"""Fail-closed NVML empty-GPU gate and continuous process/memory telemetry."""
import datetime
import fcntl
import json
import os
from pathlib import Path
import threading
import time
import pynvml as nv

class GpuMonitor:
 def __init__(self, uuid=None, max_idle_mib=128, max_idle_util=1):
  nv.nvmlInit();self.max_idle_bytes=max_idle_mib*1024**2;self.max_idle_util=max_idle_util
  self.handles=[nv.nvmlDeviceGetHandleByIndex(i) for i in range(nv.nvmlDeviceGetCount())]
  self.uuid=uuid;self.lock=None
 def snapshot(self,handle):
  memory=nv.nvmlDeviceGetMemoryInfo(handle,nv.nvmlMemory_v2);util=nv.nvmlDeviceGetUtilizationRates(handle)
  processes={}
  # An unavailable process query is an error, not proof of an idle card.
  for kind,query in [('compute',nv.nvmlDeviceGetComputeRunningProcesses),('graphics',nv.nvmlDeviceGetGraphicsRunningProcesses)]:
   for proc in query(handle):
    item=processes.setdefault(proc.pid,{'pid':proc.pid,'kinds':[],'used_memory_bytes':None})
    item['kinds'].append(kind);item['used_memory_bytes']=proc.usedGpuMemory
    try:item['name']=nv.nvmlSystemGetProcessName(proc.pid)
    except nv.NVMLError:item['name']=None
  return {'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'monotonic_ns':time.monotonic_ns(),
          'uuid':nv.nvmlDeviceGetUUID(handle),'name':nv.nvmlDeviceGetName(handle),
          'memory_used_bytes':memory.used,'memory_reserved_bytes':memory.reserved,'memory_total_bytes':memory.total,
          'gpu_util_percent':util.gpu,'memory_util_percent':util.memory,'processes':list(processes.values()),
          'temperature_c':nv.nvmlDeviceGetTemperature(handle,nv.NVML_TEMPERATURE_GPU),
          'power_mw':nv.nvmlDeviceGetPowerUsage(handle),'sm_clock_mhz':nv.nvmlDeviceGetClockInfo(handle,nv.NVML_CLOCK_SM)}
 def background_graphics(self,process):
  # Verified server Xorg occupies ~4.5 MiB on every GPU. Never exempt compute clients.
  return (process['kinds']==['graphics'] and process.get('name')=='/usr/lib/xorg/Xorg'
          and isinstance(process.get('used_memory_bytes'),int) and 0<=process['used_memory_bytes']<=16*1024**2)
 def outsiders(self,sample,allowed_pid=None):
  def allowed(pid):
   if pid==allowed_pid or pid in getattr(self,'owned_pids',set()):return True
   if getattr(self,'allowed_process_group',None) is not None:
    try:
     if os.getpgid(pid)==self.allowed_process_group:
      if not hasattr(self,'owned_pids'):self.owned_pids=set()
      self.owned_pids.add(pid);return True
     return False
    except (ProcessLookupError,PermissionError):return False
   return False
  return [p for p in sample['processes'] if not allowed(p['pid']) and not self.background_graphics(p)]
 def idle(self,sample):
  return not self.outsiders(sample) and sample['memory_used_bytes']<=self.max_idle_bytes and sample['gpu_util_percent']<=self.max_idle_util
 def scan(self):return [self.snapshot(h) for h in self.handles]
 def acquire(self,wait_seconds=0,consecutive=5,interval=.5):
  deadline=time.monotonic()+(wait_seconds if wait_seconds>0 else consecutive*interval+1);counts={};history=[]
  while True:
   samples=self.scan();history.extend(samples)
   for sample in samples:
    uuid=sample['uuid']
    if self.uuid and uuid!=self.uuid:continue
    counts[uuid]=counts.get(uuid,0)+1 if self.idle(sample) else 0
    if counts[uuid]>=consecutive:
     lock=open('/tmp/my_llm_inference_gpu_'+uuid+'.lock','a+')
     try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
     except BlockingIOError:lock.close();continue
     requested_uuid=self.uuid
     self.uuid=uuid;self.lock=lock
     self.handle=nv.nvmlDeviceGetHandleByUUID(uuid)
     # Recheck after acquiring the cooperative lock.
     last=self.snapshot(self.handle);history.append(last)
     if self.idle(last):return history
     self.release();self.uuid=requested_uuid;counts[uuid]=0
   if (wait_seconds==0 and not any(self.idle(x) for x in samples if not self.uuid or x["uuid"]==self.uuid)) or time.monotonic()>=deadline:
    raise NoIdleGpu(history)
   time.sleep(min(interval,max(0,deadline-time.monotonic())))
 def release(self):
  if self.lock:
   fcntl.flock(self.lock,fcntl.LOCK_UN);self.lock.close();self.lock=None
 def watch(self,allowed_pid,interval=.02):return Watch(self,allowed_pid,interval)

class NoIdleGpu(RuntimeError):
 def __init__(self,history):
  super().__init__('No empty GPU: process, memory, utilization and consecutive-sample gates have not passed')
  self.history=history

class Watch:
 def __init__(self,monitor,allowed_pid,interval):
  self.monitor=monitor;self.pid=allowed_pid;self.interval=interval;self.samples=[];self.errors=[];self.foreign=[];self.stop_event=threading.Event()
 def poll(self):
  try:
   sample=self.monitor.snapshot(self.monitor.handle);self.samples.append(sample)
   outsiders=self.monitor.outsiders(sample,self.pid)
   if outsiders:self.foreign.append({'sample_index':len(self.samples)-1,'processes':outsiders})
  except Exception as error:self.errors.append(repr(error))
 def run(self):
  while not self.stop_event.is_set():self.poll();self.stop_event.wait(self.interval)
 def start(self):self.poll();self.thread=threading.Thread(target=self.run,daemon=True);self.thread.start()
 def stop(self):self.stop_event.set();self.thread.join();self.poll()
 @property
 def valid(self):return bool(self.samples) and not self.errors and not self.foreign
 def report(self):
  deltas=[(b['monotonic_ns']-a['monotonic_ns'])/1e6 for a,b in zip(self.samples,self.samples[1:])]
  return {'valid':self.valid,'allowed_pid':self.pid,'observed_owned_pids':sorted(getattr(self.monitor,'owned_pids',set())|{self.pid}),'foreign_processes':self.foreign,'errors':self.errors,
          'poll_interval_ms':self.interval*1000,'max_observation_gap_ms':max(deltas,default=0),
          'peak_sampled_device_memory_bytes':max((s['memory_used_bytes'] for s in self.samples),default=0),
          'samples':self.samples}

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path);a=p.parse_args()
 monitor=GpuMonitor();report={'gpus':monitor.scan()};report['idle_uuids']=[s['uuid'] for s in report['gpus'] if monitor.idle(s)]
 if a.output:a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps(report,indent=2))
