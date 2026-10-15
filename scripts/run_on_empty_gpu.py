#!/usr/bin/env python3
"""Run a correctness/diagnostic command on an empty GPU with fail-closed telemetry."""
import argparse,gzip,json,os,subprocess,time,signal
from pathlib import Path
from empty_gpu import GpuMonitor,NoIdleGpu
from measurement_records import archive_attempt,harness_provenance
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--wait-seconds',type=float,default=0);p.add_argument('--gpu-uuid');p.add_argument('--record',type=Path,required=True);p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
command=a.command[1:] if a.command[:1]==['--'] else a.command
if not command:p.error('A command is required')
monitor=GpuMonitor(a.gpu_uuid);a.record.parent.mkdir(parents=True,exist_ok=True)
if a.record.exists():
 prior=json.loads(a.record.read_text())
 if prior.get('status') in ['passed','failed','contaminated']:archive_attempt(a.record)
try:gate=monitor.acquire(a.wait_seconds)
except NoIdleGpu as error:
 a.record.write_text(json.dumps({'status':'waiting_for_empty_gpu','gpu_policy':monitor.policy,'purpose':'correctness','gpu_observations':error.history},indent=2)+'\n');print('No empty GPU available');raise SystemExit(75)
try:
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=monitor.uuid)
 output_path=a.record.with_suffix('.stdout.txt');error_path=a.record.with_suffix('.stderr.txt')
 with output_path.open('w') as output,error_path.open('w') as error:
  process=subprocess.Popen(command,env=env,stdout=output,stderr=error,start_new_session=True)
  monitor.allowed_process_group=process.pid
  watcher=monitor.watch(process.pid);watcher.start()
  while process.poll() is None:
   if not watcher.valid:os.killpg(process.pid,signal.SIGTERM);break
   time.sleep(.02)
  try:code=process.wait(timeout=10)
  except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);code=process.wait()
  watcher.stop()
 report=watcher.report();trace=a.record.with_suffix('.nvml.json.gz')
 with gzip.open(trace,'wt') as f:json.dump(report,f)
 a.record.write_text(json.dumps({'status':('contaminated' if not report['valid'] else 'passed' if code==0 else 'failed'),'harness':harness_provenance(),'gpu_policy':monitor.policy,'purpose':'correctness','gpu_uuid':monitor.uuid,'command':command,'exit_code':code,'gate':gate,'monitor':{k:v for k,v in report.items() if k!='samples'},'telemetry_file':str(trace)},indent=2)+'\n')
 print(output_path.read_text());raise SystemExit(75 if not report['valid'] else 0 if code==0 else 1)
finally:monitor.release()
