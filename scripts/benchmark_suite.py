#!/usr/bin/env python3
"""Measure an immutable stage only on an empty GPU, rejecting external GPU clients.
Uses fixed token IDs, resident-model warmups/repeats, and continuous NVML telemetry.
"""
import argparse,datetime,gzip,hashlib,json,os,platform,statistics,subprocess,time
from pathlib import Path
from empty_gpu import GpuMonitor,NoIdleGpu
from freeze_stage import digest
from measurement_records import harness_provenance,prepare_existing_result

def quantile(values,q):
 x=sorted(values)
 if not x:return None
 f=(len(x)-1)*q;i=int(f);return x[i]+(x[min(i+1,len(x)-1)]-x[i])*(f-i)

def summarize(samples,prompt_length):
 metrics={}
 if samples and 'serving_mode' in samples[0]:
  for name in ['load_ms','total_ms','output_tokens_per_second','requests_per_second','ttft_ms','ttft_p95_ms','ttft_p99_ms','latency_p50_ms','latency_p95_ms','latency_p99_ms','tpot_p50_ms','tpot_p95_ms','tpot_p99_ms','queue_p95_ms','gpu_load_memory_bytes']:
   values=[sample[name] for sample in samples]
   metrics[name]={'median':statistics.median(values),'min':min(values),'max':max(values)}
  return metrics
 for name in ['load_ms','tokenization_ms','ttft_ms','prefill_ms','decode_ms','total_ms','decode_tokens_per_second','gpu_load_memory_bytes']:
  values=[s[name] for s in samples]
  metrics[name]={'median':statistics.median(values),'p95':quantile(values,.95),'min':min(values),'max':max(values)}
 for name in ['kv_reserved_bytes','kv_effective_bytes','kv_table_bytes','kv_peak_mapped_bytes','kv_peak_reserved_bytes','kv_page_allocations','kv_page_reuses','kv_page_releases','kv_table_updates','kv_release_ms']:
  if all(name in sample for sample in samples):
   values=[sample[name] for sample in samples];metrics[name]={'median':statistics.median(values),'p95':quantile(values,.95),'min':min(values),'max':max(values)}
 per_token=[x for sample in samples for x in sample['decode_token_ms']]
 metrics['tpot_ms']={'median':statistics.median(per_token),'p95':quantile(per_token,.95),'p99':quantile(per_token,.99)}
 metrics['prefill_tokens_per_second']={'median':statistics.median(prompt_length*1000/s['prefill_ms'] for s in samples)}
 return metrics

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',required=True);p.add_argument('--gpu-uuid');p.add_argument('--wait-seconds',type=float,default=0);p.add_argument('--model',action='append');p.add_argument('--extra-arg',action='append',default=[]);p.add_argument('--check-only',action='store_true');p.add_argument('--label',default='cumulative');a=p.parse_args()
 root=Path('docs/performance/stages')/a.stage;root.mkdir(parents=True,exist_ok=True)
 manifest_path=Path('.artifacts/performance')/a.stage/'manifest.json'
 manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else None
 if manifest is None and not a.check_only:raise FileNotFoundError('Freeze the stage before measuring: '+str(manifest_path))
 monitor=GpuMonitor(a.gpu_uuid,policy=manifest.get('gpu_policy',{}) if manifest else None)
 try:gate=monitor.acquire(a.wait_seconds)
 except NoIdleGpu as error:
  record={'status':'waiting_for_empty_gpu','gpu_policy':monitor.policy,'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'reason':str(error),'gpu_observations':error.history}
  (root/'waiting.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({'status':record['status'],'gpus_observed':len(error.history)}));return 75
 try:
  print(json.dumps({'status':'empty_gpu_acquired','uuid':monitor.uuid}),flush=True)
  if a.check_only:return 0
  root=root/a.label/monitor.uuid;root.mkdir(parents=True,exist_ok=True)
  manifest_path=Path('.artifacts/performance')/a.stage/'manifest.json';manifest=json.loads(manifest_path.read_text());workload=manifest['workloads']
  base_env=dict(os.environ,CUDA_VISIBLE_DEVICES=monitor.uuid,OPENBLAS_NUM_THREADS=str(workload['cpu_threads']),OMP_NUM_THREADS=str(workload['cpu_threads']))
  completed=[]
  for model in manifest['models']:
   if a.model and model['name'] not in a.model:continue
   binary=Path(model['binary']).resolve();library=Path(model['library']).resolve()
   assert digest(binary)==model['binary_sha256'] and digest(library)==model['library_sha256'],'Frozen executable changed'
   for path,info in model.get('dependencies',{}).items():assert digest(path)==info['sha256'],'Frozen dependency changed: '+path
   for path,info in model['artifacts'].items():
    if 'mtime_ns' in info:
     stat=Path(path).stat()
     assert stat.st_size==info['bytes'] and stat.st_mtime_ns==info['mtime_ns'],'Model asset changed; verify before measuring: '+path
    else:assert digest(path)==info['sha256'],'Model artifact changed: '+path
   env=dict(base_env,LD_LIBRARY_PATH=str(library.parent)+(':'+os.environ['LD_LIBRARY_PATH'] if os.environ.get('LD_LIBRARY_PATH') else ''))
   for prompt_length in workload['prompt_tokens']:
    for output_length in workload['generated_tokens']:
     if prompt_length+output_length-1>model['context']:continue
     name=f"{model['name']}_p{prompt_length}_g{output_length}";result_path=root/(name+'.json')
     if prepare_existing_result(result_path,manifest['source_digest'],monitor.uuid,a.extra_arg):
      completed.append(name);continue
     latest=monitor.snapshot(monitor.handle)
     if not monitor.idle(latest):raise NoIdleGpu([latest])
     ids=[model['seed_tokens'][i%len(model['seed_tokens'])] for i in range(prompt_length)]
     ids_path=root/(name+'.tokens');ids_path.write_text(' '.join(map(str,ids))+'\n')
     command=[str(binary),model['checkpoint'],model['tokenizer'],'--device','cuda','--prompt-ids',str(ids_path),'--max-new-tokens',str(output_length),'--ignore-eos','--warmup',str(workload['warmups']),'--repeat',str(workload['repeats']),'--json']+a.extra_arg
     stdout_path=root/(name+'.stdout.jsonl');stderr_path=root/(name+'.stderr.txt')
     with stdout_path.open('w') as output,stderr_path.open('w') as error:
      process=subprocess.Popen(command,env=env,stdout=output,stderr=error);watch=monitor.watch(process.pid);watch.start()
      while process.poll() is None:
       if not watch.valid:process.terminate();break
       time.sleep(.02)
      try:code=process.wait(timeout=10)
      except subprocess.TimeoutExpired:process.kill();code=process.wait()
      watch.stop()
     telemetry=watch.report();trace_path=root/(name+'.nvml.json.gz')
     with gzip.open(trace_path,'wt') as trace:json.dump(telemetry,trace)
     samples=[];reasons=[]
     if code!=0:reasons.append('process_exit_'+str(code))
     if not telemetry['valid']:reasons.append('foreign_gpu_client_or_nvml_error')
     if not reasons:
      samples=[json.loads(line) for line in stdout_path.read_text().splitlines() if line.strip()]
      if len(samples)!=workload['repeats']:reasons.append('wrong_sample_count')
      elif samples and 'serving_mode' in samples[0]:
       first=samples[0]
       for sample in samples:
        requests=sample['requests']
        if sample['prompt_tokens']!=ids or len(requests)!=sample['request_count'] or any(len(r['generated_tokens'])!=output_length or len(r['decode_token_ms'])!=output_length-1 for r in requests):reasons.append('wrong_serving_token_counts')
        if [r['generated_tokens'] for r in requests]!=[r['generated_tokens'] for r in first['requests']]:reasons.append('nondeterministic_serving_tokens')
      elif any(len(s['prompt_tokens'])!=prompt_length or len(s['generated_tokens'])!=output_length or len(s['decode_token_ms'])!=output_length-1 for s in samples):reasons.append('wrong_token_counts')
      elif any(s['prompt_tokens']!=ids or s['generated_tokens']!=samples[0]['generated_tokens'] for s in samples):reasons.append('nondeterministic_tokens')
     record={'accepted':not reasons,'rejection_reasons':reasons,'stage':a.stage,'source_revision':manifest['source_revision'],'source_digest':manifest['source_digest'],'gpu_uuid':monitor.uuid,'precision':manifest['precision'],'gpu_policy':monitor.policy,'model':model['name'],'prompt_tokens':prompt_length,'generated_tokens':output_length,'warmups':workload['warmups'],'repeats':workload['repeats'],'cpu_threads':workload['cpu_threads'],'harness':harness_provenance(),'command':command,'extra_args':a.extra_arg,'label':a.label,'environment':{k:env[k] for k in ['CUDA_VISIBLE_DEVICES','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','LD_LIBRARY_PATH']},'platform':platform.platform(),'cpu_affinity':sorted(os.sched_getaffinity(0)),'system_load_average':os.getloadavg(),'driver_version':__import__('pynvml').nvmlSystemGetDriverVersion(),'gate':gate,'before':latest,'monitor':{k:v for k,v in telemetry.items() if k!='samples'},'telemetry_file':str(trace_path),'samples':samples,'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
     if record['accepted']:record['metrics']=summarize(samples,prompt_length)
     result_path.write_text(json.dumps(record,indent=2)+'\n')
     if not telemetry['valid']:
      raise NoIdleGpu(telemetry['samples'][-1:])
     if reasons:raise RuntimeError(name+' rejected: '+','.join(reasons))
     completed.append(name);print(json.dumps({'status':'accepted','case':name,'metrics':record['metrics']}),flush=True)
     # Wait for utilization counters to settle; process and memory gates remain mandatory.
     for _ in range(40):
      last=monitor.snapshot(monitor.handle)
      if monitor.idle(last):break
      time.sleep(.25)
     else:raise NoIdleGpu([last])
  (root/'summary.json').write_text(json.dumps({'stage':a.stage,'source_revision':manifest['source_revision'],'source_digest':manifest['source_digest'],'gpu_uuid':monitor.uuid,'completed_cases':completed,'selected_models':a.model,'label':a.label,'extra_args':a.extra_arg,'status':'selected_models_complete' if a.model else 'complete'},indent=2)+'\n')
  return 0
 except NoIdleGpu as error:
  (root/'waiting.json').write_text(json.dumps({'status':'waiting_for_empty_gpu','gpu_policy':monitor.policy,'gpu_observations':error.history},indent=2)+'\n');print('Waiting for the selected GPU to become empty again',flush=True);return 75
 finally:monitor.release()
if __name__=='__main__':raise SystemExit(main())
