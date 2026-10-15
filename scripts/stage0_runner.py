#!/usr/bin/env python3
"""Wait for empty GPUs, finish GPU correctness, then collect the untouched S0 matrix.
This runner never implements or enables optimization features.
"""
import argparse,datetime,json,os,subprocess,time,sys,re
from pathlib import Path
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage',default='S0');args=parser.parse_args()
stage=args.stage
if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',stage):parser.error('Invalid stage name')
root=Path('docs/performance/stages')/stage;root.mkdir(parents=True,exist_ok=True)
manifest=json.loads((Path('.artifacts/performance')/stage/'manifest.json').read_text())
status_path=root/'runner_status.json'
env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',LLM_GPU_POLICY_JSON=json.dumps(manifest.get('gpu_policy',{})))
validation=root/'validation';validation.mkdir(parents=True,exist_ok=True)
def status(phase,**extra):
 data={'stage':stage,'phase':phase,'pid':os.getpid(),'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),**extra}
 status_path.write_text(json.dumps(data,indent=2)+'\n')
def gated(record,command):
 while True:
  status('waiting_or_verifying',record=str(record),command=command)
  completed=subprocess.run([sys.executable,'scripts/run_on_empty_gpu.py','--record',str(record),'--wait-seconds','20','--',*command],env=env)
  if completed.returncode==0:return
  if completed.returncode!=75:raise RuntimeError('Correctness command failed: '+str(command))
  status('waiting_for_empty_gpu',next_record=str(record));time.sleep(15)
try:
 for model in manifest['models']:
  record=validation/(model['name']+'_gpu_gate.json')
  if record.exists():
   prior=json.loads(record.read_text())
   if prior.get('status')=='passed' and prior.get('frozen_source_digest')==manifest['source_digest']:continue
  library=str(Path(model['library']).resolve().parent)
  frozen_env=['env','LD_LIBRARY_PATH='+library+(':'+os.environ['LD_LIBRARY_PATH'] if os.environ.get('LD_LIBRARY_PATH') else '')]
  if model['family']=='llama2':
   command=[sys.executable,'scripts/verify_reference.py','--binary',model['binary'],'--device','cuda']
  else:
   command=[sys.executable,'scripts/verify_hf.py','--binary',model['binary'],'--model-dir',model['hf_dir'],'--device','cuda','--output',str(validation/(model['name']+'_cuda.json'))]
  gated(validation/(model['name']+'_gpu_units_gate.json'),frozen_env+[model['test_binary']])
  gated(record,frozen_env+command)
  saved=json.loads(record.read_text());saved['frozen_source_digest']=manifest['source_digest'];record.write_text(json.dumps(saved,indent=2)+'\n')
 status('gpu_correctness_complete')
 while True:
  status('waiting_or_measuring_baseline')
  completed=subprocess.run([sys.executable,'scripts/benchmark_suite.py','--stage',stage,'--wait-seconds','20'],env=env)
  if completed.returncode==0:break
  if completed.returncode!=75:raise RuntimeError('Baseline measurement failed')
  status('waiting_for_empty_gpu');time.sleep(15)
 status('baseline_complete_no_features_implemented')
 print('S0 GPU correctness and empty-GPU baseline complete; ready to implement feature 1.',flush=True)
except Exception as error:
 status('failed',error=str(error));raise
