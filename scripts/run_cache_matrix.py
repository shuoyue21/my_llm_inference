#!/usr/bin/env python3
"""Run/audit the full paged primary matrix and full dense reference control."""
import argparse,datetime,json,subprocess,sys,time
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',required=True);p.add_argument('--gpu-uuid',required=True);a=p.parse_args()
    root=Path('docs/performance/stages')/a.stage;status_file=root/'cache_matrix_status.json';done=[]
    def status(phase,**extra):
        status_file.write_text(json.dumps({'stage':a.stage,'gpu_uuid':a.gpu_uuid,'phase':phase,'completed_labels':done,'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),**extra},indent=2)+'\n')
    try:
        status('primary_validation_and_matrix')
        subprocess.run([sys.executable,'scripts/stage_runner.py','--stage',a.stage,'--gpu-uuid',a.gpu_uuid],check=True)
        for label in ['cumulative','contiguous_reference']:
            if label!='cumulative':
                while True:
                    status('measuring_dense_reference')
                    result=subprocess.run([sys.executable,'scripts/benchmark_suite.py','--stage',a.stage,'--gpu-uuid',a.gpu_uuid,'--label',label,'--wait-seconds','20','--extra-arg=--kv-cache','--extra-arg=contiguous'])
                    if result.returncode==0:break
                    if result.returncode!=75:raise RuntimeError('Dense reference matrix failed')
                    status('waiting_for_gpu');time.sleep(15)
            status('auditing',label=label)
            output=root/('audit.json' if label=='cumulative' else 'audit_contiguous_reference.json')
            subprocess.run([sys.executable,'scripts/audit_stage.py','--stage',a.stage,'--gpu-uuid',a.gpu_uuid,'--label',label,'--output',str(output)],check=True)
            done.append(label)
        status('complete',verified_cases=72,measured_samples=360)
        print(a.stage+': full paged and contiguous matrices complete',flush=True)
    except Exception as error:
        status('failed',error=repr(error));raise
if __name__=='__main__':main()
