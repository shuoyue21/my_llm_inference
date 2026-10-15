#!/usr/bin/env python3
"""Complete a frozen primary stage and all three Graph/fusion ablation matrices."""
import argparse
import datetime
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',required=True)
    p.add_argument('--gpu-uuid',required=True)
    a=p.parse_args()
    root=Path('docs/performance/stages')/a.stage
    state=root/'ablation_status.json'
    modes=[('cumulative',[]),('graph_only',['--decode','optimized','--cuda-graph','on','--fusion','off']),
           ('fusion_only',['--decode','optimized','--cuda-graph','off','--fusion','on']),
           ('neither',['--decode','optimized','--cuda-graph','off','--fusion','off'])]
    done=[]
    def status(phase,**extra):
        state.write_text(json.dumps({'stage':a.stage,'gpu_uuid':a.gpu_uuid,'phase':phase,'completed_labels':done,
            'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),**extra},indent=2)+'\n')
    try:
        status('primary_correctness_and_matrix')
        subprocess.run([sys.executable,'scripts/stage_runner.py','--stage',a.stage,'--gpu-uuid',a.gpu_uuid],check=True)
        for label,extra in modes:
            if label!='cumulative':
                while True:
                    status('measuring_ablation',label=label)
                    command=[sys.executable,'scripts/benchmark_suite.py','--stage',a.stage,'--gpu-uuid',a.gpu_uuid,'--label',label,'--wait-seconds','20']
                    command+=['--extra-arg='+value for value in extra]
                    result=subprocess.run(command)
                    if result.returncode==0:break
                    if result.returncode!=75:raise RuntimeError('Ablation failed: '+label)
                    status('waiting_for_gpu',label=label)
                    time.sleep(15)
            status('auditing_label',label=label)
            output=root/('audit.json' if label=='cumulative' else 'audit_'+label+'.json')
            subprocess.run([sys.executable,'scripts/audit_stage.py','--stage',a.stage,'--gpu-uuid',a.gpu_uuid,'--label',label,'--output',str(output)],check=True)
            done.append(label)
        status('complete',verified_cases=144,measured_samples=720)
        print(a.stage+': primary and three ablations complete, 144 cases / 720 samples',flush=True)
    except Exception as error:
        status('failed',error=repr(error))
        raise

if __name__=='__main__':main()
