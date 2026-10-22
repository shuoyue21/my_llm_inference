#!/usr/bin/env python3
"""Run controlled S5/S6/S7 comparisons through the existing monitored suite.
Every label includes the complete four-model workload registry. No raw output
is staged in Git; immutable stages must already exist and pass validation.
"""
import argparse
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--feature', choices=['prefix','precision','attention'], required=True)
    p.add_argument('--stage', required=True)
    p.add_argument('--gpu-uuid', required=True)
    p.add_argument('--requests', type=int)
    p.add_argument('--concurrency', type=int)
    p.add_argument('--token-budget', type=int, default=128)
    a=p.parse_args()
    if a.requests is None:a.requests=4 if a.feature=='prefix' else 1
    if a.concurrency is None:a.concurrency=4 if a.feature=='prefix' else 1
    if a.feature=='prefix':
        variants=[('prefix-off',['--prefix-cache-mib','0']),('prefix-on',['--prefix-cache-mib','512'])]
    elif a.feature=='precision':
        variants=[(precision,['--precision',precision]) for precision in ['fp32','fp16','bf16','w8a16','w4a16']]
    else:
        variants=[('attention-'+mode,['--attention',mode]) for mode in ['reference','online']]
    for label, options in variants:
        extra=['--serving','continuous','--concurrency',str(a.concurrency),'--requests',str(a.requests),
               '--token-budget',str(a.token_budget)]+options
        command=[sys.executable,'scripts/benchmark_suite.py','--stage',a.stage,'--gpu-uuid',a.gpu_uuid,
                 '--wait-seconds','30','--label',label]+['--extra-arg='+arg for arg in extra]
        result=subprocess.run(command)
        if result.returncode:return result.returncode
    return 0


if __name__=='__main__':raise SystemExit(main())
