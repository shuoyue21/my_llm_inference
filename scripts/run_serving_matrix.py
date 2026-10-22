#!/usr/bin/env python3
"""Compare serial S3 inference and continuous batches on the complete registry."""
import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', required=True)
    p.add_argument('--gpu-uuid', required=True)
    p.add_argument('--requests', type=int, default=16)
    p.add_argument('--concurrency', type=int, nargs='+', default=[4, 8])
    p.add_argument('--token-budget', type=int, default=128)
    p.add_argument('--label-prefix', default='')
    p.add_argument('--skip-serial', action='store_true')
    p.add_argument('--extra-arg', action='append', default=[])
    a = p.parse_args()
    cases = ([] if a.skip_serial else [('serial', 'serial', 1)]) + [
        ('continuous-c' + str(c), 'continuous', c) for c in a.concurrency]
    for label, mode, concurrency in cases:
        command = [sys.executable, 'scripts/benchmark_suite.py', '--stage', a.stage,
                   '--gpu-uuid', a.gpu_uuid, '--label', a.label_prefix + label, '--wait-seconds', '30']
        extra = ['--serving', mode, '--concurrency', str(concurrency), '--requests', str(a.requests),
                 '--token-budget', str(a.token_budget)] + a.extra_arg
        command += ['--extra-arg=' + arg for arg in extra]
        result = subprocess.run(command)
        if result.returncode:
            return result.returncode
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
