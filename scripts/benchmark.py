#!/usr/bin/env python3
"""Repeat isolated CLI requests and save raw samples plus median TTFT/decode throughput.
Model load is measured separately; per-run startup, tokenization and prompt processing
are not included in decode throughput. Use a fixed model, prompt, hardware and thread count.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--binary", default="build/demo/llama_infer")
p.add_argument("--model", default="models/stories15M.bin")
p.add_argument("--tokenizer", default="models/tokenizer.model")
p.add_argument("--devices", nargs="+", choices=["cpu", "cuda"], default=["cpu"])
p.add_argument("--prompt", default="Once upon a time")
p.add_argument("--tokens", type=int, default=64)
p.add_argument("--runs", type=int, default=3)
p.add_argument("--threads", type=int, default=1)
p.add_argument("--output", type=Path, default=Path("docs/benchmark_baseline.json"))
a = p.parse_args()
if "cuda" in a.devices:
    p.error("GPU benchmarking requires the empty-GPU gate: use scripts/benchmark_suite.py --stage STAGE")
if a.runs < 1 or a.tokens < 2 or a.threads < 1: p.error("runs/threads must be positive; tokens must be at least 2")
env = dict(os.environ, OPENBLAS_NUM_THREADS=str(a.threads), OMP_NUM_THREADS=str(a.threads))
report = {"measured_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
          "platform": platform.platform(), "binary": a.binary, "model": a.model,
          "prompt": a.prompt, "max_new_tokens": a.tokens, "threads": a.threads,
          "warmup_requests": 1, "results": {}}
try: report["gpu"] = subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,driver_version", "--format=csv,noheader"], text=True).splitlines()
except (OSError, subprocess.CalledProcessError): report["gpu"] = []
for device in a.devices:
    samples = []
    for iteration in range(a.runs + 1):
        command = [a.binary, a.model, a.tokenizer, "--device", device, "--prompt", a.prompt,
                   "--max-new-tokens", str(a.tokens), "--json"]
        run = subprocess.run(command, capture_output=True, text=True, check=True, env=env)
        result = json.loads(run.stdout)
        if iteration: samples.append(result)
    report["results"][device] = {"samples": samples, "median": {
        metric: statistics.median(sample[metric] for sample in samples)
        for metric in ["load_ms", "ttft_ms", "decode_ms", "decode_tokens_per_second"]}}
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({k:v["median"] for k,v in report["results"].items()}, indent=2))
