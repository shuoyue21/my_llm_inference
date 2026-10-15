#!/usr/bin/env python3
"""Real-model generation and CLI error-handling regression tests (default Llama build)."""
import argparse
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile
p=argparse.ArgumentParser(description=__doc__)
p.add_argument("--binary",default="build/demo/llama_infer")
p.add_argument("--model",default="models/stories15M.bin")
p.add_argument("--tokenizer",default="models/tokenizer.model")
p.add_argument("--cuda",action="store_true")
a=p.parse_args()
env=dict(os.environ,OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1")
def run(args,good=True,model=None):
    r=subprocess.run([a.binary,model or a.model,a.tokenizer]+args,capture_output=True,text=True,env=env)
    assert (r.returncode==0) == good, (r.returncode,r.stdout,r.stderr)
    if good: return json.loads(r.stdout)
base=["--prompt","Once upon a time","--max-new-tokens","32","--json"]
cpu=run(base)
assert len(cpu["generated_tokens"])==32 and "little girl" in cpu["text"]
if a.cuda:
    gpu=run(base+["--device","cuda"])
    assert cpu["generated_tokens"]==gpu["generated_tokens"]
single=run(["--prompt","","--max-new-tokens","1","--json"])
assert len(single["prompt_tokens"])==1 and len(single["generated_tokens"])==1
assert not run(["--max-new-tokens","0","--json"])["generated_tokens"]
for flags in [["--max-new-tokens","-1"],["--max-new-tokens","999999"],["--device","bad"],["--prompt"],["--unknown"]]: run(flags,False)
run([],False,model="/nonexistent/checkpoint.bin")
with tempfile.TemporaryDirectory() as directory:
    path=Path(directory)/"bad.bin"
    path.write_bytes(b"bad")
    run([],False,model=str(path))
    path.write_bytes(struct.pack("7i",288,768,6,6,6,32000,256)+b"\x00"*64)
    run([],False,model=str(path))
    path.write_bytes(struct.pack("7i",288,768,6,0,6,32000,256)+b"\x00"*64)
    run([],False,model=str(path))
print("PASS: real generation, CPU/CUDA agreement" if a.cuda else "PASS: real CPU generation", ", BOS-only prompt, zero generation, invalid CLI, context overflow and malformed checkpoints")
