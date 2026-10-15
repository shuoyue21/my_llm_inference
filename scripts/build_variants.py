#!/usr/bin/env python3
import argparse,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('variants',nargs='*',default=['llama2','qwen2','qwen3','llama3']);p.add_argument('--jobs',type=int,default=8);a=p.parse_args()
for variant in a.variants:
 if variant not in ['llama2','qwen2','qwen3','llama3']:p.error('Unknown variant '+variant)
 build='build' if variant=='llama2' else 'build-'+variant
 command=['cmake','-S','.','-B',build,'-G','Ninja','-DCMAKE_BUILD_TYPE=Release','-DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.1/bin/nvcc','-DCMAKE_CUDA_ARCHITECTURES=89','-DFETCHCONTENT_FULLY_DISCONNECTED=ON']
 for name in ['LLAMA3','QWEN2','QWEN3']:command.append(f'-D{name}_SUPPORT='+('ON' if variant==name.lower() else 'OFF'))
 for dependency in ['googletest','glog','armadillo','sentencepiece','absl','re2','nlohmann_json']:
  source=Path('build/_deps')/(dependency+'-src')
  if source.exists():command.append(f'-DFETCHCONTENT_SOURCE_DIR_{dependency.upper()}={source.resolve()}')
 log=Path('/tmp')/('my_llm_build_'+variant+'.log')
 with log.open('w') as f:
  subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,check=True)
  subprocess.run(['cmake','--build',build,'--target','llama_infer','test_llm','-j',str(a.jobs)],stdout=f,stderr=subprocess.STDOUT,check=True)
 print(variant+' built; '+str(log),flush=True)
