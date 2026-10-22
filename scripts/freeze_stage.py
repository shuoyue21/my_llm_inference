#!/usr/bin/env python3
"""Freeze each stage's executables and core shared library before collecting benchmarks."""
import argparse,hashlib,json,shutil,subprocess,re,tarfile
from pathlib import Path
from gpu_policy import normalize_policy

def digest(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for chunk in iter(lambda:f.read(8*1024**2),b''):h.update(chunk)
 return h.hexdigest()

def source_digest():
 files=[]
 for folder in ['include','source','demo','test','scripts']:
  files.extend(p for p in Path(folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
 files.extend(Path(p) for p in ['CMakeLists.txt','CMakePresets.json'])
 h=hashlib.sha256()
 for p in sorted(files):h.update(str(p).encode()+b'\0'+p.read_bytes())
 return h.hexdigest()

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',required=True);p.add_argument('--registry',type=Path,default=Path('scripts/model_registry.json'));p.add_argument('--gpu-policy',type=Path);p.add_argument('--model-evidence',type=Path,help='Reuse prior verified asset identities when assets are unchanged');a=p.parse_args()
 if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',a.stage):p.error('Invalid stage name')
 record=Path('docs/performance/stages')/a.stage
 if (record/'manifest.json').exists():raise SystemExit(f'{record}/manifest.json already exists; choose a new machine-specific stage name')
 folder=Path('.artifacts/performance')/a.stage
 if folder.exists():raise SystemExit(f'{folder} already exists; stage snapshots are immutable')
 prior_models={m['name']:m for m in json.loads(a.model_evidence.read_text())['models']} if a.model_evidence else {}
 registry=json.loads(a.registry.read_text());manifest={'stage':a.stage,'source_revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'source_digest':source_digest(),'precision':registry['precision'],'gpu_policy':normalize_policy(json.loads(a.gpu_policy.read_text()) if a.gpu_policy else None),'models':[],'workloads':registry['workloads']}
 for m in registry['models']:
  out=folder/m['name'];(out/'bin').mkdir(parents=True);(out/'lib').mkdir()
  binary=out/'bin/llama_infer';library=out/'lib/libllama.so'
  shutil.copy2(Path(m['build'])/'demo/llama_infer',binary);shutil.copy2(Path(m['build'])/'lib/libllama.so',library)
  unit=out/'bin/test_llm';shutil.copy2(Path(m['build'])/'test/test_llm',unit)
  item=dict(m,binary=str(binary),library=str(library),test_binary=str(unit),test_binary_sha256=digest(unit),binary_sha256=digest(binary),library_sha256=digest(library))
  dependencies={}
  linked='\n'.join(subprocess.check_output(['ldd',str(Path(m['build'])/name)],text=True) for name in ['demo/llama_infer','test/test_llm'])
  for soname,original in re.findall(r'\s*(\S+) => (/\S+)',linked):
   source=Path(original)
   if source.resolve().is_relative_to(Path.cwd()):
    destination=out/'lib'/soname
    if destination!=library:shutil.copy2(source,destination)
    dependencies[str(destination)]={'sha256':digest(destination),'original':str(source)}
  item['dependencies']=dependencies
  item['artifacts']={}
  for file in [Path(m['checkpoint']),Path(m['tokenizer']),Path(m['checkpoint']+'.json')]:
   if file.exists():
    stat=file.stat();prior=prior_models.get(m['name'],{}).get('artifacts',{}).get(str(file))
    if a.model_evidence:
     if not prior or prior['bytes']!=stat.st_size:raise RuntimeError('Asset differs from prior evidence: '+str(file))
     identity=dict(prior,verification_reused_from=str(a.model_evidence))
    else:identity={'sha256':digest(file),'bytes':stat.st_size}
    item['artifacts'][str(file)]=dict(identity,mtime_ns=stat.st_mtime_ns)

  manifest['models'].append(item)
 folder.mkdir(parents=True,exist_ok=True)
 with tarfile.open(folder/'source.tar.gz','w:gz') as archive:
  for source_root in ['include','source','demo','test','scripts']:
   for file in sorted(Path(source_root).rglob('*')):
    if file.is_file() and '__pycache__' not in file.parts:archive.add(file,arcname=str(file),recursive=False)
  archive.add('CMakeLists.txt');archive.add('CMakePresets.json')
 (folder/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 record=Path('docs/performance/stages')/a.stage;record.mkdir(parents=True,exist_ok=True);(record/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 print(json.dumps({'stage':a.stage,'revision':manifest['source_revision'],'source_digest':manifest['source_digest'],'snapshot':str(folder),'models':len(manifest['models'])}))
if __name__=='__main__':main()
