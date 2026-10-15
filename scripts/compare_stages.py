#!/usr/bin/env python3
"""Produce paired stage comparisons; require complete matrices and the same physical GPU."""
import argparse,json
from pathlib import Path

def load(stage,label,uuid):
 root=Path('docs/performance/stages')/stage
 manifest=json.loads((root/'manifest.json').read_text());data=root/label/uuid
 summary=json.loads((data/'summary.json').read_text())
 if summary['status']!='complete':raise ValueError(stage+' does not contain a full matrix')
 expected={f"{m['name']}_p{p}_g{g}" for m in manifest['models'] for p in manifest['workloads']['prompt_tokens'] for g in manifest['workloads']['generated_tokens'] if p+g-1<=m['context']}
 if set(summary['completed_cases'])!=expected:raise ValueError(stage+' is missing required cases')
 records={}
 for key in sorted(expected):
  record=json.loads((data/(key+'.json')).read_text())
  if not record['accepted'] or not record['monitor']['valid'] or record['source_digest']!=manifest['source_digest'] or record['gpu_uuid']!=uuid:raise ValueError('Invalid record '+key)
  records[key]=record
 return manifest,records

def compare(base,current):
 for key in ['gpu_uuid','precision','model','prompt_tokens','generated_tokens','cpu_threads','warmups','repeats']:
  if base[key]!=current[key]:raise ValueError('Unpaired field '+key)
 if base['samples'][0]['prompt_tokens']!=current['samples'][0]['prompt_tokens']:raise ValueError('Prompt token IDs differ')
 changes={}
 for metric in ['ttft_ms','prefill_ms','decode_tokens_per_second','total_ms']:
  b=base['metrics'][metric]['median'];c=current['metrics'][metric]['median']
  changes[metric]={'baseline':b,'current':c,'change_percent':100*(c/b-1),'speedup':(c/b if metric=='decode_tokens_per_second' else b/c)}
 for metric in ['median','p95','p99']:
  b=base['metrics']['tpot_ms'][metric];c=current['metrics']['tpot_ms'][metric]
  changes['tpot_'+metric+'_ms']={'baseline':b,'current':c,'change_percent':100*(c/b-1)}
 b=base['monitor']['peak_sampled_device_memory_bytes']-base['before']['memory_used_bytes'];c=current['monitor']['peak_sampled_device_memory_bytes']-current['before']['memory_used_bytes']
 changes['sampled_peak_memory_increment_bytes']={'baseline':b,'current':c,'change_percent':100*(c/b-1) if b else None}
 return changes

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--baseline',default='S0');p.add_argument('--stage',required=True);p.add_argument('--gpu-uuid',required=True);p.add_argument('--baseline-label',default='cumulative');p.add_argument('--label',default='cumulative');a=p.parse_args()
 bm,base=load(a.baseline,a.baseline_label,a.gpu_uuid);cm,current=load(a.stage,a.label,a.gpu_uuid)
 if bm['precision']!=cm['precision'] or bm['workloads']!=cm['workloads']:raise ValueError('Workload/precision changed')
 for b,c in zip(bm['models'],cm['models']):
  if b['name']!=c['name'] or b['artifacts']!=c['artifacts']:raise ValueError('Model hashes/config differ')
 result={key:compare(base[key],current[key]) for key in base}
 root=Path('docs/performance/stages')/a.stage
 (root/'comparison.json').write_text(json.dumps({'baseline':a.baseline,'stage':a.stage,'gpu_uuid':a.gpu_uuid,'cases':result},indent=2)+'\n')
 lines=[f'# {a.stage} versus {a.baseline}',f'GPU: `{a.gpu_uuid}`. Complete matched workload matrices; all samples passed the empty-GPU gate.',
 'Latency/memory change below zero means reduction; throughput change above zero means increase. Negative results are retained. NVML peak is sampled and excludes initial device usage.',
 '| Case | TTFT change | TPOT median change | Decode throughput change | Sampled peak memory change |','|---|---:|---:|---:|---:|']
 for key,changes in result.items():
  values=[changes[x]['change_percent'] for x in ['ttft_ms','tpot_median_ms','decode_tokens_per_second','sampled_peak_memory_increment_bytes']]
  lines.append('| '+key+' | '+' | '.join('N/A' if x is None else f'{x:+.2f}%' for x in values)+' |')
 (root/'COMPARISON.md').write_text('\n\n'.join(lines[:3])+'\n\n'+'\n'.join(lines[3:])+'\n')
 print(f'Compared {len(result)} complete paired cases')
if __name__=='__main__':main()
