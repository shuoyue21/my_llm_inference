#!/usr/bin/env python3
"""Measure logits drift and text NLL on identical teacher-forced tokens.
This is a small fixed-text accuracy probe, not a task benchmark. All raw logits
and commands stay in the requested local output directory.
Run under run_on_empty_gpu.py, never concurrently with performance collection.
"""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import numpy as np

TEXT = ('Once upon a time, a little girl named Lily found a small bird in the garden. '
        'She gave it water and waited quietly. The next morning the bird could fly again. '
        'Lily smiled as it returned to the tall tree. '
        'A computer stores information in memory and follows instructions to solve a problem. '
        'The sum of three and five is eight. 春天来了，花园里的小鸟开始唱歌。')


def artifact_path(prefix, suffix):
    # Model names contain dots; with_suffix would replace part of the model ID
    # and cause different precision runs to overwrite the same evidence.
    return prefix.parent / (prefix.name + suffix)


def compare(reference, actual, tokens):
    if reference.shape != actual.shape or not np.isfinite(actual).all():
        raise RuntimeError('Mismatched/nonfinite logits')
    targets = tokens[1:]
    ref_nll, nll, kl, squared, maximum, agree = [], [], [], 0., 0., 0
    for position, (ref, row) in enumerate(zip(reference, actual)):
        ref, row = ref.astype(np.float64), row.astype(np.float64)
        delta = row - ref
        maximum = max(maximum, float(np.max(np.abs(delta))))
        squared += float(delta @ delta)
        agree += int(np.argmax(ref) == np.argmax(row))
        logp = ref - (np.max(ref) + np.log(np.exp(ref - np.max(ref)).sum()))
        logq = row - (np.max(row) + np.log(np.exp(row - np.max(row)).sum()))
        kl.append(float(np.exp(logp) @ (logp - logq)))
        if position < len(targets):
            ref_nll.append(-logp[targets[position]])
            nll.append(-logq[targets[position]])
    delta_nll = float(np.mean(nll) - np.mean(ref_nll))
    return dict(rows=len(reference), vocab=reference.shape[1], max_abs_logit_error=maximum,
                logits_rmse=math.sqrt(squared / reference.size), top1_agreement=agree / len(reference),
                mean_kl=float(np.mean(kl)), reference_nll=float(np.mean(ref_nll)),
                nll=float(np.mean(nll)), nll_delta=delta_nll, perplexity_ratio=math.exp(delta_nll))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', required=True)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--precisions', nargs='+', default=['fp32','fp16','bf16','w8a16','w4a16'])
    p.add_argument('--attention', default='reference')
    a = p.parse_args()
    manifest=json.loads((Path('.artifacts/performance') / a.stage / 'manifest.json').read_text())
    a.output.mkdir(parents=True, exist_ok=True)
    if any(a.output.iterdir()):raise SystemExit('Precision output directory must be empty; choose a new path to retain earlier evidence')
    report={'scope':'fixed text teacher-forced NLL and numerical drift; not a task benchmark', 'text':TEXT, 'models':{}}
    for model in manifest['models']:
        rows={}
        for precision in a.precisions:
            prefix=a.output/(model['name']+'_'+precision+'_'+a.attention)
            dump=artifact_path(prefix,'.logits.f32')
            command=[str(Path(model['binary']).resolve()),model['checkpoint'],model['tokenizer'],
                     '--device','cuda','--serving','continuous','--precision',precision,
                     '--concurrency','1','--requests','1','--token-budget','32',
                     '--max-new-tokens','1','--ignore-eos','--prompt',TEXT,'--dump-logits',str(dump),'--json']
            if a.attention != 'reference':command += ['--attention',a.attention]
            env=dict(os.environ,LD_LIBRARY_PATH=str(Path(model['library']).resolve().parent)+':'+os.environ.get('LD_LIBRARY_PATH',''),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
            with artifact_path(prefix,'.stdout.jsonl').open('w') as out,artifact_path(prefix,'.stderr.txt').open('w') as err:
                subprocess.run(command,env=env,stdout=out,stderr=err,check=True)
            sample=json.loads(artifact_path(prefix,'.stdout.jsonl').read_text())
            tokens=sample['prompt_tokens']
            values=np.memmap(dump,mode='r',dtype=np.float32).reshape(len(tokens),-1)
            rows[precision]=(values,tokens,sample)
        native_prefix=a.output/(model['name']+'_native_fp32')
        native_dump=artifact_path(native_prefix,'.logits.f32')
        native_command=[str(Path(model['binary']).resolve()),model['checkpoint'],model['tokenizer'],
                        '--device','cuda','--max-new-tokens','1','--ignore-eos','--prompt',TEXT,
                        '--dump-logits',str(native_dump),'--json']
        with artifact_path(native_prefix,'.stdout.jsonl').open('w') as out,artifact_path(native_prefix,'.stderr.txt').open('w') as err:
            subprocess.run(native_command,env=env,stdout=out,stderr=err,check=True)
        native_sample=json.loads(artifact_path(native_prefix,'.stdout.jsonl').read_text())
        if native_sample['prompt_tokens']!=rows['fp32'][1]:raise RuntimeError('Native/reference token inputs differ')
        reference=rows['fp32'][0]
        native=np.memmap(native_dump,mode='r',dtype=np.float32).reshape(len(native_sample['prompt_tokens']),-1)
        native_check=compare(native,reference,rows['fp32'][1])
        if native_check['max_abs_logit_error']>0.003 or abs(native_check['nll_delta'])>0.0002:
            raise RuntimeError('Serving FP32 differs from native FP32: '+str(native_check))
        model_report={}
        for precision,(values,tokens,sample) in rows.items():
            if tokens!=rows['fp32'][1]:raise RuntimeError('Token inputs changed across precisions')
            model_report[precision]=dict(compare(reference,values,tokens),packed_weight_bytes=sample['linear_weight_bytes'])
        model_report['native_fp32_control']=native_check
        report['models'][model['name']]=model_report
        (a.output/'quality.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
        print(json.dumps({model['name']:model_report}),flush=True)


if __name__=='__main__':
    main()
