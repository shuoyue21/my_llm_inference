#!/usr/bin/env python3
"""Revalidate the complete four-stage objective from raw data, not status flags."""
import argparse
import hashlib
import json
import re
from pathlib import Path
import subprocess
from audit_stage import audit, require
from compare_stages import load, compare
from freeze_stage import digest
from gpu_policy import normalize_policy


def read(path):return json.loads(Path(path).read_text())

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-id',required=True);p.add_argument('--gpu-uuid',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--s3-stage');a=p.parse_args()
    stages={name:name+'-'+a.run_id for name in ['S0','S1','S2','S3']}
    if a.s3_stage:stages['S3']=a.s3_stage
    manifests={name:read(Path('docs/performance/stages')/stage/'manifest.json') for name,stage in stages.items()}
    base=manifests['S0']
    for name,m in manifests.items():
        require(m['precision']=='fp32' and m['workloads']==base['workloads'],'Precision/workloads changed in '+name)
        require(normalize_policy(m.get('gpu_policy'))==normalize_policy(base.get('gpu_policy')),'GPU policy changed in '+name)
        require([x['name'] for x in m['models']]==[x['name'] for x in base['models']],'Four-model scope changed')
        for original,current in zip(base['models'],m['models']):require(original['artifacts']==current['artifacts'],'Model/sidecar assets changed')
    labels={'S0':['cumulative'],'S1':['cumulative'],'S2':['cumulative','graph_only','fusion_only','neither'],'S3':['cumulative','contiguous_reference']}
    checked=[];records={}
    for name in ['S0','S1','S2','S3']:
        for label in labels[name]:
            result=audit(stages[name],a.gpu_uuid,label)
            checked.append({'stage':stages[name],'label':label,'verified_cases':result['verified_cases'],'samples':result['measured_samples'],'audit_script_sha256':result['audit_script_sha256']})
            records[name,label]=load(stages[name],label,a.gpu_uuid)[1]
            print(name+' '+label+': raw-data audit passed',flush=True)
    pairs=[('S0','cumulative','S1','cumulative'),('S1','cumulative','S2','cumulative'),('S0','cumulative','S2','cumulative'),
           ('S2','neither','S2','graph_only'),('S2','neither','S2','fusion_only'),('S2','neither','S2','cumulative'),
           ('S2','graph_only','S2','cumulative'),('S2','fusion_only','S2','cumulative'),
           ('S2','cumulative','S3','cumulative'),('S0','cumulative','S3','cumulative'),('S3','contiguous_reference','S3','cumulative')]
    comparisons=[]
    for bn,bl,cn,cl in pairs:
        before,after=records[bn,bl],records[cn,cl]
        require(set(before)==set(after) and len(before)==36,'Incomplete paired comparison')
        for case in before:compare(before[case],after[case])
        comparisons.append({'baseline':stages[bn],'baseline_label':bl,'stage':stages[cn],'label':cl,'paired_cases':36,'exact_greedy_tokens':True})
    correctness=[]
    for name,filename,count in [('S1','prefill_correctness.json',28),('S2','decode_correctness.json',112),('S3','paged_correctness.json',112),('S3','contiguous_correctness.json',112)]:
        root=Path('docs/performance/stages')/stages[name]/'validation';report=read(root/filename)
        require(report['status']=='passed' and report['verified_cases']==count and report['models']==4,'Incomplete correctness '+name+' '+filename)
        require(len(report['cases'])==count and all(x['greedy_tokens_match'] for x in report['cases']),'Incorrect greedy validation scope')
        if name in ['S2','S3']:
            require(set(report['modes'])=={'neither','graph_only','fusion_only','combined'},'Missing feature-switch correctness coverage')
            require(len(report['units'])==4,'Missing four-model unit coverage')
            for model in manifests[name]['models']:
                require(sum(c['model']==model['name'] for c in report['cases'])==28,'Missing model/mode correctness cases')
        if name=='S3':
            expected_mode='paged' if filename.startswith('paged_') else 'contiguous'
            require(report['kv_cache']==expected_mode and all(c['kv_cache_mode']==expected_mode for c in report['cases']),'Missing requested cache-mode validation')
            require(all(c['kv_live_pages_end']==0 for c in report['cases']),'Correctness requests retained live page mappings')
            for model in manifests[name]['models']:
                unit_text=(Path(report['artifact_directory'])/(model['name']+'_units.stdout.txt')).read_text()
                for test in ['paged_pool_growth_reuse_trim_and_context_bounds','paged_random_mapping_causal_gqa_and_layer_separation','decode_graph_fusion_switches_and_capture_reuse']:
                    require(re.search(r'\[       OK \] runtime_cuda\.'+test+r' ',unit_text) is not None,'Required page/graph unit did not actually run: '+test)

        gate_name={'prefill_correctness.json':'prefill_correctness_gate.json','decode_correctness.json':'decode_correctness_gate.json','paged_correctness.json':'paged_correctness_gate.json','contiguous_correctness.json':'contiguous_correctness_gate.json'}[filename]
        gate=read(root/gate_name)
        require(gate['status']=='passed' and gate['monitor']['valid'] and gate['gpu_uuid']==a.gpu_uuid,'Invalid correctness GPU gate')
        for model in manifests[name]['models']:
            actual=report['candidate_artifacts'][model['name']]
            for key in ['binary_sha256','library_sha256','test_binary_sha256']:require(actual[key]==model[key],'Frozen candidate differs from validated candidate')
        correctness.append({'stage':stages[name],'report':filename,'verified_cases':count,'frozen_hashes_match':True})
    sanitizer=[]
    for name,stem in [('S1','prefill_memcheck_gate'),('S2','decode_memcheck_gate'),('S3','paged_memcheck_gate')]:
        root=Path('docs/performance/stages')/stages[name]/'validation';gate=read(root/(stem+'.json'));text=(root/(stem+'.stdout.txt')).read_text()
        require(gate['status']=='passed' and gate['exit_code']==0 and gate['monitor']['valid'] and gate['gpu_uuid']==a.gpu_uuid,'Memory safety gate failed')
        require('ERROR SUMMARY: 0 errors' in text,'Sanitizer did not report zero errors')
        sanitizer.append({'stage':stages[name],'command':gate['command'],'zero_errors':True})
    # Source review anchors supplement executable/numeric/allocation evidence.
    source_checks=[('S1','source/runtime/prefill.cpp',['cublasSgemm','CUBLAS_PEDANTIC_MATH','prefill_attention']),
                   ('S2','source/runtime/decode.cpp',['cudaStreamBeginCapture','cudaGraphInstantiateWithFlags','cudaGraphLaunch','decode_residual_norm','PackedBlock']),
                   ('S3','source/runtime/kv_cache.cpp',['cudaMalloc','std::make_unique<Page>','cudaFree(page->data)','std::shuffle']),
                   ('S3','source/op/kernels/cuda/kv_cache_view.cuh',['cache.pages[token/cache.page_tokens]']),
                   ('S3','source/model/model.cpp',['initialize_kv_cache','paged_cache_','configure_kv_cache'])]
    reviewed=[]
    for name,path,tokens in source_checks:
        revision=manifests[name]['source_revision'];data=subprocess.check_output(['git','show',revision+':'+path]);text=data.decode()
        compact=re.sub(r'\s+','',text)
        require(all(re.sub(r'\s+','',token) in compact for token in tokens),'Required frozen implementation missing: '+path)
        reviewed.append({'stage':stages[name],'source_revision':revision,'file':path,'sha256':hashlib.sha256(data).hexdigest()})
    require(sum(x['verified_cases'] for x in checked)==288 and sum(x['samples'] for x in checked)==1440,'Full main/control matrix total differs')
    result={'status':'full_objective_verified','gpu_uuid':a.gpu_uuid,'run_id':a.run_id,'precision':'fp32','models':[m['name'] for m in base['models']],
            'primary_stages':4,'primary_cases':144,'all_main_and_control_cases':288,'all_measured_samples':1440,
            'raw_data_audits':checked,'paired_comparisons':comparisons,'correctness':correctness,'memory_safety':sanitizer,'frozen_source_review':reviewed,
            'audit_script_sha256':digest(Path(__file__))}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['raw_data_audits','paired_comparisons','correctness','memory_safety','frozen_source_review']},indent=2))
if __name__=='__main__':main()
