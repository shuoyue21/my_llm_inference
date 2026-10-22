#!/usr/bin/env python3
"""Summarize accepted S4-S7 records without rehashing model assets or old stages."""
import argparse
import csv
import json
from pathlib import Path
import statistics

STAGES = {
    'S4': ('S4-Forward-20261022-runtime', ['serial','continuous-c4','continuous-c8']),
    'S5': ('S5-Forward-20261022-runtime', ['prefix-off','prefix-on']),
    'S6': ('S6-Forward-20261022-runtime', ['fp32','fp16','bf16','w8a16','w4a16']),
    'S7': ('S7-Forward-20261022-runtime', ['attention-reference','attention-online']),
}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu-uuid',default='GPU-c8c002a0-861e-e0e3-3b8d-138381f9f61d')
    p.add_argument('--output',type=Path,default=Path('docs/PERFORMANCE_EXTENSIONS.md'))
    p.add_argument('--partial',action='store_true',help='Write a clearly incomplete local progress summary')
    a=p.parse_args()
    registry=json.loads(Path('scripts/model_registry.json').read_text())
    cases=[f"{m['name']}_p{n}_g{g}" for m in registry['models'] for n in registry['workloads']['prompt_tokens']
           for g in registry['workloads']['generated_tokens'] if n+g-1<=m['context']]
    records={}; missing=[]; rows=[]
    for feature,(stage,labels) in STAGES.items():
        for label in labels:
            found={}
            for case in cases:
                source_stage=stage
                path=Path('docs/performance/stages')/source_stage/label/a.gpu_uuid/(case+'.json')
                if not path.exists():missing.append(str(path));continue
                record=json.loads(path.read_text())
                if not record['accepted'] or not record['monitor']['valid'] or record['gpu_uuid']!=a.gpu_uuid:
                    raise RuntimeError('Rejected/contaminated record: '+str(path))
                if len(record['samples'])!=registry['workloads']['repeats']:raise RuntimeError('Incomplete repeats: '+str(path))
                found[case]=record
                for metric,values in record['metrics'].items():
                    rows.append(dict(stage=source_stage,label=label,case=case,metric=metric,median=values['median']))
            records[feature,label]=found
    if missing and not a.partial:raise SystemExit(f'{len(missing)} cases remain; no final summary written')
    local=Path('docs/performance/extensions');local.mkdir(parents=True,exist_ok=True)
    with (local/'metrics.csv').open('w') as out:
        writer=csv.DictWriter(out,fieldnames=['stage','label','case','metric','median']);writer.writeheader();writer.writerows(rows)
    case_count=sum(len(values) for values in records.values())
    group_count=sum(len(record['samples']) for values in records.values() for record in values.values())
    request_count=sum(len(sample['requests']) for values in records.values() for record in values.values() for sample in record['samples'])
    (local/'audit.json').write_text(json.dumps({'complete':not missing,'accepted_cases':case_count,'measured_groups':group_count,'measured_requests':request_count,'gpu_uuid':a.gpu_uuid,'missing':missing},indent=2)+'\n')
    def ratios_for(feature, baseline, label, model, metric='output_tokens_per_second', min_prompt=0):
        before=records[feature,baseline]
        return [record['metrics'][metric]['median']/before[case]['metrics'][metric]['median']
                for case,record in records[feature,label].items() if case in before and record['model']==model and record['prompt_tokens']>=min_prompt]
    def interval_values(values):
        return f'{min(values):.3f}–{max(values):.3f}' if values else '待采集'
    lines=['# 第 4–7 项性能记录','', '状态：'+('未完成；以下仅为已有数据。' if missing else '完整矩阵已完成。'),'', f'{case_count} 个场景、{group_count} 组正式采样、{request_count} 个正式请求。', '',
           '所有比较来自同一 GPU、固定模型资产和 token 输入，每场景预热 2 组、测量 5 组。原始结果只保存在本地忽略目录。', '',
           'S4 使用构建结束后的统一修复版 native 串行对照与批处理路径；每组 16 请求，比较串行与并发 4/8；S5 每组 4 请求、并发 4；S6/S7 使用单请求隔离精度和 Attention 的影响，均使用同版本配对对照。', '',
           'S4 对比原生串行 generate 和连续后端的整体请求处理路径；原生 generate 保留最后的文本解码，连续后端返回 token IDs。吞吐包含请求处理墙钟时间，排除加载和 JSON 输出时间，不将该比值当成独立算子的加速比。', '',
           '尾延迟为每组请求经验分位数的 5 组中位数；样本量有限，不作为生产环境 p99/SLO 证明。S5 使用重复前缀和固定 512 MiB 缓存（覆盖 Qwen3 全 2048-token KV 的 448 MiB），是有利于命中的受控场景。', '',
           '| 阶段 | 对照 → 实现 | 有效场景 | 输出吞吐倍数范围 | TTFT p95 倍数范围 | 总延迟 p95 倍数范围 | 吞吐退化 |',
           '|---|---|---:|---:|---:|---:|---:|']
    if not missing:
        c8=[]
        for case,record in records['S4','continuous-c8'].items():
            c8.append(record['metrics']['output_tokens_per_second']['median']/records['S4','serial'][case]['metrics']['output_tokens_per_second']['median'])
        prefix_speed=[records['S5','prefix-off'][case]['metrics']['ttft_ms']['median']/record['metrics']['ttft_ms']['median'] for case,record in records['S5','prefix-on'].items()]
        long_attention=[]
        for model in registry['models']:
            long_attention.extend(ratios_for('S7','attention-reference','attention-online',model['name'],min_prompt=512))
        findings=['主要结果：','',
            f'- 并发 8 的吞吐在 {sum(value>1 for value in c8)}/36 场景提高，最高 {max(c8):.2f} 倍；长输入仍有回退。',
            f'- 重复前缀受控负载最高 TTFT 加速 {max(prefix_speed):.2f} 倍，短 prompt 不一定受益。',
            '- FP16/BF16/W8/W4 都有速度退化场景；W4 的固定文本困惑度为 FP32 的约 1.26–3.68 倍，不能视为无损。',
            f'- Online Attention 将全局 scores 工作区降为零，但已测长输入吞吐为参考的 {min(long_attention):.3f}–{max(long_attention):.3f} 倍，默认保留 reference。','']
        lines[6:6]=findings
    comparisons=[('S4','serial','continuous-c4'),('S4','serial','continuous-c8'),('S5','prefix-off','prefix-on')]
    comparisons += [('S6','fp32',precision) for precision in ['fp16','bf16','w8a16','w4a16']]
    comparisons += [('S7','attention-reference','attention-online')]
    for feature,baseline,label in comparisons:
        before,after=records[feature,baseline],records[feature,label]
        ratios={name:[] for name in ['output_tokens_per_second','ttft_p95_ms','latency_p95_ms']}
        for case in set(before)&set(after):
            b,s=before[case],after[case]
            for field in ['gpu_uuid','gpu_policy','model','warmups','repeats']:
                if b[field]!=s[field]:raise RuntimeError('Unpaired metadata: '+field)
            if b['samples'][0]['prompt_tokens']!=s['samples'][0]['prompt_tokens']:raise RuntimeError('Mismatched input IDs')
            if b['samples'][0]['request_count']!=s['samples'][0]['request_count']:raise RuntimeError('Mismatched request counts')
            if feature!='S6' and [r['generated_tokens'] for r in b['samples'][0]['requests']] != [r['generated_tokens'] for r in s['samples'][0]['requests']]:
                raise RuntimeError('FP32 paired outputs differ: '+case)
            for metric in ratios:ratios[metric].append(s['metrics'][metric]['median']/b['metrics'][metric]['median'])
        def interval(metric):
            values=ratios[metric]
            return f'{min(values):.3f}–{max(values):.3f}' if values else '待采集'
        values=ratios['output_tokens_per_second']
        lines.append(f'| {feature} | {baseline} → {label} | {len(values)}/36 | {interval("output_tokens_per_second")} | {interval("ttft_p95_ms")} | {interval("latency_p95_ms")} | {sum(value<1 for value in values)} |')
    lines += ['', '| 模型 | 并发度 | 场景 | 吞吐倍数范围 | TTFT p95 倍数范围 | 吞吐退化 |', '|---|---:|---:|---:|---:|---:|']
    for model in registry['models']:
        name=model['name']
        for concurrency in [4,8]:
            label='continuous-c'+str(concurrency)
            speed=ratios_for('S4','serial',label,name)
            ttft=ratios_for('S4','serial',label,name,'ttft_p95_ms')
            lines.append(f'| {name} | {concurrency} | {len(speed)} | {interval_values(speed)} | {interval_values(ttft)} | {sum(value<1 for value in speed)} |')
    quality_path=Path('docs/performance/development/S6-Forward-20261022-runtime/quality-fixed/quality.json')
    if quality_path.exists():
        quality=json.loads(quality_path.read_text())
        lines += ['', '| 模型 | 精度 | Packed 权重 MiB | 加载显存增量 MiB | 吞吐倍数范围 | NLL 差值 | 困惑度倍数 | Top-1 一致率 |', '|---|---|---:|---:|---:|---:|---:|---:|']
        for model,precisions in quality['models'].items():
            for precision,values in precisions.items():
                if precision=='native_fp32_control':continue
                loads=[record['metrics']['gpu_load_memory_bytes']['median'] for record in records['S6',precision].values() if record['model']==model]
                speed=interval_values(ratios_for('S6','fp32',precision,model))
                loaded=f'{statistics.median(loads)/1024**2:.2f}' if loads else '待采集'
                lines.append(f"| {model} | {precision} | {values['packed_weight_bytes']/1024**2:.2f} | {loaded} | {speed} | {values['nll_delta']:+.5f} | {values['perplexity_ratio']:.4f} | {values['top1_agreement']:.2%} |")
    lines += ['', '加载显存增量来自加载前后 cudaMemGetInfo 的全卡可用显存差，含后端加载工作区，不含随后增长的请求 KV；允许的图形驻留变化及整卡峰值另在原始 NVML 记录中。Packed 权重列为该模型实际分配的权重字节数。', '',
              '| 模型 | 前缀命中率（全部正式请求） | 最大驻留前缀 MiB | 参考 scores 工作区 MiB | Online scores 工作区 MiB | Online 长输入吞吐倍数 |',
              '|---|---:|---:|---:|---:|---:|']
    for model in registry['models']:
        name=model['name']
        prefix=[sample for record in records['S5','prefix-on'].values() if record['model']==name for sample in record['samples']]
        lookups=sum(sample['prefix_lookups'] for sample in prefix)
        hit=f"{sum(sample['prefix_hits'] for sample in prefix)/lookups:.2%}" if lookups else '待采集'
        resident=f"{max(sample['prefix_cache_bytes'] for sample in prefix)/1024**2:.2f}" if prefix else '待采集'
        workspaces=[]
        for mode in ['reference','online']:
            samples=[sample for record in records['S7','attention-'+mode].values() if record['model']==name for sample in record['samples']]
            workspaces.append(f"{max(sample['attention_workspace_bytes'] for sample in samples)/1024**2:.2f}" if samples else '待采集')
        long_speed=interval_values(ratios_for('S7','attention-reference','attention-online',name,min_prompt=512)) if model['context']>=512 else '不适用'
        lines.append(f'| {name} | {hit} | {resident} | {workspaces[0]} | {workspaces[1]} | {long_speed} |')
    lines += ['', '吞吐倍数越大越好；延迟倍数越小越好。完整逐场景 CSV 为本地 `docs/performance/extensions/metrics.csv`。', '',
              '低精度仅对 embedding 和投影矩阵/乘法使用 FP16/BF16 或分组 W8A16/W4A16；累加、规范化与 KV 保留 FP32。',
              '量化小批 decode 直接读取 packed 权重，大批 prefill 临时展开当前矩阵为 FP16；实际显存指标包括此临时空间。',
              '精度损失使用相同固定文本的 teacher forcing logits、NLL、KL 和 top-1 一致率，不将该小语料当成通用任务准确率。', '',
              '保留串行原生推理、连续 KV、Graph/fusion、分页和参考 Attention 路径。原始 JSON、NVML、logits、冻结程序均不进入 Git。']
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text('\n'.join(lines)+'\n')
    print(f'Wrote {a.output}; missing cases: {len(missing)}')


if __name__=='__main__':main()
