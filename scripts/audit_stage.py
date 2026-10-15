#!/usr/bin/env python3
"""Verify a full frozen 36-case stage against its raw outputs and NVML traces."""
import argparse
import gzip
import json
import math
import struct
from pathlib import Path
from benchmark_suite import summarize
from empty_gpu import GpuMonitor
from freeze_stage import digest
from gpu_policy import normalize_policy


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(path):
    return json.loads(Path(path).read_text())


def validate_decode_metadata(sample, command, measured=True):
    """Check the explicit Graph/fusion contract without invoking a GPU."""
    if 'decode_mode' not in sample:  # Immutable S0/S1 formats.
        return
    def option(name, default):
        result = default
        for i in range(len(command) - 1):
            if command[i] == name:
                result = command[i + 1]
        return result
    count = len(sample['generated_tokens']) - 1
    reference = option('--decode', 'auto') == 'reference'
    if count <= 0 or reference:
        require(sample['decode_mode'] == ('none' if count <= 0 else 'reference'), 'Incorrect reference/empty decode mode')
        require(not sample['decode_instrumented'] and not sample['fusion_enabled'], 'Reference decode claimed optimized execution')
        require(sample['decode_graph_launches'] == sample['decode_host_submissions'] == 0, 'Reference decode has optimized counters')
        return
    graph = option('--cuda-graph', 'on') == 'on'
    fusion = option('--fusion', 'on') == 'on'
    require(sample['decode_mode'] == ('cuda_graph' if graph else 'cuda_eager'), 'Requested/actual Graph mode differs')
    require(sample['decode_instrumented'] and sample['fusion_enabled'] == fusion, 'Requested/actual fusion differs')
    require(sample['decode_workspace_bytes'] > 0, 'Missing stable decode workspace')
    require((sample['decode_packed_weight_bytes'] > 0) == fusion, 'Packed weight residency differs from selected mode')
    if graph:
        require(sample['decode_graph_launches'] == count, 'Not one Graph replay per decode token')
        require(sample['decode_host_submissions'] == 1 + 2 * count, 'Unexpected Graph work-submission count')
        require(sample['decode_graph_captures_total'] == 1 and sample['decode_graph_nodes'] > 0 and sample['decode_graph_kernel_nodes'] > 0, 'Missing or repeated graph capture')
        require(sample['graph_capture_total_ms'] > 0 and sample['graph_warmup_total_ms'] > 0, 'Cold capture/warmup cost was lost')
    else:
        require(sample['decode_graph_launches'] == sample['decode_graph_captures_total'] == 0, 'Eager mode launched/captured a Graph')
        require(sample['decode_graph_nodes'] == sample['decode_graph_kernel_nodes'] == 0, 'Eager mode reports active Graph nodes')
        require(sample['decode_host_submissions'] > 1 + 2 * count, 'Eager mode has implausible submission count')
    if measured:
        require(sample['graph_capture_ms'] == sample['graph_warmup_ms'] == 0, 'Capture/warmup was repeated after resident warmups')


def validate_kv_metadata(sample, command, layers, context, kv_dim, measured=True):
    if 'kv_cache_mode' not in sample:
        return
    def option(name, default):
        value = default
        for i in range(len(command) - 1):
            if command[i] == name:
                value = command[i + 1]
        return value
    logical = len(sample['prompt_tokens']) + len(sample['generated_tokens']) - 1 if sample['generated_tokens'] else 0
    token_bytes = 2 * layers * kv_dim * 4
    require(sample['kv_logical_tokens'] == logical, 'Incorrect effective KV token count')
    require(sample['kv_effective_bytes'] == logical * token_bytes, 'Incorrect effective KV bytes')
    require(math.isfinite(sample['kv_release_ms']) and sample['kv_release_ms'] >= 0, 'Invalid KV release cost')
    paged = option('--kv-cache', 'auto') != 'contiguous'
    require(sample['kv_cache_mode'] == ('paged' if paged else 'contiguous'), 'Requested/actual KV mode differs')
    if paged:
        size = sample['kv_page_tokens']
        require(0 < size <= context and size == int(option('--kv-page-tokens', '16')), 'Unexpected physical page size')
        pages = (logical + size - 1) // size
        physical = pages * size * token_bytes
        require(sample['kv_reserved_bytes'] == sample['kv_peak_reserved_bytes'] == sample['kv_peak_mapped_bytes'] == physical, 'Incorrect paged CUDA capacity')
        require(sample['kv_table_bytes'] == ((context + size - 1) // size) * struct.calcsize('P'), 'Incorrect device page-table size')
        require(sample['kv_mapped_bytes'] == sample['kv_live_pages_end'] == 0, 'Request pages were not released')
        require(sample['kv_free_pool_pages'] == sample['kv_page_releases'] == pages, 'Page release/free-pool mismatch')
        require(sample['kv_lifetime_allocations'] == pages and sample['kv_lifetime_frees'] == 0, 'Unexpected page allocation/trim behavior')
        require(sample['kv_table_clears'] == 1 and 0 < sample['kv_table_updates'] <= pages, 'Page table was not maintained')
        if measured:
            require(sample['kv_page_allocations'] == 0 and sample['kv_page_reuses'] == pages, 'Resident warmups did not establish reusable pages')
    else:
        physical = context * token_bytes
        require(sample['kv_reserved_bytes'] == sample['kv_mapped_bytes'] == physical, 'Dense reference capacity changed')
        require(sample['kv_table_bytes'] == sample['kv_page_tokens'] == 0, 'Dense cache claimed page storage')
        require(sample['kv_page_allocations'] == sample['kv_page_reuses'] == sample['kv_page_releases'] == 0, 'Dense reference used page pool')


def audit(stage, uuid, label='cumulative'):
    root = Path('docs/performance/stages') / stage
    manifest = load(root / 'manifest.json')
    snapshot = Path('.artifacts/performance') / stage
    require(manifest == load(snapshot / 'manifest.json'), 'Published/frozen manifests differ')
    registry = load('scripts/model_registry.json')
    require(manifest['precision'] == registry['precision'] == 'fp32', 'Precision changed')
    require(manifest['workloads'] == registry['workloads'], 'Required workload settings changed')
    require([m['name'] for m in manifest['models']] == [m['name'] for m in registry['models']], 'Required four models changed')
    pinned = {m['name']: m for m in load('scripts/model_downloads.json')['models']}
    policy = normalize_policy(manifest.get('gpu_policy'))
    monitor = object.__new__(GpuMonitor)
    monitor.policy = policy
    monitor.max_idle_bytes = 128 * 1024**2
    monitor.max_idle_util = 1
    workload = manifest['workloads']
    expected = {f"{m['name']}_p{p}_g{g}": (m, p, g) for m in manifest['models']
                for p in workload['prompt_tokens'] for g in workload['generated_tokens']
                if p + g - 1 <= m['context']}
    require(len(expected) == 36, 'Full required matrix is not 36 cases')
    data = root / label / uuid
    summary = load(data / 'summary.json')
    require(summary['status'] == 'complete' and not summary.get('selected_models'), 'Full matrix is incomplete')
    require(set(summary['completed_cases']) == set(expected), 'Completed matrix differs from required cases')
    for model, registered in zip(manifest['models'], registry['models']):
        for key in ['name', 'family', 'context', 'seed_tokens', 'checkpoint', 'tokenizer']:
            require(model[key] == registered[key], f"Model configuration changed: {model['name']} {key}")
        pin = pinned[model['name']]
        golden = pin.get('checkpoint_sha256') or next(f['sha256'] for f in pin['files'] if f['path'] == model['checkpoint'])
        require(model['artifacts'][model['checkpoint']]['sha256'] == golden, 'Checkpoint differs from pinned model: ' + model['name'])
        for key, hash_key in [('binary', 'binary_sha256'), ('library', 'library_sha256'), ('test_binary', 'test_binary_sha256')]:
            require(digest(model[key]) == model[hash_key], 'Frozen file hash mismatch: ' + model[key])
        for path, metadata in {**model['dependencies'], **model['artifacts']}.items():
            require(digest(path) == metadata['sha256'], 'Model/dependency hash mismatch: ' + path)
        for suffix in ['_gpu_units_gate.json', '_gpu_gate.json']:
            gate = load(root / 'validation' / (model['name'] + suffix))
            require(gate['status'] == 'passed' and gate['monitor']['valid'], 'GPU correctness gate did not pass')
            require(gate['gpu_uuid'] == uuid, 'GPU correctness was performed on another GPU')
            require(normalize_policy(gate.get('gpu_policy')) == policy, 'GPU correctness policy differs')
    reports = []
    for name, (model, prompt, generated) in sorted(expected.items()):
        record = load(data / (name + '.json'))
        require(record['accepted'] and record['monitor']['valid'], 'Rejected record: ' + name)
        for key, value in [('stage', stage), ('gpu_uuid', uuid), ('precision', 'fp32'), ('model', model['name']),
                           ('source_digest', manifest['source_digest']), ('source_revision', manifest['source_revision']),
                           ('prompt_tokens', prompt), ('generated_tokens', generated), ('warmups', workload['warmups']),
                           ('repeats', workload['repeats']), ('cpu_threads', workload['cpu_threads'])]:
            require(record[key] == value, name + ': unpaired field ' + key)
        require(normalize_policy(record.get('gpu_policy')) == policy, name + ': GPU policy differs')
        require(record.get('extra_args', []) == summary.get('extra_args', []) and record.get('label', 'cumulative') == label, name + ': benchmark modes differ')
        require('--ignore-eos' in record['command'] and '--quant' not in record['command'], name + ': fixed FP32 generation flags changed')
        samples = record['samples']
        with open(model['checkpoint'], 'rb') as source:
            dim, _, layers, heads, kv_heads, _, context = struct.unpack('<7i', source.read(28))
        kv_dim = dim * kv_heads // heads
        require(len(samples) == workload['repeats'] == 5, name + ': not five measured samples')
        ids = [model['seed_tokens'][i % len(model['seed_tokens'])] for i in range(prompt)]
        require(list(map(int, (data / (name + '.tokens')).read_text().split())) == ids, name + ': raw token file changed')
        raw = [json.loads(line) for line in (data / (name + '.stdout.jsonl')).read_text().splitlines() if line.strip()]
        require(raw == samples, name + ': raw stdout differs from recorded samples')
        for index, sample in enumerate(samples):
            require(sample['device'] == 'cuda' and sample['context_length'] == model['context'], name + ': device/context changed')
            validate_decode_metadata(sample, record['command'])
            validate_kv_metadata(sample, record['command'], layers, context, kv_dim)
            require(sample['iteration'] == index and sample['warmup_requests'] == 2, name + ': iterations/warmups changed')
            require(sample['prompt_tokens'] == ids and len(sample['generated_tokens']) == generated, name + ': incorrect token counts')
            require(sample['generated_tokens'] == samples[0]['generated_tokens'], name + ': nondeterministic greedy tokens')
            require(len(sample['decode_token_ms']) == generated - 1, name + ': decode latency count differs')
            values = [sample[k] for k in ['load_ms', 'tokenization_ms', 'ttft_ms', 'prefill_ms', 'decode_ms', 'total_ms', 'decode_tokens_per_second']]
            values += sample['decode_token_ms']
            require(all(math.isfinite(x) and x >= 0 for x in values), name + ': invalid numeric samples')
        require(record['metrics'] == summarize(samples, prompt), name + ': aggregates differ from raw samples')
        with gzip.open(record['telemetry_file'], 'rt') as stream:
            trace = json.load(stream)
        require(trace['valid'] and not trace['errors'] and not trace['foreign_processes'], name + ': invalid raw telemetry')
        require(normalize_policy(trace.get('gpu_policy')) == policy, name + ': raw telemetry policy differs')
        observations = trace['samples']
        require(bool(observations), name + ': telemetry is empty')
        allowed = set(trace.get('observed_owned_pids', [])) | {trace['allowed_pid']}
        for sample in observations:
            require(sample['uuid'] == uuid, name + ': telemetry from another GPU')
            require(not [p for p in sample['processes'] if p['pid'] not in allowed and not monitor.background_graphics(p)], name + ': foreign client in raw telemetry')
        gaps = [(b['monotonic_ns'] - a['monotonic_ns']) / 1e6 for a, b in zip(observations, observations[1:])]
        require(max(gaps, default=0) == trace['max_observation_gap_ms'] == record['monitor']['max_observation_gap_ms'], name + ': telemetry gap mismatch')
        peak = max(s['memory_used_bytes'] for s in observations)
        require(peak == trace['peak_sampled_device_memory_bytes'] == record['monitor']['peak_sampled_device_memory_bytes'], name + ': memory peak mismatch')
        gate = [sample for sample in record['gate'] if sample['uuid'] == uuid]
        require(len(gate) >= 6 and all(monitor.idle(sample) for sample in gate[-6:]), name + ': consecutive idle gate did not pass')
        reports.append({'case': name, 'samples': len(samples), 'telemetry_samples': len(observations),
                        'max_observation_gap_ms': max(gaps, default=0), 'ttft_ms': record['metrics']['ttft_ms']['median'],
                        'tpot_ms': record['metrics']['tpot_ms']['median'], 'decode_tokens_per_second': record['metrics']['decode_tokens_per_second']['median']})
    return {'status': 'complete_verified', 'audit_script_sha256': digest(Path(__file__)), 'stage': stage, 'gpu_uuid': uuid, 'gpu_policy': policy,
            'source_revision': manifest['source_revision'], 'source_digest': manifest['source_digest'],
            'required_cases': len(expected), 'verified_cases': len(reports), 'measured_samples': sum(r['samples'] for r in reports), 'cases': reports}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--label', default='cumulative')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = audit(args.stage, args.gpu_uuid, args.label)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'cases'}, indent=2))

if __name__ == '__main__':
    main()
