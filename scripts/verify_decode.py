#!/usr/bin/env python3
"""Validate all four decode ablations against the proven frozen-S0 logits.
Reuses the full 28-case S0 oracle retained by S1 validation; it does not rerun or
merge performance baselines. Diagnostic times and I/O are not performance data.
Run through run_on_empty_gpu.py with the recorded stage GPU policy.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import re
import struct
import subprocess
import time
import numpy as np
from freeze_stage import digest

MODES = [('neither', False, False), ('graph_only', True, False),
         ('fusion_only', False, True), ('combined', True, True)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--oracle-report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--kv-cache', choices=['auto','paged','contiguous'], default='auto')
    parser.add_argument('--artifact-dir', type=Path, default=Path('.artifacts/validation/decode'))
    args = parser.parse_args()
    oracle = json.loads(args.oracle_report.read_text())
    assert oracle['status'] == 'passed' and oracle['verified_cases'] == 28 and oracle['models'] == 4
    registry = json.loads(Path('scripts/model_registry.json').read_text())
    baseline = json.loads((Path('.artifacts/performance') / oracle['baseline_stage'] / 'manifest.json').read_text())
    assert [m['name'] for m in registry['models']] == [m['name'] for m in baseline['models']]
    for model in baseline['models']:
        assert digest(model['binary']) == model['binary_sha256']
        assert digest(model['library']) == model['library_sha256']
    candidates = {}
    for model in registry['models']:
        item = {}
        for key, relative in [('binary', 'demo/llama_infer'), ('library', 'lib/libllama.so'), ('test_binary', 'test/test_llm')]:
            path = Path(model['build']) / relative
            item[key] = str(path)
            item[key + '_sha256'] = digest(path)
        candidates[model['name']] = item
    artifact = args.artifact_dir / str(time.time_ns())
    artifact.mkdir(parents=True)
    reports, units = [], []
    report = {'status': 'running', 'purpose': 'correctness_not_performance',
              'baseline_stage': oracle['baseline_stage'], 'oracle_report': str(args.oracle_report),
              'oracle_report_sha256': digest(args.oracle_report), 'verification_script_sha256': digest(Path(__file__)),
              'candidate_artifacts': candidates, 'kv_cache': args.kv_cache, 'artifact_directory': str(artifact), 'units': units,
              'modes': [m[0] for m in MODES], 'cases': reports,
              'started_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')

    def save():
        args.output.write_text(json.dumps(report, indent=2) + '\n')

    def execute(command, prefix, model):
        library = Path(model['build']) / 'lib'
        run_env = dict(env, LD_LIBRARY_PATH=str(library.resolve()) + (':' + os.environ['LD_LIBRARY_PATH'] if os.environ.get('LD_LIBRARY_PATH') else ''),
                       LLAMA_TEST_MODEL=str(Path(model['checkpoint']).resolve()), LLAMA_TEST_TOKENIZER=str(Path(model['tokenizer']).resolve()), LLAMA_TEST_KV_MODE=args.kv_cache)
        completed = subprocess.run(command, env=run_env, capture_output=True, text=True)
        Path(str(prefix) + '.stdout.txt').write_text(completed.stdout)
        Path(str(prefix) + '.stderr.txt').write_text(completed.stderr)
        Path(str(prefix) + '.command.json').write_text(json.dumps({'command': command, 'exit_code': completed.returncode,
            'environment': {k: run_env[k] for k in ['CUDA_VISIBLE_DEVICES', 'LD_LIBRARY_PATH', 'LLAMA_TEST_MODEL', 'LLAMA_TEST_TOKENIZER'] if k in run_env}}, indent=2) + '\n')
        completed.check_returncode()
        return completed.stdout

    try:
        # Fail early on primitive, cursor, switching or same-model reuse bugs.
        for model in registry['models']:
            output = execute([candidates[model['name']]['test_binary']], artifact / (model['name'] + '_units'), model)
            assert '[  SKIPPED ]' not in output
            count = int(re.search(r'\[  PASSED  \] (\d+) tests', output).group(1))
            units.append({'model': model['name'], 'tests_passed': count})
            save()
            print(json.dumps(units[-1]), flush=True)
        for model in registry['models']:
            expected_cases = [c for c in oracle['cases'] if c['model'] == model['name']]
            assert len(expected_cases) == 7
            with open(model['checkpoint'], 'rb') as source:
                vocabulary = abs(struct.unpack('<7i', source.read(28))[5])
            for case in expected_cases:
                name = case['case']
                prompt, generated = case['prompt_tokens'], case['generated_tokens']
                positions = prompt + generated - 1
                old = Path(case['raw_artifacts'])
                oracle_result = json.loads((old / 'frozen_s0.stdout.json').read_text())
                ids = [model['seed_tokens'][i % len(model['seed_tokens'])] for i in range(prompt)]
                assert oracle_result['prompt_tokens'] == ids and len(oracle_result['generated_tokens']) == generated
                assert list(map(int, (old / 'input.tokens').read_text().split())) == ids
                oracle_dump = old / 'frozen_s0.f32'
                assert oracle_dump.stat().st_size == positions * vocabulary * 4
                oracle_sha = digest(oracle_dump)
                expected = np.memmap(oracle_dump, dtype='<f4', mode='r').reshape(positions, vocabulary)
                folder = artifact / name
                folder.mkdir()
                token_file = folder / 'input.tokens'
                token_file.write_text(' '.join(map(str, ids)) + '\n')
                for mode, graph, fusion in MODES:
                    dump = folder / (mode + '.f32')
                    command = [candidates[model['name']]['binary'], model['checkpoint'], model['tokenizer'],
                               '--device', 'cuda', '--prompt-ids', str(token_file), '--max-new-tokens', str(generated),
                               '--ignore-eos', '--kv-cache', args.kv_cache, '--prefill', 'batched', '--decode', 'optimized',
                               '--cuda-graph', 'on' if graph else 'off', '--fusion', 'on' if fusion else 'off',
                               '--dump-logits', str(dump), '--json']
                    output = execute(command, folder / mode, model)
                    result = json.loads(output)
                    assert result['prompt_tokens'] == ids and result['prefill_mode'] == 'batched'
                    assert result['generated_tokens'] == oracle_result['generated_tokens'], name + ' ' + mode + ': greedy output differs'
                    assert dump.stat().st_size == positions * vocabulary * 4
                    assert result['kv_cache_mode'] == ('contiguous' if args.kv_cache == 'contiguous' else 'paged')
                    assert result['kv_logical_tokens'] == positions
                    assert result['kv_reserved_bytes'] >= result['kv_effective_bytes']
                    if result['kv_cache_mode'] == 'paged':
                        assert result['kv_live_pages_end'] == result['kv_mapped_bytes'] == 0
                        assert result['kv_free_pool_pages'] == result['kv_page_releases']
                        assert result['kv_table_bytes'] > 0 and result['kv_table_clears'] == 1
                    else:
                        assert result['kv_table_bytes'] == 0 and result['kv_mapped_bytes'] == result['kv_reserved_bytes']
                    if generated > 1:
                        assert result['decode_mode'] == ('cuda_graph' if graph else 'cuda_eager')
                        assert result['fusion_enabled'] == fusion and result['decode_instrumented']
                        if graph:
                            assert result['decode_graph_launches'] == generated - 1
                            assert result['decode_host_submissions'] == 1 + 2 * (generated - 1)
                            assert result['decode_graph_captures_total'] == 1 and result['graph_capture_ms'] > 0
                            assert result['decode_graph_nodes'] > 0 and result['decode_graph_kernel_nodes'] > 0
                        else:
                            assert result['decode_graph_launches'] == result['decode_graph_captures_total'] == 0
                            assert result['decode_host_submissions'] > 1 + 2 * (generated - 1)
                        assert (result['decode_packed_weight_bytes'] > 0) == fusion
                    else:
                        assert result['decode_mode'] == 'none' and not result['decode_instrumented']
                        assert result['decode_graph_launches'] == 0
                    actual = np.memmap(dump, dtype='<f4', mode='r').reshape(positions, vocabulary)
                    tolerance = 3e-4 if model['family'] == 'llama2' else 6e-4
                    maximum = 0.0
                    for begin in range(0, positions, 32):
                        a, b = actual[begin:begin + 32], expected[begin:begin + 32]
                        assert np.isfinite(a).all() and np.isfinite(b).all()
                        np.testing.assert_allclose(a, b, atol=tolerance, rtol=tolerance, err_msg=name + ' ' + mode + ' row=' + str(begin))
                        maximum = max(maximum, float(np.max(np.abs(a - b))))
                    item = {'case': name, 'model': model['name'], 'mode': mode, 'positions': positions,
                            'max_abs_logit_error': maximum, 'greedy_tokens_match': True,
                            'oracle_dump': str(oracle_dump), 'oracle_sha256': oracle_sha,
                            'raw_artifacts': str(folder), 'graph_capture_ms': result['graph_capture_ms'],
                            'graph_warmup_ms': result['graph_warmup_ms'], 'decode_setup_ms': result['decode_setup_ms'],
                            'host_submissions': result['decode_host_submissions'], 'graph_launches': result['decode_graph_launches'],
                            'graph_nodes': result['decode_graph_nodes'], 'packed_weight_bytes': result['decode_packed_weight_bytes'],
                            'kv_cache_mode': result['kv_cache_mode'], 'kv_reserved_bytes': result['kv_reserved_bytes'],
                            'kv_effective_bytes': result['kv_effective_bytes'], 'kv_page_allocations': result['kv_page_allocations'],
                            'kv_live_pages_end': result['kv_live_pages_end']}
                    reports.append(item)
                    save()
                    print(json.dumps(item), flush=True)
                    del actual
                del expected
        for item in candidates.values():
            for key in ['binary', 'library', 'test_binary']:
                assert digest(item[key]) == item[key + '_sha256'], 'Candidate changed during verification'
        assert len(reports) == 112 and len(units) == 4
        report.update(status='passed', verified_cases=112, models=4)
    except Exception as error:
        report.update(status='failed', error=repr(error))
        raise
    finally:
        save()
    print(json.dumps({'status': 'passed', 'verified_cases': 112, 'models': 4}))

if __name__ == '__main__':
    main()
