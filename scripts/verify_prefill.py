#!/usr/bin/env python3
"""Full per-position logits, greedy outputs and boundaries versus frozen S0.
Diagnostic timings are not performance results. Every engine invocation is serial;
run this script through run_on_empty_gpu.py with the stage's recorded GPU policy.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import struct
import time
import numpy as np
from freeze_stage import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-stage', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--artifact-dir', type=Path, default=Path('.artifacts/validation/prefill'))
    args = parser.parse_args()
    registry = json.loads(Path('scripts/model_registry.json').read_text())
    baseline = json.loads((Path('.artifacts/performance') / args.baseline_stage / 'manifest.json').read_text())
    old_models = {m['name']: m for m in baseline['models']}
    artifact = args.artifact_dir / str(time.time_ns())
    artifact.mkdir(parents=True)
    reports = []
    candidate_artifacts = {}
    for model in registry['models']:
        item = {}
        for key, relative in [('binary', 'demo/llama_infer'), ('library', 'lib/libllama.so'), ('test_binary', 'test/test_llm')]:
            path = Path(model['build']) / relative
            item[key] = str(path)
            item[key + '_sha256'] = digest(path)
        candidate_artifacts[model['name']] = item
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    report = {'status': 'running', 'purpose': 'correctness_not_performance', 'baseline_stage': args.baseline_stage,
              'artifact_directory': str(artifact), 'verification_script_sha256': digest(Path(__file__)), 'candidate_artifacts': candidate_artifacts, 'cases': reports, 'started_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def execute(command, prefix, library, json_lines=True):
        run_env = dict(env, LD_LIBRARY_PATH=str(library.resolve()) + (':' + os.environ['LD_LIBRARY_PATH'] if os.environ.get('LD_LIBRARY_PATH') else ''))
        completed = subprocess.run(command, env=run_env, capture_output=True, text=True)
        Path(str(prefix) + ('.stdout.json' if json_lines else '.stdout.txt')).write_text(completed.stdout)
        Path(str(prefix) + '.stderr.txt').write_text(completed.stderr)
        Path(str(prefix) + '.command.json').write_text(json.dumps({'command': command, 'exit_code': completed.returncode, 'environment': {key: run_env[key] for key in ['CUDA_VISIBLE_DEVICES', 'LD_LIBRARY_PATH', 'LLAMA_TEST_MODEL', 'LLAMA_TEST_TOKENIZER'] if key in run_env}}, indent=2) + '\n')
        completed.check_returncode()
        return [json.loads(line) for line in completed.stdout.splitlines() if line.strip()] if json_lines else []

    try:
        for model in registry['models']:
            frozen = old_models[model['name']]
            assert digest(frozen['binary']) == frozen['binary_sha256']
            assert digest(frozen['library']) == frozen['library_sha256']
            binary = str(Path(model['build']) / 'demo/llama_infer')
            library = Path(model['build']) / 'lib'
            old_library = Path(frozen['library']).parent
            with open(model['checkpoint'], 'rb') as source:
                vocabulary = abs(struct.unpack('<7i', source.read(28))[5])
            env['LLAMA_TEST_MODEL'] = str(Path(model['checkpoint']).resolve())
            env['LLAMA_TEST_TOKENIZER'] = str(Path(model['tokenizer']).resolve())
            execute([str(Path(model['build']) / 'test/test_llm')], artifact / (model['name'] + '_units'), library, json_lines=False)
            cases = [(1, 8), (8, 8), (32, 8), (128, 8),
                     (min(512, model['context'] - 31), 32),
                     (min(1536, model['context'] - 127), 128), (model['context'], 1)]
            for prompt, generated in cases:
                name = f"{model['name']}_p{prompt}_g{generated}"
                folder = artifact / name
                folder.mkdir()
                ids = [model['seed_tokens'][i % len(model['seed_tokens'])] for i in range(prompt)]
                token_file = folder / 'input.tokens'
                token_file.write_text(' '.join(map(str, ids)) + '\n')
                common = [model['checkpoint'], model['tokenizer'], '--device', 'cuda', '--prompt-ids', str(token_file),
                          '--max-new-tokens', str(generated), '--ignore-eos', '--json']
                runs = {}
                dumps = {}
                for mode in ['frozen_s0', 'reference', 'batched']:
                    dump = folder / (mode + '.f32')
                    command = [frozen['binary'] if mode == 'frozen_s0' else binary] + common + ['--dump-logits', str(dump)]
                    if mode != 'frozen_s0':
                        command += ['--prefill', mode]
                    results = execute(command, folder / mode, old_library if mode == 'frozen_s0' else library)
                    assert len(results) == 1
                    result = results[0]
                    assert result['prompt_tokens'] == ids and len(result['generated_tokens']) == generated
                    if mode != 'frozen_s0':
                        assert result['prefill_mode'] == mode
                    assert dump.stat().st_size == 4 * (prompt + generated - 1) * vocabulary
                    runs[mode] = result
                    dumps[mode] = dump
                assert runs['reference']['generated_tokens'] == runs['frozen_s0']['generated_tokens'], name + ': reference token mismatch'
                assert runs['batched']['generated_tokens'] == runs['frozen_s0']['generated_tokens'], name + ': batch token mismatch'
                positions = prompt + generated - 1
                expected = np.memmap(dumps['frozen_s0'], dtype='<f4', mode='r').reshape(positions, -1)
                maximum = {}
                for mode in ['reference', 'batched']:
                    actual = np.memmap(dumps[mode], dtype='<f4', mode='r').reshape(positions, -1)
                    assert actual.shape == expected.shape
                    worst = 0.0
                    for begin in range(0, positions, 32):
                        a = actual[begin:begin + 32]
                        b = expected[begin:begin + 32]
                        assert np.isfinite(a).all() and np.isfinite(b).all(), name + ': non-finite logits'
                        tolerance = 3e-4 if model['family'] == 'llama2' else 6e-4
                        np.testing.assert_allclose(a, b, atol=tolerance, rtol=tolerance, err_msg=name + ' ' + mode + ' row=' + str(begin))
                        worst = max(worst, float(np.max(np.abs(a - b))))
                    maximum[mode] = worst
                    del actual
                vocab = expected.shape[1]
                del expected
                item = {'case': name, 'model': model['name'], 'prompt_tokens': prompt, 'generated_tokens': generated,
                        'positions': positions, 'vocab_size': vocab, 'max_abs_logit_errors': maximum,
                        'greedy_tokens_match': True, 'raw_artifacts': str(folder)}
                reports.append(item)
                args.output.write_text(json.dumps(report, indent=2) + '\n')
                print(json.dumps(item), flush=True)
            # Same resident model, warmed and repeated; its workspace/cache is reused.
            repeat_folder = artifact / (model['name'] + '_resident_repeats')
            repeat_folder.mkdir()
            token_file = repeat_folder / 'input.tokens'
            ids = [model['seed_tokens'][i % len(model['seed_tokens'])] for i in range(32)]
            token_file.write_text(' '.join(map(str, ids)) + '\n')
            command = [binary, model['checkpoint'], model['tokenizer'], '--device', 'cuda', '--prompt-ids', str(token_file),
                       '--max-new-tokens', '16', '--ignore-eos', '--prefill', 'batched', '--warmup', '2', '--repeat', '3', '--json']
            repeated = execute(command, repeat_folder / 'batched', library)
            assert len(repeated) == 3 and all(r['generated_tokens'] == repeated[0]['generated_tokens'] and r['prefill_mode'] == 'batched' for r in repeated)
        for item in candidate_artifacts.values():
            for key in ['binary', 'library', 'test_binary']:
                assert digest(item[key]) == item[key + '_sha256'], 'Candidate executable changed during validation'
        report['status'] = 'passed'
        report['verified_cases'] = len(reports)
        report['models'] = len(registry['models'])
        assert report['verified_cases'] == 28 and report['models'] == 4
    except Exception as error:
        report['status'] = 'failed'
        report['error'] = repr(error)
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': report['status'], 'verified_cases': report['verified_cases'], 'models': report['models']}))

if __name__ == '__main__':
    main()
