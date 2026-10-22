#!/usr/bin/env python3
"""Build model variants with automatic CUDA discovery and fresh-machine dependency fetching."""
import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

VARIANTS = ['llama2', 'qwen2', 'qwen3', 'llama3']
DEPENDENCIES = ['googletest', 'glog', 'armadillo', 'sentencepiece']

def find_cuda_compiler(explicit=None):
    candidates = [explicit] if explicit else [os.environ.get('CUDACXX'), shutil.which('nvcc'), '/usr/local/cuda/bin/nvcc', '/usr/local/cuda-12.1/bin/nvcc']
    for candidate in candidates:
        if candidate:
            resolved = shutil.which(candidate) or candidate
            if Path(resolved).is_file():
                return str(Path(resolved).resolve())
    raise FileNotFoundError('nvcc was not found; pass --cuda-compiler /path/to/nvcc or set CUDACXX')

def configure_command(variant, compiler, architecture, offline=False, cache=Path('build/_deps')):
    build = 'build' if variant == 'llama2' else 'build-' + variant
    command = ['cmake', '-S', '.', '-B', build, '-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release',
               '-DCMAKE_CUDA_COMPILER=' + compiler, '-DCMAKE_CUDA_ARCHITECTURES=' + architecture,
               '-DFETCHCONTENT_FULLY_DISCONNECTED=' + ('ON' if offline else 'OFF'),
               '-DFETCHCONTENT_UPDATES_DISCONNECTED=ON']
    for name in ['LLAMA3', 'QWEN2', 'QWEN3']:
        command.append('-D' + name + '_SUPPORT=' + ('ON' if variant == name.lower() else 'OFF'))
    required = DEPENDENCIES + ([] if variant == 'llama2' else ['absl', 're2', 'nlohmann_json'])
    for dependency in required:
        source = cache / (dependency + '-src')
        if source.is_dir():
            command.append('-DFETCHCONTENT_SOURCE_DIR_' + dependency.upper() + '=' + str(source.resolve()))
        elif offline:
            raise FileNotFoundError('Offline dependency is missing: ' + str(source))
    return build, command

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('variants', nargs='*', default=VARIANTS)
    p.add_argument('--jobs', type=int, default=8)
    p.add_argument('--cuda-compiler')
    p.add_argument('--cuda-arch', default=os.environ.get('CUDAARCHS', 'native'), help='native requires CMake >= 3.24; e.g. 80 for A100, 89 for RTX 4090')
    p.add_argument('--dependency-cache', type=Path, default=Path('build/_deps'))
    p.add_argument('--offline', action='store_true', help='Require pre-populated dependency source directories')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--build-only', action='store_true', help='Reuse configured directories for incremental builds (configure first after adding sources)')
    a = p.parse_args()
    if any(variant not in VARIANTS for variant in a.variants):
        p.error('Unknown variant; choose from ' + ', '.join(VARIANTS))
    if a.jobs < 1:
        p.error('--jobs must be positive')
    try:
        compiler = find_cuda_compiler(a.cuda_compiler)
        plans = []
        for variant in a.variants:
            build, command = configure_command(variant, compiler, a.cuda_arch, a.offline, a.dependency_cache)
            plans.append({'variant': variant, 'configure': command, 'build': ['cmake', '--build', build, '--target', 'llama_infer', 'test_llm', '-j', str(a.jobs)]})
    except FileNotFoundError as error:
        p.error(str(error))
    if a.dry_run:
        print(json.dumps(plans, indent=2)); return
    for plan in plans:
        log = Path('/tmp') / ('my_llm_build_' + plan['variant'] + '.log')
        with log.open('w') as output:
            for command in ([plan['build']] if a.build_only else [plan['configure'], plan['build']]):
                subprocess.run(command, stdout=output, stderr=subprocess.STDOUT, check=True)
        print(plan['variant'] + ' built; ' + str(log), flush=True)

if __name__ == '__main__':
    main()
