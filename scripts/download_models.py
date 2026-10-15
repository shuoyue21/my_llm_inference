#!/usr/bin/env python3
"""Download pinned, hash-verified model assets and export the same FP32 checkpoints."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from freeze_stage import digest

def download_file(info):
    target = Path(info['path'])
    if target.exists():
        if target.stat().st_size == info['bytes'] and digest(target) == info['sha256']:
            print('Verified existing ' + str(target), flush=True); return
        raise RuntimeError('Existing model asset differs from the pinned version: ' + str(target) + '; move it aside before downloading')
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + '.partial')
    subprocess.run(['curl', '--fail', '--location', '--retry', '3', '--connect-timeout', '20', info['url'], '--output', str(temporary)], check=True)
    if temporary.stat().st_size != info['bytes'] or digest(temporary) != info['sha256']:
        raise RuntimeError('Downloaded file failed size/SHA256 verification: ' + str(target))
    temporary.replace(target)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+')
    parser.add_argument('--no-export', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    manifest = json.loads(Path('scripts/model_downloads.json').read_text())
    names = {item['name'] for item in manifest['models']}
    if args.models and not set(args.models) <= names:
        parser.error('Unknown model; choose from ' + ', '.join(sorted(names)))
    selected = [item for item in manifest['models'] if not args.models or item['name'] in args.models]
    if args.dry_run:
        print(json.dumps(selected, indent=2)); return
    for model in selected:
        for info in model['files']:
            download_file(info)
        if model['family'] == 'llama2':
            continue
        folder = Path(model['folder'])
        (folder / 'source.json').write_text(json.dumps({'repo': model['repo'], 'revision': model['revision']}, indent=2) + '\n')
        if args.no_export:
            continue
        checkpoint = Path(model['checkpoint'])
        metadata = Path(str(checkpoint) + '.json')
        if checkpoint.exists() and digest(checkpoint) == model['checkpoint_sha256'] and metadata.exists():
            print('Verified existing exported ' + str(checkpoint), flush=True); continue
        if checkpoint.exists() and digest(checkpoint) != model['checkpoint_sha256']:
            raise RuntimeError('Existing checkpoint differs from pinned export: ' + str(checkpoint))
        subprocess.run([sys.executable, 'scripts/export_hf.py', str(folder), str(checkpoint), '--context', str(model['context'])], check=True)
        if digest(checkpoint) != model['checkpoint_sha256']:
            raise RuntimeError('Exported checkpoint SHA256 differs from the tested checkpoint: ' + str(checkpoint))
        print('Verified exported ' + str(checkpoint), flush=True)

if __name__ == '__main__':
    main()
