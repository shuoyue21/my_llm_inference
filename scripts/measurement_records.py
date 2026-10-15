"""Preserve rejected attempts and identify the exact measurement harness."""
import hashlib
import json
import shutil
import time
from pathlib import Path

METRIC_DEFINITION_VERSION = 1

def harness_provenance():
    names = ['benchmark_suite.py', 'empty_gpu.py', 'measurement_records.py', 'run_on_empty_gpu.py', 'gpu_policy.py']
    root = Path(__file__).resolve().parent
    return {'metric_definition_version': METRIC_DEFINITION_VERSION,
            'files': {str(root / name): hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}}

def archive_attempt(record_path):
    """Move a prior result and its raw evidence together; never discard a measurement."""
    record_path = Path(record_path)
    folder = record_path.parent / 'attempts' / record_path.stem / str(time.time_ns())
    folder.mkdir(parents=True)
    files = [record_path, *record_path.parent.glob(record_path.stem + '.*')]
    moved = []
    for source in dict.fromkeys(files):
        if source.is_file():
            shutil.move(str(source), str(folder / source.name))
            moved.append(source.name)
    (folder / 'archive.json').write_text(json.dumps({'original_record': str(record_path), 'files': moved}, indent=2) + '\n')
    return folder

def prepare_existing_result(path, source_digest, gpu_uuid, extra_args):
    path = Path(path)
    if not path.exists():
        return False
    previous = json.loads(path.read_text())
    if (previous.get('source_digest') != source_digest or previous.get('gpu_uuid') != gpu_uuid
            or previous.get('extra_args', []) != extra_args):
        raise RuntimeError('Existing measurement has different provenance: ' + str(path))
    if previous.get('accepted') and previous.get('monitor', {}).get('valid'):
        return True
    if previous.get('monitor', {}).get('valid') is False:
        archive_attempt(path)
        return False
    raise RuntimeError('Prior measurement failed independently of GPU contamination: ' + str(path))
