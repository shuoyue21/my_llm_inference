"""Fresh-machine command planning and pinned-asset safeguards; no downloads or builds."""
import hashlib
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from build_variants import configure_command
from download_models import download_file

class BuildPortabilityTests(unittest.TestCase):
    def test_fresh_checkout_can_fetch_dependencies(self):
        with tempfile.TemporaryDirectory() as folder:
            build, command = configure_command('qwen3', '/opt/cuda/bin/nvcc', '80', cache=Path(folder))
            self.assertEqual(build, 'build-qwen3')
            self.assertIn('-DFETCHCONTENT_FULLY_DISCONNECTED=OFF', command)
            self.assertIn('-DCMAKE_CUDA_ARCHITECTURES=80', command)
            self.assertIn('-DQWEN3_SUPPORT=ON', command)
            self.assertIn('-DLLAMA3_SUPPORT=OFF', command)
            self.assertFalse(any('FETCHCONTENT_SOURCE_DIR_' in item for item in command))
    def test_offline_requires_all_variant_dependencies(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(FileNotFoundError):
                configure_command('qwen3', '/opt/cuda/bin/nvcc', '80', offline=True, cache=Path(folder))
    def test_offline_cache_reuse_uses_current_machine_paths(self):
        with tempfile.TemporaryDirectory() as folder:
            cache = Path(folder)
            for name in ['googletest','glog','armadillo','sentencepiece','absl','re2','nlohmann_json']:
                (cache / (name + '-src')).mkdir()
            _, command = configure_command('llama3', '/opt/cuda/bin/nvcc', 'native', offline=True, cache=cache)
            self.assertIn('-DFETCHCONTENT_FULLY_DISCONNECTED=ON', command)
            overrides = [item for item in command if 'FETCHCONTENT_SOURCE_DIR_' in item]
            self.assertEqual(len(overrides), 7)
            self.assertTrue(all(str(cache.resolve()) in item for item in overrides))

class AssetHandoffTests(unittest.TestCase):
    def test_all_registry_models_have_pinned_assets(self):
        downloads=json.loads(Path('scripts/model_downloads.json').read_text())['models']
        registry=json.loads(Path('scripts/model_registry.json').read_text())['models']
        self.assertEqual({m['name'] for m in downloads},{m['name'] for m in registry})
        for model in downloads:
            for file in model['files']:
                self.assertRegex(file['url'],r'/resolve/[0-9a-f]{40}/')
                self.assertRegex(file['sha256'],r'^[0-9a-f]{64}$')
                self.assertGreater(file['bytes'],0)
    def test_existing_correct_asset_does_not_download(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'asset';path.write_bytes(b'known data')
            info={'path':str(path),'bytes':10,'sha256':hashlib.sha256(b'known data').hexdigest()}
            with patch('download_models.subprocess.run') as command:
                download_file(info)
                command.assert_not_called()
    def test_existing_different_asset_is_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'asset';path.write_bytes(b'user data')
            info={'path':str(path),'bytes':10,'sha256':hashlib.sha256(b'known data').hexdigest()}
            with self.assertRaises(RuntimeError):download_file(info)
            self.assertEqual(path.read_bytes(),b'user data')

if __name__=='__main__':unittest.main()
