"""Check that memory claims prove real capacities, release and warm reuse."""
import struct
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_stage import validate_kv_metadata

class KVMetadataTests(unittest.TestCase):
    def sample(self):
        return {'prompt_tokens':[1]*8,'generated_tokens':[2]*8,'kv_cache_mode':'paged',
                'kv_logical_tokens':15,'kv_effective_bytes':15*192,'kv_release_ms':.02,
                'kv_page_tokens':16,'kv_reserved_bytes':16*192,'kv_peak_reserved_bytes':16*192,
                'kv_peak_mapped_bytes':16*192,'kv_table_bytes':2*struct.calcsize('P'),
                'kv_mapped_bytes':0,'kv_live_pages_end':0,'kv_free_pool_pages':1,
                'kv_page_releases':1,'kv_lifetime_allocations':1,'kv_lifetime_frees':0,
                'kv_table_clears':1,'kv_table_updates':1,'kv_page_allocations':0,'kv_page_reuses':1}
    def check(self,sample,command=[]):validate_kv_metadata(sample,command,2,32,12)
    def test_complete_paged_memory_contract(self):self.check(self.sample())
    def test_fake_capacity_or_unreleased_pages_rejected(self):
        for change in [{'kv_reserved_bytes':6144},{'kv_effective_bytes':1},{'kv_logical_tokens':16},
                       {'kv_live_pages_end':1},{'kv_mapped_bytes':192},{'kv_table_bytes':0},
                       {'kv_page_releases':0},{'kv_page_allocations':1},{'kv_page_reuses':0},
                       {'kv_table_clears':0},{'kv_lifetime_allocations':0}]:
            with self.subTest(change=change),self.assertRaises(ValueError):self.check(dict(self.sample(),**change))
    def test_mode_and_page_size_are_paired(self):
        with self.assertRaises(ValueError):self.check(self.sample(),['--kv-cache','contiguous'])
        with self.assertRaises(ValueError):self.check(self.sample(),['--kv-page-tokens','8'])
    def test_dense_reference_has_full_tensor_capacity(self):
        sample=dict(self.sample(),kv_cache_mode='contiguous',kv_page_tokens=0,kv_reserved_bytes=32*192,
                    kv_mapped_bytes=32*192,kv_table_bytes=0,kv_page_allocations=0,kv_page_reuses=0,kv_page_releases=0)
        self.check(sample,['--kv-cache','contiguous'])
        with self.assertRaises(ValueError):self.check(dict(sample,kv_reserved_bytes=16*192),['--kv-cache','contiguous'])
    def test_old_frozen_formats_remain_valid(self):validate_kv_metadata({'generated_tokens':[1]*32},[],2,32,12)

if __name__=='__main__':unittest.main()
