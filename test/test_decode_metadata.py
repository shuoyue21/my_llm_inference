"""Reject fake/mixed Graph telemetry while retaining immutable older formats."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_stage import validate_decode_metadata

class DecodeMetadataTests(unittest.TestCase):
    def sample(self):
        return {'generated_tokens':[1]*32,'decode_mode':'cuda_graph','decode_instrumented':True,'fusion_enabled':True,
                'decode_workspace_bytes':1024,'decode_packed_weight_bytes':2048,'decode_graph_launches':31,
                'decode_host_submissions':63,'decode_graph_captures_total':1,'decode_graph_nodes':100,
                'decode_graph_kernel_nodes':90,'graph_capture_total_ms':2,'graph_warmup_total_ms':1,
                'graph_capture_ms':0,'graph_warmup_ms':0}
    def test_real_warmed_graph_contract(self):validate_decode_metadata(self.sample(),[])
    def test_claimed_graph_needs_nodes_and_replays(self):
        for change in [{'decode_graph_nodes':0},{'decode_graph_launches':0},{'decode_graph_captures_total':2},{'graph_capture_total_ms':0},{'decode_host_submissions':0},{'graph_capture_ms':1}]:
            with self.subTest(change=change),self.assertRaises(ValueError):validate_decode_metadata(dict(self.sample(),**change),[])
    def test_requested_flags_must_match_actual_mode(self):
        with self.assertRaises(ValueError):validate_decode_metadata(self.sample(),['--cuda-graph','off'])
        with self.assertRaises(ValueError):validate_decode_metadata(self.sample(),['--fusion','off'])
    def test_eager_ablation_requires_no_graph_and_no_packed_weights(self):
        data=dict(self.sample(),decode_mode='cuda_eager',fusion_enabled=False,decode_graph_launches=0,
                  decode_graph_captures_total=0,decode_graph_nodes=0,decode_graph_kernel_nodes=0,
                  decode_host_submissions=3000,decode_packed_weight_bytes=0,graph_capture_total_ms=0,graph_warmup_total_ms=0)
        validate_decode_metadata(data,['--cuda-graph','off','--fusion','off'])
        with self.assertRaises(ValueError):validate_decode_metadata(dict(data,decode_graph_launches=1),['--cuda-graph','off','--fusion','off'])
    def test_last_flag_value_matches_cli_behavior(self):
        data=dict(self.sample(),fusion_enabled=False,decode_packed_weight_bytes=0)
        validate_decode_metadata(data,['--cuda-graph','off','--cuda-graph','on','--fusion','off'])
    def test_legacy_and_zero_decode_formats(self):
        validate_decode_metadata({'generated_tokens':[1]*32},[])
        data=dict(self.sample(),generated_tokens=[1],decode_mode='none',decode_instrumented=False,fusion_enabled=False,decode_graph_launches=0,decode_host_submissions=0)
        validate_decode_metadata(data,[])

if __name__=='__main__':unittest.main()
