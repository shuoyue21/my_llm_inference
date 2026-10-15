"""Gate and contamination detection tests; no GPU workload is launched."""
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from empty_gpu import GpuMonitor,NoIdleGpu
from benchmark_suite import summarize
from unittest.mock import patch
class GateTests(unittest.TestCase):
 def setUp(self):
  self.monitor=object.__new__(GpuMonitor);self.monitor.max_idle_bytes=128*1024**2;self.monitor.max_idle_util=1
 def sample(self,processes=None,util=0,memory=16*1024**2):return {'processes':processes or [],'memory_used_bytes':memory,'gpu_util_percent':util}
 def test_idle_compute_client_still_rejected(self):
  self.assertFalse(self.monitor.idle(self.sample([{'pid':1,'kinds':['compute'],'used_memory_bytes':1,'name':'python'}])))
 def test_empty_low_utilization_passes(self):self.assertTrue(self.monitor.idle(self.sample()))
 def test_busy_without_visible_process_still_rejected(self):self.assertFalse(self.monitor.idle(self.sample(util=50)))
 def test_hidden_memory_still_rejected(self):self.assertFalse(self.monitor.idle(self.sample(memory=1024**3)))
 def test_only_confirmed_small_graphics_daemon_exempt(self):
  process={'pid':4037,'kinds':['graphics'],'used_memory_bytes':5*1024**2,'name':'/usr/lib/xorg/Xorg'}
  self.assertTrue(self.monitor.idle(self.sample([process])))
  for change in [{'name':None},{'used_memory_bytes':100*1024**2},{'kinds':['graphics','compute']},{'name':'other'}]:self.assertFalse(self.monitor.idle(self.sample([dict(process,**change)])))
 def test_foreign_process_invalidates_whole_measurement(self):
  self.monitor.handle=None
  self.monitor.snapshot=lambda _:dict(self.sample([{'pid':10,'kinds':['compute']},{'pid':20,'kinds':['compute']}]),monotonic_ns=1)
  watch=self.monitor.watch(10);watch.poll();self.assertFalse(watch.valid);self.assertEqual(watch.foreign[0]['processes'][0]['pid'],20)
 def test_nvml_error_is_not_a_clean_gpu(self):
  self.monitor.handle=None
  def failed(_):raise RuntimeError('NVML unavailable')
  self.monitor.snapshot=failed;watch=self.monitor.watch(10);watch.poll();self.assertFalse(watch.valid);self.assertTrue(watch.errors)
 def test_latency_statistics_use_actual_decode_tokens(self):
  sample=dict(load_ms=1,tokenization_ms=2,ttft_ms=10,prefill_ms=8,decode_ms=6,total_ms=16,decode_tokens_per_second=500,gpu_load_memory_bytes=100,decode_token_ms=[1,2,3])
  result=summarize([sample,sample],4);self.assertEqual(result['tpot_ms']['median'],2);self.assertEqual(result['prefill_tokens_per_second']['median'],500)
class PairingTests(unittest.TestCase):
 def record(self,ttft,throughput):
  return {'gpu_uuid':'GPU-a','precision':'fp32','model':'example','prompt_tokens':8,'generated_tokens':32,'cpu_threads':1,'warmups':2,'repeats':5,'samples':[{'prompt_tokens':[1]*8}],
   'metrics':{'ttft_ms':{'median':ttft},'prefill_ms':{'median':ttft},'total_ms':{'median':ttft},'decode_tokens_per_second':{'median':throughput},'tpot_ms':{'median':ttft,'p95':ttft,'p99':ttft}},'monitor':{'peak_sampled_device_memory_bytes':1000},'before':{'memory_used_bytes':100}}
 def test_known_latency_reduction_and_throughput_increase(self):
  from compare_stages import compare
  result=compare(self.record(10,100),self.record(5,200));self.assertEqual(result['ttft_ms']['change_percent'],-50);self.assertEqual(result['decode_tokens_per_second']['speedup'],2)
 def test_different_card_is_not_a_paired_comparison(self):
  from compare_stages import compare
  base=self.record(10,100);current=self.record(5,200);current['gpu_uuid']='GPU-b'
  with self.assertRaises(ValueError):compare(base,current)
 def test_different_actual_token_sequence_is_rejected(self):
  from compare_stages import compare
  base=self.record(10,100);current=self.record(5,200);current['samples'][0]['prompt_tokens'][0]=2
  with self.assertRaises(ValueError):compare(base,current)
if __name__=='__main__':unittest.main()
