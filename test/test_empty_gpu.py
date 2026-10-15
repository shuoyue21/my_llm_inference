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
class DesktopPolicyTests(unittest.TestCase):
 def setUp(self):
  from gpu_policy import normalize_policy
  self.monitor=object.__new__(GpuMonitor);self.monitor.max_idle_bytes=128*1024**2;self.monitor.max_idle_util=1
  self.monitor.policy=normalize_policy({'approved_graphics_process_names':['/Xwayland'],'idle_counter_mode':'observe_only_with_approved_graphics'})
  self.desktop={'pid':27,'kinds':['graphics'],'name':'/Xwayland','used_memory_bytes':None}
 def sample(self,processes):return {'processes':processes,'memory_used_bytes':4*1024**3,'gpu_util_percent':10}
 def test_explicitly_approved_desktop_counters_are_observational(self):
  self.assertTrue(self.monitor.idle(self.sample([self.desktop])))
 def test_desktop_name_never_exempts_compute_or_mixed_client(self):
  for kinds in [['compute'],['graphics','compute']]:
   self.assertFalse(self.monitor.idle(self.sample([dict(self.desktop,kinds=kinds)])))
 def test_other_compute_or_graphics_still_rejected(self):
  for kinds in [['compute'],['graphics']]:
   outsider={'pid':99,'kinds':kinds,'name':'other','used_memory_bytes':1}
   self.assertFalse(self.monitor.idle(self.sample([self.desktop,outsider])))
 def test_counter_exception_requires_exact_approved_desktop(self):
  self.assertFalse(self.monitor.idle(self.sample([])))
  self.assertFalse(self.monitor.idle(self.sample([dict(self.desktop,name='/other/Xwayland')])))
 def test_strict_policy_still_rejects_desktop(self):
  from gpu_policy import normalize_policy
  self.monitor.policy=normalize_policy()
  self.assertFalse(self.monitor.idle(self.sample([self.desktop])))
 def test_invalid_policy_cannot_disable_checks_accidentally(self):
  from gpu_policy import normalize_policy
  for policy in [{'idle_counter_mode':'observe_only_with_approved_graphics'},{'approved_graphics_process_names':'/Xwayland'},{'idle_counter_mode':'ignore_everything'},{'schema_version':True},{'unknown':True}]:
   with self.assertRaises(ValueError):normalize_policy(policy)
 def test_explicit_policy_from_environment(self):
  import json
  from gpu_policy import policy_from_environment
  with patch.dict('os.environ',{'LLM_GPU_POLICY_JSON':json.dumps(self.monitor.policy)}):
   self.assertEqual(policy_from_environment(),self.monitor.policy)
class PairingTests(unittest.TestCase):
 def record(self,ttft,throughput):
  return {'gpu_uuid':'GPU-a','precision':'fp32','model':'example','prompt_tokens':8,'generated_tokens':32,'cpu_threads':1,'warmups':2,'repeats':5,'samples':[{'prompt_tokens':[1]*8,'generated_tokens':[2]*32}],
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
 def test_different_greedy_outputs_are_not_a_correct_stage_comparison(self):
  from compare_stages import compare
  base=self.record(10,100);current=self.record(5,200);current['samples'][0]['generated_tokens'][0]=3
  with self.assertRaises(ValueError):compare(base,current)
 def test_mismatched_gpu_policy_is_not_a_paired_comparison(self):
  from compare_stages import compare
  base=self.record(10,100);current=self.record(5,200)
  current['gpu_policy']={'approved_graphics_process_names':['/Xwayland'],'idle_counter_mode':'observe_only_with_approved_graphics'}
  with self.assertRaises(ValueError):compare(base,current)
class RecoveryTests(unittest.TestCase):
 def record(self,accepted,valid):
  return {'source_digest':'source','gpu_uuid':'GPU-a','extra_args':[],'accepted':accepted,'monitor':{'valid':valid}}
 def test_contaminated_attempt_is_preserved_before_retry(self):
  import tempfile,json
  from measurement_records import prepare_existing_result
  with tempfile.TemporaryDirectory() as temp:
   path=Path(temp)/'case.json';path.write_text(json.dumps(self.record(False,False)))
   raw=Path(temp)/'case.stdout.jsonl';raw.write_text('partial output')
   self.assertFalse(prepare_existing_result(path,'source','GPU-a',[]))
   self.assertFalse(path.exists());self.assertFalse(raw.exists())
   archived=list((Path(temp)/'attempts'/'case').glob('*/case.json'));self.assertEqual(len(archived),1)
   self.assertEqual(archived[0].with_name('case.stdout.jsonl').read_text(),'partial output')
 def test_valid_measurement_is_reused_without_overwrite(self):
  import tempfile,json
  from measurement_records import prepare_existing_result
  with tempfile.TemporaryDirectory() as temp:
   path=Path(temp)/'case.json';path.write_text(json.dumps(self.record(True,True)))
   self.assertTrue(prepare_existing_result(path,'source','GPU-a',[]));self.assertTrue(path.exists())
 def test_different_provenance_never_archived_as_contamination(self):
  import tempfile,json
  from measurement_records import prepare_existing_result
  with tempfile.TemporaryDirectory() as temp:
   path=Path(temp)/'case.json';path.write_text(json.dumps(self.record(False,False)))
   with self.assertRaises(RuntimeError):prepare_existing_result(path,'different-source','GPU-a',[])
   self.assertTrue(path.exists())
 def test_model_failure_does_not_retry_as_resource_wait(self):
  import tempfile,json
  from measurement_records import prepare_existing_result
  with tempfile.TemporaryDirectory() as temp:
   path=Path(temp)/'case.json';path.write_text(json.dumps(self.record(False,True)))
   with self.assertRaises(RuntimeError):prepare_existing_result(path,'source','GPU-a',[])
if __name__=='__main__':unittest.main()
