#include "runtime/generation.h"
#include "runtime/serving_benchmark.h"
#include "model/llama3.h"
#if defined(QWEN2_SUPPORT)
#include "model/qwen2.h"
#elif defined(QWEN3_SUPPORT)
#include "model/qwen3.h"
#endif
#include <cuda_runtime_api.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace {
std::string json_string(const std::string& s) {
  std::ostringstream out;
  out << '"';
  for (size_t i=0;i<s.size();) {
    unsigned char c=s[i];
    if (c < 0x80) {
      if (c == '"' || c == '\\') out << '\\' << c;
      else if (c < 0x20) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c) << std::dec;
      else out << c;
      ++i;
      continue;
    }
    size_t length=c>=0xC2 && c<=0xDF ? 2 : c>=0xE0 && c<=0xEF ? 3 : c>=0xF0 && c<=0xF4 ? 4 : 0;
    bool valid=length && i+length<=s.size();
    for(size_t j=1;valid && j<length;++j)valid=(static_cast<unsigned char>(s[i+j])&0xC0)==0x80;
    if(valid && length>=3) {
      unsigned char second=s[i+1];
      valid=!(c==0xE0 && second<0xA0) && !(c==0xED && second>=0xA0) && !(c==0xF0 && second<0x90) && !(c==0xF4 && second>=0x90);
    }
    if(valid) {out.write(s.data()+i,length);i+=length;}
    else {out << "\\ufffd"; ++i;}
  }
  out << '"';
  return out.str();
}

void ids(const std::vector<int32_t>& values) {
  std::cout << '[';
  for (size_t i = 0; i < values.size(); ++i) { if (i) std::cout << ','; std::cout << values[i]; }
  std::cout << ']';
}
void usage() {
  std::cout << "Usage: llama_infer CHECKPOINT TOKENIZER [--device cpu|cuda] [--prompt TEXT]\n"
               "       [--max-new-tokens N] [--json] [--quant] [--dump-logits FILE]\n"
               "       [--prompt-ids FILE] [--ignore-eos] [--warmup N] [--repeat N] [--prefill auto|batched|reference]\n"
               "       [--decode auto|optimized|reference] [--cuda-graph on|off] [--fusion on|off] [--kv-cache auto|paged|contiguous] [--kv-page-tokens N]\n"
               "       [--serving serial|continuous] [--concurrency N] [--requests N] [--token-budget N] [--arrival-ms N]\n"
               "       [--prefix-cache-mib N] [--precision fp32|fp16|bf16|w8a16|w4a16]\n"
               "       [--attention reference|online]\n"
               "Greedy decoding. JSON reports generated text, token IDs and separate TTFT/decode metrics.\n";
}
}
int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_logtostderr = true;
  if (argc == 2 && std::string(argv[1]) == "--help") { usage(); return 0; }
  if (argc < 3) { usage(); return 2; }
  try {
    std::string prompt = "Once upon a time", device = "cpu", dump_path;
    bool json = false, quant = false;
    int repeats = 1, warmups = 0;
    std::string prompt_ids_path;
    std::string kv_cache_mode = "auto";
    int kv_page_tokens = 16;
    runtime::GenerationOptions options;
    runtime::ServingBenchmarkOptions serving;
    bool serving_enabled = false, serving_host_weights = false, native_switch_selected = false;
    for (int i = 3; i < argc; ++i) {
      std::string flag = argv[i];
      if (flag == "--prefill" || flag == "--decode" || flag == "--cuda-graph" || flag == "--fusion") native_switch_selected = true;
      if (flag == "--json") { json = true; continue; }
      if (flag == "--quant") { quant = true; continue; }
      if (flag == "--ignore-eos") { options.ignore_eos = true; continue; }
      if (flag == "--attention") {
        if (++i == argc) throw std::invalid_argument("Missing attention mode");
        serving.attention = argv[i]; continue;
      }
      if (flag == "--precision") {
        if (++i == argc) throw std::invalid_argument("Missing precision");
        serving.precision = argv[i]; serving_host_weights = true; continue;
      }
      if (flag == "--serving" || flag == "--concurrency" || flag == "--requests" || flag == "--token-budget" || flag == "--prefix-cache-mib" || flag == "--arrival-ms") {
        if (++i == argc) throw std::invalid_argument("Missing serving option value");
        std::string value = argv[i]; size_t end = 0;
        if (flag == "--serving") { serving.mode = value; serving_enabled = true; }
        else if (flag == "--arrival-ms") {
          serving.arrival_interval_ms = std::stod(value, &end);
          if (end != value.size() || !std::isfinite(serving.arrival_interval_ms)) throw std::invalid_argument("Invalid arrival interval");
        } else {
          int number = std::stoi(value, &end);
          if (end != value.size()) throw std::invalid_argument("Invalid serving integer");
          if (flag == "--concurrency") serving.concurrency = number;
          if (flag == "--requests") serving.requests = number;
          if (flag == "--token-budget") serving.token_budget = number;
          if (flag == "--prefix-cache-mib") { if (number < 0) throw std::invalid_argument("Negative prefix capacity"); serving.prefix_cache_bytes = uint64_t(number) * 1024 * 1024; }
        }
        continue;
      }
      if (flag != "--prompt" && flag != "--device" && flag != "--max-new-tokens" && flag != "--dump-logits" && flag != "--prompt-ids" && flag != "--repeat" && flag != "--warmup" && flag != "--prefill" && flag != "--decode" && flag != "--cuda-graph" && flag != "--fusion" && flag != "--kv-cache" && flag != "--kv-page-tokens")
        throw std::invalid_argument("Unknown option: " + flag);
      if (++i == argc) throw std::invalid_argument("Missing value for " + flag);
      std::string value = argv[i];
      if (flag == "--prompt") prompt = value;
      else if (flag == "--device") device = value;
      else if (flag == "--prefill") options.prefill_mode = value;
      else if (flag == "--decode") options.decode_mode = value;
      else if (flag == "--kv-cache") kv_cache_mode = value;
      else if (flag == "--kv-page-tokens") {
        size_t end = 0; kv_page_tokens = std::stoi(value, &end);
        if (end != value.size() || kv_page_tokens <= 0) throw std::invalid_argument("Invalid KV page size");
      }
      else if (flag == "--cuda-graph" || flag == "--fusion") {
        if (value != "on" && value != "off") throw std::invalid_argument(flag + " must be on or off");
        if (flag == "--cuda-graph") options.cuda_graph = value == "on";
        else options.fusion = value == "on";
      }
      else if (flag == "--dump-logits") dump_path = value;
      else if (flag == "--prompt-ids") prompt_ids_path = value;
      else if (flag == "--repeat" || flag == "--warmup") {
        size_t end = 0; int number = std::stoi(value,&end);
        if (end!=value.size() || number<0 || (flag=="--repeat" && number==0)) throw std::invalid_argument("Invalid iteration count");
        if (flag=="--repeat") repeats=number; else warmups=number;
      }
      else { size_t end = 0; options.max_new_tokens = std::stoi(value, &end);
             if (end != value.size() || options.max_new_tokens < 0) throw std::invalid_argument("Invalid max-new-tokens"); }
    }
    if (device != "cpu" && device != "cuda") throw std::invalid_argument("Device must be cpu or cuda");
    if (device == "cuda") {
      int count = 0;
      auto error = cudaGetDeviceCount(&count);
      if (error != cudaSuccess || count == 0)
        throw std::runtime_error(std::string("CUDA unavailable: ") + cudaGetErrorString(error));
    }
    if (!std::ifstream(argv[1]).good() || !std::ifstream(argv[2]).good())
      throw std::invalid_argument("Checkpoint or tokenizer file does not exist");
    if (!prompt_ids_path.empty()) {
      std::ifstream file(prompt_ids_path);
      if (!file) throw std::invalid_argument("Cannot open prompt token file");
      int32_t token;
      while (file >> token) options.prompt_token_ids.push_back(token);
      if (!file.eof() || options.prompt_token_ids.empty()) throw std::invalid_argument("Invalid prompt token file");
    }
    size_t gpu_free_before=0,gpu_total=0,gpu_free_after=0;
    if(device=="cuda") cudaMemGetInfo(&gpu_free_before,&gpu_total);
    auto load_start = std::chrono::steady_clock::now();
    std::unique_ptr<model::Model> model;
#if defined(QWEN3_SUPPORT)
    model = std::make_unique<model::Qwen3Model>(base::TokenizerType::kEncodeBpe, argv[2], argv[1], quant);
#elif defined(QWEN2_SUPPORT)
    model = std::make_unique<model::Qwen2Model>(base::TokenizerType::kEncodeBpe, argv[2], argv[1], quant);
#elif defined(LLAMA3_SUPPORT)
    model = std::make_unique<model::LLama2Model>(base::TokenizerType::kEncodeBpe, argv[2], argv[1], quant);
#else
    model = std::make_unique<model::LLama2Model>(base::TokenizerType::kEncodeSpe, argv[2], argv[1], quant);
#endif
    if (serving_enabled && serving.mode == "continuous") {
      if (native_switch_selected) throw std::invalid_argument("prefill/decode/Graph/fusion switches apply to native or serial inference");
      if (kv_cache_mode == "contiguous") throw std::invalid_argument("Continuous serving requires paged KV");
    }
    auto status = model->configure_kv_cache(serving_host_weights ? "auto" : kv_cache_mode, kv_page_tokens);
    if (!status) throw std::runtime_error(status.get_err_msg());
    if (serving_host_weights && (!serving_enabled || serving.mode != "continuous")) throw std::invalid_argument("Precision selection requires continuous serving");
    status = model->init(device == "cpu" || serving_host_weights ? base::DeviceType::kDeviceCPU : base::DeviceType::kDeviceCUDA);
    if (!status) throw std::runtime_error(status.get_err_msg());
    if (device == "cuda") {
      auto error = cudaDeviceSynchronize();
      if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
    }
    double load_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - load_start).count();
    if (serving_enabled) {
      if (device != "cuda") throw std::invalid_argument("Serving benchmark requires CUDA");
      cudaMemGetInfo(&gpu_free_after, &gpu_total);
      serving.logits_path = dump_path;
      runtime::benchmark_serving(*model, prompt, options, serving, warmups, repeats, load_ms,
          gpu_free_before > gpu_free_after ? gpu_free_before - gpu_free_after : 0);
      return 0;
    }
    std::ofstream dump;
    if (!dump_path.empty()) {
      dump.open(dump_path, std::ios::binary);
      if (!dump) throw std::runtime_error("Cannot open logits output file");
      options.on_logits = [&](int, const tensor::Tensor& logits) {
        std::vector<float> values(logits.size());
        if (device == "cuda") {
          auto error = cudaDeviceSynchronize();
          if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
          error = cudaMemcpy(values.data(), logits.ptr<float>(), logits.byte_size(), cudaMemcpyDeviceToHost);
          if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
        } else std::copy_n(logits.ptr<float>(), logits.size(), values.data());
        dump.write(reinterpret_cast<const char*>(values.data()), logits.byte_size());
        if (!dump) throw std::runtime_error("Failed to write logits");
      };
    }
    if(device=="cuda") cudaMemGetInfo(&gpu_free_after,&gpu_total);
    if(!dump_path.empty() && (repeats!=1 || warmups!=0)) throw std::invalid_argument("Logit dump requires one iteration and no warmup");
    for (int iteration=-warmups; iteration<repeats; ++iteration) {
    runtime::GenerationResult result;
    status = runtime::generate(*model, prompt, options, result);
    if (!status) throw std::runtime_error(status.get_err_msg());
    if (iteration<0) continue;
    if (json) {
      std::cout << std::setprecision(9) << "{\"device\":" << json_string(device)
                << ",\"text\":" << json_string(result.text) << ",\"prompt_tokens\":";
      ids(result.prompt_tokens);
      std::cout << ",\"generated_tokens\":"; ids(result.generated_tokens);
      std::cout << ",\"stop_reason\":" << json_string(result.stop_reason)
                << ",\"prefill_mode\":" << json_string(result.prefill_mode)
                << ",\"decode_mode\":" << json_string(result.decode_mode)
                << ",\"fusion_enabled\":" << (result.fusion_enabled ? "true" : "false")
                << ",\"decode_instrumented\":" << (result.decode_instrumented ? "true" : "false")
                << ",\"decode_setup_ms\":" << result.decode_stats.setup_ms
                << ",\"decode_setup_total_ms\":" << result.decode_stats.setup_total_ms
                << ",\"graph_capture_ms\":" << result.decode_stats.capture_ms
                << ",\"graph_capture_total_ms\":" << result.decode_stats.capture_total_ms
                << ",\"graph_warmup_ms\":" << result.decode_stats.warmup_ms
                << ",\"graph_warmup_total_ms\":" << result.decode_stats.warmup_total_ms
                << ",\"decode_host_submissions\":" << result.decode_stats.host_submissions
                << ",\"decode_preparation_submissions\":" << result.decode_stats.preparation_submissions
                << ",\"graph_capture_api_calls\":" << result.decode_stats.capture_api_calls
                << ",\"decode_graph_launches\":" << result.decode_stats.graph_launches
                << ",\"decode_graph_nodes\":" << result.decode_stats.graph_nodes
                << ",\"decode_graph_kernel_nodes\":" << result.decode_stats.graph_kernel_nodes
                << ",\"decode_graph_captures_total\":" << result.decode_stats.captures_total
                << ",\"decode_packed_weight_bytes\":" << result.decode_stats.packed_weight_bytes
                << ",\"decode_workspace_bytes\":" << result.decode_stats.workspace_bytes
                << ",\"kv_cache_mode\":" << json_string(result.kv_stats.paged ? "paged" : "contiguous")
                << ",\"kv_page_tokens\":" << result.kv_stats.page_tokens
                << ",\"kv_logical_tokens\":" << result.kv_stats.logical_tokens
                << ",\"kv_reserved_bytes\":" << result.kv_stats.reserved_bytes
                << ",\"kv_mapped_bytes\":" << result.kv_stats.mapped_bytes
                << ",\"kv_effective_bytes\":" << result.kv_stats.effective_bytes
                << ",\"kv_table_bytes\":" << result.kv_stats.table_bytes
                << ",\"kv_peak_mapped_bytes\":" << result.kv_stats.peak_mapped_bytes
                << ",\"kv_peak_reserved_bytes\":" << result.kv_stats.peak_reserved_bytes
                << ",\"kv_live_pages_end\":" << result.kv_stats.mapped_pages
                << ",\"kv_free_pool_pages\":" << result.kv_stats.free_pages
                << ",\"kv_page_allocations\":" << result.kv_stats.allocations
                << ",\"kv_page_reuses\":" << result.kv_stats.reuses
                << ",\"kv_page_releases\":" << result.kv_stats.releases
                << ",\"kv_table_updates\":" << result.kv_stats.table_updates
                << ",\"kv_table_clears\":" << result.kv_stats.table_clears
                << ",\"kv_lifetime_allocations\":" << result.kv_stats.lifetime_allocations
                << ",\"kv_lifetime_frees\":" << result.kv_stats.lifetime_frees
                << ",\"kv_lifetime_trimmed_bytes\":" << result.kv_stats.lifetime_trimmed_bytes
                << ",\"kv_release_ms\":" << result.kv_release_ms
                << ",\"context_length\":" << model->context_length()
                << ",\"iteration\":" << iteration
                << ",\"warmup_requests\":" << warmups
                << ",\"tokenization_ms\":" << result.tokenization_ms
                << ",\"prefill_ms\":" << result.prefill_ms
                << ",\"gpu_load_memory_bytes\":" << (gpu_free_before>=gpu_free_after ? gpu_free_before-gpu_free_after : 0)
                << ",\"load_ms\":" << load_ms << ",\"ttft_ms\":" << result.ttft_ms
                << ",\"decode_ms\":" << result.decode_ms << ",\"total_ms\":" << result.total_ms
                << ",\"decode_tokens_per_second\":" << result.decode_tokens_per_second << ",\"decode_token_ms\":[";
      for(size_t i=0;i<result.decode_token_ms.size();++i) {if(i)std::cout<<',';std::cout<<result.decode_token_ms[i];}
      std::cout << "]}\n";
    } else {
      std::cout << result.text << '\n';
      std::cerr << "prompt_tokens=" << result.prompt_tokens.size() << " generated_tokens=" << result.generated_tokens.size()
                << " stop=" << result.stop_reason << " load_ms=" << load_ms << " ttft_ms=" << result.ttft_ms
                << " decode_tokens/s=" << result.decode_tokens_per_second << '\n';
    }
    }
    return 0;
  } catch (const std::exception& error) { std::cerr << "Error: " << error.what() << '\n'; return 1; }
}
