#include "runtime/serving_benchmark.h"
#include "runtime/cuda_batch.h"
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <thread>
#include <stdexcept>
#include <fstream>
namespace runtime {
namespace {
using Clock = std::chrono::steady_clock;
double since(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
template<class T> void array(const std::vector<T>& values) {
  std::cout << '[';
  for (size_t i=0; i<values.size(); ++i) { if(i) std::cout << ','; std::cout << values[i]; }
  std::cout << ']';
}
double percentile(std::vector<double> values, double q) {
  if (values.empty()) return 0;
  std::sort(values.begin(), values.end()); double p = (values.size()-1)*q;
  auto i = size_t(p); return values[i] + (values[std::min(i+1, values.size()-1)]-values[i])*(p-i);
}
}
void benchmark_serving(const model::Model& model, const std::string& prompt,
    const GenerationOptions& generation, const ServingBenchmarkOptions& options,
    int warmups, int repeats, double load_ms, size_t load_memory) {
  if ((options.mode != "continuous" && options.mode != "serial") || options.concurrency <= 0 ||
      options.requests <= 0 || options.requests > 65536 || options.token_budget <= 0 ||
      options.token_budget > 1024 || options.arrival_interval_ms < 0)
    throw std::invalid_argument("Invalid serving benchmark options");
  auto ids = generation.prompt_token_ids.empty() ? model.encode(prompt) : generation.prompt_token_ids;
  size_t backend_free_before=0, backend_free_after=0, backend_total=0;
  cudaMemGetInfo(&backend_free_before,&backend_total);
  auto backend_start=Clock::now();
  std::unique_ptr<CudaBatchBackend> backend;
  if (options.mode == "continuous") backend = model.create_batch_backend(options.token_budget, options.precision);
  cudaDeviceSynchronize();
  cudaMemGetInfo(&backend_free_after,&backend_total);
  const double backend_setup_ms=since(backend_start);
  load_memory += backend_free_before > backend_free_after ? backend_free_before-backend_free_after : 0;
  if (backend) backend->configure_attention(options.attention);
  std::ofstream dump;
  if (!options.logits_path.empty()) {
    if (!backend || warmups || repeats != 1 || options.requests != 1) throw std::invalid_argument("Serving logits dump requires one request/repeat and no warmups");
    dump.open(options.logits_path, std::ios::binary);
    if (!dump) throw std::runtime_error("Cannot open serving logits dump");
    backend->set_logits_callback([&](RequestId, int, const std::vector<float>& values) {
      dump.write(reinterpret_cast<const char*>(values.data()), values.size()*sizeof(float));
      if (!dump) throw std::runtime_error("Serving logits dump failed");
    });
  }
  if (backend) backend->configure_prefix_cache(options.prefix_cache_bytes);
  for (int iteration = -warmups; iteration < repeats; ++iteration) {
    auto prefix_before = backend ? backend->prefix_stats() : PrefixCacheStats{};
    auto start = Clock::now();
    std::vector<ServingResult> completed;
    if (backend) {
      Scheduler scheduler(*backend, {size_t(options.concurrency), size_t(options.token_budget), size_t(options.requests)});
      int submitted = 0;
      while (submitted < options.requests || !scheduler.empty()) {
        while (submitted < options.requests && since(start) >= submitted * options.arrival_interval_ms) {
          scheduler.submit({uint64_t(submitted), ids, generation.max_new_tokens, generation.ignore_eos,
              std::max(0.0, since(start) - submitted * options.arrival_interval_ms)});
          ++submitted;
        }
        if (!scheduler.step() && submitted < options.requests)
          std::this_thread::sleep_for(std::chrono::microseconds(50));
        auto ready = scheduler.take_completed();
        for (auto& result : ready) completed.push_back(std::move(result));
      }
    } else {
      for (int i = 0; i < options.requests; ++i) {
        auto arrival = i * options.arrival_interval_ms;
        if (since(start) < arrival)
          std::this_thread::sleep_until(start + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double,std::milli>(arrival)));
        double queue = since(start) - arrival;
        GenerationResult single;
        auto status = generate(model, prompt, generation, single);
        if (!status) throw std::runtime_error(status.get_err_msg());
        ServingResult result; result.id = i; result.tokens = std::move(single.generated_tokens);
        result.queue_ms = queue; result.ttft_ms = queue + single.ttft_ms;
        result.total_ms = since(start) - arrival; result.inter_token_ms = std::move(single.decode_token_ms);
        completed.push_back(std::move(result));
      }
    }
    const double total = since(start);
    if (iteration < 0) continue;
    std::sort(completed.begin(), completed.end(), [](const auto& a, const auto& b) { return a.id < b.id; });
    std::vector<double> ttft, latency, tpot, queue;
    size_t generated = 0;
    for (const auto& result : completed) {
      ttft.push_back(result.ttft_ms); latency.push_back(result.total_ms); queue.push_back(result.queue_ms);
      generated += result.tokens.size();
      tpot.insert(tpot.end(), result.inter_token_ms.begin(), result.inter_token_ms.end());
    }
    auto prefix_after = backend ? backend->prefix_stats() : PrefixCacheStats{};
    std::cout << "{\"attention\":\"" << options.attention << "\",\"attention_workspace_bytes\":" << (backend ? backend->attention_workspace_bytes() : 0) << ",\"precision\":\"" << options.precision << "\",\"linear_weight_bytes\":" << (backend ? backend->linear_weight_bytes() : 0) << ",\"prefix_cache_bytes\":" << prefix_after.resident_bytes
        << ",\"prefix_lookups\":" << prefix_after.lookups - prefix_before.lookups
        << ",\"prefix_hits\":" << prefix_after.hits - prefix_before.hits
        << ",\"prefix_reused_tokens\":" << prefix_after.reused_tokens - prefix_before.reused_tokens
        << ",\"prefix_evictions\":" << prefix_after.evictions - prefix_before.evictions << ',';
    std::cout << std::setprecision(12) << "\"serving_mode\":\"" << options.mode << "\",\"concurrency\":" << options.concurrency
        << ",\"request_count\":" << options.requests << ",\"token_budget\":" << options.token_budget
        << ",\"arrival_interval_ms\":" << options.arrival_interval_ms << ",\"backend_setup_ms\":" << backend_setup_ms << ",\"load_ms\":" << load_ms + backend_setup_ms
        << ",\"gpu_load_memory_bytes\":" << load_memory << ",\"total_ms\":" << total
        << ",\"output_tokens_per_second\":" << generated * 1000.0 / total
        << ",\"requests_per_second\":" << completed.size() * 1000.0 / total
        << ",\"ttft_ms\":" << percentile(ttft,.5) << ",\"ttft_p95_ms\":" << percentile(ttft,.95)
        << ",\"ttft_p99_ms\":" << percentile(ttft,.99) << ",\"latency_p50_ms\":" << percentile(latency,.5)
        << ",\"latency_p95_ms\":" << percentile(latency,.95) << ",\"latency_p99_ms\":" << percentile(latency,.99)
        << ",\"tpot_p50_ms\":" << percentile(tpot,.5) << ",\"tpot_p95_ms\":" << percentile(tpot,.95)
        << ",\"tpot_p99_ms\":" << percentile(tpot,.99) << ",\"queue_p95_ms\":" << percentile(queue,.95)
        << ",\"prompt_tokens\":";
    array(ids); std::cout << ",\"requests\":[";
    for (size_t i=0; i<completed.size(); ++i) {
      if (i) std::cout << ',';
      const auto& r = completed[i];
      std::cout << "{\"id\":" << r.id << ",\"generated_tokens\":"; array(r.tokens);
      std::cout << ",\"ttft_ms\":" << r.ttft_ms << ",\"total_ms\":" << r.total_ms << ",\"queue_ms\":" << r.queue_ms << ",\"decode_token_ms\":";
      array(r.inter_token_ms); std::cout << '}';
    }
    std::cout << "]}" << std::endl;
  }
}
}
