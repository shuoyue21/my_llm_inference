#pragma once
#include "runtime/generation.h"
namespace runtime {
struct ServingBenchmarkOptions {
  std::string mode = "continuous";
  std::string precision = "fp32";
  std::string attention = "reference";
  std::string logits_path;
  int concurrency = 4;
  int requests = 32;
  int token_budget = 128;
  uint64_t prefix_cache_bytes = 0;
  double arrival_interval_ms = 0;
};
void benchmark_serving(const model::Model& model, const std::string& prompt,
    const GenerationOptions& generation, const ServingBenchmarkOptions& options,
    int warmups, int repeats, double load_ms, size_t load_memory);
}
