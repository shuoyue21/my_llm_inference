#pragma once
#include "runtime/generation.h"
namespace runtime {
struct ServingBenchmarkOptions {
  std::string mode = "continuous";
  int concurrency = 4;
  int requests = 32;
  int token_budget = 128;
  double arrival_interval_ms = 0;
};
void benchmark_serving(const model::Model& model, const std::string& prompt,
    const GenerationOptions& generation, const ServingBenchmarkOptions& options,
    int warmups, int repeats, double load_ms, size_t load_memory);
}
