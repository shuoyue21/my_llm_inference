#pragma once
#include "model/model.h"
#include <functional>
namespace runtime {
struct GenerationOptions {
  int max_new_tokens = 64;
  bool ignore_eos = false;
  std::string prefill_mode = "auto";
  std::string decode_mode = "auto";
  bool cuda_graph = true;
  bool fusion = true;
  std::vector<int32_t> prompt_token_ids;
  // Optional diagnostic callback; synchronizing/copying logits affects timings.
  std::function<void(int, const tensor::Tensor&)> on_logits;
};
struct GenerationResult {
  std::vector<int32_t> prompt_tokens;
  std::vector<int32_t> generated_tokens;
  std::string text;
  std::string stop_reason;
  std::string prefill_mode;
  std::string decode_mode = "none";
  bool fusion_enabled = false;
  bool decode_instrumented = false;
  DecodeStats decode_stats;
  KVCacheStats kv_stats;
  double kv_release_ms = 0;
  double tokenization_ms = 0;
  double prefill_ms = 0;
  std::vector<double> decode_token_ms;
  double ttft_ms = 0;
  double decode_ms = 0;
  double total_ms = 0;
  double decode_tokens_per_second = 0;
};
base::Status generate(const model::Model& model, const std::string& prompt,
                      const GenerationOptions& options, GenerationResult& result);
}
