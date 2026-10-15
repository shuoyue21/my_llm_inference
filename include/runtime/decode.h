#pragma once
#include "runtime/prefill.h"
#include <cstdint>

namespace runtime {
struct DecodeStats {
  double setup_ms = 0, setup_total_ms = 0;
  double capture_ms = 0, capture_total_ms = 0;
  double warmup_ms = 0, warmup_total_ms = 0;
  uint64_t host_submissions = 0, graph_launches = 0;
  uint64_t preparation_submissions = 0, capture_api_calls = 0;
  uint64_t graph_nodes = 0, graph_kernel_nodes = 0, captures_total = 0;
  uint64_t packed_weight_bytes = 0, workspace_bytes = 0;
};

// A stable device cursor and workspaces permit capture once and replay at any
// valid decode position. Fusion and graph replay are independent choices.
class CudaDecode {
 public:
  CudaDecode(PrefillWeights weights, std::shared_ptr<kernel::CudaConfig> config,
             tensor::Tensor logits, KVCacheView cache,
             tensor::Tensor sin_cache, tensor::Tensor cos_cache);
  ~CudaDecode();
  CudaDecode(const CudaDecode&) = delete;
  CudaDecode& operator=(const CudaDecode&) = delete;
  void begin(int token, int position, bool graph, bool fusion);
  int step();
  int next_position() const;
  DecodeStats stats() const;
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace runtime
