#pragma once
#include "runtime/scheduler.h"
#include "runtime/prefill.h"
#include <functional>
namespace runtime {
// Shared immutable model tensors and one GEMM workspace; request KV is private.
struct PrefixCacheStats {
  uint64_t lookups = 0, hits = 0, reused_tokens = 0, evictions = 0, resident_bytes = 0;
};
class CudaBatchBackend : public BatchBackend {
 public:
  CudaBatchBackend(PrefillWeights weights, std::shared_ptr<kernel::CudaConfig> config,
                   tensor::Tensor sine, tensor::Tensor cosine, size_t max_rows,
                   int page_tokens, std::function<bool(int)> is_eos);
  ~CudaBatchBackend();
  int context_length() const override;
  int vocab_size() const override;
  void admit(RequestId id) override;
  std::vector<TokenOutput> execute(const std::vector<ScheduledToken>& batch) override;
  void release(RequestId id) override;
  KVCacheStats cache_stats() const;
  void configure_prefix_cache(uint64_t max_bytes);
  int reuse_prefix(RequestId id, const std::vector<int32_t>& tokens) override;
  PrefixCacheStats prefix_stats() const;
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}
