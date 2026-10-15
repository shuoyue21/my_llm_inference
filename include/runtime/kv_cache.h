#pragma once
#include "base/cuda_config.h"
#include <cstdint>
#include <memory>
#include <utility>

namespace runtime {
// Device-readable view: a real dense cache OR independently allocated pages.
struct KVCacheView {
  float* keys = nullptr;
  float* values = nullptr;
  float* const* pages = nullptr;
  int layers = 0, context = 0, kv_dim = 0, page_tokens = 0;
};
struct KVCacheStats {
  bool paged = false;
  int page_tokens = 0, logical_tokens = 0;
  uint64_t reserved_bytes = 0, mapped_bytes = 0, effective_bytes = 0, table_bytes = 0;
  uint64_t peak_mapped_bytes = 0, peak_reserved_bytes = 0;
  uint64_t mapped_pages = 0, free_pages = 0;
  uint64_t allocations = 0, reuses = 0, releases = 0, trimmed_pages = 0;
  uint64_t table_updates = 0, table_clears = 0;
  uint64_t lifetime_allocations = 0, lifetime_frees = 0, lifetime_trimmed_bytes = 0;
};
class PagedKVCache {
 public:
  PagedKVCache(int layers, int context, int kv_dim, int page_tokens,
               std::shared_ptr<kernel::CudaConfig> config);
  ~PagedKVCache();
  PagedKVCache(const PagedKVCache&) = delete;
  PagedKVCache& operator=(const PagedKVCache&) = delete;
  void begin_request();
  void ensure_tokens(int count);
  std::pair<float*, float*> slot(int layer, int position) const;
  void release_request();
  void trim();
  void shuffle_free_pages(uint32_t seed);
  KVCacheView view() const;
  KVCacheStats stats() const;
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace runtime
