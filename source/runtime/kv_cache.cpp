#include "runtime/kv_cache.h"
#include <algorithm>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
namespace runtime {
namespace {
void check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(std::string("Paged KV: ") + cudaGetErrorString(status));
}
struct Page {
  float* data = nullptr;
  bool mapped = false;
  ~Page() { if (data) cudaFree(data); }
};
}
struct PagedKVCache::Impl {
  int layers, context, kv_dim, page_tokens, logical_pages;
  size_t page_bytes, token_bytes;
  std::shared_ptr<kernel::CudaConfig> config;
  float** device_table = nullptr;
  float** host_table = nullptr;
  std::vector<std::unique_ptr<Page>> physical;
  std::vector<Page*> mapping, free_list;
  bool active = false;
  uint64_t mapped_count = 0, allocations = 0, frees = 0, trimmed_bytes = 0;
  KVCacheStats current;
  Impl(int l, int c, int d, int p, std::shared_ptr<kernel::CudaConfig> cuda)
      : layers(l), context(c), kv_dim(d), page_tokens(p), config(std::move(cuda)) {
    if (!config || l <= 0 || c <= 0 || d <= 0 || p <= 0 || p > c)
      throw std::invalid_argument("Invalid CUDA paged KV dimensions/page size");
    logical_pages = (context + page_tokens - 1) / page_tokens;
    token_bytes = size_t(2) * layers * kv_dim * sizeof(float);
    page_bytes = token_bytes * page_tokens;
    mapping.resize(logical_pages, nullptr);
    try {
      check(cudaMalloc(reinterpret_cast<void**>(&device_table), logical_pages * sizeof(float*)));
      check(cudaMallocHost(reinterpret_cast<void**>(&host_table), logical_pages * sizeof(float*)));
      std::fill_n(host_table, logical_pages, nullptr);
      check(cudaMemsetAsync(device_table, 0, logical_pages * sizeof(float*), config->stream));
    } catch (...) {
      if (device_table) cudaFree(device_table);
      if (host_table) cudaFreeHost(host_table);
      throw;
    }
  }
  ~Impl() {
    if (config) cudaStreamSynchronize(config->stream);
    physical.clear();
    if (device_table) cudaFree(device_table);
    if (host_table) cudaFreeHost(host_table);
  }
  void begin() {
    if (active) release();
    current = {};
    current.paged = true;
    current.page_tokens = page_tokens;
    current.peak_reserved_bytes = physical.size() * page_bytes;
    active = true;
  }
  void ensure(int count) {
    if (count < 0 || count > context) throw std::out_of_range("Paged KV token count exceeds context");
    if (!active) begin();
    const int needed = (count + page_tokens - 1) / page_tokens;
    const int first_new = static_cast<int>(mapped_count);
    for (int i = first_new; i < needed; ++i) {
      if (mapping[i]) continue;
      Page* page = nullptr;
      if (!free_list.empty()) {
        page = free_list.back(); free_list.pop_back(); ++current.reuses;
      } else {
        auto allocated = std::make_unique<Page>();
        check(cudaMalloc(reinterpret_cast<void**>(&allocated->data), page_bytes));
        page = allocated.get(); physical.push_back(std::move(allocated));
        ++current.allocations; ++allocations;
      }
      page->mapped = true; mapping[i] = page; host_table[i] = page->data; ++mapped_count;
      current.peak_mapped_bytes = std::max(current.peak_mapped_bytes, mapped_count * page_bytes);
      current.peak_reserved_bytes = std::max(current.peak_reserved_bytes, physical.size() * page_bytes);
    }
    if (needed > first_new) {
      check(cudaMemcpyAsync(device_table + first_new, host_table + first_new,
                            (needed - first_new) * sizeof(float*), cudaMemcpyHostToDevice, config->stream));
      ++current.table_updates;
    }
    current.logical_tokens = std::max(current.logical_tokens, count);
  }
  std::pair<float*, float*> slot(int layer, int position) const {
    if (layer < 0 || layer >= layers || position < 0 || position >= context)
      throw std::out_of_range("Invalid paged KV layer/position");
    Page* page = mapping[position / page_tokens];
    if (!page) throw std::logic_error("Paged KV position has no physical mapping");
    size_t offset = size_t(layer * page_tokens + position % page_tokens) * kv_dim;
    return {page->data + offset, page->data + size_t(layers) * page_tokens * kv_dim + offset};
  }
  void release() {
    if (!active) return;
    check(cudaStreamSynchronize(config->stream));
    for (auto& page : mapping) if (page) {
      page->mapped = false; free_list.push_back(page); page = nullptr; ++current.releases;
    }
    mapped_count = 0;
    std::fill_n(host_table, logical_pages, nullptr);
    check(cudaMemsetAsync(device_table, 0, logical_pages * sizeof(float*), config->stream));
    ++current.table_clears;
    check(cudaStreamSynchronize(config->stream));
    active = false;
  }
  void trim() {
    check(cudaStreamSynchronize(config->stream));
    for (auto& page : physical) if (!page->mapped && page->data) {
      check(cudaFree(page->data)); page->data = nullptr;
      ++current.trimmed_pages; ++frees; trimmed_bytes += page_bytes;
    }
    free_list.clear();
    physical.erase(std::remove_if(physical.begin(), physical.end(), [](const auto& p) { return p->data == nullptr; }), physical.end());
  }
  KVCacheStats stats() const {
    auto result = current;
    result.paged = true; result.page_tokens = page_tokens;
    result.reserved_bytes = physical.size() * page_bytes;
    result.mapped_bytes = mapped_count * page_bytes;
    result.effective_bytes = size_t(current.logical_tokens) * token_bytes;
    result.table_bytes = logical_pages * sizeof(float*);
    result.mapped_pages = mapped_count; result.free_pages = free_list.size();
    result.lifetime_allocations = allocations; result.lifetime_frees = frees; result.lifetime_trimmed_bytes = trimmed_bytes;
    return result;
  }
};
PagedKVCache::PagedKVCache(int l,int c,int d,int p,std::shared_ptr<kernel::CudaConfig> config)
    : impl_(std::make_unique<Impl>(l,c,d,p,std::move(config))) {}
PagedKVCache::~PagedKVCache() = default;
void PagedKVCache::begin_request() { impl_->begin(); }
void PagedKVCache::ensure_tokens(int n) { impl_->ensure(n); }
std::pair<float*,float*> PagedKVCache::slot(int l,int p) const { return impl_->slot(l,p); }
void PagedKVCache::release_request() { impl_->release(); }
void PagedKVCache::trim() { impl_->trim(); }
void PagedKVCache::shuffle_free_pages(uint32_t seed) {
  if (impl_->active) throw std::logic_error("Release KV request before remapping free pages");
  std::mt19937 generator(seed); std::shuffle(impl_->free_list.begin(),impl_->free_list.end(),generator);
}
KVCacheView PagedKVCache::view() const {
  return {nullptr,nullptr,impl_->device_table,impl_->layers,impl_->context,impl_->kv_dim,impl_->page_tokens};
}
KVCacheStats PagedKVCache::stats() const { return impl_->stats(); }
}  // namespace runtime
