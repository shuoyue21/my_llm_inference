#pragma once
#include "runtime/kv_cache.h"
namespace kernel {
__device__ __forceinline__ float* kv_key(runtime::KVCacheView cache,int layer,int token) {
  if (cache.pages) return cache.pages[token/cache.page_tokens] + size_t(layer*cache.page_tokens+token%cache.page_tokens)*cache.kv_dim;
  return cache.keys + (size_t(layer)*cache.context+token)*cache.kv_dim;
}
__device__ __forceinline__ float* kv_value(runtime::KVCacheView cache,int layer,int token) {
  if (cache.pages) return cache.pages[token/cache.page_tokens] + size_t((cache.layers+layer)*cache.page_tokens+token%cache.page_tokens)*cache.kv_dim;
  return cache.values + (size_t(layer)*cache.context+token)*cache.kv_dim;
}
}  // namespace kernel
