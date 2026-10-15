#pragma once
#include "runtime/kv_cache.h"
namespace kernel {
void paged_store_batch(const float* keys,const float* values,runtime::KVCacheView cache,int layer,int rows,cudaStream_t stream);
void paged_prefill_attention(const float* query,runtime::KVCacheView cache,int layer,float* scores,float* output,int rows,int heads,int kv_heads,int head_size,cudaStream_t stream);
}  // namespace kernel
