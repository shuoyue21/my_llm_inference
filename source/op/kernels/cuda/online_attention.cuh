#pragma once
#include "runtime/kv_cache.h"
namespace kernel {
void batch_online_attention(const float* query, const runtime::KVCacheView* caches,
    const int* positions, int layer, float* output, int rows,
    int heads, int kv_heads, int head_size, cudaStream_t stream);
}
