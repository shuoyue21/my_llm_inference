#pragma once
#include <cuda_runtime_api.h>
#include "runtime/kv_cache.h"
namespace kernel {
void decode_embedding(const int* token,const float* table,float* output,int width,cudaStream_t stream);
void decode_residual_norm(float* residual,const float* update,const float* weight,float* output,int width,float epsilon,cudaStream_t stream);
void decode_rope(float* query,float* key,const int* position,const float* sine,const float* cosine,int query_width,int key_width,int head_size,bool split_half,cudaStream_t stream);
void decode_store_kv(const float* key,const float* value,float* keys,float* values,const int* position,int width,cudaStream_t stream);
void decode_prepare_qkv(float* qkv,const float* bias,const float* query_norm,const float* key_norm,
                        float* keys,float* values,const int* position,const float* sine,const float* cosine,
                        int heads,int kv_heads,int head_size,float epsilon,bool split_half,cudaStream_t stream);
void decode_packed_swiglu(float* packed,const float* bias,int width,cudaStream_t stream);
void decode_attention(const float* query,const float* keys,const float* values,float* scores,float* output,
                      const int* position,int context,int heads,int kv_heads,int head_size,cudaStream_t stream);
void decode_sample(const float* logits,int vocab,int* token,int* position,cudaStream_t stream);
void decode_store_cache(const float* key,const float* value,runtime::KVCacheView cache,int layer,const int* position,cudaStream_t stream);
void decode_prepare_cache_qkv(float* qkv,const float* bias,const float* query_norm,const float* key_norm,
                              runtime::KVCacheView cache,int layer,const int* position,const float* sine,const float* cosine,
                              int heads,int kv_heads,int head_size,float epsilon,bool split_half,cudaStream_t stream);
void decode_cache_attention(const float* query,runtime::KVCacheView cache,int layer,float* scores,float* output,
                            const int* position,int host_position,int heads,int kv_heads,int head_size,cudaStream_t stream);
}  // namespace kernel
