#pragma once
#include <cuda_runtime_api.h>
#include <cstddef>
namespace kernel {
void prefill_embedding(const int* tokens, const float* table, float* output, int rows, int width, cudaStream_t stream);
void prefill_norm(const float* input, const float* weight, float* output, int rows, int width, float epsilon, cudaStream_t stream);
void prefill_bias(float* output, const float* bias, int rows, int width, cudaStream_t stream);
void prefill_add(float* residual, const float* update, size_t count, cudaStream_t stream);
void prefill_swiglu(float* gate, const float* up, size_t count, cudaStream_t stream);
void prefill_rope(float* query, float* key, const float* sin_cache, const float* cos_cache, int rows, int query_width, int key_width, int head_size, bool split_half, cudaStream_t stream);
void prefill_attention(const float* query, const float* key, const float* value, float* scores, float* output, int rows, int heads, int kv_heads, int head_size, cudaStream_t stream);
}  // namespace kernel
