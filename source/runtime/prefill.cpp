#include "runtime/prefill.h"
#include <cublas_v2.h>
#include <stdexcept>
#include <string>
#include <utility>
#include "../op/kernels/cuda/prefill_kernels.cuh"
#include "../op/kernels/cuda/paged_kernels.cuh"

namespace runtime {
namespace {
void cuda_check(cudaError_t status) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string("Batched prefill CUDA: ") + cudaGetErrorString(status));
}
void blas_check(cublasStatus_t status) {
  if (status != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error("Batched prefill cuBLAS status " + std::to_string(int(status)));
}
struct DeviceBuffer {
  void* data = nullptr;
  size_t capacity = 0;
  ~DeviceBuffer() {
    if (data) cudaFree(data);
  }
  void reserve(size_t bytes) {
    if (bytes <= capacity) return;
    void* replacement = nullptr;
    cuda_check(cudaMalloc(&replacement, bytes));
    if (data) cuda_check(cudaFree(data));
    data = replacement;
    capacity = bytes;
  }
  float* floats(size_t count) {
    reserve(count * sizeof(float));
    return static_cast<float*>(data);
  }
  float* ptr() { return static_cast<float*>(data); }
};
}  // namespace
struct CudaPrefill::Impl {
  PrefillWeights weights;
  std::shared_ptr<kernel::CudaConfig> config;
  cublasHandle_t blas = nullptr;
  DeviceBuffer ids, hidden, normalized, query, key, value, attention, projected, gate, up, scores,
      all_logits;
  Impl(PrefillWeights data, std::shared_ptr<kernel::CudaConfig> cuda)
      : weights(std::move(data)), config(std::move(cuda)) {
    if (!config || weights.embedding.device_type() != base::DeviceType::kDeviceCUDA)
      throw std::invalid_argument("Batched prefill requires CUDA FP32 weights");
    blas_check(cublasCreate(&blas));
    blas_check(cublasSetStream(blas, config->stream));
    // Keep every input/accumulation FP32; do not silently select TF32.
    blas_check(cublasSetMathMode(blas, CUBLAS_PEDANTIC_MATH));
  }
  ~Impl() {
    if (config) cudaStreamSynchronize(config->stream);
    if (blas) cublasDestroy(blas);
  }
  void linear(const float* input, const Projection& projection, float* output, int rows) {
    const auto& weight = projection.weight;
    const int n = weight.get_dim(0), k = weight.get_dim(1);
    const float alpha = 1.f, beta = 0.f;
    // Row-major Y[T,N] = X[T,K] W[N,K]^T, expressed in column-major BLAS.
    blas_check(cublasSgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, n, rows, k, &alpha, weight.ptr<float>(),
                           k, input, k, &beta, output, n));
    if (!projection.bias.is_empty())
      kernel::prefill_bias(output, projection.bias.ptr<float>(), rows, n, config->stream);
  }
};
CudaPrefill::CudaPrefill(PrefillWeights weights, std::shared_ptr<kernel::CudaConfig> config)
    : impl_(std::make_unique<Impl>(std::move(weights), std::move(config))) {}
CudaPrefill::~CudaPrefill() = default;

void CudaPrefill::run(const std::vector<int32_t>& tokens, tensor::Tensor& logits,
                      const tensor::Tensor& sin_cache, const tensor::Tensor& cos_cache,
                      KVCacheView cache, const LogitsCallback& callback) {
  auto& s = *impl_;
  const auto& w = s.weights;
  const int rows = static_cast<int>(tokens.size()), kv = w.kv_heads * w.head_size;
  if (rows <= 0 || rows > w.context) throw std::invalid_argument("Invalid batched prompt length");
  for (int token : tokens)
    if (token < 0 || token >= w.vocab) throw std::invalid_argument("Invalid batched prompt token");
  const size_t hidden_count = size_t(rows) * w.hidden;
  s.ids.reserve(size_t(rows) * sizeof(int));
  float* x = s.hidden.floats(hidden_count);
  float* z = s.normalized.floats(hidden_count);
  float* q = s.query.floats(size_t(rows) * w.attention);
  float* k = s.key.floats(size_t(rows) * kv);
  float* v = s.value.floats(size_t(rows) * kv);
  float* a = s.attention.floats(size_t(rows) * w.attention);
  float* y = s.projected.floats(hidden_count);
  float* gate = s.gate.floats(size_t(rows) * w.intermediate);
  float* up = s.up.floats(size_t(rows) * w.intermediate);
  float* scores = s.scores.floats(size_t(rows) * w.heads * rows);
  auto stream = s.config->stream;
  cuda_check(cudaMemcpyAsync(s.ids.data, tokens.data(), size_t(rows) * sizeof(int),
                             cudaMemcpyHostToDevice, stream));
  kernel::prefill_embedding(static_cast<int*>(s.ids.data), w.embedding.ptr<float>(), x, rows,
                            w.hidden, stream);
  for (size_t layer = 0; layer < w.blocks.size(); ++layer) {
    const auto& block = w.blocks[layer];
    kernel::prefill_norm(x, block.attention_norm.ptr<float>(), z, rows, w.hidden, w.epsilon,
                         stream);
    s.linear(z, block.query, q, rows);
    s.linear(z, block.key, k, rows);
    s.linear(z, block.value, v, rows);
    if (!block.query_norm.is_empty()) {
      kernel::prefill_norm(q, block.query_norm.ptr<float>(), q, rows * w.heads, w.head_size,
                           w.epsilon, stream);
      kernel::prefill_norm(k, block.key_norm.ptr<float>(), k, rows * w.kv_heads, w.head_size,
                           w.epsilon, stream);
    }
    kernel::prefill_rope(q, k, sin_cache.ptr<float>(), cos_cache.ptr<float>(), rows, w.attention,
                         kv, w.head_size, w.split_half_rope, stream);
    if (cache.pages) {
      kernel::paged_store_batch(k, v, cache, static_cast<int>(layer), rows, stream);
      kernel::paged_prefill_attention(q, cache, static_cast<int>(layer), scores, a, rows,
                                     w.heads, w.kv_heads, w.head_size, stream);
    } else {
      const size_t cache_offset = layer * w.context * kv;
      cuda_check(cudaMemcpyAsync(cache.keys + cache_offset, k, size_t(rows) * kv * sizeof(float),
                                cudaMemcpyDeviceToDevice, stream));
      cuda_check(cudaMemcpyAsync(cache.values + cache_offset, v, size_t(rows) * kv * sizeof(float),
                                cudaMemcpyDeviceToDevice, stream));
      kernel::prefill_attention(q, k, v, scores, a, rows, w.heads, w.kv_heads, w.head_size, stream);
    }
    s.linear(a, block.output, y, rows);
    kernel::prefill_add(x, y, hidden_count, stream);
    kernel::prefill_norm(x, block.ffn_norm.ptr<float>(), z, rows, w.hidden, w.epsilon, stream);
    s.linear(z, block.gate, gate, rows);
    s.linear(z, block.up, up, rows);
    kernel::prefill_swiglu(gate, up, size_t(rows) * w.intermediate, stream);
    s.linear(gate, block.down, y, rows);
    kernel::prefill_add(x, y, hidden_count, stream);
  }
  if (callback) {
    kernel::prefill_norm(x, w.final_norm.ptr<float>(), z, rows, w.hidden, w.epsilon, stream);
    float* all = s.all_logits.floats(size_t(rows) * w.vocab);
    s.linear(z, w.classifier, all, rows);
    cuda_check(cudaMemcpyAsync(logits.ptr<float>(), all + size_t(rows - 1) * w.vocab,
                               w.vocab * sizeof(float), cudaMemcpyDeviceToDevice, stream));
    cuda_check(cudaStreamSynchronize(stream));
    for (int row = 0; row < rows; ++row) {
      tensor::Tensor view(base::DataType::kDataTypeFp32, w.vocab, false, nullptr,
                          all + size_t(row) * w.vocab);
      view.set_device_type(base::DeviceType::kDeviceCUDA);
      callback(row, view);
    }
  } else {
    kernel::prefill_norm(x + size_t(rows - 1) * w.hidden, w.final_norm.ptr<float>(), z, 1, w.hidden,
                         w.epsilon, stream);
    s.linear(z, w.classifier, logits.ptr<float>(), 1);
  }
  cuda_check(cudaGetLastError());
}
}  // namespace runtime
