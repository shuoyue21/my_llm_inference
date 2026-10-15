#pragma once
#include "base/cuda_config.h"
#include "tensor/tensor.h"
#include "runtime/kv_cache.h"
#include <functional>
#include <memory>
#include <vector>

namespace runtime {
using LogitsCallback = std::function<void(int, const tensor::Tensor&)>;
struct Projection {
  tensor::Tensor weight;
  tensor::Tensor bias;
};
struct PrefillBlock {
  tensor::Tensor attention_norm, ffn_norm, query_norm, key_norm;
  Projection query, key, value, output, gate, up, down;
};
struct PrefillWeights {
  int hidden = 0, attention = 0, intermediate = 0;
  int heads = 0, kv_heads = 0, head_size = 0, context = 0, vocab = 0;
  float epsilon = 1e-5f;
  bool split_half_rope = false;
  tensor::Tensor embedding, final_norm;
  Projection classifier;
  std::vector<PrefillBlock> blocks;
};

// One workspace/BLAS handle per resident model; K/V are shared with reference decode.
class CudaPrefill {
 public:
  CudaPrefill(PrefillWeights weights, std::shared_ptr<kernel::CudaConfig> config);
  ~CudaPrefill();
  CudaPrefill(const CudaPrefill&) = delete;
  CudaPrefill& operator=(const CudaPrefill&) = delete;
  void run(const std::vector<int32_t>& tokens, tensor::Tensor& logits,
           const tensor::Tensor& sin_cache, const tensor::Tensor& cos_cache,
           KVCacheView cache, const LogitsCallback& callback);
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace runtime
