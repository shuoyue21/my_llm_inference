#pragma once
#include "runtime/prefill.h"
#include <cublas_v2.h>
namespace runtime {
// FP16/BF16 operands with FP32 accumulation; groupwise symmetric W8/W4
// multiply FP16 activations directly without materializing FP16 weights.
class PrecisionLinear {
 public:
  PrecisionLinear(const tensor::Tensor& source, const std::string& precision, cudaStream_t stream);
  ~PrecisionLinear();
  void run(const float* input, float* output, int rows, void* activation_workspace, void* weight_workspace,
           cublasHandle_t handle, cudaStream_t stream) const;
  void embedding(const int* tokens, float* output, int rows, cudaStream_t stream) const;
  uint64_t bytes() const;
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}
