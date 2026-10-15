#ifndef LLAMA_INFER_RMSNORM_KERNEL_H
#define LLAMA_INFER_RMSNORM_KERNEL_H
#include "tensor/tensor.h"
namespace kernel {
void rmsnorm_kernel_cpu(const tensor::Tensor& input, const tensor::Tensor& weight,
                        tensor::Tensor& output, void* stream = nullptr);
void rmsnorm_dim_kernel_cpu(const tensor::Tensor&, const tensor::Tensor&, tensor::Tensor&, int32_t, void*);
}  // namespace kernel
#endif  // LLAMA_INFER_RMSNORM_KERNEL_H