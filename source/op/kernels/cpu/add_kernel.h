#ifndef FORWARD_INCLUDE_OP_KERNEL
#define FORWARD_INCLUDE_OP_KERNEL
#include "tensor/tensor.h"
namespace kernel {
void add_kernel_cpu(const tensor::Tensor& input1, const tensor::Tensor& input2,
                    tensor::Tensor& output, void* stream = nullptr);
}  // namespace kernel
#endif