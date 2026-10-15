#pragma once
#include "tensor/tensor.h"
namespace kernel {
void softmax_inplace_cu(const tensor::Tensor&, void*);
void scale_inplace_cu(float, const tensor::Tensor&, void*);
}
