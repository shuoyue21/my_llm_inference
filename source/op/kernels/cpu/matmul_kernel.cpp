#include "matmul_kernel.h"
#include "../kernels_interface.h"
#include "base/base.h"

namespace kernel {
void matmul_kernel_cpu(const tensor::Tensor& input, const tensor::Tensor& weight,
                       tensor::Tensor& output, float scale, const CudaConfig* config) {
  UNUSED(config);
  CHECK(input.is_empty() == false);
  CHECK(weight.is_empty() == false);
  CHECK(output.is_empty() == false);
  CHECK(input.device_type() == base::DeviceType::kDeviceCPU);
  CHECK(weight.device_type() == base::DeviceType::kDeviceCPU);
  CHECK(output.device_type() == base::DeviceType::kDeviceCPU);

  const float* input_ptr = input.ptr<float>();
  const float* weight_ptr = weight.ptr<float>();
  float* output_ptr = output.ptr<float>();

  CHECK_EQ(input.dims_size(), 1);
  CHECK_EQ(weight.dims_size(), 2);
  const int32_t rows = weight.get_dim(0);
  const int32_t cols = weight.get_dim(1);
  CHECK_EQ(input.size(), cols);
  CHECK_EQ(output.size(), rows);
  // Checkpoints store W as row-major [out_features, in_features].
  arma::fvec x(const_cast<float*>(input_ptr), cols, false, true);
  arma::fmat w(const_cast<float*>(weight_ptr), cols, rows, false, true);
  arma::fvec y(output_ptr, rows, false, true);
  y = scale * (w.t() * x);
}
}  // namespace kernel
