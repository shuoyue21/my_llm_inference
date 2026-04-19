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

  int32_t input_dim0 = 1;
  int32_t input_dim1 = 1;

  if (input.dims_size() == 1) {
    input_dim1 = input.get_dim(0);
  } else if (input.dims_size() == 2) {
    input_dim0 = input.get_dim(0);
    input_dim1 = input.get_dim(1);
  } else {
    LOG(FATAL) << "The input tensor has a wrong dims size in the matmul kernel.";
  }

  CHECK_EQ(weight.dims_size(), 2);
  int32_t weight_dim0 = weight.get_dim(0);
  int32_t weight_dim1 = weight.get_dim(1);
  CHECK_EQ(input_dim1, weight_dim0);

  CHECK_EQ(output.size(), weight_dim0 * input_dim1);

  arma::fmat input_mat(const_cast<float*>(input_ptr), input_dim1, input_dim0, false, true);
  arma::fmat weight_mat(const_cast<float*>(weight_ptr), weight_dim1, weight_dim0, false, true);
  arma::fmat output_mat(output_ptr, weight_dim1, input_dim0, false, true);
  output_mat = ((input_mat * weight_mat)) * scale;
}
}  // namespace kernel