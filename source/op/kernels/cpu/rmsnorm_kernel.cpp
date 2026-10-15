#include "rmsnorm_kernel.hpp"

namespace kernel {
void rmsnorm_kernel_cpu(const tensor::Tensor& input, const tensor::Tensor& weight,
                        tensor::Tensor& output, void* stream) {
  UNUSED(stream);
  CHECK(!input.is_empty());
  CHECK(!weight.is_empty());
  CHECK(!output.is_empty());

  CHECK(input.device_type() == base::DeviceType::kDeviceCPU &&
        weight.device_type() == base::DeviceType::kDeviceCPU &&
        output.device_type() == base::DeviceType::kDeviceCPU);

  const float* in_ptr = input.ptr<float>();
  const float* wei_ptr = weight.ptr<float>();
  const float* out_ptr = output.ptr<float>();
  const int32_t dim = static_cast<int32_t>(input.size());

  arma::fvec in_tensor(const_cast<float*>(in_ptr), dim, false, true);
  arma::fvec out_tensor(const_cast<float*>(out_ptr), dim, false, true);
  arma::fvec wei_tensor(const_cast<float*>(wei_ptr), dim, false, true);

#if defined(QWEN2_SUPPORT) || defined(QWEN3_SUPPORT)
  const float eps = 1e-6f;
#else
  const float eps = 1e-5f;
#endif

  const float mean = arma::as_scalar(arma::mean(arma::pow(in_tensor, 2))) + eps;
  const float rsqrt = 1.f / std::sqrt(mean);
  out_tensor = wei_tensor % (rsqrt * in_tensor);
}
void rmsnorm_dim_kernel_cpu(const tensor::Tensor& input, const tensor::Tensor& weight,
                            tensor::Tensor& output, int32_t dim, void* stream) {
  CHECK_GT(dim, 0);
  CHECK_EQ(input.size() % dim, 0);
  CHECK_EQ(output.size(), input.size());
  for (size_t offset = 0; offset < input.size(); offset += dim) {
    tensor::Tensor row(base::DataType::kDataTypeFp32, dim, false, nullptr, const_cast<float*>(input.ptr<float>(offset)));
    tensor::Tensor out(base::DataType::kDataTypeFp32, dim, false, nullptr, output.ptr<float>(offset));
    row.set_device_type(base::DeviceType::kDeviceCPU); out.set_device_type(base::DeviceType::kDeviceCPU);
    rmsnorm_kernel_cpu(row, weight, out, stream);
  }
}
}  // namespace kernel