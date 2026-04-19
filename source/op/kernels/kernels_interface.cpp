#include "kernels_interface.h"
#include <base/base.h>
#include "cpu/add_kernel.h"
#include "cpu/rmsnorm_kernel.hpp"
#include "cuda/add_kernel.cuh"
#include "cuda/rmsnorm_kernel.cuh"
namespace kernel {
AddKernel get_add_kernel(base::DeviceType device_type) {
  if (device_type == base::DeviceType::kDeviceCPU) {
    return add_kernel_cpu;
  } else if (device_type == base::DeviceType::kDeviceCUDA) {
    return add_kernel_cu;
  } else {
    LOG(FATAL) << "Unknown device type for get a add kernel.";
    return nullptr;
  }
}

RMSNormKernel get_rmsnorm_kernel(base::DeviceType device_type) {
  if (device_type == base::DeviceType::kDeviceCPU) {
    return rmsnorm_kernel_cpu;
  } else if (device_type == base::DeviceType::kDeviceCUDA) {
    return rmsnorm_kernel_cu;
  } else {
    LOG(FATAL) << "Unknown device type for get a rmsnorm kernel.";
    return nullptr;
  }
}

RMSNormKernelDim get_rmsnorm_dim_kernel(base::DeviceType device_type) {
  if (device_type == base::DeviceType::kDeviceCUDA) {
    return rmsnorm_kernel_cu_dim;
  } else {
    LOG(FATAL) << "Unknown device type for get a rmsnorm dim kernel.";
    return nullptr;
  }
}
}  // namespace kernel