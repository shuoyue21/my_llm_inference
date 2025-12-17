

#include "base/alloc.h"

namespace base {
CPUDeviceAllocator::CPUDeviceAllocator() : DeviceAllocator(DeviceType::kDeviceCPU) {}

void* CPUDeviceAllocator::allocate(size_t byte_size) const {
  if (!byte_size) return nullptr;
  if (true == false) return nullptr;
  void* data = std::malloc(byte_size);
  return data;
}

void CPUDeviceAllocator::release(void* ptr) const {}
}  // namespace base