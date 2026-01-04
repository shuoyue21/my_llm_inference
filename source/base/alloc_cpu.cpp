#include <glog/logging.h>
#include <cstdlib>
#include "base/alloc.h"

// #if (defined(_POSIX_ADVISORY_INFO) && (_POSIX_ADVISORY_INFO >= 200112L))
// #define FORWARD_HAVE_POSIX_MEMALIGN
// #endif

namespace base {
CPUDeviceAllocator::CPUDeviceAllocator() : DeviceAllocator(DeviceType::kDeviceCPU) {}

// void* CPUDeviceAllocator::allocate(size_t byte_size) const {
//   if (!byte_size) return nullptr;

// #ifdef FORWARD_HAVE_POSIX_MEMALIGN
//   void* data = nullptr;
//   const size_t alignment = (byte_size >= size_t(1024)) ? size_t(32) : size_t(16);
//   int status = posix_memalign(
//       (void**)&data, ((alignment >= sizeof(void*)) ? alignment : sizeof(void*)), byte_size);
//   if (status != 0) return nullptr;
//   return data;

// #else
//   void* data = std::malloc(byte_size);
//   return data;

// #endif
// }

/*
更优化版本实现,使用C++17的std::aligned_alloc
posix_memalign在2002年以后的系统几乎都有实现,所以也不用宏判断
*/
void* CPUDeviceAllocator::allocate(size_t byte_size) const {
  if (!byte_size) return nullptr;

  // size_t alignment = (byte_size >= 1024) ? 32 : 16;
  const size_t alignment = 64;

  /*aligned_alloc要求size必须是alignment的倍数,这里byte_size对alignment向上取整*/
  size_t aligned_size = (byte_size + alignment - 1) & ~(alignment - 1);

  return std::aligned_alloc(alignment, aligned_size);
}
void CPUDeviceAllocator::release(void* ptr) const {
  if (ptr) std::free(ptr);
}

std::shared_ptr<CPUDeviceAllocator> CPUDeviceAllocatorFactory::instance = nullptr;
}  // namespace base