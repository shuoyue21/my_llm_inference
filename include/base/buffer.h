#ifndef FORWARD_INCLUDE_BASE_BUFFER_H_
#define FORWARD_INCLUDE_BASE_BUFFER_H_
#include <memory>
#include "alloc.h"
namespace base {
class Buffer : public NoCopyable, public std::enable_shared_from_this<Buffer> {
 private:
  size_t byte_size_ = 0;
  void* ptr_ = nullptr;
  bool use_external_ = false;
  DeviceType device_type_ = DeviceType::kDeviceUnknown;
  std::shared_ptr<DeviceAllocator> allocator_;

 public:
  explicit Buffer() = default;

  explicit Buffer(size_t byte_size, std::shared_ptr<DeviceAllocator> allocator = nullptr,
                  void* ptr = nullptr, bool use_external = false);
  virtual ~Buffer();

  bool allocate();

  void copy_from(const Buffer& buffer) const;
};
}  // namespace base
#endif