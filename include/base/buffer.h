#ifndef FORWARD_INCLUDE_BASE_BUFFER_H_
#define FORWARD_INCLUDE_BASE_BUFFER_H_
#include <memory>
#include "base/alloc.h"
namespace base {
// buffer类用来实现对"一段内存"的管理
class Buffer : public NoCopyable, public std::enable_shared_from_this<Buffer> {
 private:
  size_t byte_size_ = 0;
  void* ptr_ = nullptr;
  bool use_external_ = false;  // 区分外部,内部内存,外部内存在析构是不自动释放
  DeviceType device_type_ = DeviceType::kDeviceUnknown;
  std::shared_ptr<DeviceAllocator> allocator_;

 public:
  explicit Buffer() = default;

  explicit Buffer(size_t byte_size, std::shared_ptr<DeviceAllocator> allocator = nullptr,
                  void* ptr = nullptr, bool use_external = false);
  virtual ~Buffer();
  // 手动分配内存(延迟分配场景)
  bool allocate();
  // 从另一个buffer拷贝数据
  void copy_from(const Buffer& buffer) const;

  void copy_from(const Buffer* buffer) const;
  // 返回数据指针
  void* ptr();

  const void* ptr() const;

  size_t byte_size() const;
  // 返回关联的分配器
  std::shared_ptr<DeviceAllocator> allocator() const;

  DeviceType device_type() const;

  void set_device_type(DeviceType device_type);

  std::shared_ptr<Buffer> get_shared_from_this();

  bool is_external() const;
};
}  // namespace base
#endif