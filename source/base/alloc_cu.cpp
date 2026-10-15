#include <cuda_runtime_api.h>
#include "base/alloc.h"
namespace base {
CUDADeviceAllocator::CUDADeviceAllocator() : DeviceAllocator(DeviceType::kDeviceCUDA) {}

void* CUDADeviceAllocator::allocate(size_t byte_size) const {
  int id = -1;
  cudaError_t state = cudaGetDevice(&id);
  CHECK(state == cudaSuccess);
  if (byte_size > 1024 * 1024) {
    auto& big_buffers = big_buffers_map_[id];
    int sel_id = -1;
    for (int i = 0; i < big_buffers.size(); ++i) {
      if (big_buffers[i].byte_size >= byte_size and !big_buffers[i].busy and
          big_buffers[i].byte_size - byte_size < 1 * 1024 * 1024) {
        if (sel_id == -1 or big_buffers[sel_id].byte_size > big_buffers[i].byte_size) {
          sel_id = i;
        }
      }
    }
    if (sel_id != -1) {
      big_buffers[sel_id].busy = true;
      return big_buffers[sel_id].data;
    }

    void* ptr = nullptr;
    state = cudaMalloc(&ptr, byte_size);
    if (state != cudaSuccess) {
      char buf[256];
      snprintf(buf, 256,
               "Error: CUDA error when allocating %lu MB memory! maybe there's no enough memory "
               "left on  device.",
               byte_size >> 20);
      LOG(ERROR) << buf;
      return nullptr;
    }
    big_buffers.emplace_back(ptr, byte_size, true);
    return ptr;
  }

  auto& cuda_buffers = cuda_buffers_map_[id];
  for (int i = 0; i < cuda_buffers.size(); ++i) {
    if (cuda_buffers[i].byte_size >= byte_size and !cuda_buffers[i].busy) {
      cuda_buffers[i].busy = true;
      no_busy_cnt_[id] -= cuda_buffers[i].byte_size;
      return cuda_buffers[i].data;
    }
  }

  void* ptr = nullptr;
  state = cudaMalloc(&ptr, byte_size);
  if (state != cudaSuccess) {
    char buf[256];
    snprintf(buf, 256,
             "Error: CUDA error when allocating %lu MB memory! maybe there's no enough memory "
             "left on  device.",
             byte_size >> 20);
    LOG(ERROR) << buf;
    return nullptr;
  }
  cuda_buffers.emplace_back(ptr, byte_size, true);
  return ptr;
}

void CUDADeviceAllocator::release(void* ptr) const {
  if (!ptr) {
    return;
  }
  if (cuda_buffers_map_.empty()) {
    return;
  }
  cudaError_t state = cudaSuccess;
  for (auto& [id, buffers] : cuda_buffers_map_) {
    // 如果某张卡的"空闲小内存"总和超过1GB
    if (no_busy_cnt_[id] > 1024 * 1024 * 1024) {
      std::vector<CudaMemoryBuffer> temp;
      // 遍历该卡的所有缓存块
      for (int i = 0; i < buffers.size(); ++i) {
        // 如果是空闲的,则释放
        if (!buffers[i].busy) {
          state = cudaSetDevice(id);  // 切换到对应的卡
          state = cudaFree(buffers[i].data);
          CHECK(state == cudaSuccess) << "Error:CUDA error when release memory on device" << id;
        } else {
          temp.push_back(buffers[i]);
        }
      }
      buffers = temp;
      no_busy_cnt_[id] = 0;
    }
  }

  for (auto& [id, buffers] : cuda_buffers_map_) {
    for (int i = 0; i < buffers.size(); ++i) {
      if (buffers[i].data == ptr) {
        no_busy_cnt_[id] += buffers[i].byte_size;
        buffers[i].busy = false;
        return;
      }
    }
    auto& big_buffers = big_buffers_map_[id];
    for (int i = 0; i < big_buffers.size(); ++i) {
      if (big_buffers[i].data == ptr) {
        big_buffers[i].busy = false;
        return;
      }
    }
  }
  state = cudaFree(ptr);
  CHECK(state == cudaSuccess) << "Error:CUDA error when release memory on device";
}
std::shared_ptr<CUDADeviceAllocator> CUDADeviceAllocatorFactory::instance = nullptr;
}  // namespace base