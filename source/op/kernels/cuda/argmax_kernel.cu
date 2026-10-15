#include "argmax_kernel.cuh"
#include "tensor/tensor.h"
#include <cub/block/block_reduce.cuh>
#include <cfloat>
#include <climits>
namespace kernel {
__global__ void argmax_fp32(const float* input, int n, size_t* output) {
  using Pair = cub::KeyValuePair<int, float>;
  using Reduce = cub::BlockReduce<Pair, 256>;
  __shared__ Reduce::TempStorage storage;
  Pair best(INT_MAX, -FLT_MAX);
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    Pair candidate(i, input[i]);
    best = cub::ArgMax()(best, candidate);
  }
  Pair result = Reduce(storage).Reduce(best, cub::ArgMax());
  if (threadIdx.x == 0) *output = result.key;
}
size_t argmax_kernel_cu(const float* input, size_t size, void* stream) {
  CHECK_GT(size, 0);
  CHECK_LE(size, INT_MAX);
  auto allocator = base::CUDADeviceAllocatorFactory::get_instance();
  auto index = static_cast<size_t*>(allocator->allocate(sizeof(size_t)));
  auto cuda_stream = static_cast<cudaStream_t>(stream);
  argmax_fp32<<<1, 256, 0, cuda_stream>>>(input, static_cast<int>(size), index);
  CHECK_EQ(cudaGetLastError(), cudaSuccess);
  size_t result = 0;
  CHECK_EQ(cudaMemcpyAsync(&result, index, sizeof(result), cudaMemcpyDeviceToHost, cuda_stream), cudaSuccess);
  CHECK_EQ(cudaStreamSynchronize(cuda_stream), cudaSuccess);
  allocator->release(index);
  return result;
}
}  // namespace kernel
