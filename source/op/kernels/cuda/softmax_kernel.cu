#include "softmax_kernel.cuh"
#include <cub/block/block_reduce.cuh>
#include <cfloat>
namespace kernel {
__global__ void scale_values(float* x, size_t n, float scale) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) x[i] *= scale;
}
__global__ void softmax_values(float* x, size_t n) {
  using Reduce = cub::BlockReduce<float, 256>;
  __shared__ Reduce::TempStorage storage;
  __shared__ float shared;
  float value = -FLT_MAX;
  for (size_t i = threadIdx.x; i < n; i += blockDim.x) value = fmaxf(value, x[i]);
  value = Reduce(storage).Reduce(value, cub::Max());
  if (threadIdx.x == 0) shared = value;
  __syncthreads();
  float maximum = shared;
  __syncthreads();
  float sum = 0;
  for (size_t i = threadIdx.x; i < n; i += blockDim.x) { x[i] = expf(x[i] - maximum); sum += x[i]; }
  sum = Reduce(storage).Sum(sum);
  if (threadIdx.x == 0) shared = sum;
  __syncthreads();
  for (size_t i = threadIdx.x; i < n; i += blockDim.x) x[i] /= shared;
}
void scale_inplace_cu(float scale, const tensor::Tensor& x, void* stream) {
  CHECK_GT(x.size(), 0);
  scale_values<<<(x.size()+255)/256, 256, 0, static_cast<cudaStream_t>(stream)>>>(const_cast<float*>(x.ptr<float>()), x.size(), scale);
}
void softmax_inplace_cu(const tensor::Tensor& x, void* stream) {
  CHECK_GT(x.size(), 0);
  softmax_values<<<1, 256, 0, static_cast<cudaStream_t>(stream)>>>(const_cast<float*>(x.ptr<float>()), x.size());
}
}
