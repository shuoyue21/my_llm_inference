#include "add_kernel.cuh"

namespace kernel {
__global__ void add_kernel_naive(int32_t size, const float* in1, const float* in2, float* out) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < size) {
    out[idx] = in1[idx] + in2[idx];
  }
}

__global__ void add_kernel_grid_stride(int32_t N, const float* input1, const float* input2,
                                       float* output) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;  // 一个grid的总线程数
  for (int i = tid; i < N; i += stride) {
    output[i] = input1[i] + input2[i];
  }
}

__global__ void add_kernel_float4_safe(const float* input1, const float* input2, float* output,
                                       int N) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  // 处理能被4整除的部分
  int N4 = N / 4;
  const float4* input1_4 = reinterpret_cast<const float4*>(input1);
  const float4* input2_4 = reinterpret_cast<const float4*>(input2);
  float4* output4 = reinterpret_cast<float4*>(output);

  for (int i = tid; i < N4; i += stride) {
    float4 a = input1_4[i];
    float4 b = input2_4[i];
    float4 c;
    c.x = a.x + b.x;
    c.y = a.y + b.y;
    c.z = a.z + b.z;
    c.w = a.w + b.w;
    output4[i] = c;
  }

  // 标量方式处理尾部数据
  // 计算剩余未处理的元素起始索引
  int tail_start = N4 * 4;
  for (int i = tail_start + tid; i < N; i += stride) {
    output[i] = input1[i] + input2[i];  // 会退到普通的float访存
  }
}

void add_kernel_cu(const tensor::Tensor& input1, const tensor::Tensor& input2,
                   tensor::Tensor output, void* stream) {
  CHECK_EQ(input1.is_empty(), false);
  CHECK_EQ(input2.is_empty(), false);
  CHECK_EQ(output.is_empty(), false);
  int32_t size = static_cast<int32_t>(input1.size());
  CHECK_EQ(size, input2.size());
  CHECK_EQ(size, output.size());
  int32_t thread_num = 512;
  int32_t block_num = (size + thread_num - 1) / thread_num;
  if (stream) {
    cudaStream_t stream_ = static_cast<CUstream_st*>(stream);
    add_kernel_grid_stride<<<thread_num, block_num, 0, stream_>>>(
        size, input1.ptr<const float>(), input2.ptr<const float>(), output.ptr<float>());
  } else {
    add_kernel_grid_stride<<<thread_num, block_num>>>(
        size, input1.ptr<const float>(), input2.ptr<const float>(), output.ptr<float>());
  }
}
}  // namespace kernel