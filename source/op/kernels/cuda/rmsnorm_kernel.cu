#include <device_launch_parameters.h>
#include <cub/block/block_reduce.cuh>
#include "rmsnorm_kernel.cuh"
namespace kernel {
/**
 * 计算多维输入 in = (dim1, dim2), 计算在dim2维度上的rmsnorm
 */
static __global__ void row_rmsnorm_f32_dim(float* in, float* wei, float* out, int dim_size,
                                           int size, float eps) {
  // 想想输入是一个二维矩阵,然后每个block处理一行,如果是[b,s,d],那处理起来就是[b*s,d]
  // 然后这里的weight是一个和最后一维大小相同的一维向量,每一行的weight都是一样的
  // 参数的size是dim2的大小, dim_size是dim1的大小
  // 这里的weight是
  const int bid = blockIdx.x;
  const int tid = threadIdx.x;

  // 每个block处理一个dim2维度上的rmsnorm, 因此block数量等于dim1的大小,
  // block内线程并行计算dim2维度上的rmsnorm
  if (bid >= dim_size) {
    return;
  }

  float* block_in = in + bid * size;  // 每个block处理一个dim2维度上的rmsnorm,
                                      // 因此block_in是当前block处理的dim2维度上的输入数据
  float* block_out = out + bid * size;
  // 后续和一维版本的处理一样
  constexpr int pack_size = 4;
  const int pack_num = size / pack_size;
  const int pack_off = pack_size * pack_num;

  float sum = 0.0f;
  float4* in_pack = reinterpret_cast<float4*>(block_in);
  for (int i = tid; i < pack_num; i += blockDim.x) {
    float4 in_float4 = *(in_pack + i);
    sum += in_float4.x * in_float4.x;
    sum += in_float4.y * in_float4.y;
    sum += in_float4.z * in_float4.z;
    sum += in_float4.w * in_float4.w;
  }

  for (int i = pack_off + tid; i < size; i += blockDim.x) {
    sum += block_in[i] * block_in[i];
  }

  using BlockReduce = cub::BlockReduce<float, 128>;
  __shared__ typename BlockReduce::TempStorage temp;
  __shared__ float shared_val;
  sum = BlockReduce(temp).Sum(sum);
  if (threadIdx.x == 0) {
    shared_val = sum;
  }
  __syncthreads();
  sum = shared_val;
  const float scale = rsqrtf(sum / static_cast<float>(size) + eps);

  float4* wei_pack = reinterpret_cast<float4*>(wei);
  float4* out_pack = reinterpret_cast<float4*>(block_out);
  for (int i = tid; i < pack_num; i += blockDim.x) {
    float4 in_float4 = *(in_pack + i);
    float4 wei_float4 = *(wei_pack + i);
    *(out_pack + i) =
        make_float4(scale * in_float4.x * wei_float4.x, scale * in_float4.y * wei_float4.y,
                    scale * in_float4.z * wei_float4.z, scale * in_float4.w * wei_float4.w);
  }

  for (int i = pack_off + tid; i < size; i += blockDim.x) {
    block_out[i] = wei[i] * block_in[i] * scale;
  }
}

template <int32_t BLOCK_DIM>
static __global__ void row_rmsnorm_f32(float* in, float* wei, float* out, int size, float eps) {
  // 把in作为一个长度为size的一维向量
  // wei是权重，也是一个长度为size的一维向量
  const int tid = threadIdx.x;

  constexpr int pack_size = 4;            // 一次读取4个float
  const int pack_num = size / pack_size;  // pack_num是可以被4整除的部分的长度
  const int pack_off =
      pack_size * pack_num;  // pack_off是可以被4整除的部分的长度，剩余的部分需要单独处理

  float sum = 0.0f;
  // 先计算可以被4整除的部分，使用float4类型一次读取4个float，减少内存访问次数
  float4* in_pack = reinterpret_cast<float4*>(in);
  for (int i = tid; i < pack_num; i += blockDim.x) {
    float4 in_float4 = *(in_pack + i);
    sum += in_float4.x * in_float4.x;
    sum += in_float4.y * in_float4.y;
    sum += in_float4.z * in_float4.z;
    sum += in_float4.w * in_float4.w;
  }

  // 处理剩余的部分
  for (int i = pack_off + tid; i < size; i += blockDim.x) {
    sum += in[i] * in[i];
  }
  // 这里的sum是每个线程计算的部分和.
  // 怎么理解呢,先处理被4整除的部分,想象成把4个元素压缩成一个,然后每个线程以stride的方式处理这些压缩后的元素,最后在单独处理剩余的部分

  using BlockReduce =
      cub::BlockReduce<float, BLOCK_DIM>;  // 在block内进行reduce操作,参与的线程数是BLOCK_DIM
  __shared__ typename BlockReduce::TempStorage temp;  // BlockReduce需要的临时存储,每个block共享
  __shared__ float shared_val;                        // 用于存储所有线程总和
  sum = BlockReduce(temp).Sum(sum);
  if (threadIdx.x == 0) {
    shared_val = sum;
  }
  __syncthreads();
  sum = shared_val;
  const float scale = rsqrtf(sum / static_cast<float>(size) + eps);  // 计算输入平方根的倒数

  // 类似地,计算输出,先处理被4整除的部分,再处理剩余的部分
  float4* wei_pack = reinterpret_cast<float4*>(wei);
  float4* out_pack = reinterpret_cast<float4*>(out);
  for (int i = tid; i < pack_num; i += blockDim.x) {
    float4 in_float4 = *(in_pack + i);
    float4 wei_float4 = *(wei_pack + i);
    *(out_pack + i) =
        make_float4(scale * in_float4.x * wei_float4.x, scale * in_float4.y * wei_float4.y,
                    scale * in_float4.z * wei_float4.z, scale * in_float4.w * wei_float4.w);
  }

  for (int i = pack_off + tid; i < size; i += blockDim.x) {
    out[i] = wei[i] * in[i] * scale;
  }
}

void rmsnorm_kernel_cu(const tensor::Tensor& input, const tensor::Tensor& weight,
                       tensor::Tensor& output, void* stream) {
  CHECK(!input.is_empty());
  CHECK(!weight.is_empty());
  CHECK(!output.is_empty());

  CHECK(input.device_type() == base::DeviceType::kDeviceCUDA &&
        weight.device_type() == base::DeviceType::kDeviceCUDA &&
        output.device_type() == base::DeviceType::kDeviceCUDA);

#if defined(QWEN2_SUPPORT) || defined(QWEN3_SUPPORT)
  const float eps = 1e-6f;
#else
  const float eps = 1e-5f;
#endif
  const int32_t size = static_cast<int32_t>(input.size());
  float* in_ptr = const_cast<float*>(input.ptr<float>());
  float* wei_ptr = const_cast<float*>(weight.ptr<float>());
  float* out_ptr = const_cast<float*>(output.ptr<float>());
  constexpr int threads_num = 128;
  if (stream) {
    cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
    row_rmsnorm_f32<128><<<1, threads_num, 0, stream_>>>(in_ptr, wei_ptr, out_ptr, size, eps);
  } else {
    row_rmsnorm_f32<128><<<1, threads_num>>>(in_ptr, wei_ptr, out_ptr, size, eps);
  }
}

void rmsnorm_kernel_cu_dim(const tensor::Tensor& input, const tensor::Tensor& weight,
                           tensor::Tensor& output, int32_t dim, void* stream) {
  CHECK(!input.is_empty());
  CHECK(!weight.is_empty());
  CHECK(!output.is_empty());

  CHECK(input.device_type() == base::DeviceType::kDeviceCUDA &&
        weight.device_type() == base::DeviceType::kDeviceCUDA &&
        output.device_type() == base::DeviceType::kDeviceCUDA);

  const float eps = 1e-6f;
  const int32_t total_size = static_cast<int32_t>(input.size());
  const int32_t size = input.get_dim(input.dims_size() - 1);
  const int32_t dim_size = total_size / size;

  float* in_ptr = const_cast<float*>(input.ptr<float>());
  float* wei_ptr = const_cast<float*>(weight.ptr<float>());
  float* out_ptr = const_cast<float*>(output.ptr<float>());
  constexpr int threads_num = 128;
  if (stream) {
    cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
    row_rmsnorm_f32_dim<<<dim_size, threads_num, 0, stream_>>>(in_ptr, wei_ptr, out_ptr, dim_size,
                                                               size, eps);
  } else {
    row_rmsnorm_f32_dim<<<dim_size, threads_num>>>(in_ptr, wei_ptr, out_ptr, dim_size, size, eps);
  }
}
}  // namespace kernel