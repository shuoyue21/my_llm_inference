#include <gtest/gtest.h>
#include "source/op/kernels/kernels_interface.h"
#include "source/op/kernels/cpu/rope_kernel.h"
#include "source/op/kernels/cuda/rope_kernel.cuh"
namespace {
void compare_rope(int heads, int kv_heads, int head_size, bool use_stream) {
  int count=0; if(cudaGetDeviceCount(&count)!=cudaSuccess || count==0) GTEST_SKIP();
  auto cpu=base::CPUDeviceAllocatorFactory::get_instance();
  auto gpu=base::CUDADeviceAllocatorFactory::get_instance();
  const int seq=8, position=7;
  tensor::Tensor q(base::DataType::kDataTypeFp32, heads*head_size,true,cpu);
  tensor::Tensor k(base::DataType::kDataTypeFp32, kv_heads*head_size,true,cpu);
  for(int i=0;i<q.size();++i) q.index<float>(i)=std::sin(i*0.1f);
  for(int i=0;i<k.size();++i) k.index<float>(i)=std::cos(i*0.2f);
  auto q_gpu=q.clone(); auto k_gpu=k.clone(); q_gpu.to_cuda(); k_gpu.to_cuda();
  tensor::Tensor pos(base::DataType::kDataTypeInt32,1,true,cpu); pos.index<int32_t>(0)=position;
  tensor::Tensor sin_cpu(base::DataType::kDataTypeFp32,head_size*seq,true,cpu);
  tensor::Tensor cos_cpu(base::DataType::kDataTypeFp32,head_size*seq,true,cpu);
  tensor::Tensor sin_gpu(base::DataType::kDataTypeFp32,head_size*seq,true,gpu);
  tensor::Tensor cos_gpu(base::DataType::kDataTypeFp32,head_size*seq,true,gpu);
  cudaStream_t stream=nullptr;
  if(use_stream) ASSERT_EQ(cudaStreamCreate(&stream),cudaSuccess);
  kernel::sin_cos_cache_calc_cpu(head_size,seq,sin_cpu.ptr<float>(),cos_cpu.ptr<float>());
  kernel::sin_cos_cache_calc_cu(head_size,seq,sin_gpu,cos_gpu,stream);
  kernel::get_rope_kernel(base::DeviceType::kDeviceCPU)(heads*head_size,kv_heads*head_size,head_size,q,k,pos,sin_cpu,cos_cpu,nullptr);
  kernel::get_rope_kernel(base::DeviceType::kDeviceCUDA)(heads*head_size,kv_heads*head_size,head_size,q_gpu,k_gpu,pos,sin_gpu,cos_gpu,stream);
  ASSERT_EQ(cudaStreamSynchronize(stream),cudaSuccess);
  q_gpu.to_cpu(); k_gpu.to_cpu();
  for(int i=0;i<q.size();++i) EXPECT_NEAR(q.index<float>(i),q_gpu.index<float>(i),1e-5);
  for(int i=0;i<k.size();++i) EXPECT_NEAR(k.index<float>(i),k_gpu.index<float>(i),1e-5);
  if(use_stream) ASSERT_EQ(cudaStreamDestroy(stream),cudaSuccess);
}
}
TEST(test_rope_cu, mha_default_stream) { compare_rope(6,6,48,false); }
TEST(test_rope_cu, gqa_stream) { compare_rope(8,2,64,true); }
TEST(test_rope_cu, partial_block) { compare_rope(6,2,18,true); }
