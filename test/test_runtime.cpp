#include <gtest/gtest.h>
#include "op/matmul.h"
#include "sampler/argmax_sampler.h"
#include "source/op/kernels/kernels_interface.h"
TEST(runtime_cpu, non_square_matmul) {
  auto alloc = base::CPUDeviceAllocatorFactory::get_instance();
  tensor::Tensor input(base::DataType::kDataTypeFp32, 3, true, alloc);
  tensor::Tensor weight(base::DataType::kDataTypeFp32, 2, 3, true, alloc);
  tensor::Tensor output(base::DataType::kDataTypeFp32, 2, true, alloc);
  input.index<float>(0)=1; input.index<float>(1)=2; input.index<float>(2)=-1;
  for (int i=0;i<6;++i) weight.index<float>(i)=i+1;
  kernel::get_matmul_kernel(base::DeviceType::kDeviceCPU)(input, weight, output, 0.5f, nullptr);
  EXPECT_FLOAT_EQ(output.index<float>(0),1);
  EXPECT_FLOAT_EQ(output.index<float>(1),4);
}
TEST(runtime_cpu, reshape_preserves_data_and_metadata) {
  auto alloc = base::CPUDeviceAllocatorFactory::get_instance();
  tensor::Tensor input(base::DataType::kDataTypeFp32, 4, true, alloc);
  for (int i=0;i<4;++i) input.index<float>(i)=i+1;
  input.reshape({2,4});
  EXPECT_EQ(input.size(),8); EXPECT_EQ(input.get_dim(0),2); EXPECT_EQ(input.get_dim(1),4);
  for (int i=0;i<4;++i) EXPECT_FLOAT_EQ(input.index<float>(i),i+1);
  input.reshape({2}); EXPECT_EQ(input.size(),2); EXPECT_EQ(input.get_dim(0),2);
  input.reset(base::DataType::kDataTypeFp32,{3,4});
  EXPECT_EQ(input.size(),12); EXPECT_TRUE(input.is_empty());
}
TEST(runtime_cpu, argmax_tie_and_negative) {
  sampler::ArgmaxSampler sample(base::DeviceType::kDeviceCPU);
  float logits[] = {-5,-2,-2,-10};
  EXPECT_EQ(sample.sample(logits,4,nullptr),1);
}
TEST(runtime_cuda, argmax_small_vocab_and_stream) {
  int count=0; if (cudaGetDeviceCount(&count)!=cudaSuccess || count==0) GTEST_SKIP();
  auto alloc = base::CPUDeviceAllocatorFactory::get_instance();
  tensor::Tensor logits(base::DataType::kDataTypeFp32,4,true,alloc);
  logits.index<float>(0)=-5; logits.index<float>(1)=-2; logits.index<float>(2)=-2; logits.index<float>(3)=-10;
  logits.to_cuda();
  sampler::ArgmaxSampler sample(base::DeviceType::kDeviceCUDA);
  cudaStream_t stream; ASSERT_EQ(cudaStreamCreate(&stream),cudaSuccess);
  for(int i=0;i<20;++i) EXPECT_EQ(sample.sample(logits.ptr<float>(),4,stream),1);
  EXPECT_EQ(cudaStreamDestroy(stream),cudaSuccess);
}

#include "runtime/generation.h"
#include "model/llama3.h"
#include <cstdlib>
TEST(runtime_cpu, causal_gqa_matches_scalar_reference) {
  auto alloc=base::CPUDeviceAllocatorFactory::get_instance();
  const int heads=6, kv_heads=2, head_size=6, seq=7, pos=4, layer=1, kv=kv_heads*head_size;
  tensor::Tensor query(base::DataType::kDataTypeFp32,heads*head_size,true,alloc);
  tensor::Tensor key(base::DataType::kDataTypeFp32,2*seq*kv,true,alloc);
  tensor::Tensor value(base::DataType::kDataTypeFp32,2*seq*kv,true,alloc);
  tensor::Tensor score(base::DataType::kDataTypeFp32,heads*seq,true,alloc);
  tensor::Tensor output(base::DataType::kDataTypeFp32,heads*head_size,true,alloc);
  for(int i=0;i<query.size();++i) query.index<float>(i)=std::sin(float(i));
  for(int i=0;i<key.size();++i) { key.index<float>(i)=std::cos(i*0.2f); value.index<float>(i)=std::sin(i*0.3f); }
  // Future positions contain extreme values and must never enter the attention result.
  for(int t=pos+1;t<seq;++t) for(int i=0;i<kv;++i) {key.index<float>(layer*seq*kv+t*kv+i)=10000; value.index<float>(layer*seq*kv+t*kv+i)=-10000;}
  kernel::get_mha_kernel(base::DeviceType::kDeviceCPU)(pos,heads,layer,seq,kv,heads/kv_heads,head_size,output,query,score,key,value,base::DeviceType::kDeviceCPU,nullptr);
  for(int h=0;h<heads;++h) {
    double weights[pos+1], sum=0;
    for(int t=0;t<=pos;++t) {
      double dot=0;
      for(int i=0;i<head_size;++i) dot+=query.index<float>(h*head_size+i)*key.index<float>(layer*seq*kv+t*kv+(h/(heads/kv_heads))*head_size+i);
      weights[t]=std::exp(dot/std::sqrt(double(head_size))); sum+=weights[t];
    }
    for(int i=0;i<head_size;++i) {
      double expected=0;
      for(int t=0;t<=pos;++t) expected+=weights[t]/sum*value.index<float>(layer*seq*kv+t*kv+(h/(heads/kv_heads))*head_size+i);
      EXPECT_NEAR(output.index<float>(h*head_size+i),expected,1e-5);
    }
  }
  int devices=0;
  if(cudaGetDeviceCount(&devices)==cudaSuccess && devices>0) {
    auto expected=output.clone(); query.to_cuda(); key.to_cuda(); value.to_cuda(); score.to_cuda(); output.to_cuda();
    kernel::CudaConfig config;
    ASSERT_EQ(cudaStreamCreate(&config.stream),cudaSuccess);
    kernel::get_mha_kernel(base::DeviceType::kDeviceCUDA)(pos,heads,layer,seq,kv,heads/kv_heads,head_size,output,query,score,key,value,base::DeviceType::kDeviceCUDA,&config);
    ASSERT_EQ(cudaStreamSynchronize(config.stream),cudaSuccess); output.to_cpu();
    for(int i=0;i<output.size();++i) EXPECT_NEAR(output.index<float>(i),expected.index<float>(i),1e-5);
  }
}
#if !defined(LLAMA3_SUPPORT) && !defined(QWEN2_SUPPORT) && !defined(QWEN3_SUPPORT)
TEST(runtime_cpu, generation_repeated_request_and_boundaries) {
  const char* path=std::getenv("LLAMA_TEST_MODEL"); const char* tokenizer=std::getenv("LLAMA_TEST_TOKENIZER");
  if(!path || !tokenizer) GTEST_SKIP() << "Set LLAMA_TEST_MODEL and LLAMA_TEST_TOKENIZER";
  model::LLama2Model model(base::TokenizerType::kEncodeSpe,tokenizer,path,false);
  ASSERT_TRUE(bool(model.init(base::DeviceType::kDeviceCPU)));
  runtime::GenerationOptions options; options.max_new_tokens=4;
  runtime::GenerationResult first,second;
  ASSERT_TRUE(bool(runtime::generate(model,"Once upon a time",options,first)));
  ASSERT_TRUE(bool(runtime::generate(model,"Once upon a time",options,second)));
  EXPECT_EQ(first.generated_tokens,second.generated_tokens);
  options.max_new_tokens=1;
  ASSERT_TRUE(bool(runtime::generate(model,"",options,second)));
  EXPECT_EQ(second.prompt_tokens.size(),1); EXPECT_EQ(second.generated_tokens.size(),1);
  options.max_new_tokens=0;
  ASSERT_TRUE(bool(runtime::generate(model,"hello",options,second))); EXPECT_TRUE(second.generated_tokens.empty());
  options.max_new_tokens=model.context_length()+1;
  EXPECT_FALSE(bool(runtime::generate(model,"hello",options,second)));
  options.max_new_tokens=-1;
  EXPECT_FALSE(bool(runtime::generate(model,"hello",options,second)));
}
#endif
