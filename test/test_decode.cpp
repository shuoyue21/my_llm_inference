#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include "runtime/generation.h"
#include "model/llama3.h"
#if defined(QWEN2_SUPPORT)
#include "model/qwen2.h"
#elif defined(QWEN3_SUPPORT)
#include "model/qwen3.h"
#endif
#include "source/op/kernels/cuda/decode_kernels.cuh"
namespace {
bool has_cuda(){int count=0;return cudaGetDeviceCount(&count)==cudaSuccess && count>0;}
}
TEST(runtime_cuda, decode_fused_qkv_norm_rope_and_causal_gqa) {
  if(!has_cuda())GTEST_SKIP();
  constexpr int heads=6,kv_heads=2,hs=6,kw=kv_heads*hs,qw=heads*hs,context=7,pos=4;
  auto alloc=base::CPUDeviceAllocatorFactory::get_instance();
  for(bool split:{false,true})for(bool normalize:{false,true}) {
    tensor::Tensor packed(base::DataType::kDataTypeFp32,qw+2*kw,true,alloc);
    tensor::Tensor bias(base::DataType::kDataTypeFp32,qw+2*kw,true,alloc);
    tensor::Tensor qnorm(base::DataType::kDataTypeFp32,hs,true,alloc),knorm(base::DataType::kDataTypeFp32,hs,true,alloc);
    tensor::Tensor keys(base::DataType::kDataTypeFp32,context*kw,true,alloc),values(base::DataType::kDataTypeFp32,context*kw,true,alloc);
    tensor::Tensor sine(base::DataType::kDataTypeFp32,context*hs,true,alloc),cosine(base::DataType::kDataTypeFp32,context*hs,true,alloc);
    tensor::Tensor position(base::DataType::kDataTypeInt32,1,true,alloc);
    tensor::Tensor scores(base::DataType::kDataTypeFp32,heads*context,true,alloc),out(base::DataType::kDataTypeFp32,qw,true,alloc);
    std::vector<float> expected(qw+2*kw),ek(context*kw),ev(context*kw),eo(qw);
    for(int i=0;i<packed.size();++i){packed.index<float>(i)=std::sin(i*.1f);bias.index<float>(i)=std::cos(i*.7f)*.01f;expected[i]=packed.index<float>(i)+bias.index<float>(i);}
    for(int i=0;i<hs;++i){qnorm.index<float>(i)=1.f+i*.1f;knorm.index<float>(i)=.5f+i*.2f;}
    for(int i=0;i<keys.size();++i){ek[i]=keys.index<float>(i)=std::cos(i*.2f);ev[i]=values.index<float>(i)=std::sin(i*.3f);}
    for(int i=(pos+1)*kw;i<keys.size();++i)keys.index<float>(i)=values.index<float>(i)=std::numeric_limits<float>::quiet_NaN();
    for(int i=0;i<sine.size();++i){sine.index<float>(i)=1;cosine.index<float>(i)=0;}
    position.index<int32_t>(0)=pos;
    for(int head=0;head<heads+kv_heads;++head){
      int base=head<heads?head*hs:qw+(head-heads)*hs;
      if(normalize){double sum=0;for(int i=0;i<hs;++i)sum+=double(expected[base+i])*expected[base+i];
        for(int i=0;i<hs;++i)expected[base+i]=expected[base+i]/std::sqrt(sum/hs+1e-6)*(head<heads?qnorm.index<float>(i):knorm.index<float>(i));}
      for(int pair=0;pair<hs/2;++pair){int left=split?pair:pair*2,right=left+(split?hs/2:1);float a=expected[base+left],b=expected[base+right];expected[base+left]=-b;expected[base+right]=a;}
    }
    for(int i=0;i<kw;++i){ek[pos*kw+i]=expected[qw+i];ev[pos*kw+i]=expected[qw+kw+i];}
    for(int head=0;head<heads;++head){
      std::vector<double> probability(pos+1);double total=0;
      for(int t=0;t<=pos;++t){double dot=0;for(int i=0;i<hs;++i)dot+=double(expected[head*hs+i])*ek[t*kw+(head/3)*hs+i];probability[t]=std::exp(dot/std::sqrt(double(hs)));total+=probability[t];}
      for(int i=0;i<hs;++i){double value=0;for(int t=0;t<=pos;++t)value+=probability[t]/total*ev[t*kw+(head/3)*hs+i];eo[head*hs+i]=value;}
    }
    kernel::CudaConfig config;ASSERT_EQ(cudaStreamCreate(&config.stream),cudaSuccess);
    packed.to_cuda(config.stream);bias.to_cuda(config.stream);qnorm.to_cuda(config.stream);knorm.to_cuda(config.stream);
    keys.to_cuda(config.stream);values.to_cuda(config.stream);position.to_cuda(config.stream);sine.to_cuda(config.stream);cosine.to_cuda(config.stream);scores.to_cuda(config.stream);out.to_cuda(config.stream);
    kernel::decode_prepare_qkv(packed.ptr<float>(),bias.ptr<float>(),normalize?qnorm.ptr<float>():nullptr,normalize?knorm.ptr<float>():nullptr,keys.ptr<float>(),values.ptr<float>(),position.ptr<int32_t>(),sine.ptr<float>(),cosine.ptr<float>(),heads,kv_heads,hs,1e-6f,split,config.stream);
    kernel::decode_attention(packed.ptr<float>(),keys.ptr<float>(),values.ptr<float>(),scores.ptr<float>(),out.ptr<float>(),position.ptr<int32_t>(),context,heads,kv_heads,hs,config.stream);
    ASSERT_EQ(cudaStreamSynchronize(config.stream),cudaSuccess);
    packed.to_cpu();keys.to_cpu();values.to_cpu();out.to_cpu();
    for(int i=0;i<qw;++i){EXPECT_NEAR(packed.index<float>(i),expected[i],2e-5);EXPECT_NEAR(out.index<float>(i),eo[i],2e-5);}
    for(int i=0;i<kw;++i){EXPECT_NEAR(keys.index<float>(pos*kw+i),ek[pos*kw+i],2e-5);EXPECT_NEAR(values.index<float>(pos*kw+i),ev[pos*kw+i],2e-5);}
  }
}
TEST(runtime_cuda, decode_residual_norm_odd_width) {
  if(!has_cuda())GTEST_SKIP();
  auto alloc=base::CPUDeviceAllocatorFactory::get_instance();
  tensor::Tensor x(base::DataType::kDataTypeFp32,7,true,alloc),y(base::DataType::kDataTypeFp32,7,true,alloc),w(base::DataType::kDataTypeFp32,7,true,alloc),out(base::DataType::kDataTypeFp32,7,true,alloc);
  std::vector<float> residual(7),expected(7);double squared=0;
  for(int i=0;i<7;++i){x.index<float>(i)=std::sin(i*.3f);y.index<float>(i)=std::cos(i*.7f);w.index<float>(i)=.5f+i*.2f;residual[i]=x.index<float>(i)+y.index<float>(i);squared+=double(residual[i])*residual[i];}
  for(int i=0;i<7;++i)expected[i]=residual[i]/std::sqrt(squared/7+1e-5)*w.index<float>(i);
  kernel::CudaConfig config;ASSERT_EQ(cudaStreamCreate(&config.stream),cudaSuccess);
  x.to_cuda(config.stream);y.to_cuda(config.stream);w.to_cuda(config.stream);out.to_cuda(config.stream);
  kernel::decode_residual_norm(x.ptr<float>(),y.ptr<float>(),w.ptr<float>(),out.ptr<float>(),7,1e-5f,config.stream);
  ASSERT_EQ(cudaStreamSynchronize(config.stream),cudaSuccess);x.to_cpu();out.to_cpu();
  for(int i=0;i<7;++i){EXPECT_FLOAT_EQ(x.index<float>(i),residual[i]);EXPECT_NEAR(out.index<float>(i),expected[i],1e-6);}
}
TEST(runtime_cuda, decode_graph_sampling_advances_device_position) {
  if(!has_cuda())GTEST_SKIP();
  auto alloc=base::CPUDeviceAllocatorFactory::get_instance();
  tensor::Tensor logits(base::DataType::kDataTypeFp32,4,true,alloc),cursor(base::DataType::kDataTypeInt32,2,true,alloc);
  logits.index<float>(0)=-5;logits.index<float>(1)=-2;logits.index<float>(2)=-2;logits.index<float>(3)=-10;cursor.index<int32_t>(0)=0;cursor.index<int32_t>(1)=4;
  kernel::CudaConfig config;ASSERT_EQ(cudaStreamCreate(&config.stream),cudaSuccess);logits.to_cuda(config.stream);cursor.to_cuda(config.stream);ASSERT_EQ(cudaStreamSynchronize(config.stream),cudaSuccess);
  cudaGraph_t graph=nullptr;cudaGraphExec_t executable=nullptr;
  ASSERT_EQ(cudaStreamBeginCapture(config.stream,cudaStreamCaptureModeThreadLocal),cudaSuccess);
  kernel::decode_sample(logits.ptr<float>(),4,cursor.ptr<int32_t>(),cursor.ptr<int32_t>(1),config.stream);
  ASSERT_EQ(cudaStreamEndCapture(config.stream,&graph),cudaSuccess);ASSERT_EQ(cudaGraphInstantiateWithFlags(&executable,graph,0),cudaSuccess);
  for(int i=0;i<3;++i)ASSERT_EQ(cudaGraphLaunch(executable,config.stream),cudaSuccess);
  ASSERT_EQ(cudaStreamSynchronize(config.stream),cudaSuccess);cursor.to_cpu();EXPECT_EQ(cursor.index<int32_t>(0),1);EXPECT_EQ(cursor.index<int32_t>(1),7);
  EXPECT_EQ(cudaGraphExecDestroy(executable),cudaSuccess);EXPECT_EQ(cudaGraphDestroy(graph),cudaSuccess);
}
TEST(runtime_cuda, decode_graph_fusion_switches_and_capture_reuse) {
  if(!has_cuda())GTEST_SKIP();const char* path=std::getenv("LLAMA_TEST_MODEL");const char* tokenizer=std::getenv("LLAMA_TEST_TOKENIZER");
  if(!path || !tokenizer)GTEST_SKIP()<<"Set LLAMA_TEST_MODEL and LLAMA_TEST_TOKENIZER";
#if defined(QWEN3_SUPPORT)
  model::Qwen3Model model(base::TokenizerType::kEncodeBpe,tokenizer,path,false);
#elif defined(QWEN2_SUPPORT)
  model::Qwen2Model model(base::TokenizerType::kEncodeBpe,tokenizer,path,false);
#elif defined(LLAMA3_SUPPORT)
  model::LLama2Model model(base::TokenizerType::kEncodeBpe,tokenizer,path,false);
#else
  model::LLama2Model model(base::TokenizerType::kEncodeSpe,tokenizer,path,false);
#endif
  ASSERT_TRUE(bool(model.init(base::DeviceType::kDeviceCUDA)));
  for(int length:{1,33,129,8}){
    runtime::GenerationOptions options;options.max_new_tokens=8;options.ignore_eos=true;options.prefill_mode="batched";options.decode_mode="reference";
    for(int i=0;i<length;++i)options.prompt_token_ids.push_back(10+i%100);
    runtime::GenerationResult reference;ASSERT_TRUE(bool(runtime::generate(model,"",options,reference)));EXPECT_EQ(reference.decode_mode,"reference");EXPECT_FALSE(reference.decode_instrumented);
    for(auto flags:{std::pair<bool,bool>{false,false},{true,false},{false,true},{true,true}}){
      options.decode_mode="optimized";options.cuda_graph=flags.first;options.fusion=flags.second;
      runtime::GenerationResult actual;ASSERT_TRUE(bool(runtime::generate(model,"",options,actual)));
      EXPECT_EQ(actual.generated_tokens,reference.generated_tokens)<<"length="<<length<<" graph="<<flags.first<<" fusion="<<flags.second;
      EXPECT_EQ(actual.decode_mode,flags.first?"cuda_graph":"cuda_eager");EXPECT_EQ(actual.fusion_enabled,flags.second);EXPECT_TRUE(actual.decode_instrumented);
      if(flags.first){EXPECT_EQ(actual.decode_stats.graph_launches,7);EXPECT_EQ(actual.decode_stats.host_submissions,15);EXPECT_GT(actual.decode_stats.graph_nodes,0);EXPECT_GT(actual.decode_stats.graph_kernel_nodes,0);EXPECT_GT(actual.decode_stats.capture_total_ms,0);}
      else{EXPECT_EQ(actual.decode_stats.graph_launches,0);EXPECT_GT(actual.decode_stats.host_submissions,15);}
      if(flags.second)EXPECT_GT(actual.decode_stats.packed_weight_bytes,0);
      runtime::GenerationResult repeated;ASSERT_TRUE(bool(runtime::generate(model,"",options,repeated)));EXPECT_EQ(repeated.generated_tokens,reference.generated_tokens);
      EXPECT_EQ(repeated.decode_stats.capture_ms,0);EXPECT_EQ(repeated.decode_stats.captures_total,actual.decode_stats.captures_total);
    }
  }
  EXPECT_EQ(model.decode_stats().captures_total,2);
}
