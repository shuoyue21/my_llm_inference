#include <gtest/gtest.h>
#include <cmath>
#include <limits>
#include "runtime/kv_cache.h"
#include "source/op/kernels/cuda/online_attention.cuh"
#include "source/op/kernels/cuda/decode_kernels.cuh"
TEST(runtime_cuda, batched_attention_long_paged_gqa_matches_double_reference) {
  int count=0; if(cudaGetDeviceCount(&count)!=cudaSuccess || !count) GTEST_SKIP();
  auto config=std::make_shared<kernel::CudaConfig>();ASSERT_EQ(cudaStreamCreate(&config->stream),cudaSuccess);
  for(int length:{1,31,32,33,1537}) for(int hs:{6,128}) {
    const int heads=4,kv_heads=2,kv=kv_heads*hs;
    runtime::PagedKVCache cache(1,1568,kv,16,config);
    cache.begin_request();cache.ensure_tokens(length);
    std::vector<float> query(heads*hs),keys(size_t(length)*kv),values(keys.size()),actual(heads*hs),expected(heads*hs);
    for(size_t i=0;i<query.size();++i)query[i]=std::sin(i*.07f)*3;
    for(size_t i=0;i<keys.size();++i){keys[i]=std::cos(i*.031f)*3;values[i]=std::sin(i*.057f);}
    for(int t=0;t<length;++t){
      ASSERT_EQ(cudaMemcpyAsync(cache.slot(0,t).first,keys.data()+size_t(t)*kv,kv*sizeof(float),cudaMemcpyHostToDevice,config->stream),cudaSuccess);
      ASSERT_EQ(cudaMemcpyAsync(cache.slot(0,t).second,values.data()+size_t(t)*kv,kv*sizeof(float),cudaMemcpyHostToDevice,config->stream),cudaSuccess);
    }
    for(int head=0;head<heads;++head){
      std::vector<double> scores(length);double maximum=-1e30,total=0;
      for(int t=0;t<length;++t){double dot=0;for(int i=0;i<hs;++i)dot+=double(query[head*hs+i])*keys[size_t(t)*kv+(head/2)*hs+i];scores[t]=dot/std::sqrt(double(hs));maximum=std::max(maximum,scores[t]);}
      for(auto& score:scores){score=std::exp(score-maximum);total+=score;}
      for(int i=0;i<hs;++i){double value=0;for(int t=0;t<length;++t)value+=scores[t]*values[size_t(t)*kv+(head/2)*hs+i];expected[head*hs+i]=value/total;}
    }
    float *q=nullptr,*out=nullptr,*scores=nullptr;int* position=nullptr;runtime::KVCacheView* view=nullptr;
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&q),query.size()*sizeof(float)),cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&out),query.size()*sizeof(float)),cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&position),sizeof(int)),cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&view),sizeof(runtime::KVCacheView)),cudaSuccess);
    auto host_view=cache.view();int host_position=length-1;
    ASSERT_EQ(cudaMemcpyAsync(q,query.data(),query.size()*sizeof(float),cudaMemcpyHostToDevice,config->stream),cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(position,&host_position,sizeof(int),cudaMemcpyHostToDevice,config->stream),cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(view,&host_view,sizeof(host_view),cudaMemcpyHostToDevice,config->stream),cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&scores),heads*1568*sizeof(float)),cudaSuccess);
    kernel::batch_reference_attention(q,view,position,0,scores,out,1,heads,kv_heads,hs,config->stream);
    ASSERT_EQ(cudaMemcpyAsync(actual.data(),out,actual.size()*sizeof(float),cudaMemcpyDeviceToHost,config->stream),cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(config->stream),cudaSuccess);
    for(size_t i=0;i<actual.size();++i) EXPECT_NEAR(actual[i],expected[i],8e-5)<<"reference length="<<length<<" hs="<<hs;
    kernel::batch_online_attention(q,view,position,0,out,1,heads,kv_heads,hs,config->stream);
    ASSERT_EQ(cudaMemcpyAsync(actual.data(),out,actual.size()*sizeof(float),cudaMemcpyDeviceToHost,config->stream),cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(config->stream),cudaSuccess);
    for(size_t i=0;i<actual.size();++i) EXPECT_NEAR(actual[i],expected[i],8e-5)<<"length="<<length<<" hs="<<hs;
    cudaFree(scores);cudaFree(q);cudaFree(out);cudaFree(position);cudaFree(view);
  }
}
