#include <gtest/gtest.h>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>
#include <vector>
#include "runtime/kv_cache.h"
#include "tensor/tensor.h"
#include "source/op/kernels/cuda/paged_kernels.cuh"
#include "source/op/kernels/cuda/decode_kernels.cuh"
namespace {
bool paged_cuda_available(){int n=0;return cudaGetDeviceCount(&n)==cudaSuccess && n>0;}
}
TEST(runtime_cuda, paged_pool_growth_reuse_trim_and_context_bounds) {
  if(!paged_cuda_available())GTEST_SKIP();
  auto config=std::make_shared<kernel::CudaConfig>();ASSERT_EQ(cudaStreamCreate(&config->stream),cudaSuccess);
  runtime::PagedKVCache cache(2,23,12,4,config);const auto table=cache.view().pages;
  EXPECT_EQ(cache.stats().reserved_bytes,0);EXPECT_NE(table,nullptr);
  cache.begin_request();cache.ensure_tokens(1);auto first=cache.stats();
  EXPECT_EQ(first.allocations,1);EXPECT_EQ(first.mapped_pages,1);EXPECT_EQ(first.logical_tokens,1);
  cache.ensure_tokens(4);EXPECT_EQ(cache.stats().allocations,1);
  cache.ensure_tokens(5);EXPECT_EQ(cache.stats().allocations,2);
  cache.ensure_tokens(23);auto full=cache.stats();
  EXPECT_EQ(full.allocations,6);EXPECT_EQ(full.mapped_pages,6);EXPECT_GE(full.reserved_bytes,full.effective_bytes);
  EXPECT_THROW(cache.ensure_tokens(24),std::out_of_range);EXPECT_EQ(cache.stats().logical_tokens,23);
  EXPECT_THROW(cache.slot(2,0),std::out_of_range);
  EXPECT_THROW(cache.slot(0,23),std::out_of_range);
  std::set<float*> blocks;
  for(int page=0;page<6;++page){
    float* pointer=cache.slot(0,page*4).first;blocks.insert(pointer);
    cudaPointerAttributes attributes;ASSERT_EQ(cudaPointerGetAttributes(&attributes,pointer),cudaSuccess);EXPECT_EQ(attributes.type,cudaMemoryTypeDevice);
  }
  EXPECT_EQ(blocks.size(),6);EXPECT_NE(cache.slot(0,0).first,cache.slot(1,0).first);EXPECT_NE(cache.slot(0,0).first,cache.slot(0,0).second);
  cache.release_request();auto released=cache.stats();EXPECT_EQ(released.mapped_pages,0);EXPECT_EQ(released.free_pages,6);EXPECT_EQ(released.releases,6);EXPECT_EQ(released.reserved_bytes,full.reserved_bytes);
  cache.shuffle_free_pages(42);cache.begin_request();cache.ensure_tokens(23);EXPECT_EQ(cache.stats().allocations,0);EXPECT_EQ(cache.stats().reuses,6);EXPECT_EQ(cache.view().pages,table);
  cache.release_request();cache.trim();auto trimmed=cache.stats();
  EXPECT_EQ(trimmed.reserved_bytes,0);EXPECT_EQ(trimmed.free_pages,0);EXPECT_EQ(trimmed.lifetime_frees,6);EXPECT_EQ(trimmed.lifetime_trimmed_bytes,full.reserved_bytes);EXPECT_EQ(cache.view().pages,table);
  cache.begin_request();cache.ensure_tokens(1);EXPECT_EQ(cache.stats().allocations,1);EXPECT_EQ(cache.stats().lifetime_allocations,7);cache.release_request();cache.trim();EXPECT_EQ(cache.stats().reserved_bytes,0);
}
TEST(runtime_cuda, paged_random_mapping_causal_gqa_and_layer_separation) {
  if(!paged_cuda_available())GTEST_SKIP();
  constexpr int layers=2,context=23,page_tokens=4,heads=6,kv_heads=2,hs=6,kv=kv_heads*hs,valid=21;
  auto config=std::make_shared<kernel::CudaConfig>();ASSERT_EQ(cudaStreamCreate(&config->stream),cudaSuccess);
  runtime::PagedKVCache cache(layers,context,kv,page_tokens,config);const auto table=cache.view().pages;
  cache.begin_request();cache.ensure_tokens(context);cache.release_request();
  auto alloc=base::CPUDeviceAllocatorFactory::get_instance();
  tensor::Tensor query(base::DataType::kDataTypeFp32,context*heads*hs,true,alloc);
  tensor::Tensor keys(base::DataType::kDataTypeFp32,layers*context*kv,true,alloc),values(base::DataType::kDataTypeFp32,layers*context*kv,true,alloc);
  tensor::Tensor scores(base::DataType::kDataTypeFp32,context*heads*context,true,alloc),out(base::DataType::kDataTypeFp32,context*heads*hs,true,alloc);
  tensor::Tensor one_scores(base::DataType::kDataTypeFp32,heads*context,true,alloc),one_out(base::DataType::kDataTypeFp32,heads*hs,true,alloc);
  for(int i=0;i<query.size();++i)query.index<float>(i)=std::sin((i+1)*.11f);
  for(int layer=0;layer<layers;++layer)for(int t=0;t<context;++t)for(int i=0;i<kv;++i){
    int index=(layer*context+t)*kv+i;
    keys.index<float>(index)=t<valid?std::cos((index+1)*.07f):std::numeric_limits<float>::quiet_NaN();
    values.index<float>(index)=t<valid?std::sin((index+1)*.13f):std::numeric_limits<float>::quiet_NaN();
  }
  std::vector<float> expected(valid*heads*hs);
  for(int row=0;row<valid;++row)for(int head=0;head<heads;++head){
    std::vector<double> p(row+1);double sum=0;
    for(int t=0;t<=row;++t){double dot=0;for(int i=0;i<hs;++i)dot+=double(query.index<float>((row*heads+head)*hs+i))*keys.index<float>((context+t)*kv+(head/3)*hs+i);p[t]=std::exp(dot/std::sqrt(double(hs)));sum+=p[t];}
    for(int i=0;i<hs;++i){double value=0;for(int t=0;t<=row;++t)value+=p[t]/sum*values.index<float>((context+t)*kv+(head/3)*hs+i);expected[(row*heads+head)*hs+i]=value;}
  }
  query.to_cuda(config->stream);keys.to_cuda(config->stream);values.to_cuda(config->stream);scores.to_cuda(config->stream);out.to_cuda(config->stream);one_scores.to_cuda(config->stream);one_out.to_cuda(config->stream);
  std::vector<float> all(out.size()),single(heads*hs);
  std::vector<float*> previous_mapping;bool mapping_changed=false;
  for(uint32_t seed=0;seed<10;++seed){
    cache.shuffle_free_pages(seed);cache.begin_request();cache.ensure_tokens(context);EXPECT_EQ(cache.stats().allocations,0);EXPECT_EQ(cache.view().pages,table);
    std::vector<float*> mapping;for(int p=0;p<6;++p)mapping.push_back(cache.slot(0,p*page_tokens).first);
    if(!previous_mapping.empty() && mapping!=previous_mapping)mapping_changed=true;previous_mapping=mapping;
    for(int layer=0;layer<layers;++layer)kernel::paged_store_batch(keys.ptr<float>(layer*context*kv),values.ptr<float>(layer*context*kv),cache.view(),layer,context,config->stream);
    kernel::paged_prefill_attention(query.ptr<float>(),cache.view(),1,scores.ptr<float>(),out.ptr<float>(),context,heads,kv_heads,hs,config->stream);
    kernel::decode_cache_attention(query.ptr<float>((valid-1)*heads*hs),cache.view(),1,one_scores.ptr<float>(),one_out.ptr<float>(),nullptr,valid-1,heads,kv_heads,hs,config->stream);
    ASSERT_EQ(cudaMemcpyAsync(all.data(),out.ptr<float>(),out.byte_size(),cudaMemcpyDeviceToHost,config->stream),cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(single.data(),one_out.ptr<float>(),one_out.byte_size(),cudaMemcpyDeviceToHost,config->stream),cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(config->stream),cudaSuccess);
    for(size_t i=0;i<expected.size();++i)EXPECT_NEAR(all[i],expected[i],2e-5) << "seed="<<seed<<" index="<<i;
    for(int i=0;i<heads*hs;++i)EXPECT_NEAR(single[i],expected[(valid-1)*heads*hs+i],2e-5);
    cache.release_request();EXPECT_EQ(cache.stats().mapped_pages,0);
  }
  EXPECT_TRUE(mapping_changed);cache.trim();EXPECT_EQ(cache.stats().reserved_bytes,0);
}
