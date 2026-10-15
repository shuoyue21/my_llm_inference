#include "paged_kernels.cuh"
#include "kv_cache_view.cuh"
#include "reduce_compat.cuh"
#include <cub/block/block_reduce.cuh>
#include <cfloat>
#include <cmath>
namespace kernel {
namespace {
__global__ void scatter_batch(const float* k,const float* v,runtime::KVCacheView cache,int layer,int rows) {
  size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i<size_t(rows)*cache.kv_dim){int token=i/cache.kv_dim,column=i%cache.kv_dim;kv_key(cache,layer,token)[column]=k[i];kv_value(cache,layer,token)[column]=v[i];}
}
__global__ void paged_causal_attention(const float* query,runtime::KVCacheView cache,int layer,float* scores,float* out,int rows,int heads,int kv_heads,int hs) {
  int head=blockIdx.x,row=blockIdx.y,tid=threadIdx.x,offset=(head/(heads/kv_heads))*hs;
  const float* q=query+(size_t(row)*heads+head)*hs;float* score=scores+(size_t(row)*heads+head)*rows;
  extern __shared__ float shared_q[];
  for(int i=tid;i<(hs+3)/4*4;i+=blockDim.x)shared_q[i]=i<hs?q[i]:0.f;
  __syncthreads();float maximum=-FLT_MAX,scale=1.f/sqrtf(float(hs));
  for(int t=tid;t<=row;t+=blockDim.x){const float* k=kv_key(cache,layer,t)+offset;float dot=0;for(int i=0;i<hs;++i)dot+=shared_q[i]*k[i];score[t]=dot*scale;maximum=fmaxf(maximum,score[t]);}
  using Reduce=cub::BlockReduce<float,256>;__shared__ Reduce::TempStorage storage;__shared__ float shared;
  maximum=Reduce(storage).Reduce(maximum,FloatMaximum{});if(tid==0)shared=maximum;__syncthreads();maximum=shared;__syncthreads();
  float sum=0;for(int t=tid;t<=row;t+=blockDim.x){score[t]=expf(score[t]-maximum);sum+=score[t];}
  sum=Reduce(storage).Sum(sum);if(tid==0)shared=sum;__syncthreads();for(int t=tid;t<=row;t+=blockDim.x)score[t]/=shared;__syncthreads();
  for(int i=tid;i<hs;i+=blockDim.x){float value=0;for(int t=0;t<=row;++t)value+=score[t]*kv_value(cache,layer,t)[offset+i];out[(size_t(row)*heads+head)*hs+i]=value;}
}
}
void paged_store_batch(const float* k,const float* v,runtime::KVCacheView cache,int layer,int rows,cudaStream_t stream){scatter_batch<<<(size_t(rows)*cache.kv_dim+255)/256,256,0,stream>>>(k,v,cache,layer,rows);}
void paged_prefill_attention(const float* q,runtime::KVCacheView cache,int layer,float* scores,float* out,int rows,int heads,int kv_heads,int hs,cudaStream_t stream){paged_causal_attention<<<dim3(heads,rows),256,((hs+3)/4*4)*sizeof(float),stream>>>(q,cache,layer,scores,out,rows,heads,kv_heads,hs);}
}  // namespace kernel
