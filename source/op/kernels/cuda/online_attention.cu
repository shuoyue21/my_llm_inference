#include "online_attention.cuh"
#include "kv_cache_view.cuh"
#include "reduce_compat.cuh"
#include <cub/block/block_reduce.cuh>
#include <cfloat>
#include <cmath>
namespace kernel {
namespace {
// A block consumes 32 keys at a time. Only scores for this tile are stored;
// the running maximum/norm rescale the accumulated values exactly as softmax.
__global__ void online_attention(const float* query,const runtime::KVCacheView* caches,
    const int* positions,int layer,float* output,int heads,int kv_heads,int hs) {
  const int row=blockIdx.x,head=blockIdx.y,tid=threadIdx.x,warp=tid/32,lane=tid%32;
  const int kv_head=head/(heads/kv_heads),length=positions[row]+1;
  const auto cache=caches[row];
  const float* q=query+(size_t(row)*heads+head)*hs;
  using Reduce=cub::BlockReduce<float,256>;
  __shared__ Reduce::TempStorage storage;
  __shared__ float scores[32],weights[32],maximum,norm,scale;
  if(tid==0){maximum=-FLT_MAX;norm=0;}
  float accumulated=0;
  __syncthreads();
  for(int start=0;start<length;start+=32){
    const int count=min(32,length-start);
    for(int t=warp;t<count;t+=8){
      const float* key=kv_key(cache,layer,start+t)+kv_head*hs;
      float dot=0;
      for(int i=lane;i<hs;i+=32)dot+=q[i]*key[i];
      for(int offset=16;offset>0;offset/=2)dot+=__shfl_down_sync(0xffffffff,dot,offset);
      if(lane==0)scores[t]=dot*rsqrtf(float(hs));
    }
    __syncthreads();
    float tile_max=Reduce(storage).Reduce(tid<count?scores[tid]:-FLT_MAX,FloatMaximum{});
    if(tid==0){float next=fmaxf(maximum,tile_max);scale=expf(maximum-next);maximum=next;}
    __syncthreads();
    float probability=tid<count?expf(scores[tid]-maximum):0;
    if(tid<count)weights[tid]=probability;
    float tile_norm=Reduce(storage).Sum(probability);
    if(tid==0)norm=norm*scale+tile_norm;
    __syncthreads();
    if(tid<hs){
      accumulated*=scale;
      for(int t=0;t<count;++t)accumulated+=weights[t]*kv_value(cache,layer,start+t)[kv_head*hs+tid];
    }
    __syncthreads();
  }
  if(tid<hs)output[(size_t(row)*heads+head)*hs+tid]=accumulated/norm;
}
}
void batch_online_attention(const float* q,const runtime::KVCacheView* caches,const int* positions,
    int layer,float* output,int rows,int heads,int kv_heads,int hs,cudaStream_t stream){
  online_attention<<<dim3(rows,heads),256,0,stream>>>(q,caches,positions,layer,output,heads,kv_heads,hs);
}
}
