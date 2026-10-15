#include "decode_kernels.cuh"
#include "reduce_compat.cuh"
#include "kv_cache_view.cuh"
#include <cub/block/block_reduce.cuh>
#include <cfloat>
#include <climits>
#include <cmath>
namespace kernel {
namespace {
__global__ void gather_device_token(const int* token,const float* table,float* out,int width) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<width)out[i]=table[size_t(*token)*width+i];
}
__global__ void residual_norm(float* x,const float* update,const float* weight,float* out,int width,float epsilon) {
  int tid=threadIdx.x;float sum=0;
  for(int i=tid;i<width;i+=blockDim.x){float v=x[i]+update[i];x[i]=v;sum+=v*v;}
  using Reduce=cub::BlockReduce<float,128>;__shared__ Reduce::TempStorage storage;__shared__ float inverse;
  sum=Reduce(storage).Sum(sum);if(tid==0)inverse=rsqrtf(sum/width+epsilon);__syncthreads();
  for(int i=tid;i<width;i+=blockDim.x)out[i]=x[i]*inverse*weight[i];
}
__device__ void rotate(float* data,int left,int right,float s,float c) {
  float a=data[left],b=data[right];data[left]=a*c-b*s;data[right]=a*s+b*c;
}
__global__ void rotate_device_position(float* q,float* k,const int* position,const float* sine,const float* cosine,
                                      int qw,int kw,int hs,bool split) {
  int pair=blockIdx.x*blockDim.x+threadIdx.x;if(pair>=qw/2)return;
  int within=pair%(hs/2),left=split?(pair/(hs/2))*hs+within:pair*2;
  int right=left+(split?hs/2:1),offset=*position*hs+within*2;
  float s=sine[offset],c=cosine[offset];rotate(q,left,right,s,c);if(pair<kw/2)rotate(k,left,right,s,c);
}
__global__ void store_device_position(const float* k,const float* v,runtime::KVCacheView cache,int layer,const int* position,int width) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<width){kv_key(cache,layer,*position)[i]=k[i];kv_value(cache,layer,*position)[i]=v[i];}
}
__global__ void prepare_qkv(float* qkv,const float* bias,const float* qnorm,const float* knorm,
                            runtime::KVCacheView cache,int layer,const int* position,const float* sine,const float* cosine,
                            int heads,int kv_heads,int hs,float epsilon,bool split) {
  int head=blockIdx.x,tid=threadIdx.x,qw=heads*hs,kw=kv_heads*hs;
  float* q=qkv+head*hs;float* k=qkv+qw+(head<kv_heads?head*hs:0);
  float qs=0,ks=0;
  for(int i=tid;i<hs;i+=blockDim.x){
    float v=q[i]+(bias?bias[head*hs+i]:0.f);q[i]=v;if(qnorm)qs+=v*v;
    if(head<kv_heads){v=k[i]+(bias?bias[qw+head*hs+i]:0.f);k[i]=v;if(knorm)ks+=v*v;}
  }
  using Reduce=cub::BlockReduce<float,128>;__shared__ Reduce::TempStorage storage;__shared__ float qscale,kscale;
  if(qnorm){
    qs=Reduce(storage).Sum(qs);if(tid==0)qscale=rsqrtf(qs/hs+epsilon);__syncthreads();
    ks=Reduce(storage).Sum(ks);if(tid==0)kscale=rsqrtf(ks/hs+epsilon);__syncthreads();
    for(int i=tid;i<hs;i+=blockDim.x){q[i]=q[i]*qscale*qnorm[i];if(head<kv_heads)k[i]=k[i]*kscale*knorm[i];}
  }
  __syncthreads();
  int pos=*position;float* keys=kv_key(cache,layer,pos);float* values=kv_value(cache,layer,pos);
  for(int pair=tid;pair<hs/2;pair+=blockDim.x){
    int left=split?pair:pair*2,right=left+(split?hs/2:1);
    float s=sine[pos*hs+pair*2],c=cosine[pos*hs+pair*2];
    rotate(q,left,right,s,c);
    if(head<kv_heads){float a=k[left],b=k[right];size_t offset=head*hs;keys[offset+left]=a*c-b*s;keys[offset+right]=a*s+b*c;}
  }
  if(head<kv_heads)for(int i=tid;i<hs;i+=blockDim.x){
    int index=qw+kw+head*hs+i;values[head*hs+i]=qkv[index]+(bias?bias[index]:0.f);
  }
}
__global__ void packed_swiglu(float* data,const float* bias,int width) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=width)return;
  float x=data[i]+(bias?bias[i]:0.f),y=data[width+i]+(bias?bias[width+i]:0.f);
  data[i]=(x/(1.f+expf(-x)))*y;
}
__global__ void attention_device_position(const float* q,runtime::KVCacheView cache,int layer,float* scores,float* out,
                                          const int* position,int host_position,int context,int heads,int kv_heads,int hs) {
  int head=blockIdx.x,tid=threadIdx.x,pos=position?*position:host_position,kv=kv_heads*hs,kh=(head/(heads/kv_heads))*hs;
  extern __shared__ float shared_q[];
  int padded=(hs+3)/4*4;
  for(int i=tid;i<padded;i+=blockDim.x)shared_q[i]=i<hs?q[head*hs+i]:0.f;
  __syncthreads();
  float* score=scores+head*context;float maximum=-FLT_MAX,scale=1.f/sqrtf(float(hs));
  for(int t=tid;t<=pos;t+=blockDim.x){const float* k=kv_key(cache,layer,t)+kh;float dot=0;for(int i=0;i<hs;++i)dot+=k[i]*shared_q[i];score[t]=dot*scale;maximum=fmaxf(maximum,score[t]);}
  using Reduce=cub::BlockReduce<float,256>;__shared__ Reduce::TempStorage storage;__shared__ float shared;
  maximum=Reduce(storage).Reduce(maximum,FloatMaximum{});if(tid==0)shared=maximum;__syncthreads();maximum=shared;__syncthreads();
  float sum=0;for(int t=tid;t<=pos;t+=blockDim.x){score[t]=expf(score[t]-maximum);sum+=score[t];}
  sum=Reduce(storage).Sum(sum);if(tid==0)shared=sum;__syncthreads();
  for(int t=tid;t<=pos;t+=blockDim.x)score[t]/=shared;__syncthreads();
  for(int i=tid;i<hs;i+=blockDim.x){float result=0;for(int t=0;t<=pos;++t)result+=score[t]*kv_value(cache,layer,t)[kh+i];out[head*hs+i]=result;}
}
__global__ void sample_and_advance(const float* input,int n,int* token,int* position) {
  using Pair=cub::KeyValuePair<int,float>;using Reduce=cub::BlockReduce<Pair,256>;
  __shared__ Reduce::TempStorage storage;Pair best(INT_MAX,-FLT_MAX);
  for(int i=threadIdx.x;i<n;i+=blockDim.x)best=cub::ArgMax()(best,Pair(i,input[i]));
  Pair result=Reduce(storage).Reduce(best,cub::ArgMax());
  if(threadIdx.x==0){*token=result.key;*position+=1;}
}
}
void decode_embedding(const int* token,const float* table,float* out,int width,cudaStream_t stream){gather_device_token<<<(width+255)/256,256,0,stream>>>(token,table,out,width);}
void decode_residual_norm(float* x,const float* y,const float* w,float* out,int width,float epsilon,cudaStream_t stream){residual_norm<<<1,128,0,stream>>>(x,y,w,out,width,epsilon);}
void decode_rope(float* q,float* k,const int* pos,const float* s,const float* c,int qw,int kw,int hs,bool split,cudaStream_t stream){rotate_device_position<<<(qw/2+127)/128,128,0,stream>>>(q,k,pos,s,c,qw,kw,hs,split);}
void decode_store_kv(const float* k,const float* v,float* keys,float* values,const int* pos,int width,cudaStream_t stream){store_device_position<<<(width+255)/256,256,0,stream>>>(k,v,runtime::KVCacheView{keys,values,nullptr,1,0,width,0},0,pos,width);}
void decode_prepare_qkv(float* qkv,const float* bias,const float* qn,const float* kn,float* keys,float* values,const int* pos,const float* s,const float* c,int heads,int kv_heads,int hs,float epsilon,bool split,cudaStream_t stream){prepare_qkv<<<heads,128,0,stream>>>(qkv,bias,qn,kn,runtime::KVCacheView{keys,values,nullptr,1,0,kv_heads*hs,0},0,pos,s,c,heads,kv_heads,hs,epsilon,split);}
void decode_packed_swiglu(float* data,const float* bias,int width,cudaStream_t stream){packed_swiglu<<<(width+255)/256,256,0,stream>>>(data,bias,width);}
void decode_attention(const float* q,const float* k,const float* v,float* scores,float* out,const int* pos,int context,int heads,int kv_heads,int hs,cudaStream_t stream){attention_device_position<<<heads,256,((hs+3)/4*4)*sizeof(float),stream>>>(q,runtime::KVCacheView{const_cast<float*>(k),const_cast<float*>(v),nullptr,1,context,kv_heads*hs,0},0,scores,out,pos,0,context,heads,kv_heads,hs);}
void decode_sample(const float* logits,int n,int* token,int* position,cudaStream_t stream){sample_and_advance<<<1,256,0,stream>>>(logits,n,token,position);}
void decode_store_cache(const float* k,const float* v,runtime::KVCacheView cache,int layer,const int* pos,cudaStream_t stream){store_device_position<<<(cache.kv_dim+255)/256,256,0,stream>>>(k,v,cache,layer,pos,cache.kv_dim);}
void decode_prepare_cache_qkv(float* qkv,const float* bias,const float* qn,const float* kn,runtime::KVCacheView cache,int layer,const int* pos,const float* s,const float* c,int heads,int kv_heads,int hs,float epsilon,bool split,cudaStream_t stream){prepare_qkv<<<heads,128,0,stream>>>(qkv,bias,qn,kn,cache,layer,pos,s,c,heads,kv_heads,hs,epsilon,split);}
void decode_cache_attention(const float* q,runtime::KVCacheView cache,int layer,float* scores,float* out,const int* pos,int host_pos,int heads,int kv_heads,int hs,cudaStream_t stream){attention_device_position<<<heads,256,((hs+3)/4*4)*sizeof(float),stream>>>(q,cache,layer,scores,out,pos,host_pos,cache.context,heads,kv_heads,hs);}
}  // namespace kernel
