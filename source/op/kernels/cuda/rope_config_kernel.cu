#include "rope_kernel.cuh"
namespace kernel {
__global__ void configured_rope_cache(float* sin_cache,float* cos_cache,int head_size,int seq,model::RopeConfig cfg) {
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=head_size)return;
  float frequency=1.f/powf(cfg.theta,float(i)/head_size);
  if(cfg.factor>1.f) {
    float wavelength=6.283185307179586f/frequency;
    float low=float(cfg.original_context)/cfg.low_freq_factor,high=float(cfg.original_context)/cfg.high_freq_factor;
    if(wavelength>low)frequency/=cfg.factor;
    else if(wavelength>=high) {
      float smooth=(float(cfg.original_context)/wavelength-cfg.low_freq_factor)/(cfg.high_freq_factor-cfg.low_freq_factor);
      frequency=(1.f-smooth)*frequency/cfg.factor+smooth*frequency;
    }
  }
  for(int pos=0;pos<seq;++pos) {sin_cache[pos*head_size+i]=sinf(pos*frequency);cos_cache[pos*head_size+i]=cosf(pos*frequency);}
}
void sin_cos_cache_calc_cu_config(int head_size,int seq,const tensor::Tensor& sin_cache,const tensor::Tensor& cos_cache,cudaStream_t stream,const model::RopeConfig& cfg) {
  configured_rope_cache<<<(head_size+127)/128,128,0,stream>>>(const_cast<float*>(sin_cache.ptr<float>()),const_cast<float*>(cos_cache.ptr<float>()),head_size,seq,cfg);
  CHECK_EQ(cudaGetLastError(),cudaSuccess);
}
}
