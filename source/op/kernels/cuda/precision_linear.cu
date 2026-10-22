#include "runtime/precision_linear.h"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cub/block/block_reduce.cuh>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>
namespace runtime {
namespace {
void check(cudaError_t code) {
  if (code != cudaSuccess) throw std::runtime_error(cudaGetErrorString(code));
}
template<class T> __global__ void convert_activation(const float* source, T* target, size_t count) {
  size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if (i<count) target[i]=T(source[i]);
}
__global__ void gather_packed(const int* tokens, const void* weight, const float* scales,
    float* output, int rows, int k, int groups, int bits, int type) {
  size_t index=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(index>=size_t(rows)*k)return;
  int row=index/k,col=index%k,token=tokens[row];size_t source=size_t(token)*k+col;
  float value;
  if(bits){
    const auto* packed=static_cast<const unsigned char*>(weight);int q;
    if(bits==8)q=static_cast<signed char>(packed[source]);
    else {int x=(packed[source/2]>>((source%2)*4))&15;q=x>=8?x-16:x;}
    value=float(q)*scales[size_t(token)*groups+col/128];
  } else if(type==int(CUDA_R_32F))value=static_cast<const float*>(weight)[source];
  else if(type==int(CUDA_R_16F))value=__half2float(static_cast<const __half*>(weight)[source]);
  else value=__bfloat162float(static_cast<const __nv_bfloat16*>(weight)[source]);
  output[index]=value;
}
__global__ void unpack_half(const unsigned char* weight, const float* scales, __half* output,
    size_t count, int k, int groups, int bits) {
  size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i>=count)return;
  int q;
  if(bits==8)q=static_cast<signed char>(weight[i]);
  else {int x=(weight[i/2]>>((i%2)*4))&15;q=x>=8?x-16:x;}
  output[i]=__float2half_rn(float(q)*scales[(i/k)*groups+(i%k)/128]);
}
__global__ void quantized_product(const __half* input, const unsigned char* weight,
    const float* scales, float* output, int n, int k, int bits, int groups) {
  int row=blockIdx.y, out=blockIdx.x, tid=threadIdx.x;
  float sum=0;
  for (int col=tid;col<k;col+=blockDim.x) {
    size_t index=size_t(out)*k+col;
    int q;
    if(bits==8) q=static_cast<signed char>(weight[index]);
    else { int nibble=(weight[index/2]>>((index%2)*4))&15; q=nibble>=8?nibble-16:nibble; }
    sum+=__half2float(input[size_t(row)*k+col])*float(q)*scales[size_t(out)*groups+col/128];
  }
  using Reduce=cub::BlockReduce<float,256>;
  __shared__ Reduce::TempStorage storage;
  sum=Reduce(storage).Sum(sum);
  if(tid==0) output[size_t(row)*n+out]=sum;
}
}
struct PrecisionLinear::Impl {
  void* weight=nullptr; float* scales=nullptr;
  int n=0,k=0,bits=0,groups=0;
  cudaDataType_t type=CUDA_R_16F;
  uint64_t storage_bytes=0;
  ~Impl() { if(weight)cudaFree(weight);if(scales)cudaFree(scales); }
};
PrecisionLinear::PrecisionLinear(const tensor::Tensor& source,const std::string& precision,cudaStream_t stream)
    :impl_(std::make_unique<Impl>()) {
  auto& s=*impl_;s.n=source.get_dim(0);s.k=source.get_dim(1);
  if(precision!="fp32" && precision!="fp16" && precision!="bf16" && precision!="w8a16" && precision!="w4a16")
    throw std::invalid_argument("Unsupported linear precision");
  const size_t count=size_t(s.n)*s.k;
  std::vector<float> host(count);
  if(source.device_type()==base::DeviceType::kDeviceCUDA)
    check(cudaMemcpy(host.data(),source.ptr<float>(),count*sizeof(float),cudaMemcpyDeviceToHost));
  else std::copy_n(source.ptr<float>(),count,host.data());
  if(precision=="fp32") {
    s.type=CUDA_R_32F;s.storage_bytes=count*sizeof(float);
    check(cudaMalloc(&s.weight,s.storage_bytes));
    check(cudaMemcpyAsync(s.weight,host.data(),s.storage_bytes,cudaMemcpyHostToDevice,stream));
  } else if(precision=="fp16" || precision=="bf16") {
    s.type=precision=="fp16"?CUDA_R_16F:CUDA_R_16BF;s.storage_bytes=count*2;
    std::vector<unsigned short> packed(count);
    for(size_t i=0;i<count;++i) packed[i]=precision=="fp16"?__half_as_ushort(__float2half_rn(host[i])):__bfloat16_as_ushort(__float2bfloat16_rn(host[i]));
    check(cudaMalloc(&s.weight,s.storage_bytes));
    check(cudaMemcpyAsync(s.weight,packed.data(),s.storage_bytes,cudaMemcpyHostToDevice,stream));
    check(cudaStreamSynchronize(stream));
  } else {
    s.bits=precision=="w8a16"?8:4;s.groups=(s.k+127)/128;
    int maximum=s.bits==8?127:7;
    std::vector<float> scales(size_t(s.n)*s.groups);
    std::vector<unsigned char> packed((count*s.bits+7)/8,0);
    for(int row=0;row<s.n;++row) for(int group=0;group<s.groups;++group) {
      int begin=group*128,end=std::min(begin+128,s.k);float largest=0;
      for(int col=begin;col<end;++col)largest=std::max(largest,std::abs(host[size_t(row)*s.k+col]));
      float scale=largest>0?largest/maximum:1.f;scales[size_t(row)*s.groups+group]=scale;
      for(int col=begin;col<end;++col){
        size_t index=size_t(row)*s.k+col;
        int q=std::clamp(int(std::round(host[index]/scale)),-maximum,maximum);
        if(s.bits==8)packed[index]=static_cast<unsigned char>(static_cast<signed char>(q));
        else packed[index/2]|=static_cast<unsigned char>((q&15)<<((index%2)*4));
      }
    }
    check(cudaMalloc(&s.weight,packed.size()));
    check(cudaMalloc(reinterpret_cast<void**>(&s.scales),scales.size()*sizeof(float)));
    check(cudaMemcpyAsync(s.weight,packed.data(),packed.size(),cudaMemcpyHostToDevice,stream));
    check(cudaMemcpyAsync(s.scales,scales.data(),scales.size()*sizeof(float),cudaMemcpyHostToDevice,stream));
    check(cudaStreamSynchronize(stream));s.storage_bytes=packed.size()+scales.size()*sizeof(float);
  }
  check(cudaStreamSynchronize(stream));
}
PrecisionLinear::~PrecisionLinear()=default;
void PrecisionLinear::embedding(const int* tokens,float* output,int rows,cudaStream_t stream)const {
 const auto& s=*impl_;
 gather_packed<<<(size_t(rows)*s.k+255)/256,256,0,stream>>>(tokens,s.weight,s.scales,output,rows,s.k,s.groups,s.bits,int(s.type));
}
uint64_t PrecisionLinear::bytes()const{return impl_->storage_bytes;}
void PrecisionLinear::run(const float* input,float* output,int rows,void* activation_workspace,void* weight_workspace,
    cublasHandle_t handle,cudaStream_t stream)const {
  const auto& s=*impl_;size_t count=size_t(rows)*s.k;
  if(s.bits && rows<=8){
    convert_activation<<<(count+255)/256,256,0,stream>>>(input,static_cast<__half*>(activation_workspace),count);
    quantized_product<<<dim3(s.n,rows),256,0,stream>>>(static_cast<__half*>(activation_workspace),
        static_cast<const unsigned char*>(s.weight),s.scales,output,s.n,s.k,s.bits,s.groups);
    return;
  }
  const void* activation=input;
  const void* weight=s.weight;
  auto operand_type=s.bits?CUDA_R_16F:s.type;
  if(s.bits){
    unpack_half<<<(size_t(s.n)*s.k+255)/256,256,0,stream>>>(static_cast<const unsigned char*>(s.weight),
        s.scales,static_cast<__half*>(weight_workspace),size_t(s.n)*s.k,s.k,s.groups,s.bits);
    weight=weight_workspace;
  }
  if(operand_type==CUDA_R_16F){
    convert_activation<<<(count+255)/256,256,0,stream>>>(input,static_cast<__half*>(activation_workspace),count);
    activation=activation_workspace;
  } else if(operand_type==CUDA_R_16BF){
    convert_activation<<<(count+255)/256,256,0,stream>>>(input,static_cast<__nv_bfloat16*>(activation_workspace),count);
    activation=activation_workspace;
  }
  float alpha=1,beta=0;
  auto status=cublasGemmEx(handle,CUBLAS_OP_T,CUBLAS_OP_N,s.n,rows,s.k,&alpha,weight,operand_type,s.k,
      activation,operand_type,s.k,&beta,output,CUDA_R_32F,s.n,operand_type==CUDA_R_32F?CUBLAS_COMPUTE_32F_PEDANTIC:CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT);
  if(status!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("Low precision GEMM failed: "+std::to_string(int(status)));
}
}
