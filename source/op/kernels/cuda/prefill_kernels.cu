#include "prefill_kernels.cuh"
#include "reduce_compat.cuh"
#include <cub/block/block_reduce.cuh>
#include <cfloat>
#include <cmath>
namespace kernel {
namespace {
__global__ void gather(const int* tokens, const float* table, float* output, int rows, int width) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < rows * width) output[i] = table[size_t(tokens[i / width]) * width + i % width];
}
__global__ void norm_rows(const float* input, const float* weight, float* output, int width, float epsilon) {
  int row = blockIdx.x, tid = threadIdx.x;
  const float* x = input + size_t(row) * width;
  float sum = 0;
  for (int i = tid; i < width; i += blockDim.x) sum += x[i] * x[i];
  using Reduce = cub::BlockReduce<float, 128>;
  __shared__ Reduce::TempStorage storage;
  __shared__ float inverse;
  sum = Reduce(storage).Sum(sum);
  if (tid == 0) inverse = rsqrtf(sum / width + epsilon);
  __syncthreads();
  for (int i = tid; i < width; i += blockDim.x) output[size_t(row) * width + i] = x[i] * inverse * weight[i];
}
__global__ void bias_rows(float* output, const float* bias, size_t count, int width) {
  size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count) output[i] += bias[i % width];
}
__global__ void add_values(float* residual, const float* update, size_t count) {
  size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count) residual[i] += update[i];
}
__global__ void swiglu_values(float* gate, const float* up, size_t count) {
  size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count) {float x = gate[i]; gate[i] = (x / (1.f + expf(-x))) * up[i];}
}
__device__ void rotate_pair(float* x, int left, int right, float sine, float cosine) {
  float a = x[left], b = x[right];
  x[left] = a * cosine - b * sine; x[right] = a * sine + b * cosine;
}
__global__ void rotate_rows(float* query, float* key, const float* sine, const float* cosine,
                            int rows, int q_width, int k_width, int head_size, bool split_half) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int pairs = q_width / 2;
  if (index >= rows * pairs) return;
  int row = index / pairs, pair = index % pairs;
  int within = pair % (head_size / 2);
  int left = split_half ? (pair / (head_size / 2)) * head_size + within : pair * 2;
  int right = left + (split_half ? head_size / 2 : 1);
  float s = sine[row * head_size + within * 2], c = cosine[row * head_size + within * 2];
  rotate_pair(query + size_t(row) * q_width, left, right, s, c);
  if (pair < k_width / 2) rotate_pair(key + size_t(row) * k_width, left, right, s, c);
}
__global__ void causal_attention(const float* query, const float* key, const float* value,
                                 float* scores, float* output, int rows, int heads, int kv_heads, int head_size) {
  int head = blockIdx.x, row = blockIdx.y, tid = threadIdx.x;
  int kv_width = kv_heads * head_size, head_offset = (head / (heads / kv_heads)) * head_size;
  const float* q = query + (size_t(row) * heads + head) * head_size;
  float* score = scores + (size_t(row) * heads + head) * rows;
  extern __shared__ float shared_query[];
  // Keep the shared tail aligned and initialized for vectorized shared loads.
  const int padded_head = (head_size + 3) / 4 * 4;
  for (int i = tid; i < padded_head; i += blockDim.x) shared_query[i] = i < head_size ? q[i] : 0.f;
  __syncthreads();
  float maximum = -FLT_MAX;
  float scale = 1.f / sqrtf(float(head_size));
  for (int t = tid; t <= row; t += blockDim.x) {
    float dot = 0;
    for (int i = 0; i < head_size; ++i) dot += shared_query[i] * key[size_t(t) * kv_width + head_offset + i];
    score[t] = dot * scale; maximum = fmaxf(maximum, score[t]);
  }
  using Reduce = cub::BlockReduce<float, 256>;
  __shared__ Reduce::TempStorage storage;
  __shared__ float shared;
  maximum = Reduce(storage).Reduce(maximum, FloatMaximum{});
  if (tid == 0) shared = maximum;
  __syncthreads(); maximum = shared;
  __syncthreads();
  float sum = 0;
  for (int t = tid; t <= row; t += blockDim.x) {score[t] = expf(score[t] - maximum); sum += score[t];}
  sum = Reduce(storage).Sum(sum);
  if (tid == 0) shared = sum;
  __syncthreads();
  for (int t = tid; t <= row; t += blockDim.x) score[t] /= shared;
  __syncthreads();
  for (int i = tid; i < head_size; i += blockDim.x) {
    float total = 0;
    for (int t = 0; t <= row; ++t) total += score[t] * value[size_t(t) * kv_width + head_offset + i];
    output[(size_t(row) * heads + head) * head_size + i] = total;
  }
}
}  // namespace
void prefill_embedding(const int* ids,const float* table,float* out,int rows,int width,cudaStream_t stream) {gather<<<(rows*width+255)/256,256,0,stream>>>(ids,table,out,rows,width);}
void prefill_norm(const float* x,const float* w,float* y,int rows,int width,float epsilon,cudaStream_t stream) {norm_rows<<<rows,128,0,stream>>>(x,w,y,width,epsilon);}
void prefill_bias(float* x,const float* bias,int rows,int width,cudaStream_t stream) {bias_rows<<<(size_t(rows)*width+255)/256,256,0,stream>>>(x,bias,size_t(rows)*width,width);}
void prefill_add(float* x,const float* y,size_t count,cudaStream_t stream) {add_values<<<(count+255)/256,256,0,stream>>>(x,y,count);}
void prefill_swiglu(float* x,const float* y,size_t count,cudaStream_t stream) {swiglu_values<<<(count+255)/256,256,0,stream>>>(x,y,count);}
void prefill_rope(float* q,float* k,const float* s,const float* c,int rows,int qw,int kw,int hs,bool split,cudaStream_t stream) {rotate_rows<<<(rows*(qw/2)+255)/256,256,0,stream>>>(q,k,s,c,rows,qw,kw,hs,split);}
void prefill_attention(const float* q,const float* k,const float* v,float* scores,float* y,int rows,int heads,int kv_heads,int hs,cudaStream_t stream) {causal_attention<<<dim3(heads,rows),256,((hs+3)/4*4)*sizeof(float),stream>>>(q,k,v,scores,y,rows,heads,kv_heads,hs);}
}  // namespace kernel
