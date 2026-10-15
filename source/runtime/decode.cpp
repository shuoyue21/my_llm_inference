#include "runtime/decode.h"
#include <chrono>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <utility>
#include "../op/kernels/cuda/decode_kernels.cuh"
#include "../op/kernels/cuda/prefill_kernels.cuh"

namespace runtime {
namespace {
using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
void check(cudaError_t status) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string("CUDA decode: ") + cudaGetErrorString(status));
}
void check_blas(cublasStatus_t status) {
  if (status != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error("CUDA decode cuBLAS status " + std::to_string(int(status)));
}
struct Buffer {
  void* data = nullptr;
  size_t bytes = 0;
  Buffer() = default;
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  ~Buffer() {
    if (data) cudaFree(data);
  }
  void allocate(size_t size) {
    if (data) throw std::logic_error("Decode buffers must keep stable addresses");
    check(cudaMalloc(&data, size));
    bytes = size;
  }
  float* floats() { return static_cast<float*>(data); }
  const float* floats() const { return static_cast<const float*>(data); }
};
struct Packed {
  Buffer weight, bias;
  int rows = 0, columns = 0;
  const float* bias_ptr() const { return static_cast<const float*>(bias.data); }
};
struct PackedBlock {
  Packed qkv, gate_up;
};
struct Captured {
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
  size_t nodes = 0, kernels = 0;
};
}  // namespace
struct CudaDecode::Impl {
  PrefillWeights w;
  std::shared_ptr<kernel::CudaConfig> config;
  tensor::Tensor logits, sine, cosine;
  KVCacheView cache;
  cublasHandle_t blas = nullptr;
  Buffer cursor, x, z, y, qkv, attention, gate_up, scores, blas_workspace;
  int* host = nullptr;
  std::vector<std::unique_ptr<PackedBlock>> packed;
  Captured graphs[2];
  bool use_graph = false, use_fusion = false, first_begin = true, packing_complete = false;
  int host_position = 0;
  double construction_ms = 0, total_setup = 0, total_capture = 0, total_warmup = 0;
  uint64_t captures = 0, packed_bytes = 0;
  DecodeStats current;
  Impl(PrefillWeights weights, std::shared_ptr<kernel::CudaConfig> cuda, tensor::Tensor output,
       KVCacheView cache_view, tensor::Tensor sin_cache, tensor::Tensor cos_cache)
      : w(std::move(weights)),
        config(std::move(cuda)),
        logits(std::move(output)),
        cache(cache_view),
        sine(std::move(sin_cache)),
        cosine(std::move(cos_cache)) {
    auto start = Clock::now();
    if (!config || w.embedding.device_type() != base::DeviceType::kDeviceCUDA)
      throw std::invalid_argument("Optimized decode requires CUDA FP32 weights");
    try {
      cursor.allocate(2 * sizeof(int));
      x.allocate(w.hidden * sizeof(float));
      z.allocate(w.hidden * sizeof(float));
      y.allocate(w.hidden * sizeof(float));
      qkv.allocate((w.attention + 2 * w.kv_heads * w.head_size) * sizeof(float));
      attention.allocate(w.attention * sizeof(float));
      gate_up.allocate(2 * w.intermediate * sizeof(float));
      scores.allocate(size_t(w.heads) * w.context * sizeof(float));
      blas_workspace.allocate(32 * 1024 * 1024);
      check(cudaMallocHost(reinterpret_cast<void**>(&host), 3 * sizeof(int)));
      check_blas(cublasCreate(&blas));
      check_blas(cublasSetStream(blas, config->stream));
      check_blas(cublasSetMathMode(blas, CUBLAS_PEDANTIC_MATH));
      check_blas(cublasSetAtomicsMode(blas, CUBLAS_ATOMICS_NOT_ALLOWED));
      check_blas(cublasSetWorkspace(blas, blas_workspace.data, blas_workspace.bytes));
      packed.resize(w.blocks.size());
      construction_ms = elapsed(start);
    } catch (...) {
      if (blas) cublasDestroy(blas);
      if (host) cudaFreeHost(host);
      throw;
    }
  }
  ~Impl() {
    if (config) cudaStreamSynchronize(config->stream);
    for (auto& entry : graphs) {
      if (entry.executable) cudaGraphExecDestroy(entry.executable);
      if (entry.graph) cudaGraphDestroy(entry.graph);
    }
    if (blas) cublasDestroy(blas);
    if (host) cudaFreeHost(host);
  }
  int* token() { return static_cast<int*>(cursor.data); }
  int* position() { return static_cast<int*>(cursor.data) + 1; }
  uint64_t workspace_bytes() const {
    return cursor.bytes + x.bytes + z.bytes + y.bytes + qkv.bytes + attention.bytes +
           gate_up.bytes + scores.bytes + blas_workspace.bytes;
  }
  void pack(Packed& out, std::initializer_list<const Projection*> inputs) {
    out.columns = (*inputs.begin())->weight.get_dim(1);
    bool bias = false;
    for (auto* p : inputs) {
      if (p->weight.get_dim(1) != out.columns)
        throw std::logic_error("Packed projection widths differ");
      out.rows += p->weight.get_dim(0);
      bias |= !p->bias.is_empty();
    }
    out.weight.allocate(size_t(out.rows) * out.columns * sizeof(float));
    if (bias) {
      out.bias.allocate(out.rows * sizeof(float));
      check(cudaMemsetAsync(out.bias.data, 0, out.bias.bytes, config->stream));
    }
    size_t row = 0;
    for (auto* p : inputs) {
      size_t rows = p->weight.get_dim(0);
      check(cudaMemcpyAsync(out.weight.floats() + row * out.columns, p->weight.ptr<float>(),
                            rows * out.columns * sizeof(float), cudaMemcpyDeviceToDevice,
                            config->stream));
      if (!p->bias.is_empty())
        check(cudaMemcpyAsync(out.bias.floats() + row, p->bias.ptr<float>(), rows * sizeof(float),
                              cudaMemcpyDeviceToDevice, config->stream));
      row += rows;
    }
    packed_bytes += out.weight.bytes + out.bias.bytes;
  }
  void ensure_packed() {
    if (packing_complete) return;
    for (auto& p : packed) p.reset();
    packed_bytes = 0;
    for (size_t i = 0; i < w.blocks.size(); ++i) {
      auto block = std::make_unique<PackedBlock>();
      const auto& b = w.blocks[i];
      pack(block->qkv, {&b.query, &b.key, &b.value});
      pack(block->gate_up, {&b.gate, &b.up});
      packed[i] = std::move(block);
    }
    packing_complete = true;
  }
  void gemv(const float* input, const float* weight, float* output, int n, int k) {
    const float alpha = 1.f, beta = 0.f;
    check_blas(cublasSgemv(blas, CUBLAS_OP_T, k, n, &alpha, weight, k, input, 1, &beta, output, 1));
  }
  uint64_t linear(const float* input, const Projection& projection, float* output) {
    gemv(input, projection.weight.ptr<float>(), output, projection.weight.get_dim(0),
         projection.weight.get_dim(1));
    if (!projection.bias.is_empty()) {
      kernel::prefill_bias(output, projection.bias.ptr<float>(), 1, projection.weight.get_dim(0),
                           config->stream);
      return 2;
    }
    return 1;
  }
  void packed_linear(const float* input, const Packed& projection, float* output) {
    gemv(input, projection.weight.floats(), output, projection.rows, projection.columns);
  }
  uint64_t body() {
    auto stream = config->stream;
    const int kv = w.kv_heads * w.head_size;
    float* hidden = x.floats();
    float* normalized = z.floats();
    float* output = y.floats();
    float* q = qkv.floats();
    float* k = q + w.attention;
    float* v = k + kv;
    float* attn = attention.floats();
    float* gate = gate_up.floats();
    float* up = gate + w.intermediate;
    uint64_t submitted = 1;
    kernel::decode_embedding(token(), w.embedding.ptr<float>(), hidden, w.hidden, stream);
    for (size_t layer = 0; layer < w.blocks.size(); ++layer) {
      const auto& b = w.blocks[layer];
      if (!use_fusion || layer == 0) {
        kernel::prefill_norm(hidden, b.attention_norm.ptr<float>(), normalized, 1, w.hidden,
                             w.epsilon, stream);
        ++submitted;
      }
      if (use_fusion) {
        const auto& p = *packed[layer];
        packed_linear(normalized, p.qkv, q);
        ++submitted;
        kernel::decode_prepare_cache_qkv(q, p.qkv.bias_ptr(), b.query_norm.ptr<float>(),
                                   b.key_norm.ptr<float>(), cache, static_cast<int>(layer), position(),
                                   sine.ptr<float>(), cosine.ptr<float>(), w.heads, w.kv_heads,
                                   w.head_size, w.epsilon, w.split_half_rope, stream);
        ++submitted;
      } else {
        submitted += linear(normalized, b.query, q) + linear(normalized, b.key, k) +
                     linear(normalized, b.value, v);
        if (!b.query_norm.is_empty()) {
          kernel::prefill_norm(q, b.query_norm.ptr<float>(), q, w.heads, w.head_size, w.epsilon,
                               stream);
          kernel::prefill_norm(k, b.key_norm.ptr<float>(), k, w.kv_heads, w.head_size, w.epsilon,
                               stream);
          submitted += 2;
        }
        kernel::decode_rope(q, k, position(), sine.ptr<float>(), cosine.ptr<float>(), w.attention,
                            kv, w.head_size, w.split_half_rope, stream);
        kernel::decode_store_cache(k, v, cache, static_cast<int>(layer), position(), stream);
        submitted += 2;
      }
      kernel::decode_cache_attention(q, cache, static_cast<int>(layer), scores.floats(), attn,
                                     position(), 0, w.heads, w.kv_heads, w.head_size, stream);
      ++submitted;
      submitted += linear(attn, b.output, output);
      if (use_fusion) {
        kernel::decode_residual_norm(hidden, output, b.ffn_norm.ptr<float>(), normalized, w.hidden,
                                     w.epsilon, stream);
        ++submitted;
      } else {
        kernel::prefill_add(hidden, output, w.hidden, stream);
        kernel::prefill_norm(hidden, b.ffn_norm.ptr<float>(), normalized, 1, w.hidden, w.epsilon,
                             stream);
        submitted += 2;
      }
      if (use_fusion) {
        const auto& p = packed[layer]->gate_up;
        packed_linear(normalized, p, gate);
        ++submitted;
        kernel::decode_packed_swiglu(gate, p.bias_ptr(), w.intermediate, stream);
        ++submitted;
      } else {
        submitted += linear(normalized, b.gate, gate) + linear(normalized, b.up, up);
        kernel::prefill_swiglu(gate, up, w.intermediate, stream);
        ++submitted;
      }
      submitted += linear(gate, b.down, output);
      if (use_fusion) {
        const auto& norm =
            layer + 1 < w.blocks.size() ? w.blocks[layer + 1].attention_norm : w.final_norm;
        kernel::decode_residual_norm(hidden, output, norm.ptr<float>(), normalized, w.hidden,
                                     w.epsilon, stream);
        ++submitted;
      } else {
        kernel::prefill_add(hidden, output, w.hidden, stream);
        ++submitted;
      }
    }
    if (!use_fusion) {
      kernel::prefill_norm(hidden, w.final_norm.ptr<float>(), normalized, 1, w.hidden, w.epsilon,
                           stream);
      ++submitted;
    }
    submitted += linear(normalized, w.classifier, logits.ptr<float>());
    kernel::decode_sample(logits.ptr<float>(), w.vocab, token(), position(), stream);
    ++submitted;
    check(cudaGetLastError());
    return submitted;
  }
  void prepare_graph() {
    auto& entry = graphs[use_fusion ? 1 : 0];
    if (entry.executable) return;
    auto warm = Clock::now();
    current.preparation_submissions += body();
    check(cudaStreamSynchronize(config->stream));
    current.warmup_ms = elapsed(warm);
    total_warmup += current.warmup_ms;
    // The dry run only writes the next cache row. Restore the cursor so replay
    // overwrites that row with exactly the same input before it can be read.
    check(cudaMemcpyAsync(cursor.data, host, 2 * sizeof(int), cudaMemcpyHostToDevice,
                          config->stream));
    ++current.preparation_submissions;
    check(cudaStreamSynchronize(config->stream));
    auto capture = Clock::now();
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    check(cudaStreamBeginCapture(config->stream, cudaStreamCaptureModeThreadLocal));
    try {
      current.capture_api_calls = body();
      check(cudaStreamEndCapture(config->stream, &graph));
      check(cudaGraphInstantiateWithFlags(&executable, graph, 0));
    } catch (...) {
      if (!graph) cudaStreamEndCapture(config->stream, &graph);
      if (executable) cudaGraphExecDestroy(executable);
      if (graph) cudaGraphDestroy(graph);
      throw;
    }
    current.capture_ms = elapsed(capture);
    total_capture += current.capture_ms;
    ++captures;
    entry.graph = graph;
    entry.executable = executable;
    check(cudaGraphGetNodes(graph, nullptr, &entry.nodes));
    std::vector<cudaGraphNode_t> nodes(entry.nodes);
    check(cudaGraphGetNodes(graph, nodes.data(), &entry.nodes));
    for (auto node : nodes) {
      cudaGraphNodeType type;
      check(cudaGraphNodeGetType(node, &type));
      if (type == cudaGraphNodeTypeKernel) ++entry.kernels;
    }
  }
  void begin(int input, int start, bool graph, bool fusion) {
    if (input < 0 || input >= w.vocab || start < 0 || start >= w.context)
      throw std::invalid_argument("Invalid decode token/position");
    auto setup = Clock::now();
    current = {};
    use_graph = graph;
    use_fusion = fusion;
    host_position = start;
    if (use_fusion) ensure_packed();
    host[0] = input;
    host[1] = start;
    check(cudaMemcpyAsync(cursor.data, host, 2 * sizeof(int), cudaMemcpyHostToDevice,
                          config->stream));
    check(cudaStreamSynchronize(config->stream));
    current.host_submissions = 1;
    if (use_graph) prepare_graph();
    current.setup_ms = elapsed(setup) + (first_begin ? construction_ms : 0);
    first_begin = false;
    total_setup += current.setup_ms;
  }
  int step() {
    if (host_position >= w.context) throw std::invalid_argument("Decode exceeds model context");
    if (use_graph) {
      check(cudaGraphLaunch(graphs[use_fusion ? 1 : 0].executable, config->stream));
      ++current.host_submissions;
      ++current.graph_launches;
    } else
      current.host_submissions += body();
    check(cudaMemcpyAsync(host + 2, token(), sizeof(int), cudaMemcpyDeviceToHost, config->stream));
    ++current.host_submissions;
    check(cudaStreamSynchronize(config->stream));
    ++host_position;
    return host[2];
  }
  DecodeStats stats() const {
    DecodeStats result = current;
    result.setup_total_ms = total_setup;
    result.capture_total_ms = total_capture;
    result.warmup_total_ms = total_warmup;
    result.captures_total = captures;
    result.packed_weight_bytes = packed_bytes;
    result.workspace_bytes = workspace_bytes();
    if (use_graph) {
      const auto& graph = graphs[use_fusion ? 1 : 0];
      result.graph_nodes = graph.nodes;
      result.graph_kernel_nodes = graph.kernels;
    }
    return result;
  }
};
CudaDecode::CudaDecode(PrefillWeights weights, std::shared_ptr<kernel::CudaConfig> config,
                       tensor::Tensor logits, KVCacheView cache,
                       tensor::Tensor sine, tensor::Tensor cosine)
    : impl_(std::make_unique<Impl>(std::move(weights), std::move(config), std::move(logits),
                                   cache, std::move(sine), std::move(cosine))) {}
CudaDecode::~CudaDecode() = default;
void CudaDecode::begin(int token, int position, bool graph, bool fusion) {
  impl_->begin(token, position, graph, fusion);
}
int CudaDecode::step() { return impl_->step(); }
int CudaDecode::next_position() const { return impl_->host_position; }
DecodeStats CudaDecode::stats() const { return impl_->stats(); }
}  // namespace runtime
