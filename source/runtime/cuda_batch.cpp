#include "runtime/cuda_batch.h"
#include "runtime/precision_linear.h"
#include <cublas_v2.h>
#include <stdexcept>
#include <unordered_map>
#include <algorithm>
#include "../op/kernels/cuda/prefill_kernels.cuh"
#include "../op/kernels/cuda/decode_kernels.cuh"
#include "../op/kernels/cuda/online_attention.cuh"
namespace runtime {
namespace {
void check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
void blas_check(cublasStatus_t status) {
  if (status != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error("Batch cuBLAS: " + std::to_string(int(status)));
}
struct Buffer {
  void* data = nullptr;
  ~Buffer() { if (data) cudaFree(data); }
  void allocate(size_t bytes) { check(cudaMalloc(&data, bytes)); }
  float* f(size_t offset = 0) { return static_cast<float*>(data) + offset; }
  int* i(size_t offset = 0) { return static_cast<int*>(data) + offset; }
};
}
struct CudaBatchBackend::Impl {
  PrefillWeights w;
  std::shared_ptr<kernel::CudaConfig> config;
  tensor::Tensor sine, cosine;
  size_t max_rows;
  int page_tokens;
  bool online_attention = false;
  Buffer cache_views;
  std::function<bool(int)> is_eos;
  cublasHandle_t blas = nullptr;
  Buffer ids, positions, x, z, q, k, v, attention, y, gate, up, scores, logits, sampled, activation, dequantization, output_indices;
  std::unordered_map<const float*, std::unique_ptr<PrecisionLinear>> linears;
  uint64_t weight_bytes = 0;
  std::function<void(RequestId, int, const std::vector<float>&)> on_logits;
  struct Request {
    std::unique_ptr<PagedKVCache> cache;
    int position = 0;
    std::vector<int32_t> prompt;
  };
  struct PrefixEntry {
    std::vector<int32_t> tokens;
    std::shared_ptr<const KVPrefixSnapshot> pages;
    uint64_t last_used;
  };
  std::vector<PrefixEntry> prefixes;
  uint64_t prefix_limit = 0, clock = 0;
  PrefixCacheStats prefix_stats;
  void publish(Request& request) {
    if (!prefix_limit) return;
    int count = (int(request.prompt.size()) - 1) / page_tokens * page_tokens;
    if (!count) return;
    std::vector<int32_t> tokens(request.prompt.begin(), request.prompt.begin() + count);
    for (auto& entry : prefixes) if (entry.tokens == tokens) { entry.last_used = ++clock; return; }
    auto pages = request.cache->capture_prefix(count);
    if (pages->bytes() > prefix_limit) return;
    while (prefix_stats.resident_bytes + pages->bytes() > prefix_limit) {
      auto victim = std::min_element(prefixes.begin(), prefixes.end(),
          [](const auto& a, const auto& b) { return a.last_used < b.last_used; });
      prefix_stats.resident_bytes -= victim->pages->bytes();
      prefixes.erase(victim); ++prefix_stats.evictions;
    }
    prefix_stats.resident_bytes += pages->bytes();
    prefixes.push_back({std::move(tokens), std::move(pages), ++clock});
  };

  std::unordered_map<RequestId, Request> requests;
  std::vector<std::unique_ptr<PagedKVCache>> idle_caches;
  Impl(PrefillWeights weights, std::shared_ptr<kernel::CudaConfig> cuda,
       tensor::Tensor sin, tensor::Tensor cos, size_t rows, int pages,
       std::function<bool(int)> eos, const std::string& precision)
      : w(std::move(weights)), config(std::move(cuda)), sine(std::move(sin)),
        cosine(std::move(cos)), max_rows(rows), page_tokens(pages), is_eos(std::move(eos)) {
    if (!max_rows || max_rows > 1024 || !config || page_tokens <= 0 || page_tokens > w.context)
      throw std::invalid_argument("Invalid batch backend dimensions");
    const bool host_source = w.embedding.device_type() == base::DeviceType::kDeviceCPU;
    auto upload = [&](tensor::Tensor& tensor) {
      if (!tensor.is_empty() && tensor.device_type() == base::DeviceType::kDeviceCPU) {
        tensor::Tensor uploaded(base::DataType::kDataTypeFp32, tensor.dims(), true,
            base::CUDADeviceAllocatorFactory::get_instance());
        check(cudaMemcpyAsync(uploaded.ptr<float>(), tensor.ptr<float>(), tensor.byte_size(),
            cudaMemcpyHostToDevice, config->stream));
        tensor = std::move(uploaded);
      }
    };
    size_t largest_matrix = 0;
    auto prepare = [&](Projection& projection) {
      largest_matrix = std::max(largest_matrix, projection.weight.size());
      if (host_source || precision != "fp32") {
        auto key = projection.weight.ptr<float>();
        if (!linears.count(key)) {
          auto packed = std::make_unique<PrecisionLinear>(projection.weight, precision, config->stream);
          weight_bytes += packed->bytes(); linears.emplace(key, std::move(packed));
        }
      } else weight_bytes += projection.weight.byte_size();
      upload(projection.bias);
    };
    if (host_source || precision != "fp32") {
      auto packed = std::make_unique<PrecisionLinear>(w.embedding, precision, config->stream);
      weight_bytes += packed->bytes(); linears.emplace(w.embedding.ptr<float>(), std::move(packed));
    }
    upload(w.final_norm); upload(sine); upload(cosine);
    prepare(w.classifier);
    for (auto& block : w.blocks) {
      upload(block.attention_norm); upload(block.ffn_norm); upload(block.query_norm); upload(block.key_norm);
      for (auto p : {&block.query, &block.key, &block.value, &block.output, &block.gate, &block.up, &block.down}) prepare(*p);
    }
    if (precision == "w8a16" || precision == "w4a16") dequantization.allocate(largest_matrix * 2);
    activation.allocate(rows * size_t(std::max({w.hidden, w.intermediate, w.attention})) * 2);
    const size_t kv = w.kv_heads * w.head_size;
    cache_views.allocate(rows * sizeof(KVCacheView));
    ids.allocate(rows * sizeof(int)); positions.allocate(rows * sizeof(int));
    sampled.allocate(rows * sizeof(int)); output_indices.allocate(rows * sizeof(int));
    x.allocate(rows * w.hidden * sizeof(float)); z.allocate(rows * w.hidden * sizeof(float));
    q.allocate(rows * w.attention * sizeof(float));
    k.allocate(rows * kv * sizeof(float)); v.allocate(rows * kv * sizeof(float));
    attention.allocate(rows * w.attention * sizeof(float)); y.allocate(rows * w.hidden * sizeof(float));
    gate.allocate(rows * w.intermediate * sizeof(float)); up.allocate(rows * w.intermediate * sizeof(float));
    // The reference score workspace is allocated only when selected.
    logits.allocate(rows * w.vocab * sizeof(float));
    blas_check(cublasCreate(&blas));
    try {
      blas_check(cublasSetStream(blas, config->stream));
      blas_check(cublasSetMathMode(blas, CUBLAS_PEDANTIC_MATH));
    } catch (...) { cublasDestroy(blas); blas = nullptr; throw; }
  }
  ~Impl() {
    if (config) cudaStreamSynchronize(config->stream);
    if (blas) cublasDestroy(blas);
  }
  void linear(const float* input, const Projection& p, float* output, int rows) {
    int n = p.weight.get_dim(0), width = p.weight.get_dim(1);
    float alpha = 1.f, beta = 0.f;
    auto packed = linears.find(p.weight.ptr<float>());
    if (packed != linears.end()) packed->second->run(input, output, rows, activation.data, dequantization.data, blas, config->stream);
    else blas_check(cublasSgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, n, rows, width,
        &alpha, p.weight.ptr<float>(), width, input, width, &beta, output, n));
    if (!p.bias.is_empty()) kernel::prefill_bias(output, p.bias.ptr<float>(), rows, n, config->stream);
  }
};
CudaBatchBackend::CudaBatchBackend(PrefillWeights w, std::shared_ptr<kernel::CudaConfig> c,
    tensor::Tensor sin, tensor::Tensor cos, size_t rows, int pages, std::function<bool(int)> eos, const std::string& precision)
    : impl_(std::make_unique<Impl>(std::move(w), std::move(c), std::move(sin),
                                  std::move(cos), rows, pages, std::move(eos), precision)) {}
CudaBatchBackend::~CudaBatchBackend() = default;
int CudaBatchBackend::context_length() const { return impl_->w.context; }
int CudaBatchBackend::vocab_size() const { return impl_->w.vocab; }
void CudaBatchBackend::admit(RequestId id) {
  auto& s = *impl_;
  if (s.requests.count(id)) throw std::invalid_argument("Request already admitted");
  std::unique_ptr<PagedKVCache> cache;
  if (s.idle_caches.empty()) cache = std::make_unique<PagedKVCache>(s.w.blocks.size(),
      s.w.context, s.w.kv_heads * s.w.head_size, s.page_tokens, s.config);
  else { cache = std::move(s.idle_caches.back()); s.idle_caches.pop_back(); }
  cache->begin_request();
  s.requests.emplace(id, Impl::Request{std::move(cache), 0});
}
void CudaBatchBackend::release(RequestId id) {
  auto& s = *impl_;
  auto found = s.requests.find(id);
  if (found == s.requests.end()) return;
  found->second.cache->release_request();
  s.idle_caches.push_back(std::move(found->second.cache));
  s.requests.erase(found);
}
std::vector<TokenOutput> CudaBatchBackend::execute(const std::vector<ScheduledToken>& batch) {
  auto& s = *impl_; const auto& w = s.w;
  if (batch.empty()) return {};
  if (batch.size() > s.max_rows) throw std::invalid_argument("Batch exceeds workspace capacity");
  std::unordered_map<RequestId, int> expected;
  std::vector<int> ids, positions;
  for (const auto& row : batch) {
    if (!s.requests.count(row.id)) throw std::invalid_argument("Unknown request");
    auto [position, inserted] = expected.emplace(row.id, s.requests.at(row.id).position);
    if (row.token < 0 || row.token >= w.vocab || row.position != position->second++ || row.position >= w.context)
      throw std::invalid_argument("Invalid batch request/token/position");
    ids.push_back(row.token); positions.push_back(row.position);
  }
  for (const auto& row : batch) s.requests.at(row.id).cache->ensure_tokens(row.position + 1);
  if (!s.online_attention && !s.scores.data) s.scores.allocate(s.max_rows * s.w.heads * s.w.context * sizeof(float));
  std::vector<KVCacheView> views;
  for (const auto& row : batch) views.push_back(s.requests.at(row.id).cache->view());
  check(cudaMemcpyAsync(s.cache_views.data, views.data(), views.size() * sizeof(KVCacheView), cudaMemcpyHostToDevice, s.config->stream));
  const int rows = batch.size(), kv = w.kv_heads * w.head_size;
  auto stream = s.config->stream;
  check(cudaMemcpyAsync(s.ids.data, ids.data(), rows * sizeof(int), cudaMemcpyHostToDevice, stream));
  check(cudaMemcpyAsync(s.positions.data, positions.data(), rows * sizeof(int), cudaMemcpyHostToDevice, stream));
  auto embedding = s.linears.find(w.embedding.ptr<float>());
  if (embedding != s.linears.end()) embedding->second->embedding(s.ids.i(), s.x.f(), rows, stream);
  else kernel::prefill_embedding(s.ids.i(), w.embedding.ptr<float>(), s.x.f(), rows, w.hidden, stream);
  for (size_t layer = 0; layer < w.blocks.size(); ++layer) {
    const auto& b = w.blocks[layer];
    kernel::prefill_norm(s.x.f(), b.attention_norm.ptr<float>(), s.z.f(), rows, w.hidden, w.epsilon, stream);
    s.linear(s.z.f(), b.query, s.q.f(), rows);
    s.linear(s.z.f(), b.key, s.k.f(), rows);
    s.linear(s.z.f(), b.value, s.v.f(), rows);
    if (!b.query_norm.is_empty()) {
      kernel::prefill_norm(s.q.f(), b.query_norm.ptr<float>(), s.q.f(), rows * w.heads, w.head_size, w.epsilon, stream);
      kernel::prefill_norm(s.k.f(), b.key_norm.ptr<float>(), s.k.f(), rows * w.kv_heads, w.head_size, w.epsilon, stream);
    }
    auto* views = static_cast<const KVCacheView*>(s.cache_views.data);
    kernel::batch_prepare_cache(s.q.f(), s.k.f(), s.v.f(), views, s.positions.i(), layer,
        s.sine.ptr<float>(), s.cosine.ptr<float>(), rows, w.attention, kv, w.head_size, w.split_half_rope, stream);
    if (s.online_attention) kernel::batch_online_attention(s.q.f(), views, s.positions.i(), layer,
        s.attention.f(), rows, w.heads, w.kv_heads, w.head_size, stream);
    else kernel::batch_reference_attention(s.q.f(), views, s.positions.i(), layer,
        s.scores.f(), s.attention.f(), rows, w.heads, w.kv_heads, w.head_size, stream);
    s.linear(s.attention.f(), b.output, s.y.f(), rows);
    kernel::prefill_add(s.x.f(), s.y.f(), size_t(rows) * w.hidden, stream);
    kernel::prefill_norm(s.x.f(), b.ffn_norm.ptr<float>(), s.z.f(), rows, w.hidden, w.epsilon, stream);
    s.linear(s.z.f(), b.gate, s.gate.f(), rows); s.linear(s.z.f(), b.up, s.up.f(), rows);
    kernel::prefill_swiglu(s.gate.f(), s.up.f(), size_t(rows) * w.intermediate, stream);
    s.linear(s.gate.f(), b.down, s.y.f(), rows);
    kernel::prefill_add(s.x.f(), s.y.f(), size_t(rows) * w.hidden, stream);
  }
  // Intermediate prompt rows only populate KV. Project/sample the rows
  // actually emitted to callers; diagnostics explicitly request every row.
  std::vector<int> output_rows;
  for (int r = 0; r < rows; ++r) {
    const auto& request = s.requests.at(batch[r].id);
    if (s.on_logits || !batch[r].prompt || request.prompt.empty() ||
        batch[r].position + 1 == int(request.prompt.size())) output_rows.push_back(r);
  }
  const int outputs = output_rows.size();
  std::vector<int> tokens(outputs);
  if (outputs) {
    const float* final_hidden = s.x.f();
    if (outputs != rows) {
      check(cudaMemcpyAsync(s.output_indices.data, output_rows.data(), outputs * sizeof(int), cudaMemcpyHostToDevice, stream));
      kernel::prefill_embedding(s.output_indices.i(), s.x.f(), s.y.f(), outputs, w.hidden, stream);
      final_hidden = s.y.f();
    }
    kernel::prefill_norm(final_hidden, w.final_norm.ptr<float>(), s.z.f(), outputs, w.hidden, w.epsilon, stream);
    s.linear(s.z.f(), w.classifier, s.logits.f(), outputs);
    for (int j = 0; j < outputs; ++j)
      kernel::decode_sample(s.logits.f(size_t(j) * w.vocab), w.vocab, s.sampled.i(j), s.positions.i(output_rows[j]), stream);
    check(cudaMemcpyAsync(tokens.data(), s.sampled.data, outputs * sizeof(int), cudaMemcpyDeviceToHost, stream));
  }
  check(cudaGetLastError()); check(cudaStreamSynchronize(stream));
  if (s.on_logits) {
    std::vector<float> values(w.vocab);
    for (int j = 0; j < outputs; ++j) {
      const int r = output_rows[j];
      check(cudaMemcpy(values.data(), s.logits.f(size_t(j) * w.vocab), w.vocab * sizeof(float), cudaMemcpyDeviceToHost));
      s.on_logits(batch[r].id, batch[r].position, values);
    }
  }
  std::vector<TokenOutput> result(rows, {0, false});
  for (int j = 0; j < outputs; ++j) result[output_rows[j]] = {tokens[j], s.is_eos(tokens[j])};
  for (int r = 0; r < rows; ++r) {
    auto& request = s.requests.at(batch[r].id);
    ++request.position;
    if (batch[r].prompt && request.position == int(request.prompt.size())) s.publish(request);
  }
  return result;
}
void CudaBatchBackend::configure_prefix_cache(uint64_t bytes) {
  if (!impl_->requests.empty()) throw std::logic_error("Configure prefix cache before admitting requests");
  impl_->prefixes.clear(); impl_->prefix_stats = {}; impl_->prefix_limit = bytes;
}
int CudaBatchBackend::reuse_prefix(RequestId id, const std::vector<int32_t>& tokens) {
  auto& s = *impl_; auto& request = s.requests.at(id); request.prompt = tokens;
  if (!s.prefix_limit) return 0;
  ++s.prefix_stats.lookups;
  Impl::PrefixEntry* best = nullptr;
  int best_tokens = 0;
  for (auto& entry : s.prefixes) {
    int matched = 0;
    int bound = std::min(entry.tokens.size(), tokens.size() - 1);
    while (matched < bound && entry.tokens[matched] == tokens[matched]) ++matched;
    matched = matched / s.page_tokens * s.page_tokens;
    if (matched > best_tokens) { best_tokens = matched; best = &entry; }
  }
  if (!best) return 0;
  request.cache->attach_prefix(*best->pages, best_tokens);
  request.position = best_tokens; best->last_used = ++s.clock;
  ++s.prefix_stats.hits; s.prefix_stats.reused_tokens += best_tokens;
  return request.position;
}
PrefixCacheStats CudaBatchBackend::prefix_stats() const { return impl_->prefix_stats; }
void CudaBatchBackend::configure_attention(const std::string& mode) {
  if (mode != "reference" && mode != "online") throw std::invalid_argument("Attention must be reference or online");
  if (!impl_->requests.empty()) throw std::logic_error("Configure attention before admitting requests");
  if (mode == "online" && impl_->w.head_size > 256) throw std::invalid_argument("Online attention supports head sizes up to 256");
  if (impl_->scores.data) { check(cudaFree(impl_->scores.data)); impl_->scores.data = nullptr; }
  impl_->online_attention = mode == "online";
}
uint64_t CudaBatchBackend::attention_workspace_bytes() const {
  return impl_->online_attention ? 0 : impl_->max_rows * impl_->w.heads * impl_->w.context * sizeof(float);
}
uint64_t CudaBatchBackend::linear_weight_bytes() const { return impl_->weight_bytes; }
void CudaBatchBackend::set_logits_callback(std::function<void(RequestId, int, const std::vector<float>&)> callback) {
  impl_->on_logits = std::move(callback);
}
KVCacheStats CudaBatchBackend::cache_stats() const {
  KVCacheStats total; total.paged = true;
  auto add = [&](const auto& cache) { auto v = cache->stats();
    total.reserved_bytes += v.reserved_bytes; total.mapped_bytes += v.mapped_bytes;
    total.effective_bytes += v.effective_bytes; total.table_bytes += v.table_bytes;
  };
  for (const auto& [id, request] : impl_->requests) add(request.cache);
  for (const auto& cache : impl_->idle_caches) add(cache);
  return total;
}
}
