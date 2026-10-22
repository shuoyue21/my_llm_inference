#include "model/model.h"
#include "runtime/cuda_batch.h"
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <limits>
#include <memory>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#if defined(LLAMA3_SUPPORT) || defined(QWEN2_SUPPORT) || defined(QWEN3_SUPPORT)
#include <nlohmann/json.hpp>
#endif
namespace model {
Model::Model(base::TokenizerType tokenizer_type, base::ModelType model_type, std::string token_path,
             std::string model_path, bool is_quant_model)
    : tokenizer_type_(tokenizer_type),
      model_type_(model_type),
      token_path_(std::move(token_path)),
      model_path_(std::move(model_path)),
      is_quant_model_(is_quant_model) {}

std::unique_ptr<runtime::CudaBatchBackend> Model::create_batch_backend(size_t) const {
  throw std::runtime_error("This model does not provide a CUDA batch backend");
}

base::Status Model::prefill(const std::vector<int32_t>&,
    const std::function<void(int, const tensor::Tensor&)>&, int&) const {
  return base::error::FunctionNotImplement("This model does not provide CUDA batched prefill");
}

base::Status Model::begin_decode(int, int, bool, bool) const {
  return base::error::FunctionNotImplement("This model does not provide optimized CUDA decode");
}
base::Status Model::decode_next(int&) const {
  return base::error::FunctionNotImplement("This model does not provide optimized CUDA decode");
}
runtime::DecodeStats Model::decode_stats() const { return {}; }

base::Status Model::configure_kv_cache(const std::string& mode, int page_tokens) {
  if (device_type_ != base::DeviceType::kDeviceUnknown)
    return base::error::InvalidArgument("Configure KV cache before model init");
  if ((mode != "auto" && mode != "paged" && mode != "contiguous") || page_tokens <= 0)
    return base::error::InvalidArgument("Invalid KV cache mode/page size");
  requested_kv_mode_ = mode; kv_page_tokens_ = page_tokens;
  return base::error::Success();
}
base::Status Model::validate_kv_configuration() const {
  if (requested_kv_mode_ == "paged" && (device_type_ != base::DeviceType::kDeviceCUDA || is_quant_model_))
    return base::error::InvalidArgument("Paged KV requires CUDA FP32");
  return base::error::Success();
}
void Model::initialize_kv_cache(const std::shared_ptr<base::DeviceAllocator>& allocator,
                                std::shared_ptr<kernel::CudaConfig> cuda) {
  cache_config_ = std::move(cuda);
  const bool paged = requested_kv_mode_ != "contiguous" && device_type_ == base::DeviceType::kDeviceCUDA && !is_quant_model_;
  if (paged) {
    paged_cache_ = std::make_shared<runtime::PagedKVCache>(config_->layer_num_, config_->seq_len_,
        config_->kv_dim_, kv_page_tokens_, cache_config_);
  } else {
    tensor::Tensor keys(base::DataType::kDataTypeFp32, config_->layer_num_, config_->seq_len_, config_->kv_dim_, true, allocator);
    tensor::Tensor values(base::DataType::kDataTypeFp32, config_->layer_num_, config_->seq_len_, config_->kv_dim_, true, allocator);
    STATUS_CHECK(insert_buffer(ModelBufferType::kKeyCache, keys));
    STATUS_CHECK(insert_buffer(ModelBufferType::kValueCache, values));
  }
}
base::Status Model::begin_request() const {
  try {
    if (request_active_) { auto status = end_request(); if (!status) return status; }
    request_tokens_ = 0;
    if (paged_cache_) paged_cache_->begin_request();
    request_active_ = true;
    return base::error::Success();
  } catch (const std::exception& error) { return base::error::InternalError(error.what()); }
}
base::Status Model::end_request() const {
  try {
    if (request_active_ && paged_cache_) paged_cache_->release_request();
    request_active_ = false;
    return base::error::Success();
  } catch (const std::exception& error) { return base::error::InternalError(error.what()); }
}
base::Status Model::trim_kv_cache() const {
  try {
    if (paged_cache_) paged_cache_->trim();
    return base::error::Success();
  } catch (const std::exception& error) { return base::error::InternalError(error.what()); }
}
void Model::reserve_cache_tokens(int count) const {
  if (count < 0 || count > config_->seq_len_) throw std::out_of_range("KV token count exceeds context");
  if (!request_active_) { auto status = begin_request(); if (!status) throw std::runtime_error(status.get_err_msg()); }
  if (paged_cache_) paged_cache_->ensure_tokens(count);
  request_tokens_ = std::max(request_tokens_, count);
}
runtime::KVCacheView Model::kv_cache_view() const {
  if (paged_cache_) return paged_cache_->view();
  return {const_cast<float*>(get_buffer(ModelBufferType::kKeyCache).ptr<float>()),
          const_cast<float*>(get_buffer(ModelBufferType::kValueCache).ptr<float>()), nullptr,
          config_->layer_num_, config_->seq_len_, config_->kv_dim_, 0};
}
runtime::KVCacheStats Model::kv_cache_stats() const {
  if (paged_cache_) return paged_cache_->stats();
  runtime::KVCacheStats stats;
  stats.logical_tokens = request_tokens_;
  stats.reserved_bytes = get_buffer(ModelBufferType::kKeyCache).byte_size() + get_buffer(ModelBufferType::kValueCache).byte_size();
  stats.mapped_bytes = stats.peak_mapped_bytes = stats.peak_reserved_bytes = stats.reserved_bytes;
  stats.effective_bytes = size_t(request_tokens_) * config_->layer_num_ * config_->kv_dim_ * 2 * sizeof(float);
  return stats;
}

base::ModelType Model::model_type() const { return model_type_; }

const std::string& Model::token_path() const { return token_path_; }

const std::string& Model::model_path() const { return model_path_; }

base::Status Model::insert_buffer(ModelBufferType buffer_idx, const tensor::Tensor& tensor) {
  if (buffers_.count(buffer_idx) > 0) {
    return base::error::KeyHasExits(std::to_string(int(buffer_idx)) + " has exits in the buffers");
  }
  if (tensor.is_empty()) {
    return base::error::InvalidArgument("The tensor is empty for inserting buffer.");
  }
  buffers_.insert({buffer_idx, tensor});
  return base::error::Success();
}

tensor::Tensor& Model::get_buffer(ModelBufferType buffer_idx) {
  CHECK_GT(buffers_.count(buffer_idx), 0) << int(buffer_idx);
  return buffers_.at(buffer_idx);
}

const tensor::Tensor& Model::get_buffer(ModelBufferType buffer_idx) const {
  CHECK_GT(buffers_.count(buffer_idx), 0);
  return buffers_.at(buffer_idx);
}

base::Status Model::read_model_file() {
  using namespace base;
  std::unique_ptr<FILE, decltype(&fclose)> file(fopen(model_path_.c_str(), "rb"), fclose);
  if (!file) return error::PathNotValid("Cannot open checkpoint: " + model_path_);
  ModelConfig config{};
  if (fread(&config, sizeof(config), 1, file.get()) != 1)
    return error::ModelParseError("Checkpoint header is truncated");
  if (is_quant_model_ && (fread(&group_size_, sizeof(group_size_), 1, file.get()) != 1 || group_size_ <= 0))
    return error::ModelParseError("Invalid quantization group size");
  auto status = generate_model_infos(config);
  if (!status) return status;
  std::shared_ptr<RawModelData> raw;
  if (is_quant_model_) raw = std::make_shared<RawModelDataInt8>();
  else raw = std::make_shared<RawModelDataFp32>();
  raw->fd = dup(fileno(file.get()));
  struct stat info{};
  if (raw->fd < 0 || fstat(raw->fd, &info) != 0)
    return error::ModelParseError("Cannot stat checkpoint");
  const size_t header_bytes = sizeof(ModelConfig) + (is_quant_model_ ? sizeof(group_size_) : 0);
  if (info.st_size <= static_cast<off_t>(header_bytes))
    return error::ModelParseError("Checkpoint has no weights");
  raw->file_size = static_cast<size_t>(info.st_size);
#if !defined(QWEN3_SUPPORT)
  if (!is_quant_model_) {
    const uint64_t d = config_->dim_, h = config_->hidden_dim_, l = config_->layer_num_;
    const uint64_t v = config_->vocab_size_, kv = config_->kv_dim_;
    uint64_t count = v*d + 2*l*d + 2*l*d*d + 2*l*d*kv + 3*l*d*h + d +
        (config_->is_shared_weight_ ? 0 : uint64_t(config_->seq_len_)*config_->head_size_ + v*d);
#if defined(QWEN2_SUPPORT)
    count += l * (d + 2*kv);
#endif
    if (count > (raw->file_size - header_bytes) / sizeof(float))
      return error::ModelParseError("Checkpoint weight data is truncated or format is incompatible");
  }
#else
  {
    const uint64_t d=config_->hidden_dim_, a=config_->dim_, kv=config_->kv_dim_;
    const uint64_t l=config_->layer_num_, h=config_->immediate_dim_, v=config_->vocab_size_;
    const uint64_t count=(2*l+1)*d+2*v*d+2*l*d*a+2*l*d*kv+2*l*config_->head_size_+3*l*d*h;
    if(count>(raw->file_size-header_bytes)/sizeof(float))
      return error::ModelParseError("Qwen3 checkpoint is truncated");
  }
#endif
  raw->data = mmap(nullptr, raw->file_size, PROT_READ, MAP_PRIVATE, raw->fd, 0);
  if (raw->data == MAP_FAILED) return error::ModelParseError("Cannot mmap checkpoint");
  raw->weight_data = static_cast<int8_t*>(raw->data) + header_bytes;
  raw_model_data_ = std::move(raw);
  return error::Success();
}

base::Status Model::generate_model_infos(const ModelConfig& config) const {
  if (config.dim <= 0 || config.hidden_dim <= 0 || config.layer_num <= 0 ||
      config.head_num <= 0 || config.kv_head_num <= 0 || config.seq_len <= 0 ||
      config.vocab_size == 0 || config.vocab_size == std::numeric_limits<int32_t>::min() ||
      config.dim % config.head_num != 0 || config.head_num % config.kv_head_num != 0 ||
      (config.dim / config.head_num) % 2 != 0)
    return base::error::ModelParseError("Invalid model dimensions or attention head configuration");
  // Bound dimensions before multiplication and allocation of runtime tensors.
  if (config.dim > 65536 || config.hidden_dim > 1048576 || config.layer_num > 1024 ||
      std::abs(config.vocab_size) > 2000000 || config.seq_len > 1048576 ||
      int64_t(config.layer_num) * config.seq_len * config.dim > std::numeric_limits<int32_t>::max())
    return base::error::ModelParseError("Model dimensions exceed this runtime's supported limits");

  config_->dim_ = config.dim;
  config_->hidden_dim_ = config.hidden_dim;
  config_->layer_num_ = config.layer_num;
  config_->head_num_ = config.head_num;
  config_->kv_head_num_ = config.kv_head_num;
  config_->seq_len_ = config.seq_len;

  config_->kv_dim_ = (config.dim * config.kv_head_num) / config.head_num;
  config_->kv_mul_ = config.head_num / config.kv_head_num;
  config_->head_size_ = config.dim / config.head_num;
#if defined(QWEN3_SUPPORT)
  if (config.immediate_dim_ <= 0) return base::error::ModelParseError("Invalid intermediate size");
  config_->immediate_dim_ = config.immediate_dim_;
#endif
  if (config.vocab_size > 0) {
    config_->is_shared_weight_ = true;
  } else {
    config_->is_shared_weight_ = false;
  }

  // Qwen tokenizer size and embedding size is mismatched
  // refer: https://github.com/QwenLM/Qwen2.5/issues/29
  // if (std::abs(config.vocab_size) != config_->vocab_size_) {
  //   return base::error::ModelParseError(
  //       "Vocabulary size mismatch between the model file and the token list.");
  // }
  config_->vocab_size_ = std::abs(config.vocab_size);
#if defined(LLAMA3_SUPPORT)
  config_->rope.theta = 500000.f;
#elif defined(QWEN2_SUPPORT) || defined(QWEN3_SUPPORT)
  config_->rope.theta = 1000000.f;
#endif
#if defined(LLAMA3_SUPPORT) || defined(QWEN2_SUPPORT) || defined(QWEN3_SUPPORT)
  std::ifstream metadata(model_path_ + ".json");
  if (metadata) {
    try {
      auto value = nlohmann::json::parse(metadata);
      config_->rope.theta = value.value("rope_theta", config_->rope.theta);
      float expected_eps =
#if defined(LLAMA3_SUPPORT)
          1e-5f;
#else
          1e-6f;
#endif
      if (std::abs(value.value("rms_norm_eps", expected_eps) - expected_eps) > 1e-10f)
        return base::error::ModelParseError("RMSNorm epsilon does not match this model build");
      if (value.contains("rope_scaling") && !value["rope_scaling"].is_null()) {
        const auto& scaling = value["rope_scaling"];
        if (scaling.value("rope_type", scaling.value("type", std::string{})) != "llama3")
          return base::error::ModelParseError("Unsupported RoPE scaling type");
        config_->rope.factor = scaling.at("factor");
        config_->rope.low_freq_factor = scaling.at("low_freq_factor");
        config_->rope.high_freq_factor = scaling.at("high_freq_factor");
        config_->rope.original_context = scaling.at("original_max_position_embeddings");
      }
      if (config_->rope.theta <= 0 || config_->rope.factor < 1 ||
          config_->rope.low_freq_factor <= 0 || config_->rope.high_freq_factor <= config_->rope.low_freq_factor ||
          config_->rope.original_context <= 0)
        return base::error::ModelParseError("Invalid RoPE configuration");
    } catch (const std::exception& error) {
      return base::error::ModelParseError(std::string("Invalid runtime metadata: ") + error.what());
    }
  }
#endif

  return base::error::Success();
}

base::Status Model::create_encode_layer() {
  using namespace base;

  // create token encode decode layer
  if (tokenizer_type_ == TokenizerType::kEncodeSpe) {
    encode_layer_ = std::make_unique<op::SpeEncodeLayer>(this->token_path_, true, false);
  } else {
#ifdef LLAMA3_SUPPORT
    encode_layer_ = std::make_unique<op::BpeEncodeLayer>(this->token_path_, true, false);
#endif

#if defined(QWEN2_SUPPORT) || defined(QWEN3_SUPPORT)
    encode_layer_ = std::make_unique<op::QwenEncodeLayer>(this->token_path_, false, false);
#endif
  }
  if (!encode_layer_) {
    return error::InternalError("Create the encode layer failed.");
  }

  config_->vocab_size_ = encode_layer_->vocab_size();
  if (config_->vocab_size_ <= 0) {
    return error::InternalError("The vocab size param read error from the model file!");
  }
  return error::Success();
}

base::Status Model::gen_model_from_file() {
  using namespace base;
  config_ = std::make_unique<TransformerConfig>();

  // init sentence piece processor
  // google sentence piece
  auto create_encode_status = create_encode_layer();
  if (!create_encode_status) {
    LOG(ERROR) << "Create the encode layer failed!";
    return create_encode_status;
  }
  // mmap
  auto mmap_status = read_model_file();
  if (!mmap_status) {
    LOG(ERROR) << "Handle model file " << model_path_ << " failed!";
    return mmap_status;
  }
  auto layer_create_status = create_layers();
  if (!layer_create_status) {
    LOG(ERROR) << "Create layers for the model file " << model_path_ << " failed!";
    return layer_create_status;
  }

  return error::Success();
}

std::vector<int32_t> Model::encode(const std::string& sentence) const {
  CHECK(encode_layer_ != nullptr);
  return encode_layer_->encode(sentence);
}

bool Model::is_sentence_ending(int32_t token_idx) const {
  CHECK(this->encode_layer_ != nullptr);
  return this->encode_layer_->is_sentence_ending(token_idx);
}

std::string Model::decode(int32_t token_idx) const {
  CHECK(this->encode_layer_ != nullptr);
  return this->encode_layer_->decode(token_idx);
}

std::string Model::decode(std::vector<int32_t> token_idxs) const {
  CHECK(this->encode_layer_ != nullptr);
  return this->encode_layer_->decode(token_idxs);
}

std::pair<tensor::Tensor, tensor::Tensor> Model::slice_kv_cache(int32_t layer_idx,
                                                                int32_t token_pos) const {
  CHECK_GE(layer_idx, 0);
  CHECK_LT(layer_idx, config_->layer_num_);
  CHECK_GE(token_pos, 0);
  CHECK_LT(token_pos, config_->seq_len_);
  reserve_cache_tokens(token_pos + 1);
  if (paged_cache_) {
    auto [key_pointer, value_pointer] = paged_cache_->slot(layer_idx, token_pos);
    tensor::Tensor key(base::DataType::kDataTypeFp32, config_->kv_dim_, false, nullptr, key_pointer);
    tensor::Tensor value(base::DataType::kDataTypeFp32, config_->kv_dim_, false, nullptr, value_pointer);
    key.set_device_type(device_type_); value.set_device_type(device_type_);
    return {key, value};
  }
  int32_t layer_offset = layer_idx * config_->seq_len_ * config_->kv_dim_;
  int32_t cache_offset = layer_offset + token_pos * config_->kv_dim_;

  float* key_cache_ptr =
      const_cast<float*>(get_buffer(ModelBufferType::kKeyCache).ptr<float>(cache_offset));
  float* val_cache_ptr =
      const_cast<float*>(get_buffer(ModelBufferType::kValueCache).ptr<float>(cache_offset));

  tensor::Tensor key(base::DataType::kDataTypeFp32, config_->kv_dim_, false, nullptr,
                     key_cache_ptr);
  tensor::Tensor val(base::DataType::kDataTypeFp32, config_->kv_dim_, false, nullptr,
                     val_cache_ptr);
  key.set_device_type(device_type_);
  val.set_device_type(device_type_);
  return {key, val};
}

tensor::Tensor Model::fill_input(const tensor::Tensor& pos_tensor,
                                 const op::EmbeddingOutput& embedding_output,
                                 bool is_prompt) const {
  const int32_t pos = pos_tensor.index<int32_t>(0);
  auto [input_tokens, input_embeddings, input_token_num] = embedding_output;

  int32_t index = 0;
  if (is_prompt) {
    index = pos;
  }
#if defined(QWEN3_SUPPORT)
  std::shared_ptr<base::Buffer> input_emb_buffer = std::make_shared<base::Buffer>(
      config_->hidden_dim_ * sizeof(float), nullptr,
      input_embeddings.ptr<float>(index * config_->hidden_dim_), true);
  tensor::Tensor input(base::DataType::kDataTypeFp32, config_->hidden_dim_);

#else
  std::shared_ptr<base::Buffer> input_emb_buffer =
      std::make_shared<base::Buffer>(config_->dim_ * sizeof(float), nullptr,
                                     input_embeddings.ptr<float>(index * config_->dim_), true);
  tensor::Tensor input(base::DataType::kDataTypeFp32, config_->dim_);
#endif
  input.assign(input_emb_buffer);
  input.set_device_type(device_type_);
  return input;
}

}  // namespace model