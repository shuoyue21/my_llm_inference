#ifndef KUIPER_INCLUDE_MODEL_MODEL_H_
#define KUIPER_INCLUDE_MODEL_MODEL_H_
#include <op/embedding.h>
#include <map>
#include <functional>
#include <string>
#include "config.h"
#include "op/encode.h"
#include "op/layer.h"
#include "raw_model_data.h"
#include "sampler/argmax_sampler.h"
#include "sentencepiece_processor.h"
#include "tensor/tensor.h"
#include "runtime/decode.h"

namespace runtime { class CudaPrefill; class CudaBatchBackend; }
namespace model {
class Model {
 public:
  virtual ~Model() = default;
  // The model must outlive its serving backend (tokenizer EOS predicate).
  virtual std::unique_ptr<runtime::CudaBatchBackend> create_batch_backend(size_t max_rows) const;
  int32_t context_length() const { return config_->seq_len_; }
  int32_t vocab_size() const { return config_->vocab_size_; }
  base::DeviceType device_type() const { return device_type_; }
  bool is_quantized() const { return is_quant_model_; }
  base::Status configure_kv_cache(const std::string& mode, int page_tokens = 16);
  base::Status begin_request() const;
  base::Status end_request() const;
  base::Status trim_kv_cache() const;
  runtime::KVCacheView kv_cache_view() const;
  runtime::KVCacheStats kv_cache_stats() const;
  virtual base::Status prefill(const std::vector<int32_t>& tokens,
      const std::function<void(int, const tensor::Tensor&)>& on_logits, int& next) const;
  virtual base::Status begin_decode(int token, int position, bool graph, bool fusion) const;
  virtual base::Status decode_next(int& next) const;
  virtual runtime::DecodeStats decode_stats() const;
  explicit Model(base::TokenizerType tokenizer_type, base::ModelType model_type,
                 std::string token_path, std::string model_path, bool is_quant_model);

  virtual base::Status init(base::DeviceType device_type) = 0;

  virtual base::Status predict(const tensor::Tensor& input, const tensor::Tensor& pos_tensor,
                               bool is_prompt, int& next) const = 0;

  virtual base::Status forward(const tensor::Tensor& input, const tensor::Tensor& pos_tensor,
                               int& next) const = 0;

  base::ModelType model_type() const;

  const std::string& token_path() const;

  const std::string& model_path() const;

  virtual tensor::Tensor& get_buffer(ModelBufferType buffer_idx);

  virtual const tensor::Tensor& get_buffer(ModelBufferType buffer_idx) const;

  virtual bool is_sentence_ending(int32_t token_idx) const;

  virtual std::string decode(int32_t token_idx) const;

  virtual std::string decode(std::vector<int32_t> token_idxs) const;

  /////////////////////////////////////////////////////
  /////////////////////////////////////////////////////
  virtual std::vector<int32_t> encode(const std::string& sentence) const;

  virtual std::pair<tensor::Tensor, tensor::Tensor> slice_kv_cache(int32_t layer_idx,
                                                                   int32_t token_pos) const;

  virtual op::EmbeddingOutput embedding(const std::vector<int>& tokens) const = 0;

  virtual tensor::Tensor fill_input(const tensor::Tensor& pos_tensor,
                                    const op::EmbeddingOutput& embedding_output,
                                    bool is_prompt) const;

 protected:
  virtual base::Status insert_buffer(ModelBufferType buffer_idx, const tensor::Tensor& tensor);

  virtual base::Status read_model_file();

  virtual base::Status create_encode_layer();

  virtual base::Status gen_model_from_file();

  virtual base::Status generate_model_infos(const ModelConfig& config) const;

  virtual int32_t post_processing(const tensor::Tensor& pos, bool is_prompt) const = 0;

 private:
  virtual void init_mem() = 0;

  virtual base::Status create_layers() = 0;

  virtual void create_param_layers() = 0;

  virtual void create_nonparam_layers() = 0;

  virtual void create_param_quant_layers() = 0;

 protected:
  base::Status validate_kv_configuration() const;
  void initialize_kv_cache(const std::shared_ptr<base::DeviceAllocator>& allocator,
                           std::shared_ptr<kernel::CudaConfig> config);
  void reserve_cache_tokens(int count) const;
  std::string requested_kv_mode_ = "auto";
  int kv_page_tokens_ = 16;
  mutable int request_tokens_ = 0;
  mutable bool request_active_ = false;
  std::shared_ptr<kernel::CudaConfig> cache_config_;
  mutable std::shared_ptr<runtime::PagedKVCache> paged_cache_;
  int32_t group_size_ = 1;
  bool is_quant_model_ = false;
  std::unique_ptr<TransformerConfig> config_;

  std::string token_path_;
  std::string model_path_;
  std::unique_ptr<op::EncodeLayerBase> encode_layer_;
  std::map<ModelBufferType, tensor::Tensor> buffers_;
  std::unique_ptr<sampler::Sampler> sampler_;
  std::shared_ptr<RawModelData> raw_model_data_;
  base::DeviceType device_type_ = base::DeviceType::kDeviceUnknown;
  base::ModelType model_type_ = base::ModelType::kModelTypeUnknown;
  base::TokenizerType tokenizer_type_ = base::TokenizerType::kEncodeUnknown;
};
}  // namespace model
#endif  // KUIPER_INCLUDE_MODEL_MODEL_H_
