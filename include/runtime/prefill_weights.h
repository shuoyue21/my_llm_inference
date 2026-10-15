#pragma once
#include "runtime/prefill.h"
#include "model/config.h"
#include "op/matmul.h"

namespace runtime {
inline tensor::Tensor layer_weight(const std::shared_ptr<op::Layer>& layer) {
  auto parameter = std::dynamic_pointer_cast<op::LayerParam>(layer);
  CHECK(parameter != nullptr);
  return parameter->get_weight(0);
}
inline Projection projection(const std::shared_ptr<op::Layer>& layer) {
  auto linear = std::dynamic_pointer_cast<op::MatmulLayer>(layer);
  CHECK(linear != nullptr);
  return {linear->get_weight(0), linear->has_bias() ? linear->get_bias(0) : tensor::Tensor{}};
}
template <class Layers>
PrefillWeights collect_prefill_weights(const Layers& layers, const model::TransformerConfig& config) {
  PrefillWeights result;
  result.embedding = layer_weight(layers.embedding_layer_);
  result.hidden = result.embedding.get_dim(1);
  result.attention = layer_weight(layers.wq_layers_.at(0)).get_dim(0);
  result.intermediate = layer_weight(layers.w1_layers_.at(0)).get_dim(0);
  result.heads = config.head_num_; result.kv_heads = config.kv_head_num_;
  result.head_size = config.head_size_; result.context = config.seq_len_;
  result.vocab = config.vocab_size_;
#if defined(QWEN2_SUPPORT) || defined(QWEN3_SUPPORT)
  result.epsilon = 1e-6f;
#endif
#if defined(LLAMA3_SUPPORT) || defined(QWEN2_SUPPORT) || defined(QWEN3_SUPPORT)
  result.split_half_rope = true;
#endif
  const int count = config.layer_num_;
  const bool head_norm = layers.rmsnorm_layers_.size() == static_cast<size_t>(4 * count + 1);
  result.final_norm = layer_weight(layers.rmsnorm_layers_.at(2 * count));
  result.classifier = projection(layers.cls_layer_);
  for (int i = 0; i < count; ++i) {
    PrefillBlock block;
    block.attention_norm = layer_weight(layers.rmsnorm_layers_.at(i));
    block.ffn_norm = layer_weight(layers.rmsnorm_layers_.at(i + count));
    if (head_norm) {
      block.query_norm = layer_weight(layers.rmsnorm_layers_.at(i + 2 * count + 1));
      block.key_norm = layer_weight(layers.rmsnorm_layers_.at(i + 3 * count + 1));
    }
    block.query = projection(layers.wq_layers_.at(i));
    block.key = projection(layers.wk_layers_.at(i));
    block.value = projection(layers.wv_layers_.at(i));
    block.output = projection(layers.wo_layers_.at(i));
    block.gate = projection(layers.w1_layers_.at(i));
    block.up = projection(layers.w3_layers_.at(i));
    block.down = projection(layers.w2_layers_.at(i));
    result.blocks.push_back(std::move(block));
  }
  return result;
}
}  // namespace runtime
