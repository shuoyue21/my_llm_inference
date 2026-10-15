#include "runtime/generation.h"
#include <chrono>
namespace runtime {
base::Status generate(const model::Model& model, const std::string& prompt,
                      const GenerationOptions& options, GenerationResult& result) {
  using Clock = std::chrono::steady_clock;
  auto start = Clock::now();
  result = {};
  if (options.max_new_tokens < 0)
    return base::error::InvalidArgument("max-new-tokens must be nonnegative");
  if (options.prefill_mode != "auto" && options.prefill_mode != "batched" && options.prefill_mode != "reference")
    return base::error::InvalidArgument("prefill must be auto, batched or reference");
  const bool can_batch = model.device_type() == base::DeviceType::kDeviceCUDA && !model.is_quantized();
  if (options.prefill_mode == "batched" && !can_batch)
    return base::error::InvalidArgument("Batched prefill requires CUDA FP32");
  const bool batched = can_batch && options.prefill_mode != "reference";
  result.prefill_mode = options.max_new_tokens == 0 ? "none" : batched ? "batched" : "reference";
  result.prompt_tokens = options.prompt_token_ids.empty() ? model.encode(prompt) : options.prompt_token_ids;
  auto prefill_start = Clock::now();
  result.tokenization_ms = std::chrono::duration<double,std::milli>(prefill_start-start).count();
  if (result.prompt_tokens.empty())
    return base::error::InvalidArgument("Prompt produced no tokens");
  if (result.prompt_tokens.size() > static_cast<size_t>(model.context_length()) ||
      int64_t(result.prompt_tokens.size()) + options.max_new_tokens - 1 > model.context_length())
    return base::error::InvalidArgument("Prompt plus generation exceeds model context length");
  for (int token : result.prompt_tokens)
    if (token < 0 || token >= model.vocab_size())
      return base::error::InvalidArgument("Tokenizer produced an out-of-vocabulary token");
  result.stop_reason = "length";
  auto first_token_time = start;
  const int prompt_size = static_cast<int>(result.prompt_tokens.size());
  auto pos_tensor = model.get_buffer(model::ModelBufferType::kInputPos);
  int next = -1;
  auto previous_token_time = start;
  bool stopped = false;
  auto emit = [&] {
    result.generated_tokens.push_back(next);
    auto token_time = Clock::now();
    if (result.generated_tokens.size() > 1)
      result.decode_token_ms.push_back(std::chrono::duration<double,std::milli>(token_time-previous_token_time).count());
    previous_token_time = token_time;
    if (result.generated_tokens.size() == 1) {
      first_token_time = token_time;
      result.prefill_ms = std::chrono::duration<double,std::milli>(token_time-prefill_start).count();
      result.ttft_ms = std::chrono::duration<double,std::milli>(token_time-start).count();
    }
    if (!options.ignore_eos && model.is_sentence_ending(next)) {result.stop_reason = "eos"; stopped = true;}
  };
  int begin = 0;
  if (batched && options.max_new_tokens > 0) {
    pos_tensor.index<int32_t>(0) = prompt_size - 1;
    auto status = model.prefill(result.prompt_tokens, options.on_logits, next);
    if (!status) return status;
    if (next < 0 || next >= model.vocab_size()) return base::error::InternalError("Sampler returned an invalid token");
    emit(); begin = prompt_size;
  }
  // Reference mode retains the original per-token prefill. Decode is shared.
  for (int pos = begin; !stopped && options.max_new_tokens > 0 &&
                       pos < prompt_size + options.max_new_tokens - 1; ++pos) {
    const int token = pos < prompt_size ? result.prompt_tokens[pos] : next;
    pos_tensor.index<int32_t>(0) = pos;
    auto embedding = model.embedding({token});
    auto input = model.fill_input(pos_tensor, embedding, false);
    auto status = model.predict(input, pos_tensor, pos < prompt_size - 1, next);
    if (!status) return status;
    if (options.on_logits)
      options.on_logits(pos, model.get_buffer(model::ModelBufferType::kForwardOutput));
    if (pos < prompt_size - 1) continue;
    if (next < 0 || next >= model.vocab_size()) return base::error::InternalError("Sampler returned an invalid token");
    emit();
  }
  auto end = Clock::now();
  result.total_ms = std::chrono::duration<double,std::milli>(end-start).count();
  if (result.generated_tokens.size() > 1) {
    result.decode_ms = std::chrono::duration<double,std::milli>(end-first_token_time).count();
    result.decode_tokens_per_second = (result.generated_tokens.size()-1)*1000.0/result.decode_ms;
  }
  auto visible_tokens = result.generated_tokens;
  if (result.stop_reason == "eos" && !visible_tokens.empty()) visible_tokens.pop_back();
  result.text = model.decode(visible_tokens);
  return base::error::Success();
}
}  // namespace runtime
