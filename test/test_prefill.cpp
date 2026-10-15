#include <gtest/gtest.h>
#include <cmath>
#include <cstdlib>
#include <limits>
#include "runtime/generation.h"
#include "model/llama3.h"
#if defined(QWEN2_SUPPORT)
#include "model/qwen2.h"
#elif defined(QWEN3_SUPPORT)
#include "model/qwen3.h"
#endif
#include "source/op/kernels/cuda/prefill_kernels.cuh"

namespace {
bool cuda_available() {int count = 0; return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;}
}
TEST(runtime_cuda, batched_attention_gqa_masks_poisoned_future) {
  if (!cuda_available()) GTEST_SKIP();
  constexpr int rows = 7, heads = 6, kv_heads = 2, hs = 6, kv = kv_heads * hs;
  auto alloc = base::CPUDeviceAllocatorFactory::get_instance();
  tensor::Tensor q(base::DataType::kDataTypeFp32, rows * heads * hs, true, alloc);
  tensor::Tensor k(base::DataType::kDataTypeFp32, rows * kv, true, alloc);
  tensor::Tensor v(base::DataType::kDataTypeFp32, rows * kv, true, alloc);
  tensor::Tensor scores(base::DataType::kDataTypeFp32, rows * heads * rows, true, alloc);
  tensor::Tensor out(base::DataType::kDataTypeFp32, rows * heads * hs, true, alloc);
  for (int i = 0; i < q.size(); ++i) q.index<float>(i) = std::sin(i * .1f);
  for (int i = 0; i < k.size(); ++i) {k.index<float>(i) = std::cos(i * .2f); v.index<float>(i) = std::sin(i * .3f);}
  std::vector<float> expected(5 * heads * hs);
  for (int row = 0; row < 5; ++row) for (int head = 0; head < heads; ++head) {
    std::vector<double> weights(row + 1);
    double sum = 0;
    for (int t = 0; t <= row; ++t) {
      double dot = 0;
      for (int i = 0; i < hs; ++i) dot += double(q.index<float>((row * heads + head) * hs + i)) * k.index<float>(t * kv + head / 3 * hs + i);
      weights[t] = std::exp(dot / std::sqrt(double(hs))); sum += weights[t];
    }
    for (int i = 0; i < hs; ++i) {
      double value = 0;
      for (int t = 0; t <= row; ++t) value += weights[t] / sum * v.index<float>(t * kv + head / 3 * hs + i);
      expected[(row * heads + head) * hs + i] = value;
    }
  }
  // These rows exist in the physical batch but are causally invisible above.
  for (int i = 5 * kv; i < k.size(); ++i) k.index<float>(i) = v.index<float>(i) = std::numeric_limits<float>::quiet_NaN();
  kernel::CudaConfig config; ASSERT_EQ(cudaStreamCreate(&config.stream), cudaSuccess);
  q.to_cuda(config.stream); k.to_cuda(config.stream); v.to_cuda(config.stream); scores.to_cuda(config.stream); out.to_cuda(config.stream);
  kernel::prefill_attention(q.ptr<float>(), k.ptr<float>(), v.ptr<float>(), scores.ptr<float>(), out.ptr<float>(), rows, heads, kv_heads, hs, config.stream);
  ASSERT_EQ(cudaStreamSynchronize(config.stream), cudaSuccess); out.to_cpu();
  for (size_t i = 0; i < expected.size(); ++i) EXPECT_NEAR(out.index<float>(i), expected[i], 1e-5);
}
TEST(runtime_cuda, batched_norm_odd_width_and_model_epsilon) {
  if (!cuda_available()) GTEST_SKIP();
  auto alloc = base::CPUDeviceAllocatorFactory::get_instance();
  tensor::Tensor x(base::DataType::kDataTypeFp32, 3, 7, true, alloc);
  tensor::Tensor weight(base::DataType::kDataTypeFp32, 7, true, alloc);
  std::vector<float> expected(x.size());
  for (int i = 0; i < x.size(); ++i) x.index<float>(i) = std::sin(i * .4f) * .01f;
  for (int i = 0; i < 7; ++i) weight.index<float>(i) = 1.f + i * .1f;
  for (int row = 0; row < 3; ++row) {
    double squared = 0;
    for (int i = 0; i < 7; ++i) squared += double(x.index<float>(row * 7 + i)) * x.index<float>(row * 7 + i);
    for (int i = 0; i < 7; ++i) expected[row * 7 + i] = x.index<float>(row * 7 + i) / std::sqrt(squared / 7 + 1e-5) * weight.index<float>(i);
  }
  kernel::CudaConfig config; ASSERT_EQ(cudaStreamCreate(&config.stream), cudaSuccess);
  x.to_cuda(config.stream); weight.to_cuda(config.stream);
  kernel::prefill_norm(x.ptr<float>(), weight.ptr<float>(), x.ptr<float>(), 3, 7, 1e-5f, config.stream);
  ASSERT_EQ(cudaStreamSynchronize(config.stream), cudaSuccess); x.to_cpu();
  for (size_t i = 0; i < expected.size(); ++i) EXPECT_NEAR(x.index<float>(i), expected[i], 1e-6);
}
TEST(runtime_cuda, batched_prefill_repeated_shorter_and_longer_requests) {
  if (!cuda_available()) GTEST_SKIP();
  const char* path = std::getenv("LLAMA_TEST_MODEL");
  const char* tokenizer = std::getenv("LLAMA_TEST_TOKENIZER");
  if (!path || !tokenizer) GTEST_SKIP() << "Set LLAMA_TEST_MODEL and LLAMA_TEST_TOKENIZER";
#if defined(QWEN3_SUPPORT)
  model::Qwen3Model model(base::TokenizerType::kEncodeBpe, tokenizer, path, false);
#elif defined(QWEN2_SUPPORT)
  model::Qwen2Model model(base::TokenizerType::kEncodeBpe, tokenizer, path, false);
#elif defined(LLAMA3_SUPPORT)
  model::LLama2Model model(base::TokenizerType::kEncodeBpe, tokenizer, path, false);
#else
  model::LLama2Model model(base::TokenizerType::kEncodeSpe, tokenizer, path, false);
#endif
  ASSERT_TRUE(bool(model.init(base::DeviceType::kDeviceCUDA)));
  for (int length : {1, 33, 129, 8, 1}) {
    runtime::GenerationOptions options; options.max_new_tokens = 8; options.ignore_eos = true;
    for (int i = 0; i < length; ++i) options.prompt_token_ids.push_back(10 + i % 100);
    runtime::GenerationResult batch, reference;
    options.prefill_mode = "batched";
    ASSERT_TRUE(bool(runtime::generate(model, "", options, batch)));
    EXPECT_EQ(batch.prefill_mode, "batched");
    options.prefill_mode = "reference";
    ASSERT_TRUE(bool(runtime::generate(model, "", options, reference)));
    EXPECT_EQ(batch.generated_tokens, reference.generated_tokens) << "length=" << length;
  }
  runtime::GenerationOptions options; options.prefill_mode = "batched"; options.max_new_tokens = 0;
  options.prompt_token_ids = {10}; runtime::GenerationResult result;
  ASSERT_TRUE(bool(runtime::generate(model, "", options, result))); EXPECT_TRUE(result.generated_tokens.empty());
  options.max_new_tokens = model.context_length() + 1;
  EXPECT_FALSE(bool(runtime::generate(model, "", options, result)));
  options.max_new_tokens = 1; options.prefill_mode = "invalid";
  EXPECT_FALSE(bool(runtime::generate(model, "", options, result)));
}
