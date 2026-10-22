#include <gtest/gtest.h>
#include "runtime/scheduler.h"
#include "runtime/cuda_batch.h"
#include "runtime/generation.h"
#include "model/llama3.h"
#if defined(QWEN2_SUPPORT)
#include "model/qwen2.h"
#elif defined(QWEN3_SUPPORT)
#include "model/qwen3.h"
#endif
#include <cstdlib>
#include <map>
#include <set>
namespace {
class ReferenceBackend : public runtime::BatchBackend {
 public:
  std::map<runtime::RequestId, int> positions;
  size_t max_batch = 0;
  bool fail = false;
  int context_length() const override { return 256; }
  int vocab_size() const override { return 1024; }
  void admit(runtime::RequestId id) override { ASSERT_TRUE(positions.emplace(id, 0).second); }
  void release(runtime::RequestId id) override { EXPECT_EQ(positions.erase(id), 1); }
  std::vector<runtime::TokenOutput> execute(const std::vector<runtime::ScheduledToken>& batch) override {
    max_batch = std::max(max_batch, batch.size());
    std::vector<runtime::TokenOutput> out;
    for (auto row : batch) {
      EXPECT_EQ(positions.at(row.id)++, row.position);
      out.push_back({row.token + 1, false});
    }
    if (fail) throw std::runtime_error("Injected execution failure");
    return out;
  }
};
}
TEST(scheduler_cpu, dynamic_arrivals_budget_cancellation_and_isolation) {
  ReferenceBackend backend;
  runtime::Scheduler scheduler(backend, {3, 2, 8});
  scheduler.submit({1, {10, 11, 12}, 4, true});
  scheduler.submit({2, {100}, 6, true});
  ASSERT_TRUE(scheduler.step());
  scheduler.submit({3, {200, 201}, 2, true});
  scheduler.submit({4, {300}, 2, true});
  ASSERT_TRUE(scheduler.cancel(4));
  while (!scheduler.empty()) scheduler.step();
  auto results = scheduler.take_completed();
  ASSERT_EQ(results.size(), 4);
  for (const auto& result : results) {
    if (result.id == 1) EXPECT_EQ(result.tokens, (std::vector<int>{13, 14, 15, 16}));
    if (result.id == 2) EXPECT_EQ(result.tokens, (std::vector<int>{101, 102, 103, 104, 105, 106}));
    if (result.id == 3) EXPECT_EQ(result.tokens, (std::vector<int>{202, 203}));
    if (result.id == 4) { EXPECT_TRUE(result.cancelled); EXPECT_TRUE(result.tokens.empty()); }
  }
  EXPECT_EQ(backend.max_batch, 2); EXPECT_TRUE(backend.positions.empty());
  EXPECT_THROW(scheduler.submit({1, {1}, 1, true}), std::invalid_argument);
}
TEST(scheduler_cpu, failed_iteration_releases_only_affected_requests) {
  ReferenceBackend backend; runtime::Scheduler scheduler(backend, {3, 1, 8});
  scheduler.submit({1, {1}, 1, true}); scheduler.submit({2, {2}, 1, true});
  backend.fail = true; EXPECT_THROW(scheduler.step(), std::runtime_error);
  backend.fail = false;
  while (!scheduler.empty()) scheduler.step();
  auto results = scheduler.take_completed(); ASSERT_EQ(results.size(), 2);
  EXPECT_TRUE(results[0].cancelled); EXPECT_FALSE(results[1].cancelled);
  EXPECT_TRUE(backend.positions.empty());
}
TEST(scheduler_cuda, mixed_requests_match_serial_generation) {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) GTEST_SKIP();
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
  std::vector<runtime::ServingRequest> requests;
  std::map<uint64_t, std::vector<int32_t>> references;
  for (int i = 0; i < 4; ++i) {
    runtime::ServingRequest request{uint64_t(i), {}, 8 + i, true};
    for (int t = 0; t < std::vector<int>{1, 17, 33, 8}[i]; ++t) request.prompt.push_back(10 + (t + i * 7) % 100);
    runtime::GenerationOptions options; options.prompt_token_ids = request.prompt;
    options.max_new_tokens = request.max_new_tokens; options.ignore_eos = true;
    runtime::GenerationResult result;
    ASSERT_TRUE(bool(runtime::generate(model, "", options, result)));
    references[i] = result.generated_tokens; requests.push_back(request);
  }
  auto backend = model.create_batch_backend(3);
  backend->configure_prefix_cache(64 * 1024 * 1024);
  if (const char* mode = std::getenv("LLAMA_TEST_ATTENTION")) backend->configure_attention(mode);
  for (int repeat = 0; repeat < 2; ++repeat) {
    runtime::Scheduler scheduler(*backend, {3, 2, 8});
    scheduler.submit(requests[0]); scheduler.submit(requests[1]);
    ASSERT_TRUE(scheduler.step()); scheduler.submit(requests[2]); scheduler.submit(requests[3]);
    while (!scheduler.empty()) scheduler.step();
    auto results = scheduler.take_completed(); ASSERT_EQ(results.size(), 4);
    for (const auto& result : results) {
      EXPECT_EQ(result.tokens, references.at(result.id)) << "request=" << result.id;
      EXPECT_EQ(result.inter_token_ms.size(), result.tokens.size() - 1);
    }
    EXPECT_EQ(backend->cache_stats().mapped_bytes, 0);
    if (repeat) EXPECT_GE(backend->prefix_stats().hits, 2);
  }
  const auto page_bytes = backend->prefix_stats().resident_bytes / 3;
  ASSERT_GT(page_bytes, 0);
  backend->configure_prefix_cache(page_bytes);
  auto changed = requests[1]; changed.id = 99; ++changed.prompt[0];
  runtime::GenerationOptions gold_options; gold_options.prompt_token_ids = changed.prompt;
  gold_options.max_new_tokens = changed.max_new_tokens; gold_options.ignore_eos = true;
  runtime::GenerationResult gold;
  ASSERT_TRUE(bool(runtime::generate(model, "", gold_options, gold)));
  runtime::Scheduler eviction(*backend, {1, 2, 8});
  eviction.submit(requests[1]); eviction.submit(changed);
  while (!eviction.empty()) eviction.step();
  auto evicted_results = eviction.take_completed(); ASSERT_EQ(evicted_results.size(), 2);
  EXPECT_EQ(evicted_results[0].tokens, references.at(1));
  EXPECT_EQ(evicted_results[1].tokens, gold.generated_tokens);
  EXPECT_GE(backend->prefix_stats().evictions, 1);
  EXPECT_LE(backend->prefix_stats().resident_bytes, page_bytes);

}
