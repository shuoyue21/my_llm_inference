#pragma once
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <unordered_map>
#include <vector>

namespace runtime {
using RequestId = uint64_t;
// One backend call is one scheduling iteration. Implementations may merge all
// selected rows into GPU GEMMs; the scheduler never assumes serial execution.
struct ScheduledToken {
  RequestId id;
  int32_t token;
  int position;
  bool prompt;
};
struct TokenOutput {
  int32_t token;
  bool eos = false;
};
class BatchBackend {
 public:
  virtual ~BatchBackend() = default;
  virtual int context_length() const = 0;
  virtual int vocab_size() const = 0;
  virtual void admit(RequestId id) = 0;
  virtual std::vector<TokenOutput> execute(const std::vector<ScheduledToken>& batch) = 0;
  virtual void release(RequestId id) = 0;
  virtual int reuse_prefix(RequestId, const std::vector<int32_t>&) { return 0; }
};
struct ServingRequest {
  RequestId id;
  std::vector<int32_t> prompt;
  int max_new_tokens = 64;
  bool ignore_eos = false;
  double initial_queue_ms = 0;
};
struct ServingResult {
  RequestId id;
  std::vector<int32_t> tokens;
  bool cancelled = false;
  double queue_ms = 0, ttft_ms = 0, total_ms = 0;
  std::vector<double> inter_token_ms;
};
struct SchedulerOptions {
  size_t max_active = 8;
  size_t token_budget = 32;
  size_t max_pending = 1024;
};
// Single host owner. submit/cancel can be interleaved with step(), including
// while earlier requests are decoding. IDs cannot be reused in this instance.
class Scheduler {
 public:
  Scheduler(BatchBackend& backend, SchedulerOptions options = {});
  ~Scheduler();
  Scheduler(const Scheduler&) = delete;
  Scheduler& operator=(const Scheduler&) = delete;
  void submit(ServingRequest request);
  bool cancel(RequestId id);
  bool step();
  bool empty() const;
  std::vector<ServingResult> take_completed();

 private:
  struct State;
  void finish(RequestId id, bool cancelled);
  BatchBackend& backend_;
  SchedulerOptions options_;
  std::unordered_map<RequestId, std::unique_ptr<State>> requests_;
  std::unordered_map<RequestId, bool> used_;
  std::deque<RequestId> pending_, active_;
  std::vector<ServingResult> completed_;
};
}  // namespace runtime
