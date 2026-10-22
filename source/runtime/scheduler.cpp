#include "runtime/scheduler.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>

namespace runtime {
namespace {
using Clock = std::chrono::steady_clock;
double ms(Clock::time_point end, Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}
}
struct Scheduler::State {
  ServingRequest request;
  ServingResult result;
  Clock::time_point submitted = Clock::now(), previous;
  int position = 0;
  int32_t next = 0;
  bool admitted = false;
};
Scheduler::Scheduler(BatchBackend& backend, SchedulerOptions options)
    : backend_(backend), options_(options) {
  if (!options.max_active || !options.token_budget || !options.max_pending)
    throw std::invalid_argument("Scheduler limits must be positive");
}
Scheduler::~Scheduler() {
  for (const auto& [id, state] : requests_)
    if (state->admitted) { try { backend_.release(id); } catch (...) {} }
}
void Scheduler::submit(ServingRequest request) {
  if (!std::isfinite(request.initial_queue_ms) || request.initial_queue_ms < 0) throw std::invalid_argument("Invalid initial queue delay");
  if (used_.count(request.id)) throw std::invalid_argument("Duplicate request id");
  if (pending_.size() >= options_.max_pending) throw std::runtime_error("Pending queue is full");
  if (request.prompt.empty() || request.max_new_tokens < 0 ||
      request.prompt.size() > size_t(backend_.context_length()) ||
      uint64_t(request.prompt.size()) + uint64_t(std::max(0, request.max_new_tokens - 1)) >
          uint64_t(backend_.context_length()))
    throw std::invalid_argument("Invalid request length");
  for (auto token : request.prompt)
    if (token < 0 || token >= backend_.vocab_size())
      throw std::invalid_argument("Invalid prompt token");
  auto state = std::make_unique<State>();
  state->submitted -= std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double, std::milli>(request.initial_queue_ms));
  state->result.id = request.id;
  state->request = std::move(request);
  auto id = state->request.id;
  requests_.emplace(id, std::move(state));
  used_.emplace(id, true);
  pending_.push_back(id);
}
void Scheduler::finish(RequestId id, bool cancelled) {
  auto& state = *requests_.at(id);
  if (state.admitted) {
    backend_.release(id);
    state.admitted = false;
  }
  state.result.cancelled = cancelled;
  state.result.total_ms = ms(Clock::now(), state.submitted);
  completed_.push_back(std::move(state.result));
  requests_.erase(id);
  std::erase(pending_, id);
  std::erase(active_, id);
}
bool Scheduler::cancel(RequestId id) {
  if (!requests_.count(id)) return false;
  finish(id, true);
  return true;
}
bool Scheduler::step() {
  while (!pending_.empty() && active_.size() < options_.max_active) {
    auto id = pending_.front();
    auto& state = *requests_.at(id);
    if (!state.request.max_new_tokens) { finish(id, false); continue; }
    backend_.admit(id);
    state.admitted = true;
    try {
      state.position = backend_.reuse_prefix(id, state.request.prompt);
      if (state.position < 0 || state.position >= int(state.request.prompt.size()))
        throw std::runtime_error("Invalid reused prefix length");
    } catch (...) { finish(id, true); throw; }
    state.result.queue_ms = ms(Clock::now(), state.submitted);
    pending_.pop_front();
    active_.push_back(id);
  }
  if (active_.empty()) return false;
  // Give every selected request a fair share of this iteration's budget.
  // Long prompts are chunked; decode consumes one row per request.
  const auto count = std::min(options_.token_budget, active_.size());
  std::vector<ScheduledToken> batch;
  batch.reserve(options_.token_budget);
  for (size_t i = 0; i < count; ++i) {
    auto id = active_.front(); active_.pop_front(); active_.push_back(id);
    const auto& state = *requests_.at(id);
    const int remaining_prompt = int(state.request.prompt.size()) - state.position;
    const int share = (options_.token_budget - batch.size()) / (count - i);
    const int rows = remaining_prompt > 0 ? std::min(remaining_prompt, share) : 1;
    for (int r = 0; r < rows; ++r) {
      const int position = state.position + r;
      const bool prompt = position < int(state.request.prompt.size());
      batch.push_back({id, prompt ? state.request.prompt[position] : state.next,
                       position, prompt});
    }
  }
  // An execution failure can leave partially written KV. Fail closed: discard
  // all affected requests, propagate the error, and keep unrelated requests.
  std::vector<TokenOutput> output;
  try {
    output = backend_.execute(batch);
    if (output.size() != batch.size()) throw std::runtime_error("Backend row count mismatch");
    for (auto row : output)
      if (row.token < 0 || row.token >= backend_.vocab_size())
        throw std::runtime_error("Backend returned invalid token");
  } catch (...) {
    for (const auto& row : batch) if (requests_.count(row.id)) { try { finish(row.id, true); } catch (...) {} }
    throw;
  }
  auto now = Clock::now();
  for (size_t i = 0; i < batch.size(); ++i) {
    auto& state = *requests_.at(batch[i].id);
    ++state.position;
    if (state.position < int(state.request.prompt.size())) continue;
    state.next = output[i].token;
    if (state.result.tokens.empty()) state.result.ttft_ms = ms(now, state.submitted);
    else state.result.inter_token_ms.push_back(ms(now, state.previous));
    state.previous = now;
    state.result.tokens.push_back(state.next);
    if (int(state.result.tokens.size()) >= state.request.max_new_tokens ||
        (output[i].eos && !state.request.ignore_eos)) finish(batch[i].id, false);
  }
  return true;
}
bool Scheduler::empty() const { return requests_.empty(); }
std::vector<ServingResult> Scheduler::take_completed() {
  std::vector<ServingResult> result;
  result.swap(completed_);
  return result;
}
}  // namespace runtime
