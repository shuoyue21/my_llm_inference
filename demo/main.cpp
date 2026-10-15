#include "runtime/generation.h"
#include "model/llama3.h"
#if defined(QWEN2_SUPPORT)
#include "model/qwen2.h"
#elif defined(QWEN3_SUPPORT)
#include "model/qwen3.h"
#endif
#include <cuda_runtime_api.h>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace {
std::string json_string(const std::string& s) {
  std::ostringstream out;
  out << '"';
  for (unsigned char c : s) {
    if (c == '"' || c == '\\') out << '\\' << c;
    else if (c < 0x20) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c) << std::dec;
    else out << c;
  }
  out << '"';
  return out.str();
}
void ids(const std::vector<int32_t>& values) {
  std::cout << '[';
  for (size_t i = 0; i < values.size(); ++i) { if (i) std::cout << ','; std::cout << values[i]; }
  std::cout << ']';
}
void usage() {
  std::cout << "Usage: llama_infer CHECKPOINT TOKENIZER [--device cpu|cuda] [--prompt TEXT]\n"
               "       [--max-new-tokens N] [--json] [--quant] [--dump-logits FILE]\n"
               "Greedy decoding. JSON reports generated text, token IDs and separate TTFT/decode metrics.\n";
}
}
int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_logtostderr = true;
  if (argc == 2 && std::string(argv[1]) == "--help") { usage(); return 0; }
  if (argc < 3) { usage(); return 2; }
  try {
    std::string prompt = "Once upon a time", device = "cpu", dump_path;
    bool json = false, quant = false;
    runtime::GenerationOptions options;
    for (int i = 3; i < argc; ++i) {
      std::string flag = argv[i];
      if (flag == "--json") { json = true; continue; }
      if (flag == "--quant") { quant = true; continue; }
      if (flag != "--prompt" && flag != "--device" && flag != "--max-new-tokens" && flag != "--dump-logits")
        throw std::invalid_argument("Unknown option: " + flag);
      if (++i == argc) throw std::invalid_argument("Missing value for " + flag);
      std::string value = argv[i];
      if (flag == "--prompt") prompt = value;
      else if (flag == "--device") device = value;
      else if (flag == "--dump-logits") dump_path = value;
      else { size_t end = 0; options.max_new_tokens = std::stoi(value, &end);
             if (end != value.size() || options.max_new_tokens < 0) throw std::invalid_argument("Invalid max-new-tokens"); }
    }
    if (device != "cpu" && device != "cuda") throw std::invalid_argument("Device must be cpu or cuda");
    if (device == "cuda") {
      int count = 0;
      auto error = cudaGetDeviceCount(&count);
      if (error != cudaSuccess || count == 0)
        throw std::runtime_error(std::string("CUDA unavailable: ") + cudaGetErrorString(error));
    }
    if (!std::ifstream(argv[1]).good() || !std::ifstream(argv[2]).good())
      throw std::invalid_argument("Checkpoint or tokenizer file does not exist");
    auto load_start = std::chrono::steady_clock::now();
    std::unique_ptr<model::Model> model;
#if defined(QWEN3_SUPPORT)
    model = std::make_unique<model::Qwen3Model>(base::TokenizerType::kEncodeBpe, argv[2], argv[1], quant);
#elif defined(QWEN2_SUPPORT)
    model = std::make_unique<model::Qwen2Model>(base::TokenizerType::kEncodeBpe, argv[2], argv[1], quant);
#elif defined(LLAMA3_SUPPORT)
    model = std::make_unique<model::LLama2Model>(base::TokenizerType::kEncodeBpe, argv[2], argv[1], quant);
#else
    model = std::make_unique<model::LLama2Model>(base::TokenizerType::kEncodeSpe, argv[2], argv[1], quant);
#endif
    auto status = model->init(device == "cpu" ? base::DeviceType::kDeviceCPU : base::DeviceType::kDeviceCUDA);
    if (!status) throw std::runtime_error(status.get_err_msg());
    if (device == "cuda") {
      auto error = cudaDeviceSynchronize();
      if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
    }
    double load_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - load_start).count();
    std::ofstream dump;
    if (!dump_path.empty()) {
      dump.open(dump_path, std::ios::binary);
      if (!dump) throw std::runtime_error("Cannot open logits output file");
      options.on_logits = [&](int, const tensor::Tensor& logits) {
        std::vector<float> values(logits.size());
        if (device == "cuda") {
          auto error = cudaDeviceSynchronize();
          if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
          error = cudaMemcpy(values.data(), logits.ptr<float>(), logits.byte_size(), cudaMemcpyDeviceToHost);
          if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
        } else std::copy_n(logits.ptr<float>(), logits.size(), values.data());
        dump.write(reinterpret_cast<const char*>(values.data()), logits.byte_size());
        if (!dump) throw std::runtime_error("Failed to write logits");
      };
    }
    runtime::GenerationResult result;
    status = runtime::generate(*model, prompt, options, result);
    if (!status) throw std::runtime_error(status.get_err_msg());
    if (json) {
      std::cout << std::setprecision(9) << "{\"device\":" << json_string(device)
                << ",\"text\":" << json_string(result.text) << ",\"prompt_tokens\":";
      ids(result.prompt_tokens);
      std::cout << ",\"generated_tokens\":"; ids(result.generated_tokens);
      std::cout << ",\"stop_reason\":" << json_string(result.stop_reason)
                << ",\"context_length\":" << model->context_length()
                << ",\"load_ms\":" << load_ms << ",\"ttft_ms\":" << result.ttft_ms
                << ",\"decode_ms\":" << result.decode_ms << ",\"total_ms\":" << result.total_ms
                << ",\"decode_tokens_per_second\":" << result.decode_tokens_per_second << "}\n";
    } else {
      std::cout << result.text << '\n';
      std::cerr << "prompt_tokens=" << result.prompt_tokens.size() << " generated_tokens=" << result.generated_tokens.size()
                << " stop=" << result.stop_reason << " load_ms=" << load_ms << " ttft_ms=" << result.ttft_ms
                << " decode_tokens/s=" << result.decode_tokens_per_second << '\n';
    }
    return 0;
  } catch (const std::exception& error) { std::cerr << "Error: " << error.what() << '\n'; return 1; }
}
