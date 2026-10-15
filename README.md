# my_llm_inference

**换机器接着跑：先读 [完整交接文档](docs/HANDOFF.md)。** 当前四模型 CPU/CUDA 正确性已验证，原机器 S0 有效 23/36 场景，三个优化特性尚未实现。

基于 [KuiperLLama](https://github.com/zjhellofss/KuiperLLama) 的 C++20 / CUDA 推理项目。当前已跑通 **TinyStories 15M、Qwen2.5-0.5B-Instruct、Qwen3-0.6B、Llama3.2-1B-Instruct 的 FP32、单请求、贪心生成**，含权重加载、分词、模型前向、GQA/KV Cache、EOS/长度停止和数值对照。

本轮保留原项目的 Tensor/Buffer/Layer，接入上游缺失模块，修正阻塞推理的实现，并新增 CLI、独立验证与性能基准。来源见 [UPSTREAM](docs/UPSTREAM.md)；结果见 [VALIDATION](docs/VALIDATION.md)；后续亮点见 [AI Infra 改进清单](docs/IMPROVEMENTS.md)。

## 立即运行（当前工作区已构建并下载模型）

```bash
# Llama 风格小模型，约 61 MB
OPENBLAS_NUM_THREADS=1 ./build/demo/llama_infer \
  models/stories15M.bin models/tokenizer.model \
  --device cpu --prompt 'Once upon a time' --max-new-tokens 64

# 同模型 CUDA 路径
./build/demo/llama_infer models/stories15M.bin models/tokenizer.model \
  --device cuda --prompt 'Once upon a time' --max-new-tokens 64 --json

# Qwen 中文指令生成（封装了该模型要求的聊天模板）
bash scripts/run_qwen_chat.sh '请用一句话解释什么是大模型推理。'
# CPU 路径
QWEN_DEVICE=cpu OPENBLAS_NUM_THREADS=1 bash scripts/run_qwen_chat.sh '你好'
```

`--json` 输出生成文本、prompt/generated token IDs、停止原因、模型加载耗时、TTFT、decode 吞吐。文本不包含 prompt 和终止标记；generated token IDs 保留 EOS，便于对齐。`--dump-logits FILE` 保存每次 forward 的 FP32 logits，仅用于正确性诊断，会干扰性能计时。

## 从源码构建

需要 GCC 的 C++20 支持、CMake >= 3.18、Ninja、CUDA Toolkit、BLAS/LAPACK 开发库。**当前 CPU 执行也链接 CUDA runtime，因此编译 CPU 路径仍需要 Toolkit，但 CPU 运行不需要可用 GPU。** 依赖通过 CMake FetchContent 下载。

```bash
cmake --preset default
cmake --build --preset default --target llama_infer test_llm -j 8
```

预设从 CUDACXX/PATH 发现 CUDA，并使用当前机器的 native 架构（CMake >= 3.24）。也可通过构建脚本自动发现或显式指定编译器和架构：

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/path/to/nvcc -DCMAKE_CUDA_ARCHITECTURES=YOUR_ARCH \
  -DLLAMA3_SUPPORT=OFF -DQWEN2_SUPPORT=OFF -DQWEN3_SUPPORT=OFF
cmake --build build --target llama_infer test_llm -j 8
```

默认构建使用 Llama2 的 SentencePiece / RoPE，适配 TinyStories legacy FP32 文件。Qwen2 使用独立构建目录：

```bash
cmake --preset qwen2
cmake --build --preset qwen2 --target llama_infer test_llm -j 8
```

`llama3`、`qwen3` 已通过真实权重的 CPU/CUDA 数值验证；正式 S0 全矩阵尚未完成，进展见 [交接状态](docs/performance/S0_PARTIAL.md)。模型变体的 RoPE、RMSNorm epsilon、tokenizer 与导出格式不同，每个构建只允许打开一个模型选项。不要用 Qwen/Llama3 构建读取 TinyStories 权重。

## 模型下载与转换

```bash
bash scripts/download_smoke_model.sh
# 下载约 1 GB BF16 权重；导出后的 FP32 文件约 2 GB
bash scripts/download_qwen2.sh
```

模型目录被 Git 忽略。Qwen 脚本固定了官方仓库 revision；直接从 safetensors 转换，仅需 NumPy，无需安装 transformers，也不重新加载训练模型。导出上下文默认设为 2048，以控制预分配 KV Cache 的占用；这不改变权重，可在导出命令中设定其他受支持长度。

上游导出工具保留在 `tools/`。注意运行时读取的是 **legacy 自定义 bin**，不能直接读取 Hugging Face safetensors/GGUF，也不接受上游所有带新版本头的 bin 文件。上游 INT8 路径保留，但本轮没有完成端到端量化质量验证；Qwen3 INT8 明确拒绝。

## 验证

```bash
# 不需要 GPU 的测试；提供模型变量才能运行真实模型重复请求测试
export LLAMA_TEST_MODEL="$PWD/models/stories15M.bin"
export LLAMA_TEST_TOKENIZER="$PWD/models/tokenizer.model"
OPENBLAS_NUM_THREADS=1 ctest --test-dir build -R '^cpu_unit$' --output-on-failure

# 有 GPU 时运行完整算子/运行时测试
OPENBLAS_NUM_THREADS=1 ./build/test/test_llm
python3 scripts/smoke_test.py --cuda

# 逐步 logits / greedy token 对照，依赖 numpy + torch
OPENBLAS_NUM_THREADS=1 python3 scripts/verify_reference.py --device cpu
OPENBLAS_NUM_THREADS=1 python3 scripts/verify_reference.py --device cuda

# Qwen 对照原始 safetensors 和官方 tokenizer，另需 safetensors + tokenizers
OPENBLAS_NUM_THREADS=1 python3 scripts/verify_qwen2.py --device cuda
```

当前机器的附加验证依赖安装在被忽略的 `build/validation-deps`，没有修改全局 Python 环境；这里可使用 `PYTHONPATH="$PWD/build/validation-deps"` 执行 Qwen 验证。

```bash
# 旧脚本仅保留 CPU 微基准；正式 GPU 实验按 HANDOFF 建立本机阶段后使用 benchmark_suite.py
python3 scripts/benchmark.py --devices cpu
```

吞吐指标排除模型加载与 prefill，decode tokens/s 使用首个生成 token 之后的 token 数 / decode 时间。TTFT 包含分词和 prompt 计算，不包含模型加载。这是单请求基线，不是并发服务吞吐。服务器有其他 GPU 负载，短序列和小模型的数字不能外推到大模型或生产环境。

## 代码入口

- `demo/main.cpp`：CLI、模型选择、JSON 和诊断输出。
- `source/runtime/generation.cpp`：完整自回归生成循环。
- `source/model/`：模型加载、层组装、KV Cache、前向。
- `source/op/kernels/`：CPU/CUDA 算子及调度。
- `scripts/verify_reference.py` / `verify_qwen2.py`：独立数值对照。
- `docs/IMPROVEMENTS.md`：拟议改进、工作量、指标和简历表述边界。

当前仍是单请求推理基线；没有 continuous batching、分页 KV、共享前缀缓存和正式服务调度。这些是下一轮可自行实现并通过实验展示的 AI Infra 亮点。

## 分阶段优化与空卡实验

当前目标依次为：采集完整 S0 原始基线、实现批量 prefill、实现 CUDA Graph/算子融合、实现分页 KV。新机器应先按 HANDOFF 设置 LLM_BASELINE_STAGE 并冻结本机版本，不能复用旧 S0 路径。完整计划和模型/长度矩阵见 [PLAN](docs/performance/PLAN.md)、[BRINGUP](docs/performance/BRINGUP.md)。旧的共享 GPU 数字只作历史记录。

```bash
# 全部变体使用同一套代码构建（复用本机缓存依赖）
python3 scripts/build_variants.py --cuda-arch native
# 新模型原始 safetensors -> FP32 legacy bin，保留正确 RoPE 配置
python3 scripts/export_hf.py models/qwen3-0.6b models/qwen3-0.6b/model.bin --context 2048
python3 scripts/export_hf.py models/llama3.2-1b-instruct models/llama3.2-1b-instruct/model.bin --context 2048
# 原始权重 CPU 数值校验
PYTHONPATH="$PWD/build/validation-deps" OPENBLAS_NUM_THREADS=1 python3 scripts/verify_hf.py \
  --binary build-qwen3/demo/llama_infer --model-dir models/qwen3-0.6b --device cpu
# 空卡门槛与阶段基线（快照由 freeze_stage.py 创建，阶段不可覆盖）
python3 scripts/benchmark_suite.py --stage "$LLM_BASELINE_STAGE" --wait-seconds 30
```

空卡监控依赖 `nvidia-ml-py`，已安装在 `build/validation-deps`。无空卡时退出码 75，并保存占用证据；不会在繁忙 GPU 上降级测速。GPU 原始性能通过 `benchmark_suite.py` 收集，旧 `benchmark.py` 仅保留 CPU 用途。
