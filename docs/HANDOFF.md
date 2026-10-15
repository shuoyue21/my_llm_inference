# 换机器运行与后续开发交接

最后整理：2026-10-15。工作分支：`codex/complete-inference`。仓库：<https://github.com/shuoyue21/my_llm_inference>。

## 任务目标与真实进度

用户要求：先跑通 Qwen3/Llama3；在**未实现任何新特性**的 S0 上记录各种模型和输入/输出长度的性能；再依次实现前三个特性，每完成一项记录正确性、性能改善或退化。全部 GPU 性能测试必须在空卡上运行。用户已同意自动使用任何空闲卡，不需要再次询问卡号。

| 项目 | 当前状态 |
|---|---|
| TinyStories 15M / Llama2 风格、Qwen2.5-0.5B-Instruct、Qwen3-0.6B、Llama3.2-1B-Instruct | CPU 和 CUDA 推理、独立逐步 logits、分词、greedy token 对齐通过 |
| Qwen3 / Llama3 完整问答与 EOS | 通过；CPU 示例在 docs/performance/bringup/ |
| S0 原始基线 | 旧机器有效 23/36 项，尚未完成完整矩阵；[明细和证据](performance/S0_PARTIAL.md) |
| S1：批量 prefill | 未实现 |
| S2：CUDA Graph + 算子融合 | 未实现 |
| S3：分页 KV Cache | 未实现 |

基线推理代码快照是 `6d36c27`；后续提交只改测速可靠性及交接/构建工具，没有实现三个优化。原机器的后台队列已经停止，保留了已完成记录，不要从旧状态文件推断它还活着。全局聊天 goal 曾因无空卡标记 blocked；换机器后直接按本文继续任务。

## 新机器从零开始

以下命令在 Linux / Bash 中执行。建议 Python 3.11、支持 C++20 的 GCC、CMake >= 3.24、CUDA Toolkit >= 12.1，以及至少 25 GB 可用磁盘。已验证环境是 GCC 12 / CUDA 12.1 / RTX 4090。其他 GPU/工具链需要在目标机器重新构建和校验。

### 1. 获取代码

```bash
git clone --branch codex/complete-inference --single-branch \
  https://github.com/shuoyue21/my_llm_inference.git
cd my_llm_inference
```

如果使用随交接提供的 Git bundle（不含模型），也可以：

```bash
git clone --branch codex/complete-inference my_llm_inference.bundle my_llm_inference
cd my_llm_inference
git remote set-url origin https://github.com/shuoyue21/my_llm_inference.git
```

模型权重、编译产物、Python 环境和 `.artifacts/` 冻结二进制不上传 GitHub。仓库里保留源码、下载/转换脚本、模型资产哈希、测试证据和有效/被拒绝的原始实验记录。旧机器的 docs/performance/stages/S0/manifest.json 仅用于追溯，不是新机器可直接执行的二进制快照。

### 2. 准备工具和 Python 环境

系统依赖（按本机权限安装，也可使用已有环境）：

```bash
sudo apt-get update
sudo apt-get install -y build-essential git curl python3-venv pkg-config libopenblas-dev liblapack-dev
nvcc --version
nvidia-smi

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install 'cmake>=3.24' ninja
python -m pip install -r requirements-validation.txt
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
```

PyTorch 只用于独立正确性参考，CPU wheel 足够；实际 CUDA 推理由 C++ 程序执行。不要将旧机器 `build/validation-deps` 的绝对路径写到新机器。激活 `.venv` 后无需设置旧 PYTHONPATH。

### 3. 构建四个模型变体

```bash
python scripts/build_variants.py --jobs 8 --cuda-arch native
```

构建脚本会自动从 `CUDACXX` / PATH / 常见路径发现 nvcc，首次联网下载 CMake 依赖，随后复用源码缓存。`native` 要求 CMake >= 3.24 和可访问的 GPU 驱动。

需要手工指定时：

```bash
python scripts/build_variants.py --cuda-compiler /usr/local/cuda/bin/nvcc --cuda-arch 80 --jobs 8
```

上例 `80` 用于 A100；RTX 4090 使用 `89`。以实际 GPU 为准，不要沿用旧机器写死的架构。编译日志为 `/tmp/my_llm_build_<variant>.log`。只有依赖缓存齐全时才使用 `--offline`；默认允许首次下载。若编译器不在 CUDA 支持范围内，请选择兼容 GCC 后重新建立构建目录。

### 4. 下载完全相同的模型资产

```bash
python scripts/download_models.py
```

下载器使用 [model_downloads.json](../scripts/model_downloads.json) 中固定的 Hugging Face revision、文件长度和 SHA256，随后转换为经过验证的 FP32 legacy bin，继续校验导出后的哈希。既有文件若不是指定版本会报错，不会直接覆盖。所有模型和导出文件合计约 14 GB。

也可以先下载少量模型调试：

```bash
python scripts/download_models.py --models tinystories15m
python scripts/download_models.py --models qwen3-0.6b llama3.2-1b
```

完整基线必须准备四个模型；不能仅用小模型替代。Llama3.2 使用公开可访问的 `unsloth/Llama-3.2-1B-Instruct` 分发，精确 revision 记录在清单。Qwen3/Llama3 的 `model.bin.json` 是 RoPE 配置的必要交接资产，不要只拷贝 bin 而漏掉 sidecar。

如网络不便，可从旧机器复制 models/ 和 CMake 的 `build/_deps/*-src` 到对应路径，再由脚本验证。不要复制旧 CMakeCache.txt 或旧 `.venv` 到不同机器。

### 5. 快速确认正确性

```bash
python test/test_handoff.py
python test/test_empty_gpu.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/verify_reference.py --device cpu
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/verify_hf.py \
  --binary build-qwen2/demo/llama_infer --model-dir models/qwen2.5-0.5b-instruct --device cpu
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/verify_hf.py \
  --binary build-qwen3/demo/llama_infer --model-dir models/qwen3-0.6b --device cpu
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/verify_hf.py \
  --binary build-llama3/demo/llama_infer --model-dir models/llama3.2-1b-instruct --device cpu
```

CPU 校验不要求空 GPU，其中的执行时间不作为正式 GPU 性能结果。完整 GPU 测试由下一步的队列自动在空卡上执行。

### 6. 创建新机器自己的 S0 并采集完整基线

不要覆盖已经发布的 `S0` 目录。冻结工具会拒绝覆盖现有阶段。

```bash
export LLM_RUN_ID="$(hostname -s)-$(date +%Y%m%d-%H%M%S)"
export LLM_BASELINE_STAGE="S0-$LLM_RUN_ID"
python scripts/freeze_stage.py --stage "$LLM_BASELINE_STAGE"

set -o pipefail
python -u scripts/stage0_runner.py --stage "$LLM_BASELINE_STAGE" 2>&1 \
  | tee ".artifacts/performance/$LLM_BASELINE_STAGE/runner.log"
```

建议在 tmux/screen 里运行。队列先执行四模型 GPU 单元测试和独立数值对照，再测完整矩阵。找不到空卡时等待；测试中出现其他计算进程则保留污染证据并重新排队。不要终止别人的进程，也不要降低空卡门槛来凑数据。

矩阵来自 [model_registry.json](../scripts/model_registry.json)：FP32，prompt 长度 8/32/128/512/1536，输出 32/128，受各模型上下文限制；每场景同一驻留模型预热 2 次，再测 5 次，共 **36 场景**。忽略 EOS 以固定输出长度。TinyStories 上下文 256，其余导出上下文 2048。

空卡门槛：连续 5 次采样无外部计算进程、实际显存 <=128 MiB、GPU 利用率 <=1%。驱动保留显存单独计算。仅豁免已确认的 `/usr/lib/xorg/Xorg` 小型图形驻留（<=16 MiB），所有其他图形/计算任务均拒绝。测量中约每 20 ms 监控并记录实际最大采样间隔。

### 7. 判断是否真正完成

查看本次阶段下的：

- `runner_status.json`：`phase` 应为 `baseline_complete_no_features_implemented`。
- `cumulative/<GPU_UUID>/summary.json`：`status=complete` 且 `completed_cases` 有 36 项。
- 36 个场景 JSON：必须 `accepted=true` 且 `monitor.valid=true`；每个应有 5 次样本、固定 token 输入和 NVML 压缩轨迹。
- `validation/`：四模型单元测试 gate、独立数值校验 gate 和报告，均应通过。

退出码 75 是资源等待；独立模型/程序失败会停止队列，需要修复后再跑。被污染的旧场景放在 `attempts/`，不能计入有效统计。S0 冻结程序和项目内动态库在 `.artifacts/performance/<阶段>/`，后续代码改动不会替换它们。

## 后续三个特性：必须按顺序做

**S0 完整数据采齐后才开始 S1。** 每阶段先提交代码与正确性结果，再冻结本阶段的程序，空卡跑同一矩阵，记录正收益和退化。

1. **S1 批量 prefill**：当前 `source/runtime/generation.cpp` 对每个 prompt token 分别执行模型。实现真正的 token 批量 QKV/FFN GEMM 与因果 Attention，保留逐 token 参考模式；逐步/最终 logits、greedy token、长短 prompt 和 GQA 都要对齐。主要观察 TTFT、prefill tokens/s、瞬时显存。
2. **S2 CUDA Graph + 算子融合**：使用稳定输入/输出地址、设备端 position/token，处理 embedding 临时分配和采样同步等 capture 障碍；Graph 与 fusion 要能独立开关并做消融。区分首次捕获成本和预热后 decode，观察 TPOT、提交次数和端到端时间。
3. **S3 分页 KV Cache**：实现真实页表、按需物理页分配、Attention 寻址、扩展和释放，保留连续 KV 参考模式。验证随机页映射、边界、重复请求和显存回收；报告实际/有效 KV 显存与吞吐，不把连续大块内存改个索引就称为分页缓存。

每阶段沿用同一个 LLM_RUN_ID，例如 S1-$LLM_RUN_ID、S2-$LLM_RUN_ID、S3-$LLM_RUN_ID。实现完成后可使用：

```bash
# 以下仅在实际实现并校验该特性后执行
python scripts/freeze_stage.py --stage "S1-$LLM_RUN_ID"
python scripts/benchmark_suite.py --stage "S1-$LLM_RUN_ID" --gpu-uuid GPU_UUID --wait-seconds 60
python scripts/compare_stages.py --baseline "$LLM_BASELINE_STAGE" \
  --stage "S1-$LLM_RUN_ID" --gpu-uuid GPU_UUID
```

将 GPU_UUID 替换为 S0 实际 GPU UUID；不能跨卡拼出严格配对的加速比。若换了卡，要在该空卡重放冻结 S0 后再与新阶段对比。比较工具要求完整矩阵、相同精度、模型哈希和实际 token 输入，生成 COMPARISON.md/JSON。

## 代码入口和已知约束

- `demo/main.cpp`：CLI、模型选择、重复/预热、固定 token 输入、JSON 和 logits 导出。
- `include/runtime/generation.h` / `source/runtime/generation.cpp`：生成循环和指标定义。
- `source/model/{llama3,qwen2,qwen3}.cpp`：模型组装、权重层、连续 KV Cache、前向。Llama2 和 Llama3 共用 LLama2Model，靠构建选项选择。
- `scripts/export_hf.py`：Qwen2/Qwen3/Llama 的原始 safetensors 转 FP32。Qwen3 的 attention_dim 与 hidden_size 不同；Q/K Norm 和 sidecar RoPE 都已适配。
- `source/op/kernels/{cpu,cuda}/`：算子；Llama3.2 的分段 RoPE 缩放在配置驱动缓存中实现。
- `scripts/empty_gpu.py` / `benchmark_suite.py` / `measurement_records.py`：空卡检查、测量、污染恢复与脚本哈希。
- `docs/IMPROVEMENTS.md`：原始候选亮点，当前授权的是前三个。

当前验证范围为 FP32、单请求、贪心解码与导出上下文长度；没有实现 continuous batching。INT8 路径保留上游接口但没有完整验证，Qwen3 INT8 明确拒绝。CPU 执行仍链接 CUDA runtime，因此也需要 Toolkit 才能编译。

## 给下一台机器上的代理的继续指令

> 先读 AGENTS.md 和 docs/HANDOFF.md。继续原目标：在本机器重建四模型 S0，全部 GPU 性能测试仅在空卡运行，完成 36 场景原始基线后依次实现批量 prefill、CUDA Graph/融合、分页 KV。每阶段冻结版本、校验正确性、跑同样矩阵、记录改善或退化。旧机有 23/36 部分 S0 数据，不能跨 GPU 拼基线；模型和编译产物未进 Git，需要按文档重建。不要把已有集成和上游功能宣称为新优化，也不要缩小到只跑小模型。

## 本机继续运行更新（2026-10-15）

用户最新明确允许 WSL `/Xwayland` 桌面进程。此本机例外优先于上文旧机的严格门槛。四模型固定资产、本机构建和 CPU 对照已通过；本机 S0 使用 `scripts/gpu_policies/wsl_desktop.json` 重新冻结并运行。所有外部计算进程仍禁止，所有阶段使用同一规则。当前工作目录、运行标识、快照和队列状态详见 [LOCAL_MACHINE.md](performance/LOCAL_MACHINE.md)。

本机继续进度：S0 和 S1 均已完成并审计 36/36；S1 的全部改善和退化见 [RESULTS.md](performance/stages/S1-Forward-20261015-163413-desktop/RESULTS.md)。当前工作为 S2，然后 S3；请读取本机 state.json 和 NEXT_STAGE_NOTES.md 继续。
