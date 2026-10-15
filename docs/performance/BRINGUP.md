# S0 模型适配与测量准备

优化前代码最初快照：`c3344263e7d168d9a6e62d2c08d5abaf462c8cff`。

已下载并固定 Qwen/Qwen3-0.6B (`c1899de289a04d12100db370d81485cdf75e47ca`) 和 unsloth/Llama-3.2-1B-Instruct (`5a8abab4a5d6f164389b1079fb721cfab8d7126c`)。后者是公开可访问的 Llama3.2 指令模型分发，实验将明确记录这一来源。权重导出为 FP32，并保留原始 safetensors、配置及版本。

已通过 CPU 独立数值校验：Qwen3 的 Q/K Norm、非 hidden_size 的 attention_dim、分组注意力与 Llama3.2 的分段 RoPE 缩放均从原始模型配置/权重计算对照。分词和 greedy token 一致。原始报告见 bringup/*.json；这些是正确性报告，不是空卡性能数据。

新增的计时和重复运行选项只用于测量，三个优化特性均未实现。`--prompt-ids` 固定确切 token 序列，`--ignore-eos` 保证固定输出长度，`--warmup 2 --repeat 5` 在同一个已加载模型上测量。TTFT 包含分词和 prompt forward；prefill_ms 包含 prompt forward、末位置输出层和首 token 采样；decode_token_ms 是相邻生成 token 的主机可见间隔；模型加载单独报告。GPU 显存同时报告 CUDA 分配变化及 NVML 采样峰值，采样峰值不声称捕捉所有瞬时分配。

空卡工具使用 NVML v2 将驱动预留的约 632 MiB 与实际显存分开。要求连续 5 次采样没有外部计算进程、实际显存不超过 128 MiB、GPU 利用率不超过 1%。仅允许已确认的 `/usr/lib/xorg/Xorg` 图形驻留（最多 16 MiB；此机约 4.5 MiB）；所有其他图形应用和所有计算进程都拒绝。测量中每约 20 ms 检查进程和显存，保存实际最大采样间隔；发生外部任务或 NVML 查询失败则废弃整次场景。

`scripts/freeze_stage.py` 冻结推理程序、测试程序和核心动态库，记录代码版本、SHA256 与模型哈希。`scripts/benchmark_suite.py` 从冻结文件运行，结果按阶段、实验标签和 GPU UUID 分开保存；跨卡结果不能直接当作严格配对的改进。每次改进的对照要求同一 GPU UUID、模型哈希、token 输入、输出长度和精度；必要时在当前空卡重放 S0 冻结版本以建立配对。

原来的 docs/benchmark_baseline.json 和 docs/benchmark_qwen2.json 是历史共享卡结果，**不计入本次空卡基线**。当前所有 GPU 有其他计算进程，等待自动选择空卡后，先完成四模型 GPU 正确性校验与 S0 完整矩阵，再开始特性 1。

## 测量中断恢复

外部 GPU 任务或 NVML 查询错误导致的污染会返回资源等待状态（退出码 75），队列重新选择空卡。被污染的场景及 stdout/stderr/NVML 原始记录移入 `attempts/` 保留；有效数据不会被覆盖，模型自身执行失败也不会被当成资源等待而无限重试。

`source_revision` / `source_digest` 指向被冻结的推理版本。每条新测量额外记录 `harness.files` 中实际测速工具的 SHA256 和 `metric_definition_version`，因此测量工具的可靠性修复不会改变或冒充 S0 推理实现。S0 二进制、动态依赖和模型哈希保持不变。
