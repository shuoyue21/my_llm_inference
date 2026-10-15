# 完成范围与实测记录

本轮日期：2026-10-04。工作分支：`codex/complete-inference`。参考仓库通过重新 `git clone` 成功获取，固定为 `83030c894da046902c8134110eb70ed59cb49486`。

## 已完成

- 保留原有 Tensor/Buffer/Layer，接入模型配置/权重映射、Llama/Qwen 模型组装、Attention/GQA、KV Cache、Argmax。
- 新增独立 generation API 和 CLI，支持 CPU/CUDA、prompt、生成长度、EOS、上下文检查、JSON 和逐步 logits 导出。
- 修复 Tensor reshape/reset、CPU Matmul、embedding token 边界、BPE 单 token 解码与原始字节处理、Unicode 预分词连续空白、CUDA RoPE 边界、Argmax 小词表/异步同步/缓存复用、Matmul scale 与非对齐行、Add launch 维度、SwiGLU 部分 block 等问题。
- 构建按模型变体隔离，恢复可执行算子测试，新增因果 GQA、重复请求、形状和生成边界测试。
- 新增无 transformers 依赖的 Qwen safetensors 导出、固定版本下载、独立 PyTorch/官方 tokenizer 对照、可复现性能测量。

## 验证范围

| 路径 | 结果 |
|---|---|
| TinyStories 15M，Llama2 风格 FP32 CPU/CUDA | 真实生成、逐步 logits、生成 token 对齐通过 |
| Qwen2.5-0.5B-Instruct，FP32 CPU/CUDA | 原始 BF16 权重转 FP32，中文聊天/EOS、原始权重 logits 与官方分词对齐通过 |
| Llama3 / Qwen3 | 上游实现与构建入口已接入，未用真实权重完成端到端验证 |
| INT8 | 保留上游接口，不作为本次已验证能力；Qwen3 INT8 明确拒绝 |
| 并发服务 / continuous batching / 分页 KV | 尚未实现，属于后续改进方向 |

CPU 后端执行仍链接 CUDA runtime，编译需要 CUDA Toolkit。模型当前按编译选项选择；不要混用不同家族的文件格式。15M 模型只用于快速正确性验证，不能替代大模型性能测试。

## 正确性

Llama 构建的 45 项测试全部通过，Qwen 构建的 44 项测试全部通过（Llama 专用的真实模型重复请求测试不加入 Qwen 构建）。默认构建 CTest 的 3 个条目也全部通过，其中含依赖 Armadillo 的 smoke test。端到端 smoke 覆盖真实文本、CPU/CUDA token 一致性、BOS-only prompt、零生成、非法参数、上下文溢出、截断及非法维度模型文件。

| 对照路径 | forward 位置数 | 最大绝对 logit 误差 | 生成 token |
|---|---:|---:|---|
| llama_cpu | 12 | 3.1471252e-05 | 一致 |
| llama_cuda | 12 | 1.6689301e-05 | 一致 |
| qwen2_cpu | 19 | 0.0001783371 | 一致 |
| qwen2_cuda | 19 | 9.4652176e-05 | 一致 |

Llama 误差阈值 `atol=rtol=3e-4`；Qwen 使用 `5e-4`。每个位置比较整个词表，而非只看最终文本。Qwen 的计算参考从官方原始 safetensors 读取权重，能够同时覆盖自定义导出布局；分词同时对照官方 tokenizer.json。6 项自动化分词样例及另 8 项 Unicode/空白/缩写/特殊 token 样例一致。没有执行全量困惑度、长上下文或任意模型兼容性验证。

Compute Sanitizer 对 5 项 CUDA Argmax/RoPE/GQA 测试执行 memcheck，报告 `0 errors`。这不是所有 CUDA 路径的完整 sanitizer 覆盖。

原始证据在 [validation](validation/)：`ctest.txt`、`unit_llama.txt`、`unit_qwen2.txt`、`smoke.txt`、`cuda_memcheck.txt`、各后端 JSON 与 `model_manifest.json`。

## 性能基线

本机 NVIDIA GeForce RTX 4090，CUDA 12.1，Release，FP32，单请求；CPU 的 OpenBLAS/OMP 线程数为 1。每个后端先执行 1 次预热请求，随后 3 次独立进程请求取中位数。模型加载独立统计，TTFT 包含分词和 prompt forward；decode 吞吐排除首 token。当前是共享服务器，GPU 负载可能波动。

| 模型 | 后端 | prompt token | 实际生成 token | TTFT ms（中位） | decode token/s（中位） |
|---|---|---:|---:|---:|---:|
| TinyStories 15M | cpu | 5 | 64 | 116.981 | 47.23 |
| TinyStories 15M | cuda | 5 | 64 | 3.067 | 2292.83 |
| Qwen2.5-0.5B-Instruct | cuda | 28 | 41 | 96.375 | 277.58 |

Qwen 的实际生成数包含 EOS；EOS 不出现在展示文本。不同模型的 throughput 不能互相当作优化前后对比。原始采样见 [小模型基线](benchmark_baseline.json) 和 [Qwen 基线](benchmark_qwen2.json)。这些数字是当前实现的 baseline，尚无新增优化的加速比结论。

## 使用与后续

运行命令见 [README](../README.md)。候选改进与验收指标见 [IMPROVEMENTS](IMPROVEMENTS.md)。本轮集成与修复为后续实验提供基础；简历应如实标注基于 KuiperLLama，并将自己的优化与上游能力区分。
