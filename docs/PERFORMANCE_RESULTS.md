# 四模型性能结果

同一 RTX 5070 Ti（`GPU-c8c002a0-861e-e0e3-3b8d-138381f9f61d`）完成 S0–S3。四个固定 FP32 模型为 TinyStories 15M、Qwen2.5-0.5B-Instruct、Qwen3-0.6B、Llama3.2-1B-Instruct。旧机器的 23 项结果未混入比较。

每模型测试 prompt 8/32/128/512/1536 与生成 32/128，按固定上下文排除越界组合，共 36 场景。每场景同一驻留模型预热 2 次、正式采样 5 次；固定 token 输入、忽略 EOS。模型版本与矩阵在 `scripts/model_downloads.json` 和 `scripts/model_registry.json`。

| 阶段 | 实现 | 完整场景 | 观测结果 |
|---|---|---:|---|
| S0 | 原始单 token prefill、连续 KV | 36 | 本机独立原始基线 |
| S1 | 批量 QKV/FFN GEMM、因果 GQA | 36 | 相对 S0，TTFT 加速 3.42–84.82 倍；TinyStories P8/G32 总延迟增加 27.66% |
| S2 | CUDA Graph、算子融合，独立开关 | 144 | 主矩阵加三套消融；相对 S1，主矩阵 decode 加速 1.05–8.03 倍 |
| S3 | 真实物理页、设备页表、按需扩展、复用与 trim | 72 | 分页与同版本连续缓存对照；KV 容量节省 0–97.66%；相对 S2，33/36 项 decode 退化，最大 26.54% |

选定版本共 288 场景、1440 次正式采样；四阶段主矩阵均为 36/36。另一次分页寻址实验 r2 完成 72 场景、360 样本，但相对初版分页有 34/36 项 decode 退化，未采用。当前实现保留初版分页。KV 节省指实际页池容量相对连续 KV 容量，不代表整卡显存同比下降。

四模型独立 CPU/CUDA 数值对照通过。S1、S2、S3 的完整 logits/greedy 对照分别为 28、112、224 项；新增 CUDA 路径专项 memcheck 为 0 errors。测试覆盖随机页映射、因果掩码、跨页 Graph、变长重复请求、复用和实际物理回收。

运行标识为 `Forward-20261015-163413-desktop`。用户允许 `/Xwayland` 图形驻留，所有阶段统一使用 `scripts/gpu_policies/wsl_desktop.json`，没有终止其他人的任务。

Git 只保存源码、测试、复现配置与本文汇总。逐场景 JSON、stdout/stderr、tokens、NVML 轨迹、失败尝试、完整 CSV 和审计输出均留在被忽略的 `docs/performance/`；冻结二进制与动态库在 `.artifacts/performance/`，正确性原始 logits 在 `.artifacts/validation/`。这些目录不随克隆分发。冻结记录中的旧源码版本号保留用于本机追溯，不因提交整理而改写原始实验记录。
