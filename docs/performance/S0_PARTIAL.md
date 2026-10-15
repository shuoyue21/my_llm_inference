# 迁移前的 S0 实验记录

日期：2026-10-15。原始推理快照：`6d36c278612f6755f66973de4325db0cc064f44f`。测速工具的污染恢复修复另见 `46de40c`。**S1/S2/S3 三个优化特性均未实现。**

后台队列后来捕获到了空卡窗口，已完成四个模型的 GPU 数值校验和单元测试。GPU 校验的分词、逐步 logits 和 greedy token 全部通过，与 CPU 独立参考一致。

| 模型 | GPU 数值校验（最大 logit 绝对误差） | GPU 测试通过数 | 有效基线场景/要求场景 |
|---|---|---:|---:|
| tinystories15m | 通过，误差 1.67e-05 | 45 | 6/6 |
| qwen2.5-0.5b | 通过，误差 9.47e-05 | 45 | 10/10 |
| qwen3-0.6b | 通过，误差 4.48e-05 | 45 | 7/10 |
| llama3.2-1b | 通过，误差 2.31e-05 | 45 | 0/10 |

TinyStories 的 GPU 测试中有 1 项依赖 `LLAMA_TEST_MODEL` 环境变量的 CPU 模型测试被跳过；同一测试已经在 CPU 回归中执行通过，见 bringup/llama2_cpu_units.txt。

有效性能结果共 **23/36**，来自同一块 GPU `GPU-65eb0868-8c9c-735a-db98-49ecae46b4a4`。Qwen3 p512/g128 场景曾被外部计算任务打断，记录标为 `accepted=false`，不计入有效数据。其后重新等待空卡。按用户迁移指令，原机器队列已停止，停止前没有活动的测试子进程。

原始数据、固定 token 输入和压缩 NVML 轨迹保存在 [stages/S0](stages/S0/)，GPU 正确性证据在 [bringup](bringup/)。该 S0 还没有 `summary.json` 的 complete 状态，不能作为完整基线宣称已完成。

另一台机器必须使用独立阶段名重建并测完完整 S0，不能将不同 GPU UUID 的数据拼成优化前后比较。迁移步骤见 [HANDOFF](../HANDOFF.md)。
