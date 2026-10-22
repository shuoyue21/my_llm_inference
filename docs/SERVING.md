# 第 4–7 项实现与复现

S4–S7 的主工作负载仍来自 `scripts/model_registry.json`，包含四模型全部 36 个输入/输出组合。性能汇总由 `scripts/summarize_extensions.py` 从已接受的原始记录生成；未完成采集时不会生成最终结论。

## Continuous batching

`runtime::Scheduler` 支持请求在轮次之间动态加入、取消和完成；`max_active` 限制驻留请求数，`max_pending` 限制等待队列，`token_budget` 限制每轮实际输入的 token 行数。长 prompt 按公平份额分块，decode 每轮每请求输入一行。完成请求释放 KV 映射，后续请求可以立即占用位置。

`CudaBatchBackend` 在同一轮中合并不同请求的 Q/K/V、输出投影及 FFN GEMM；RoPE/KV 准备和参考 Attention 也使用请求维度的批量 kernel，避免按行主机提交；prefill 中间行只更新 KV，只有最终 prompt 行和 decode 行投影全词表/采样（诊断回调保留全部 logits）；每个请求保留独立 KV 页表和位置，权重及 GEMM 工作区共享。调用者在一个 host 线程中交错调用 `submit`、`cancel`、`step` 和 `take_completed`。CLI 提供可复现的动态请求流。

```bash
build-qwen3/demo/llama_infer models/qwen3-0.6b/model.bin \
  models/qwen3-0.6b/tokenizer.json --device cuda \
  --serving continuous --concurrency 4 --requests 16 --token-budget 128 \
  --prompt 'Explain how memory works.' --max-new-tokens 32 --ignore-eos \
  --warmup 2 --repeat 5 --json
```

`--arrival-ms` 指定请求计划到达间隔，默认所有请求在实验开始时到达；延迟包括等待下一轮调度的时间。JSON 保留逐请求 token、TTFT、等待时间、总延迟和每 token 间隔，并输出吞吐及经验分位数。

## 共享前缀缓存

增加 `--prefix-cache-mib 512` 开启缓存，默认关闭。匹配同一后端实例内最长相同 token 前缀，只共享完整物理页，至少重新计算 prompt 最后一个 token。共享页通过引用计数保持存活，追加内容分配私有页；LRU 淘汰移除缓存引用，活跃请求仍持有的页继续有效。每个后端绑定一份模型权重和计算精度，缓存不跨模型/精度复用。

预算约束缓存条目的页租约字节总和；多个条目共享页时这是保守计数。活跃请求及可复用物理页池另占显存，LRU 预算不是整卡显存硬上限。请求释放返回物理页池；`PagedKVCache::trim()` 会保护仍被引用的共享页。后端拥有自己的页池，随 `CudaBatchBackend` 销毁释放；`Model::trim_kv_cache()` 仍控制原生单请求缓存。输出中记录命中、复用 token、淘汰和缓存容量。

## FP16/BF16 和 W8A16/W4A16

`--precision fp32|fp16|bf16|w8a16|w4a16` 选择 continuous 后端的权重/矩阵乘法路径。显式选择精度时从 CPU checkpoint 直接压缩上传，不在 GPU 保留一份原始 FP32 大矩阵。Embedding 和绑定的分类权重复用同一份存储。

FP16/BF16 使用低精度操作数、FP32 累加。W8/W4 使用每输出行、每 128 元素一组的对称量化，分别存储实际 int8 或两个 int4/nibble，加 FP32 scale；激活为 FP16。小批次直接读取 packed 权重，大批次临时展开当前矩阵为 FP16 使用 GEMM。该临时缓冲计入显存；规范化、残差、Attention 和 KV 保留 FP32。

加载/压缩成本单独计入 `backend_setup_ms`，正式吞吐使用驻留后端预热后采样。`scripts/verify_precision.py` 在完全相同文本 token 上保存 logits 并计算 NLL、KL、RMSE 和 top-1 一致率。这是小规模固定文本数值/语言模型损失探针，不等同于通用任务准确率或完整评测集。

## 分块 Attention

`--attention reference|online` 选择参考实现或 32-token 分块的 online softmax。Online 路径逐块更新最大值、归一化和及加权 V 累加值，不向全局内存写入整段序列的 scores，且一层的全部行/头共用一次 kernel 启动。支持 GQA、实际物理页和最长已导出上下文；head size 上限为 256。

原有单请求 native/reference、连续 KV、Graph/fusion 路径保留。Continuous 后端使用自身的合批执行循环，目前没有请求批大小动态变化的 CUDA Graph 捕获；Graph/fusion 开关仍控制原有单请求/串行路径。

## 性能数据与 Git

所有 GPU 验证使用 `scripts/run_on_empty_gpu.py`；性能矩阵使用 `scripts/benchmark_suite.py`。各阶段统一冻结 `scripts/gpu_policies/wsl_desktop.json`，只允许用户已授权的 `/Xwayland` 图形进程。

```bash
python scripts/run_serving_matrix.py --stage S4-STAGE --gpu-uuid GPU_UUID
python scripts/run_extension_matrices.py --feature prefix --stage S5-STAGE --gpu-uuid GPU_UUID
python scripts/run_extension_matrices.py --feature precision --stage S6-STAGE --gpu-uuid GPU_UUID
python scripts/run_extension_matrices.py --feature attention --stage S7-STAGE --gpu-uuid GPU_UUID
```

冻结程序、源码快照及私有库位于 `.artifacts/performance/`。原始性能、NVML、失败尝试和数值探针位于忽略的 `docs/performance/`。只提交实现、必要验证、复现配置及最终人类可读汇总。未变化的模型复用既有验证记录，资产变化时再验证，不对每个性能场景重新读取多 GB 模型做哈希。
