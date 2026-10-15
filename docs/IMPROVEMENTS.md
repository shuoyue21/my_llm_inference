# AI Infra 改进方向（待审阅）

本轮完成的是基于 KuiperLLama 的可运行、可验证推理基线。下表是后续候选工作，不代表已经实现，也不预先承诺加速倍数。工作量为熟悉当前代码、单人全职开发的粗略估计，包含基本正确性与性能验证；学习、调试和硬件差异会影响实际时间。

## 当前结构与瓶颈

- `source/runtime/generation.cpp`：单请求、自回归循环；prompt 的每个 token 分别执行整个网络，prefill 尚未批量化。生成指标区分 TTFT 和 decode，模型加载另计。
- `source/model/llama3.cpp`、`qwen2.cpp`：权重、算子和请求状态集中在同一个 Model；同一个实例暂不支持并发请求。FFN、QKV、Attention 通过独立算子调用。
- `Model::init_mem` / `slice_kv_cache`：为每层预分配最大上下文长度的连续 FP32 KV Cache。实际 token 很少时仍占用全长存储。
- `source/op/kernels/cuda/matmul_kernel.cu`：主要是逐 token GEMV；没有批量 GEMM 和 Tensor Core 路径。
- `source/op/kernels/cuda/mha_kernel.cu`：每个头计算整个历史序列的 scores，再 softmax 和加权求和；长序列时重复读写较多。
- 生成过程每步将采样结果返回 CPU；embedding 的 token ID 传输含同步。现在是便于验证的正确性基线，后续可以减少主机参与。

## 候选列表

| 优先级 | 方向 | 具体实现 | 验收证据 | 工作量 | 简历价值 |
|---|---|---|---|---|---|
| 1 | 批量 prefill + decode 分离 | 建立 `[tokens, hidden]` 张量路径，QKV/FFN 改为 GEMM，prefill 用因果 attention；保留单 token decode | prompt 长度 32/128/512/2048 的 TTFT、prefill tokens/s；对照逐 token logits | 5–10 天 | 高；能说明 GEMV/GEMM、算术强度和长 prompt 瓶颈 |
| 2 | CUDA Graph + 算子融合 | 持久化输入 token/position/采样缓冲；消除 capture 内主机同步和临时分配；捕获 decode 图；融合 residual+RMSNorm 或 bias+activation | Nsight 的 kernel launch 数、CPU launch 时间、TPOT；分别关闭 graph/fusion 做消融 | 4–8 天 | 高；范围可控，适合快速产生可信性能结果 |
| 3 | 分页 KV Cache | block allocator、逻辑到物理页表、引用计数、按需扩容与释放；attention 按页表寻址 | 并发/长度矩阵下有效 KV 占比、峰值显存、最大可容纳请求数；随机页表和释放测试 | 7–14 天 | 很高；内存管理和 attention 内核相结合 |
| 4 | Continuous batching + 调度 | 先把模型权重与 RequestState 拆开；每轮加入新请求、移除 EOS 请求；增加 token budget 与 chunked prefill | 到达率/输入输出长度变化下吞吐、p50/p95 TTFT/TPOT、完成率；对照静态 batch | 7–14 天 | 很高；与推理服务岗位最直接相关；依赖 batched kernel 和请求隔离 |
| 5 | 共享前缀缓存 | 在分页 KV 上增加前缀哈希、引用计数、LRU 淘汰和部分命中；缓存键包含模型与相关配置 | 0/50/90% 前缀命中率下 TTFT、命中率、复用 token 数；跨请求正确性与隔离测试 | 3–6 天，分页之后 | 高；适合系统提示词、重复文档/RAG 场景 |
| 6 | FP16/BF16 + W8A16/W4A16 | 先补 dtype/加载/转换和 FP16 GEMM/GEMV；再做分组量化、反量化融合或成熟库调用 | 同一模型的权重/峰值显存、decode 吞吐、logit 误差与困惑度/任务准确率 | 7–14 天起 | 高；上游已有 INT8 路径，必须明确自己增加的精度、内核和验证 |
| 7 | 分块 attention / online softmax | 分块遍历 K/V，维护在线最大值与分母；分别处理 prefill 与 decode，并对 GQA 复用 KV | 序列长度曲线、HBM 读写、临时 scores 内存、端到端 TTFT/TPOT；验证因果 mask 和不整齐头维度 | 7–14 天 | 高；不能把普通融合 attention 直接称作完整 FlashAttention 实现 |
| 8 | 推理服务与可观测性 | 在请求调度之上增加 HTTP/SSE、取消、超时、背压、队列长度与延迟指标 | 并发压测、客户端取消后的显存回收、排队时间/失败率 | 3–6 天，调度之后 | 中高；单独包 API 技术含量有限，与调度组合更有说服力 |
| 9 | Speculative decoding | draft/target 双模型、候选批量验证和拒绝后的 KV 回滚；先限定 greedy，再考虑随机采样 | 与 target-only 的 token 一致性、接受率、每轮接受 token 数、不同模型/负载下净收益 | 10–20 天 | 高但风险大；需要已完成 batched prefill，不建议作为第一项 |

## 建议审阅的两条路线

**快速形成一组可写成果：方向 1 + 2。** 先用 profile 确认长 prompt 与 decode 的瓶颈，再实现 batched prefill 和 CUDA Graph/融合。交付两个独立优化、关闭各优化的消融实验，以及 0.5B/更大模型上的对比。粗估 2–3 周。

**更贴近 AI Infra 核心岗位：方向 1 + 3 + 4，之后按时间加 5。** 交付一个支持并发请求的最小推理引擎，能解释请求生命周期、KV 页表、调度预算、吞吐与延迟权衡。粗估 4–6 周。不要同时启动所有方向。

## 实验和简历写法

本轮的 `scripts/benchmark.py` 和数值对照脚本作为后续基线。每次性能比较固定模型/权重精度、GPU、CUDA/编译选项、prompt token 数、输出 token 数、线程数；预热后多次测量并保留原始数据。并发实验另报告请求到达分布、排队时间和分位延迟。当前服务器由多人共享，一次测量的加速比不能直接作为稳定结论。

推荐表述模板（只有实验完成后填写数字）：

> 基于 KuiperLLama 构建 C++/CUDA 推理引擎，完成 Llama/Qwen 的数值对齐与端到端验证；设计并实现【自己的改进】，在【模型、硬件、输入输出长度、并发】下，将【TTFT/TPOT/显存/吞吐】从【A】改善到【B】，并通过【正确性测试与消融】验证。

上游已有的算子、INT8 和模型结构不能单独包装成原创亮点；项目说明保留上游来源，突出自己完成的设计、实现、测试和实验结论。

## 参考原始资料

- [PagedAttention 论文](https://arxiv.org/abs/2309.06180)：分页 KV 与 serving 内存管理。
- [NVIDIA CUDA Graphs](https://developer.nvidia.com/blog/cuda-graphs/)：执行图及减少 kernel 提交开销。
- [FlashAttention-2 论文](https://arxiv.org/abs/2307.08691)：attention 分块和并行组织。

上述资料用于确定技术方向；表中优先级与工作量是针对当前仓库的工程判断。
