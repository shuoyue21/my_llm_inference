# Ordered implementation and empty-GPU measurement plan

Objective: validate Qwen3 and Llama3, then implement (1) batched prefill, (2) CUDA Graph and operator fusion, (3) paged KV cache, in that order. Preserve the full model/workload baseline before implementing feature 1, then measure and document every cumulative stage.

## Stages

- Initial snapshot: prior Llama2/TinyStories and Qwen2.5 integration, before Qwen3/Llama3 bring-up.
- S0: all four model families validated; no requested optimization enabled or implemented. Freeze the source revision and baseline executables.
- S1: batched prefill, with a token-by-token reference mode.
- S2: CUDA Graph decode and fused operators, with independent switches for ablation.
- S3: paged KV allocation and attention, with contiguous-KV reference mode.

Each stage must include correctness and empty-GPU benchmark evidence. Old docs/benchmark_*.json were collected on a shared GPU and are historical records only; they do not satisfy this goal.

## Models and workload matrix

TinyStories 15M (Llama2-style), Qwen2.5-0.5B-Instruct, Qwen3-0.6B, Llama-3.2-1B-Instruct. Fix model hashes, precision, GPU UUID, context length, prompt token IDs, generated token budget, build flags and CPU threads. Cover short/medium/long prompts and short/long decode; run 2 resident-model warmups plus 5 measured samples, retain raw data, medians and latency quantiles. Report load, TTFT, prefill time/throughput, TPOT/decode throughput, total latency and GPU memory, with scope of every metric specified.

## Empty GPU gate

Use a GPU with no other compute clients, negligible baseline device memory/utilization, and no active graphics applications for consecutive samples. NVML confirmed that every card has the same server `/usr/lib/xorg/Xorg` client using 4.5 MiB; this fixed graphics-only daemon (at most 16 MiB, zero idle GPU utilization) is the sole permitted background client and is retained in all telemetry. All compute clients, including idle ones, disqualify the GPU. Set CUDA_VISIBLE_DEVICES to its UUID. Check all clients before, throughout and after each measurement. A foreign client invalidates the sample; never stop another user's process. Save gate/monitor telemetry alongside data. A cooperative lock does not substitute for actual GPU checks.

## Current status

Qwen3-0.6B and Llama3.2-1B-Instruct pass CPU full-logit/greedy-token/tokenizer comparison against original HF weights. Max absolute errors are 1.335e-4 and 9.537e-5 respectively. All four build variants and the empty-GPU benchmark matrix are being finalized. All 8 GPUs have other compute clients; no empty-GPU baseline has been measured. The user confirmed that no card can currently be reserved and asked for automatic selection of whichever becomes empty. Feature work starts only after S0 performance data is complete.
