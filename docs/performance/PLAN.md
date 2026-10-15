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

TinyStories 15M (Llama2-style), Qwen2.5-0.5B-Instruct, Qwen3-0.6B, Llama-3.2-1B-Instruct. Fix model hashes, precision, GPU UUID, context length, prompt token IDs, generated token budget, build flags and CPU threads. Cover short/medium/long prompts and short/long decode; run 1 warmup plus repeated measured samples, retain raw data, medians and latency quantiles. Report load, TTFT, prefill time/throughput, TPOT/decode throughput, total latency and GPU memory, with scope of every metric specified.

## Empty GPU gate

Use a GPU with no compute or graphics clients and negligible baseline device memory/utilization for consecutive samples. Set CUDA_VISIBLE_DEVICES to its UUID. Check all clients before, throughout and after each measurement. A foreign client invalidates the sample; never stop another user's process. Save gate/monitor telemetry alongside data. A cooperative lock does not substitute for actual GPU checks.

## Current status

Model bring-up and benchmark harness are in progress. All 8 GPUs were occupied at first inspection; no empty-GPU baseline has been measured. Feature work starts only after S0 performance data is complete.
