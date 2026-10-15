# Project handoff

Read `docs/HANDOFF.md` before continuing this project. It records the user's full objective, actual validation status and exact commands for another machine.

Current inference remains the unoptimized S0 baseline. The required order is: finish a complete empty-GPU baseline for all four registered models, then implement (1) batched prefill, (2) CUDA Graph plus operator fusion, (3) paged KV cache. Preserve each stage and record correctness and performance changes.

Use `scripts/build_variants.py` for portable builds and `scripts/download_models.py` for pinned model assets. Models, build directories, virtual environments and `.artifacts/` are intentionally excluded from Git; recreate them on the target machine.

All GPU performance measurements must pass `scripts/empty_gpu.py` and use `scripts/benchmark_suite.py`. Never stop another user's GPU process. Compare complete workloads on the same GPU UUID with the same model hashes, precision and token IDs. Keep rejected/contaminated attempts as evidence; do not count them as valid data.

The old machine's S0 results are partial (23/36) and its queue was stopped for migration. Use a new machine-specific stage name and do not overwrite published manifests. See `scripts/model_registry.json` for the full required workload matrix.
