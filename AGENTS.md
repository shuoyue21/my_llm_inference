# Project handoff

Read `docs/HANDOFF.md` and `docs/PERFORMANCE_RESULTS.md` before changing this project. Raw experiment data in `docs/performance/` is local and may be absent in a fresh clone.

## Completed objective

The user-requested four-model sequence is complete: local S0 original baseline, S1 true batched prefill, S2 independent CUDA Graph/operator fusion, S3 actual paged KV. Every primary stage contains the complete 36-case registry on the same GPU UUID with identical fixed FP32 models and token IDs. S2 additionally has three full ablations; S3 has a full dense reference control. `scripts/audit_project.py` has revalidated all selected raw data: 288 cases / 1440 measured samples. A further 72-case r2 experiment and its regressions remain preserved.

Selected inference source revisions: S0 `f36fa56`, S1 `fffbc89`, S2 `35af606`, S3 `0aff906`. The r2 addressing experiment `cab953e` was not adopted because 34/36 paged decode cases regressed relative to the initial paging version. Current C++ implementation and build programs/private libraries have been restored to the selected S3 version; frozen artifacts remain immutable in `.artifacts/performance/`.

## Workspace and retained evidence

Active worktree: `/home/forward/mycode/my_llm_inference_handoff`, branch `codex/local-staged-inference-20261015`. Original `/home/forward/mycode/my_llm_inference` main and its uncommitted user changes remain untouched.

Keep existing stage manifests, raw outputs, tokens, NVML traces, failed/contaminated attempts and r2 negative results. Keep `/docs/performance/` ignored and untracked. Never commit experiment JSON, logs, NVML traces, temporary state or failed-attempt outputs. Commit source, tests, reproducible configuration and the final human-readable summary only; do not create result-only checkpoint commits. Models, builds, virtual environment and `.artifacts/` also remain local. The final raw-data audit and export are `docs/performance/FINAL_AUDIT.json` and `ALL_STAGE_METRICS.csv`.

## Reproduction rules

Use `scripts/build_variants.py` for portable builds and `scripts/download_models.py` for fixed-revision/hash-verified assets. Registered scope is all four models and the full workload matrix in `scripts/model_registry.json`. Never combine the old machine's partial 23/36 data with this GPU's baseline.

All GPU performance experiments must pass `scripts/empty_gpu.py` and use `scripts/benchmark_suite.py`; never stop another user's task. The user explicitly permits local `/Xwayland` graphics residency. Freeze every paired local stage with `--gpu-policy scripts/gpu_policies/wsl_desktop.json`; the exception is graphics-only and never permits external compute clients. Keep the policy, GPU UUID, model hashes, precision, input IDs, warmups and repeats identical across comparisons.

Paged KV uses separately allocated CUDA pages, a stable device translation table, on-demand growth and a reusable physical pool. Request release clears mappings and returns pages to the pool; explicit `Model::trim_kv_cache()` performs actual driver frees. Keep both native per-token and continuous-KV reference modes and all Graph/fusion switches functional. No continuous batching or shared-prefix serving scheduler is claimed.

Verify pinned model assets when downloading them or when there is evidence they changed. Do not repeatedly hash unchanged multi-GB models or rerun full audits for Git/documentation cleanup.
