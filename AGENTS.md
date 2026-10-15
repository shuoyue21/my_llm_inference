# Project handoff

Read `docs/HANDOFF.md` before continuing this project. It records the user's full objective, actual validation status and exact commands for another machine.

The completed local S0 remains frozen; S1 batched prefill is complete and current work is S2 CUDA Graph/operator fusion. The required order is: finish a complete empty-GPU baseline for all four registered models, then implement (1) batched prefill, (2) CUDA Graph plus operator fusion, (3) paged KV cache. Preserve each stage and record correctness and performance changes.

Use `scripts/build_variants.py` for portable builds and `scripts/download_models.py` for pinned model assets. Models, build directories, virtual environments and `.artifacts/` are intentionally excluded from Git; recreate them on the target machine.

All GPU performance measurements must pass `scripts/empty_gpu.py` and use `scripts/benchmark_suite.py`. Never stop another user's GPU process. Compare complete workloads on the same GPU UUID with the same model hashes, precision and token IDs. Keep rejected/contaminated attempts as evidence; do not count them as valid data.

The old machine's S0 results are partial (23/36) and its queue was stopped for migration. Use a new machine-specific stage name and do not overwrite published manifests. See `scripts/model_registry.json` for the full required workload matrix.

## Local user instruction, 2026-10-15

The user explicitly permits the existing WSL `/Xwayland` graphics process. On this local machine freeze every S0/S1/S2/S3 stage with `--gpu-policy scripts/gpu_policies/wsl_desktop.json`. This exception is graphics-only: all external compute clients and other non-exempt graphics clients remain forbidden. With the approved desktop present, aggregate memory/utilization counters are recorded rather than used as idle rejection thresholds. Freeze and compare the same explicit policy for every paired stage. The previous strict-policy local queue was stopped by us; its snapshot has zero accepted cases and is preserved. See `docs/performance/LOCAL_MACHINE.md` and `docs/performance/local-machine/state.json` for the current stage.

Local S0 `S0-Forward-20261015-163413-desktop` is now complete and audited (36 cases / 180 samples on the same GPU). See its `audit.json` and `BASELINE.md`; continue with S1 batched prefill, then S2, then S3. Do not rerun or replace this completed baseline unless its frozen artifacts fail verification.

Local S1 `S1-Forward-20261015-163413-desktop` is complete: four models, 28 full-logit correctness cases, zero-error batch memcheck, and 36 performance cases / 180 samples audited and paired with S0. See its `RESULTS.md`, `comparison.json`, and `audit.json`. Continue S2 (independent graph/fusion switches and ablations), then S3 true paged KV.
