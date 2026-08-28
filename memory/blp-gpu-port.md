---
name: blp-gpu-port
description: BLP GPU estimation (blp_1/blp_2 _gpu.jl) — on-device SQUAREM rewrite, Float32 1/R precision bug, verify_gpu_shares guard, Bouchet deploy/verify workflow
metadata:
  type: project
---

BLP demand estimation has GPU variants `blp_1_estimation_gpu.jl` (numerical-gradient) and `blp_2_estimation_gpu.jl` (IFT analytical gradient). blp_2 includes blp_1's GPU file, which guards its include of `blp_engine_cpu.jl` with `if !isdefined(Main, :X_COLS)` to avoid double-load.

Work done 2026-06-02:
- **On-device SQUAREM**: rewrote `blp_contraction_gpu!` to keep δ, the contraction map, norms, and acceleration all on-device (new helper `model_shares_dev!` with pre-gathered `mu_B`/`mu_D`). Fixes YCRC "you did not use the GPU" emails — previously per-step CPU↔GPU sync left H200 at ~0% sampled utilization. `compute_model_shares_gpu!` kept as-is for blp_2's IFT forward passes (they need `buf.s_B`/`s_D` on CPU).
- **Float32 1/R precision bug**: `GPU_T(1f0 / R)` divided in Float32 → ~3.6e-9 relative error on every share → constant `norm=3.627e-9` SQUAREM floor, above `tol_inner=1e-10`, so inner loop ran all 5000 iters and never converged (blp_1 hit the day time limit; blp_2 then threw `SingularException` from the rank-deficient θ₁ projection). Fixed to `GPU_T(1.0 / R)`. The earlier "it converged" runs were the CPU path (`s / R`, Float64).
- **gmm_fg_gpu! arg bug**: blp_2 called it with a spurious leading `1.0` (16 args vs 15 params). Removed.
- **verify_gpu_shares guard**: both runners now compare CPU vs GPU shares at the warm-start δ before optimizing, erroring if max rel diff > `rtol=args["tol_inner"]`. Catches precision regressions (like 1f0/R) loudly instead of as silent non-convergence. Lesson: in this GPU code, scaling factors must use Float64 literals (`1.0`), not Float32 (`1f0`); clamp-floor literals (`1f-15`, `1f-30`) are fine since they don't systematically scale results.

Bouchet workflow: production on `--partition=gpu_h200 --gpus=h200:1`; interactive smoke test needs `salloc -p gpu_devel --gpus=1` or `--qos=normal` (plain salloc on gpu_h200 errors "Invalid qos"). Verify with `--dry-run` (times 10 contraction iters + IFT passes) and `jobstats <id>` (GPU util should be clearly off 0%). Sanity-check GPU `theta1`/`Q` against the CPU result for the same E/spec. Related: [[panel-2025-extension]].
