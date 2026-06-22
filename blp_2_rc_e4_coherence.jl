"""
blp_2_rc_e4_coherence.jl
========================
Coherence RC-BLP — routine E4 (Pooled Constrained Linear), specification 12. One GPU job.

Reads  demand_4_constrained_spec_12.parquet
Warm-start  BLP_RESULTS/logit_delta_E4_spec_12_coherence.bin (logit-coherence output)
Runs the full sigma -> … -> extended RC sequence on the GPU (IFT engine by
default; set BLP_COHERENCE_ENGINE=numerical for the finite-difference engine).

Forwards engine flags, e.g.:
  julia --project=. --threads=auto blp_2_rc_e4_coherence.jl --hpc --R 2000
  julia --project=. blp_2_rc_e4_coherence.jl --dry-run
"""
const _USER_ARGS = copy(ARGS)
include(joinpath(@__DIR__, "blp_2_rc_coherence.jl"))
run_coherence_routine(4; passthrough = _USER_ARGS)
