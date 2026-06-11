"""
blp_2_rc_e1_coherence.jl
========================
Coherence RC-BLP — routine E1 (Local B-type), specification 12. One GPU job.

Reads  demand_1_final_spec_12.parquet
Warm-start  BLP_RESULTS/logit_delta_E1_spec_12_coherence.bin (logit-coherence output)
Runs the full sigma -> … -> extended RC sequence on the GPU (IFT engine by
default; set BLP_COHERENCE_ENGINE=numerical for the finite-difference engine).

Forwards engine flags, e.g.:
  julia --project=. --threads=auto blp_2_rc_e1_coherence.jl --hpc --R 2000
  julia --project=. blp_2_rc_e1_coherence.jl --dry-run
"""
const _USER_ARGS = copy(ARGS)
include(joinpath(@__DIR__, "blp_2_rc_coherence.jl"))
run_coherence_routine(1; passthrough = _USER_ARGS)
