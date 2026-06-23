"""
blp_2_rc_e7_coherence.jl
========================
Coherence RC-BLP — routine E7 (Joint Single-Index, sieve), specification 12. One GPU job.

Reads  demand_7_sijoint_spec_12.parquet
Warm-start  BLP_RESULTS/logit_delta_E7_spec_12_coherence.bin (logit-coherence output)
Runs the full sigma -> … -> extended RC sequence on the GPU.

  julia --project=. --threads=auto blp_2_rc_e7_coherence.jl --hpc --R 2000
"""
const _USER_ARGS = copy(ARGS)
include(joinpath(@__DIR__, "blp_2_rc_coherence.jl"))
run_coherence_routine(7; passthrough = _USER_ARGS)
