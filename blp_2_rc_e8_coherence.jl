"""
blp_2_rc_e8_coherence.jl
========================
Coherence RC-BLP — routine E8 (Joint Single-Index, kernel), specification 12. One GPU job.

Reads  demand_8_sikernel_spec_12.parquet
Warm-start  BLP_RESULTS/logit_delta_E8_spec_12_coherence.bin (logit-coherence output)
Runs the full sigma -> … -> extended RC sequence on the GPU.

  julia --project=. --threads=auto blp_2_rc_e8_coherence.jl --hpc --R 2000
"""
const _USER_ARGS = copy(ARGS)
include(joinpath(@__DIR__, "blp_2_rc_coherence.jl"))
run_coherence_routine(8; passthrough = _USER_ARGS)
