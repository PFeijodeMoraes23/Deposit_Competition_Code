"""
blp_1_logit_e8_coherence.jl
===========================
Thin entrypoint: run ONLY estimation routine E8 (Joint Single-Index kernel, demand_8_sikernel)
of the coherence logit build, then merge its results into the combined coherence summary.

Shares all logic with blp_1_logit_coherence.jl (included below). Spec 12 only.

Usage
-----
  julia --project=. --threads=auto blp_1_logit_e8_coherence.jl
"""

include(joinpath(@__DIR__, "blp_1_logit_coherence.jl"))

function main_e8()
    estim = ESTIM_STRATEGIES[findfirst(s -> s.id == 8, ESTIM_STRATEGIES)]
    println("=" ^ 70)
    println("  BLP Logit (Non-RC) — COHERENCE — single routine $(estim.label) — Spec $SPEC_ID")
    println("=" ^ 70)
    routine_results = run_strategy(estim)
    merge_into_combined_summary(routine_results)
    println("  [DONE] Coherence logit routine $(estim.label) complete.")
end

main_e8()
