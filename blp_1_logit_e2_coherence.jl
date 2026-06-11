"""
blp_1_logit_e2_coherence.jl
===========================
Thin entrypoint: run ONLY estimation routine E2 (Pooled B+D Linear, demand_2) of the
coherence logit build, then merge its results into the combined coherence summary.

Shares all logic with blp_1_logit_coherence.jl (included below). Run all three routines
at once with `julia blp_1_logit_coherence.jl` instead.

Usage
-----
  julia --project=. --threads=auto blp_1_logit_e2_coherence.jl
"""

include(joinpath(@__DIR__, "blp_1_logit_coherence.jl"))

function main_e2()
    estim = ESTIM_STRATEGIES[findfirst(s -> s.id == 2, ESTIM_STRATEGIES)]
    println("=" ^ 70)
    println("  BLP Logit (Non-RC) — COHERENCE — single routine $(estim.label) — Spec $SPEC_ID")
    println("=" ^ 70)
    routine_results = run_strategy(estim)
    merge_into_combined_summary(routine_results)
    println("  [DONE] Coherence logit routine $(estim.label) complete.")
end

main_e2()
