"""
cf_4_pix.jl
===========
CF4 — Pix as a switching function (DESCRIPTIVE, volume reallocation).

Idea: Pix lowered account-switching frictions. In this model Pix enters the
SLEEPINESS function φ(S_mt) = Υ'·S_mt via the `pix_exists` state (it does NOT enter
the logit mean utility X_COLS; under RC it also enters demographic interactions).
The descriptive counterfactual sets the regime to "no Pix" (`pix_exists = 0`),
recomputes φ, re-simulates the deposit law of motion at observed spreads, and
measures the volume REALLOCATION (which institutions gain/lose deposits) — no costs
and no equilibrium re-solve needed (the user's CF4: "requires volume reallocation —
no need for decomposition").

  φ_cf_mt   = φ̂_mt − Υ_pix · pix_exists_mt          (force the pre-Pix regime)
  Dep_cf    = simulate_deposits(ctx, st with φ=φ_cf, spreads = ρ̂)   [cf_0_deposit_sim]
  Δvolume_j = Dep_cf_j − Dep_obs_j                                  (per institution)

⚠ DEPENDENCY (why this is a scaffold, not yet runnable): it needs the estimated
sleepiness coefficient on `pix_exists` (Υ_pix). The sleep step
(`estimation_1_sleep.py`, control-function regression with `s_base = ['constant',
'pix_exists', …]`) produces φ̂_mt but does NOT currently persist Υ. To run CF4:
  (1) have the sleep step save its Υ vector (or the φ control-function model) to a
      file keyed by routine/spec, OR
  (2) recover Υ_pix by re-running the φ regression here from the demand-prep inputs.
Then fill in `pix_coefficient(...)` below and the rest executes on the existing
cf_0_deposit_sim engine (which already has the bps / accrual-stability / per-type
d̄ fixes from CF1).

This file deliberately does the safe parts (load context, locate pix/phi columns)
and STOPS at the Υ_pix hookup with a clear error, so running it fails loudly rather
than silently producing wrong numbers.
"""

include(joinpath(@__DIR__, "cf_0_deposit_sim.jl"))

using DataFrames, Statistics, Printf

"""
    pix_coefficient(cf_dir, estim, spec) -> Float64

Return the estimated sleepiness coefficient on `pix_exists` (Υ_pix) for the routine,
read from `CF_FOUNDATION/upsilon_pix_E{e}_spec_{s}.json` (recovered read-only from the
sleep pickle by `cf_4_upsilon_export.py` — see that script and the module docstring).
"""
function pix_coefficient(cf_dir::String, estim::Int, spec_id::Int)::Float64
    f = joinpath(cf_dir, "upsilon_pix_E$(estim)_spec_$(spec_id).json")
    isfile(f) || error("CF4 needs Υ_pix. Run first:  " *
                       "python cf_4_upsilon_export.py --estim $estim --spec $spec_id   (missing $f)")
    m = match(r"\"upsilon_pix\"\s*:\s*(-?[0-9.eE+]+)", read(f, String))
    m === nothing && error("upsilon_pix not found in $f")
    return parse(Float64, m.captures[1])
end

"""
    cf4_pix_reallocation(ctx, st; upsilon_pix) -> NamedTuple

Force pix_exists=0 in φ, re-simulate deposits at observed spreads, and return the
per-institution volume reallocation vs the observed (φ̂) deposits.
"""
function cf4_pix_reallocation(ctx::CFDemandCtx, st::DepositSimState; upsilon_pix::Float64, T::Int=8)
    pix, pc = _first_present(ctx.df, ["pix_exists", "pix_active"]; default=0.0)
    phi_cf  = clamp.(st.phi .- upsilon_pix .* pix, 0.0, 0.999)   # pre-Pix regime (drop the Pix term)
    sim_obs = simulate_deposits(ctx, st; T=T)                    # observed φ̂
    sim_cf  = simulate_deposits(ctx, st; T=T, phi_override=phi_cf)
    dvol    = sim_cf.Dep[:, end] .- sim_obs.Dep[:, end]
    firm    = string.(ctx.df.CodConglomeradoPrudencial)
    return (firm=firm, is_B=st.is_B, dep_type=st.dep_type, phi=st.phi, phi_cf=phi_cf, pix=pix,
            dep_obs=sim_obs.Dep[:, end], dep_cf=sim_cf.Dep[:, end], dvol=dvol)
end

function main_cf4()
    a = _parse_cf_args()
    ctx = build_cf_context(a["estim"], a["spec"], a["stage"];
                           R=a["R"], seed=a["seed"], hpc=a["hpc"],
                           local_dir=a["local-dir"], suffix=a["suffix"])
    st  = load_sim_state(ctx)
    _, _, out_dir = get_paths(a["hpc"]; local_dir=a["local-dir"])
    cf_dir = joinpath(dirname(out_dir), "CF_FOUNDATION")
    υ   = pix_coefficient(cf_dir, a["estim"], a["spec"])
    T   = 8   # medium-run (2y) reallocation horizon
    log_status("  [CF4] Υ_pix=$(round(υ, sigdigits=4)) | horizon=$T")
    res = cf4_pix_reallocation(ctx, st; upsilon_pix=υ, T=T)
    log_status("  [CF4] φ_cf ≤ φ̂ for $(round(100*mean(res.phi_cf .<= res.phi .+ 1e-12), digits=1))% of obs " *
               "(mean φ̂=$(round(mean(res.phi), digits=3)) → φ_cf=$(round(mean(res.phi_cf), digits=3)))")

    # Per-obs export + firm/type summary.
    df = DataFrame(CodConglomeradoPrudencial=res.firm, is_B=res.is_B, deposit_type=res.dep_type,
                   pix_exists=res.pix, phi=res.phi, phi_cf=res.phi_cf,
                   dep_obs=res.dep_obs, dep_cf=res.dep_cf, dvol=res.dvol)
    out_path = joinpath(cf_dir, "cf4_pix_realloc_E$(a["estim"])_spec_$(a["spec"])_$(a["stage"])$(a["suffix"]).parquet")
    mkpath(cf_dir); Parquet2.writefile(out_path, df)

    tot_obs = sum(res.dep_obs); tot_cf = sum(res.dep_cf)
    @printf("\n  === CF4 Pix reallocation (descriptive, no-Pix vs observed, T=%d) ===\n", T)
    @printf("  Σ Dep obs=%.4g  no-Pix=%.4g  ΔΣ=%.4g (%.2f%%)\n",
            tot_obs, tot_cf, tot_cf - tot_obs, 100*(tot_cf-tot_obs)/max(abs(tot_obs),1e-12))
    for (lbl, mask) in (("B-firms", res.is_B), ("D-firms", .!res.is_B))
        any(mask) || continue
        @printf("    %-8s Δvolume=%.4g  (winners %d / losers %d)\n", lbl,
                sum(res.dvol[mask]), count(>(0), res.dvol[mask]), count(<(0), res.dvol[mask]))
    end
    for k in sort(unique(res.dep_type))
        m = res.dep_type .== k
        @printf("    k=%d      Δvolume=%.4g\n", k, sum(res.dvol[m]))
    end
    log_status("  [CF4] wrote $(basename(out_path))")
    log_status("[DONE] cf_4_pix (descriptive Pix reallocation)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_cf4()
end
