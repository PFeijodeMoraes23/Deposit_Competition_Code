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

using DataFrames, Statistics

"""
    pix_coefficient(estim, spec; suffix) -> Float64

Return the estimated sleepiness coefficient on `pix_exists` (Υ_pix) for the routine.
NOT YET WIRED — needs the sleep step to persist Υ (see module docstring).
"""
function pix_coefficient(estim::Int, spec_id::Int; suffix::String="")
    error("CF4 needs Υ_pix: have estimation_1_sleep.py persist the φ control-function " *
          "coefficients (incl. pix_exists), then load it here. See cf_4_pix.jl docstring.")
end

"""
    cf4_pix_reallocation(ctx, st; upsilon_pix) -> NamedTuple

Force pix_exists=0 in φ, re-simulate deposits at observed spreads, and return the
per-institution volume reallocation vs the observed (φ̂) deposits.
"""
function cf4_pix_reallocation(ctx::CFDemandCtx, st::DepositSimState; upsilon_pix::Float64)
    pix, pc = _first_present(ctx.df, ["pix_exists", "pix_active"]; default=0.0)
    phi_cf  = clamp.(st.phi .- upsilon_pix .* pix, 0.0, 0.999)   # pre-Pix regime
    sim_obs = simulate_deposits(ctx, st; T=1)                    # observed φ̂ (1-step) — or T as needed
    sim_cf  = simulate_deposits(ctx, st; T=1, phi_override=phi_cf)
    dvol    = sim_cf.Dep[:, end] .- sim_obs.Dep[:, end]
    firm    = string.(ctx.df.CodConglomeradoPrudencial)
    return (firm=firm, is_B=st.is_B, dep_type=st.dep_type,
            dep_obs=sim_obs.Dep[:, end], dep_cf=sim_cf.Dep[:, end], dvol=dvol)
end

function main_cf4()
    a = _parse_cf_args()
    ctx = build_cf_context(a["estim"], a["spec"], a["stage"];
                           R=a["R"], seed=a["seed"], hpc=a["hpc"],
                           local_dir=a["local-dir"], suffix=a["suffix"])
    st  = load_sim_state(ctx)
    υ   = pix_coefficient(a["estim"], a["spec"]; suffix=a["suffix"])   # fails until wired
    res = cf4_pix_reallocation(ctx, st; upsilon_pix=υ)
    log_status("  [CF4] total Δvolume (no-Pix vs observed) = $(round(sum(res.dvol), sigdigits=4))")
    log_status("[DONE] cf_4_pix (descriptive Pix reallocation)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_cf4()
end
