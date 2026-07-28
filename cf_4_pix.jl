"""
cf_4_pix.jl
===========
CF4 — Pix as a switching function (DESCRIPTIVE, volume reallocation).

Idea: Pix lowered account-switching frictions. In this model Pix enters the
SLEEPINESS function φ(S_mt) = G(S_mt′θ) via the `pix_exists` state (it does NOT enter
the logit mean utility X_COLS; under RC it also enters demographic interactions).
The descriptive counterfactual sets the regime to "no Pix" (`pix_exists = 0`),
recomputes φ, re-simulates the deposit law of motion at observed spreads, and
measures the volume REALLOCATION (which institutions gain/lose deposits) — no costs
and no equilibrium re-solve needed (the user's CF4: "requires volume reallocation —
no need for decomposition").

  φ_cf_mt   = G( S_mt′θ − θ_pix·pix_exists_mt )     (EXACT no-Pix φ under a nonlinear link G;
              built by phi_from_native with pix=0, loaded from the cf_4_upsilon_export.py parquet)
            = φ̂_mt − Υ_pix · pix_exists_mt          (identity-link fallback E1/E2 only; here
              Υ_pix is the AME, so subtracting it is a 1st-order approximation when G≠identity)
  Dep_cf    = simulate_deposits(ctx, st with φ=φ_cf, spreads = ρ̂)   [foundation_deposit_sim]
  Δvolume_j = Dep_cf_j − Dep_obs_j                                  (per institution)

DEPENDENCY: the no-Pix φ is prepared read-only by `cf_4_upsilon_export.py` (it does NOT touch
the sleep estimation). Keyed by routine/spec it writes:
  • `upsilon_pix_E{e}_spec_{s}.json`    — Υ_pix (the Pix AME; reporting + the identity-link
    fallback), read here by `pix_coefficient`;
  • `phi_nopix_E{e}_spec_{s}.parquet`   — the EXACT per-row no-Pix φ for the NONLINEAR links
    (E5/E6 'index', E7/E8 'sieve', E3/E4 logit), computed via `phi_from_native` with pix=0,
    read here by `load_phi_nopix` and joined 1:1 to `ctx.df` on (entity_id, time_id).
CF4 prefers the exact parquet and falls back to the scalar subtraction only when it is absent
(identity-link specs E1/E2, where the subtraction is exact). A missing JSON errors loudly with
the exact command to run, so a run without the recovery fails rather than silently producing
wrong numbers.
"""

include(joinpath(@__DIR__, "foundation_deposit_sim.jl"))

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
    pix_level_zero(cf_dir, estim, spec) -> Float64

The value the `pix_exists` COLUMN takes when Pix does not exist, in estimation units.
The state block is grand-mean centred, so that is −mean(pix_exists) ≈ −0.53, NOT 0.0:
the identity-link fallback below subtracts Υ_pix·(pix − this), and using 0.0 instead
would evaluate a "no Pix" world in which 53% of markets still have Pix. Written by
cf_4_upsilon_export.py alongside Υ_pix; falls back to 0.0 for pre-centering exports.
"""
function pix_level_zero(cf_dir::String, estim::Int, spec_id::Int)::Float64
    f = joinpath(cf_dir, "upsilon_pix_E$(estim)_spec_$(spec_id).json")
    isfile(f) || return 0.0
    m = match(r"\"pix_level_zero\"\s*:\s*(-?[0-9.eE+]+)", read(f, String))
    return m === nothing ? 0.0 : parse(Float64, m.captures[1])
end

"""
    load_phi_nopix(cf_dir, estim, spec, ctx) -> Union{Nothing,Vector{Float64}}

Exact link-aware no-Pix φ, precomputed by `cf_4_upsilon_export.py` via `phi_from_native`
with `pix_exists` zeroed — the correct counterfactual under a NONLINEAR link G, where the
level subtraction φ̂ − Υ_pix·pix is only a first-order approximation (Υ_pix is the AME, not
∂φ/∂pix). Reads `CF_FOUNDATION/phi_nopix_E{e}_spec_{s}.parquet` and returns a per-row vector
aligned to `ctx.df` by an (entity_id, time_id) join, or `nothing` if the file is absent (→
caller falls back to the scalar subtraction, exact for the identity-link specs E1/E2).
"""
function load_phi_nopix(cf_dir::String, estim::Int, spec_id::Int, ctx::CFDemandCtx)
    f = joinpath(cf_dir, "phi_nopix_E$(estim)_spec_$(spec_id).parquet")
    isfile(f) || return nothing
    e = DataFrame(Parquet2.Dataset(f); copycols=true)
    for c in ("entity_id", "time_id", "phi_mt_nopix")
        c in names(e) || error("phi_nopix export $(basename(f)) missing column $c")
    end
    ("entity_id" in names(ctx.df) && "time_id" in names(ctx.df)) ||
        error("ctx.df lacks entity_id/time_id — cannot join the exact no-Pix φ export")
    lut = Dict{Tuple{String,String},Float64}()
    for r in eachrow(e)
        lut[(string(r.entity_id), string(r.time_id))] = Float64(r.phi_mt_nopix)
    end
    out = Vector{Float64}(undef, nrow(ctx.df))
    miss = 0
    for (i, r) in enumerate(eachrow(ctx.df))
        k = (string(r.entity_id), string(r.time_id))
        haskey(lut, k) ? (out[i] = lut[k]) : (miss += 1; out[i] = NaN)
    end
    miss == 0 || error("phi_nopix export misses $miss/$(nrow(ctx.df)) ctx rows (stale? " *
                       "regenerate: python cf_4_upsilon_export.py --estim $estim --spec $spec_id)")
    return out
end

"""
    cf4_pix_reallocation(ctx, st; upsilon_pix[, phi_nopix]) -> NamedTuple

Force the pre-Pix regime in φ, re-simulate deposits at observed spreads, and return the
per-institution volume reallocation vs the observed (φ̂) deposits. If `phi_nopix` (the exact
link-aware no-Pix φ from `load_phi_nopix`) is supplied it is used directly; otherwise the
identity-link fallback φ̂ − Υ_pix·pix is used (exact only for the linear specs E1/E2).
"""
function cf4_pix_reallocation(ctx::CFDemandCtx, st::DepositSimState; upsilon_pix::Float64,
                              T::Int=8, phi_nopix::Union{Nothing,Vector{Float64}}=nothing,
                              pix_zero::Float64=0.0)
    pix, pc = _first_present(ctx.df, ["pix_exists", "pix_active"]; default=0.0)
    if phi_nopix === nothing
        # Identity-link fallback (exact only for E1/E2): drop the Pix term from the level φ̂.
        # The DEVIATION from the no-Pix level is what Υ_pix multiplies -- with the state block
        # centred, the column's "no Pix" value is pix_zero ≈ −0.53, not 0. For a nonlinear link
        # G this is a first-order approximation (Υ_pix is the AME); prefer the exact per-row
        # export from cf_4_upsilon_export.py.
        phi_cf = clamp.(st.phi .- upsilon_pix .* (pix .- pix_zero), 0.0, 0.999)
    else
        # Exact: φ^noPix = G(index − θ_pix·pix), already in [0,1] from phi_from_native.
        phi_cf = clamp.(phi_nopix, 0.0, 0.999)
    end
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
    pix0 = pix_level_zero(cf_dir, a["estim"], a["spec"])          # centred "no Pix" level (≈ −0.53)
    phi_np = load_phi_nopix(cf_dir, a["estim"], a["spec"], ctx)   # exact link-aware φ^noPix, or nothing
    T   = 8   # medium-run (2y) reallocation horizon
    if phi_np === nothing
        log_status("  [CF4] Υ_pix=$(round(υ, sigdigits=4)) | horizon=$T | no-Pix φ via LEVEL " *
                   "subtraction (exact for identity-link E1/E2; for a NONLINEAR link run " *
                   "cf_4_upsilon_export.py to get the exact per-row φ^noPix)")
    else
        log_status("  [CF4] exact link-aware no-Pix φ loaded (phi_from_native, Pix zeroed), " *
                   "$(length(phi_np)) rows | horizon=$T | Υ_pix=$(round(υ, sigdigits=4)) [AME, reporting]")
    end
    res = cf4_pix_reallocation(ctx, st; upsilon_pix=υ, T=T, phi_nopix=phi_np, pix_zero=pix0)
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
