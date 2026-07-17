"""
cf_1_franchise_value.jl
=======================
CF1 — franchise value attributable to depositor sleepiness (the paper's headline
counterfactual object): the discounted deposit-funding value under the estimated
sleeper share φ̂ versus the no-inertia counterfactual φ=0.

GROSS version (this file): the deposit-FUNDING franchise value, i.e. the markdown
income the bank earns by funding at r^dep = r^f − ρ instead of r^f:

    V^gross_j = Σ_{t=0}^{T} β^t · Σ_{k} Dep_jk,t · ρ^q_jk        (per-period markdown)

    ΔV^sleep_j = V^gross_j(φ̂) − V^gross_j(φ=0)

This needs NEITHER marginal costs NOR an equilibrium re-solve — only the demand
shares (foundation_demand_eval) and the deposit law of motion (foundation_deposit_sim). The
NET-of-cost version (subtracting ĉ and adding the asset-return term) is produced
later, once CF2 delivers cost parameters; see counterfactuals_plan.md.

UNITS / TIMING (confirm with author before headline run):
  * Periods are QUARTERS (panel frequency). β is per-period; default 0.9 per the
    draft ("β=0.9, 50 periods forward"). Override with --beta.
  * The per-period markdown is the QUARTERLY spread ρ^q (spread_qoq), NOT the
    annualized ρ used by the demand shares. We hold ρ flat over the horizon
    (banks-believe-state-constant), so shares are computed once per scenario.
  * Market-size level d̄ scales V in levels but cancels in the φ̂/φ=0 RATIO; the
    headline statistic ΔV/V(φ̂) and the share of franchise value due to inertia are
    d̄-invariant.

Usage (after data is downloaded AND running is authorized):
  # local dev (approximate, one quarter, low memory):
  julia --project=. --threads=4 cf_1_franchise_value.jl --estim 6 --spec 12 \\
      --stage extended --R 300 --time-filter 2024Q4 --beta 0.9 --horizon 50
  # cluster headline:
  julia --project=\${PROJECT_DIR} --threads=8 cf_1_franchise_value.jl --estim 6 \\
      --spec 12 --stage extended --R 2000 --hpc --beta 0.9 --horizon 50
"""

include(joinpath(@__DIR__, "foundation_psi_basis.jl"))   # → foundation_deposit_sim + load_Z / load_cost_params (for --net)

using DataFrames, Statistics, Printf

# ==========================================================================
# Discounted gross franchise value along a simulated deposit path
# ==========================================================================
"""
    gross_value(Dep, markdown_q; beta) -> Vector{Float64}

Per-observation discounted deposit-funding value
`Σ_{t=0}^{T} β^t · Dep_t · ρ^q`, where `Dep` is N×(T+1) (col t+1 = period t) and
`markdown_q` is the per-period (quarterly) markdown ρ^q (length N, held flat).
"""
function gross_value(Dep::Matrix{Float64}, markdown_q::Vector{Float64}; beta::Float64)
    N, Tp1 = size(Dep)
    v = zeros(N)
    @inbounds for t in 0:(Tp1 - 1)
        bt = beta^t
        @views v .+= bt .* Dep[:, t+1] .* markdown_q
    end
    return v
end

"""
    franchise_decomposition(ctx, st; beta, T, markdown_q) -> NamedTuple

Compute V^gross(φ̂), V^gross(φ=0), and ΔV^sleep per observation, plus the
inertia share ΔV/V(φ̂). Shares are evaluated at the observed (flat) spreads.
"""
function franchise_decomposition(ctx::CFDemandCtx, st::DepositSimState;
                                 beta::Float64=0.9, T::Int=50,
                                 markdown_q::Vector{Float64})
    sim_phi = simulate_deposits(ctx, st; T=T)                                 # φ = φ̂
    sim_0   = simulate_deposits(ctx, st; T=T, phi_override=zeros(length(st.phi)))  # φ = 0
    V_phi = gross_value(sim_phi.Dep, markdown_q; beta=beta)
    V_0   = gross_value(sim_0.Dep,   markdown_q; beta=beta)
    dV    = V_phi .- V_0
    share = dV ./ max.(abs.(V_phi), 1e-12)
    return (V_phi=V_phi, V_0=V_0, dV_sleep=dV, inertia_share=share,
            Dep_phi_T=sim_phi.Dep[:, end], Dep_0_T=sim_0.Dep[:, end])
end

# ==========================================================================
# Reporting: aggregate to firm / type / deposit-type and export
# ==========================================================================
function summarize_and_export(ctx::CFDemandCtx, st::DepositSimState, dec;
                              out_path::String)
    df = DataFrame(
        CodConglomeradoPrudencial = string.(ctx.df.CodConglomeradoPrudencial),
        is_B         = st.is_B,
        deposit_type = st.dep_type,
        V_phi        = dec.V_phi,
        V_0          = dec.V_0,
        dV_sleep     = dec.dV_sleep,
    )
    mkpath(dirname(out_path))
    Parquet2.writefile(out_path, df)
    log_status("  [CF1] Wrote per-obs decomposition → $(basename(out_path))")

    # Console summary: totals and inertia share by firm type and deposit type.
    tot_phi = sum(dec.V_phi); tot_0 = sum(dec.V_0); tot_dV = sum(dec.dV_sleep)
    @printf("\n  === CF1 Franchise value (GROSS, deposit-funding) ===\n")
    @printf("  Total V(φ̂)=%.4g  V(φ=0)=%.4g  ΔV_sleep=%.4g  (%.1f%% of V(φ̂))\n",
            tot_phi, tot_0, tot_dV, 100 * tot_dV / max(abs(tot_phi), 1e-12))
    for (lbl, mask) in (("B-firms", st.is_B), ("D-firms", .!st.is_B))
        any(mask) || continue
        vp = sum(dec.V_phi[mask]); dv = sum(dec.dV_sleep[mask])
        @printf("    %-8s V(φ̂)=%.4g  ΔV_sleep=%.4g  (%.1f%%)\n",
                lbl, vp, dv, 100 * dv / max(abs(vp), 1e-12))
    end
    for k in sort(unique(st.dep_type))
        mask = st.dep_type .== k
        vp = sum(dec.V_phi[mask]); dv = sum(dec.dV_sleep[mask])
        @printf("    k=%d      V(φ̂)=%.4g  ΔV_sleep=%.4g  (%.1f%%)\n",
                k, vp, dv, 100 * dv / max(abs(vp), 1e-12))
    end
    return out_path
end

# ==========================================================================
# CLI
# ==========================================================================
function _parse_cf1_args()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--estim";       arg_type = Int;     default = 6
        "--spec";        arg_type = Int;     default = 12
        "--stage";       arg_type = String;  default = "extended"
        "--R";           arg_type = Int;     default = 2000
        "--seed";        arg_type = Int;     default = 42
        "--hpc";         action   = :store_true
        "--local-dir";   arg_type = String;  default = nothing
        "--suffix";      arg_type = String;  default = ""
        "--beta";        arg_type = Float64; default = 0.9
        "--horizon";     arg_type = Int;     default = 50
        "--time-filter"; arg_type = String;  default = nothing   # e.g. "2024Q4" (local dev)
        "--dbar";        arg_type = Float64; default = -1.0   # <=0 => auto-calibrate globally
        "--net";         action   = :store_true             # net-of-cost value flow: ρ^q − ĉ (needs CF2 costs)
        "--cost-json";   arg_type = String;  default = nothing  # default COST_FWD/cost_params_{tag}.json
    end
    return parse_args(s)
end

function main_cf1()
    a = _parse_cf1_args()
    tf = a["time-filter"] === nothing ? nothing : String[a["time-filter"]]
    ctx = build_cf_context(a["estim"], a["spec"], a["stage"];
                           R=a["R"], seed=a["seed"], hpc=a["hpc"],
                           local_dir=a["local-dir"], suffix=a["suffix"],
                           time_filter=tf)
    # Market size: load_sim_state takes M_mt/M_nat from the demand parquet — V_Main's
    # M_mt = d̄_mt·Pop_mt with d̄_mt = bc_mt·r̂_max, i.e. the market size the BLP was estimated under.
    # The per-type d-bar auto-calibration (with its DBAR_CAP_K=5 guardrail) that used to live here
    # back-solved a scalar from pop_total, ignoring banked_correction entirely, so CF1 simulated
    # under a different market size than the demand model. Retired; --dbar is an override only.
    # See counterfactuals_plan.md §9.8.
    dbar = a["dbar"] > 0.0 ? a["dbar"] : 1.0
    st = load_sim_state(ctx; dbar=dbar)
    # Per-period (quarterly) markdown ρ^q for the value flow.
    markdown_q, mc = _first_present(ctx.df, ["spread_qoq", "spread_q"]; default=NaN)
    if all(isnan, markdown_q)
        markdown_q = ctx.rho_hat ./ 4.0   # annualized → quarterly fallback (confirm convention)
        mc = "rho_hat/4"
    else
        markdown_q = clamp.(markdown_q ./ 1e4, -0.1, 0.1)  # bps -> per-quarter fraction; clip residual artifact tail (±10%/qtr)
        mc = "$(mc)/1e4"
    end
    log_status("  [CF1] markdown ρ^q ← $mc | β=$(a["beta"]) | horizon=$(a["horizon"])")
    _, _, out_dir = get_paths(a["hpc"]; local_dir=a["local-dir"])

    # NET-of-cost value flow (needs CF2 costs): replace ρ^q with (r^j−r^f)+ρ^q−ĉ. With the
    # r^j−r^f=0 default this is ρ^q − c^q, c^q = ω+ζ·r^f_q+γ′Z per obs (V_Main eq 8). The cost
    # is φ-invariant so it shifts both the φ̂ and φ=0 legs identically.
    kind = "gross"
    if a["net"]
        tag = "E$(a["estim"])_spec_$(a["spec"])_$(a["stage"])$(a["suffix"])"
        cj = a["cost-json"] === nothing ?
            joinpath(dirname(out_dir), "COST_FWD", "cost_params_$tag.json") : a["cost-json"]
        Z, znames = load_Z(ctx)
        cost = load_cost_params(cj, znames)
        rfq, _ = _first_present(ctx.df, ["risk_free_qoq", "risk_free_qoq_lag", "selic_qoq"]; default=0.0)
        isBv = BitVector(Bool.(coalesce.(ctx.df.is_B, false)))
        cq = marginal_cost_per_obs(cost, isBv, rfq, Z)
        markdown_q = markdown_q .- cq
        kind = "net"
        log_status("  [CF1] NET: markdown ρ^q − ĉ (mean c^q=$(round(mean(cq), sigdigits=3))) ← $(basename(cj))")
    end

    dec = franchise_decomposition(ctx, st; beta=a["beta"], T=a["horizon"],
                                  markdown_q=markdown_q)
    cf_dir = joinpath(dirname(out_dir), "CF_FOUNDATION")
    out_path = joinpath(cf_dir,
        "cf1_franchise_$(kind)_E$(a["estim"])_spec_$(a["spec"])_$(a["stage"])$(a["suffix"]).parquet")
    summarize_and_export(ctx, st, dec; out_path=out_path)
    log_status("[DONE] CF1 franchise value ($kind)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_cf1()
end
