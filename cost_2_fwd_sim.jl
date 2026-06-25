"""
cost_2_fwd_sim.jl
=================
CF2 — BBL Step 2, part 1: forward-simulate the value-function basis ψ under the
EQUILIBRIUM strategy σ̂ and under a battery of DEVIATING strategies σ̃, then export
the firm-level ψ's for the Eq-18 minimization (estimation_1_cost_3_solve.py).

This REPLACES the simulation core of the legacy prototype estimation_1_cost_2_fwd.py
(which used a crude exp(δ-shift) share proxy, hard-coded r_f, only k=4,5, and never
solved Eq 18). Here shares come from the real RC demand (cf_0_demand_eval), deposits
from the real law of motion (cf_0_deposit_sim), and ψ from the exact basis
(cf_0_psi_basis).

Pipeline
--------
  σ̂  (equilibrium)  → simulate deposits → accumulate ψ_eq         (per firm)
  σ̃₁…σ̃_S (deviations)→ simulate deposits → accumulate ψ_dev[s]     (per firm)
  export {ψ_eq, ψ_dev, firms, firm_is_B, Z_names}  → COST_FWD/psi_bbl_*.jls
  estimation_1_cost_3_solve.py reads these and minimizes Σ min{g,0}² (Eq 18).

DEVIATING STRATEGY σ̃ (knob — confirm scheme/scale with author):
  σ̃ perturbs the CHOICE spreads (k∈{4,5}) by a draw and holds the perturbed policy
  for the whole horizon (a stationary deviation, as in BBL forward simulation).
  Default: σ̃ = σ̂ + ε with ε ~ N(0, perturb_scale) on k∈{4,5} only (regulated types
  unchanged). `--perturb-scale` in annualized-spread units (ρ=spread_ann/100).

EQUILIBRIUM σ̂ (knob):
  Default σ̂ = observed spreads ρ̂ (the data IS the equilibrium). Pass
  `--policy-csv` to instead use the smoothed fitted policy from
  estimation_1_cost_1_polfunc.py (polfunc_fitted_spec_*.csv).

⚠ COMPUTE: each σ̃ costs one deposit simulation (≈ one share evaluation when spreads
are held flat). With S deviations on the full panel at R=2000 this is the heavy,
GPU/cluster step. Develop locally with --R small, --time-filter one quarter, and
--shocks small; run headline on Bouchet.

Usage (write-only here; run only after data is downloaded AND author authorizes):
  julia --project=. --threads=4 cost_2_fwd_sim.jl --estim 6 --spec 12 \\
      --stage extended --R 300 --time-filter 2024Q4 --shocks 20 --beta 0.9 --horizon 50
"""

include(joinpath(@__DIR__, "cf_0_psi_basis.jl"))

using DataFrames, Random, Serialization, Statistics

# ==========================================================================
# Spread scenarios
# ==========================================================================
"""
    equilibrium_spreads(ctx; policy_csv=nothing) -> Vector{Float64}

The equilibrium choice-spread vector σ̂ (annualized units). Default = observed ρ̂.
If `policy_csv` is given, overwrite k∈{4,5} rows with the fitted policy spreads
(matched on CodConglomeradoPrudencial × mca_code × deposit_type × time_id).
"""
function equilibrium_spreads(ctx::CFDemandCtx; policy_csv::Union{Nothing,String}=nothing)
    σ̂ = copy(ctx.rho_hat)
    policy_csv === nothing && return σ̂
    @warn "policy-csv merge is a stub: confirm key columns in polfunc_fitted_*.csv before use."
    # TODO(author-confirm): read polfunc_fitted_spec_*.csv, match rows, and replace
    # σ̂ on k∈{4,5}. Left as observed until the fitted-policy keys are confirmed.
    return σ̂
end

"""
    draw_deviation(σ̂, endog, rng; scale) -> Vector{Float64}

A stationary deviating strategy σ̃: perturb the choice spreads (k∈{4,5}) by an
i.i.d. N(0, scale) shock; regulated types pass through unchanged.
"""
function draw_deviation(σ̂::Vector{Float64}, endog::BitVector, rng::AbstractRNG; scale::Float64)
    σ̃ = copy(σ̂)
    @inbounds for i in eachindex(σ̃)
        endog[i] && (σ̃[i] += scale * randn(rng))
    end
    return σ̃
end

# ==========================================================================
# ψ under a given (stationary) spread scenario
# ==========================================================================
"""
    psi_under(ctx, st, Z, markdown_q, spreads_ann; beta, T, asset_return_q, rf_path_q)
        -> (psi_firm, firms)

Simulate deposits forward under the stationary spread vector `spreads_ann`
(annualized, for shares) and accumulate the firm-level ψ basis. The per-period
markdown for the value flow is `markdown_q` (quarterly); for a deviation that
changes spreads, the markdown should move with it — by default we recompute the
quarterly markdown from the scenario spreads as `spreads_ann/4` shifted by the
observed wedge (see note) to stay consistent.
"""
function psi_under(ctx::CFDemandCtx, st::DepositSimState, Z::Matrix{Float64},
                   markdown_q0::Vector{Float64}, spreads_ann::Vector{Float64};
                   beta::Float64=0.9, T::Int=50,
                   asset_return_q::Union{Nothing,Vector{Float64}}=nothing,
                   rf_path_q::Union{Nothing,Vector{Float64}}=nothing)
    sim = simulate_deposits(ctx, st; T=T, spreads_ann=spreads_ann)
    # Move the quarterly markdown with the scenario spread: Δρ^q ≈ Δρ_ann/4.
    # (Confirm annual→quarterly convention; only matters for k∈{4,5} deviations.)
    # markdown_q0 is the observed quarterly markdown (fraction); a scenario change in
    # the ANNUAL spread (pp = spread_ann/100) maps to a quarterly-fraction change via
    # /400 (×0.01 pp→fraction, ÷4 annual→quarter).
    markdown_q = markdown_q0 .+ (spreads_ann .- ctx.rho_hat) ./ 400.0
    res = accumulate_psi(ctx, st, sim.Dep, markdown_q, Z;
                         beta=beta, asset_return_q=asset_return_q, rf_path_q=rf_path_q)
    return res.psi_firm, res.firms
end

# ==========================================================================
# Firm type (B/D) for the κ-specific cost blocks
# ==========================================================================
"""
    firm_is_B(ctx, firms) -> BitVector

Classify each firm (CodConglomeradoPrudencial) as B-type if the majority of its
observations are B (brick-and-mortar). The cost parameters (ω,ζ,γ) differ by κ∈{B,D}.
"""
function firm_is_B(ctx::CFDemandCtx, firms::Vector{String})::BitVector
    key = string.(ctx.df.CodConglomeradoPrudencial)
    isB = BitVector(Bool.(coalesce.(ctx.df.is_B, false)))
    nB = Dict{String,Int}(); nT = Dict{String,Int}()
    for i in eachindex(key)
        nT[key[i]] = get(nT, key[i], 0) + 1
        isB[i] && (nB[key[i]] = get(nB, key[i], 0) + 1)
    end
    return BitVector([get(nB, f, 0) >= get(nT, f, 1) / 2 for f in firms])
end

# ==========================================================================
# Driver
# ==========================================================================
function _parse_cost2_args()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--estim";         arg_type = Int;     default = 6
        "--spec";          arg_type = Int;     default = 12
        "--stage";         arg_type = String;  default = "extended"
        "--R";             arg_type = Int;     default = 2000
        "--seed";          arg_type = Int;     default = 42
        "--hpc";           action   = :store_true
        "--local-dir";     arg_type = String;  default = nothing
        "--suffix";        arg_type = String;  default = ""
        "--beta";          arg_type = Float64; default = 0.9
        "--horizon";       arg_type = Int;     default = 50
        "--shocks";        arg_type = Int;     default = 50      # TOTAL number of σ̃ deviations
        "--perturb-scale"; arg_type = Float64; default = 0.05    # σ̃ shock sd (annualized ρ units)
        "--dbar";          arg_type = Float64; default = -1.0   # <=0 => per-type auto-calibrate
        "--time-filter";   arg_type = String;  default = nothing
        "--policy-csv";    arg_type = String;  default = nothing
        "--n-shards";      arg_type = Int;     default = 1       # split deviations across jobs
        "--shard-id";      arg_type = Int;     default = 0       # 0-based; = SLURM_ARRAY_TASK_ID
    end
    return parse_args(s)
end

function main_cost2()
    a = _parse_cost2_args()
    tf = a["time-filter"] === nothing ? nothing : String[a["time-filter"]]
    ctx = build_cf_context(a["estim"], a["spec"], a["stage"];
                           R=a["R"], seed=a["seed"], hpc=a["hpc"],
                           local_dir=a["local-dir"], suffix=a["suffix"], time_filter=tf)
    # Per-type d-bar auto-calibration (same as CF1): structural active demand
    # (1-phi)*M*s reproduces observed active deposits within B and D separately, so
    # the deposit path (and hence ψ) is real-scale. A global/unit d-bar mis-scales
    # the active channel and makes the Eq-18 costs degenerate.
    local dbar
    if a["dbar"] <= 0.0
        s0 = cf_model_shares(ctx)
        pop0, _    = _first_present(ctx.df, ["pop_total", "M_mt", "pop"]; default=NaN)
        phi0, _    = _first_present(ctx.df, ["phi_mt", "phi_local_mt", "phi_local", "phi"]; default=NaN)
        depact0, _ = _first_present(ctx.df, ["Dep_Act", "active_deposits", "deposit_active"]; default=NaN)
        phi0 = clamp.(phi0, 0.0, 0.999)
        isBcal = BitVector(Bool.(coalesce.(ctx.df.is_B, false)))
        dbar = ones(nrow(ctx.df))
        for (lbl, mask) in (("B", isBcal), ("D", .!isBcal))
            m = mask .& isfinite.(pop0) .& isfinite.(s0) .& isfinite.(phi0) .& isfinite.(depact0)
            den = sum((1.0 .- phi0[m]) .* pop0[m] .* s0[m]); nm = sum(max.(depact0[m], 0.0))
            db = (den > 0 && isfinite(nm)) ? nm / den : 1.0
            dbar[mask] .= db
            log_status("  [CF2] d-bar[$lbl] = $(round(db, sigdigits=5))")
        end
    else
        dbar = a["dbar"]
    end
    st  = load_sim_state(ctx; dbar=dbar)
    Z, znames = load_Z(ctx)
    markdown_q0, mc = _first_present(ctx.df, ["spread_qoq", "spread_q"]; default=NaN)
    if all(isnan, markdown_q0)
        markdown_q0 = ctx.rho_hat ./ 400.0; mc = "rho_hat/400"          # annual pp -> quarterly fraction
    else
        markdown_q0 = clamp.(markdown_q0 ./ 1e4, -0.1, 0.1); mc = "$(mc)/1e4"  # bps -> quarterly fraction
    end
    log_status("  [CF2] markdown ρ^q ← $mc | β=$(a["beta"]) | T=$(a["horizon"]) | shocks=$(a["shocks"])")

    # Forward r^f path for the ψ4 funding base (else ζ is unidentified). Placeholder:
    # flat at the panel-median quarterly risk-free; replace with the BCB forward curve
    # (open knob). asset_return r^j defaults to 0 (pure deposit-funding value).
    rf_q0, rfc = _first_present(ctx.df, ["risk_free_qoq", "risk_free_qoq_lag", "selic_qoq"]; default=NaN)
    rf_level = median(filter(isfinite, rf_q0))
    rf_path  = fill(rf_level, a["horizon"])
    log_status("  [CF2] forward r^f flat at median($rfc)=$(round(rf_level, sigdigits=4)) [placeholder]")

    # Equilibrium ψ
    σ̂ = equilibrium_spreads(ctx; policy_csv=a["policy-csv"])
    psi_eq, firms = psi_under(ctx, st, Z, markdown_q0, σ̂; beta=a["beta"], T=a["horizon"], rf_path_q=rf_path)
    isB = firm_is_B(ctx, firms)
    log_status("  [CF2] ψ_eq: $(size(psi_eq)) over $(length(firms)) firms " *
               "($(sum(isB)) B / $(sum(.!isB)) D)")

    # ── Deviation ψ's (SHARDABLE across SLURM jobs to fit Bouchet time walls) ──
    nf, nb = size(psi_eq)
    S = a["shocks"]; nsh = a["n-shards"]; sid = a["shard-id"]
    (0 <= sid < nsh) || error("shard-id ($sid) must be in 0:$(nsh-1)")
    # Global shock indices for THIS shard (stride). A per-shock RNG seeded by the
    # GLOBAL index makes σ̃ identical whether or not the run is sharded, so shard
    # outputs merge into exactly the unsharded result.
    s_list = [s for s in 1:S if (s - 1) % nsh == sid]
    log_status("  [CF2] shard $sid/$nsh → $(length(s_list)) of $S deviations")
    psi_dev = Array{Float64,3}(undef, length(s_list), nf, nb)
    for (li, s) in enumerate(s_list)
        rng_s = MersenneTwister(a["seed"] * 100003 + s)
        σ̃ = draw_deviation(σ̂, st.endog, rng_s; scale=a["perturb-scale"])
        pd, _ = psi_under(ctx, st, Z, markdown_q0, σ̃; beta=a["beta"], T=a["horizon"], rf_path_q=rf_path)
        psi_dev[li, :, :] .= pd
        li % 5 == 0 && log_status("    [CF2] shard $sid: $li/$(length(s_list)) done")
    end

    _, _, out_dir = get_paths(a["hpc"]; local_dir=a["local-dir"])
    cost_dir = joinpath(dirname(out_dir), "COST_FWD"); mkpath(cost_dir)
    tag = "E$(a["estim"])_spec_$(a["spec"])_$(a["stage"])$(a["suffix"])"
    blocks = vcat(["psi1", "psi2_omega"], ["psi3_gamma_$z" for z in znames], ["psi4_zeta"])

    # Equilibrium ψ is shard-invariant → write once (shard 0).
    if sid == 0
        eq_df = DataFrame(firm=firms, is_B=collect(isB))
        for (j, b) in enumerate(blocks); eq_df[!, b] = psi_eq[:, j]; end
        Parquet2.writefile(joinpath(cost_dir, "psi_eq_$tag.parquet"), eq_df)
        log_status("  [CF2] wrote psi_eq_$tag.parquet")
    end

    # Deviation ψ for this shard (firm-fastest within shock; GLOBAL shock ids).
    nloc = length(s_list)
    dev_df = DataFrame(shock=repeat(s_list, inner=nf),
                       firm=repeat(firms, outer=nloc),
                       is_B=repeat(collect(isB), outer=nloc))
    for (j, b) in enumerate(blocks)
        dev_df[!, b] = vec([psi_dev[li, f, j] for f in 1:nf, li in 1:nloc])
    end
    shard_tag = nsh == 1 ? "" : "_shard$(sid)of$(nsh)"
    Parquet2.writefile(joinpath(cost_dir, "psi_dev_$tag$shard_tag.parquet"), dev_df)
    log_status("  [CF2] wrote psi_dev_$tag$shard_tag.parquet ($nloc deviations)")
    log_status("[DONE] cost_2_fwd_sim shard $sid — run estimation_1_cost_3_solve.py after ALL shards")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_cost2()
end
