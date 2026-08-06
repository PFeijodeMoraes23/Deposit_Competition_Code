"""
foundation_psi_basis.jl
=================
Foundation 0c: accumulate the BBL value-function basis ψ along a simulated deposit
path. The value function is LINEAR in the cost parameters (eqs 16-B / 16-D):

    V_jσt = ψ_jσt' · [ 1 , −ω , −γ' , −(1+ζ) ]

with the four discounted, deposit-weighted blocks (β per quarter, T quarters):

    ψ1 (coef +1)      = Σ_t β^t Σ_k Dep_jk,t · ( r^j_t + ρ_jk,t )      # revenue/markdown
    ψ2 (coef −ω)      = Σ_t β^t Σ_k Dep_jk,t                           # deposit base
    ψ3 (coef −γ', vec)= Σ_t β^t Σ_k Dep_jk,t · Z_j,t                   # cost-shifter base
    ψ4 (coef −(1+ζ))  = Σ_t β^t r^f_t · Σ_k Dep_jk,t                   # funding base

so ψ is a length (3 + n_Z) vector PER FIRM (aggregating k and markets within firm),
and the cost-parameter vector is θ_c = [1, ω, γ(n_Z), (1+ζ)] with sign pattern
[+, −, −, −]. Recovered in CF2 by minimizing squared FOC-inequality violations.

OPEN MODELING KNOBS (confirm before headline run — see counterfactuals_plan.md):
  * r^j_t (asset return entering ψ1): not directly observed. `asset_return` defaults
    to r^f_t (i.e. (r^j − r^f)=0, treating the object as the deposit-FUNDING value).
    Supply a margin or column to include a positive asset spread.
  * Z_j cost shifters: the γ regressors. Default = the lagged cost/capital ratios
    used by the policy-function step (estimation_bbl_1_polfunc.py COST_SHIFTERS +
    CAPITAL_WHOLESALE), resolved by name from the parquet/sidecar.
  * β (per quarter, default 0.9), horizon T (default 50), r^f path (default flat).

This module is `include`d by estimation_bbl_2_fwd_sim.jl; it is not a standalone entry point.
"""

include(joinpath(@__DIR__, "foundation_deposit_sim.jl"))

using DataFrames, LinearAlgebra, Statistics
import JSON3

# Cost-shifter (Z) columns entering c = ω + ζ·r^f_q + γ′Z (V_Main eq 8).
# ACTIVE SET — the four actually present in the demand-prep parquets. Mirrors the corresponding
# subset of estimation_bbl_1_polfunc.py's COST_SHIFTERS (personnel/admin/tax, COSIF DRE) +
# CAPITAL_WHOLESALE (indice_basileia), so γ stays comparable with the policy function.
const Z_COST_COLS = ["personnel_cost_ratio_lag", "admin_cost_ratio_lag",
                     "tax_cost_ratio_lag", "indice_basileia_lag"]

# DEFERRED (2026-08-02) — these live in market_panel.csv and ARE used by polfunc
# (CAPITAL_WHOLESALE), but were never propagated into the demand-prep parquets, so the ψ basis
# never saw them: load_Z silently kept whatever was present and γ was identified on 4, not 6.
# This is a PREP-WIRING gap, not missing data. Until estimation_1_demand_1_prep.py carries them
# through (needs a re-prep + re-upload), their explanatory power loads onto ω.
const Z_COST_DEFERRED = ["wholesale_ratio_lag", "lci_lca_ratio_lag"]

"""
    load_Z(ctx; sidecar=nothing) -> (Z::Matrix, names::Vector{String})

Assemble the per-observation cost-shifter matrix Z (N × n_Z) from the parquet (or a
sidecar), keeping only columns that are present and non-degenerate.
"""
function load_Z(ctx::CFDemandCtx; sidecar::Union{Nothing,DataFrame}=nothing)
    df = sidecar === nothing ? ctx.df : hcat(ctx.df, sidecar; makeunique=true)
    cols = [c for c in Z_COST_COLS if c in names(df)]
    isempty(cols) && error("No cost-shifter (Z) columns found; pass a sidecar with $(Z_COST_COLS).")
    # Never drop an expected shifter silently: a missing column just removes a γ, changing what ω
    # absorbs, with no other trace. (2026-08-02: two shifters had been vanishing this way for the
    # whole project because they were absent from the demand parquets — see Z_COST_DEFERRED.)
    absent = setdiff(Z_COST_COLS, cols)
    isempty(absent) || @warn "  [ψ] Z cost-shifter(s) MISSING from the panel — γ is identified " *
                             "WITHOUT them and their effect loads onto ω: $(absent)"
    # ---- Missing / non-finite: impute the WITHIN-TYPE MEDIAN, not 0.0 -----------------------
    # Until 2026-08-06 this was `coalesce(v, 0.0)` plus `replace!(Inf => 0.0)`. For a COST RATIO
    # zero is not a neutral filler — it reads as "this bank has zero personnel cost" — so every
    # unobserved row entered γ̂ as an extreme-LOW observation, pulling the slope. It was ~2,094
    # rows per column on demand_6_spec_12 (0.28%), small but systematically signed, and invisible
    # because 0.0 is a perfectly plausible-looking value.
    # The within-type median is the natural filler here: it is the same B/D split the winsorising
    # below uses (D balance sheets differ from B by construction), it is robust to the very tail
    # this function then clips, and imputing at the median means those rows contribute no
    # leverage to γ̂ instead of contributing wrong leverage.
    # (polfunc leaves them NaN and statsmodels drops the rows; the BBL cannot drop rows because ψ
    # is per-observation, so imputation is the only option here.)
    isB_all = "is_B" in names(df) ? BitVector(Bool.(coalesce.(df.is_B, false))) : trues(nrow(df))
    Z = zeros(nrow(df), length(cols))
    for (i, c) in enumerate(cols)
        raw = df[!, c]
        v = Float64[(x === missing || x === nothing) ? NaN : Float64(x) for x in raw]
        replace!(v, Inf => NaN, -Inf => NaN)          # ±Inf is missing information, not a value
        bad = .!isfinite.(v)
        if any(bad)
            nimp = 0
            for grp in (true, false)
                rows = findall(isB_all .== grp)
                isempty(rows) && continue
                good = [v[r] for r in rows if isfinite(v[r])]
                isempty(good) && continue
                med = median(good)
                for r in rows
                    isfinite(v[r]) || (v[r] = med; nimp += 1)
                end
            end
            # Anything still non-finite has no finite value anywhere in its own type group.
            leftover = count(!isfinite, v)
            leftover == 0 || (v[.!isfinite.(v)] .= 0.0)
            log_status("  [ψ] $c: imputed $nimp non-finite obs at the within-type median" *
                       (leftover == 0 ? "" : " ($leftover had no finite value in-type → 0.0)"))
        end
        Z[:, i] .= v
    end

    # ---- WINSORIZE, matching estimation_bbl_1_polfunc.py -----------------------------------
    # Until 2026-08-06 the BBL read these RAW while polfunc winsorized the same columns at the
    # 1st/99th percentile within firm type — so the policy function and the cost equation were
    # fitted on DIFFERENT versions of the same regressors, and γ̂ was not what the write-up
    # described. Measured on demand_6_spec_12 (n=755,438) before this fix:
    #     personnel_cost_ratio_lag  p99 0.0109  max   1.995   (183x p99)
    #     admin_cost_ratio_lag      p99 0.0110  max   4.567   (415x p99)
    #     indice_basileia_lag       p99 21.21   max  53093.6  (2503x p99)
    # and **93.1% of Σx² for the Basel ratio came from 0.034% of rows** — i.e. γ̂ was set by ~257
    # observations, dominated by one. The extremes are all is_B=false (digital) banks: capital
    # over near-zero risk-weighted assets, a divide-by-tiny artifact, not a real capital ratio
    # (p50 = 16.4, so the column is a PERCENT and anything >100 is implausible).
    # polfunc's own note records the consequence: the corrupt tail "drives the point estimates and
    # collapses the wild-cluster-bootstrap SEs (the Basel row printed 0.0001 with a 0.0000 SE at
    # *** before this was applied)". Same rule here, same reason.
    # WITHIN-type is the point: B and D have genuinely different balance sheets, so pooled
    # percentiles would clip real cross-type variation instead of the corrupt tail.
    # BBL_WINSOR_Z=0 restores the raw columns (for a with/without comparison).
    if get(ENV, "BBL_WINSOR_Z", "1") == "1"
        pct = try parse(Float64, get(ENV, "BBL_WINSOR_PCT", "0.01")) catch; 0.01 end
        isB = isB_all      # same B/D split the median imputation above used — keep them in step
        "is_B" in names(df) || @warn "  [ψ] no is_B column — imputing and winsorizing Z POOLED, " *
                                     "not within type"
        for (i, c) in enumerate(cols)
            nclip = 0; before = maximum(view(Z, :, i))
            for grp in (true, false)
                rows = findall(isB .== grp)
                length(rows) < 100 && continue          # too few for stable percentiles
                s = view(Z, rows, i)
                lo, hi = quantile(s, pct), quantile(s, 1 - pct)
                (isfinite(lo) && isfinite(hi) && lo < hi) || continue
                for r in rows
                    z = Z[r, i]
                    zc = clamp(z, lo, hi)
                    zc == z || (nclip += 1)
                    Z[r, i] = zc
                end
            end
            nclip == 0 || log_status("  [ψ] winsorized $c: $nclip obs, max " *
                                     "$(round(before, sigdigits=6)) → $(round(maximum(view(Z,:,i)), sigdigits=6))")
        end
        log_status("  [ψ] Z winsorized at $(round(100*pct, digits=1))%/$(round(100*(1-pct), digits=1))% within firm type " *
                   "(matches estimation_bbl_1_polfunc.py)")
    else
        @warn "  [ψ] BBL_WINSOR_Z=0 — Z cost-shifters RAW; γ̂ will be driven by the corrupt tail " *
              "and is NOT comparable to the policy function's regressors."
    end

    log_status("  [ψ] Z cost-shifters: $(cols)")
    return Z, cols
end

# ==========================================================================
# Marginal-cost parameters (CF2 output) — shared by CF3 (equilibrium) and CF1-net
# ==========================================================================
"""
    load_cost_params(path, znames) -> Dict("B"=>(omega,zeta,gamma::Vector), "D"=>(…))

Read `COST_FWD/cost_params_{tag}.json` (estimation_bbl_3_solve.py) and align each type's
γ to the ψ-basis Z-column order `znames`.
"""
function load_cost_params(path::String, znames::Vector{String})
    isfile(path) || error("Missing cost params $path — run estimation_bbl_2_fwd_sim.jl → " *
                          "estimation_bbl_3_solve.py first (or pass --cost-json).")
    j = JSON3.read(read(path, String))
    out = Dict{String,Any}()
    for κ in ("B", "D")
        haskey(j, Symbol(κ)) || continue
        b = j[Symbol(κ)]; γmap = b["gamma"]
        γ = Float64[haskey(γmap, Symbol(z)) ? Float64(γmap[Symbol(z)]) : 0.0 for z in znames]
        out[κ] = (omega=Float64(b["omega"]), zeta=Float64(b["zeta"]), gamma=γ)
    end
    isempty(out) && error("No B/D blocks in $path")
    return out
end

"""
    theta_c(cost_κ, n_Z) -> Vector{Float64}

Cost-contraction vector matching the ψ_firm layout [ψ1, ψ2, ψ3(1..n_Z), ψ4]:
`[1, −ω, −γ_1..−γ_{n_Z}, −(1+ζ)]` (V_Main eq 16). V_j = ψ_j · θ_c.
"""
function theta_c(costκ, n_Z::Int)::Vector{Float64}
    v = Vector{Float64}(undef, 3 + n_Z)
    v[1] = 1.0; v[2] = -costκ.omega
    @inbounds for z in 1:n_Z; v[2+z] = -costκ.gamma[z]; end
    v[3+n_Z] = -(1.0 + costκ.zeta)
    return v
end

"""
    marginal_cost_per_obs(cost, is_B, rf_q, Z) -> Vector{Float64}

Per-observation quarterly marginal cost `c = ω^κ + ζ^κ·r^f_q + (γ^κ)′Z` (V_Main eq 8), with
κ∈{B,D} selected by `is_B`. `rf_q` is the per-obs quarterly risk-free; `Z` is the (N×n_Z)
cost-shifter matrix from `load_Z` (same column order as the γ in `cost`).
"""
function marginal_cost_per_obs(cost, is_B::BitVector, rf_q::Vector{Float64}, Z::Matrix{Float64})
    N = length(is_B); c = Vector{Float64}(undef, N)
    cB = cost["B"]; cD = cost["D"]
    @inbounds for i in 1:N
        κ = is_B[i] ? cB : cD
        c[i] = κ.omega + κ.zeta * rf_q[i] + dot(@view(Z[i, :]), κ.gamma)
    end
    return c
end

"""
    accumulate_psi(ctx, st, Dep, markdown_q, Z; beta, asset_return_q, rf_path_q)
        -> (psi_firm::Matrix, firm_ids::Vector, blocks::NamedTuple)

Given a deposit path `Dep` (N×(T+1)) and per-obs/per-period inputs, accumulate the
ψ blocks discounted at `beta`, then aggregate rows to the firm
(`CodConglomeradoPrudencial`) level. Returns `psi_firm` of size (n_firms × (3+n_Z))
with column layout [ψ1, ψ2, ψ3(1..n_Z), ψ4].

`markdown_q` (ρ^q, length N), `asset_return_q` (r^j_q, length N; default r^f_q),
`rf_path_q` (length T; default flat from per-obs r^f implied by st). All quarterly.
"""
function accumulate_psi(ctx::CFDemandCtx, st::DepositSimState,
                        Dep::Matrix{Float64}, markdown_q::Vector{Float64},
                        Z::Matrix{Float64};
                        beta::Float64=0.9,
                        asset_return_q::Union{Nothing,Vector{Float64}}=nothing,
                        rf_path_q::Union{Nothing,Vector{Float64}}=nothing)
    N, Tp1 = size(Dep); T = Tp1 - 1
    n_Z = size(Z, 2)
    rj = asset_return_q === nothing ? zeros(N) : asset_return_q   # (r^j − r^f); 0 ⇒ funding value
    # Per-obs flat r^f for ψ4 if no path supplied: infer from accrual identity is
    # ambiguous, so default to a zero contribution unless rf_path_q is given.
    rf_flat = rf_path_q === nothing ? zeros(T) : rf_path_q

    # Per-observation ψ blocks (then summed to firms).
    psi1 = zeros(N); psi2 = zeros(N); psi4 = zeros(N)
    psi3 = zeros(N, n_Z)
    @inbounds for t in 0:T
        bt = beta^t
        dep_t = @view Dep[:, t+1]
        rf_t = t == 0 ? 0.0 : rf_flat[t]
        # ψ1 (V_Main eq 16, row 1) carries the GROSS asset return r^j: Dep·(r^j + ρ).
        # `asset_return_q` is supplied as the NET margin (r^j − r^f), so add r^f back here.
        # The single −r^f (funding cost) is then delivered by the −(1+ζ)·ψ4 term in θ_c.
        # Using the NET rj here subtracted r^f TWICE (V/Dep = ρ − c − r^f), which made the
        # deposit franchise value negative and forced ω to its ≥0 bound in the eq:17 solve.
        psi1 .+= bt .* dep_t .* (rj .+ rf_t .+ markdown_q)
        psi2 .+= bt .* dep_t
        for z in 1:n_Z
            @views psi3[:, z] .+= bt .* dep_t .* Z[:, z]
        end
        psi4 .+= bt .* rf_t .* dep_t
    end

    # Aggregate observations → firms.
    firm_key = string.(ctx.df.CodConglomeradoPrudencial)
    firms = sort(unique(firm_key))
    fidx = Dict(f => i for (i, f) in enumerate(firms))
    nf = length(firms)
    psi_firm = zeros(nf, 3 + n_Z)
    @inbounds for i in 1:N
        r = fidx[firm_key[i]]
        psi_firm[r, 1] += psi1[i]
        psi_firm[r, 2] += psi2[i]
        for z in 1:n_Z; psi_firm[r, 2+z] += psi3[i, z]; end
        psi_firm[r, 3+n_Z] += psi4[i]
    end
    return (psi_firm=psi_firm, firms=firms,
            blocks=(psi1=psi1, psi2=psi2, psi3=psi3, psi4=psi4))
end
