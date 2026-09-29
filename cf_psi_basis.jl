"""
cf_psi_basis.jl
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
    used by the policy-function step (bbl_polfunc.py COST_SHIFTERS +
    CAPITAL_WHOLESALE), resolved by name from the parquet/sidecar.
  * β (per quarter) and the BBL horizon T: BBL_BETA / BBL_HORIZON from bbl_discount.env (see
    `bbl_discount` below) unless a caller passes them; r^f path (default flat).

This module is `include`d by bbl_fwd_sim.jl; it is not a standalone entry point.
"""

include(joinpath(@__DIR__, "cf_deposit_sim.jl"))

using DataFrames, LinearAlgebra, Statistics
import JSON3

# ==========================================================================
# The discount registry: bbl_discount.env
# ==========================================================================
"""
    bbl_discount(key) -> String

One value of `bbl_discount.env`, the single place the BBL discount factor (`BBL_BETA`) and the
forward-simulation horizon (`BBL_HORIZON`) are set. The file sits beside this one (the scripts
folder on the cluster); `BBL_DISCOUNT_ENV` names another file. Plain `KEY=VALUE` lines with `#`
comments, read with Base alone. A missing file or key is an error, never a fallback literal:
a silent default is how a discount factor nobody chose reaches a run.
"""
function bbl_discount(key::AbstractString)
    path = get(ENV, "BBL_DISCOUNT_ENV", joinpath(@__DIR__, "bbl_discount.env"))
    isfile(path) || error("BBL discount registry not found: $path\n" *
                          "  It sets BBL_BETA and BBL_HORIZON. Upload bbl_discount.env beside the " *
                          "scripts, or pass --beta/--horizon explicitly.")
    val = nothing
    for ln in eachline(path)
        s = strip(ln)
        (isempty(s) || startswith(s, '#')) && continue
        i = findfirst('=', s)
        i === nothing && continue
        strip(s[1:prevind(s, i)]) == key || continue
        v = strip(s[nextind(s, i):end])
        j = findfirst('#', v)
        j === nothing || (v = strip(v[1:prevind(v, j)]))
        isempty(v) || (val = String(v))
    end
    val === nothing && error("$key is not set in $path")
    return val
end
bbl_beta()    = parse(Float64, bbl_discount("BBL_BETA"))
bbl_horizon() = parse(Int, bbl_discount("BBL_HORIZON"))

"""
    resolve_discount!(a; horizon=true, who="BBL") -> a

Fill the parsed-argument dict's `"beta"` (and, when `horizon`, `"horizon"`) from bbl_discount.env
when the flag was not given (`nothing`), and log where each value came from. `horizon=false` is
for the counterfactual entry points: their horizon is their own, and only β is shared with the
BBL cost estimation.
"""
function resolve_discount!(a::AbstractDict; horizon::Bool=true, who::AbstractString="BBL")
    src = String[]
    if a["beta"] === nothing
        a["beta"] = bbl_beta(); push!(src, "β=$(a["beta"]) ← bbl_discount.env")
    else
        push!(src, "β=$(a["beta"]) ← --beta")
    end
    if horizon
        if a["horizon"] === nothing
            a["horizon"] = bbl_horizon(); push!(src, "T=$(a["horizon"]) ← bbl_discount.env")
        else
            push!(src, "T=$(a["horizon"]) ← --horizon")
        end
    end
    log_status("  [$who] discount: " * join(src, " | "))
    return a
end

# Cost-shifter (Z) columns entering c = ω + ζ·r^f_q + γ′Z (V_Main eq 8).
# ACTIVE SET — the four actually present in the demand-prep parquets. Mirrors the corresponding
# subset of bbl_polfunc.py's COST_SHIFTERS (personnel/admin/tax, COSIF DRE) +
# CAPITAL_WHOLESALE (indice_basileia), so γ stays comparable with the policy function.
const Z_COST_COLS = ["personnel_cost_ratio_lag", "admin_cost_ratio_lag",
                     "tax_cost_ratio_lag", "indice_basileia_lag"]

# DEFERRED (2026-08-02) — these live in market_panel.csv and ARE used by polfunc
# (CAPITAL_WHOLESALE), but were never propagated into the demand-prep parquets, so the ψ basis
# never saw them: load_Z silently kept whatever was present and γ was identified on 4, not 6.
# This is a PREP-WIRING gap, not missing data. Until sleep_demand_prep_e1.py carries them
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
    # rows per column on demand_4_spec_12 (0.28%), small but systematically signed, and invisible
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

    # ---- WINSORIZE, matching bbl_polfunc.py -----------------------------------
    # Until 2026-08-06 the BBL read these RAW while polfunc winsorized the same columns at the
    # 1st/99th percentile within firm type — so the policy function and the cost equation were
    # fitted on DIFFERENT versions of the same regressors, and γ̂ was not what the write-up
    # described. Measured on demand_4_spec_12 (n=755,438) before this fix:
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
                   "(matches bbl_polfunc.py)")
    else
        @warn "  [ψ] BBL_WINSOR_Z=0 — Z cost-shifters RAW; γ̂ will be driven by the corrupt tail " *
              "and is NOT comparable to the policy function's regressors."
    end

    log_status("  [ψ] Z cost-shifters: $(cols)")
    return Z, cols
end

# ==========================================================================
# Cost shifters along the simulated path
# ==========================================================================
"""
    ZEvolution

The cost shifters' own law of motion over the forward horizon, as bbl_transitions.py estimates
it (`cost_shifters`): mean reversion to a FIRM-SPECIFIC level with a persistence per firm type,

    z_{j,t+1} = (1 − ρ_κ)·μ_j + ρ_κ·z_{j,t} + u       ⇒       E z_{j,t} = μ_j + ρ_κ^t·(z_{j,0} − μ_j),

ρ_κ the pooled within-firm OLS slope of type κ ∈ {B, D} and μ_j firm j's own level — the mean the
within transform removes. It is stored as the equivalent SHIFT of the launch value,

    Z_{i,t} = Z_{i,0} + (ρ_κ^t − 1)·(Z_{i,0} − μ_{j(i)}),

which is exactly Z_{i,0} at t = 0 and for ρ = 1, so ρ = 1 reproduces the constant-Z ψ3 bit for
bit. μ_j is the firm's mean over the quarters the panel observes, taken on the SAME Z that enters
ψ3 (load_Z: within-type median imputation, 1/99 within-type winsorisation, parquet units — the
Basel index is in percentage points there and a fraction in the market panel, which ρ does not
see but a level does). A row's ρ is its own type's, the split bbl_transitions.py estimates on.
"""
struct ZEvolution
    dz   ::Matrix{Float64}   # N × n_Z: Z_{i,0} − μ_{j(i)}
    rhoB ::Vector{Float64}   # n_Z, B firms (1.0 = the shifter does not move)
    rhoD ::Vector{Float64}   # n_Z, D firms
    isB  ::BitVector         # N, the row's firm type
    src  ::String            # the transitions file the ρ came from
end

"""
    _firm_level(zcol, firm, tid) -> μ per row

The firm's long-run level of one shifter: the firm-quarter value (mean over the firm's rows in
that quarter; a firm-level ratio is the same on all of them) averaged over the firm's quarters.
"""
function _firm_level(zcol::AbstractVector{Float64}, firm::Vector{String}, tid::Vector{String})
    N = length(zcol)
    fq = Dict{Tuple{String,String},Tuple{Float64,Int}}()
    @inbounds for i in 1:N
        k = (firm[i], tid[i]); s, c = get(fq, k, (0.0, 0))
        fq[k] = (s + zcol[i], c + 1)
    end
    fs = Dict{String,Tuple{Float64,Int}}()
    for ((f, _), (s, c)) in fq
        a, n = get(fs, f, (0.0, 0))
        fs[f] = (a + s / c, n + 1)
    end
    return [(p = fs[firm[i]]; p[1] / p[2]) for i in 1:N]
end

"""
    build_z_evolution(df, Z, znames, isB; transitions_path, rho_field="rho") -> ZEvolution

ρ from `cost_shifters[z][B|D][rho_field]` of bbl_transitions.json, kept when 0 < ρ < 1 and
otherwise 1.0 (frozen, logged) — the guard the market states use. `df` supplies the firm and
quarter keys (CodConglomeradoPrudencial, time_id) of the rows `Z` belongs to.
"""
function build_z_evolution(df::DataFrame, Z::Matrix{Float64}, znames::Vector{String},
                           isB::AbstractVector{Bool}; transitions_path::AbstractString,
                           rho_field::String="rho")
    isfile(transitions_path) || error("bbl_transitions.json not found: $transitions_path")
    cs = get(JSON3.read(read(transitions_path, String)), :cost_shifters, nothing)
    cs === nothing && error("$(basename(transitions_path)) has no `cost_shifters` block.")
    nZ = length(znames); N = size(Z, 1)
    rhoB = ones(nZ); rhoD = ones(nZ)
    firm = string.(df.CodConglomeradoPrudencial); tid = string.(df.time_id)
    dz = Matrix{Float64}(undef, N, nZ)
    for (z, name) in enumerate(znames)
        e = get(cs, Symbol(name), nothing)
        for (κ, dest) in (("B", rhoB), ("D", rhoD))
            r = e === nothing ? nothing : get(get(e, Symbol(κ), Dict()), Symbol(rho_field), nothing)
            if r isa Real && isfinite(r) && 0.0 < Float64(r) < 1.0
                dest[z] = Float64(r)
            else
                log_status("  [ψ-Z] $name ($κ): no usable cost_shifters.$κ.$rho_field ($(r === nothing ? "missing" : r)) -> frozen")
            end
        end
        dz[:, z] .= view(Z, :, z) .- _firm_level(view(Z, :, z), firm, tid)
    end
    ev = ZEvolution(dz, rhoB, rhoD, BitVector(isB), String(transitions_path))
    for (z, name) in enumerate(znames)
        hl(r) = r < 1.0 ? string(round(log(0.5) / log(r), digits=1)) : "frozen"
        d = view(dz, :, z)
        log_status("  [ψ-Z] $(rpad(name, 26)) ρ_B=$(round(rhoB[z], digits=4)) (half-life $(hl(rhoB[z]))q) " *
                   "ρ_D=$(round(rhoD[z], digits=4)) (half-life $(hl(rhoD[z]))q) | " *
                   "|Z_0 − μ_j|: mean $(round(mean(abs.(d)), sigdigits=4)) max $(round(maximum(abs.(d)), sigdigits=4))")
    end
    return ev
end

# ==========================================================================
# Marginal-cost parameters (CF2 output) — shared by CF3 (equilibrium) and CF1-net
# ==========================================================================
"""
    load_cost_params(path, znames) -> Dict("B"=>(omega,zeta,gamma::Vector), "D"=>(…))

Read `COST_FWD/cost_params_{tag}.json` (bbl_solve.py) and align each type's
γ to the ψ-basis Z-column order `znames`.
"""
function load_cost_params(path::String, znames::Vector{String})
    isfile(path) || error("Missing cost params $path — run bbl_fwd_sim.jl → " *
                          "bbl_solve.py first (or pass --cost-json).")
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

`z_evol` moves the cost shifters in ψ3 along their estimated mean reversion (ZEvolution):
period t uses Z_t = Z_0 + (ρ_κ^t − 1)(Z_0 − μ_j). `nothing` holds Z at its launch values.
"""
function accumulate_psi(ctx::CFDemandCtx, st::DepositSimState,
                        Dep::Matrix{Float64}, markdown_q::Vector{Float64},
                        Z::Matrix{Float64};
                        beta::Float64=bbl_beta(),
                        asset_return_q::Union{Nothing,Vector{Float64}}=nothing,
                        rf_path_q::Union{Nothing,Vector{Float64}}=nothing,
                        rf_curves::Union{Nothing,Matrix{Float64}}=nothing,
                        row_curve::Union{Nothing,Vector{Int}}=nothing,
                        row_start::Union{Nothing,Vector{String}}=nothing,
                        z_evol::Union{Nothing,ZEvolution}=nothing)
    N, Tp1 = size(Dep); T = Tp1 - 1
    n_Z = size(Z, 2)
    z_evol === nothing || (size(z_evol.dz) == size(Z) && length(z_evol.isB) == N) ||
        error("z_evol is $(size(z_evol.dz)) for a Z of $(size(Z))")
    rj = asset_return_q === nothing ? zeros(N) : asset_return_q   # (r^j − r^f); 0 ⇒ funding value
    # Per-obs flat r^f for ψ4 if no path supplied: infer from accrual identity is
    # ambiguous, so default to a zero contribution unless rf_path_q is given.
    rf_flat = rf_path_q === nothing ? zeros(T) : rf_path_q

    # MULTI-START. `rf_curves` (S×T) with `row_curve` (row → curve index) gives every row the
    # forward curve of ITS OWN launch quarter, so one pass over the panel is S per-quarter
    # simulations at once. That is exact, not an approximation: a market is keyed (MCA,
    # quarter), so rows from different quarters never enter one another's share denominator
    # and never interact in the deposit recursion. Rows are then grouped by (firm, start) via
    # `row_start`, which is what gives eq:16 the time index it has always been written with
    # and never had — until now a firm's rows from every quarter were summed into ONE ψ as
    # though they were simultaneous.
    per_row_rf = rf_curves !== nothing && row_curve !== nothing
    per_row_rf && (length(row_curve) == N || error("row_curve length ≠ N"))
    per_row_rf && (size(rf_curves, 2) >= T || error("rf_curves has < T periods"))
    rfv = zeros(N)

    # Per-observation ψ blocks (then summed to firms).
    psi1 = zeros(N); psi2 = zeros(N); psi4 = zeros(N)
    psi3 = zeros(N, n_Z)
    @inbounds for t in 0:T
        bt = beta^t
        dep_t = @view Dep[:, t+1]
        # ψ1 (V_Main eq 16, row 1) carries the GROSS asset return r^j: Dep·(r^j + ρ).
        # `asset_return_q` is supplied as the NET margin (r^j − r^f), so add r^f back here.
        # The single −r^f (funding cost) is then delivered by the −(1+ζ)·ψ4 term in θ_c.
        # Using the NET rj here subtracted r^f TWICE (V/Dep = ρ − c − r^f), which made the
        # deposit franchise value negative and forced ω to its ≥0 bound in the eq:17 solve.
        if per_row_rf
            if t == 0
                fill!(rfv, 0.0)
            else
                for i in 1:N; rfv[i] = rf_curves[row_curve[i], t]; end
            end
            psi1 .+= bt .* dep_t .* (rj .+ rfv .+ markdown_q)
            psi4 .+= bt .* rfv .* dep_t
        else
            rf_t = t == 0 ? 0.0 : rf_flat[t]
            psi1 .+= bt .* dep_t .* (rj .+ rf_t .+ markdown_q)
            psi4 .+= bt .* rf_t .* dep_t
        end
        psi2 .+= bt .* dep_t
        if z_evol === nothing
            for z in 1:n_Z
                @views psi3[:, z] .+= bt .* dep_t .* Z[:, z]
            end
        else
            for z in 1:n_Z
                fB = z_evol.rhoB[z]^t - 1.0; fD = z_evol.rhoD[z]^t - 1.0   # 0 at t = 0 and ρ = 1
                for i in 1:N
                    zt = Z[i, z] + (z_evol.isB[i] ? fB : fD) * z_evol.dz[i, z]
                    psi3[i, z] += bt * dep_t[i] * zt
                end
            end
        end
    end

    # Aggregate observations → firms, or → (firm, start) when a start label is supplied.
    # The key is joined with a unit separator that cannot occur in either component, so the
    # solver can split it back apart unambiguously; `firm_of_key` saves it the trouble.
    firm_key = string.(ctx.df.CodConglomeradoPrudencial)
    keys_row = row_start === nothing ? firm_key :
               (length(row_start) == N ? firm_key .* "\x1f" .* row_start :
                error("row_start length ≠ N"))
    firms = sort(unique(keys_row))
    fidx = Dict(f => i for (i, f) in enumerate(firms))
    nf = length(firms)
    psi_firm = zeros(nf, 3 + n_Z)
    @inbounds for i in 1:N
        r = fidx[keys_row[i]]
        psi_firm[r, 1] += psi1[i]
        psi_firm[r, 2] += psi2[i]
        for z in 1:n_Z; psi_firm[r, 2+z] += psi3[i, z]; end
        psi_firm[r, 3+n_Z] += psi4[i]
    end
    firm_of_key = row_start === nothing ? firms :
                  [String(split(k, "\x1f")[1]) for k in firms]
    start_of_key = row_start === nothing ? fill("all", nf) :
                   [String(split(k, "\x1f")[2]) for k in firms]
    return (psi_firm=psi_firm, firms=firms,
            firm_of_key=firm_of_key, start_of_key=start_of_key,
            blocks=(psi1=psi1, psi2=psi2, psi3=psi3, psi4=psi4))
end
