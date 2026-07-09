"""
cf_0_psi_basis.jl
=================
Foundation 0c: accumulate the BBL value-function basis ψ along a simulated deposit
path. The value function is LINEAR in the cost parameters (eqs 17-B / 17-D):

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
    used by the policy-function step (estimation_1_cost_1_polfunc.py COST_SHIFTERS +
    CAPITAL_WHOLESALE), resolved by name from the parquet/sidecar.
  * β (per quarter, default 0.9), horizon T (default 50), r^f path (default flat).

This module is `include`d by cost_2_fwd_sim.jl; it is not a standalone entry point.
"""

include(joinpath(@__DIR__, "cf_0_deposit_sim.jl"))

using DataFrames, LinearAlgebra, Statistics
import JSON3

# Cost-shifter (Z) columns — mirror estimation_1_cost_1_polfunc.py so γ is comparable.
const Z_COST_COLS = ["personnel_cost_ratio_lag", "admin_cost_ratio_lag",
                     "tax_cost_ratio_lag", "indice_basileia_lag",
                     "wholesale_ratio_lag", "lci_lca_ratio_lag"]

"""
    load_Z(ctx; sidecar=nothing) -> (Z::Matrix, names::Vector{String})

Assemble the per-observation cost-shifter matrix Z (N × n_Z) from the parquet (or a
sidecar), keeping only columns that are present and non-degenerate.
"""
function load_Z(ctx::CFDemandCtx; sidecar::Union{Nothing,DataFrame}=nothing)
    df = sidecar === nothing ? ctx.df : hcat(ctx.df, sidecar; makeunique=true)
    cols = [c for c in Z_COST_COLS if c in names(df)]
    isempty(cols) && error("No cost-shifter (Z) columns found; pass a sidecar with $(Z_COST_COLS).")
    Z = zeros(nrow(df), length(cols))
    for (i, c) in enumerate(cols)
        v = Float64.(coalesce.(df[!, c], 0.0)); replace!(v, Inf=>0.0, -Inf=>0.0)
        Z[:, i] .= v
    end
    log_status("  [ψ] Z cost-shifters: $(cols)")
    return Z, cols
end

# ==========================================================================
# Marginal-cost parameters (CF2 output) — shared by CF3 (equilibrium) and CF1-net
# ==========================================================================
"""
    load_cost_params(path, znames) -> Dict("B"=>(omega,zeta,gamma::Vector), "D"=>(…))

Read `COST_FWD/cost_params_{tag}.json` (estimation_1_cost_3_solve.py) and align each type's
γ to the ψ-basis Z-column order `znames`.
"""
function load_cost_params(path::String, znames::Vector{String})
    isfile(path) || error("Missing cost params $path — run cost_2_fwd_sim.jl → " *
                          "estimation_1_cost_3_solve.py first (or pass --cost-json).")
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
        # ψ1: Dep·(r^j + ρ). (r^j defaults to 0 over r^f ⇒ pure markdown value.)
        psi1 .+= bt .* dep_t .* (rj .+ markdown_q)
        psi2 .+= bt .* dep_t
        for z in 1:n_Z
            @views psi3[:, z] .+= bt .* dep_t .* Z[:, z]
        end
        rf_t = t == 0 ? 0.0 : rf_flat[t]
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
