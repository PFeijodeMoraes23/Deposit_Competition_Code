"""
cf_3_equilibrium_spreads.jl
===========================
CF3 — equilibrium deposit spreads (dynamic MPE, Egan-style forward-sim continuation).

Banks choose the spreads on the ENDOGENOUS deposit types k∈{4 (CDB), 5 (prepaid)}; the
regulated types k∈{1,2,3} pass through at their observed level. Given recovered marginal
costs ĉ (CF2), each bank best-responds by choosing (σ_{j4}, σ_{j5}) to maximize its
franchise value

    V_j(σ) = ψ_j(σ)′ · [ 1, −ω^κ, −γ^κ′, −(1+ζ^κ) ]                       (V_Main eq 16)

where ψ_j is the discounted, deposit-weighted value basis (cf_0_psi_basis / psi_under) and
κ∈{B,D} is the firm type. The equilibrium is the fixed point of the best-response map,
found by iterating firm best responses (Gauss-Seidel or Jacobi) from σ⁰=ρ̂ until
‖σ^{n+1}−σ^n‖ < tol. VALIDATION GATE (V_Main): at ĉ the fixed point should reproduce the
observed spreads ρ̂ (a bank cannot profitably deviate in equilibrium).

SCAFFOLD SIMPLIFICATIONS (first pass; refine for the headline):
  * Each firm sets ONE k=4 spread and ONE k=5 spread, uniform across its markets (a 2-D
    best response), rather than the per-market vector the draft allows for B firms.
  * `V_j(σ)` is read from a FULL-panel `psi_under` each evaluation (correct, but re-simulates
    every firm; a production version would re-simulate only firm j's markets).

Local dev (validate the machinery at logit with the logit costs already on disk):
  julia --project=. cf_3_equilibrium_spreads.jl --estim 6 --spec 12 --stage logit --R 50 \\
      --time-filter 2025Q4 --fixed-point gauss-seidel

Headline (after credible RC costs land): --stage extended --R 2000 on the cluster.
"""

include(joinpath(@__DIR__, "cost_2_fwd_sim.jl"))

using DataFrames, Statistics, Printf, LinearAlgebra
# load_cost_params / theta_c / marginal_cost_per_obs come from cf_0_psi_basis.jl (shared with CF1-net).

# ==========================================================================
# Firm value V_j(σ) under a stationary spread vector
# ==========================================================================
"""
    firm_values(psi_firm, isB, θc_B, θc_D) -> Vector{Float64}

V_j = ψ_j · θ_c(κ_j) for every firm (row of `psi_firm`), using the firm-type mask `isB`.
"""
function firm_values(psi_firm::Matrix{Float64}, isB::BitVector,
                     θc_B::Vector{Float64}, θc_D::Vector{Float64})::Vector{Float64}
    nf = size(psi_firm, 1)
    V = Vector{Float64}(undef, nf)
    @inbounds for f in 1:nf
        V[f] = dot(@view(psi_firm[f, :]), isB[f] ? θc_B : θc_D)
    end
    return V
end

# ==========================================================================
# Per-firm best response over its (σ4, σ5)
# ==========================================================================
_grid(c, w, n, lo, hi) = unique(clamp.(collect(range(c - w, c + w, length=n)), lo, hi))

"""
    firm_best_response(ctx, st, Z, mq0, σ, rows4, rows5, θc_j, fj; …) -> (s4, s5, Vstar)

Maximize firm j's value over a uniform (σ4, σ5) applied to its k∈{4,5} rows, holding
rivals' spreads fixed at `σ`, by a **bounded grid search** centered on the firm's current
spread (so the grid always contains the incumbent value — best response can stay put in
equilibrium). Each evaluation is one `psi_under`; grid keeps that count small and bounded
(`ngrid` points per choice type). Returns optimal absolute levels (NaN for an absent type).
"""
function firm_best_response(ctx, st, Z, mq0, σ, rows4, rows5, θc_j, fj;
                            beta, T, aret, rf, lo, hi, ngrid, window)
    have4 = !isempty(rows4); have5 = !isempty(rows5)
    (have4 || have5) || return (NaN, NaN, NaN)
    s4_0 = have4 ? clamp(mean(@view σ[rows4]), lo, hi) : NaN
    s5_0 = have5 ? clamp(mean(@view σ[rows5]), lo, hi) : NaN
    g4 = have4 ? _grid(s4_0, window, ngrid, lo, hi) : [NaN]
    g5 = have5 ? _grid(s5_0, window, ngrid, lo, hi) : [NaN]

    bestV = -Inf; bs4 = s4_0; bs5 = s5_0
    σ2 = copy(σ)
    for a4 in g4, a5 in g5
        have4 && (@inbounds σ2[rows4] .= a4)
        have5 && (@inbounds σ2[rows5] .= a5)
        pf, _ = psi_under(ctx, st, Z, mq0, σ2; beta=beta, T=T, asset_return_q=aret, rf_path_q=rf)
        V = dot(@view(pf[fj, :]), θc_j)
        if V > bestV; bestV = V; bs4 = a4; bs5 = a5; end
    end
    return (bs4, bs5, bestV)
end

# ==========================================================================
# MPE fixed-point solve
# ==========================================================================
function solve_equilibrium(ctx, st, Z, mq0, θc_B, θc_D, isB, firms;
                           beta, T, aret, rf, scheme::Symbol, damping::Float64,
                           tol::Float64, max_iter::Int, lo::Float64, hi::Float64,
                           ngrid::Int, window::Float64)
    N = nrow(ctx.df)
    dep = st.dep_type
    firm_key = string.(ctx.df.CodConglomeradoPrudencial)
    fidx = Dict(f => i for (i, f) in enumerate(firms))
    # Per-firm k=4 / k=5 row indices (only firms with a choice participate).
    rows4 = Dict{Int,Vector{Int}}(); rows5 = Dict{Int,Vector{Int}}()
    @inbounds for i in 1:N
        f = get(fidx, firm_key[i], 0); f == 0 && continue
        dep[i] == 4 && push!(get!(rows4, f, Int[]), i)
        dep[i] == 5 && push!(get!(rows5, f, Int[]), i)
    end
    choosers = sort(collect(union(keys(rows4), keys(rows5))))
    log_status("  [CF3] $(length(choosers)) firms with a choice spread (of $(length(firms))) | scheme=$scheme damping=$damping")

    σ = copy(ctx.rho_hat)
    hist = Float64[]
    for it in 1:max_iter
        base = scheme == :jacobi ? copy(σ) : σ     # gauss-seidel mutates σ in place
        Δmax = 0.0
        for f in choosers
            r4 = get(rows4, f, Int[]); r5 = get(rows5, f, Int[])
            θc = isB[f] ? θc_B : θc_D
            s4, s5, _ = firm_best_response(ctx, st, Z, mq0, base, r4, r5, θc, f;
                                           beta=beta, T=T, aret=aret, rf=rf,
                                           lo=lo, hi=hi, ngrid=ngrid, window=window)
            if !isnan(s4) && !isempty(r4)
                new = damping * s4 + (1 - damping) * mean(@view σ[r4])
                Δmax = max(Δmax, maximum(abs.(new .- @view σ[r4]))); σ[r4] .= new
            end
            if !isnan(s5) && !isempty(r5)
                new = damping * s5 + (1 - damping) * mean(@view σ[r5])
                Δmax = max(Δmax, maximum(abs.(new .- @view σ[r5]))); σ[r5] .= new
            end
        end
        push!(hist, Δmax)
        log_status("  [CF3] iter $it  ‖Δσ‖∞ = $(round(Δmax, sigdigits=4))")
        Δmax < tol && break
    end
    return (sigma=σ, hist=hist, choosers=choosers, rows4=rows4, rows5=rows5)
end

# ==========================================================================
# CLI
# ==========================================================================
function _parse_cf3_args()
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
        "--time-filter"; arg_type = String;  default = nothing
        "--cost-json";   arg_type = String;  default = nothing  # default COST_FWD/cost_params_{tag}.json
        "--fixed-point"; arg_type = String;  default = "gauss-seidel"  # gauss-seidel | jacobi
        "--damping";     arg_type = Float64; default = 1.0
        "--tol";         arg_type = Float64; default = 1e-4
        "--max-iter";    arg_type = Int;     default = 25
        "--br-bound";    arg_type = Float64; default = 0.30   # spread box [0, br-bound] (annual ρ)
        "--br-grid";     arg_type = Int;     default = 7      # grid points per choice type (BR search)
        "--br-window";   arg_type = Float64; default = 0.02   # BR grid half-width around current (annual ρ)
        "--n-markets";   arg_type = Int;     default = 0      # >0: restrict to K biggest markets (fast local validation)
        "--selic-shock"; arg_type = Float64; default = 0.0    # CF5: annual Selic shock (0 ⇒ base); applied to r^f in cf3_setup
        "--merge";       arg_type = String;  default = ""     # CF6: "firmA,firmB" conglomerate pair to merge
        # ── Firm-sharded Jacobi (full-panel cluster solve): one SWEEP = an array of shard jobs
        #    (each best-responds a subset of firms vs the FROZEN σ) then a merge. Sweeps chain. ──
        "--n-firm-shards"; arg_type = Int;   default = 0      # 0 = in-process solve; >0 = one shard of a Jacobi sweep
        "--firm-shard-id"; arg_type = Int;   default = 0      # 0-based; = SLURM_ARRAY_TASK_ID
        "--jacobi-merge";  action   = :store_true             # merge shard outputs → next-sweep σ
        "--write-sigma0";  action   = :store_true             # write the initial σ⁰=ρ̂ (full-N) and exit
        "--sigma-in";      arg_type = String; default = ""    # input σ parquet (empty ⇒ observed ρ̂)
        "--sigma-out";     arg_type = String; default = ""    # output σ parquet
        "--sigma-glob";    arg_type = String; default = ""    # merge: glob of this sweep's shard files
        # CF5/CF6 compare: read two solved equilibria (base + scenario σ) and report the effect.
        "--compare";       action   = :store_true
        "--sigma-base";    arg_type = String; default = ""    # base-scenario equilibrium σ parquet
        "--sigma-scn";     arg_type = String; default = ""    # shocked (CF5) / merged (CF6) equilibrium σ
    end
    return parse_args(s)
end

"""
    cf3_setup(a) -> NamedTuple

Assemble the CF3 problem — context, sim state, ψ inputs, forward r^f, marginal-cost vectors,
firms, spread box — from parsed args `a`. Shared by CF3, CF5 (pass-through) and CF6 (mergers).
`a["n-markets"]>0` restricts to the K biggest markets for fast local validation. Fields:
`(ctx, st, Z, znames, mq0, rf, aret, cost, θc_B, θc_D, firms, isB, lo, hi, out_dir, tag)`.
"""
function cf3_setup(a)
    tf = a["time-filter"] === nothing ? nothing : String[a["time-filter"]]
    # Optional: restrict to the K biggest (mca,time) markets for a FAST local validation of the
    # machinery (each best-response evals a full-panel psi_under, so a small panel is what makes
    # the fixed point tractable locally; the full run is a cluster job).
    keep = nothing
    if a["n-markets"] > 0
        input_dir, _, _ = get_paths(a["hpc"]; local_dir=a["local-dir"])
        meta = DataFrame(Parquet2.Dataset(discover_demand_parquet(input_dir, a["estim"], a["spec"])))
        tid = string.(meta.time_id)
        inwin = tf === nothing ? trues(length(tid)) : [t in tf for t in tid]
        keyv = string.(string.(meta.mca_code), "|", tid)
        counts = Dict{String,Int}()
        for i in eachindex(keyv); inwin[i] && (counts[keyv[i]] = get(counts, keyv[i], 0) + 1); end
        topmk = Set(first.(sort(collect(counts), by=x -> -x[2])[1:min(a["n-markets"], length(counts))]))
        keep = BitVector([inwin[i] && (keyv[i] in topmk) for i in eachindex(keyv)])
        log_status("  [CF3] validation subset: $(sum(keep)) rows in $(length(topmk)) of $(length(counts)) markets")
        tf = nothing
    end
    ctx = build_cf_context(a["estim"], a["spec"], a["stage"];
                           R=a["R"], seed=a["seed"], hpc=a["hpc"],
                           local_dir=a["local-dir"], suffix=a["suffix"], time_filter=tf, keep=keep)

    # CF6 scenario: merge a conglomerate pair into ONE decision-maker (relabel B→A) BEFORE firms
    # are formed, so the merged entity best-responds jointly (internalizes cross-elasticities).
    if !isempty(a["merge"])
        parts = strip.(split(a["merge"], ",")); length(parts) == 2 || error("--merge needs \"firmA,firmB\"")
        fA, fB = String(parts[1]), String(parts[2])
        key = string.(ctx.df.CodConglomeradoPrudencial)
        (fA in key && fB in key) || error("--merge firms not found: $fA / $fB")
        ctx.df[!, :CodConglomeradoPrudencial] = [k == fB ? fA : k for k in key]
        log_status("  [CF3] merged $fB → $fA")
    end

    # Per-type d̄ + markdown + forward r^f, identical to cost_2_fwd_sim (so ψ matches CF2).
    s0 = cf_model_shares(ctx)
    pop0, _    = _first_present(ctx.df, ["pop_total", "M_mt", "pop"]; default=NaN)
    phi0, _    = _first_present(ctx.df, ["phi_mt", "phi_local_mt", "phi_local", "phi"]; default=NaN)
    depact0, _ = _first_present(ctx.df, ["Dep_Act", "active_deposits", "deposit_active"]; default=NaN)
    phi0 = clamp.(phi0, 0.0, 0.999); isBcal = BitVector(Bool.(coalesce.(ctx.df.is_B, false)))
    dbar = ones(nrow(ctx.df))
    for (_, mask) in (("B", isBcal), ("D", .!isBcal))
        m = mask .& isfinite.(pop0) .& isfinite.(s0) .& isfinite.(phi0) .& isfinite.(depact0)
        den = sum((1.0 .- phi0[m]) .* pop0[m] .* s0[m]); nm = sum(max.(depact0[m], 0.0))
        dbar[mask] .= (den > 0 && isfinite(nm)) ? nm / den : 1.0
    end
    st  = load_sim_state(ctx; dbar=dbar)
    Z, znames = load_Z(ctx)
    mq0, _ = _first_present(ctx.df, ["spread_qoq", "spread_q"]; default=NaN)
    mq0 = all(isnan, mq0) ? ctx.rho_hat ./ 400.0 : clamp.(mq0 ./ 1e4, -0.1, 0.1)
    _, _, out_dir = get_paths(a["hpc"]; local_dir=a["local-dir"])
    rf = load_forward_rf(nothing, out_dir, a["horizon"], ctx; require=a["hpc"])
    # CF5 scenario: shift the whole forward Selic (r^f) path by an annual shock (0 ⇒ no-op).
    if a["selic-shock"] != 0.0
        Δq = (1.0 + a["selic-shock"])^0.25 - 1.0
        rf = rf .+ Δq
        log_status("  [CF3] Selic shock $(a["selic-shock"]) → Δr^f_q=$(round(Δq, sigdigits=3)) on forward r^f")
    end
    aret = zeros(nrow(ctx.df))

    tag = "E$(a["estim"])_spec_$(a["spec"])_$(a["stage"])$(a["suffix"])"
    cost_json = a["cost-json"] === nothing ?
        joinpath(dirname(out_dir), "COST_FWD", "cost_params_$tag.json") : a["cost-json"]
    cost = load_cost_params(cost_json, znames)
    n_Z = length(znames)
    haskey(cost, "B") && haskey(cost, "D") || error("cost_params must have B and D blocks")
    θc_B = theta_c(cost["B"], n_Z); θc_D = theta_c(cost["D"], n_Z)
    log_status("  [CF3] costs ← $(basename(cost_json)) | B ω=$(round(cost["B"].omega,sigdigits=3)) ζ=$(round(cost["B"].zeta,sigdigits=3)) | D ω=$(round(cost["D"].omega,sigdigits=3))")

    _, firms = psi_under(ctx, st, Z, mq0, ctx.rho_hat; beta=a["beta"], T=a["horizon"],
                         asset_return_q=aret, rf_path_q=rf)
    isB = firm_is_B(ctx, firms)
    # Spread box: observed range on the choice types ± a margin (a fixed [0,br-bound] box would
    # clamp firms whose ρ̂ lies outside it — CDB spreads are large / can be negative).
    ρe = ctx.rho_hat[st.endog]
    lo = minimum(ρe) - a["br-bound"]; hi = maximum(ρe) + a["br-bound"]
    return (; ctx, st, Z, znames, mq0, rf, aret, cost, θc_B, θc_D, firms, isB, lo, hi, out_dir, tag)
end

# ==========================================================================
# Firm-sharded Jacobi (full-panel cluster solve)
# ==========================================================================
# A SWEEP = an array of shard jobs, each best-responding a subset of firms against the
# FROZEN σ from the previous sweep, then a merge into the next σ. Sweeps chain (afterok).
# Correct (uses the exact full-panel psi_under) and embarrassingly parallel across firms.

_read_sigma(path) = Float64.(DataFrame(Parquet2.Dataset(path)).sigma)
_write_sigma(path, σ) = (Parquet2.writefile(path, DataFrame(sigma=σ)); log_status("  [CF3] wrote σ ($(length(σ))) → $(basename(path))"))

"""Per-firm k=4 / k=5 row indices, keyed by firm INDEX in `firms`."""
function _firm_endog_rows(ctx, st, firms)
    firm_key = string.(ctx.df.CodConglomeradoPrudencial)
    fidx = Dict(f => i for (i, f) in enumerate(firms))
    rows4 = Dict{Int,Vector{Int}}(); rows5 = Dict{Int,Vector{Int}}()
    @inbounds for i in eachindex(firm_key)
        f = get(fidx, firm_key[i], 0); f == 0 && continue
        st.dep_type[i] == 4 && push!(get!(rows4, f, Int[]), i)
        st.dep_type[i] == 5 && push!(get!(rows5, f, Int[]), i)
    end
    return rows4, rows5
end

"""One shard of a Jacobi sweep: best-respond this shard's firms vs the frozen σ; write
their new choice spreads (per endog row) to `--sigma-out`."""
function jacobi_shard(a, P)
    σ = isempty(a["sigma-in"]) ? copy(P.ctx.rho_hat) : _read_sigma(a["sigma-in"])
    length(σ) == nrow(P.ctx.df) || error("σ length $(length(σ)) ≠ N $(nrow(P.ctx.df))")
    ns = a["n-firm-shards"]; sid = a["firm-shard-id"]
    rows4, rows5 = _firm_endog_rows(P.ctx, P.st, P.firms)
    out_rows = Int[]; out_sig = Float64[]; ndone = 0
    for fi in eachindex(P.firms)
        ((fi - 1) % ns == sid) || continue
        r4 = get(rows4, fi, Int[]); r5 = get(rows5, fi, Int[])
        (isempty(r4) && isempty(r5)) && continue
        θc = P.isB[fi] ? P.θc_B : P.θc_D
        s4, s5, _ = firm_best_response(P.ctx, P.st, P.Z, P.mq0, σ, r4, r5, θc, fi;
                                       beta=a["beta"], T=a["horizon"], aret=P.aret, rf=P.rf,
                                       lo=P.lo, hi=P.hi, ngrid=a["br-grid"], window=a["br-window"])
        if !isnan(s4); for i in r4; push!(out_rows, i); push!(out_sig, s4); end; end
        if !isnan(s5); for i in r5; push!(out_rows, i); push!(out_sig, s5); end; end
        ndone += 1
        ndone % 20 == 0 && log_status("  [CF3-shard $sid] $ndone firms done")
    end
    Parquet2.writefile(a["sigma-out"], DataFrame(row_idx=out_rows, sigma=out_sig))
    log_status("  [CF3-shard $sid/$ns] $ndone firms, $(length(out_rows)) rows → $(basename(a["sigma-out"]))")
end

"""Merge a sweep's shard files onto the previous σ (`--sigma-in`), write the next σ
(`--sigma-out`), and report the sweep's ‖Δσ‖∞ on the choice types."""
function jacobi_merge(a)
    σprev = _read_sigma(a["sigma-in"]); σ = copy(σprev); n = 0
    for f in sort(glob_files(a["sigma-glob"]))
        d = DataFrame(Parquet2.Dataset(f)); n += 1
        @inbounds for (ri, sv) in zip(Int.(d.row_idx), Float64.(d.sigma)); σ[ri] = sv; end
    end
    touched = σ .!= σprev
    Δ = any(touched) ? maximum(abs.(σ[touched] .- σprev[touched])) : 0.0
    _write_sigma(a["sigma-out"], σ)
    log_status("  [CF3-merge] $n shards | ‖Δσ‖∞=$(round(Δ, sigdigits=4)) | $(sum(touched)) rows changed")
end

# Minimal glob (avoid a Glob.jl dep): split dir + wildcard filename.
function glob_files(pattern)
    dir = dirname(pattern); pat = basename(pattern)
    rx = Regex("^" * replace(replace(pat, "." => "\\."), "*" => ".*") * "\$")
    return [joinpath(dir, f) for f in readdir(isempty(dir) ? "." : dir) if occursin(rx, f)]
end

function main_cf3()
    a = _parse_cf3_args()
    a["jacobi-merge"] && return jacobi_merge(a)   # light: no context build
    scheme = Symbol(replace(a["fixed-point"], "-" => "_") == "gauss_seidel" ? :gauss_seidel : :jacobi)
    P = cf3_setup(a)
    a["write-sigma0"] && return _write_sigma(a["sigma-out"], P.ctx.rho_hat)
    a["n-firm-shards"] > 0 && return jacobi_shard(a, P)   # one shard of a sweep (cluster full-panel)
    log_status("  [CF3] spread box [$(round(P.lo,sigdigits=3)), $(round(P.hi,sigdigits=3))] | BR grid $(a["br-grid"]) × window $(a["br-window"])")
    eq = solve_equilibrium(P.ctx, P.st, P.Z, P.mq0, P.θc_B, P.θc_D, P.isB, P.firms;
                           beta=a["beta"], T=a["horizon"], aret=P.aret, rf=P.rf,
                           scheme=scheme, damping=a["damping"], tol=a["tol"],
                           max_iter=a["max-iter"], lo=P.lo, hi=P.hi,
                           ngrid=a["br-grid"], window=a["br-window"])
    ctx = P.ctx; st = P.st; out_dir = P.out_dir; tag = P.tag

    # Validation gate: does the fixed point reproduce observed spreads on the choice types?
    endog = st.endog
    dσ = eq.sigma[endog] .- ctx.rho_hat[endog]
    conv = !isempty(eq.hist) && last(eq.hist) < a["tol"]
    @printf("\n  === CF3 equilibrium (%s) : gate = recover ρ̂ at ĉ ===\n", a["fixed-point"])
    @printf("  converged=%s in %d iters (final ‖Δσ‖∞=%.3g)\n", conv, length(eq.hist),
            isempty(eq.hist) ? NaN : last(eq.hist))
    @printf("  σ* − ρ̂ on k∈{4,5}:  mean=%.4g  median=%.4g  max|·|=%.4g  (annual ρ units)\n",
            mean(dσ), median(dσ), maximum(abs.(dσ)))

    df = DataFrame(CodConglomeradoPrudencial=string.(ctx.df.CodConglomeradoPrudencial),
                   deposit_type=st.dep_type, is_B=st.is_B,
                   rho_hat=ctx.rho_hat, sigma_star=eq.sigma, endog=endog)
    cf_dir = joinpath(dirname(out_dir), "CF_FOUNDATION"); mkpath(cf_dir)
    out_path = joinpath(cf_dir, "cf3_equilibrium_$tag.parquet")
    Parquet2.writefile(out_path, df)
    log_status("  [CF3] wrote $(basename(out_path))")
    log_status("[DONE] cf_3_equilibrium_spreads")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_cf3()
end
