"""
cf_0_demand_eval.jl
===================
Foundation 0a for the counterfactual pipeline: a demand-evaluation + composite
elasticity module that REUSES the estimator's exact share kernels so that
counterfactual shares are identical to the in-sample BLP shares.

What it provides
----------------
  * `build_cf_context(...)`  — load the demand parquet, the precomputed
    draws, and the estimated (θ̂₁, θ̂₂, δ̂) for a routine/stage; assemble the
    same `Precomp`/`HotBuffers`/`prod_vec` the estimator builds.
  * `cf_model_shares(ctx)`   — in-sample active shares s^Act_jkmt (validation gate).
  * `cf_shares_at(ctx, ρ′)`  — active shares under a COUNTERFACTUAL spread vector ρ′,
    holding ξ̂ and all other characteristics fixed. δ′ = δ̂ + α̂·(ρ′ − ρ̂) and μ is
    recomputed for ρ′ (the spread enters the random coefficient, coef index 1).
    This is the building block reused by cf_0_deposit_sim.jl / CF1 / CF2.
  * `cf_dDep_dρ(ctx; ...)`   — composite deposit semi-elasticity ∂Dep/∂ρ via finite
    differences on `cf_shares_at`, scaled by the deposit law of motion (1−φ)·M.

Design notes
------------
  * REUSE, do not reimplement: we `include("blp_1_estimation.jl")` (CPU baseline,
    no CUDA) and call its `compute_mu!`, `compute_model_shares!`, `build_precomp`,
    `build_regressor_matrices`, `project_endogenous_spreads`, `estimate_theta1`,
    `build_theta2_structure`, `unpack_theta2`, `load_precomputed_draws`,
    `precompute_pi_products!`. The CPU path is portable to a laptop; the GPU path is
    only needed for the heavy CF2 forward simulation (see cost_2_fwd_sim.jl).
  * Spread convention (must match the estimator exactly): ρ = spread_ann / 100
    (bps → percentage points). θ̂₁[1] = α̂ is the mean spread coefficient.
  * s^Act: for B-firms `buf.s_B` is the CONDITIONAL within-market share
    (model s^Act,B); for D-firms `buf.s_D` is the unconditional national share
    (model s^Act,D). See eqs (13-B)/(13-D) and (16-B)/(16-D) in the proposal.
  * MEMORY: peak RAM is dominated by `buf.mu` (N×R) and the π-products
    (n_pi × N×R). For E6 "extended" (n_pi=6) at R=2000 this is tens of GB → cluster.
    At R≈200–500 it is a few GB → laptop-feasible, but shares are then a
    Monte-Carlo approximation of the R=2000 estimates (use only for development /
    magnitudes, NOT for the exact-reproduction validation gate).

Usage
-----
  # Local quick-look (approximate, low memory):
  julia --project=. --threads=4 cf_0_demand_eval.jl --estim 6 --spec 12 \\
      --stage extended --R 200 --seed 42

  # Cluster exact (matches the estimated shares):
  julia --project=\${PROJECT_DIR} --threads=8 cf_0_demand_eval.jl --estim 6 \\
      --spec 12 --stage extended --R 2000 --seed 42 --hpc

Outputs `CF_FOUNDATION/shares_elas_E{estim}_spec_{spec}_{stage}{suffix}.parquet`.
"""

using Parquet2, DataFrames, Serialization, Statistics, LinearAlgebra, ArgParse

# Load CUDA + the GPU share engine ONLY when the CF opts in (CF_GPU!=0). CPU-only steps (cost2,
# cost_solve, cf1, the Jacobi init/merge) set CF_GPU=0 so they never import CUDA — critical on the
# cluster, where many concurrent CPU jobs sharing one NFS depot would otherwise stampede the Julia
# precompile/load lock just to load a package they don't use.
#
# blp_1_estimation.jl (the CPU kernels & constants: X_COLS, get_paths, log_status, …) is pulled in
# exactly once on either branch — via the engine when we take the GPU path, directly otherwise. Do
# NOT also include it unconditionally above: that would load it twice on the GPU branch (the engine
# loads it too), guarded only by the engine's isdefined check. The CPU-branch load goes through
# `Base.include(Main, …)` — the same thing a top-level `include` expands to — so the two branches
# don't read as two static edges to the same file.
const _CF_GPU_REQUESTED = get(ENV, "CF_GPU", "1") != "0"
if _CF_GPU_REQUESTED
    # blp_gpu_engine brings in CUDA + the GPU share kernels AND the CPU baseline. It LOADS on a
    # no-GPU machine (verified): CUDA.functional()==false and we fall back to the CPU share path.
    include(joinpath(@__DIR__, "blp_gpu_engine.jl"))
    import CUDA
else
    Base.include(Main, joinpath(@__DIR__, "blp_1_estimation.jl"))
end
# Use the GPU share kernel only when opted in AND a GPU is actually present. The `&&` short-circuits,
# so CUDA is never referenced when CF_GPU=0 (and thus need not be imported).
const _CF_USE_GPU = _CF_GPU_REQUESTED && (try CUDA.functional() catch; false end)

# ==========================================================================
# Context: everything needed to evaluate (counterfactual) shares
# ==========================================================================
struct CFDemandCtx
    estim          ::Int
    spec_id        ::Int
    stage          ::String
    R              ::Int
    coef_dim       ::Int
    df             ::DataFrame
    pc             ::Precomp
    buf            ::HotBuffers
    prod_vec       ::Matrix{Float64}      # col 1 = ρ̂ (spread_ann/100), cols 2:end = X
    nu_draws       ::Matrix{Float64}
    draws_3d       ::Array{Float64,3}
    obs_key_idx    ::Vector{Int}
    sigma_indices  ::Vector{Int}
    pi_interactions::Vector{Tuple{Int,Int}}
    theta1         ::Vector{Float64}
    theta2         ::Vector{Float64}
    delta_hat      ::Vector{Float64}
    rho_hat        ::Vector{Float64}      # in-sample spread (= prod_vec[:,1])
    alpha          ::Float64              # θ̂₁[1], mean spread coefficient
    gbuf           ::Any                          # GPU share buffers (nothing ⇒ CPU share path; typed Any so the struct loads without CUDA when CF_GPU=0)
end

"""
    _result_path(out_dir, estim, spec, stage, suffix) -> String

Path of the serialized RC result Dict for the counterfactuals.

After the 2026-06-25 `BLP_RESULTS/` reorg (handoff), the canonical CF input is the
consolidated **`cluster_processed/blp_E{estim}_spec_{spec}.jls`** — a byte-identical
copy of the cluster's `blp_results_E{estim}_spec_{spec}_extended.jls` (final/most-complex
`extended` stage, IFT engine; unchanged Julia Dict schema: δ̂, θ̂₁, θ̂₂, Q). It is treated
as `stage="extended"`, `suffix=""`. The metadata `.json` (per routine) and `INDEX.json`
sit alongside it; intermediate stages and the numerical-engine `*_num` results remain in
`cluster_raw/`.

`out_dir` is `get_paths()[2]` = the `BLP_RESULTS/` root. For the `extended` stage we point
at `cluster_processed/`; any other stage falls back to the legacy flat name
`blp_results_E{estim}_spec_{spec}_{stage}{suffix}.jls` (so an intermediate stage from
`cluster_raw/` still loads if explicitly requested). The `extended` consolidated path also
falls back to the flat name if the consolidated artifact is absent. (Logit is loaded by a
separate branch from `out_dir/logit/`.)
"""
function _result_path(out_dir, estim, spec_id, stage, suffix)
    flat = joinpath(out_dir, "blp_results_E$(estim)_spec_$(spec_id)_$(stage)$(suffix).jls")
    if stage == "extended"
        consolidated = joinpath(out_dir, "cluster_processed", "blp_E$(estim)_spec_$(spec_id).jls")
        return isfile(consolidated) ? consolidated : flat
    end
    return flat
end

"""
    collect_shares(buf, pc, N) -> Vector{Float64}

CPU analogue of the GPU `collect_model_shares`: gather `buf.s_B`/`buf.s_D`
(filled by `compute_model_shares!`) into an N-vector in original row order.
"""
function collect_shares(buf::HotBuffers, pc::Precomp, N::Int)::Vector{Float64}
    s = Vector{Float64}(undef, N)
    b = 0; d = 0
    @inbounds for i in 1:N
        if pc.b_mask[i]
            b += 1; s[i] = max(buf.s_B[b], 1e-15)
        else
            d += 1; s[i] = max(buf.s_D[d], 1e-15)
        end
    end
    return s
end

# ==========================================================================
# Routine → demand-parquet mapping (AUTO-DISCOVERED, new 8-routine scheme)
# ==========================================================================
"""
    discover_demand_parquet(input_dir, estim, spec_id) -> String

Find the demand parquet for routine `estim`, spec `spec_id`, by scanning
`demand_<estim>_*_spec_<spec>.parquet` (any tag; excludes `_final`), newest mtime wins.

This mirrors the BLP side's auto-discovery (the note: "the routine list — id AND
prefix — is AUTO-DISCOVERED … newest mtime per id wins so leftover old-scheme files
can't shadow rebuilds"). We do NOT use the static `DEMAND_PREFIXES` dict — its E4+
entries are stale after the 2026-06-24 relabel to the 8-routine scheme:
  E1 LocalB · E2 PooledLinear · E3 Logistic · E4 Logistic+Time ·
  E5 Single-Index · E6 Single-Index+Time (headline) · E7 Joint · E8 Joint+Time.
The cluster default set is {5, 6, 7, 8} (single-index links + their +Time variants).
"""
function discover_demand_parquet(input_dir::String, estim::Int, spec_id::Int)::String
    pat = Regex("^demand_$(estim)_.*spec_$(spec_id)\\.parquet\$")
    cands = String[]
    for f in readdir(input_dir)
        (occursin(pat, f) && !occursin("final", lowercase(f))) &&
            push!(cands, joinpath(input_dir, f))
    end
    isempty(cands) && error(
        "No demand parquet for estim=$estim spec=$spec_id in $input_dir " *
        "(scanned demand_$(estim)_*spec_$(spec_id).parquet, excluding _final). " *
        "New 8-routine scheme is auto-discovered from the parquet; ensure the routine's " *
        "demand-prep parquet exists (the sleep/BLP side produces it).")
    path = cands[argmax(mtime.(cands))]
    length(cands) > 1 && log_status("  [CF] $(length(cands)) parquets for E$estim; using newest: $(basename(path))")
    return path
end

# ==========================================================================
# Build the evaluation context (mirrors run_blp_estimation_ift_gpu assembly)
# ==========================================================================
"""
    build_cf_context(estim, spec_id, stage; R, seed, hpc, local_dir, suffix) -> CFDemandCtx

Replicates the estimator's data/precomp assembly and loads the saved (θ̂₁, θ̂₂, δ̂)
for `(estim, spec_id, stage)`. Requires the demand input parquet (local) and the
draws + result `.jls` (downloaded from the cluster).
"""
function build_cf_context(estim::Int, spec_id::Int, stage::String;
                          R::Int=2000, seed::Int=42, hpc::Bool=false,
                          local_dir=nothing,
                          time_filter::Union{Nothing,Vector{String}}=nothing,
                          keep::Union{Nothing,BitVector}=nothing,
                          placeholder::Bool=false,
                          draws_dir_override::Union{Nothing,String}=nothing,
                          suffix::String=get(ENV, "BLP_OUTPUT_SUFFIX", ""))
    input_dir, draws_dir, out_dir = get_paths(hpc; local_dir=local_dir)
    path = discover_demand_parquet(input_dir, estim, spec_id)   # new 8-routine scheme (auto-discover; not DEMAND_PREFIXES)

    df_full = DataFrame(Parquet2.Dataset(path); copycols=true)
    nrow(df_full) == 0 && error("Empty demand parquet: $path")
    N_full = nrow(df_full)

    # ── Optional subsample (KEEPS δ̂ aligned): build a row mask over the FULL
    # parquet order so we can subset both df and the saved δ̂ identically. Use for
    # local-dev memory relief (one quarter, or a market fraction). The share
    # aggregation is exact on the kept rows as long as whole (mca,time) markets are
    # kept together — `time_filter` does this; arbitrary masks may split markets. ──
    row_keep = keep === nothing ? trues(N_full) : copy(keep)
    length(row_keep) == N_full ||
        error("keep mask length $(length(row_keep)) ≠ parquet rows $N_full")
    if time_filter !== nothing
        tids = string.(df_full.time_id)
        row_keep .&= [t in time_filter for t in tids]
    end
    df = all(row_keep) ? df_full : df_full[row_keep, :]
    log_status("  [CF] Loaded $(nrow(df))/$N_full obs from $(basename(path))" *
               (all(row_keep) ? "" : "  [subsampled]"))

    sigma_indices, pi_interactions, _ = build_theta2_structure(stage)
    coef_dim = 1 + L_PROD
    N_obs    = nrow(df)

    # ── Draws + obs→draw mapping (identical to the estimator) ───────────────
    dd = draws_dir_override === nothing ? draws_dir : draws_dir_override
    nu_draws, draws_3d, key_index = load_precomputed_draws(dd, R, seed)
    mca_codes = string.(df.mca_code)
    time_ids  = string.(df.time_id)
    pad_idx   = size(draws_3d, 1)
    obs_key_idx = [get(key_index, (mca_codes[i], time_ids[i]), pad_idx) for i in 1:N_obs]

    # ── prod_vec: col 1 = ρ (spread_ann/100), cols 2:1+L_PROD = X_COLS ──────
    prod_vec = zeros(N_obs, coef_dim)
    prod_vec[:, 1] .= coalesce.(df.spread_ann, 0.0) ./ 100.0
    for (i, col) in enumerate(X_COLS)
        col in names(df) && (prod_vec[:, 1+i] .= coalesce.(df[!, col], 0.0))
    end

    # ── Precomp (shares aggregation structure) ──────────────────────────────
    spread_cols, x_mat, Z_mat, iv_cols = build_regressor_matrices(df)
    iv_avail = [c for c in iv_cols
                if std(replace(coalesce.(df[!, c], 0.0), Inf=>0.0, -Inf=>0.0)) > 1e-10]
    Z = zeros(N_obs, length(iv_avail))
    for (i, c) in enumerate(iv_avail)
        v = coalesce.(df[!, c], 0.0); replace!(v, Inf=>0.0, -Inf=>0.0); Z[:, i] .= v
    end
    H          = hcat(x_mat, Z_mat)
    dep_types  = Int.(coalesce.(df.deposit_type, 0))
    spread_hat = project_endogenous_spreads(spread_cols, H, dep_types)
    X_full     = hcat(spread_cols, x_mat)
    X_hat      = hcat(spread_hat,  x_mat)
    valid      = BitVector(all.(isfinite, eachrow(X_hat)))
    clusters   = string.(df.CodConglomeradoPrudencial)
    pc         = build_precomp(df, Z, X_full, X_hat, valid, clusters)

    N_B = sum(pc.b_mask); N_D = sum(pc.d_mask)
    n_pairs = length(pc.unique_pairs); n_times = length(pc.unique_times)
    buf = allocate_hot_buffers(N_obs, N_B, N_D, R, n_pairs, n_times, coef_dim,
                               length(pi_interactions))
    precompute_pi_products!(buf, prod_vec, draws_3d, obs_key_idx,
                            pi_interactions, coef_dim)
    # GPU share buffers (H200 on the cluster; nothing on a CPU-only machine ⇒ CPU share path).
    gbuf = _CF_USE_GPU ? allocate_gpu_buffers(buf, pc, N_obs, N_B, N_D, R, n_pairs, n_times) : nothing
    gbuf === nothing || log_status("  [CF] GPU share kernel enabled ($(CUDA.name(CUDA.device())))")

    # ── Load estimated parameters & δ̂ for this stage ───────────────────────
    # PLACEHOLDER mode (smoke test before the BLP finishes): no RC result needed —
    # set θ₂=0, θ₁=0, and δ = log-share init (the estimator's own fallback, lines
    # 357–361 of blp_1_estimation.jl). This exercises the WHOLE pipeline
    # (parquet→pc→buf→μ→shares→deposits→ψ→CF1/CF2) on real local data with draws,
    # catching plumbing bugs now; swap in δ̂/θ̂ once results land.
    rpath = _result_path(out_dir, estim, spec_id, stage, suffix)
    if !placeholder && stage == "logit"
        # LOGIT stage: δ̂ is saved as logit_delta_E{e}_spec_{s}{suffix}.bin
        # (there is no RC-style result dict). θ₂ is empty ⇒ μ=0 ⇒ plain logit shares.
        # θ₁ is only needed for SPREAD counterfactuals; pull α from the 'full' logit
        # sub-model if present, else 0 (in-sample share reproduction is unaffected).
        dbin  = joinpath(out_dir, "logit", "logit_delta_E$(estim)_spec_$(spec_id)$(suffix).bin")
        isfile(dbin) || (dbin = joinpath(out_dir, "logit_delta_E$(estim)_spec_$(spec_id)$(suffix).bin"))
        dfull = load_delta_bin(dbin)
        dfull === nothing && error("Missing logit δ̂ bin: $dbin")
        length(dfull) == N_full ||
            error("logit δ̂ length $(length(dfull)) ≠ parquet rows $N_full")
        delta_hat = all(row_keep) ? dfull : dfull[row_keep]
        theta2 = Float64[]
        theta1 = zeros(coef_dim)
        lpath  = joinpath(out_dir, "logit", "logit_E$(estim)_full_spec_$(spec_id)$(suffix).jls")
        isfile(lpath) || (lpath = joinpath(out_dir, "logit_E$(estim)_full_spec_$(spec_id)$(suffix).jls"))
        if isfile(lpath)
            lr = deserialize(lpath); t1 = get(lr, "theta1", nothing)
            (t1 !== nothing && !isempty(t1)) && (theta1[1] = Float64(t1[1]))
        else
            @warn "No logit full sub-model ($(basename(lpath))) — α=0 (in-sample shares OK; spread CFs disabled)."
        end
    elseif placeholder || !isfile(rpath)
        placeholder || @warn "RC result missing ($(basename(rpath))) — using PLACEHOLDER params (θ₂=0, δ=log-share init). Results are NOT valid until the BLP finishes."
        n_params  = length(sigma_indices) + length(pi_interactions)
        theta1    = zeros(coef_dim)
        theta2    = zeros(n_params)
        delta_hat = zeros(N_obs)
        delta_hat[pc.d_mask] .= pc.ln_s_data_D[pc.d_mask]
        delta_hat[pc.b_mask] .= pc.ln_s_data_B_cond[pc.b_mask]
    else
        res = deserialize(rpath)
        theta1 = Float64.(res["theta1"])
        # The logit-stage result carries no random coefficients → θ₂ may be absent
        # or empty; treat it as the zero-RC case (shares reduce to plain logit).
        theta2 = Float64.(get(res, "theta2", Float64[]))
        haskey(res, "delta") || error("Result $(basename(rpath)) has no 'delta' field.")
        delta_full = Float64.(res["delta"])
        length(delta_full) == N_full ||
            error("δ̂ length $(length(delta_full)) ≠ parquet rows $N_full — results/parquet mismatch")
        delta_hat = all(row_keep) ? delta_full : delta_full[row_keep]
    end
    # Sanity: the saved θ₂ structure must match the requested stage.
    (@isdefined(res) && res["sigma_indices"] != sigma_indices) &&
        @warn "sigma_indices in result ≠ build_theta2_structure($stage)"

    return CFDemandCtx(estim, spec_id, stage, R, coef_dim, df, pc, buf, prod_vec,
                       nu_draws, draws_3d, obs_key_idx, sigma_indices, pi_interactions,
                       theta1, theta2, delta_hat, copy(prod_vec[:, 1]), theta1[1], gbuf)
end

# ==========================================================================
# Share evaluation
# ==========================================================================
# Dispatch the share aggregation to the GPU kernel (H200) when a GPU context is present,
# else the CPU kernel. Both consume ctx.buf.mu (filled by compute_mu! on CPU) and write
# results into ctx.buf.s_B / ctx.buf.s_D (Float64), so callers/collect_shares are unchanged.
@inline function _cf_model_shares!(ctx::CFDemandCtx, delta::Vector{Float64})
    if ctx.gbuf === nothing
        compute_model_shares!(ctx.buf, delta, ctx.pc, ctx.R)
    else
        compute_model_shares_gpu!(ctx.buf, ctx.gbuf, delta, ctx.pc, ctx.R)
    end
end

"""
    cf_model_shares(ctx) -> Vector{Float64}

In-sample active shares s^Act at (δ̂, θ̂₂). Used by the validation gate to check
that this module reproduces the estimator's shares.
"""
function cf_model_shares(ctx::CFDemandCtx)::Vector{Float64}
    sv, pv = unpack_theta2(ctx.theta2, ctx.sigma_indices, ctx.pi_interactions)
    compute_mu!(ctx.buf, ctx.prod_vec, ctx.nu_draws, sv, ctx.sigma_indices, pv,
                ctx.R, ctx.coef_dim)
    _cf_model_shares!(ctx, ctx.delta_hat)
    return collect_shares(ctx.buf, ctx.pc, nrow(ctx.df))
end

"""
    cf_shares_at(ctx, rho_new) -> Vector{Float64}

Active shares under a counterfactual spread vector `rho_new` (same units as
ρ = spread_ann/100), holding ξ̂ and all non-spread characteristics fixed.

  δ′ = δ̂ + α̂·(ρ′ − ρ̂)      (linear mean-utility channel)
  μ  recomputed with ρ′      (spread is coef index 1 in the random coefficient)

`rho_new` must be length N (one spread per jkmt row). For the deposit simulator,
only the endogenous types k=4,5 will differ from ρ̂; regulated types are passed
through unchanged.
"""
function cf_shares_at(ctx::CFDemandCtx, rho_new::Vector{Float64})::Vector{Float64}
    length(rho_new) == nrow(ctx.df) || error("rho_new length ≠ N_obs")
    # Update the spread column and the π-products that interact with it (cidx==1).
    ctx.prod_vec[:, 1] .= rho_new
    precompute_pi_products!(ctx.buf, ctx.prod_vec, ctx.draws_3d, ctx.obs_key_idx,
                            ctx.pi_interactions, ctx.coef_dim)
    sv, pv = unpack_theta2(ctx.theta2, ctx.sigma_indices, ctx.pi_interactions)
    compute_mu!(ctx.buf, ctx.prod_vec, ctx.nu_draws, sv, ctx.sigma_indices, pv,
                ctx.R, ctx.coef_dim)
    delta_cf = ctx.delta_hat .+ ctx.alpha .* (rho_new .- ctx.rho_hat)
    _cf_model_shares!(ctx, delta_cf)
    s = collect_shares(ctx.buf, ctx.pc, nrow(ctx.df))
    # Restore in-sample spread so the context is reusable.
    ctx.prod_vec[:, 1] .= ctx.rho_hat
    precompute_pi_products!(ctx.buf, ctx.prod_vec, ctx.draws_3d, ctx.obs_key_idx,
                            ctx.pi_interactions, ctx.coef_dim)
    return s
end

"""
    cf_dDep_dρ(ctx; eps=1e-4, market_size=nothing, phi=nothing) -> NamedTuple

Composite deposit semi-elasticity ∂Dep^Act/∂ρ at the observed equilibrium, via a
UNIFORM finite-difference perturbation of all spreads by `eps` (one extra forward
pass). Returns the share derivative `ds_dρ = (s(ρ̂+eps) − s(ρ̂))/eps` and, when
`market_size` (M_mt proxy) and `phi` (φ̂_mt) are supplied, the deposit semi-elasticity
`dDep_dρ = (1−φ)·M·ds_dρ` per the law of motion (eq 8).

NOTE: this is the response to a *uniform* spread shift, which folds in within-market
cross-substitution; it is the right object for aggregate pass-through intuition but
is NOT the pure own-spread Jacobian diagonal. The exact own/cross Jacobian is built
inside CF2 by perturbing each strategy and re-simulating (see cost_2_fwd_sim.jl).
"""
function cf_dDep_dρ(ctx::CFDemandCtx; eps::Float64=1e-4,
                    market_size=nothing, phi=nothing)
    s0 = cf_model_shares(ctx)
    s1 = cf_shares_at(ctx, ctx.rho_hat .+ eps)
    ds = (s1 .- s0) ./ eps
    dDep = nothing
    if market_size !== nothing && phi !== nothing
        dDep = (1.0 .- phi) .* market_size .* ds
    end
    return (s0=s0, ds_dρ=ds, dDep_dρ=dDep)
end

# ==========================================================================
# Export (validation artifact)
# ==========================================================================
function cf_export_shares(ctx::CFDemandCtx; out_path::String)
    s = cf_model_shares(ctx)
    out = DataFrame(
        CodConglomeradoPrudencial = string.(ctx.df.CodConglomeradoPrudencial),
        mca_code      = string.(ctx.df.mca_code),
        time_id       = string.(ctx.df.time_id),
        deposit_type  = Int.(coalesce.(ctx.df.deposit_type, 0)),
        is_B          = Bool.(coalesce.(ctx.df.is_B, false)),
        spread        = ctx.rho_hat,
        s_act_model   = s,
    )
    # Carry the data-implied shares when present, for the validation comparison.
    "share_B_cond" in names(ctx.df) && (out.share_B_cond = coalesce.(ctx.df.share_B_cond, NaN))
    "share_D"      in names(ctx.df) && (out.share_D      = coalesce.(ctx.df.share_D, NaN))
    mkpath(dirname(out_path))
    Parquet2.writefile(out_path, out)
    log_status("  [CF] Wrote $(nrow(out)) rows → $(basename(out_path))")
    return out_path
end

# ==========================================================================
# CLI
# ==========================================================================
function _parse_cf_args()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--estim";     arg_type = Int;    default = 6
        "--spec";      arg_type = Int;    default = 12
        "--stage";     arg_type = String; default = "extended"
        "--R";         arg_type = Int;    default = 2000
        "--seed";      arg_type = Int;    default = 42
        "--hpc";       action   = :store_true
        "--local-dir"; arg_type = String; default = nothing
        "--draws-dir"; arg_type = String; default = nothing   # override draws location
        "--suffix";    arg_type = String; default = ""
        "--verify-gpu"; action  = :store_true                 # assert GPU shares == CPU shares, then exit
    end
    return parse_args(s)
end

"""
    verify_cf_gpu(ctx; rtol) — assert the GPU share kernel matches the CPU one at (δ̂, θ̂₂).

Run once on the cluster (`cf_0_demand_eval.jl … --verify-gpu`) to gate the GPU path. Errors if
the max relative / log-floor share difference exceeds `rtol`. No-op (warns) without a GPU.
"""
function verify_cf_gpu(ctx::CFDemandCtx; rtol::Float64=1e-7)
    ctx.gbuf === nothing && (@warn "  [CF] --verify-gpu: no GPU present (CUDA.functional()==false) — nothing to verify"; return)
    sv, pv = unpack_theta2(ctx.theta2, ctx.sigma_indices, ctx.pi_interactions)
    compute_mu!(ctx.buf, ctx.prod_vec, ctx.nu_draws, sv, ctx.sigma_indices, pv, ctx.R, ctx.coef_dim)
    verify_gpu_shares(ctx.buf, ctx.gbuf, ctx.delta_hat, ctx.pc, ctx.R; rtol=rtol)  # errors on mismatch
    log_status("  [CF] --verify-gpu PASSED: GPU shares == CPU shares to rtol=$rtol")
end

function main_cf_demand()
    a = _parse_cf_args()
    log_status("CF demand-eval — E$(a["estim"]) spec $(a["spec"]) stage $(a["stage"]) " *
               "| R=$(a["R"]) seed=$(a["seed"]) hpc=$(a["hpc"])")
    ctx = build_cf_context(a["estim"], a["spec"], a["stage"];
                           R=a["R"], seed=a["seed"], hpc=a["hpc"],
                           local_dir=a["local-dir"], suffix=a["suffix"],
                           draws_dir_override=a["draws-dir"])
    if a["verify-gpu"]; verify_cf_gpu(ctx); return; end
    _, _, out_dir = get_paths(a["hpc"]; local_dir=a["local-dir"])
    cf_dir = joinpath(dirname(out_dir), "CF_FOUNDATION")
    out_path = joinpath(cf_dir,
        "shares_elas_E$(a["estim"])_spec_$(a["spec"])_$(a["stage"])$(a["suffix"]).parquet")
    cf_export_shares(ctx; out_path=out_path)
    log_status("[DONE] CF demand-eval")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_cf_demand()
end
