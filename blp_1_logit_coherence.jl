"""
blp_1_logit_coherence.jl
========================
Non-random-coefficients logit demand estimation for BLP (θ₂ = 0) — COHERENCE build.

This is the post-"coherence-fix" replacement for blp_1_logit_local.jl. The sleepiness
function was re-estimated locally, changing the demand-prep outputs. There are now
**3 estimation routines** (instead of 5), and the demand-prep writes self-contained
`demand_X_spec_12.parquet` files (NOT the old `demand_X_final_spec_12.parquet`):

  E1  Local B-type          → demand_1_spec_12.parquet
  E2  Pooled B+D Linear     → demand_2_spec_12.parquet
  E3  Pooled B+D Logistic   → demand_3_logistic_spec_12.parquet

Each parquet already carries every column the logit needs (spread_ann in bps, share_D /
share_B_cond, is_B, deposit_type, CodConglomeradoPrudencial, the X_COLS, and all
LOO/cost/capital instruments), so no separate "finalization" step is required.

This file doubles as the **shared library** for the three thin per-routine entrypoints
(blp_1_logit_e{1,2,3}_coherence.jl). They `include` this file and call `run_strategy(...)`.
When executed directly, this file's `main()` runs all three routines and writes the
combined summary. All outputs are suffixed `_coherence` and are fully parallel to the old
pipeline — nothing canonical is clobbered.

Four sub-models per routine (identical keys to the legacy build, so the LaTeX table
generator works unchanged):
  (a) priceonly:           δ = α · spread
  (b) core:                δ = α · spread + X_core · β
  (c) full:                δ = α · spread + X · β
  (d) full_dtype:          δ = α · spread + X · β + γ · dummy_D_type

Outputs cluster-robust standard errors (conglomerate clustering), GMM Q-values, and
IK2016 effective clusters G*.

Usage
-----
  # All three routines + combined summary:
  julia --project=. --threads=auto blp_1_logit_coherence.jl

  # A single routine (via the thin entrypoints):
  julia --project=. blp_1_logit_e3_coherence.jl

References
----------
  Berry, Levinsohn & Pakes (1995, Econometrica)
  Conlon & Gortmaker (2020, RAND J. Econ.)
  Egan, Hortaçsu & Matvos (2017, AER)
  Imbens & Kolesár (2016, REStat)
"""

using Parquet2, DataFrames, LinearAlgebra, Statistics
using JSON3, Serialization, Printf, Dates

# ==========================================================================
# 0. Constants
# ==========================================================================
const SPEC_ID = 12
const COH_SUFFIX = "coherence"

const X_COLS = ["fgc_covered", "has_ip", "seg_S2", "seg_S3", "seg_S4", "seg_S5",
                "log_total_assets_lag"]
const CORE_COLS = ["fgc_covered", "has_ip", "log_total_assets_lag"]

const IV_BLP_LOO = ["loo_log_assets", "mean_loo_log_assets",
                    "loo_equity_ratio", "mean_loo_equity_ratio",
                    "loo_basileia", "mean_loo_basileia",
                    "loo_credit_assets", "mean_loo_credit_assets",
                    "loo_npl_provision", "mean_loo_npl_provision",
                    "n_rivals"]
const IV_COST    = ["personnel_cost_ratio_lag", "admin_cost_ratio_lag",
                    "tax_cost_ratio_lag"]
const IV_CAPITAL = ["indice_basileia_lag"]

# Coherence estimation routines (3, not 5). E3 input is the logistic-pooled prep.
const ESTIM_STRATEGIES = [
    (id=1, label="E1", prefix="demand_1"),
    (id=2, label="E2", prefix="demand_2"),
    (id=3, label="E3", prefix="demand_3_logistic"),
]

# Sub-model definitions (keys must match make_blp_logit_table.py expectations)
const SUB_MODELS = [
    (name="priceonly",  xcols=String[],    add_dtype=false),
    (name="core",       xcols=CORE_COLS,   add_dtype=false),
    (name="full",       xcols=X_COLS,      add_dtype=false),
    (name="full_dtype", xcols=X_COLS,      add_dtype=true),
]

# ==========================================================================
# 0b. Paths
# ==========================================================================
function get_paths()
    _root     = dirname(dirname(dirname(abspath(@__FILE__))))
    data_dir  = joinpath(_root, "BCB", "Egan_et_al_2025_Rep", "processed")
    input_dir = joinpath(data_dir, "ESTIMATION_OUTPUT", "DEMAND_PREP")
    output_dir= joinpath(data_dir, "ESTIMATION_OUTPUT", "BLP_RESULTS")
    return input_dir, output_dir
end

# Canonical combined-summary path for the coherence build.
combined_summary_path() = joinpath(get_paths()[2],
                                   "logit_summary_spec_$(SPEC_ID)_$(COH_SUFFIX).json")

# ==========================================================================
# 1. Data Loading
# ==========================================================================
"""Load the coherence demand-prep parquet for one routine (drops the old `_final`)."""
function load_spec_data(estim)
    input_dir, _ = get_paths()
    fname = "$(estim.prefix)_spec_$(SPEC_ID).parquet"
    path  = joinpath(input_dir, fname)
    isfile(path) || error("Missing coherence demand-prep parquet: $path")
    df = DataFrame(Parquet2.Dataset(path); copycols=true)
    return df
end

# ==========================================================================
# 2. Regressor and IV Construction
# ==========================================================================
"""Build spread vector, product-characteristics matrix, and IV matrix."""
function build_matrices(df::DataFrame, xcols::Vector{String}, add_dtype::Bool)
    N = nrow(df)

    # Spread in percentage points (÷100 to convert from bps stored in parquet)
    spread = Float64.(coalesce.(df.spread_ann, 0.0)) ./ 100.0

    # Product characteristics matrix X
    n_x = length(xcols) + (add_dtype ? 1 : 0)
    X = zeros(N, n_x)
    for (i, col) in enumerate(xcols)
        if col in names(df)
            X[:, i] .= Float64.(coalesce.(df[!, col], 0.0))
        end
    end
    if add_dtype
        # D-type dummy: is_B == false (i.e., national D-type institutions)
        is_B = Bool.(coalesce.(df.is_B, false))
        X[:, n_x] .= Float64.(.!is_B)
    end

    # Full regressor matrix: [spread, X]
    X_full = hcat(spread, X)

    # IV matrix
    all_iv_names = vcat(IV_BLP_LOO, IV_COST, IV_CAPITAL)
    iv_avail = [c for c in all_iv_names if c in names(df)]
    # Drop IVs with zero variance
    iv_avail = [c for c in iv_avail if
        std(replace(Float64.(coalesce.(df[!, c], 0.0)), Inf=>0.0, -Inf=>0.0)) > 1e-10]

    Z = zeros(N, length(iv_avail))
    for (i, col) in enumerate(iv_avail)
        v = Float64.(coalesce.(df[!, col], 0.0))
        replace!(v, Inf=>0.0, -Inf=>0.0)
        Z[:, i] .= v
    end

    return spread, X, X_full, Z, iv_avail
end

"""First-stage projection of endogenous spreads for k=4,5."""
function project_spreads(spread::Vector{Float64}, X::Matrix{Float64},
                         Z::Matrix{Float64}, dep_types::Vector{Int})
    spread_hat = copy(spread)
    H = hcat(X, Z)  # full set of exogenous regressors + instruments

    for k_val in [4, 5]
        k_mask = dep_types .== k_val
        any(k_mask) || continue
        H_k      = H[k_mask, :]
        spread_k = spread[k_mask]
        valid    = all.(isfinite, eachrow(H_k)) .& isfinite.(spread_k)
        sum(valid) > size(H_k, 2) || continue
        try
            beta_fs = H_k[valid, :] \ spread_k[valid]
            spread_hat[k_mask] .= H_k * beta_fs
        catch; end
    end
    return spread_hat
end

# ==========================================================================
# 3. 2SLS/IV Estimation with Cluster-Robust SEs
# ==========================================================================
"""
2SLS estimation: δ = X_hat \\ delta, where X_hat uses projected spreads.
Returns (theta1, se, xi, Q_value, n_clusters, G_star).
"""
function estimate_2sls(delta::Vector{Float64}, X_full::Matrix{Float64},
                       X_hat::Matrix{Float64}, Z::Matrix{Float64},
                       clusters::Vector{String})
    N, K = size(X_full)
    L    = size(Z, 2)

    # Validity mask
    valid = all.(isfinite, eachrow(X_hat)) .& isfinite.(delta) .&
            all.(isfinite, eachrow(Z))

    X_v  = X_hat[valid, :]
    Xf_v = X_full[valid, :]
    d_v  = delta[valid]
    Z_v  = Z[valid, :]
    cl_v = clusters[valid]
    N_v  = sum(valid)

    # --- 2SLS point estimates ---
    # Project X onto Z-space: X̃ = Z(Z'Z)⁻¹Z'X
    # Avoid materializing the N×N projection matrix Pz = Z(Z'Z)⁻¹Z'
    ZtZ     = Z_v' * Z_v
    ZtZ_inv = try inv(ZtZ) catch; pinv(ZtZ) end
    ZtX     = Z_v' * X_v             # (L, K)
    X_proj  = Z_v * (ZtZ_inv * ZtX)  # (N_v, K) — projected regressors

    # IV estimator: (X̃'X)⁻¹ X̃'δ
    XpX    = X_proj' * X_v
    XpX_inv= try inv(XpX) catch; pinv(XpX) end
    theta1 = XpX_inv * (X_proj' * d_v)

    # Residuals (use original X, not projected)
    xi_v  = d_v .- Xf_v * theta1
    xi    = delta .- X_full * theta1

    # --- GMM Q-value: g'Wg, g = Z'ξ/N ---
    W      = try inv(Z_v' * Z_v ./ N_v) catch; Matrix(1.0I, L, L) end
    g      = vec(mean(xi_v .* Z_v; dims=1))
    Q      = dot(g, W * g)

    # --- Cluster-robust SEs (sandwich estimator) ---
    # Bread: (X̃'X)⁻¹
    # Meat: Σ_c (X̃_c' ξ_c)(X̃_c' ξ_c)'
    unique_cl = unique(cl_v)
    G         = length(unique_cl)
    meat      = zeros(K, K)
    for c in unique_cl
        c_mask  = cl_v .== c
        X_c     = X_proj[c_mask, :]
        xi_c    = xi_v[c_mask]
        score_c = X_c' * xi_c     # (K,)
        meat   .+= score_c * score_c'
    end
    # Small-sample correction: G/(G-1) × N/(N-K)
    correction = (G / (G - 1)) * (N_v / (N_v - K))
    V_cluster  = XpX_inv * meat * XpX_inv' .* correction
    se         = sqrt.(max.(diag(V_cluster), 0.0))

    # IK2016 effective clusters: G* = G / (1 + CV²)
    cl_sizes = [count(==(c), cl_v) for c in unique_cl]
    mu_G     = Statistics.mean(cl_sizes)
    cv_G     = mu_G > 0 ? Statistics.std(cl_sizes; corrected=false) / mu_G : 0.0
    G_star   = max(1.0, G / (1.0 + cv_G^2))

    return theta1, se, xi, Q, G, G_star
end

# ==========================================================================
# 4. Per-routine driver
# ==========================================================================
"""
Estimate all four sub-models for a single coherence routine.

Loads the routine's demand-prep parquet, builds δ from data shares, runs 2SLS for each
sub-model, writes per-(routine, sub-model) JLS and the δ warm-start checkpoints (all
`_coherence`), and returns a Dict keyed `"{label}_{submodel}"`.
"""
function run_strategy(estim)
    _, output_dir = get_paths()
    mkpath(output_dir)

    println("\n  ── Estimation $(estim.label) [coherence] ──")

    df = load_spec_data(estim)
    println("    Loaded: $(nrow(df)) observations ($(estim.prefix)_spec_$(SPEC_ID).parquet)")

    # Construct delta = ln(s_data): D-type use share_D, B-type use conditional share_B_cond
    is_B = BitVector(Bool.(coalesce.(df.is_B, false)))
    share_D      = Float64.(coalesce.(df.share_D,      0.0))
    share_B_cond = Float64.(coalesce.(df.share_B_cond, 0.0))

    delta = zeros(nrow(df))
    delta[.!is_B] .= log.(clamp.(share_D[.!is_B],      1e-15, Inf))
    delta[is_B]   .= log.(clamp.(share_B_cond[is_B],   1e-15, Inf))

    # Cluster identifiers: CodConglomeradoPrudencial (matches sleep estimation)
    clusters = string.(df.CodConglomeradoPrudencial)

    # Deposit types for first-stage projection
    dep_types = Int.(coalesce.(df.deposit_type, 0))

    # ── Save logit δ checkpoint for BLP σ-stage warm-start (coherence-suffixed) ──
    delta_chk_path = joinpath(output_dir,
        "logit_delta_E$(estim.id)_spec_$(SPEC_ID)_$(COH_SUFFIX).jls")
    try
        serialize(delta_chk_path, Dict{String,Any}(
            "delta"    => delta,
            "estim_id" => estim.id,
            "spec_id"  => SPEC_ID,
            "N"        => nrow(df),
            "build"    => COH_SUFFIX,
        ))
        println("    [δ checkpoint] $(basename(delta_chk_path)) ($(nrow(df)) obs)")
    catch _e
        println("    [δ checkpoint] WARN: could not save — $(_e)")
    end
    # Version-agnostic binary (no Julia serialization-version dependency)
    bin_path = replace(delta_chk_path, ".jls" => ".bin")
    try
        open(bin_path, "w") do io
            write(io, Int64(length(delta)))
            write(io, delta)
        end
        println("    [δ .bin] $(basename(bin_path))")
    catch _e
        println("    [δ .bin] WARN: $(_e)")
    end

    routine_results = Dict{String, Any}()

    for sm in SUB_MODELS
        println("    Sub-model: $(sm.name)")

        # Build regressor and IV matrices
        spread, X, X_full, Z, iv_names = build_matrices(df, sm.xcols, sm.add_dtype)

        # First-stage projection of spreads
        spread_hat = project_spreads(spread, X, Z, dep_types)
        X_hat = hcat(spread_hat, X)

        # Parameter names
        pnames = vcat(["alpha"], sm.xcols)
        if sm.add_dtype
            pnames = vcat(pnames, ["dummy_D_type"])
        end

        # 2SLS estimation (residual ξ unused here; estimate_2sls returns it for callers
        # that need it, e.g. a future BLP moment-construction step)
        theta1, se, _xi, Q, n_cl, G_star = estimate_2sls(delta, X_full, X_hat, Z, clusters)

        # t-statistics
        tstat = theta1 ./ max.(se, 1e-15)

        # Display
        println("      Q-value: $(round(Q, sigdigits=6))")
        println("      ┌─────────────────────────────┬───────────┬──────────┬──────────┐")
        println("      │ Parameter                   │   Coeff   │    SE    │  t-stat  │")
        println("      ├─────────────────────────────┼───────────┼──────────┼──────────┤")
        for (i, pn) in enumerate(pnames)
            @printf("      │ %-27s │ %9.5f │ %8.5f │ %8.3f │\n",
                    pn, theta1[i], se[i], tstat[i])
        end
        println("      └─────────────────────────────┴───────────┴──────────┴──────────┘")

        # Store results
        key = "$(estim.label)_$(sm.name)"
        res = Dict{String,Any}(
            "estim"       => estim.label,
            "estim_id"    => estim.id,
            "spec_id"     => SPEC_ID,
            "build"       => COH_SUFFIX,
            "sub_model"   => sm.name,
            "param_names" => pnames,
            "theta1"      => theta1,
            "se"          => se,
            "tstat"       => tstat,
            "Q_value"     => Q,
            "n_obs"       => nrow(df),
            "n_iv"        => length(iv_names),
            "iv_names"    => iv_names,
            "n_clusters"  => n_cl,
            "G_star"      => G_star,
            "converged"   => true,
        )
        routine_results[key] = res

        # Save individual JLS (coherence-suffixed)
        jls_path = joinpath(output_dir,
            "logit_$(key)_spec_$(SPEC_ID)_$(COH_SUFFIX).jls")
        try
            serialize(jls_path, res)
        catch; end
    end

    return routine_results
end

# ==========================================================================
# 5. Combined-summary serialization
# ==========================================================================
"""Convert a results Dict into JSON-friendly form (round Float64 vectors)."""
function _to_json_data(all_results::Dict{String,Any})
    json_data = Dict{String,Any}()
    for (k, v) in all_results
        jv = Dict{String,Any}()
        for (k2, v2) in v
            jv[k2] = v2 isa Vector{Float64} ? round.(v2, sigdigits=8) : v2
        end
        json_data[k] = jv
    end
    return json_data
end

"""Write the combined coherence summary JSON from a full results Dict."""
function write_combined_summary(all_results::Dict{String,Any})
    json_path = combined_summary_path()
    open(json_path, "w") do f
        JSON3.pretty(f, _to_json_data(all_results))
    end
    println("\n  Combined summary saved: $(basename(json_path))")
    return json_path
end

"""
Merge one routine's keys into the existing combined coherence summary (creating it if
absent). Lets single-routine entrypoints compose into the same summary file the table
generator reads.
"""
function merge_into_combined_summary(routine_results::Dict{String,Any})
    json_path = combined_summary_path()
    existing = Dict{String,Any}()
    if isfile(json_path)
        try
            existing = copy(JSON3.read(read(json_path, String), Dict{String,Any}))
        catch _e
            println("  [merge] WARN: could not read existing summary — $(_e). Recreating.")
            existing = Dict{String,Any}()
        end
    end
    for (k, v) in _to_json_data(routine_results)
        existing[k] = v
    end
    open(json_path, "w") do f
        JSON3.pretty(f, existing)
    end
    println("  [merge] Updated $(basename(json_path)) with $(length(routine_results)) sub-model key(s).")
    return json_path
end

# ==========================================================================
# 6. Orchestrator entrypoint
# ==========================================================================
function main()
    _, output_dir = get_paths()
    mkpath(output_dir)

    println("=" ^ 70)
    println("  BLP Logit (Non-RC) — COHERENCE build — Spec $SPEC_ID")
    println("  $(length(ESTIM_STRATEGIES)) routines × $(length(SUB_MODELS)) sub-models")
    println("=" ^ 70)

    all_results = Dict{String, Any}()
    for estim in ESTIM_STRATEGIES
        try
            merge!(all_results, run_strategy(estim))
        catch e
            println("  [!] Routine $(estim.label) failed: $e. Skipping.")
        end
    end

    write_combined_summary(all_results)
    println("  [DONE] Coherence logit estimation complete.")
end

# Only auto-run when executed directly (not when `include`d by a per-routine entrypoint).
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
