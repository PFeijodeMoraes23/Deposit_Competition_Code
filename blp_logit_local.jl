"""
blp_logit_local.jl
==================
Non-random-coefficients logit demand estimation for BLP (θ₂ = 0).

For specification 12 only, runs across estimation strategies E1–E5.
Three sub-models per strategy:
  (a) price-only:          δ = α · spread
  (b) price + X_COLS:      δ = α · spread + X · β
  (c) price + X_COLS + D:  δ = α · spread + X · β + γ · dummy_D_type

Outputs cluster-robust standard errors (conglomerate × year clustering),
GMM Q-values, and residuals ξ.

This script runs locally and completes in under 1 minute.

Usage
-----
  julia --project=. --threads=4 blp_logit_local.jl

References
----------
  Berry, Levinsohn & Pakes (1995, Econometrica)
  Conlon & Gortmaker (2020, RAND J. Econ.)
  Egan, Hortaçsu & Matvos (2017, AER)
"""

using Parquet2, DataFrames, LinearAlgebra, Statistics
using JSON3, Serialization, Printf, Dates

# ==========================================================================
# 0. Constants
# ==========================================================================
const SPEC_ID = 12

const X_COLS = ["fgc_covered", "has_ip", "seg_S2", "seg_S3", "seg_S4", "seg_S5",
                "log_total_assets_lag", "equity_ratio_lag"]

const IV_BLP_LOO = ["loo_log_assets", "mean_loo_log_assets",
                    "loo_equity_ratio", "mean_loo_equity_ratio",
                    "loo_basileia", "mean_loo_basileia",
                    "loo_credit_assets", "mean_loo_credit_assets",
                    "loo_npl_provision", "mean_loo_npl_provision",
                    "n_rivals"]
const IV_COST    = ["personnel_cost_ratio_lag", "admin_cost_ratio_lag",
                    "tax_cost_ratio_lag"]
const IV_CAPITAL = ["indice_basileia_lag"]

# Estimation strategies and their input filename conventions
const ESTIM_STRATEGIES = [
    (id=1, label="E1", prefix="demand_1"),
    (id=2, label="E2", prefix="demand_2"),
    (id=3, label="E3", prefix="demand_3"),
    (id=4, label="E4", prefix="demand_4"),
    (id=5, label="E5", prefix="demand_5_logistic"),
]

# Sub-model definitions
const SUB_MODELS = [
    (name="priceonly",     xcols=String[],  add_dtype=false),
    (name="full",          xcols=X_COLS,    add_dtype=false),
    (name="full_dtype",    xcols=X_COLS,    add_dtype=true),
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

# ==========================================================================
# 1. Data Loading
# ==========================================================================
function load_spec_data(estim)
    input_dir, _ = get_paths()
    fname = "$(estim.prefix)_final_spec_$(SPEC_ID).parquet"
    path  = joinpath(input_dir, fname)
    isfile(path) || error("Missing: $path")
    df = DataFrame(Parquet2.Dataset(path); copycols=true)
    return df
end

# ==========================================================================
# 2. Regressor and IV Construction
# ==========================================================================
"""Build spread vector, product-characteristics matrix, and IV matrix."""
function build_matrices(df::DataFrame, xcols::Vector{String}, add_dtype::Bool)
    N = nrow(df)

    # Spread (endogenous price variable)
    spread = Float64.(coalesce.(df.spread_qoq, 0.0))

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
Returns (theta1, se, xi, Q_value, param_names).
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
# 4. Main Loop
# ==========================================================================
function main()
    input_dir, output_dir = get_paths()
    mkpath(output_dir)

    println("=" ^ 70)
    println("  BLP Logit (Non-RC) Estimation — Spec $SPEC_ID")
    println("  $(length(ESTIM_STRATEGIES)) strategies × $(length(SUB_MODELS)) sub-models")
    println("=" ^ 70)

    all_results = Dict{String, Any}()

    for estim in ESTIM_STRATEGIES
        println("\n  ── Estimation $(estim.label) ──")

        df = try
            load_spec_data(estim)
        catch e
            println("  [!] Failed to load $(estim.label): $e. Skipping.")
            continue
        end
        println("    Loaded: $(nrow(df)) observations")

        # Construct delta = ln(s_data) (logit shares, no outside option normalization)
        is_B = BitVector(Bool.(coalesce.(df.is_B, false)))
        share_D     = Float64.(coalesce.(df.share_D,      0.0))
        share_B_cond= Float64.(coalesce.(df.share_B_cond, 0.0))

        delta = zeros(nrow(df))
        delta[.!is_B] .= log.(clamp.(share_D[.!is_B],      1e-15, Inf))
        delta[is_B]   .= log.(clamp.(share_B_cond[is_B],    1e-15, Inf))

        # Cluster identifiers: CodConglomeradoPrudencial (matches sleep estimation)
        clusters = string.(df.CodConglomeradoPrudencial)

        # Deposit types for first-stage projection
        dep_types = Int.(coalesce.(df.deposit_type, 0))

        # ── Save logit δ checkpoint for BLP σ-stage warm-start ──────────────
        # blp_estimation.jl will load this file before the sigma outer loop,
        # saving ~50–200 SQUAREM iterations per GMM objective evaluation.
        # After running this script locally, rsync logit_delta_E*.jls to the
        # cluster output dir (../data/output/) before submitting BLP jobs.
        _delta_chk_path = joinpath(output_dir,
                                   "logit_delta_E$(estim.id)_spec_$(SPEC_ID).jls")
        try
            serialize(_delta_chk_path, Dict{String,Any}(
                "delta"    => delta,
                "estim_id" => estim.id,
                "spec_id"  => SPEC_ID,
                "N"        => nrow(df),
            ))
            println("    [δ checkpoint] $(basename(_delta_chk_path)) ($(nrow(df)) obs)")
        catch _e
            println("    [δ checkpoint] WARN: could not save — $(_e)")
        end
        # Also save as version-agnostic binary (no Julia serialization version dependency)
        # rsync this .bin alongside the .jls before submitting sigma jobs to cluster
        _bin_path = replace(_delta_chk_path, ".jls" => ".bin")
        try
            open(_bin_path, "w") do io
                write(io, Int64(length(delta)))
                write(io, delta)
            end
            println("    [δ .bin] $(basename(_bin_path))")
        catch _e
            println("    [δ .bin] WARN: $(_e)")
        end

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

            # 2SLS estimation
            theta1, se, xi, Q, n_cl, G_star = estimate_2sls(delta, X_full, X_hat, Z, clusters)

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
            all_results[key] = res

            # Save individual JLS
            jls_path = joinpath(output_dir, "logit_$(key)_spec_$(SPEC_ID).jls")
            try
                serialize(jls_path, res)
            catch; end
        end
    end

    # Save combined JSON summary
    json_path = joinpath(output_dir, "logit_summary_spec_$(SPEC_ID).json")
    # Convert for JSON serialization (strip non-serializable types)
    json_data = Dict{String,Any}()
    for (k, v) in all_results
        jv = Dict{String,Any}()
        for (k2, v2) in v
            if v2 isa Vector{Float64}
                jv[k2] = round.(v2, sigdigits=8)
            else
                jv[k2] = v2
            end
        end
        json_data[k] = jv
    end
    open(json_path, "w") do f
        JSON3.pretty(f, json_data)
    end
    println("\n  Summary saved: $json_path")
    println("  [DONE] Logit estimation complete.")
end

main()
