"""
blp_1_logit.jl
==============
Non-random-coefficients logit demand estimation for BLP (θ₂ = 0).

The sleepiness function was re-estimated locally, changing the demand-prep outputs. The
estimation routines are **auto-discovered** from the demand-prep parquets (see
`discover_estim_strategies()` — id AND prefix, newest file per id wins), so a relabelled
or new routine needs no code change, just its `demand_<id>_*_spec_12.parquet`. The
2026-06-24 scheme (base links + their +Time variants):

  E1 Local B-type   E2 Pooled Linear
  E3 Pooled Logistic            E4 Pooled Logistic + Time
  E5 Pooled Single-Index        E6 Pooled Single-Index + Time
  E7 Pooled Joint Single-Index  E8 Pooled Joint Single-Index + Time

Each parquet already carries every column the logit needs (spread_ann in bps, share_D /
share_B_cond, is_B, deposit_type, CodConglomeradoPrudencial, the X_COLS, and all
LOO/cost/capital instruments), so no separate "finalization" step is required. The
link-based routines (E4+) are produced by estimation_demand_link_common.py, which mirrors
estimation_3's demand prep exactly (same columns/scaling), so the logit treats every
routine identically.

Run all discovered routines with `julia blp_1_logit.jl`, or a single one with
`julia blp_1_logit.jl --est 8`.

Four sub-models per routine (the in-file LaTeX table generator below reads these keys):
  (a) priceonly:           δ = α · spread
  (b) core:                δ = α · spread + X_core · β
  (c) full:                δ = α · spread + X · β
  (d) full_dtype:          δ = α · spread + X · β + γ · dummy_D_type

Outputs cluster-robust standard errors (conglomerate clustering), GMM Q-values, and
IK2016 effective clusters G*.

Usage
-----
  # All discovered routines + combined summary + LaTeX tables:
  julia --project=. --threads=auto blp_1_logit.jl

  # A single routine:
  julia --project=. blp_1_logit.jl --est 8

  # Rebuild all LaTeX tables from the existing combined summary (no estimation):
  julia --project=. blp_1_logit.jl --tables-only

LaTeX outputs (→ ESTIMATION_OUTPUT/Rout + Drafts/Deposit Competition):
  est{id}_spec12_logit.tex                 per-routine, 4 sub-model columns
  est5-8_spec12_logit_comparison.tex       cross-routine, `+ D-Type` column each
                                           (tab:demand_logit_spec12_comparison)

References
----------
  Berry, Levinsohn & Pakes (1995, Econometrica)
  Conlon & Gortmaker (2020, RAND J. Econ.)
  Egan, Hortaçsu & Matvos (2017, AER)
  Imbens & Kolesár (2016, REStat)
"""

using Parquet2, DataFrames, LinearAlgebra, Statistics
using JSON3, Serialization, Printf, Dates
using Distributions   # t/χ² p-values for the LaTeX result tables
using Random          # MersenneTwister for the wild cluster bootstrap

# ==========================================================================
# 0. Constants
# ==========================================================================
const SPEC_ID = 12

const X_COLS = ["fgc_covered", "has_ip", "seg_S2", "seg_S3", "seg_S4", "seg_S5",
                "log_total_assets_lag"]
const CORE_COLS = ["fgc_covered", "has_ip", "log_total_assets_lag"]

# Full instrument set (15 = these 11 + IV_COST + IV_CAPITAL). A `mean_loo_`-trimmed set was TESTED
# 2026-07-09 and REJECTED: dropping the aggregates roughly doubled the logit α SE and made it
# insignificant (t −2.2→−0.75), even though the subsample eff-F rose (13→21, a mechanical dilution
# effect) — the `mean_loo_` contribute identifying variation despite their collinearity. Kept only
# as a robustness discussion (review §4). See weak_iv_analysis.py PARSIMONIOUS_IV for the diagnostic.
const IV_BLP_LOO = ["loo_log_assets", "mean_loo_log_assets",
                    "loo_equity_ratio", "mean_loo_equity_ratio",
                    "loo_basileia", "mean_loo_basileia",
                    "loo_credit_assets", "mean_loo_credit_assets",
                    "loo_npl_provision", "mean_loo_npl_provision",
                    "n_rivals"]
const IV_COST    = ["personnel_cost_ratio_lag", "admin_cost_ratio_lag",
                    "tax_cost_ratio_lag"]
const IV_CAPITAL = ["indice_basileia_lag"]
# ESTBAN branch-competition IV (panel_10): log1p lagged RIVAL-branch count in the MCA. The only
# instrument with within-conglomerate variation (across municipalities) — independent + relevant for
# demand-deposit spreads; validated against reserve requirements (rejected: macro-endogenous).
const IV_ESTBAN  = ["estban_rival_branches_lag"]

# Estimation routines are AUTO-DISCOVERED from the demand-prep parquets — see
# discover_estim_strategies() / const ESTIM_STRATEGIES below (defined after get_paths()).

# Sub-model definitions (keys consumed by the in-file LaTeX table generator below)
const SUB_MODELS = [
    (name="priceonly",  xcols=String[],    add_dtype=false),
    (name="core",       xcols=CORE_COLS,   add_dtype=false),
    (name="core_dtype", xcols=CORE_COLS,   add_dtype=true),   # Price + Core + D-Type (no seg/"Chars")
    (name="full",       xcols=X_COLS,      add_dtype=false),
    (name="full_dtype", xcols=X_COLS,      add_dtype=true),
]

# ==========================================================================
# 0c. Standard-error method (shared convention with the RC engine + sleepiness)
# ==========================================================================
# BLP_SE_METHOD: "wcb" (wild cluster bootstrap, default — matches the sleepiness estimation's
# inference) or "sandwich" (analytical cluster-robust). BLP_WCB_REPS / BLP_WCB_SCHEME mirror the
# sleepiness SLEEP_BOOT_B / SLEEP_BOOT_SCHEME defaults (999 replications, Webb 6-point weights).
se_method()  = lowercase(get(ENV, "BLP_SE_METHOD", "wcb"))
wcb_reps()   = parse(Int, get(ENV, "BLP_WCB_REPS", get(ENV, "SLEEP_BOOT_B", "999")))
wcb_scheme() = lowercase(get(ENV, "BLP_WCB_SCHEME", get(ENV, "SLEEP_BOOT_SCHEME", "webb")))

"""Wild bootstrap weights, one per cluster: 6-point Webb (default; robust to few/imbalanced
clusters) or 2-point Rademacher. Matches utils/sleep_links.py `_wild_weights`."""
function wild_weights(n::Int, scheme::AbstractString, rng::AbstractRNG)
    if scheme == "webb"
        vals = (-sqrt(1.5), -1.0, -sqrt(0.5), sqrt(0.5), 1.0, sqrt(1.5))
        return Float64[vals[rand(rng, 1:6)] for _ in 1:n]
    end
    return Float64[rand(rng) < 0.5 ? -1.0 : 1.0 for _ in 1:n]   # Rademacher
end

"""Score/no-refit wild cluster bootstrap (Kline–Santos; same primitive as the sleepiness
`cluster_wild_bootstrap`). Given per-cluster influence functions `IF_cl` (G×K) for θ̂, draws
`B` wild-weighted perturbations θ_b = θ̂ + Σ_g w_g·IF_g and returns (se, pval): the bootstrap SD
and a symmetric Wald p, 2·Φ̄(|θ̂_k|/se_k). No refit, no small-sample factor (unit-variance
weights supply it)."""
function wcb_se(theta::Vector{Float64}, IF_cl::Matrix{Float64};
                B::Int=wcb_reps(), scheme::AbstractString=wcb_scheme(), seed::Int=0)
    G, K = size(IF_cl)
    (G < 2 || K == 0) && return (fill(NaN, K), fill(NaN, K))
    rng   = MersenneTwister(seed)
    draws = Matrix{Float64}(undef, B, K)
    for b in 1:B
        wv = wild_weights(G, scheme, rng)          # one weight per cluster
        @views draws[b, :] .= theta .+ IF_cl' * wv  # θ_b = θ̂ + Σ_g w_g IF_g
    end
    se   = Float64[std(@view(draws[:, k]); corrected=true) for k in 1:K]
    pval = Float64[se[k] > 0 ? 2 * ccdf(Normal(), abs(theta[k]) / se[k]) : 0.0 for k in 1:K]
    return se, pval
end

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

# Local logit outputs live in a dedicated `logit/` subfolder of BLP_RESULTS, kept separate
# from the cluster RC outputs (see process_blp_outputs.py). Created on demand.
function logit_dir()
    d = joinpath(get_paths()[2], "logit")
    isdir(d) || mkpath(d)
    return d
end

# Canonical combined-summary path.
combined_summary_path() = joinpath(logit_dir(),
                                   "logit_summary_spec_$(SPEC_ID).json")

"""
    discover_estim_strategies() -> Vector of (id, label, prefix)

Auto-discover the estimation routines for spec SPEC_ID by scanning the
demand-prep directory for `demand_<id>_*_spec_<SPEC_ID>.parquet`, excluding the legacy
`*_final_*` files. The prefix is the filename minus the `_spec_<SPEC_ID>.parquet` tail
(e.g. `demand_1`, `demand_3_logistic`). Sorted by id.

Newly-added/relabelled routines are picked up with NO code change. If MULTIPLE non-final
parquets exist for the same id (e.g. a leftover old-scheme file alongside a freshly
rebuilt one), the **most recently modified** wins — so a stale file can't shadow the new
one. (Still: deleting old `demand_*_spec_<SPEC_ID>.parquet` after a relabel is tidiest.)
"""
function discover_estim_strategies()
    input_dir, _ = get_paths()
    isdir(input_dir) || return NamedTuple[]
    best = Dict{Int, Tuple{Float64, String}}()   # id => (mtime, prefix); newest wins
    pat  = Regex("^demand_(\\d+)(?:_.*)?_spec_$(SPEC_ID)\\.parquet\$")
    for f in readdir(input_dir)
        occursin("_final_", f) && continue
        m = match(pat, f)
        m === nothing && continue
        id = parse(Int, m.captures[1])
        mt = mtime(joinpath(input_dir, f))
        if !haskey(best, id) || mt > best[id][1]
            best[id] = (mt, replace(f, "_spec_$(SPEC_ID).parquet" => ""))
        end
    end
    return [(id = id, label = "E$id", prefix = best[id][2]) for id in sort!(collect(keys(best)))]
end

const ESTIM_STRATEGIES = discover_estim_strategies()

# ==========================================================================
# 1. Data Loading
# ==========================================================================
"""Load the demand-prep parquet for one routine (drops the old `_final`)."""
function load_spec_data(estim)
    input_dir, _ = get_paths()
    fname = "$(estim.prefix)_spec_$(SPEC_ID).parquet"
    path  = joinpath(input_dir, fname)
    isfile(path) || error("Missing demand-prep parquet: $path")
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
    all_iv_names = vcat(IV_BLP_LOO, IV_ESTBAN, IV_COST, IV_CAPITAL)

    # A missing instrument COLUMN used to be dropped in silence, which is how the entire IV_BLP_LOO
    # block vanished when a market_panel rebuild skipped panel_7_instruments.py (it OVERWRITES
    # market_panel.csv with the LOO instruments + FGC dummy). The logit then ran on the cost shifters
    # alone. A missing column is a broken panel, not a modelling choice — say so.
    _absent = [c for c in all_iv_names if !(c in names(df))]
    if !isempty(_absent)
        error("Demand parquet is missing instrument column(s): " * join(_absent, ", ") *
              ".\nRebuild in order: panel_6 → panel_7 (LOO instruments + FGC; OVERWRITES " *
              "market_panel.csv) → panel_9 --patch-market, then re-run the demand prep.")
    end

    # Zero-variance IVs are still dropped (a constant column carries no identifying information and
    # would make Z'Z singular) — but say WHICH, so a silently-degenerate instrument is visible.
    _zerovar = [c for c in all_iv_names if
        std(replace(Float64.(coalesce.(df[!, c], 0.0)), Inf=>0.0, -Inf=>0.0)) <= 1e-10]
    if !isempty(_zerovar)
        @warn "Dropping zero-variance instrument(s): " * join(_zerovar, ", ")
    end
    iv_avail = [c for c in all_iv_names if !(c in _zerovar)]

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

    # --- Cluster-robust sandwich + per-cluster influence functions (for the WCB) ---
    # Bread: (X̃'X)⁻¹.  Meat: Σ_c (X̃_c' ξ_c)(X̃_c' ξ_c)'.  IF_g = (X̃'X)⁻¹ (X̃_c' ξ_c).
    unique_cl = unique(cl_v)
    G         = length(unique_cl)
    meat      = zeros(K, K)
    IF_cl     = zeros(G, K)
    for (gi, c) in enumerate(unique_cl)
        c_mask  = cl_v .== c
        score_c = X_proj[c_mask, :]' * xi_v[c_mask]   # (K,)
        meat   .+= score_c * score_c'
        IF_cl[gi, :] = XpX_inv * score_c
    end
    # Small-sample correction: G/(G-1) × N/(N-K)
    correction = (G / (G - 1)) * (N_v / (N_v - K))
    se_sand    = sqrt.(max.(diag(XpX_inv * meat * XpX_inv' .* correction), 0.0))

    # IK2016 effective clusters: G* = G / (1 + CV²)
    cl_sizes = [count(==(c), cl_v) for c in unique_cl]
    mu_G     = Statistics.mean(cl_sizes)
    cv_G     = mu_G > 0 ? Statistics.std(cl_sizes; corrected=false) / mu_G : 0.0
    G_star   = max(1.0, G / (1.0 + cv_G^2))

    # --- SE method: analytical sandwich or wild cluster bootstrap (default) ---
    if se_method() == "sandwich"
        se   = se_sand
        pval = Float64[2 * ccdf(TDist(G_star), abs(theta1[k]) / max(se[k], 1e-15)) for k in 1:K]
    else  # "wcb" — score wild cluster bootstrap, matching the sleepiness estimation
        se, pval = wcb_se(theta1, IF_cl)
    end

    return theta1, se, pval, xi, Q, G, G_star
end

# ==========================================================================
# 4. Per-routine driver
# ==========================================================================
"""
Estimate all four sub-models for a single routine.

Loads the routine's demand-prep parquet, builds δ from data shares, runs 2SLS for each
sub-model, writes per-(routine, sub-model) JLS and the δ warm-start checkpoints, and
returns a Dict keyed `"{label}_{submodel}"`.
"""
function run_strategy(estim)
    _, output_dir = get_paths()
    mkpath(output_dir)

    println("\n  ── Estimation $(estim.label) ──")

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

    # ── Save logit δ checkpoint for BLP σ-stage warm-start ──
    delta_chk_path = joinpath(logit_dir(),
        "logit_delta_E$(estim.id)_spec_$(SPEC_ID).jls")
    try
        serialize(delta_chk_path, Dict{String,Any}(
            "delta"    => delta,
            "estim_id" => estim.id,
            "spec_id"  => SPEC_ID,
            "N"        => nrow(df),
            "build"    => "logit",
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

        # 2SLS estimation. SEs by BLP_SE_METHOD (WCB by default); `pval` are the matching
        # p-values (WCB Wald or sandwich t(G*)) used for the table significance stars.
        theta1, se, pval, _xi, Q, n_cl, G_star = estimate_2sls(delta, X_full, X_hat, Z, clusters)

        # t-statistics
        tstat = theta1 ./ max.(se, 1e-15)

        # Display
        println("      SE method: $(se_method())  |  Q-value: $(round(Q, sigdigits=6))")
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
            "build"       => "logit",
            "sub_model"   => sm.name,
            "param_names" => pnames,
            "theta1"      => theta1,
            "se"          => se,
            "tstat"       => tstat,
            "pval"        => pval,
            "se_method"   => se_method(),
            "Q_value"     => Q,
            "n_obs"       => nrow(df),
            "n_iv"        => length(iv_names),
            "iv_names"    => iv_names,
            "n_clusters"  => n_cl,
            "G_star"      => G_star,
            "converged"   => true,
        )
        routine_results[key] = res

        # Save individual JLS
        jls_path = joinpath(logit_dir(),
            "logit_$(key)_spec_$(SPEC_ID).jls")
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

"""Write the combined summary JSON from a full results Dict."""
function write_combined_summary(all_results::Dict{String,Any})
    json_path = combined_summary_path()
    open(json_path, "w") do f
        JSON3.pretty(f, _to_json_data(all_results))
    end
    println("\n  Combined summary saved: $(basename(json_path))")
    return json_path
end

"""
Merge one routine's keys into the existing combined summary (creating it if
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
# 5b. LaTeX result tables (ported from make_blp_logit_table.py)
# ==========================================================================
# Writes est{id}_spec12_logit.tex (xltabular: 4 sub-model cols, parameter rows
# with SE underneath + IK2016 t(G*) significance stars, footer with obs/Q/dof/G*). Output
# matches the former Python generator so V_Main.tex \input{} stays unchanged.

const VAR_MAP = Dict(
    "alpha"                => raw"Price coefficient ($\alpha$)",
    "fgc_covered"          => "FGC Covered",
    "has_ip"               => "Has Payment Institution",
    "log_total_assets_lag" => raw"$\ln(\text{Total Assets}_{t-1})$",
    "seg_S2" => "Segment S2", "seg_S3" => "Segment S3",
    "seg_S4" => "Segment S4", "seg_S5" => "Segment S5",
    "dummy_D_type"         => "D Type",
)
const ROW_ORDER = ["alpha", "fgc_covered", "has_ip", "log_total_assets_lag",
                   "seg_S2", "seg_S3", "seg_S4", "seg_S5", "dummy_D_type"]
const TABLE_SUBMODELS = [("priceonly", "Price Only"), ("core", "Price + Core"),
                         ("full", "Price + Chars"), ("full_dtype", "+ D-Type"),
                         ("core_dtype", "Price + Core + D-Type")]
# Cross-estimator comparison table (est5-8_spec12_logit_comparison.tex): one column per
# demand routine, each showing its final `+ D-Type` sub-model. Column headers \ref{} the
# sleepiness-strategy enumerate items in V_Main §(sec:empirical:sleep) — same convention as
# est5-8_spec12_stage2_comparison.tex. (E3/E4's `estimation:logistic` item is commented out
# in V_Main, so they fall back to a plain E<id> header if ever included.)
const COMPARISON_IDS  = [5, 6, 7, 8]
const COMPARISON_ROWS = ["alpha", "fgc_covered", "has_ip", "log_total_assets_lag",
                         "dummy_D_type"]   # seg_S2-S5 included in the spec, not reported
const COMPARISON_ROWS_SEG = ["alpha", "fgc_covered", "has_ip", "log_total_assets_lag",
                             "seg_S2", "seg_S3", "seg_S4", "seg_S5", "dummy_D_type"]  # segments shown
const ESTIMATION_ENUM_REF = Dict(
    1 => raw"\ref{estimation:local}",
    2 => raw"\ref{estimation:pooled}",
    5 => raw"\ref{estimation:single_idx}",
    6 => raw"\ref{estimation:single_idx_time}",
    7 => raw"\ref{estimation:joint_sieve}",
    8 => raw"\ref{estimation:joint_sieve_time}",
)
const DRAFTS_DIR = raw"C:\Users\pedro\OneDrive\Documentos\Yale\Year 3 (2024 - 2025)\Open Finance\Open-Finance\Drafts\Deposit Competition"
const TROW = " \\\\"   # LaTeX row terminator ` \\` (a raw " \\" would collapse to one backslash)

map_var(name::AbstractString) = get(VAR_MAP, name, replace(name, "_" => raw"\_"))

_stars(p) = p < 0.01 ? raw"^{***}" : p < 0.05 ? raw"^{**}" : p < 0.10 ? raw"^{*}" : ""

"""Thousands-separated integer (mirrors Python's f'{n:,}')."""
function _commas(n::Integer)
    s = string(abs(n)); parts = String[]
    while length(s) > 3; pushfirst!(parts, s[end-2:end]); s = s[1:end-3]; end
    pushfirst!(parts, s)
    return (n < 0 ? "-" : "") * join(parts, ",")
end

"""(coef_cell, se_cell) with significance stars from the stored p-value `pval` (WCB Wald p)
when supplied, else the IK2016 t(G*) p; ('-','') if missing."""
function format_cell(coef, se, gstar, pval=nothing)
    (coef === nothing || se === nothing || isnan(coef) || isnan(se)) && return ("-", "")
    t = coef / max(se, 1e-15)
    p = (pval !== nothing && !isnan(pval)) ? pval :
        (gstar !== nothing && gstar > 1)   ? 2 * ccdf(TDist(gstar), abs(t)) :
                                             2 * ccdf(Normal(), abs(t))
    return (@sprintf("\$%.4f%s\$", coef, _stars(p)), @sprintf("\$(%.4f)\$", se))
end

"""Q-value cell with χ²(L) overidentification p-value stars; '---' if missing."""
function format_q_value(qv, L)
    (qv === nothing || isnan(qv) || L <= 0) && return "---"
    return @sprintf("\$%.4f%s\$", qv, _stars(ccdf(Chisq(L), qv)))
end

# ── plain-text formatting (comparison table matches est5-8_spec12_stage2_comparison.tex,
#    which prints `-0.1494***` / `(0.0677)` without math mode) ──────────────────────────
_stars_plain(p) = p < 0.01 ? "***" : p < 0.05 ? "**" : p < 0.10 ? "*" : ""

"""(coef_cell, se_cell) in the stage2-comparison plain style; stars from `pval` (WCB) when
supplied, else t(G*); ('-','-') if missing."""
function format_cell_plain(coef, se, gstar, pval=nothing)
    (coef === nothing || se === nothing || isnan(coef) || isnan(se)) && return ("-", "-")
    t = coef / max(se, 1e-15)
    p = (pval !== nothing && !isnan(pval)) ? pval :
        (gstar !== nothing && gstar > 1)   ? 2 * ccdf(TDist(gstar), abs(t)) :
                                             2 * ccdf(Normal(), abs(t))
    return (@sprintf("%.4f%s", coef, _stars_plain(p)), @sprintf("(%.4f)", se))
end

"""Sample mean of ρ(1−s) for one routine — multiplied by that routine's α̂ it gives the
mean own-price semi-elasticity row of the comparison table: α̂·ρ_jkmt·(1−s_jkmt) averaged
over the estimation sample. ρ = spread in pp (spread_ann/100); s = the share used to build
δ (share_D for D-type products, conditional share_B_cond for B-type). NaN if unavailable."""
function _mean_rho_one_minus_s(estim)
    df   = load_spec_data(estim)
    ρ    = Float64.(coalesce.(df.spread_ann, NaN)) ./ 100.0
    is_B = Bool.(coalesce.(df.is_B, false))
    sD   = Float64.(coalesce.(df.share_D, NaN))
    sB   = Float64.(coalesce.(df.share_B_cond, NaN))
    s    = ifelse.(is_B, sB, sD)
    m    = isfinite.(ρ) .& isfinite.(s) .& (s .>= 0.0) .& (s .< 1.0)
    return any(m) ? mean(ρ[m] .* (1.0 .- s[m])) : NaN
end

"""Build est5-8_spec12_logit_comparison.tex: columns = routines (each its `+ D-Type`
sub-model), rows = COMPARISON_ROWS, stats block = semi-elasticity / N / Q(dof) / G*.
Layout, notes and label conventions mirror est5-8_spec12_stage2_comparison.tex."""
function build_logit_comparison_tex(data::AbstractDict, ids::Vector{Int},
                                    elas::AbstractDict;
                                    rows::Vector{String}=COMPARISON_ROWS,
                                    with_seg::Bool=false)::String
    n    = length(ids)
    hdr  = "Variable & " * join([get(ESTIMATION_ENUM_REF, id, "E$id") for id in ids], " & ") * TROW
    seg_sentence = with_seg ? "" :
        raw"Segment dummies (S2--S5) are included in every strategy but not reported. "
    note = raw"\multicolumn{" * string(n + 1) *
        raw"}{p{\dimexpr\textwidth-2\tabcolsep\relax}}{\scriptsize\textit{Notes:} Each " *
        raw"column reports the demand logit's final specification, estimated on the demand " *
        raw"sample implied by the corresponding sleepiness strategy, on Specification~12; columns " *
        raw"index the estimation strategies enumerated in Section~\ref{sec:empirical:sleep}. " *
        seg_sentence *
        raw"Wild cluster bootstrap standard errors (conglomerate clusters) in parentheses. " *
        raw"Significance: *** $p<0.01$, ** $p<0.05$, * $p<0.1$. $Q$ is the GMM " *
        raw"overidentification statistic ($\chi^2_L$, $L$ = \# instruments); $G^*$ is " *
        raw"effective clusters. The mean own-price semi-elasticity is " *
        raw"$\hat{\alpha}\,\rho_{jkmt}(1-s_{jkmt})$ averaged over the estimation sample " *
        raw"(spread $\rho$ in percentage points).}"   # no trailing TROW (matches stage2 template)
    lines = String[
        raw"\setstretch{1.0}",
        raw"\begin{xltabular}{\textwidth}{>{\raggedright\arraybackslash}p{0.26\textwidth} *{" *
            string(n) * raw"}{>{\centering\arraybackslash}X}}",
        "\\caption{Demand Logit Estimation}\\label{tab:demand_logit_spec12_comparison" *
            (with_seg ? "_seg" : "") * "}" * TROW,
        raw"\toprule", hdr, raw"\midrule", raw"\endfirsthead",
        raw"\multicolumn{" * string(n + 1) *
            raw"}{c}{{\bfseries \tablename\ \thetable{} (continued from previous page)}}" * TROW,
        raw"\toprule", hdr, raw"\midrule", raw"\endhead",
        raw"\midrule",
        raw"\multicolumn{" * string(n + 1) * raw"}{r}{{Continued on next page}}" * TROW,
        raw"\endfoot",
        raw"\bottomrule", note, raw"\endlastfoot",
    ]
    for (i, p) in enumerate(rows)
        row_c = String[]; row_s = String[]
        for id in ids
            entry  = get(data, "E$(id)_full_dtype", Dict{String,Any}())
            pnames = String.(get(entry, "param_names", String[]))
            j = findfirst(==(p), pnames)
            if j !== nothing
                gs = get(entry, "G_star", nothing)
                pv = get(entry, "pval", nothing)
                c, s = format_cell_plain(Float64(entry["theta1"][j]), Float64(entry["se"][j]),
                                         gs === nothing ? nothing : Float64(gs),
                                         (pv !== nothing && j <= length(pv)) ? Float64(pv[j]) : nothing)
                push!(row_c, c); push!(row_s, s)
            else
                push!(row_c, "-"); push!(row_s, "-")
            end
        end
        push!(lines, raw"\multirow[t]{2}{0.26\textwidth}{\raggedright " * map_var(p) *
                     "} & " * join(row_c, " & ") * " \\\\*")
        push!(lines, " & " * join(row_s, " & ") * TROW)
        i < length(rows) && push!(lines, raw"\addlinespace")
    end
    push!(lines, raw"\midrule")
    elas_l = String[]; obs_l = String[]; q_l = String[]; niv_l = String[]; gstar_l = String[]
    for id in ids
        entry = get(data, "E$(id)_full_dtype", Dict{String,Any}())
        ev = get(elas, id, NaN)
        push!(elas_l, isfinite(ev) ? @sprintf("%.3f", ev) : "---")
        obs = get(entry, "n_obs", nothing)
        push!(obs_l, (obs === nothing || obs == 0) ? "---" : _commas(Int(obs)))
        qv = get(entry, "Q_value", nothing); niv = Int(get(entry, "n_iv", 0))
        push!(niv_l, string(niv))
        push!(q_l, qv === nothing ? "---" :
                   @sprintf("%.4f%s", Float64(qv), _stars_plain(ccdf(Chisq(niv), Float64(qv)))))
        gs = get(entry, "G_star", nothing)
        push!(gstar_l, gs === nothing ? "---" : @sprintf("%.2f", Float64(gs)))
    end
    append!(lines, [
        "Mean own-price semi-elasticity & " * join(elas_l, " & ") * TROW,
        "Observations & " * join(obs_l, " & ") * TROW,
        "\$Q\$ & " * join(q_l, " & ") * TROW,
        "Degrees of Freedom & " * join(niv_l, " & ") * TROW,
        "Effective Clusters (\$G^*\$) & " * join(gstar_l, " & ") * TROW,
        raw"\end{xltabular}",
        raw"\doublespacing",
    ])
    return join(lines, "\n")
end

"""Write est5-8_spec12_logit_comparison.tex (COMPARISON_IDS × `+ D-Type`) to Rout + Drafts.
Computes the mean own-price semi-elasticity per routine from its demand parquet (skipped
with a '---' cell if the parquet is unavailable)."""
function write_logit_comparison_table(data::AbstractDict)
    ids = [id for id in COMPARISON_IDS if haskey(data, "E$(id)_full_dtype")]
    if length(ids) < 2
        println("    [table] comparison skipped — need ≥2 of E$(COMPARISON_IDS) full_dtype entries")
        return
    end
    elas = Dict{Int,Float64}()
    for id in ids
        entry  = data["E$(id)_full_dtype"]
        pnames = String.(get(entry, "param_names", String[]))
        j = findfirst(==("alpha"), pnames)
        estim = findfirst(s -> s.id == id, ESTIM_STRATEGIES)
        if j === nothing || estim === nothing
            elas[id] = NaN; continue
        end
        elas[id] = try
            Float64(entry["theta1"][j]) * _mean_rho_one_minus_s(ESTIM_STRATEGIES[estim])
        catch _e
            println("    [table] WARN: E$id semi-elasticity failed — $_e"); NaN
        end
    end
    _, output_dir = get_paths()
    rout_dir = joinpath(dirname(output_dir), "Rout")
    mkpath(rout_dir)
    dests = isdir(DRAFTS_DIR) ? [rout_dir, DRAFTS_DIR] : [rout_dir]
    # Two versions: segment dummies suppressed (default) and segment dummies shown (`_seg`).
    for (rws, wseg, fn) in ((COMPARISON_ROWS,     false, "est5-8_spec12_logit_comparison.tex"),
                            (COMPARISON_ROWS_SEG, true,  "est5-8_spec12_logit_comparison_seg.tex"))
        tex = build_logit_comparison_tex(data, ids, elas; rows=rws, with_seg=wseg)
        for d in dests
            path = joinpath(d, fn)
            try
                open(path, "w") do f; write(f, tex); end
                println("    [table] $fn → $(basename(d))/")
            catch _e
                println("    [table] WARN: could not write $path — $_e")
            end
        end
    end
end

"""Build the xltabular LaTeX for one routine from the combined-summary `data`."""
function build_logit_table_tex(est_id::Int, data::AbstractDict)::String
    ncols   = length(TABLE_SUBMODELS)
    col_fmt = raw">{\raggedright\arraybackslash}p{0.24\textwidth} *{" * string(ncols) *
              raw"}{>{\centering\arraybackslash}X}"
    hdr = "    Parameter & " * join([sm[2] for sm in TABLE_SUBMODELS], " & ") * TROW
    lines = String[
        raw"\begin{spacing}{1.0}",
        "\\begin{xltabular}{\\textwidth}{$col_fmt}",
        "    \\caption{Demand Logit Estimation --- Estimation " *
            "$(get(ESTIMATION_ENUM_REF, est_id, "E$est_id"))}",
        "    \\label{tab:demand_logit_est$(est_id)_spec12} \\\\",
        raw"    \toprule", hdr, raw"    \midrule", raw"    \endfirsthead", "",
        raw"    \multicolumn{" * string(ncols+1) *
            raw"}{c}{\bfseries Table \thetable\ continued from previous page}" * TROW,
        raw"    \toprule", hdr, raw"    \midrule", raw"    \endhead", "",
        raw"    \midrule",
        raw"    \multicolumn{" * string(ncols+1) * raw"}{r}{\textit{Continued on next page}}" * TROW,
        raw"    \endfoot", "",
        raw"    \bottomrule",
        raw"    \multicolumn{" * string(ncols+1) *
            raw"}{p{\dimexpr\textwidth-2\tabcolsep\relax}}{\scriptsize \textit{Notes:} " *
            raw"Wild cluster bootstrap standard errors (conglomerate clusters) in parentheses. " *
            raw"Significance: *** $p<0.01$, ** $p<0.05$, * $p<0.1$. $Q$ denotes the GMM " *
            raw"overidentification test statistic ($\chi^2_L$, $L$ = \# instruments); $G^*$ is " *
            raw"effective clusters.}" * TROW,
        raw"    \endlastfoot", "",
    ]
    for (i, p) in enumerate(ROW_ORDER)
        row_c = String[map_var(p)]; row_s = String[""]
        for (sm_key, _) in TABLE_SUBMODELS
            entry  = get(data, "E$(est_id)_$(sm_key)", Dict{String,Any}())
            pnames = String.(get(entry, "param_names", String[]))
            j = findfirst(==(p), pnames)
            if j !== nothing
                gs = get(entry, "G_star", nothing)
                pv = get(entry, "pval", nothing)
                c, s = format_cell(Float64(entry["theta1"][j]), Float64(entry["se"][j]),
                                   gs === nothing ? nothing : Float64(gs),
                                   (pv !== nothing && j <= length(pv)) ? Float64(pv[j]) : nothing)
                push!(row_c, c); push!(row_s, s)
            else
                push!(row_c, "-"); push!(row_s, "")
            end
        end
        push!(lines, "    " * join(row_c, " & ") * TROW)
        push!(lines, "    " * join(row_s, " & ") * TROW)
        i < length(ROW_ORDER) && push!(lines, raw"    \addlinespace")
    end
    push!(lines, raw"    \midrule")
    obs_l = String[]; q_l = String[]; gstar_l = String[]; niv_l = String[]
    for (sm_key, _) in TABLE_SUBMODELS
        entry = get(data, "E$(est_id)_$(sm_key)", Dict{String,Any}())
        obs   = get(entry, "n_obs", nothing)
        push!(obs_l, (obs === nothing || obs == 0) ? "---" : _commas(Int(obs)))
        qv  = get(entry, "Q_value", nothing); niv = Int(get(entry, "n_iv", 0))
        push!(niv_l, string(niv))
        push!(q_l, qv === nothing ? "---" : format_q_value(Float64(qv), niv))
        gs = get(entry, "G_star", nothing)
        push!(gstar_l, gs === nothing ? "---" : @sprintf("%.2f", Float64(gs)))
    end
    append!(lines, [
        "    Observations & " * join(obs_l, " & ") * TROW,
        "    \$Q\$ & " * join(q_l, " & ") * TROW,
        "    Degrees of Freedom & " * join(niv_l, " & ") * TROW,
        "    Effective Clusters (\$G^*\$) & " * join(gstar_l, " & ") * TROW,
        raw"\end{xltabular}", raw"\end{spacing}",
    ])
    return join(lines, "\n")
end

"""Read the combined summary JSON back into a Dict (for table generation)."""
function read_combined_summary()
    p = combined_summary_path()
    isfile(p) || return Dict{String,Any}()
    try
        return copy(JSON3.read(read(p, String), Dict{String,Any}))
    catch _e
        println("  [table] WARN: could not read summary — $_e"); return Dict{String,Any}()
    end
end

"""Write est{id}_spec12_logit.tex for each id to BLP_RESULTS/../Rout and Drafts."""
function write_logit_tables(ids::Vector{Int}, data::AbstractDict)
    _, output_dir = get_paths()
    rout_dir = joinpath(dirname(output_dir), "Rout")     # ESTIMATION_OUTPUT/Rout
    mkpath(rout_dir)
    dests = isdir(DRAFTS_DIR) ? [rout_dir, DRAFTS_DIR] : [rout_dir]
    for id in ids
        tex = build_logit_table_tex(id, data)
        for d in dests
            path = joinpath(d, "est$(id)_spec12_logit.tex")
            try
                open(path, "w") do f; write(f, tex); end
                println("    [table] est$(id)_spec12_logit.tex → $(basename(d))/")
            catch _e
                println("    [table] WARN: could not write $path — $_e")
            end
        end
    end
end

"""Discover routine ids present in the combined summary (keys `E<id>_<submodel>`)."""
function _summary_ids(data::AbstractDict)
    ids = Set{Int}()
    for k in keys(data)
        m = match(r"^E(\d+)_", String(k))
        m === nothing || push!(ids, parse(Int, m.captures[1]))
    end
    return sort!(collect(ids))
end

# ==========================================================================
# 6. Orchestrator entrypoint
# ==========================================================================
"""Parse an optional single-routine selector from ARGS: `--est N` or a bare integer.
Returns the id (Int) or nothing (run all discovered routines)."""
function _parse_est_arg()
    for (i, a) in enumerate(ARGS)
        if a == "--est" && i < length(ARGS)
            return tryparse(Int, ARGS[i + 1])
        elseif (v = tryparse(Int, a)) !== nothing
            return v
        end
    end
    return nothing
end

function main()
    _, output_dir = get_paths()
    mkpath(output_dir)

    sel = _parse_est_arg()

    # --tables-only: rebuild every LaTeX table from the existing combined summary,
    # skipping estimation entirely (the semi-elasticity row still reads the parquets).
    if "--tables-only" in ARGS
        summary = read_combined_summary()
        isempty(summary) && error("--tables-only: no combined summary at $(combined_summary_path()).")
        table_ids = sel === nothing ? _summary_ids(summary) : [sel]
        println("  [tables-only] Regenerating LaTeX tables for $(join("E" .* string.(table_ids), ", "))…")
        write_logit_tables(table_ids, summary)
        write_logit_comparison_table(summary)
        println("  [DONE] Tables regenerated (no estimation run).")
        return
    end

    routines = sel === nothing ? ESTIM_STRATEGIES : filter(s -> s.id == sel, ESTIM_STRATEGIES)
    if isempty(routines)
        avail = isempty(ESTIM_STRATEGIES) ? "(none)" :
                join([s.label for s in ESTIM_STRATEGIES], ", ")
        error(sel === nothing ?
            "No demand-prep parquets found for spec $SPEC_ID in $(get_paths()[1])." :
            "No demand-prep parquet for E$sel (spec $SPEC_ID). Available: $avail.")
    end

    println("=" ^ 70)
    println("  BLP Logit (Non-RC) — Spec $SPEC_ID")
    println("  $(length(routines)) routine(s): $(join([s.label for s in routines], ", ")) " *
            "× $(length(SUB_MODELS)) sub-models")
    println("=" ^ 70)

    if sel === nothing
        # Full build: run every discovered routine, write the combined summary.
        all_results = Dict{String, Any}()
        for estim in ESTIM_STRATEGIES
            try
                merge!(all_results, run_strategy(estim))
            catch e
                println("  [!] Routine $(estim.label) failed: $e. Skipping.")
            end
        end
        write_combined_summary(all_results)
    else
        # Single routine: run it and merge into the existing combined summary.
        merge_into_combined_summary(run_strategy(routines[1]))
    end

    # LaTeX result tables (replaces the former make_blp_logit_table.py step).
    summary   = read_combined_summary()
    table_ids = sel === nothing ? _summary_ids(summary) : [sel]
    println("\n  Generating LaTeX tables for $(join("E" .* string.(table_ids), ", "))…")
    write_logit_tables(table_ids, summary)
    write_logit_comparison_table(summary)

    println("  [DONE] Logit estimation + tables complete.")
end

# Only auto-run when executed directly.
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
