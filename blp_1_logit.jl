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

# ==========================================================================
# 0. Constants
# ==========================================================================
const SPEC_ID = 12

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

# Estimation routines are AUTO-DISCOVERED from the demand-prep parquets — see
# discover_estim_strategies() / const ESTIM_STRATEGIES below (defined after get_paths()).

# Sub-model definitions (keys consumed by the in-file LaTeX table generator below)
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

# Canonical combined-summary path.
combined_summary_path() = joinpath(get_paths()[2],
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
    delta_chk_path = joinpath(output_dir,
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
            "build"       => "logit",
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

        # Save individual JLS
        jls_path = joinpath(output_dir,
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
                         ("full", "Price + Chars"), ("full_dtype", "+ D-Type")]
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

"""(coef_cell, se_cell) with IK2016 t(G*) significance stars; ('-','') if missing."""
function format_cell(coef, se, gstar)
    (coef === nothing || se === nothing || isnan(coef) || isnan(se)) && return ("-", "")
    t = coef / max(se, 1e-15)
    p = (gstar !== nothing && gstar > 1) ? 2 * ccdf(TDist(gstar), abs(t)) :
                                           2 * ccdf(Normal(), abs(t))
    return (@sprintf("\$%.4f%s\$", coef, _stars(p)), @sprintf("\$(%.4f)\$", se))
end

"""Q-value cell with χ²(L) overidentification p-value stars; '---' if missing."""
function format_q_value(qv, L)
    (qv === nothing || isnan(qv) || L <= 0) && return "---"
    return @sprintf("\$%.4f%s\$", qv, _stars(ccdf(Chisq(L), qv)))
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
        "    \\caption{Demand Logit Estimation -- Estimation $est_id, Specification 12}",
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
            raw"Cluster-robust standard errors in parentheses, clustered at conglomerate level " *
            raw"following \textcite{imbens2016robust} and \textcite{carter2017asymptotic}. " *
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
                c, s = format_cell(Float64(entry["theta1"][j]), Float64(entry["se"][j]),
                                   gs === nothing ? nothing : Float64(gs))
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
        "    \$Q\$ (GMM overidentification) & " * join(q_l, " & ") * TROW,
        "    Degrees of freedom (overidentification) & " * join(niv_l, " & ") * TROW,
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

    sel      = _parse_est_arg()
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

    println("  [DONE] Logit estimation + tables complete.")
end

# Only auto-run when executed directly.
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
