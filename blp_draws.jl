"""
blp_draws.jl
============
Pre-compute simulation draws for BLP demand estimation.

Generates:
  1. Scrambled Halton ν-draws (CG2020-compliant) for random coefficients
  2. Demographic d-draws using market-specific σ from demographics_sigma.parquet

Outputs are serialised to BLP_DRAWS/ for consumption by blp_loop.jl or
blp_estimation.jl.  Separating draw generation from estimation ensures:
  - Cross-strategy comparability (same draws for E1–E5)
  - Transparent diagnostics (fallback rate, σ statistics)
  - No re-generation on cluster job restarts

Usage
-----
  # Local validation (small R):
  julia --project=. blp_draws.jl --R 50 --seed 42

  # Cluster production:
  julia --project=. blp_draws.jl --R 2000 --seed 42 --hpc

  # Large R (up to 5000):
  julia --project=. blp_draws.jl --R 5000 --seed 42 --hpc

References
----------
  Conlon & Gortmaker (2020, RAND J. Econ.) — §3.1: scrambled Halton
    preferred over standard Halton for d > 8.
  Owen (2003) — nested uniform scrambling for QMC integration.
"""

using Parquet2, DataFrames, LinearAlgebra, Statistics
using Random, QuasiMonteCarlo, Distributions
using JSON3, Serialization, ArgParse, Printf, Dates

# ==========================================================================
# 0. Constants (must match blp_loop.jl)
# ==========================================================================
const X_COLS = ["fgc_covered", "has_ip", "seg_S2", "seg_S3", "seg_S4", "seg_S5",
                "log_total_assets_lag", "equity_ratio_lag"]
const L_PROD  = length(X_COLS)
const COEF_DIM = 1 + L_PROD   # spread + 8 product chars = 9

const D_COLS  = ["gdp_per_capita", "fraction_65plus", "fraction_young",
                 "pix_users_pf_per1000", "connections_per100", "frac_4g5g",
                 "branches_per1000", "cadunico_families_per1000"]
const D_DIM   = length(D_COLS)

# ==========================================================================
# 0b. Paths
# ==========================================================================
function get_paths(is_hpc::Bool)
    if is_hpc
        input_dir  = "/home/pf382/dep_comp/data/input"
        output_dir = "/home/pf382/dep_comp/data/output/BLP_DRAWS"
    else
        _root      = dirname(dirname(dirname(abspath(@__FILE__))))
        data_dir   = joinpath(_root, "BCB", "Egan_et_al_2025_Rep", "processed")
        input_dir  = joinpath(data_dir, "ESTIMATION_OUTPUT", "DEMAND_PREP")
        output_dir = joinpath(data_dir, "ESTIMATION_OUTPUT", "BLP_DRAWS")
    end
    return input_dir, output_dir
end

# ==========================================================================
# 1. Scrambled Halton Draws (CG2020)
# ==========================================================================
"""
Generate R draws in COEF_DIM dimensions using scrambled Halton sequences.

Per CG2020: use scrambled (Owen) Halton with fixed burn-in of 100 points.
Inverse-CDF mapped to N(0,1).
"""
function generate_halton_draws(R::Int, dim::Int, seed::Int)::Matrix{Float64}
    println("  Generating scrambled Halton draws: R=$R, dim=$dim, seed=$seed")

    # CG2020: fixed burn-in, not proportional to seed
    skip    = 100
    n_needed = R + skip

    # OwenScramble requires n_total to be a power of 2; round up
    n_total = nextpow(2, n_needed)

    # Generate raw Halton points in [0,1]^dim — shape (dim, n_total)
    pts_raw = QuasiMonteCarlo.sample(n_total, zeros(dim), ones(dim), HaltonSample())

    # Apply Owen scrambling for inter-dimensional decorrelation
    # pad >= log2(n_total) for proper digit-level scrambling
    Random.seed!(seed)
    pad = max(ceil(Int, log2(n_total)) + 2, 20)
    try
        pts_scrambled = QuasiMonteCarlo.randomize(pts_raw, OwenScramble(base=2, pad=pad))
        pts = pts_scrambled[:, (skip+1):(skip+R)]
        println("    Using Owen-scrambled Halton (n_total=$n_total, padded to power-of-2)")
    catch e
        # Fallback: Shift randomization (Cranley-Patterson rotation)
        println("  [WARN] OwenScramble failed ($e), using Shift randomization")
        pts_shifted = QuasiMonteCarlo.randomize(pts_raw, Shift())
        pts = pts_shifted[:, (skip+1):(skip+R)]
    end

    # Clamp to avoid Inf from quantile at 0 or 1
    pts = clamp.(pts, 1e-6, 1.0 - 1e-6)

    # Inverse-CDF: uniform → N(0,1)
    nu = quantile.(Normal(), pts)'  # shape (R, dim)

    println("    ν-draws: $(size(nu)) | mean=$(round(mean(nu), sigdigits=4)) " *
            "| std=$(round(std(nu), sigdigits=4))")
    return Matrix{Float64}(nu)
end

# ==========================================================================
# 2. Demographics Sigma Loading
# ==========================================================================
"""Load demographics_sigma.parquet → Dict{(mca_code, time_id), Vector{Float64}}."""
function load_sigma_table(input_dir::String)
    path = joinpath(input_dir, "demographics_sigma.parquet")
    if !isfile(path)
        println("  [WARN] demographics_sigma.parquet not found at $path")
        return nothing
    end
    df = DataFrame(Parquet2.Dataset(path); copycols=true)
    sigma_cols = [c * "_sigma" for c in D_COLS]
    avail = [c for c in sigma_cols if c in names(df)]
    println("  Loaded sigma table: $(nrow(df)) rows, $(length(avail)) σ columns")

    tbl = Dict{Tuple{String,String}, Vector{Float64}}()
    for row in eachrow(df)
        key = (string(row.mca_code), string(row.time_id))
        tbl[key] = Float64[coalesce(row[c], 0.0) for c in avail]
    end
    return tbl
end

# ==========================================================================
# 3. Demographic Draws
# ==========================================================================
"""
Generate demographic draws for all (mca_code, time_id) market-time pairs.

Returns:
  - draws_3d:   Array{Float64,3} of shape (n_keys+1, R, D_dim)
                Row 1 is a zero-padding row for unmapped observations.
  - key_index:  Dict{(mca, time), Int} mapping to row index in draws_3d
  - diagnostics: Dict with fallback statistics
"""
function generate_demographic_draws(df::DataFrame, R::Int, seed::Int;
                                     sigma_table=nothing)
    rng    = MersenneTwister(seed + 1)  # offset seed to decorrelate from ν
    d_cols = [c for c in D_COLS if c in names(df)]
    D      = length(d_cols)

    # Extract unique (mca_code, time_id) pairs with demographic means
    mca_time = unique(df[:, vcat(["mca_code", "time_id"], d_cols)])

    # National σ fallback
    nat_std = [std(skipmissing(mca_time[!, c])) for c in d_cols]
    nat_std = [s <= 0.0 ? 1.0 : s for s in nat_std]

    # Build key list and draws
    keys_list = Tuple{String,String}[]
    draws_list = Matrix{Float64}[]

    n_market = 0; n_fallback = 0
    per_dim_sigma_used = [Float64[] for _ in 1:D]

    for row in eachrow(mca_time)
        key = (string(row.mca_code), string(row.time_id))
        push!(keys_list, key)

        mu = [coalesce(row[c], 0.0) for c in d_cols]

        # Market-specific σ from sigma table
        if sigma_table !== nothing && haskey(sigma_table, key)
            mkt_std = sigma_table[key]
            # Guard: zero σ → national fallback for that dimension
            mkt_std = [s <= 0.0 ? nat_std[i] : s for (i, s) in enumerate(mkt_std)]
            n_market += 1
        else
            mkt_std = nat_std .* 0.1  # legacy fallback
            n_fallback += 1
        end

        # Track σ used per dimension
        for i in 1:D
            push!(per_dim_sigma_used[i], mkt_std[min(i, length(mkt_std))])
        end

        # Draw: d_r ~ N(μ_m, σ_m), shape (R, D)
        raw = mu' .+ mkt_std' .* randn(rng, R, D)
        push!(draws_list, raw)
    end

    n_keys = length(keys_list)

    # Stack into 3D array: (n_keys+1, R, D) with zero-padding row at index 1
    draws_3d = zeros(n_keys + 1, R, D)
    for (i, mat) in enumerate(draws_list)
        draws_3d[i + 1, :, :] .= mat  # index 1 is padding, data starts at 2
    end

    # Build key → row index mapping (offset by 1 for padding)
    key_index = Dict{Tuple{String,String}, Int}()
    for (i, k) in enumerate(keys_list)
        key_index[k] = i + 1
    end

    # Diagnostics
    fallback_rate = n_fallback / max(n_keys, 1)
    diagnostics = Dict{String,Any}(
        "n_keys"        => n_keys,
        "n_market"      => n_market,
        "n_fallback"    => n_fallback,
        "fallback_rate" => round(fallback_rate, sigdigits=4),
        "R"             => R,
        "D_dim"         => D,
        "d_cols"        => d_cols,
        "per_dim_stats" => Dict{String,Any}(),
    )

    println("\n  Demographic draws summary:")
    println("    Keys: $n_keys | Market-specific σ: $n_market | Fallback: $n_fallback " *
            "($(round(100*fallback_rate, digits=1))%)")

    for (i, col) in enumerate(d_cols)
        vals = per_dim_sigma_used[i]
        stats = Dict(
            "mean"   => round(mean(vals), sigdigits=4),
            "std"    => round(std(vals), sigdigits=4),
            "min"    => round(minimum(vals), sigdigits=4),
            "max"    => round(maximum(vals), sigdigits=4),
            "median" => round(median(vals), sigdigits=4),
        )
        diagnostics["per_dim_stats"][col] = stats
        @printf("    %-30s  σ: mean=%.4f  med=%.4f  [%.4f, %.4f]\n",
                col, stats["mean"], stats["median"], stats["min"], stats["max"])
    end

    if fallback_rate > 0.5
        println("  [WARNING] >50% fallback rate — check demographics_sigma.parquet coverage")
    end

    mem_gb = sizeof(draws_3d) / 1e9
    println("    draws_3d size: $(size(draws_3d)) = $(round(mem_gb, digits=2)) GB")

    return draws_3d, key_index, diagnostics
end

# ==========================================================================
# 4. CLI + Main
# ==========================================================================
function parse_args_draws()
    s = ArgParseSettings(description="BLP Draw Generator (CG2020-compliant)")
    @add_arg_table! s begin
        "--R";       arg_type=Int;    default=2000; help="Number of simulation draws"
        "--seed";    arg_type=Int;    default=42
        "--spec";    arg_type=Int;    default=12;   help="Spec ID for key extraction"
        "--estim";   arg_type=Int;    default=1;    help="Estimation strategy for key panel"
        "--hpc";     action=:store_true
    end
    return parse_args(s)
end

function main()
    args = parse_args_draws()
    R    = args["R"]
    seed = args["seed"]
    spec = args["spec"]
    estim= args["estim"]
    is_hpc = args["hpc"]

    input_dir, output_dir = get_paths(is_hpc)
    mkpath(output_dir)

    println("=" ^ 60)
    println("  BLP Draw Generator — CG2020 Compliant")
    println("  R=$R | seed=$seed | spec=$spec | HPC=$is_hpc")
    println("=" ^ 60)

    # ── 1. Halton ν-draws ────────────────────────────────────────────────
    nu_draws = generate_halton_draws(R, COEF_DIM, seed)

    nu_path = joinpath(output_dir, "halton_nu_R$(R)_seed$(seed).jls")
    serialize(nu_path, nu_draws)
    println("  Saved: $(basename(nu_path))")

    # ── 2. Load reference spec data for market keys ──────────────────────
    prefix = estim == 5 ? "demand_5_alt2logistic" : "demand_$(estim)"
    fname  = "$(prefix)_final_spec_$(spec).parquet"
    path   = joinpath(input_dir, fname)
    println("\n  Loading reference panel: $fname")
    isfile(path) || error("Reference parquet not found: $path")
    df = DataFrame(Parquet2.Dataset(path); copycols=true)
    println("    $(nrow(df)) observations")

    # ── 3. Load sigma table ──────────────────────────────────────────────
    sigma_table = load_sigma_table(input_dir)

    # ── 4. Demographic draws ─────────────────────────────────────────────
    draws_3d, key_index, diagnostics = generate_demographic_draws(
        df, R, seed; sigma_table=sigma_table)

    demo_path = joinpath(output_dir, "demo_draws_R$(R)_seed$(seed).jls")
    serialize(demo_path, draws_3d)
    println("  Saved: $(basename(demo_path))")

    key_path = joinpath(output_dir, "demo_key_index_R$(R)_seed$(seed).jls")
    serialize(key_path, key_index)
    println("  Saved: $(basename(key_path))")

    diag_path = joinpath(output_dir, "draws_diagnostics_R$(R)_seed$(seed).json")
    open(diag_path, "w") do f
        JSON3.pretty(f, diagnostics)
    end
    println("  Saved: $(basename(diag_path))")

    println("\n  [DONE] All draws serialised to $output_dir")
end

main()
