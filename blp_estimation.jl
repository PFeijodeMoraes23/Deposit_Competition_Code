"""
blp_estimation.jl
=================
BLP demand estimation consuming pre-computed draws from blp_draws.jl.

Identical to blp_loop.jl except:
  1. ν-draws and demographic draws are LOADED from BLP_DRAWS/ (not generated)
  2. Tighter outer bounds for sigma/pi parameters
  3. Clustering matches sleep estimation (CodConglomeradoPrudencial only)

Usage (cluster):
  julia --project=\${PROJECT_DIR} --threads=32 \\
      blp_estimation.jl --estim 1 --spec 12 --stage sequence \\
      --R 2000 --seed 42 --hpc

Local test:
  julia --project=. --threads=4 blp_estimation.jl \\
      --estim 1 --spec 12 --stage sigma --R 50 --seed 42

References
----------
  Berry, Levinsohn & Pakes (1995, Econometrica)
  Conlon & Gortmaker (2020, RAND J. Econ.)
"""

using Parquet2, DataFrames, SparseArrays, LinearAlgebra, Statistics
using Random, Optim, QuasiMonteCarlo, Distributions
using JSON3, Serialization, ArgParse, Printf, Dates

# ── Set BLAS threads ─────────────────────────────────────────────────────
BLAS.set_num_threads(Threads.nthreads())

# ==========================================================================
# 0. Constants (must match blp_loop.jl and blp_draws.jl)
# ==========================================================================
const X_COLS = ["fgc_covered", "has_ip", "seg_S2", "seg_S3", "seg_S4", "seg_S5",
                "log_total_assets_lag", "equity_ratio_lag"]
const L_PROD = length(X_COLS)
const K_TYPES = 4
const K_LIST  = [1, 2, 4, 5]
const D_COLS  = ["gdp_per_capita", "fraction_65plus", "fraction_young",
                 "pix_users_pf_per1000", "connections_per100", "frac_4g5g",
                 "branches_per1000", "cadunico_families_per1000"]
const D_DIM   = length(D_COLS)

const IV_BLP_LOO = ["loo_log_assets", "mean_loo_log_assets",
                    "loo_equity_ratio", "mean_loo_equity_ratio",
                    "loo_basileia", "mean_loo_basileia",
                    "loo_credit_assets", "mean_loo_credit_assets",
                    "loo_npl_provision", "mean_loo_npl_provision",
                    "n_rivals"]
const IV_COST    = ["personnel_cost_ratio_lag", "admin_cost_ratio_lag",
                    "tax_cost_ratio_lag"]
const IV_CAPITAL = ["indice_basileia_lag"]

const _log_buf  = String[]
const _log_lock = ReentrantLock()

# Note: get_paths is defined AFTER the blp_loop.jl include (see below)
# to override its 2-tuple version with our 3-tuple version.

function input_filename(estim::Int, spec_id::Int)::String
    prefix = estim == 5 ? "demand_5_alt2logistic" : "demand_$(estim)"
    return "$(prefix)_final_spec_$(spec_id).parquet"
end

function log_status(msg::String)
    stamped = "[$(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))] $msg"
    lock(_log_lock) do; push!(_log_buf, stamped); end
    println(stamped); flush(stdout)
end

const _blp_loop_path = joinpath(dirname(abspath(@__FILE__)), "blp_loop.jl")

"""Strip CLI + Main section from blp_loop.jl so we can include only the functions."""
function _strip_main(src::String)::String
    lines = split(src, '\n')
    out   = String[]
    for l in lines
        s = strip(l)
        # Stop at the CLI/Main section — everything below is blp_loop.jl's own driver
        if contains(s, "CLI") && contains(s, "Main")
            break
        end
        push!(out, l)
    end
    return join(out, '\n')
end

if isfile(_blp_loop_path)
    _clean_src = _strip_main(read(_blp_loop_path, String))
    try
        include_string(Main, _clean_src, _blp_loop_path)
        log_status("Loaded BLP core functions from blp_loop.jl")
    catch e
        error("Failed to load blp_loop.jl functions: $e")
    end
else
    error("blp_loop.jl not found at $_blp_loop_path — required for core BLP functions")
end

# Override blp_loop.jl's get_paths (2-tuple) with 3-tuple adding draws_dir
function get_paths(is_hpc::Bool; local_dir::Union{String,Nothing}=nothing)
    if is_hpc
        input_dir  = "/home/pf382/dep_comp/data/input"
        draws_dir  = "/home/pf382/dep_comp/data/output/BLP_DRAWS"
        output_dir = "/home/pf382/dep_comp/data/output"
    else
        if local_dir !== nothing
            data_dir = local_dir
        else
            _root    = dirname(dirname(dirname(abspath(@__FILE__))))
            data_dir = joinpath(_root, "BCB", "Egan_et_al_2025_Rep", "processed")
        end
        input_dir  = joinpath(data_dir, "ESTIMATION_OUTPUT", "DEMAND_PREP")
        draws_dir  = joinpath(data_dir, "ESTIMATION_OUTPUT", "BLP_DRAWS")
        output_dir = joinpath(data_dir, "ESTIMATION_OUTPUT", "BLP_RESULTS")
    end
    return input_dir, draws_dir, output_dir
end

# ==========================================================================
# 1. Load Pre-Computed Draws
# ==========================================================================
function load_precomputed_draws(draws_dir::String, R::Int, seed::Int)
    nu_path  = joinpath(draws_dir, "halton_nu_R$(R)_seed$(seed).jls")
    demo_path = joinpath(draws_dir, "demo_draws_R$(R)_seed$(seed).jls")
    key_path  = joinpath(draws_dir, "demo_key_index_R$(R)_seed$(seed).jls")

    isfile(nu_path) || error("Missing ν-draws: $nu_path")
    isfile(demo_path) || error("Missing demo draws: $demo_path")
    isfile(key_path) || error("Missing key index: $key_path")

    log_status("Loading pre-computed draws from $draws_dir")

    nu_draws   = deserialize(nu_path)::Matrix{Float64}
    draws_3d   = deserialize(demo_path)::Array{Float64,3}
    key_index  = deserialize(key_path)::Dict{Tuple{String,String},Int}

    log_status("  ν-draws: $(size(nu_draws)) | demo_draws: $(size(draws_3d)) | keys: $(length(key_index))")
    return nu_draws, draws_3d, key_index
end

# ==========================================================================
# 2. Run BLP for one spec — using pre-computed draws
# ==========================================================================
function run_blp_estimation(estim::Int, spec_id::Int, args,
                             nu_draws::Matrix{Float64},
                             draws_3d::Array{Float64,3},
                             key_index::Dict{Tuple{String,String},Int})
    println("\n" * "=" ^ 60)
    println("  Estimation $estim — Specification $spec_id")
    println("=" ^ 60)

    input_dir, _, _ = get_paths(args["hpc"]; local_dir=get(args, "local_dir", nothing))
    fname = input_filename(estim, spec_id)
    path  = joinpath(input_dir, fname)
    isfile(path) || (println("  [!] Missing: $path"); return nothing)

    df = DataFrame(Parquet2.Dataset(path); copycols=true)
    nrow(df) == 0 && (println("  [!] Empty dataframe."); return nothing)
    println("  Loaded: $(nrow(df)) observations")

    sigma_indices, pi_interactions, n_params = build_theta2_structure(args["stage"])
    R        = args["R"]
    seed     = args["seed"]
    coef_dim = 1 + L_PROD

    results = Dict{String,Any}("spec_id" => spec_id, "estim" => estim,
                                "stage" => args["stage"])

    if args["stage"] == "logit"
        println("  Stage: LOGIT — delegating to blp_logit_local.jl logic")
        is_B = Bool.(coalesce.(df.is_B, false))
        share_D      = Float64.(coalesce.(df.share_D,      0.0))
        share_B_cond = Float64.(coalesce.(df.share_B_cond, 0.0))
        delta = zeros(nrow(df))
        delta[.!is_B] .= log.(clamp.(share_D[.!is_B],      1e-15, Inf))
        delta[is_B]   .= log.(clamp.(share_B_cond[is_B],    1e-15, Inf))

        spread_cols, x_mat, Z_mat, iv_cols = build_regressor_matrices(df)
        H          = hcat(x_mat, Z_mat)
        dep_types  = Int.(coalesce.(df.deposit_type, 0))
        spread_hat = project_endogenous_spreads(spread_cols, H, dep_types)
        X_full     = hcat(spread_cols, x_mat)
        X_hat      = hcat(spread_hat,  x_mat)
        valid      = BitVector(all.(isfinite, eachrow(X_hat)))
        # Clustering: CodConglomeradoPrudencial only (matches sleep estimation)
        clusters   = string.(df.CodConglomeradoPrudencial)
        iv_avail   = [c for c in iv_cols
                      if std(replace(coalesce.(df[!, c], 0.0),
                                     Inf=>0.0, -Inf=>0.0)) > 1e-10]
        Z_clean = zeros(nrow(df), length(iv_avail))
        for (i, c) in enumerate(iv_avail)
            v = coalesce.(df[!, c], 0.0)
            replace!(v, Inf=>0.0, -Inf=>0.0)
            Z_clean[:, i] .= v
        end
        pc_logit = Precomp(Int[], Int[], Tuple{String,String}[], String[], Int[],
                           Float64[],
                           sparse(zeros(0,0)), sparse(zeros(0,0)), sparse(zeros(0,0)),
                           Int[], Int[], Int[], Int[], Int[], Int[], Int[], Int[],
                           BitVector(Bool.(coalesce.(df.is_B, false))),
                           .!BitVector(Bool.(coalesce.(df.is_B, false))),
                           log.(clamp.(share_D,      1e-15, Inf)),
                           log.(clamp.(share_B_cond, 1e-15, Inf)),
                           Z_clean, X_full, X_hat, valid, clusters)

        theta1, xi = estimate_theta1(delta, pc_logit)
        G  = compute_gmm_moments(xi, pc_logit)
        W  = Matrix(I(length(iv_avail)) * 1.0)
        Q  = dot(G, W * G)
        results["theta1"] = theta1; results["theta2"] = Float64[]
        results["delta"]  = delta;  results["xi"] = xi
        results["Q_value"] = Q;     results["converged"] = true
        println("  theta1 (alpha): $(round(theta1[1], sigdigits=6))")
        println("  Q(0) = $(round(Q, sigdigits=6))")
        return results
    end

    # ── BLP stages (sigma / full / extended) ─────────────────────────────
    println("  Stage: $(uppercase(args["stage"])) ($n_params parameters)")

    # Map observations to pre-computed draw indices
    N_obs     = nrow(df)
    mca_codes = string.(df.mca_code)
    time_ids  = string.(df.time_id)
    pad_idx   = size(draws_3d, 1)  # padding row (all zeros)

    obs_key_idx = [get(key_index, (mca_codes[i], time_ids[i]), pad_idx)
                   for i in 1:N_obs]

    n_mapped  = count(i -> i != pad_idx, obs_key_idx)
    log_status("  Obs mapped to draws: $n_mapped / $N_obs ($(round(100*n_mapped/N_obs, digits=1))%)")

    prod_vec = zeros(N_obs, coef_dim)
    spreads  = coalesce.(df.spread_qoq, 0.0)
    dep_types = Int.(coalesce.(df.deposit_type, 0))
    prod_vec[:, 1] .= spreads
    for (i, col) in enumerate(X_COLS)
        col in names(df) && (prod_vec[:, 1+i] .= coalesce.(df[!, col], 0.0))
    end

    spread_cols, x_mat, Z_mat, iv_cols = build_regressor_matrices(df)
    iv_avail = [c for c in iv_cols
                if std(replace(coalesce.(df[!, c], 0.0),
                               Inf=>0.0, -Inf=>0.0)) > 1e-10]
    Z = zeros(N_obs, length(iv_avail))
    for (i, c) in enumerate(iv_avail)
        v = coalesce.(df[!, c], 0.0)
        replace!(v, Inf=>0.0, -Inf=>0.0)
        Z[:, i] .= v
    end
    W = try inv(Z' * Z ./ N_obs) catch; Matrix(I(length(iv_avail)) * 1.0) end

    H          = hcat(x_mat, Z_mat)
    spread_hat = project_endogenous_spreads(spread_cols, H, dep_types)
    X_full     = hcat(spread_cols, x_mat)
    X_hat      = hcat(spread_hat,  x_mat)
    valid      = BitVector(all.(isfinite, eachrow(X_hat)))
    # Clustering: CodConglomeradoPrudencial only
    clusters   = string.(df.CodConglomeradoPrudencial)
    pc = build_precomp(df, Z, X_full, X_hat, valid, clusters)

    N_B     = sum(pc.b_mask)
    N_D     = sum(pc.d_mask)
    n_pairs = length(pc.unique_pairs)
    n_times = length(pc.unique_times)

    buf = allocate_hot_buffers(N_obs, N_B, N_D, R, n_pairs, n_times, coef_dim,
                                length(pi_interactions))
    precompute_pi_products!(buf, prod_vec, draws_3d, obs_key_idx,
                             pi_interactions, coef_dim)

    # ── Warm-start from previous stage checkpoint ────────────────────────
    _, _, out_dir = get_paths(args["hpc"])
    prev_stages = Dict("full" => "sigma", "extended" => "full")
    theta2_0 = nothing
    if args["stage"] in keys(prev_stages)
        prev_path = joinpath(out_dir,
            "blp_checkpoint_E$(estim)_spec_$(spec_id)_$(prev_stages[args["stage"]]).jls")
        if isfile(prev_path)
            try
                prev = deserialize(prev_path)
                t2p  = get(prev, "theta2_star", nothing)
                if t2p !== nothing && length(t2p) <= n_params
                    theta2_0 = zeros(n_params)
                    theta2_0[1:length(t2p)] .= t2p
                    println("  [WARM-START] theta2_0 from $prev_path")
                end
            catch e
                println("  [WARM-START] Could not load checkpoint: $e")
            end
        end
    end
    theta2_0 === nothing && (theta2_0 = randn(MersenneTwister(seed), n_params) .* 0.01)

    # ── Tighter bounds per CG2020 ────────────────────────────────────────
    lo = fill(-5.0, n_params)
    hi = fill( 5.0, n_params)

    println("  Outer minimisation: $(args["method"])")
    println("  Inner tolerance: $(args["tol_inner"])")
    println("  Bounds: [$(lo[1]), $(hi[1])]")

    delta_work = zeros(N_obs)
    delta_work[pc.d_mask] .= pc.ln_s_data_D[pc.d_mask]
    delta_work[pc.b_mask] .= pc.ln_s_data_B_cond[pc.b_mask]

    if get(args, "dry_run", false)
        println("  [DRY RUN] Running 10 contraction iterations for timing...")
        sv, pv = unpack_theta2(theta2_0, sigma_indices, pi_interactions)
        compute_mu!(buf, prod_vec, nu_draws, sv, sigma_indices, pv, R, coef_dim)
        t0 = time()
        blp_contraction!(buf, copy(delta_work), pc, R;
                          tol=args["tol_inner"], max_iter=10)
        elapsed = time() - t0
        println("  [DRY RUN] 10 iterations in $(round(elapsed, digits=2))s " *
                "($(round(elapsed/10, digits=3)) s/iter).")
        return Dict("dry_run" => true)
    end

    outer_iter = Ref(0)
    function obj_fn(t2)
        val = gmm_objective!(buf, t2, prod_vec, nu_draws,
                              sigma_indices, pi_interactions, R, coef_dim,
                              W, args["tol_inner"], args["max_inner"],
                              delta_work, pc)
        outer_iter[] += 1
        outer_iter[] % 10 == 0 &&
            log_status("  [OUTER iter=$(outer_iter[])] Q=$(round(val, sigdigits=6)) " *
                       "theta2=$(round.(t2, digits=4))")
        return val
    end

    result = optimize(obj_fn, lo, hi, theta2_0, Fminbox(LBFGS()),
                      Optim.Options(iterations=500, f_reltol=args["tol_outer"],
                                    show_trace=false))

    theta2_star = Optim.minimizer(result)
    println("  Optimiser converged: $(Optim.converged(result))")
    println("  Q(theta2*) = $(round(Optim.minimum(result), sigdigits=6))")
    println("  theta2* = $theta2_star")

    # Final contraction at optimum
    sv, pv = unpack_theta2(theta2_star, sigma_indices, pi_interactions)
    compute_mu!(buf, prod_vec, nu_draws, sv, sigma_indices, pv, R, coef_dim)
    delta_final = copy(delta_work)
    conv, n_it, _ = blp_contraction!(buf, delta_final, pc, R;
                                      tol=args["tol_inner"], max_iter=args["max_inner"])

    theta1_star, xi_star = estimate_theta1(delta_final, pc)

    results["theta1"]             = theta1_star
    results["theta2"]             = theta2_star
    results["delta"]              = delta_final
    results["xi"]                 = xi_star
    results["Q_value"]            = Optim.minimum(result)
    results["converged"]          = Optim.converged(result)
    results["n_outer_iter"]       = Optim.iterations(result)
    results["param_names_theta1"] = vcat(["alpha"], X_COLS)
    results["sigma_indices"]      = sigma_indices
    results["pi_interactions"]    = pi_interactions
    println("  theta1 (alpha): $(round(theta1_star[1], sigdigits=6))")

    # Save checkpoint
    mkpath(out_dir)
    chk_path = joinpath(out_dir,
        "blp_checkpoint_E$(estim)_spec_$(spec_id)_$(args["stage"]).jls")
    try
        serialize(chk_path, Dict("theta2_star" => theta2_star,
                                  "delta_star"  => delta_final))
        println("  [CHECKPOINT] Saved $(basename(chk_path))")
    catch e
        println("  [CHECKPOINT-WARN] $e")
    end

    return results
end

# ==========================================================================
# 3. CLI + Main
# ==========================================================================
function parse_args_est()
    s = ArgParseSettings(description="BLP Demand Estimation — Pre-Computed Draws")
    @add_arg_table! s begin
        "--estim";   arg_type=Int;     required=true;  help="Estimation strategy (1–5)"
        "--spec";    arg_type=String;  default="12"
        "--stage";   arg_type=String;  default="logit"
        "--R";       arg_type=Int;     default=2000
        "--seed";    arg_type=Int;     default=42
        "--tol-inner"; arg_type=Float64; default=1e-12; dest_name="tol_inner"
        "--max-inner"; arg_type=Int;     default=5000;  dest_name="max_inner"
        "--tol-outer"; arg_type=Float64; default=1e-6;  dest_name="tol_outer"
        "--method";  arg_type=String;  default="l-bfgs-b"
        "--workers"; arg_type=Int;     default=Threads.nthreads()
        "--hpc";     action=:store_true
        "--local-dir"; arg_type=String; default=nothing; dest_name="local_dir"
        "--dry-run"; action=:store_true; dest_name="dry_run"
    end
    return parse_args(s)
end

function main()
    args     = parse_args_est()
    estim    = args["estim"]
    spec_ids = args["spec"] == "all" ? collect(1:12) : [parse(Int, args["spec"])]

    log_status("BLP Estimation (Pre-Computed Draws) — START")
    log_status("  E$estim | Stage: $(args["stage"]) | Specs: $spec_ids")
    log_status("  R=$(args["R"]) | seed=$(args["seed"]) | HPC=$(args["hpc"])")
    log_status("  tol_inner=$(args["tol_inner"]) | tol_outer=$(args["tol_outer"]) | threads=$(Threads.nthreads())")

    _, draws_dir, out_dir = get_paths(args["hpc"]; local_dir=get(args, "local_dir", nothing))
    mkpath(out_dir)

    # Load pre-computed draws once (shared across all specs and stages)
    nu_draws, draws_3d, key_index = load_precomputed_draws(
        draws_dir, args["R"], args["seed"])

    stages_to_run = args["stage"] == "sequence" ?
                    ["logit", "sigma", "full", "extended"] : [args["stage"]]

    for current_stage in stages_to_run
        args["stage"] = current_stage

        # Stage-skipping
        all_done = true
        for sp in spec_ids
            rpath = joinpath(out_dir,
                "blp_results_E$(estim)_spec_$(sp)_$(current_stage).jls")
            if !isfile(rpath); all_done = false; break; end
        end
        if all_done
            log_status("[SKIP] Stage '$current_stage' already complete.")
            continue
        end

        all_results = Dict{Int,Any}()
        lk          = ReentrantLock()

        println("\n  Starting stage '$current_stage' for $(length(spec_ids)) specs " *
                "with $(Threads.nthreads()) threads...")

        Threads.@threads for sp in spec_ids
            out_path = joinpath(out_dir,
                "blp_results_E$(estim)_spec_$(sp)_$(current_stage).jls")
            if isfile(out_path)
                log_status("  [SKIP] Spec $sp already done for '$current_stage'")
                continue
            end

            res = try
                run_blp_estimation(estim, sp, args, nu_draws, draws_3d, key_index)
            catch e
                println("  [!] Spec $sp failed: $e"); nothing
            end
            res === nothing && continue
            get(res, "dry_run", false) && continue

            serialize(out_path, res)
            println("  Saved: $(basename(out_path))")
            lock(lk) do
                all_results[sp] = Dict(
                    "Q_value"      => get(res, "Q_value", 0.0),
                    "converged"    => get(res, "converged", true),
                    "theta1_alpha" => isempty(get(res, "theta1", Float64[])) ?
                                     [] : [res["theta1"][1]],
                    "theta2"       => get(res, "theta2", Float64[]),
                    "stage"        => current_stage)
            end
        end

        summary_path = joinpath(out_dir, "blp_summary_E$(estim)_$(current_stage).json")
        open(summary_path, "w") do f; JSON3.write(f, all_results); end
        log_status("Summary saved to: $summary_path")
        log_status("[DONE] BLP E$estim ($current_stage) complete for specs $spec_ids.")
    end
end

main()
