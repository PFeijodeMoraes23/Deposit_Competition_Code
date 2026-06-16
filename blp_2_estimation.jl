"""
blp_2_estimation.jl
===================
BLP demand estimation with IFT (Implicit Function Theorem) analytical gradient.

Key difference from blp_1_estimation.jl
-----------------------------------------
blp_1 computes ∂Q/∂θ₂ via Optim's built-in numerical finite differences, which
requires (n_params + 1) full inner-loop evaluations per outer L-BFGS step.

blp_2 computes ∂Q/∂θ₂ analytically: after the inner contraction converges to
δ*(θ₂), each parameter requires ONE forward share pass (no contraction), giving:

  blp_1: (n_params + 1) × full_inner_loop  per outer iter
  blp_2: 1 × full_inner_loop + n_params × one_forward_pass  per outer iter

For extended stage (n_params = 8) with ~200 SQUAREM iters per full solve:
  blp_1 ≈ 9 × 200 = 1 800 SQUAREM steps per outer iter
  blp_2 ≈ 1 × 200 + 8 × 1 =  208 SQUAREM steps per outer iter
  → ~8.7× cheaper per outer iteration; L-BFGS also converges in fewer iters.

IFT gradient derivation
-----------------------
The BLP contraction T(δ, θ₂) = δ + ln s_data − ln s(δ, θ₂) has fixed point δ*.
Differentiating T(δ*(θ₂), θ₂) = δ*(θ₂) w.r.t. θ₂_k:

  (I − ∂T/∂δ) ∂δ*/∂θ₂_k = ∂T/∂θ₂_k = −∂ ln s/∂θ₂_k

Diagonal approximation: ∂T_i/∂δ_i ≈ s_i  (own-share derivative of logit T),
so (1 − s_i) ∂δ*_i/∂θ₂_k ≈ −∂ ln s_i/∂θ₂_k

Forward-difference estimate:
  ∂δ*_i/∂θ₂_k ≈ −[log s_i(δ*, θ₂+ε·eₖ) − log s_i(δ*, θ₂)] / ε / (1 − s_i)

Projection to ∂Q/∂θ₂_k:
  θ₁* = X̂[valid] \\ δ*[valid]           (2SLS on converged δ*)
  ∂θ₁*/∂θ₂_k = X̂[valid] \\ d_k[valid]  (QR reuse, O(N·K) per k)
  ∂ξ/∂θ₂_k   = d_k − X_full · ∂θ₁/∂θ₂_k
  ∂g/∂θ₂_k   = mean(Z ⊙ ∂ξ/∂θ₂_k)
  ∂Q/∂θ₂_k   = 2 · (Wg)ᵀ ∂g/∂θ₂_k

Usage (cluster)
---------------
  julia --project=\${PROJECT_DIR} --threads=6 \\
      blp_2_estimation.jl --estim 5 --spec 12 --stage sigma \\
      --R 2000 --seed 42 --hpc

Local test
----------
  julia --project=. --threads=4 blp_2_estimation.jl \\
      --estim 1 --spec 12 --stage sigma --R 50 --seed 42
"""

# Load all shared BLP functions from blp_1_estimation.jl.
# The `if abspath(PROGRAM_FILE) == @__FILE__` guard there prevents auto-execution.
include(joinpath(@__DIR__, "blp_1_estimation.jl"))

# ==========================================================================
# 8. IFT Analytical Gradient
# ==========================================================================

"""
    collect_model_shares(buf, pc, N) → Vector{Float64}

Collect model shares from `buf.s_B` / `buf.s_D` into a full N-length vector
in original observation order (interleaved by `pc.b_mask`).
Clamps to 1e-15 to avoid log(0) in the IFT calculation.
"""
function collect_model_shares(buf::HotBuffers, pc::Precomp, N::Int)::Vector{Float64}
    s = Vector{Float64}(undef, N)
    b_idx = 0; d_idx = 0
    @inbounds for i in 1:N
        if pc.b_mask[i]
            b_idx += 1; s[i] = max(buf.s_B[b_idx], 1e-15)
        else
            d_idx += 1; s[i] = max(buf.s_D[d_idx], 1e-15)
        end
    end
    return s
end


"""
    compute_ift_gradient!(grad, g_moments, delta_star, s_base, theta2,
                          buf, prod_vec, nu_draws, sigma_indices, pi_interactions,
                          R, coef_dim, W, pc; ε=1e-5)

Fill `grad[k] = ∂Q/∂θ₂_k` for k = 1..n_params using the diagonal IFT
approximation.  Each k requires ONE forward share computation — no inner
contraction — so cost is O(n_params × one_forward_pass).

After returning, `buf.mu` is restored to the base θ₂ values.
"""
function compute_ift_gradient!(grad::Vector{Float64},
                                g_moments::Vector{Float64},
                                delta_star::Vector{Float64},
                                s_base::Vector{Float64},
                                theta2::Vector{Float64},
                                buf::HotBuffers,
                                prod_vec::Matrix{Float64},
                                nu_draws::Matrix{Float64},
                                sigma_indices::Vector{Int},
                                pi_interactions::Vector{Tuple{Int,Int}},
                                R::Int, coef_dim::Int,
                                W::Matrix{Float64},
                                pc::Precomp;
                                ε::Float64=1e-5)
    n_params = length(theta2)
    N        = length(delta_star)
    valid    = pc.theta1_valid .& isfinite.(delta_star)

    Wg      = W * g_moments                         # precompute W·g (reused for all k)
    X_hat_v = pc.X_hat[valid, :]
    Xhat_qr = qr(X_hat_v)                          # QR factored once; O(N·K) per k solve

    for k in 1:n_params
        # ── Perturb θ₂_k ──────────────────────────────────────────────────
        theta2_k     = copy(theta2)
        theta2_k[k] += ε
        sv_k, pv_k   = unpack_theta2(theta2_k, sigma_indices, pi_interactions)

        # ── ONE forward share pass at fixed δ* (NO inner contraction) ─────
        compute_mu!(buf, prod_vec, nu_draws, sv_k, sigma_indices, pv_k, R, coef_dim)
        compute_model_shares!(buf, delta_star, pc, R)
        s_k = collect_model_shares(buf, pc, N)

        # ── Diagonal IFT: ∂δ*_i/∂θ₂_k ≈ −Δln(sᵢ)/ε/(1−sᵢ) ─────────────
        # Sign: (1−sᵢ)∂δ*ᵢ/∂θₖ = ∂Tᵢ/∂θₖ = −∂ln sᵢ/∂θₖ  →  NEGATIVE.
        d_k = Vector{Float64}(undef, N)
        @inbounds for i in 1:N
            s_b         = s_base[i]
            one_minus_s = max(1.0 - s_b, 1e-4)     # clamp: avoids ÷0 when sᵢ ≈ 1
            d_k[i]      = -(log(max(s_k[i], 1e-15)) - log(s_b)) / ε / one_minus_s
        end

        # ── Project: ∂θ₁/∂θ₂_k = X̂[valid] \ d_k[valid] ──────────────────
        dtheta1_k = Xhat_qr \ d_k[valid]

        # ── ∂ξ/∂θ₂_k = d_k − X_full · ∂θ₁/∂θ₂_k ─────────────────────────
        dxi_k = d_k - pc.X_full * dtheta1_k

        # ── ∂g/∂θ₂_k = mean(Z ⊙ ∂ξ, dims=1); ∂Q/∂θ₂_k = 2·Wgᵀ·∂g ────────
        dg_k    = vec(Statistics.mean(dxi_k .* pc.Z_moments; dims=1))
        grad[k] = 2.0 * dot(Wg, dg_k)
    end

    # ── Restore buf.mu to base θ₂ so subsequent optimizer calls are consistent
    sv_base, pv_base = unpack_theta2(theta2, sigma_indices, pi_interactions)
    compute_mu!(buf, prod_vec, nu_draws, sv_base, sigma_indices, pv_base, R, coef_dim)
    return nothing
end


"""
    gmm_fg!(G, theta2, ...) → Float64

Compute GMM objective Q and, when `G` is not nothing, fill the analytical
gradient ∇Q via the diagonal IFT approximation.  Both share a single inner
contraction solve — ~n_params× cheaper per outer iter than numerical gradients.
"""
function gmm_fg!(G,
                  theta2::Vector{Float64},
                  buf::HotBuffers,
                  prod_vec::Matrix{Float64},
                  nu_draws::Matrix{Float64},
                  sigma_indices::Vector{Int},
                  pi_interactions::Vector{Tuple{Int,Int}},
                  R::Int, coef_dim::Int,
                  W::Matrix{Float64},
                  tol_inner::Float64, max_inner::Int,
                  delta::Vector{Float64},
                  pc::Precomp)::Float64

    sigma_vals, pi_vals = unpack_theta2(theta2, sigma_indices, pi_interactions)
    compute_mu!(buf, prod_vec, nu_draws, sigma_vals, sigma_indices, pi_vals, R, coef_dim)
    converged, n_iter, _ = blp_contraction!(buf, delta, pc, R;
                                             tol=tol_inner, max_iter=max_inner)
    converged || println("  [!] Inner loop did not converge in $n_iter iterations")

    theta1, xi = estimate_theta1(delta, pc)
    g_moments  = compute_gmm_moments(xi, pc)
    Q          = dot(g_moments, W * g_moments)

    if G !== nothing
        # Recompute shares at δ* to ensure buf.s_B/s_D are consistent.
        # (SQUAREM may exit after an accelerated step that displaced buf from δ*.)
        compute_model_shares!(buf, delta, pc, R)
        s_base = collect_model_shares(buf, pc, length(delta))

        compute_ift_gradient!(G, g_moments, delta, s_base, theta2,
                               buf, prod_vec, nu_draws,
                               sigma_indices, pi_interactions,
                               R, coef_dim, W, pc)
    end
    return Q
end


# ==========================================================================
# 2b. Run BLP Estimation — IFT version
# ==========================================================================

function run_blp_estimation_ift(estim::Int, spec_id::Int, args,
                                 nu_draws::Matrix{Float64},
                                 draws_3d::Array{Float64,3},
                                 key_index::Dict{Tuple{String,String},Int})
    println("\n" * "=" ^ 60)
    println("  Estimation $estim — Specification $spec_id  [IFT CPU]")
    println("=" ^ 60)

    # Logit stage has no θ₂ → no IFT needed; delegate to blp_1 CPU path
    sigma_indices_pre, _, n_params_pre = build_theta2_structure(
        get(args, "stage", "sigma"))
    if get(args, "stage", "sigma") == "logit" || n_params_pre == 0
        return run_blp_estimation(estim, spec_id, args, nu_draws, draws_3d, key_index)
    end

    input_dir, _, _ = get_paths(args["hpc"]; local_dir=args["local_dir"])
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
    println("  Stage: $(uppercase(args["stage"])) ($n_params parameters)")
    println("  IFT: 1 inner loop + $n_params forward pass(es) per outer iter")

    N_obs     = nrow(df)
    mca_codes = string.(df.mca_code)
    time_ids  = string.(df.time_id)
    pad_idx   = size(draws_3d, 1)

    obs_key_idx = [get(key_index, (mca_codes[i], time_ids[i]), pad_idx) for i in 1:N_obs]
    n_mapped    = count(i -> i != pad_idx, obs_key_idx)
    log_status("  Obs mapped to draws: $n_mapped / $N_obs ($(round(100*n_mapped/N_obs, digits=1))%)")

    prod_vec  = zeros(N_obs, coef_dim)
    spreads   = coalesce.(df.spread_ann, 0.0) ./ 100.0
    dep_types = Int.(coalesce.(df.deposit_type, 0))
    prod_vec[:, 1] .= spreads
    for (i, col) in enumerate(X_COLS)
        col in names(df) && (prod_vec[:, 1+i] .= coalesce.(df[!, col], 0.0))
    end

    spread_cols, x_mat, Z_mat, iv_cols = build_regressor_matrices(df)
    iv_avail = [c for c in iv_cols
                if std(replace(coalesce.(df[!, c], 0.0), Inf=>0.0, -Inf=>0.0)) > 1e-10]
    Z = zeros(N_obs, length(iv_avail))
    for (i, c) in enumerate(iv_avail)
        v = coalesce.(df[!, c], 0.0); replace!(v, Inf=>0.0, -Inf=>0.0)
        Z[:, i] .= v
    end
    W = try inv(Z' * Z ./ N_obs) catch; Matrix(I(length(iv_avail)) * 1.0) end

    H          = hcat(x_mat, Z_mat)
    spread_hat = project_endogenous_spreads(spread_cols, H, dep_types)
    X_full     = hcat(spread_cols, x_mat)
    X_hat      = hcat(spread_hat,  x_mat)
    valid      = BitVector(all.(isfinite, eachrow(X_hat)))
    clusters   = string.(df.CodConglomeradoPrudencial)
    pc         = build_precomp(df, Z, X_full, X_hat, valid, clusters)

    N_B     = sum(pc.b_mask)
    N_D     = sum(pc.d_mask)
    n_pairs = length(pc.unique_pairs)
    n_times = length(pc.unique_times)

    buf = allocate_hot_buffers(N_obs, N_B, N_D, R, n_pairs, n_times, coef_dim,
                                length(pi_interactions))
    precompute_pi_products!(buf, prod_vec, draws_3d, obs_key_idx,
                             pi_interactions, coef_dim)

    # ── θ₂ warm-start from previous stage checkpoint ──────────────────────
    _, _, out_dir = get_paths(args["hpc"])
    prev_stages   = Dict("rc2" => "sigma", "rc3" => "rc2", "rc4" => "rc3",
                         "full" => "rc4", "ext1" => "full", "ext2" => "ext1",
                         "extended" => "ext2")
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
                    println("  [WARM-START] theta2_0 from $(basename(prev_path))")
                end
            catch e
                println("  [WARM-START] Could not load checkpoint: $e")
            end
        end
    end
    theta2_0 === nothing && (theta2_0 = randn(MersenneTwister(seed), n_params) .* 0.01)

    lo = fill(-2.0, n_params)
    hi = fill( 2.0, n_params)

    println("  Optimizer: L-BFGS-B + IFT analytical gradient")
    println("  Inner tolerance: $(args["tol_inner"]) | bounds: [$(lo[1]), $(hi[1])]")

    # ── δ warm-start ──────────────────────────────────────────────────────
    delta_work = zeros(N_obs)
    let _loaded = false
        _bin_cand = joinpath(out_dir, "logit_delta_E$(estim)_spec_$(spec_id).bin")
        if isfile(_bin_cand)
            _d = load_delta_bin(_bin_cand)
            if _d !== nothing && length(_d) == N_obs
                copyto!(delta_work, _d)
                log_status("  [δ WARM-START] Loaded $(basename(_bin_cand)) [binary] (n=$(N_obs))")
                _loaded = true
            end
        end
        if !_loaded
            for _cand in [
                joinpath(out_dir, "logit_delta_E$(estim)_spec_$(spec_id).jls"),
                joinpath(out_dir, "blp_checkpoint_E$(estim)_spec_$(spec_id)_logit.jls"),
                joinpath(out_dir, "blp_results_E$(estim)_spec_$(spec_id)_logit.jls"),
            ]
                isfile(_cand) || continue
                try
                    _ck = deserialize(_cand)
                    _d  = get(_ck, "delta", nothing)
                    if _d !== nothing && length(_d) == N_obs
                        copyto!(delta_work, _d)
                        log_status("  [δ WARM-START] Loaded $(basename(_cand)) (n=$(N_obs))")
                        _loaded = true; break
                    end
                catch _e
                    log_status("  [δ WARM-START] Skipped $(basename(_cand)): $(_e)")
                end
            end
        end
        if !_loaded
            delta_work[pc.d_mask] .= pc.ln_s_data_D[pc.d_mask]
            delta_work[pc.b_mask] .= pc.ln_s_data_B_cond[pc.b_mask]
            log_status("  [δ WARM-START] No logit checkpoint — log-share init")
        end
    end

    if get(args, "dry_run", false)
        println("  [DRY RUN] 10 contraction iterations for timing...")
        sv, pv = unpack_theta2(theta2_0, sigma_indices, pi_interactions)
        compute_mu!(buf, prod_vec, nu_draws, sv, sigma_indices, pv, R, coef_dim)
        t0 = time()
        blp_contraction!(buf, copy(delta_work), pc, R; tol=args["tol_inner"], max_iter=10)
        elapsed = time() - t0
        println("  [DRY RUN] 10 iters: $(round(elapsed, digits=2))s " *
                "($(round(elapsed/10, digits=3)) s/iter).")
        return Dict("dry_run" => true)
    end

    # ── Outer optimisation with IFT analytical gradient ───────────────────
    # Use separate f / g! with a one-point cache so the inner contraction is
    # computed once per unique θ₂, regardless of how many times Optim calls
    # f vs g! at the same point.  Compatible with all Optim versions.
    outer_iter  = Ref(0)
    cache_t2    = fill(NaN, n_params)
    cache_Q     = Ref(0.0)
    cache_grad  = zeros(n_params)

    function ift_compute!(t2::Vector{Float64})
        isequal(t2, cache_t2) && return
        cache_Q[] = gmm_fg!(cache_grad, t2,
                             buf, prod_vec, nu_draws,
                             sigma_indices, pi_interactions, R, coef_dim,
                             W, args["tol_inner"], args["max_inner"],
                             delta_work, pc)
        outer_iter[] += 1
        outer_iter[] % 10 == 0 &&
            log_status("  [OUTER iter=$(outer_iter[])] Q=$(round(cache_Q[], sigdigits=6)) " *
                       "theta2=$(round.(t2, digits=4))")
        copyto!(cache_t2, t2)
    end

    result = optimize(
        t2 -> (ift_compute!(t2); cache_Q[]),
        (G, t2) -> (ift_compute!(t2); G .= cache_grad),
        lo, hi, theta2_0, Fminbox(LBFGS()),
        Optim.Options(iterations=500, f_reltol=args["tol_outer"], show_trace=false))

    theta2_star = Optim.minimizer(result)
    println("  Optimizer converged: $(Optim.converged(result))")
    println("  Q(θ₂*) = $(round(Optim.minimum(result), sigdigits=6))")
    println("  theta2* = $theta2_star")

    # ── Final contraction at optimum ──────────────────────────────────────
    sv, pv = unpack_theta2(theta2_star, sigma_indices, pi_interactions)
    compute_mu!(buf, prod_vec, nu_draws, sv, sigma_indices, pv, R, coef_dim)
    delta_final = copy(delta_work)
    conv, n_it, _ = blp_contraction!(buf, delta_final, pc, R;
                                      tol=args["tol_inner"], max_iter=args["max_inner"])

    theta1_star, xi_star   = estimate_theta1(delta_final, pc)
    theta1_se, n_cl, G_s   = compute_cluster_se(theta1_star, delta_final, pc)

    results["theta1"]             = theta1_star
    results["theta1_se"]          = theta1_se
    results["theta2"]             = theta2_star
    results["theta2_se"]          = fill(0.0, length(theta2_star))
    results["delta"]              = delta_final
    results["xi"]                 = xi_star
    results["Q_value"]            = Optim.minimum(result)
    results["converged"]          = Optim.converged(result)
    results["n_outer_iter"]       = Optim.iterations(result)
    results["param_names_theta1"] = vcat(["alpha"], X_COLS)
    results["sigma_indices"]      = sigma_indices
    results["pi_interactions"]    = pi_interactions
    results["n_clusters"]         = n_cl
    results["G_star"]             = G_s
    results["n_obs"]              = N_obs
    println("  theta1 (alpha): $(round(theta1_star[1], sigdigits=6))")

    # ── Save checkpoint ───────────────────────────────────────────────────
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
    chk_bin_path = replace(chk_path, ".jls" => ".bin")
    save_delta_bin(chk_bin_path, delta_final)
    println("  [CHECKPOINT BIN] Saved $(basename(chk_bin_path))")

    return results
end


# ==========================================================================
# 3b. CLI + Main (IFT)
# ==========================================================================

# Overwrites blp_1's main() when this file is executed directly or included.
function main()
    args     = parse_args_est()
    estim    = args["estim"]
    spec_ids = args["spec"] == "all" ? collect(1:12) : [parse(Int, args["spec"])]

    log_status("BLP Estimation IFT (Pre-Computed Draws) — START")
    log_status("  E$estim | Stage: $(args["stage"]) | Specs: $spec_ids")
    log_status("  R=$(args["R"]) | seed=$(args["seed"]) | HPC=$(args["hpc"])")
    log_status("  tol_inner=$(args["tol_inner"]) | tol_outer=$(args["tol_outer"]) | threads=$(Threads.nthreads())")

    _, draws_dir, out_dir = get_paths(args["hpc"]; local_dir=args["local_dir"])
    mkpath(out_dir)

    nu_draws, draws_3d, key_index = load_precomputed_draws(
        draws_dir, args["R"], args["seed"])

    stages_to_run = args["stage"] == "sequence" ?
                    ["sigma", "rc2", "rc3", "rc4", "full", "ext1", "ext2", "extended"] :
                    [args["stage"]]

    for current_stage in stages_to_run
        args["stage"] = current_stage

        all_done = all(isfile(joinpath(out_dir,
            "blp_results_E$(estim)_spec_$(sp)_$(current_stage).jls"))
            for sp in spec_ids)
        if all_done
            log_status("[SKIP] Stage '$current_stage' already complete."); continue
        end

        all_results = Dict{Int,Any}()
        lk          = ReentrantLock()
        println("\n  Starting stage '$current_stage' for $(length(spec_ids)) specs " *
                "with $(Threads.nthreads()) threads...")

        Threads.@threads for sp in spec_ids
            out_path = joinpath(out_dir,
                "blp_results_E$(estim)_spec_$(sp)_$(current_stage).jls")
            if isfile(out_path)
                log_status("  [SKIP] Spec $sp already done for '$current_stage'"); continue
            end

            res = try
                run_blp_estimation_ift(estim, sp, args, nu_draws, draws_3d, key_index)
            catch e
                println("  [!] Spec $sp failed: $e"); nothing
            end
            res === nothing && continue
            get(res, "dry_run", false) && continue

            serialize(out_path, res)
            println("  Saved: $(basename(out_path))")

            json_path = replace(out_path, ".jls" => ".json")
            try
                json_out = Dict{String,Any}(
                    "estim"              => get(res, "estim",   estim),
                    "spec_id"            => get(res, "spec_id", sp),
                    "stage"              => get(res, "stage",   current_stage),
                    "theta1"             => round.(get(res, "theta1",    Float64[]), sigdigits=8),
                    "theta1_se"          => replace(round.(get(res, "theta1_se", Float64[]), sigdigits=8), NaN=>0.0),
                    "theta2"             => round.(get(res, "theta2",    Float64[]), sigdigits=8),
                    "theta2_se"          => replace(round.(get(res, "theta2_se", Float64[]), sigdigits=8), NaN=>0.0),
                    "Q_value"            => get(res, "Q_value",   0.0),
                    "converged"          => get(res, "converged", false),
                    "param_names_theta1" => get(res, "param_names_theta1", String[]),
                    "sigma_indices"      => get(res, "sigma_indices",     Int[]),
                    "pi_interactions"    => get(res, "pi_interactions",   []),
                    "n_clusters"         => get(res, "n_clusters",        0),
                    "G_star"             => get(res, "G_star",            0.0),
                    "n_obs"              => get(res, "n_obs",             0),
                )
                open(json_path, "w") do f; JSON3.write(f, json_out); end
            catch e
                println("  [JSON-WARN] $e")
            end

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
        log_status("[DONE] BLP IFT E$estim ($current_stage) complete for specs $spec_ids.")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
