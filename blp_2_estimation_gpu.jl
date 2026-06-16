"""
blp_2_estimation_gpu.jl
========================
GPU-accelerated BLP demand estimation with IFT analytical gradient.

Inherits from blp_1_estimation_gpu.jl (all GPU infrastructure: CUDA kernels,
GpuBuffers, blp_contraction_gpu!, compute_model_shares_gpu!, …).

IFT extension for GPU
---------------------
After the inner contraction on GPU, the IFT gradient requires n_params forward
share passes.  On the H200 each forward pass (compute_model_shares_gpu!) takes
~1-2 ms vs ~200 ms for a full contraction, so the overhead is negligible.

  blp_1_gpu: (n_params+1) × full_inner_loop (Optim numerical gradient)
  blp_2_gpu: 1 × full_inner_loop + n_params × one_GPU_forward_pass

GPU memory note
---------------
The IFT gradient computes n_params forward passes sequentially — no extra VRAM
is needed.  All perturbation passes reuse the same GpuBuffers.

Usage (Bouchet H200 cluster)
-----------------------------
  julia --project=\${PROJECT_DIR} --threads=6 \\
      blp_2_estimation_gpu.jl --estim 5 --spec 12 --stage sigma \\
      --R 2000 --seed 42 --hpc
"""

# ── Load CPU baseline (IFT) ──────────────────────────────────────────────────
# blp_2_estimation.jl includes blp_1_estimation.jl and adds IFT gradient.
include(joinpath(@__DIR__, "blp_2_estimation.jl"))

using CUDA
CUDA.allowscalar(false)

# ── GPU infrastructure from blp_1 ──────────────────────────────────────────────
# Include GPU buffer definitions and kernels. The guard in blp_1_estimation_gpu.jl
# checks if X_COLS is already defined to avoid constant redefinition warnings.
include(joinpath(@__DIR__, "blp_1_estimation_gpu.jl"))

# ==========================================================================
# 8. IFT Analytical Gradient (GPU)
# ==========================================================================

"""
    collect_model_shares(buf, pc, N) → Vector{Float64}

Collect model shares from `buf.s_B` / `buf.s_D` (already on CPU after GPU sync)
into a full N-length vector in original observation order.
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
    compute_ift_gradient_gpu!(grad, g_moments, delta_star, s_base, theta2,
                               buf, gbuf, prod_vec, nu_draws,
                               sigma_indices, pi_interactions,
                               R, coef_dim, W, pc; ε=1e-5)

GPU version of the IFT gradient.  Each of the n_params perturbations runs
`compute_model_shares_gpu!` — a single GPU forward pass (~1-2 ms on H200),
much faster than a full blp_contraction_gpu! (~200+ ms).

CPU handles: μ computation (MKL, tiny k-dim), d_k scaling, linear projection.
GPU handles: the n_params share forward passes (logsumexp + GEMM).

After returning, `buf.mu` is restored to the base θ₂.
"""
function compute_ift_gradient_gpu!(grad::Vector{Float64},
                                    g_moments::Vector{Float64},
                                    delta_star::Vector{Float64},
                                    s_base::Vector{Float64},
                                    theta2::Vector{Float64},
                                    buf::HotBuffers,
                                    gbuf::GpuBuffers,
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

    Wg      = W * g_moments
    X_hat_v = pc.X_hat[valid, :]
    Xhat_qr = qr(X_hat_v)

    for k in 1:n_params
        # ── Perturb θ₂_k ──────────────────────────────────────────────────
        theta2_k     = copy(theta2)
        theta2_k[k] += ε
        sv_k, pv_k   = unpack_theta2(theta2_k, sigma_indices, pi_interactions)

        # ── ONE GPU forward pass at fixed δ* (NO inner contraction) ───────
        # compute_mu! stays on CPU (MKL GEMM over small k-dim)
        compute_mu!(buf, prod_vec, nu_draws, sv_k, sigma_indices, pv_k, R, coef_dim)
        compute_model_shares_gpu!(buf, gbuf, delta_star, pc, R)
        s_k = collect_model_shares(buf, pc, N)

        # ── Diagonal IFT: ∂δ*_i/∂θ₂_k ≈ −Δln(sᵢ)/ε/(1−sᵢ)  (NEGATIVE sign)
        # (1−sᵢ)∂δ*ᵢ/∂θₖ = ∂Tᵢ/∂θₖ = −∂ln sᵢ/∂θₖ — see blp_2_estimation.jl docstring.
        d_k = Vector{Float64}(undef, N)
        @inbounds for i in 1:N
            s_b         = s_base[i]
            one_minus_s = max(1.0 - s_b, 1e-4)
            d_k[i]      = -(log(max(s_k[i], 1e-15)) - log(s_b)) / ε / one_minus_s
        end

        # ── Project ───────────────────────────────────────────────────────
        dtheta1_k = Xhat_qr \ d_k[valid]
        dxi_k     = d_k - pc.X_full * dtheta1_k
        dg_k      = vec(Statistics.mean(dxi_k .* pc.Z_moments; dims=1))
        grad[k]   = 2.0 * dot(Wg, dg_k)
    end

    # Restore buf.mu to base θ₂
    sv_base, pv_base = unpack_theta2(theta2, sigma_indices, pi_interactions)
    compute_mu!(buf, prod_vec, nu_draws, sv_base, sigma_indices, pv_base, R, coef_dim)
    return nothing
end


"""
    gmm_fg_gpu!(F, G, theta2, ...) → Float64

Optim.only_fg! interface for GPU estimation: GPU inner contraction + GPU IFT
gradient.  Shares one inner-loop solve between Q and ∇Q.
"""
function gmm_fg_gpu!(G,
                      theta2::Vector{Float64},
                      buf::HotBuffers,
                      gbuf::GpuBuffers,
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
    converged, n_iter, _ = blp_contraction_gpu!(buf, gbuf, delta, pc, R;
                                                 tol=tol_inner, max_iter=max_inner)
    converged || println("  [!] Inner loop (GPU) did not converge in $n_iter iterations")

    theta1, xi = estimate_theta1(delta, pc)
    g_moments  = compute_gmm_moments(xi, pc)
    Q          = dot(g_moments, W * g_moments)

    if G !== nothing
        # Recompute shares at δ* on GPU to ensure buf.s_B/s_D are consistent
        compute_model_shares_gpu!(buf, gbuf, delta, pc, R)
        s_base = collect_model_shares(buf, pc, length(delta))

        compute_ift_gradient_gpu!(G, g_moments, delta, s_base, theta2,
                                   buf, gbuf, prod_vec, nu_draws,
                                   sigma_indices, pi_interactions,
                                   R, coef_dim, W, pc)
    end
    return Q
end


# ==========================================================================
# GPU Estimation Runner — IFT version
# ==========================================================================

"""
    run_blp_estimation_ift_gpu(estim, spec_id, args, nu_draws, draws_3d, key_index)

GPU-accelerated BLP estimation with IFT analytical gradient.
Logit stage delegates to the CPU path; all RC stages use GPU inner loops and
GPU forward passes for the gradient.
"""
function run_blp_estimation_ift_gpu(estim::Int, spec_id::Int, args,
                                     nu_draws::Matrix{Float64},
                                     draws_3d::Array{Float64,3},
                                     key_index::Dict{Tuple{String,String},Int})
    println("\n" * "=" ^ 60)
    println("  Estimation $estim — Specification $spec_id  [IFT GPU]")
    println("=" ^ 60)

    if args["stage"] == "logit"
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
    println("  IFT: 1 GPU inner loop + $n_params GPU forward pass(es) per outer iter")

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

    # ── Design-rank guard (aborts on degenerate/collinear X_hat column) ───
    check_design_rank(pc)

    buf  = allocate_hot_buffers(N_obs, N_B, N_D, R, n_pairs, n_times, coef_dim,
                                 length(pi_interactions))
    precompute_pi_products!(buf, prod_vec, draws_3d, obs_key_idx,
                             pi_interactions, coef_dim)

    log_status("  [GPU] Device: $(CUDA.name(CUDA.device()))")
    gbuf = allocate_gpu_buffers(buf, pc, N_obs, N_B, N_D, R, n_pairs, n_times)

    # ── θ₂ warm-start ─────────────────────────────────────────────────────
    _, _, out_dir = get_paths(args["hpc"])
    prev_stages   = Dict("rc2" => "sigma", "rc3" => "rc2", "rc4" => "rc3",
                         "full" => "rc4", "ext1" => "full", "ext2" => "ext1",
                         "extended" => "ext2")
    theta2_0 = nothing
    if args["stage"] in keys(prev_stages)
        prev_path = joinpath(out_dir,
            "blp_checkpoint_E$(estim)_spec_$(spec_id)_$(prev_stages[args["stage"]])$(output_suffix()).jls")
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
    theta2_0 === nothing &&
        (theta2_0 = randn(MersenneTwister(seed), n_params) .* 0.01)

    lo = fill(-2.0, n_params)
    hi = fill( 2.0, n_params)

    println("  Optimizer: L-BFGS-B + IFT analytical gradient (GPU)")
    println("  Inner tolerance: $(args["tol_inner"]) | bounds: [$(lo[1]), $(hi[1])]")

    # ── δ warm-start ──────────────────────────────────────────────────────
    delta_work = zeros(N_obs)
    let _loaded = false
        # BLP_DELTA_SUFFIX lets a coherence run warm-start from the suffixed delta
        # (logit_delta_E{k}_spec_{s}_coherence.*) that the logit-coherence step writes;
        # default "" keeps the legacy unsuffixed name. Suffixed is preferred, legacy is fallback.
        _suffix = get(ENV, "BLP_DELTA_SUFFIX", "")
        for _bin_cand in unique([
                joinpath(out_dir, "logit_delta_E$(estim)_spec_$(spec_id)$(_suffix).bin"),
                joinpath(out_dir, "logit_delta_E$(estim)_spec_$(spec_id).bin"),
            ])
            isfile(_bin_cand) || continue
            _d = load_delta_bin(_bin_cand)
            if _d !== nothing && length(_d) == N_obs
                copyto!(delta_work, _d)
                log_status("  [δ WARM-START] Loaded $(basename(_bin_cand)) [binary]")
                _loaded = true
                break
            end
        end
        if !_loaded
            for _cand in unique([
                joinpath(out_dir, "logit_delta_E$(estim)_spec_$(spec_id)$(_suffix).jls"),
                joinpath(out_dir, "logit_delta_E$(estim)_spec_$(spec_id).jls"),
                joinpath(out_dir, "blp_checkpoint_E$(estim)_spec_$(spec_id)_logit.jls"),
                joinpath(out_dir, "blp_results_E$(estim)_spec_$(spec_id)_logit.jls"),
            ])
                isfile(_cand) || continue
                try
                    _ck = deserialize(_cand)
                    _d  = get(_ck, "delta", nothing)
                    if _d !== nothing && length(_d) == N_obs
                        copyto!(delta_work, _d)
                        log_status("  [δ WARM-START] Loaded $(basename(_cand))")
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

    # ── GPU↔CPU share consistency guard (catches precision regressions) ───
    let (sv, pv) = unpack_theta2(theta2_0, sigma_indices, pi_interactions)
        compute_mu!(buf, prod_vec, nu_draws, sv, sigma_indices, pv, R, coef_dim)
        verify_gpu_shares(buf, gbuf, delta_work, pc, R; rtol=args["tol_inner"])
    end

    # ── Dry-run timing ────────────────────────────────────────────────────
    if get(args, "dry_run", false)
        println("  [DRY RUN GPU] 10 contraction iterations for timing...")
        sv, pv = unpack_theta2(theta2_0, sigma_indices, pi_interactions)
        compute_mu!(buf, prod_vec, nu_draws, sv, sigma_indices, pv, R, coef_dim)
        CUDA.synchronize()
        t0 = time()
        blp_contraction_gpu!(buf, gbuf, copy(delta_work), pc, R;
                              tol=args["tol_inner"], max_iter=10)
        CUDA.synchronize()
        elapsed = time() - t0
        println("  [DRY RUN GPU] 10 iters: $(round(elapsed, digits=2))s " *
                "($(round(elapsed/10, digits=3)) s/iter)")

        println("  [DRY RUN GPU] Timing $n_params IFT forward pass(es)...")
        compute_model_shares_gpu!(buf, gbuf, delta_work, pc, R)
        s_base = collect_model_shares(buf, pc, N_obs)
        g_dummy = zeros(n_params)
        t1 = time()
        for k in 1:n_params
            theta2_k = copy(theta2_0); theta2_k[k] += 1e-5
            sv_k, pv_k = unpack_theta2(theta2_k, sigma_indices, pi_interactions)
            compute_mu!(buf, prod_vec, nu_draws, sv_k, sigma_indices, pv_k, R, coef_dim)
            compute_model_shares_gpu!(buf, gbuf, delta_work, pc, R)
        end
        CUDA.synchronize()
        elapsed_ift = time() - t1
        println("  [DRY RUN GPU] $n_params forward passes: $(round(elapsed_ift*1000, digits=1)) ms " *
                "($(round(elapsed_ift/n_params*1000, digits=2)) ms/pass)")
        return Dict("dry_run" => true)
    end

    # ── Outer optimisation with IFT analytical gradient (GPU) ────────────
    outer_iter  = Ref(0)
    cache_t2    = fill(NaN, n_params)
    cache_Q     = Ref(0.0)
    cache_grad  = zeros(n_params)

    function ift_compute_gpu!(t2::Vector{Float64})
        isequal(t2, cache_t2) && return
        cache_Q[] = gmm_fg_gpu!(cache_grad, t2,
                                 buf, gbuf, prod_vec, nu_draws,
                                 sigma_indices, pi_interactions, R, coef_dim,
                                 W, args["tol_inner"], args["max_inner"],
                                 delta_work, pc)
        outer_iter[] += 1
        outer_iter[] % 10 == 0 &&
            log_status("  [OUTER iter=$(outer_iter[])] Q=$(round(cache_Q[], sigdigits=6)) " *
                       "theta2=$(round.(t2, digits=4))")
        copyto!(cache_t2, t2)
    end

    # True L-BFGS-B (LBFGSB.jl) instead of Optim's Fminbox barrier; same KKT point,
    # far fewer outer evals. f and g! share ift_compute_gpu!'s one-point cache, so the
    # GPU inner contraction is solved once per unique θ₂.
    f_obj(t2)     = (ift_compute_gpu!(t2); cache_Q[])
    g_obj!(G, t2) = (ift_compute_gpu!(t2); copyto!(G, cache_grad); G)
    theta2_star, Q_min, n_outer, conv_outer = solve_box_lbfgsb(
        f_obj, g_obj!, theta2_0, lo, hi;
        tol_outer = args["tol_outer"], maxiter = 500)

    println("  Optimizer converged: $(conv_outer)")
    println("  Q(θ₂*) = $(round(Q_min, sigdigits=6))")
    println("  theta2* = $theta2_star")

    # ── Final contraction at optimum (GPU) ────────────────────────────────
    sv, pv = unpack_theta2(theta2_star, sigma_indices, pi_interactions)
    compute_mu!(buf, prod_vec, nu_draws, sv, sigma_indices, pv, R, coef_dim)
    delta_final = copy(delta_work)
    conv, n_it, _ = blp_contraction_gpu!(buf, gbuf, delta_final, pc, R;
                                          tol=args["tol_inner"],
                                          max_iter=args["max_inner"])

    theta1_star, xi_star   = estimate_theta1(delta_final, pc)
    theta1_se, n_cl, G_s   = compute_cluster_se(theta1_star, delta_final, pc)

    results["theta1"]             = theta1_star
    results["theta1_se"]          = theta1_se
    results["theta2"]             = theta2_star
    results["theta2_se"]          = fill(0.0, length(theta2_star))
    results["delta"]              = delta_final
    results["xi"]                 = xi_star
    results["Q_value"]            = Q_min
    results["converged"]          = conv_outer
    results["n_outer_iter"]       = n_outer
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
        "blp_checkpoint_E$(estim)_spec_$(spec_id)_$(args["stage"])$(output_suffix()).jls")
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
# CLI + Main (GPU IFT)
# ==========================================================================

function main_gpu()
    if !CUDA.functional()
        error("[GPU] CUDA not functional on this node.  " *
              "Use blp_2_estimation.jl for CPU-only IFT estimation.")
    end
    dev = CUDA.device()
    log_status("[GPU] Using device: $(CUDA.name(dev)) " *
               "($(round(CUDA.totalmem(dev)/2^30, digits=1)) GB VRAM)")

    args     = parse_args_est()
    estim    = args["estim"]
    spec_ids = args["spec"] == "all" ? collect(1:12) : [parse(Int, args["spec"])]

    log_status("BLP Estimation GPU+IFT — START")
    log_status("  E$estim | Stage: $(args["stage"]) | Specs: $spec_ids")
    log_status("  R=$(args["R"]) | seed=$(args["seed"]) | HPC=$(args["hpc"])")
    log_status("  tol_inner=$(args["tol_inner"]) | tol_outer=$(args["tol_outer"]) " *
               "| threads=$(Threads.nthreads())")

    _, draws_dir, out_dir = get_paths(args["hpc"]; local_dir=args["local_dir"])
    mkpath(out_dir)

    nu_draws, draws_3d, key_index = load_precomputed_draws(
        draws_dir, args["R"], args["seed"])

    # "sequence" → all 8 stages; a comma list (e.g. "sigma,rc2,rc3,rc4,full,ext1")
    # → that SUBSET in one process (option-6 grouping: amortise Julia/CUDA/parquet
    # startup over several stages, each warm-starting the next from its on-disk
    # checkpoint); a single name → just that stage.
    stages_to_run = if args["stage"] == "sequence"
        ["sigma", "rc2", "rc3", "rc4", "full", "ext1", "ext2", "extended"]
    elseif occursin(',', args["stage"])
        String.(strip.(split(args["stage"], ',')))
    else
        [args["stage"]]
    end

    for current_stage in stages_to_run
        args["stage"] = current_stage

        all_done = all(isfile(joinpath(out_dir,
            "blp_results_E$(estim)_spec_$(sp)_$(current_stage)$(output_suffix()).jls"))
            for sp in spec_ids)
        if all_done
            log_status("[SKIP] Stage '$current_stage' already complete."); continue
        end

        println("\n  Starting stage '$current_stage' — $(length(spec_ids)) spec(s) " *
                "(sequential GPU execution)...")

        all_results = Dict{Int,Any}()

        for sp in spec_ids
            out_path = joinpath(out_dir,
                "blp_results_E$(estim)_spec_$(sp)_$(current_stage)$(output_suffix()).jls")
            if isfile(out_path)
                log_status("  [SKIP] Spec $sp already done for '$current_stage'"); continue
            end

            res = try
                run_blp_estimation_ift_gpu(estim, sp, args, nu_draws, draws_3d, key_index)
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

            all_results[sp] = Dict(
                "Q_value"      => get(res, "Q_value", 0.0),
                "converged"    => get(res, "converged", true),
                "theta1_alpha" => isempty(get(res, "theta1", Float64[])) ?
                                  [] : [res["theta1"][1]],
                "theta2"       => get(res, "theta2", Float64[]),
                "stage"        => current_stage)
        end

        summary_path = joinpath(out_dir,
            "blp_summary_E$(estim)_$(current_stage)_gpu_ift$(output_suffix()).json")
        open(summary_path, "w") do f; JSON3.write(f, all_results); end
        log_status("Summary saved to: $summary_path")
        log_status("[DONE] BLP GPU+IFT E$estim ($current_stage) complete for specs $spec_ids.")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_gpu()
end
