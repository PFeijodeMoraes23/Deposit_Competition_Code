# blp_gpu_engine.jl
# =================
# AUTO-GENERATED 2026-06-24 — merge of the three GPU engine files, in the order Julia
# loaded them on the IFT path:
#   blp_2_estimation.jl (IFT CPU) + blp_1_estimation_gpu.jl (numerical GPU)
#   + blp_2_estimation_gpu.jl (IFT GPU).
# The shared CPU baseline blp_1_estimation.jl is KEPT as a separate file (it's also used
# CPU-only, without CUDA, by foundation_demand_eval.jl) and is include()d below.
# Inter-file include()s were dropped; the two GPU entrypoints were renamed
#   main_gpu (blp_1 numerical) -> main_gpu_numerical ;  main_gpu (blp_2 IFT) -> main_gpu_ift
# so both engines coexist. blp_2_rc.jl include()s this file and calls the right
# one per BLP_ENGINE; for DIRECT execution the single dispatch block at the bottom
# picks the engine from BLP_ENGINE (default ift).
# EDIT THIS FILE for engine changes (the three originals were removed).

# CPU baseline — separate file (shared CPU-only with foundation_demand_eval.jl).
# The isdefined guard makes this load-once: several drivers reach the baseline both through this
# engine and on their own CPU-only path. `Base.include(Main, …)` is what a top-level `include` call
# expands to, spelled so the static include graph carries one edge per driver instead of two.
if !isdefined(Main, :X_COLS)
    Base.include(Main, joinpath(@__DIR__, "blp_1_estimation.jl"))
end

# The three entry points below resolve their demand parquet through `demand_parquet_path`,
# defined in blp_1_estimation.jl alongside `get_paths`/`input_filename` and pulled in by the
# include above (data/input first, then data/output/DEMAND_PREP).

# ===================== blp_2_estimation.jl =====================
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
# [merged] include(joinpath(@__DIR__, "blp_1_estimation.jl"))

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

    path = demand_parquet_path(estim, spec_id, args)
    path === nothing && return nothing

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
    # in_dir is the FIRST get_paths return (on HPC: data/input, where the uploaded
    # logit_delta warm-starts live); out_dir stays the checkpoint/result dir.
    in_dir, _, out_dir = get_paths(args["hpc"])
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
        for _bin_cand in [
                joinpath(in_dir,  "logit_delta_E$(estim)_spec_$(spec_id).bin"),
                joinpath(out_dir, "logit_delta_E$(estim)_spec_$(spec_id).bin"),
            ]
            isfile(_bin_cand) || continue
            _d = load_delta_bin(_bin_cand)
            if _d !== nothing && length(_d) == N_obs
                copyto!(delta_work, _d)
                log_status("  [δ WARM-START] Loaded $(basename(_bin_cand)) [binary] (n=$(N_obs))")
                _loaded = true
                break
            end
        end
        if !_loaded
            for _cand in [
                joinpath(in_dir,  "logit_delta_E$(estim)_spec_$(spec_id).jls"),
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
    results["theta2_se"]          = fill(NaN, length(theta2_star))   # CPU-IFT path: θ₂ SEs not computed (sentinel)
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
                    "theta1_pval"        => replace(round.(get(res, "theta1_pval", Float64[]), sigdigits=6), NaN=>1.0),
                    "theta2_pval"        => replace(round.(get(res, "theta2_pval", Float64[]), sigdigits=6), NaN=>1.0),
                    "se_method"          => get(res, "se_method", "none"),
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

# [merged: per-file autorun removed — see dispatch at end]

# ===================== blp_1_estimation_gpu.jl =====================
"""
blp_estimation_gpu.jl
=====================
GPU-accelerated BLP demand estimation.

Reuses all CPU utilities from blp_estimation.jl via include().
Only the hot-path inner loop (logsumexp grouped reductions + share
computation) runs on GPU; SQUAREM delta updates and outer GMM
optimisation remain on CPU.

Design principles
-----------------
* Float32 on device — halves bandwidth vs Float64; on Tensor-Core GPUs
  (A100/V100) the logsumexp kernels run ~16× faster than Float64 BLAS.
* Static index arrays (sort_b, grp_start, …) are uploaded once in
  allocate_gpu_buffers and never touched again.
* PT_agg is tiny (n_times × n_pairs ≈ 60×500) — stored dense on device
  so CUBLAS GEMM handles sum_wtd without a sparse-matrix call.
* mu is computed by MKL on CPU (fast, small k-dim), then bulk-copied to
  mu_gpu once per outer optimiser iteration before each contraction.

Requirements
------------
* NVIDIA GPU with CUDA >= 11 / CUDA.jl >= 5
* Yale Bouchet: `module load CUDA/12.x` before julia

Usage (cluster GPU node)
------------------------
  julia --project=\${PROJECT_DIR} --threads=4 \\
      blp_estimation_gpu.jl --estim 5 --spec 12 --stage sigma \\
      --R 2000 --seed 42 --hpc
"""

# ── Load CPU baseline ────────────────────────────────────────────────────────
# Check if baseline has already been loaded (e.g., via blp_2_estimation.jl)
# to avoid constant redefinition warnings when both blp_1 and blp_2 GPU scripts run.
if !isdefined(Main, :X_COLS)
# [merged] include(joinpath(@__DIR__, "blp_1_estimation.jl"))
end

using CUDA
CUDA.allowscalar(false)   # hard-fail on accidental scalar GPU indexing

# ── Device floating-point precision ─────────────────────────────────────────
# Change to Float64 if the GPU has fast FP64 (e.g. A100 SXM) and you need
# tighter numerical tolerances on the inner contraction.
const GPU_T = Float64

# ==========================================================================
# GPU Buffers
# ==========================================================================
"""
    GpuBuffers

Holds `CuArray` mirrors of the hot-path matrices used inside the BLP inner
loop.  Fields are organised into four groups:

1. **Per-observation tensors** (`V_B`, `V_D`, `q_B`, `mu_gpu`, …) —
   overwritten every `T_inplace!` call.
2. **Grouped reduction results** (`log_sum_D_time`, `log_denom`, …) —
   intermediate outputs of the CUDA kernels.
3. **Output vectors** (`s_B_gpu`, `s_D_gpu`) — copied back to CPU after
   each call so SQUAREM can update `delta_work`.
4. **Static index arrays** (`sort_b_gpu`, `b_grp_start_gpu`, …) —
   uploaded once in `allocate_gpu_buffers`, never modified.

CPU fields that require arbitrary-precision arithmetic (delta, mu, theta1
solve, SQUAREM step) live in the ordinary `HotBuffers`.
"""
mutable struct GpuBuffers
    # ── Per-observation tensors (rewritten every contraction step) ────────
    V_B              ::CuMatrix{GPU_T}   # N_B × R
    V_D              ::CuMatrix{GPU_T}   # N_D × R
    q_B              ::CuMatrix{GPU_T}   # N_B × R
    delta_B          ::CuVector{GPU_T}   # N_B
    delta_D          ::CuVector{GPU_T}   # N_D
    mu_gpu           ::CuMatrix{GPU_T}   # N   × R  (bulk-copy of buf.mu each outer step)

    # ── Grouped reduction intermediate results ────────────────────────────
    log_sum_D_time   ::CuMatrix{GPU_T}   # n_times × R
    log_sum_B_mkt    ::CuMatrix{GPU_T}   # n_pairs × R
    log_D_sum_pair   ::CuMatrix{GPU_T}   # n_pairs × R
    log_denom        ::CuMatrix{GPU_T}   # n_pairs × R
    neg_log_denom    ::CuMatrix{GPU_T}   # n_pairs × R
    max_neg_ld       ::CuMatrix{GPU_T}   # n_times × R
    shifted_inv      ::CuMatrix{GPU_T}   # n_pairs × R
    sum_wtd          ::CuMatrix{GPU_T}   # n_times × R
    log_inv_wtd      ::CuMatrix{GPU_T}   # n_times × R
    log_s_D_r        ::CuMatrix{GPU_T}   # N_D  × R
    row_max_D        ::CuVector{GPU_T}   # N_D  (per-obs row max for s_D mean)

    # ── Output (copied back to CPU after each T_inplace! call) ────────────
    s_B_gpu          ::CuVector{GPU_T}   # N_B
    s_D_gpu          ::CuVector{GPU_T}   # N_D

    # ── Static index arrays (uploaded once at allocation time) ────────────
    # Observation-to-group mappings
    b_mkt_idx_gpu    ::CuVector{Int32}   # N_B  → pair index  (1-based)
    d_time_enc_gpu   ::CuVector{Int32}   # N_D  → time index  (1-based)
    pair_time_enc_gpu::CuVector{Int32}   # n_pairs → time index (1-based)

    # Original position of B / D observations in the full N-vector
    b_mask_idx_gpu   ::CuVector{Int32}   # N_B  (position in 1:N)
    d_mask_idx_gpu   ::CuVector{Int32}   # N_D  (position in 1:N)

    # Sorted-order arrays for grouped reductions (B groups = mkt-pairs)
    sort_b_gpu       ::CuVector{Int32}   # N_B
    b_uval_gpu       ::CuVector{Int32}   # n_pairs  (== 1:n_pairs; kept for uniform kernel)
    b_grp_start_gpu  ::CuVector{Int32}   # n_pairs+1  (sentinel-ended)

    # Sorted-order arrays for D groups (= time periods)
    sort_d_gpu       ::CuVector{Int32}   # N_D
    d_uval_gpu       ::CuVector{Int32}   # n_d_groups (unique time indices in d_time_enc)
    d_grp_start_gpu  ::CuVector{Int32}   # n_d_groups+1 (sentinel-ended)

    # Sorted-order arrays for pair-time groups (used for max_neg_ld)
    sort_pt_gpu      ::CuVector{Int32}   # n_pairs
    pt_uval_gpu      ::CuVector{Int32}   # n_pt_groups
    pt_grp_start_gpu ::CuVector{Int32}   # n_pt_groups+1 (sentinel-ended)

    # PT_agg dense mirror for CUBLAS GEMM (n_times × n_pairs)
    PT_agg_gpu       ::CuMatrix{GPU_T}

    # ── Dimensions (stored for kernel-launch arithmetic) ──────────────────
    N                ::Int
    N_B              ::Int
    N_D              ::Int
    R                ::Int
    n_pairs          ::Int
    n_times          ::Int
    n_b_groups       ::Int   # == n_pairs
    n_d_groups       ::Int   # == length(pc.d_uval)
    n_pt_groups      ::Int   # == length(pc.pt_uval)
end

# ==========================================================================
# Helpers
# ==========================================================================

"""
    _sentinel_grp_start(grp_start, n_obs) → Vector{Int32}

Appends a sentinel entry `n_obs + 1` so kernels can compute group bounds
as `grp_start[g] : grp_start[g+1] - 1` without a bounds check.
"""
function _sentinel_grp_start(grp_start::Vector{Int}, n_obs::Int)::Vector{Int32}
    return Int32.(vcat(grp_start, n_obs + 1))
end

# ==========================================================================
# Allocator
# ==========================================================================

"""
    allocate_gpu_buffers(buf, pc, N, N_B, N_D, R, n_pairs, n_times) → GpuBuffers

Allocates all device arrays and uploads static index/weight arrays once.
`buf` and `pc` are the existing CPU `HotBuffers` / `Precomp` from the
normal CPU estimation path.

Call once per spec; reuse across all outer optimiser iterations.
"""
function allocate_gpu_buffers(buf::HotBuffers, pc::Precomp,
                               N::Int, N_B::Int, N_D::Int,
                               R::Int, n_pairs::Int, n_times::Int)::GpuBuffers

    n_d_groups  = length(pc.d_uval)
    n_pt_groups = length(pc.pt_uval)
    n_b_groups  = n_pairs   # one group per market-pair

    # Footprint estimate (GPU_T bytes per element)
    bytes_per = sizeof(GPU_T)
    gb = (
        N_B * R * 3 +   # V_B, q_B, (reused for shifted_inv borrowing)
        N_D * R * 2 +   # V_D, log_s_D_r
        N   * R     +   # mu_gpu
        N_B         +   # delta_B, s_B_gpu  (×2)
        N_D         +   # delta_D, s_D_gpu, row_max_D  (×3)
        n_times * R * 4 +   # log_sum_D_time, max_neg_ld, sum_wtd, log_inv_wtd
        n_pairs * R * 5     # log_sum_B_mkt, log_D_sum_pair, log_denom,
                            # neg_log_denom, shifted_inv
    ) * bytes_per / 1e9
    println("  [GPU] Allocating buffers — estimated footprint: $(round(gb, digits=2)) GB")

    # Static index arrays
    b_mask_idx_cpu  = Int32.(findall(pc.b_mask))
    d_mask_idx_cpu  = Int32.(findall(pc.d_mask))
    b_uval_cpu      = Int32.(1:n_pairs)

    b_grp_sent  = _sentinel_grp_start(pc.b_grp_start,  N_B)
    d_grp_sent  = _sentinel_grp_start(pc.d_grp_start,  N_D)
    pt_grp_sent = _sentinel_grp_start(pc.pt_grp_start, n_pairs)

    # PT_agg: sparse → dense for CUBLAS (n_times × n_pairs, tiny)
    PT_agg_dense = Matrix{GPU_T}(pc.PT_agg)

    gbuf = GpuBuffers(
        # Per-observation tensors
        CUDA.zeros(GPU_T, N_B, R),           # V_B
        CUDA.zeros(GPU_T, N_D, R),           # V_D
        CUDA.zeros(GPU_T, N_B, R),           # q_B
        CUDA.zeros(GPU_T, N_B),              # delta_B
        CUDA.zeros(GPU_T, N_D),              # delta_D
        CUDA.zeros(GPU_T, N,  R),            # mu_gpu

        # Grouped reduction intermediates
        CUDA.fill(GPU_T(-Inf32), n_times, R), # log_sum_D_time
        CUDA.fill(GPU_T(-Inf32), n_pairs, R), # log_sum_B_mkt
        CUDA.zeros(GPU_T, n_pairs, R),        # log_D_sum_pair
        CUDA.zeros(GPU_T, n_pairs, R),        # log_denom
        CUDA.zeros(GPU_T, n_pairs, R),        # neg_log_denom
        CUDA.fill(GPU_T(-Inf32), n_times, R), # max_neg_ld
        CUDA.zeros(GPU_T, n_pairs, R),        # shifted_inv
        CUDA.zeros(GPU_T, n_times, R),        # sum_wtd
        CUDA.zeros(GPU_T, n_times, R),        # log_inv_wtd
        CUDA.zeros(GPU_T, N_D, R),            # log_s_D_r
        CUDA.fill(GPU_T(-Inf32), N_D),        # row_max_D

        # Output
        CUDA.zeros(GPU_T, N_B),              # s_B_gpu
        CUDA.zeros(GPU_T, N_D),              # s_D_gpu

        # Static index arrays — uploaded once
        CuVector{Int32}(Int32.(pc.b_mkt_idx)),
        CuVector{Int32}(Int32.(pc.d_time_enc)),
        CuVector{Int32}(Int32.(pc.pair_time_enc)),
        CuVector{Int32}(b_mask_idx_cpu),
        CuVector{Int32}(d_mask_idx_cpu),
        CuVector{Int32}(Int32.(pc.sort_b)),
        CuVector{Int32}(b_uval_cpu),
        CuVector{Int32}(b_grp_sent),
        CuVector{Int32}(Int32.(pc.sort_d)),
        CuVector{Int32}(Int32.(pc.d_uval)),
        CuVector{Int32}(d_grp_sent),
        CuVector{Int32}(Int32.(pc.sort_pt)),
        CuVector{Int32}(Int32.(pc.pt_uval)),
        CuVector{Int32}(pt_grp_sent),
        CuMatrix{GPU_T}(PT_agg_dense),

        # Dimensions
        N, N_B, N_D, R,
        n_pairs, n_times,
        n_b_groups, n_d_groups, n_pt_groups,
    )

    free_b, total_b = CUDA.memory_info()
    @printf("  [GPU] VRAM after alloc: %.2f / %.2f GB free\n", free_b / 2^30, total_b / 2^30)
    return gbuf
end

# ==========================================================================
# CUDA Kernels
# ==========================================================================

# ── Kernel tuning constants ──────────────────────────────────────────────────
# RBLOCK: number of draws handled per CUDA thread block column.
# 32 = one warp; keeps shared-memory pressure low while hiding memory latency.
const RBLOCK = Int32(32)

"""
    logsumexp_groups_kernel!(result, V, sort_idx, uval, grp_start,
                              n_groups, N_obs, R)

For each group `g` (1-based) and draw `r`:

  result[uval[g], r] = max_i V[sort_idx[i], r]
                     + log Σ_i exp(V[sort_idx[i], r] − max_i)

where the sum ranges over `i ∈ grp_start[g] : grp_start[g+1]-1`.

**Grid / block layout**

  gridDim  = (n_groups, cld(R, RBLOCK))
  blockDim = (RBLOCK, 1, 1)

Each thread handles exactly one `(g, r)` pair: sequential scan over the
group for the max pass, then a second scan for the exp-sum.  No shared
memory or atomics are needed because every thread writes to a distinct
`result` cell.

**Arguments** (all device arrays)

- `result`    — output matrix, row-major layout `[n_result_rows, R]`
- `V`         — value matrix `[N_obs, R]`
- `sort_idx`  — observation sort order `[N_obs]`, 1-based
- `uval`      — group → result-row mapping `[n_groups]`, 1-based
- `grp_start` — sentinel-ended group starts `[n_groups+1]`, 1-based
- `n_groups`  — number of groups (Int32)
- `N_obs`     — total observations (Int32, for bounds guard)
- `R`         — number of draws (Int32)
"""
function logsumexp_groups_kernel!(
        result   ::CuDeviceMatrix{GPU_T},
        V        ::CuDeviceMatrix{GPU_T},
        sort_idx ::CuDeviceVector{Int32},
        uval     ::CuDeviceVector{Int32},
        grp_start::CuDeviceVector{Int32},
        n_groups ::Int32,
        N_obs    ::Int32,
        R        ::Int32)

    g = Int32(blockIdx().x)
    r = Int32((blockIdx().y - 1) * blockDim().x + threadIdx().x)

    (g > n_groups || r > R) && return nothing

    gs  = grp_start[g]
    ge  = grp_start[g + Int32(1)] - Int32(1)
    row = uval[g]

    # Single-pass online log-sum-exp: maintain running max `mx` and running sum
    # `s = Σ exp(v_j − mx)`; when a larger value appears, rescale `s` to the new
    # max before adding. Reads each V element ONCE (vs twice for the max-then-sum
    # two-pass), halving the gather traffic of this bandwidth-bound kernel.
    # Numerically identical to the two-pass form (verify_gpu_shares guards it).
    #   First element: mx=-Inf, s=0 → s = 0·exp(-Inf) + 1 = 1, mx = v  (exact).
    mx = GPU_T(-Inf32)
    s  = GPU_T(0)
    @inbounds for ii in gs:ge
        v = V[sort_idx[ii], r]
        if v > mx
            s  = s * CUDA.exp(mx - v) + GPU_T(1)
            mx = v
        else
            s += CUDA.exp(v - mx)
        end
    end

    @inbounds result[row, r] = mx + CUDA.log(max(s, GPU_T(1e-30)))
    return nothing
end

"""
    groupmax_kernel!(result, V, sort_idx, uval, grp_start,
                     n_groups, N_obs, R)

Fills `result[uval[g], r]` with the **maximum** of `V[sort_idx[i], r]`
over all `i` in group `g`.  Same grid/block layout as
`logsumexp_groups_kernel!`.

Used to compute `max_neg_ld[t, r]` — the per-time-period maximum of
`neg_log_denom` over all market-pairs in period `t`.  This maximum is
needed to numerically stabilise the `shifted_inv` computation:

  shifted_inv[p, r] = exp(neg_log_denom[p, r] − max_neg_ld[time(p), r])
"""
function groupmax_kernel!(
        result   ::CuDeviceMatrix{GPU_T},
        V        ::CuDeviceMatrix{GPU_T},
        sort_idx ::CuDeviceVector{Int32},
        uval     ::CuDeviceVector{Int32},
        grp_start::CuDeviceVector{Int32},
        n_groups ::Int32,
        N_obs    ::Int32,
        R        ::Int32)

    g = Int32(blockIdx().x)
    r = Int32((blockIdx().y - 1) * blockDim().x + threadIdx().x)

    (g > n_groups || r > R) && return nothing

    gs  = grp_start[g]
    ge  = grp_start[g + Int32(1)] - Int32(1)
    row = uval[g]

    mx = GPU_T(-Inf32)
    @inbounds for ii in gs:ge
        v = V[sort_idx[ii], r]
        v > mx && (mx = v)
    end

    @inbounds result[row, r] = mx
    return nothing
end

# ── Convenience launch wrappers ──────────────────────────────────────────────

"""
    launch_logsumexp!(result, V, sort_idx, uval, grp_start, n_groups, N_obs, R)

Compute grouped log-sum-exp on the GPU.

`result` is filled **in-place** (`fill!(-Inf)` is called first so rows
belonging to absent groups remain `-Inf`).
"""
function launch_logsumexp!(result::CuMatrix{GPU_T},
                            V        ::CuMatrix{GPU_T},
                            sort_idx ::CuVector{Int32},
                            uval     ::CuVector{Int32},
                            grp_start::CuVector{Int32},
                            n_groups ::Int, N_obs::Int, R::Int)
    fill!(result, GPU_T(-Inf32))
    n_groups == 0 && return
    grid  = (n_groups, cld(R, Int(RBLOCK)))
    block = (Int(RBLOCK),)
    CUDA.@cuda threads=block blocks=grid logsumexp_groups_kernel!(
        result, V, sort_idx, uval, grp_start,
        Int32(n_groups), Int32(N_obs), Int32(R))
end

"""
    launch_groupmax!(result, V, sort_idx, uval, grp_start, n_groups, N_obs, R)

Compute grouped maximum on the GPU (`result` filled in-place).
"""
function launch_groupmax!(result::CuMatrix{GPU_T},
                           V        ::CuMatrix{GPU_T},
                           sort_idx ::CuVector{Int32},
                           uval     ::CuVector{Int32},
                           grp_start::CuVector{Int32},
                           n_groups ::Int, N_obs::Int, R::Int)
    fill!(result, GPU_T(-Inf32))
    n_groups == 0 && return
    grid  = (n_groups, cld(R, Int(RBLOCK)))
    block = (Int(RBLOCK),)
    CUDA.@cuda threads=block blocks=grid groupmax_kernel!(
        result, V, sort_idx, uval, grp_start,
        Int32(n_groups), Int32(N_obs), Int32(R))
end

# ==========================================================================
# GPU Share Computation
# ==========================================================================

"""
    compute_model_shares_gpu!(buf, gbuf, delta, pc, R)

GPU-accelerated drop-in replacement for `compute_model_shares!`.

Steps and GPU ops:
  1. Copy buf.mu (CPU Float64) → gbuf.mu_gpu (GPU Float32)
  2. Scatter δ → delta_B / delta_D via index gather
  3. V_B = clamp(δ_B + μ_B, -500, 500); same for V_D  [broadcast]
  4. log_sum_D_time via logsumexp_groups_kernel!
  5. Broadcast log_sum_D_time → log_D_sum_pair (pair-level)
  6. log_sum_B_mkt via logsumexp_groups_kernel!
  7. log_denom = log(outside + Σ_B + Σ_D)  [broadcast]
  8. q_B = exp(V_B − log_denom[mkt])        [broadcast]
  9. s_B = mean_r q_B                        [sum + scale]
 10. max_neg_ld via groupmax_kernel!
 11. shifted_inv = exp(neg_log_denom − max_neg_ld)  [broadcast]
 12. sum_wtd = PT_agg × shifted_inv          [CUBLAS GEMM]
 13. log_inv_wtd = max_neg_ld + log(sum_wtd) [broadcast]
 14. log_s_D_r = V_D + log_inv_wtd           [broadcast]
 15. s_D via row-max numerically stable mean  [broadcast + sum]
 16. Copy s_B, s_D back to CPU Float64

GPU_T precision (Float32) is used throughout; s_B and s_D are widened
to Float64 when written back to buf.
"""
function compute_model_shares_gpu!(buf::HotBuffers, gbuf::GpuBuffers,
                                    delta::Vector{Float64},
                                    pc::Precomp, R::Int)

    N_B     = gbuf.N_B
    N_D     = gbuf.N_D
    n_pairs = gbuf.n_pairs
    n_times = gbuf.n_times

    # ── 1. Copy mu (CPU Float64 → GPU Float32) ──────────────────────────
    copyto!(gbuf.mu_gpu, GPU_T.(buf.mu))

    # ── 2. Scatter δ → delta_B / delta_D ────────────────────────────────
    delta_gpu_full = CuVector{GPU_T}(GPU_T.(delta))
    gbuf.delta_B .= @view delta_gpu_full[gbuf.b_mask_idx_gpu]
    gbuf.delta_D .= @view delta_gpu_full[gbuf.d_mask_idx_gpu]

    # ── 3. V_B / V_D ────────────────────────────────────────────────────
    gbuf.V_B .= clamp.(gbuf.delta_B .+ @view(gbuf.mu_gpu[gbuf.b_mask_idx_gpu, :]),
                       GPU_T(-500f0), GPU_T(500f0))
    gbuf.V_D .= clamp.(gbuf.delta_D .+ @view(gbuf.mu_gpu[gbuf.d_mask_idx_gpu, :]),
                       GPU_T(-500f0), GPU_T(500f0))

    # ── 4. log_sum_D_time ────────────────────────────────────────────────
    launch_logsumexp!(gbuf.log_sum_D_time,
                      gbuf.V_D, gbuf.sort_d_gpu, gbuf.d_uval_gpu,
                      gbuf.d_grp_start_gpu,
                      gbuf.n_d_groups, N_D, R)

    # ── 5. Broadcast to pairs ────────────────────────────────────────────
    gbuf.log_D_sum_pair .= @view gbuf.log_sum_D_time[gbuf.pair_time_enc_gpu, :]

    # ── 6. log_sum_B_mkt ────────────────────────────────────────────────
    launch_logsumexp!(gbuf.log_sum_B_mkt,
                      gbuf.V_B, gbuf.sort_b_gpu, gbuf.b_uval_gpu,
                      gbuf.b_grp_start_gpu,
                      gbuf.n_b_groups, N_B, R)

    # ── 7. log_denom ─────────────────────────────────────────────────────
    # log_outside = 0 (one outside good with utility 0)
    let lsB = gbuf.log_sum_B_mkt, lsD = gbuf.log_D_sum_pair
        jm = max.(GPU_T(0f0), lsB, lsD)
        gbuf.log_denom .= jm .+ log.(max.(
            exp.(GPU_T(0f0) .- jm) .+ exp.(lsB .- jm) .+ exp.(lsD .- jm),
            GPU_T(1e-30)))
    end

    # ── 8–9. q_B → s_B ──────────────────────────────────────────────────
    gbuf.q_B     .= exp.(gbuf.V_B .- @view(gbuf.log_denom[gbuf.b_mkt_idx_gpu, :]))
    gbuf.s_B_gpu .= vec(sum(gbuf.q_B; dims=2)) .* GPU_T(1.0 / R)

    # ── 10. max_neg_ld ───────────────────────────────────────────────────
    gbuf.neg_log_denom .= .-gbuf.log_denom
    launch_groupmax!(gbuf.max_neg_ld,
                     gbuf.neg_log_denom, gbuf.sort_pt_gpu, gbuf.pt_uval_gpu,
                     gbuf.pt_grp_start_gpu,
                     gbuf.n_pt_groups, n_pairs, R)

    # ── 11. shifted_inv ──────────────────────────────────────────────────
    gbuf.shifted_inv .= exp.(gbuf.neg_log_denom .-
                             @view(gbuf.max_neg_ld[gbuf.pair_time_enc_gpu, :]))

    # ── 12. sum_wtd = PT_agg × shifted_inv  (CUBLAS GEMM) ───────────────
    mul!(gbuf.sum_wtd, gbuf.PT_agg_gpu, gbuf.shifted_inv)

    # ── 13. log_inv_wtd ──────────────────────────────────────────────────
    gbuf.log_inv_wtd .= gbuf.max_neg_ld .+
                        log.(max.(gbuf.sum_wtd, GPU_T(1e-30)))

    # ── 14. log_s_D_r ────────────────────────────────────────────────────
    gbuf.log_s_D_r .= gbuf.V_D .+ @view(gbuf.log_inv_wtd[gbuf.d_time_enc_gpu, :])

    # ── 15. s_D via row-max trick ────────────────────────────────────────
    gbuf.row_max_D .= vec(maximum(gbuf.log_s_D_r; dims=2))
    gbuf.s_D_gpu   .= exp.(gbuf.row_max_D) .*
                      (vec(sum(exp.(gbuf.log_s_D_r .- gbuf.row_max_D); dims=2)) .*
                       GPU_T(1.0 / R))

    # ── 16. Copy results back to CPU Float64 ─────────────────────────────
    CUDA.synchronize()
    buf.s_B .= Float64.(Array(gbuf.s_B_gpu))
    buf.s_D .= Float64.(Array(gbuf.s_D_gpu))
    return nothing
end

# ==========================================================================
# GPU↔CPU Consistency Guard
# ==========================================================================

"""
    verify_gpu_shares(buf, gbuf, delta, pc, R; rtol=1e-9) → Float64

The callers pass `rtol = tol_inner`: the SQUAREM residual floor equals the
share relative error, so shares must agree to `tol_inner` for the inner loop to
reach it. (The original `1f0/R` bug gave a ~3.6e-9 share error — above a
`tol_inner` of 1e-10, hence non-convergence — which this guard now flags.)

Compute model shares at `delta` on BOTH the CPU reference
(`compute_model_shares!`) and the GPU (`compute_model_shares_gpu!`) and compare.
Returns the max relative difference; **errors** if it exceeds `rtol`.

This is a cheap startup guard against silent GPU precision regressions — e.g. a
Float32 `1/R` scaling factor — that don't crash but bias every share by a constant
factor, pinning the SQUAREM residual above `tol_inner` so the inner loop spins to
`max_iter` every time. Such a bug shows up here as a clear failure instead of a
day-long job that quietly never converges.

`buf.mu` must already be populated (call `compute_mu!` at the same θ₂ first).
Both share routines read `buf.mu` and write `buf.s_B`/`buf.s_D`, so the GPU call
leaves `buf` holding the GPU shares on return.
"""
function verify_gpu_shares(buf::HotBuffers, gbuf::GpuBuffers,
                            delta::Vector{Float64}, pc::Precomp, R::Int;
                            rtol::Float64=1e-9)
    # CPU reference shares
    compute_model_shares!(buf, delta, pc, R)
    cpu_sB = copy(buf.s_B); cpu_sD = copy(buf.s_D)

    # GPU shares (overwrites buf.s_B / buf.s_D)
    compute_model_shares_gpu!(buf, gbuf, delta, pc, R)

    maxrel(c::Vector{Float64}, g::Vector{Float64}) = begin
        m = 0.0
        @inbounds for i in eachindex(c)
            d = abs(g[i] - c[i]) / max(abs(c[i]), 1e-12)
            d > m && (m = d)
        end
        m
    end
    eB = maxrel(cpu_sB, buf.s_B)
    eD = maxrel(cpu_sD, buf.s_D)
    emax = max(eB, eD)

    # Floored log-share diff — the *actual* SQUAREM residual ingredient
    # (T(δ) = δ + ln_s_data − log(max(s, 1e-15))).  Raw-share agreement can hide a
    # mismatch that only appears after the log-floor: a tiny-share obs floored at a
    # slightly-off constant (e.g. Float32 `1f-15` = 1.0000000363e-15) leaves the raw
    # shares equal yet gaps log(s) by ~3.6e-9, pinning the contraction above tol_inner.
    # Floor at the SAME exact 1e-15 the CPU/GPU contraction uses.
    maxabs_logfloor(c::Vector{Float64}, g::Vector{Float64}) = begin
        m = 0.0
        @inbounds for i in eachindex(c)
            d = abs(log(max(g[i], 1e-15)) - log(max(c[i], 1e-15)))
            d > m && (m = d)
        end
        m
    end
    lB = maxabs_logfloor(cpu_sB, buf.s_B)
    lD = maxabs_logfloor(cpu_sD, buf.s_D)
    lmax = max(lB, lD)

    log_status("  [GPU CHECK] max rel share diff CPU vs GPU: " *
               "B=$(round(eB, sigdigits=3)) | D=$(round(eD, sigdigits=3))")
    log_status("  [GPU CHECK] max |Δ log(max(s,1e-15))| (residual ingredient): " *
               "B=$(round(lB, sigdigits=3)) | D=$(round(lD, sigdigits=3))")
    if emax > rtol || lmax > rtol
        error("[GPU CHECK FAILED] GPU vs CPU mismatch: raw-share rel=$(round(emax, sigdigits=4)), " *
              "log-floor abs=$(round(lmax, sigdigits=4)) (rtol=$rtol). Likely a GPU precision " *
              "regression — a Float32 scaling factor (`1f0/R`) or log-floor literal (`1f-15`). " *
              "The inner loop will not converge to tol_inner — aborting before wasting the job.")
    end
    log_status("  [GPU CHECK] ✓ GPU matches CPU within rtol=$rtol (shares and log-floor)")
    return max(emax, lmax)
end

# ==========================================================================
# GPU Device-Resident Share Computation (for on-device SQUAREM)
# ==========================================================================

"""
    model_shares_dev!(gbuf, delta_gpu, mu_B, mu_D, s_model_full, R)

Device-resident sibling of `compute_model_shares_gpu!`.  Differences:

* `delta_gpu` is a **CuVector already on device** — no host→device copy per call.
* `gbuf.mu_gpu` is assumed already populated (uploaded **once** by the caller
  before the inner loop) — no per-step mu copy.
* Model shares are scattered into the device vector `s_model_full` (length N)
  — **no device→host copy**.

This keeps the entire SQUAREM contraction on the GPU so the device stays busy
across all inner iterations (raises GPU utilisation; eliminates per-step PCIe
transfers).  `compute_model_shares_gpu!` is retained unchanged for the IFT
forward passes, which need `buf.s_B`/`buf.s_D` back on the CPU.
"""
function model_shares_dev!(gbuf::GpuBuffers, delta_gpu::CuVector{GPU_T},
                            mu_B::CuMatrix{GPU_T}, mu_D::CuMatrix{GPU_T},
                            s_model_full::CuVector{GPU_T}, R::Int)
    N_B = gbuf.N_B; N_D = gbuf.N_D
    n_pairs = gbuf.n_pairs

    # Gather δ → delta_B / delta_D (delta already on device).
    # @view fuses the indexed gather into the assignment — no materialised temp.
    gbuf.delta_B .= @view delta_gpu[gbuf.b_mask_idx_gpu]
    gbuf.delta_D .= @view delta_gpu[gbuf.d_mask_idx_gpu]

    # V_B / V_D (mu_B / mu_D pre-gathered once per outer iter — μ is constant
    # during the inner loop, so we avoid re-gathering the N×R μ matrix per step)
    gbuf.V_B .= clamp.(gbuf.delta_B .+ mu_B, GPU_T(-500f0), GPU_T(500f0))
    gbuf.V_D .= clamp.(gbuf.delta_D .+ mu_D, GPU_T(-500f0), GPU_T(500f0))

    launch_logsumexp!(gbuf.log_sum_D_time, gbuf.V_D, gbuf.sort_d_gpu,
                      gbuf.d_uval_gpu, gbuf.d_grp_start_gpu, gbuf.n_d_groups, N_D, R)
    gbuf.log_D_sum_pair .= @view gbuf.log_sum_D_time[gbuf.pair_time_enc_gpu, :]
    launch_logsumexp!(gbuf.log_sum_B_mkt, gbuf.V_B, gbuf.sort_b_gpu,
                      gbuf.b_uval_gpu, gbuf.b_grp_start_gpu, gbuf.n_b_groups, N_B, R)
    let lsB = gbuf.log_sum_B_mkt, lsD = gbuf.log_D_sum_pair
        jm = max.(GPU_T(0f0), lsB, lsD)
        gbuf.log_denom .= jm .+ log.(max.(
            exp.(GPU_T(0f0) .- jm) .+ exp.(lsB .- jm) .+ exp.(lsD .- jm),
            GPU_T(1e-30)))
    end
    gbuf.q_B     .= exp.(gbuf.V_B .- @view(gbuf.log_denom[gbuf.b_mkt_idx_gpu, :]))
    gbuf.s_B_gpu .= vec(sum(gbuf.q_B; dims=2)) .* GPU_T(1.0 / R)

    gbuf.neg_log_denom .= .-gbuf.log_denom
    launch_groupmax!(gbuf.max_neg_ld, gbuf.neg_log_denom, gbuf.sort_pt_gpu,
                     gbuf.pt_uval_gpu, gbuf.pt_grp_start_gpu, gbuf.n_pt_groups, n_pairs, R)
    gbuf.shifted_inv .= exp.(gbuf.neg_log_denom .-
                             @view(gbuf.max_neg_ld[gbuf.pair_time_enc_gpu, :]))
    mul!(gbuf.sum_wtd, gbuf.PT_agg_gpu, gbuf.shifted_inv)
    gbuf.log_inv_wtd .= gbuf.max_neg_ld .+ log.(max.(gbuf.sum_wtd, GPU_T(1e-30)))
    gbuf.log_s_D_r .= gbuf.V_D .+ @view(gbuf.log_inv_wtd[gbuf.d_time_enc_gpu, :])
    gbuf.row_max_D .= vec(maximum(gbuf.log_s_D_r; dims=2))
    gbuf.s_D_gpu   .= exp.(gbuf.row_max_D) .*
                      (vec(sum(exp.(gbuf.log_s_D_r .- gbuf.row_max_D); dims=2)) .*
                       GPU_T(1.0 / R))

    # Scatter compact shares → full-N model-share vector (stays on device)
    s_model_full[gbuf.b_mask_idx_gpu] .= gbuf.s_B_gpu
    s_model_full[gbuf.d_mask_idx_gpu] .= gbuf.s_D_gpu
    return nothing
end

# ==========================================================================
# GPU BLP Contraction (SQUAREM) — fully on-device
# ==========================================================================

"""
    blp_contraction_gpu!(buf, gbuf, delta, pc, R; tol, max_iter)

SQUAREM BLP fixed-point contraction running **entirely on the GPU**.

`delta` (CPU `Vector{Float64}`) is uploaded **once** at entry and downloaded
**once** at convergence.  `buf.mu` is uploaded to `gbuf.mu_gpu` once.  Every
SQUAREM step — the contraction map `T`, the sup-norm/`‖·‖²` reductions, and the
acceleration step — executes as device broadcasts/reductions, so the H200 stays
busy throughout (no per-step PCIe transfers; only a handful of scalar reductions
sync to the host per step).

Algebraically identical to the CPU `blp_contraction!`:
  T(δ) = clamp(δ + ln s_data − ln s_model(δ), −500, 500)
"""
function blp_contraction_gpu!(buf::HotBuffers, gbuf::GpuBuffers,
                               delta::Vector{Float64},
                               pc::Precomp, R::Int;
                               tol::Float64=1e-12, max_iter::Int=5000)

    N            = length(delta)
    norm_history = Float64[]

    # ── Upload loop-constant data ONCE ────────────────────────────────────
    # mu is fixed during the inner loop (depends only on θ₂); upload once.
    copyto!(gbuf.mu_gpu, GPU_T.(buf.mu))

    # Pre-gather μ for B / D observations once (constant in the loop).
    mu_B = gbuf.mu_gpu[gbuf.b_mask_idx_gpu, :]
    mu_D = gbuf.mu_gpu[gbuf.d_mask_idx_gpu, :]

    # Full-N data log-share vector: ln_s_data_full[i] = ln_s_data_B_cond[i] for
    # B obs, ln_s_data_D[i] for D obs.  Constant across the loop.
    ln_s_data_cpu = Vector{GPU_T}(undef, N)
    @inbounds for i in 1:N
        ln_s_data_cpu[i] = pc.b_mask[i] ? GPU_T(pc.ln_s_data_B_cond[i]) :
                                          GPU_T(pc.ln_s_data_D[i])
    end
    ln_s_data = CuVector{GPU_T}(ln_s_data_cpu)

    # ── Device-resident SQUAREM work vectors (~7 × N × 8 B ≈ 17 MB) ───────
    d_cur  = CuVector{GPU_T}(GPU_T.(delta))
    x1     = CUDA.zeros(GPU_T, N)
    x2     = CUDA.zeros(GPU_T, N)
    r_vec  = CUDA.zeros(GPU_T, N)
    v_vec  = CUDA.zeros(GPU_T, N)
    x_prop = CUDA.zeros(GPU_T, N)
    s_full = CUDA.zeros(GPU_T, N)

    lo = GPU_T(-500f0); hi = GPU_T(500f0)
    floor_s = GPU_T(1e-15)   # Float64 literal — MUST match the CPU `clamp(s, 1e-15, Inf)`
                             # exactly. A Float32 `1f-15` here is 1.0000000363e-15, which
                             # gaps log(s) by ~3.627e-9 at floored shares and pins the
                             # SQUAREM residual above tol_inner (silent non-convergence).

    # Contraction map T(src) → dest, both device vectors
    Tstep!(dest::CuVector{GPU_T}, src::CuVector{GPU_T}) = begin
        model_shares_dev!(gbuf, src, mu_B, mu_D, s_full, R)
        dest .= clamp.(src .+ ln_s_data .- log.(max.(s_full, floor_s)), lo, hi)
        return nothing
    end

    # Finiteness guards are device reductions that sync a scalar to host. The
    # convergence norms (nm, nm2) and SQUAREM norms (norm_r_sq, norm_v_sq,
    # norm_prop) are needed EVERY step, but the isfinite guards on x1/x2 are not:
    # a NaN/Inf makes the norm non-finite (so `< tol` is false → no false
    # convergence), and the periodic check below aborts within FINITE_CHECK_EVERY
    # steps. The x_prop safeguard is folded into the norm_prop test (a non-finite
    # x_prop ⇒ non-finite norm_prop ⇒ fall back to the Picard step x2), removing a
    # separate sync with identical behaviour.
    FINITE_CHECK_EVERY = 10
    # SQUAREM occasionally limit-cycles: the residual norm flat-lines well above tol
    # (observed ~0.15 / ~0.007 for thousands of iters on hard trial θ₂). Detect a stall
    # (best residual not improving for STALL_PATIENCE SQUAREM steps) and escape with a
    # burst of plain Picard steps (monotone for a contraction), then resume SQUAREM. This
    # only fires on a stall, so well-behaved contractions are unaffected and the fixed
    # point is unchanged — it just stops the loop burning to max_iter at a plateau.
    best_nm        = Inf
    stall          = 0
    STALL_PATIENCE = 30      # SQUAREM steps without ≥0.1% best-residual improvement
    PICARD_BURST   = 20      # plain fixed-point steps to break the cycle
    fevals = 0
    while fevals + 2 <= max_iter
        Tstep!(x1, d_cur); fevals += 1
        if fevals % FINITE_CHECK_EVERY == 0 && !all(isfinite, x1)
            println("    [SQUAREM-GPU ABORT] non-finite at eval=$fevals")
            copyto!(delta, Float64.(Array(d_cur)))
            return false, fevals, norm_history
        end
        r_vec .= x1 .- d_cur
        nm = Float64(maximum(abs, r_vec))
        push!(norm_history, nm)
        (fevals % 50 == 0 || nm < tol) &&
            println("    [SQUAREM-GPU eval=$fevals/$max_iter] norm=$(round(nm, sigdigits=4))")
        if nm < tol
            copyto!(delta, Float64.(Array(x1))); return true, fevals, norm_history
        end

        Tstep!(x2, x1); fevals += 1
        if fevals % FINITE_CHECK_EVERY == 0 && !all(isfinite, x2)
            copyto!(delta, Float64.(Array(x1))); return false, fevals, norm_history
        end
        nm2 = Float64(maximum(abs, x2 .- x1))
        push!(norm_history, nm2)
        (fevals % 50 == 0 || nm2 < tol) &&
            println("    [SQUAREM-GPU eval=$fevals/$max_iter] norm=$(round(nm2, sigdigits=4))")
        if nm2 < tol
            copyto!(delta, Float64.(Array(x2))); return true, fevals, norm_history
        end

        # ── Stall detection + Picard-burst restart (escapes SQUAREM limit cycles) ──
        cur_best = min(nm, nm2)
        if cur_best < best_nm * (1.0 - 1e-3)
            best_nm = cur_best; stall = 0
        else
            stall += 1
        end
        if stall >= STALL_PATIENCE
            d_cur .= x2                       # restart the burst from the Picard iterate
            burst = 0
            while burst < PICARD_BURST && fevals + 1 <= max_iter
                Tstep!(x1, d_cur); fevals += 1; burst += 1
                nmb = Float64(maximum(abs, x1 .- d_cur)); push!(norm_history, nmb)
                d_cur .= x1
                (fevals % 50 == 0) &&
                    println("    [SQUAREM-GPU eval=$fevals/$max_iter] " *
                            "norm=$(round(nmb, sigdigits=4)) [picard-restart]")
                if nmb < tol
                    copyto!(delta, Float64.(Array(d_cur))); return true, fevals, norm_history
                end
            end
            best_nm = Inf; stall = 0
            continue
        end

        v_vec .= (x2 .- x1) .- r_vec
        norm_r_sq = Float64(sum(abs2, r_vec))
        norm_v_sq = Float64(sum(abs2, v_vec))
        if norm_v_sq < 1e-28
            d_cur .= x2; continue
        end
        α = GPU_T(-sqrt(norm_r_sq / norm_v_sq))
        x_prop .= clamp.(d_cur .- GPU_T(2f0) * α .* r_vec .+ α^2 .* v_vec, lo, hi)
        # Non-finite x_prop ⇒ norm_prop non-finite ⇒ fall back to Picard step x2
        # (folds the old `all(isfinite, x_prop)` sync into this existing reduction).
        norm_prop = Float64(maximum(abs, x_prop .- x2))
        if !isfinite(norm_prop) || norm_prop > 100.0 * nm2 + 1.0
            d_cur .= x2
        else
            d_cur .= x_prop
        end
    end
    copyto!(delta, Float64.(Array(d_cur)))
    return false, fevals, norm_history
end

# ==========================================================================
# GPU GMM Objective
# ==========================================================================

"""
    gmm_objective_gpu!(buf, gbuf, theta2, prod_vec, nu_draws,
                       sigma_indices, pi_interactions, R, coef_dim,
                       W, tol_inner, max_inner, delta, pc) → Float64

Drop-in GPU replacement for `gmm_objective!`.  `compute_mu!` runs on CPU
(MKL GEMM, ~0 ms for small k-dim), then `blp_contraction_gpu!` handles
the inner fixed-point entirely on device.
"""
function gmm_objective_gpu!(buf::HotBuffers, gbuf::GpuBuffers,
                             theta2::Vector{Float64},
                             prod_vec, nu_draws,
                             sigma_indices, pi_interactions,
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
    G          = compute_gmm_moments(xi, pc)
    return dot(G, W * G)
end

"""
    gmm_objective_cue_gpu!(...same as gmm_objective_gpu! + opts::CueOpts[, alpha0])

CUE twin of `gmm_objective_gpu!` (BLP_ENGINE=cue): identical GPU contraction, then the clustered
continuously-updated fixed point (`cue_fixed_point`, blp_1_estimation.jl §7b) instead of the fixed-W
one-step. W0 is retained only for the one-time self-check log. With `alpha0 !== nothing` it runs the
Stock–Wright variant (`cue_fixed_point_alpha0`): α pinned, β-only concentration → S(α₀).
Q is in Hansen-J units (N-scaled) — NOT comparable to the ift/numerical Q.
"""
function gmm_objective_cue_gpu!(buf::HotBuffers, gbuf::GpuBuffers,
                                 theta2::Vector{Float64},
                                 prod_vec, nu_draws,
                                 sigma_indices, pi_interactions,
                                 R::Int, coef_dim::Int,
                                 W0::Matrix{Float64},
                                 tol_inner::Float64, max_inner::Int,
                                 delta::Vector{Float64},
                                 pc::Precomp, opts::CueOpts;
                                 alpha0::Union{Nothing,Float64}=nothing)::Float64
    sigma_vals, pi_vals = unpack_theta2(theta2, sigma_indices, pi_interactions)
    compute_mu!(buf, prod_vec, nu_draws, sigma_vals, sigma_indices, pi_vals, R, coef_dim)
    converged, n_iter, _ = blp_contraction_gpu!(buf, gbuf, delta, pc, R;
                                                 tol=tol_inner, max_iter=max_inner)
    converged || println("  [!] Inner loop (GPU) did not converge in $n_iter iterations")
    local Q, iters, wconv
    if alpha0 === nothing
        _, _, _, Q, iters, wconv = cue_fixed_point(delta, pc, opts)
    else
        _, _, _, Q, iters, wconv = cue_fixed_point_alpha0(delta, pc, opts, alpha0)
    end
    opts.n_evals[] += 1
    wconv || (opts.n_nonconv[] += 1)
    iters > opts.itmax_seen[] && (opts.itmax_seen[] = iters)
    if opts.n_evals[] == 1   # one-time self-check: production one-step Q vs CUE Q at the same δ
        g0 = compute_gmm_moments(estimate_theta1(delta, pc)[2], pc)
        log_status("  [CUE self-check] Q_onestep(fixed W)=$(round(dot(g0, W0 * g0), sigdigits=6)) | " *
                   "Q_cue=$(round(Q, sigdigits=6)) (N-scaled, NOT comparable) | witer=$iters conv=$wconv")
    end
    return Q
end

# ==========================================================================
# GPU Estimation Runner
# ==========================================================================

"""
    run_blp_estimation_gpu(estim, spec_id, args, nu_draws, draws_3d, key_index)

GPU-accelerated version of `run_blp_estimation`.

For the **logit** stage there is no inner contraction loop, so this
function delegates directly to the CPU `run_blp_estimation`.

For **sigma / full / extended** stages, all inner-loop share computations
run on the GPU via `compute_model_shares_gpu!`; MKL handles `compute_mu!`
on CPU (small k-dim matrix multiply); SQUAREM delta updates and outer
L-BFGS-B optimisation run on CPU as before.
"""
function run_blp_estimation_gpu(estim::Int, spec_id::Int, args,
                                 nu_draws::Matrix{Float64},
                                 draws_3d::Array{Float64,3},
                                 key_index::Dict{Tuple{String,String},Int})

    println("\n" * "=" ^ 60)
    println("  Estimation $estim — Specification $spec_id  [GPU]")
    println("=" ^ 60)

    # CUE (BLP_ENGINE=cue): clustered continuously-updated weight matrix — see §7b in
    # blp_1_estimation.jl. Lives ONLY on this GPU-numerical path (FD gradients stay consistent
    # automatically when the objective changes; the IFT analytic gradient assumes ∂W/∂θ₂ = 0).
    use_cue = lowercase(get(ENV, "BLP_ENGINE", "ift")) == "cue"

    # Logit: no inner contraction → CPU path is fine
    if args["stage"] == "logit"
        use_cue && error("CUE is RC-stages/GPU-numerical only: the logit stage delegates to the CPU " *
                         "path, which ignores BLP_OUTPUT_SUFFIX and would overwrite the production " *
                         "(unsuffixed) files. Run the logit under BLP_ENGINE=ift.")
        return run_blp_estimation(estim, spec_id, args, nu_draws, draws_3d, key_index)
    end

    path = demand_parquet_path(estim, spec_id, args)
    path === nothing && return nothing

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

    # ── Map observations to pre-computed draw indices ─────────────────────
    N_obs     = nrow(df)
    mca_codes = string.(df.mca_code)
    time_ids  = string.(df.time_id)
    pad_idx   = size(draws_3d, 1)

    obs_key_idx = [get(key_index, (mca_codes[i], time_ids[i]), pad_idx)
                   for i in 1:N_obs]
    n_mapped = count(i -> i != pad_idx, obs_key_idx)
    log_status("  Obs mapped to draws: $n_mapped / $N_obs ($(round(100*n_mapped/N_obs, digits=1))%)")

    # ── Build product characteristics and IV matrices ─────────────────────
    prod_vec  = zeros(N_obs, coef_dim)
    spreads   = coalesce.(df.spread_ann, 0.0) ./ 100.0  # bps → percentage points (÷100)
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

    # ── CPU hot buffers + GPU mirror ──────────────────────────────────────
    buf  = allocate_hot_buffers(N_obs, N_B, N_D, R, n_pairs, n_times, coef_dim,
                                 length(pi_interactions))
    precompute_pi_products!(buf, prod_vec, draws_3d, obs_key_idx,
                             pi_interactions, coef_dim)
    cue_opts = use_cue ? CueOpts(pc.clusters) : nothing
    use_cue && log_status("  [CUE] clustered CU-GMM objective: G=$(cue_opts.G) clusters | " *
                          "max_witer=$(cue_opts.max_witer) wtol=$(cue_opts.wtol) " *
                          "pinv_rtol=$(cue_opts.pinv_rtol) ridge=$(cue_opts.ridge)")
    log_status("  [GPU] Device: $(CUDA.name(CUDA.device()))")
    log_status("  [GPU] Allocating GPU buffers (~$(round(21.68, digits=1)) GB)...")
    flush(stdout); flush(stderr)
    gbuf = allocate_gpu_buffers(buf, pc, N_obs, N_B, N_D, R, n_pairs, n_times)
    log_status("  [GPU] ✓ GPU buffers allocated")

    # ── θ₂ warm-start ─────────────────────────────────────────────────────
    # in_dir = data/input on HPC (uploaded logit_delta warm-starts); out_dir = results/checkpoints.
    in_dir, _, out_dir = get_paths(args["hpc"])
    prev_stages  = Dict("rc2" => "sigma", "rc3" => "rc2", "rc4" => "rc3",
                        "full" => "rc4", "ext1" => "full", "ext2" => "ext1",
                        "extended" => "ext2")
    theta2_0     = nothing
    # Explicit cross-engine seed: BLP_THETA2_INIT_FILE points at a checkpoint (e.g. the
    # IFT `extended` result) so a single-stage numerical cross-check starts from the IFT
    # optimum instead of its own (skipped) previous stage. Takes priority over prev-stage.
    let _init = get(ENV, "BLP_THETA2_INIT_FILE", "")
        if !isempty(_init) && isfile(_init)
            try
                _ck = deserialize(_init)
                _t2 = get(_ck, "theta2_star", nothing)
                if _t2 !== nothing && length(_t2) <= n_params
                    theta2_0 = zeros(n_params); theta2_0[1:length(_t2)] .= _t2
                    println("  [WARM-START] theta2_0 from BLP_THETA2_INIT_FILE=$(basename(_init))")
                end
            catch e
                println("  [WARM-START] Could not load BLP_THETA2_INIT_FILE: $e")
            end
        elseif !isempty(_init)
            # A set-but-missing seed silently degraded to a cold start before — say so loudly.
            println("  [WARM-START] BLP_THETA2_INIT_FILE set but MISSING: $_init")
        end
    end
    if theta2_0 === nothing && args["stage"] in keys(prev_stages)
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

    lo = fill(-5.0, n_params)
    hi = fill( 5.0, n_params)
    # σ parameters are standard deviations → bound ≥ 0 (same fix as the IFT engine):
    # removes the ±σ sign degeneracy that makes the outer optimiser oscillate without
    # converging. σ are the first length(sigma_indices) entries of θ₂.
    n_sigma = length(sigma_indices)
    lo[1:n_sigma] .= 0.0
    # σ upper bound (BLP_SIGMA_UB) and π box half-width (BLP_PI_BOUND), both default 2.0,
    # shared with the IFT engine so the cross-check optimises over the SAME box (the driver
    # widens the π bound to 5.0 for E3). This also overrides the wide ±5 default on the σ/π
    # slices, unifying the two engines' boxes.
    sigma_ub = parse(Float64, get(ENV, "BLP_SIGMA_UB", "2.0"))
    hi[1:n_sigma] .= sigma_ub
    pi_bound = parse(Float64, get(ENV, "BLP_PI_BOUND", "2.0"))
    if n_sigma < n_params
        lo[(n_sigma + 1):n_params] .= -pi_bound
        hi[(n_sigma + 1):n_params] .=  pi_bound
    end
    theta2_0 .= clamp.(theta2_0, lo, hi)   # a warm-start θ₂ may carry an out-of-box value

    println("  Outer minimisation: $(args["method"])")
    println("  Inner tolerance: $(args["tol_inner"])")

    # ── δ warm-start ──────────────────────────────────────────────────────
    delta_work = zeros(N_obs)
    let _loaded = false
        # BLP_DELTA_SUFFIX lets a run warm-start from a suffixed delta; default "" reads
        # the logit_delta_E{k}_spec_{s}.* the logit step writes. in_dir (data/input on
        # HPC) is checked first so the uploaded warm-start wins over any legacy data/output copy.
        _suffix = get(ENV, "BLP_DELTA_SUFFIX", "")
        for _bin_cand in unique([
                joinpath(in_dir,  "logit_delta_E$(estim)_spec_$(spec_id)$(_suffix).bin"),
                joinpath(in_dir,  "logit_delta_E$(estim)_spec_$(spec_id).bin"),
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
                joinpath(in_dir,  "logit_delta_E$(estim)_spec_$(spec_id)$(_suffix).jls"),
                joinpath(in_dir,  "logit_delta_E$(estim)_spec_$(spec_id).jls"),
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

    # ── Dry-run timing (GPU) ──────────────────────────────────────────────
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
        return Dict("dry_run" => true)
    end

    # ── Outer optimisation (true L-BFGS-B + cached forward-difference gradient) ──
    # The numerical engine has no analytical gradient, so L-BFGS-B is fed a forward-
    # difference ∇Q: n_params extra share solves per gradient — the same cost as the
    # finite-difference gradient Optim used before. A one-point cache stops the base
    # objective from being re-solved between the f and g! calls at the same θ₂.
    outer_iter = Ref(0)
    # CUE swaps ONLY the objective; grad_fn! below is a forward difference of _raw_obj, so the
    # gradient follows automatically. `alpha0_ref` is nothing except in Stock-Wright grid mode.
    alpha0_ref = Ref{Union{Nothing,Float64}}(nothing)
    _raw_obj(t2) = use_cue ?
        gmm_objective_cue_gpu!(buf, gbuf, t2, prod_vec, nu_draws,
                                sigma_indices, pi_interactions, R, coef_dim,
                                W, args["tol_inner"], args["max_inner"],
                                delta_work, pc, cue_opts; alpha0=alpha0_ref[]) :
        gmm_objective_gpu!(buf, gbuf, t2, prod_vec, nu_draws,
                            sigma_indices, pi_interactions, R, coef_dim,
                            W, args["tol_inner"], args["max_inner"],
                            delta_work, pc)
    fd_t2 = fill(NaN, n_params); fd_Q = Ref(0.0)
    function obj_fn(t2)
        if !isequal(t2, fd_t2)
            fd_Q[] = _raw_obj(t2); copyto!(fd_t2, t2)
            outer_iter[] += 1
            outer_iter[] % 10 == 0 &&
                log_status("  [OUTER iter=$(outer_iter[])] Q=$(round(fd_Q[], sigdigits=6)) " *
                           "theta2=$(round.(t2, digits=4))")
        end
        return fd_Q[]
    end
    function grad_fn!(G, t2)
        f0 = obj_fn(t2)                          # base value (cached → 1 solve)
        @inbounds for k in 1:n_params
            h    = 1e-6 * max(1.0, abs(t2[k]))   # relative forward-difference step
            tp   = copy(t2); tp[k] += h
            G[k] = (_raw_obj(tp) - f0) / h
        end
        return G
    end

    # ── Stock–Wright S-set mode (BLP_ALPHA_GRID, cue only) ────────────────
    # S(α₀) = min_{θ₂,β} N·ḡ'W(θ)ḡ with α pinned at α₀. Inverting {α₀ : S(α₀) ≤ crit} gives an
    # identification-robust set for α that ACCOUNTS for θ₂ being estimated — unlike the conditional
    # AR/LM sets computed at a fixed δ(θ̂₂). One artifact per (routine, stage); returns early.
    let _grid_spec = get(ENV, "BLP_ALPHA_GRID", "")
        if use_cue && !isempty(_grid_spec)
            parts = parse.(Float64, split(_grid_spec, ':'))
            length(parts) == 3 || error("BLP_ALPHA_GRID must be start:step:stop, got '$_grid_spec'")
            grid = collect(parts[1]:parts[2]:parts[3])
            log_status("  [S-SET] Stock-Wright grid: $(length(grid)) points " *
                       "$(parts[1]):$(parts[2]):$(parts[3]) — each is a constrained solve")
            pts = Vector{Dict{String,Any}}()
            t2_seed = copy(theta2_0)     # warm-start each point from the previous optimum
            for (gi, a0) in enumerate(grid)
                alpha0_ref[] = a0
                fill!(fd_t2, NaN)        # invalidate the one-point objective cache
                outer_iter[] = 0
                t2s, Sv, n_o, cvg = solve_box_lbfgsb(obj_fn, grad_fn!, t2_seed, lo, hi;
                                                     tol_outer = args["tol_outer"], maxiter = 500)
                # β/ξ at the constrained optimum (re-run the contraction at t2s first)
                sv_g, pv_g = unpack_theta2(t2s, sigma_indices, pi_interactions)
                compute_mu!(buf, prod_vec, nu_draws, sv_g, sigma_indices, pv_g, R, coef_dim)
                d_g = copy(delta_work)
                blp_contraction_gpu!(buf, gbuf, d_g, pc, R;
                                     tol=args["tol_inner"], max_iter=args["max_inner"])
                beta_g, _, _, S_g, it_g, wc_g = cue_fixed_point_alpha0(d_g, pc, cue_opts, a0)
                push!(pts, Dict{String,Any}("alpha0" => a0, "S" => S_g, "theta2" => t2s,
                                            "beta" => beta_g, "n_outer" => n_o,
                                            "converged" => cvg, "cue_iters" => it_g,
                                            "cue_converged" => wc_g))
                log_status("  [S-SET $gi/$(length(grid))] alpha0=$(round(a0, digits=4)) " *
                           "S=$(round(S_g, sigdigits=6)) outer=$n_o conv=$cvg")
                t2_seed = copy(t2s)
            end
            alpha0_ref[] = nothing
            L_iv = size(pc.Z_moments, 2)
            out = Dict{String,Any}(
                "estim" => estim, "spec_id" => spec_id, "stage" => args["stage"],
                "engine" => "cue", "mode" => "stock_wright_sset",
                "grid_spec" => _grid_spec, "n_obs" => N_obs,
                "n_instruments" => L_iv, "n_theta2" => n_params,
                "n_beta" => size(pc.X_full, 2) - 1,
                "df_conservative" => L_iv,
                "df_concentrated" => L_iv - (size(pc.X_full, 2) - 1 + n_params),
                "S_scale" => "N*g'*pinv(Omega_cluster)*g at the constrained optimum; " *
                             "invert {alpha0 : S <= chi2 crit}. Criticals are asymptotic chi2 " *
                             "(WCB criticals for S are out of scope).",
                "points" => pts)
            spath = joinpath(out_dir,
                "blp_results_E$(estim)_spec_$(spec_id)_$(args["stage"])_sset_cue.json")
            open(spath, "w") do io; JSON3.pretty(io, out); end
            log_status("  [S-SET] wrote $(basename(spath))  ($(length(grid)) points)")
            return Dict{String,Any}("sset" => true, "path" => spath, "n_points" => length(grid))
        end
    end

    theta2_star, Q_min, n_outer, conv_outer = solve_box_lbfgsb(
        obj_fn, grad_fn!, theta2_0, lo, hi;
        tol_outer = args["tol_outer"], maxiter = 500)

    println("  Optimiser converged: $(conv_outer)")
    println("  Q(theta2*) = $(round(Q_min, sigdigits=6))")
    println("  theta2* = $theta2_star")

    # ── Final contraction at optimum (GPU) ───────────────────────────────
    sv, pv = unpack_theta2(theta2_star, sigma_indices, pi_interactions)
    compute_mu!(buf, prod_vec, nu_draws, sv, sigma_indices, pv, R, coef_dim)
    delta_final = copy(delta_work)
    conv, n_it, _ = blp_contraction_gpu!(buf, gbuf, delta_final, pc, R;
                                          tol=args["tol_inner"],
                                          max_iter=args["max_inner"])

    theta1_star, xi_star      = estimate_theta1(delta_final, pc)
    theta1_se, n_cl, G_s      = compute_cluster_se(theta1_star, delta_final, pc)

    # ── CUE: re-solve θ₁ at the optimum with the continuously-updated weight ──
    # The production 2SLS θ₁/SE computed just above are KEPT (as theta1_2sls/alpha_2sls) so a table
    # can print α_cue vs α_2sls without re-deriving anything; the standard keys then carry the CUE
    # values, i.e. theta1[1] in a _cue JSON IS α_cue.
    if use_cue
        t1c, xic, Wc, Qc, itc, convc = cue_fixed_point(delta_final, pc, cue_opts)
        results["theta1_2sls"]       = theta1_star
        results["theta1_2sls_se"]    = theta1_se
        results["alpha_2sls"]        = theta1_star[1]
        results["cue_inner_iters"]   = itc
        results["cue_converged"]     = convc
        results["cue_witer_evals"]   = cue_opts.n_evals[]
        results["cue_witer_nonconv"] = cue_opts.n_nonconv[]
        results["cue_witer_itmax"]   = cue_opts.itmax_seen[]
        results["hansen_J"]          = Qc
        results["hansen_J_df"]       = size(pc.Z_moments, 2) - (size(pc.X_full, 2) + n_params)
        let ev = eigvals(Symmetric(cue_weight(xic, pc.Z_moments, cue_opts)))
            results["cue_omega_eig_min"] = minimum(ev)
            results["cue_omega_eig_max"] = maximum(ev)
        end
        results["Q_scale"] = "N*g'*pinv(Omega_cluster)*g (Hansen-J units; NOT comparable to " *
                             "the ift/numerical Q = g'(Z'Z/N)^-1 g)"
        results["se_method_theta1"] = "cue_linear_gmm_efficient(+CR1) at fixed theta2/W; " *
                                      "theta1_2sls_se is the production 2SLS cluster sandwich"
        println("  [CUE] alpha_cue=$(round(t1c[1], sigdigits=6)) vs " *
                "alpha_2sls=$(round(theta1_star[1], sigdigits=6)) | witer=$itc conv=$convc | " *
                "J=$(round(Qc, sigdigits=6)) | evals=$(cue_opts.n_evals[]) " *
                "nonconv=$(cue_opts.n_nonconv[]) itmax=$(cue_opts.itmax_seen[])")
        theta1_star = t1c
        xi_star     = xic
        theta1_se   = cue_linear_se(delta_final, pc, Wc, cue_opts.G)
    end

    results["theta1"]             = theta1_star
    results["theta1_se"]          = theta1_se
    results["theta2"]             = theta2_star
    results["theta2_se"]          = fill(NaN, length(theta2_star))
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
# CLI + Main (GPU)
# ==========================================================================

"""
    main_gpu_numerical()

Entry point for the GPU estimation script.  Parses the same CLI flags as
`main()` in blp_estimation.jl; processes specs **sequentially** (one GPU,
one spec at a time) so device memory is not over-subscribed.

Identical JSON / JLS output layout to the CPU path — downstream export
scripts work unchanged.
"""
function main_gpu_numerical()
    # ── GPU availability check ───────────────────────────────────────────
    log_status("[GPU] Checking CUDA functionality...")
    flush(stdout); flush(stderr)
    if !CUDA.functional()
        log_status("[GPU] ERROR: CUDA.functional() = false")
        flush(stdout); flush(stderr)
        error("[GPU] CUDA is not functional on this node.  " *
              "Use blp_estimation.jl for CPU-only estimation.")
    end
    dev = CUDA.device()
    log_status("[GPU] ✓ CUDA functional")
    log_status("[GPU] Using device: $(CUDA.name(dev)) " *
               "($(round(CUDA.totalmem(dev)/2^30, digits=1)) GB VRAM)")
    flush(stdout); flush(stderr)

    args     = parse_args_est()    # reuses CPU parser from blp_estimation.jl
    estim    = args["estim"]
    spec_ids = args["spec"] == "all" ? collect(1:12) : [parse(Int, args["spec"])]

    log_status("BLP Estimation GPU — START")
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
            log_status("[SKIP] Stage '$current_stage' already complete.")
            continue
        end

        println("\n  Starting stage '$current_stage' — $(length(spec_ids)) spec(s) " *
                "(sequential GPU execution)...")

        all_results = Dict{Int,Any}()

        # GPU: sequential spec loop — avoids multi-context VRAM contention.
        for sp in spec_ids
            out_path = joinpath(out_dir,
                "blp_results_E$(estim)_spec_$(sp)_$(current_stage)$(output_suffix()).jls")
            if isfile(out_path)
                log_status("  [SKIP] Spec $sp already done for '$current_stage'")
                continue
            end

            res = try
                run_blp_estimation_gpu(estim, sp, args, nu_draws, draws_3d, key_index)
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
                    "theta1_se"          => round.(get(res, "theta1_se", Float64[]), sigdigits=8),
                    "theta2"             => round.(get(res, "theta2",    Float64[]), sigdigits=8),
                    "theta2_se"          => round.(get(res, "theta2_se", Float64[]), sigdigits=8),
                    "Q_value"            => get(res, "Q_value",   0.0),
                    "converged"          => get(res, "converged", false),
                    "param_names_theta1" => get(res, "param_names_theta1", String[]),
                    "sigma_indices"      => get(res, "sigma_indices",     Int[]),
                    "pi_interactions"    => get(res, "pi_interactions",   []),
                    "n_clusters"         => get(res, "n_clusters",        0),
                    "G_star"             => get(res, "G_star",            0.0),
                )
                # CUE extras — present only under BLP_ENGINE=cue, so ift/num JSON is byte-identical.
                for k in ("theta1_2sls", "theta1_2sls_se", "alpha_2sls", "cue_inner_iters",
                          "cue_converged", "cue_witer_evals", "cue_witer_nonconv",
                          "cue_witer_itmax", "hansen_J", "hansen_J_df",
                          "cue_omega_eig_min", "cue_omega_eig_max", "Q_scale", "se_method_theta1")
                    haskey(res, k) && (json_out[k] = res[k])
                end
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
            haskey(res, "alpha_2sls")    && (all_results[sp]["alpha_2sls"]    = res["alpha_2sls"])
            haskey(res, "cue_converged") && (all_results[sp]["cue_converged"] = res["cue_converged"])
        end

        summary_path = joinpath(out_dir, "blp_summary_E$(estim)_$(current_stage)_gpu$(output_suffix()).json")
        open(summary_path, "w") do f; JSON3.write(f, all_results); end
        log_status("Summary saved to: $summary_path")
        log_status("[DONE] BLP GPU E$estim ($current_stage) complete for specs $spec_ids.")
    end
end

# Guard: run main_gpu_numerical only when this file is executed directly
# [merged: per-file autorun removed — see dispatch at end]


# ===================== blp_2_estimation_gpu.jl =====================
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

# ── CPU baseline (IFT) ───────────────────────────────────────────────────────
# The IFT gradient code (formerly blp_2_estimation.jl, which included blp_1_estimation.jl)
# is merged inline into this engine.

using CUDA
CUDA.allowscalar(false)

# ── GPU infrastructure ─────────────────────────────────────────────────────────
# GPU buffer definitions and kernels (formerly blp_1_estimation_gpu.jl) are merged inline;
# the X_COLS guard avoids constant-redefinition warnings.

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

"""Collect the raw diagonal-IFT Jacobian ∂δ*/∂θ₂ (N × dim θ₂) at the optimum, for standard errors.
Same per-column forward pass as `compute_ift_gradient_gpu!`, but returns the raw `d_k` columns
(BEFORE the θ₁ projection) — the θ₂ block of the moment Jacobian D=[−Z'X, Z'∂δ/∂θ₂]. Restores
buf.mu to base θ₂ on exit. Inherits the diagonal-IFT + forward-difference approximation."""
function compute_delta_jacobian_gpu(delta_star::Vector{Float64}, s_base::Vector{Float64},
                                    theta2::Vector{Float64}, buf::HotBuffers, gbuf::GpuBuffers,
                                    prod_vec::Matrix{Float64}, nu_draws::Matrix{Float64},
                                    sigma_indices::Vector{Int},
                                    pi_interactions::Vector{Tuple{Int,Int}},
                                    R::Int, coef_dim::Int, pc::Precomp; ε::Float64=1e-5)
    n_params = length(theta2)
    N        = length(delta_star)
    Ddelta   = Matrix{Float64}(undef, N, n_params)
    for k in 1:n_params
        theta2_k     = copy(theta2); theta2_k[k] += ε
        sv_k, pv_k   = unpack_theta2(theta2_k, sigma_indices, pi_interactions)
        compute_mu!(buf, prod_vec, nu_draws, sv_k, sigma_indices, pv_k, R, coef_dim)
        compute_model_shares_gpu!(buf, gbuf, delta_star, pc, R)
        s_k = collect_model_shares(buf, pc, N)
        @inbounds for i in 1:N
            one_minus_s   = max(1.0 - s_base[i], 1e-4)
            Ddelta[i, k]  = -(log(max(s_k[i], 1e-15)) - log(s_base[i])) / ε / one_minus_s
        end
    end
    sv_base, pv_base = unpack_theta2(theta2, sigma_indices, pi_interactions)
    compute_mu!(buf, prod_vec, nu_draws, sv_base, sigma_indices, pv_base, R, coef_dim)
    return Ddelta
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

    path = demand_parquet_path(estim, spec_id, args)
    path === nothing && return nothing

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
    # in_dir = data/input on HPC (uploaded logit_delta warm-starts); out_dir = results/checkpoints.
    in_dir, _, out_dir = get_paths(args["hpc"])
    prev_stages   = Dict("rc2" => "sigma", "rc3" => "rc2", "rc4" => "rc3",
                         "full" => "rc4", "ext1" => "full", "ext2" => "ext1",
                         "extended" => "ext2")
    theta2_0 = nothing
    # Explicit cross-engine seed: BLP_THETA2_INIT_FILE points at a checkpoint (e.g. the
    # IFT `extended` result) so a single-stage numerical cross-check starts from the IFT
    # optimum instead of its own (skipped) previous stage. Takes priority over prev-stage.
    let _init = get(ENV, "BLP_THETA2_INIT_FILE", "")
        if !isempty(_init) && isfile(_init)
            try
                _ck = deserialize(_init)
                _t2 = get(_ck, "theta2_star", nothing)
                if _t2 !== nothing && length(_t2) <= n_params
                    theta2_0 = zeros(n_params); theta2_0[1:length(_t2)] .= _t2
                    println("  [WARM-START] theta2_0 from BLP_THETA2_INIT_FILE=$(basename(_init))")
                end
            catch e
                println("  [WARM-START] Could not load BLP_THETA2_INIT_FILE: $e")
            end
        elseif !isempty(_init)
            # A set-but-missing seed silently degraded to a cold start before — say so loudly.
            println("  [WARM-START] BLP_THETA2_INIT_FILE set but MISSING: $_init")
        end
    end
    if theta2_0 === nothing && args["stage"] in keys(prev_stages)
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
    # σ parameters are standard deviations → bound ≥ 0. This removes the ±σ sign
    # degeneracy (σ and −σ give identical shares because ν is mean-zero symmetric),
    # which otherwise makes L-BFGS-B oscillate between +σ and −σ and never converge
    # (Q flat, all 500 outer iters wasted — observed on E4 sigma). σ are the first
    # length(sigma_indices) entries of θ₂; the π interactions keep the [-2,2] box.
    n_sigma = length(sigma_indices)
    lo[1:n_sigma] .= 0.0
    # σ upper bound (BLP_SIGMA_UB, default 2.0) and π box half-width (BLP_PI_BOUND, default
    # 2.0) are configurable. In E3 the demographic interaction π(FGC×Age65+) pins at the π
    # bound 2.0 (the earlier "σ₇" reading was a positional mislabel — it is a π), so the
    # driver widens BLP_PI_BOUND to 5.0 for E3. σ are the first n_sigma θ₂ entries; the π
    # interactions are the rest. Keep IFT and the numerical cross-check on the SAME box.
    sigma_ub = parse(Float64, get(ENV, "BLP_SIGMA_UB", "2.0"))
    hi[1:n_sigma] .= sigma_ub
    pi_bound = parse(Float64, get(ENV, "BLP_PI_BOUND", "2.0"))
    if n_sigma < n_params
        lo[(n_sigma + 1):n_params] .= -pi_bound
        hi[(n_sigma + 1):n_params] .=  pi_bound
    end
    theta2_0 .= clamp.(theta2_0, lo, hi)   # a warm-start θ₂ may carry an out-of-box value

    println("  Optimizer: L-BFGS-B + IFT analytical gradient (GPU)")
    println("  Inner tolerance: $(args["tol_inner"]) | bounds: [$(lo[1]), $(hi[1])]")

    # ── δ warm-start ──────────────────────────────────────────────────────
    delta_work = zeros(N_obs)
    let _loaded = false
        # BLP_DELTA_SUFFIX lets a run warm-start from a suffixed delta; default "" reads
        # the logit_delta_E{k}_spec_{s}.* that the logit step writes. A suffixed delta is
        # preferred when set, the unsuffixed one is the fallback. in_dir (data/input on
        # HPC) is checked first so the uploaded warm-start wins over any legacy data/output copy.
        _suffix = get(ENV, "BLP_DELTA_SUFFIX", "")
        for _bin_cand in unique([
                joinpath(in_dir,  "logit_delta_E$(estim)_spec_$(spec_id)$(_suffix).bin"),
                joinpath(in_dir,  "logit_delta_E$(estim)_spec_$(spec_id).bin"),
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
                joinpath(in_dir,  "logit_delta_E$(estim)_spec_$(spec_id)$(_suffix).jls"),
                joinpath(in_dir,  "logit_delta_E$(estim)_spec_$(spec_id).jls"),
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
    # ── SE-ONLY: skip the outer optimisation, recompute SEs from this stage's checkpoint ──────
    # The SE routine runs strictly AFTER optimisation, so a change confined to it (e.g. the
    # degenerate-direction profiling in se_common.jl) provably cannot move the point estimates.
    # We reload θ₂* (and δ*) and do ONE objective evaluation instead of a full L-BFGS-B solve —
    # a ~1.5 h/routine stage becomes a couple of minutes. δ* warm-starts the inner contraction at
    # its own fixed point, so that single evaluation converges immediately and leaves every GPU
    # buffer in exactly the state the SE block below expects.
    if get(args, "se_only", false)
        chk = joinpath(out_dir,
            "blp_checkpoint_E$(estim)_spec_$(spec_id)_$(args["stage"])$(output_suffix()).jls")
        isfile(chk) || error("--se-only: checkpoint not found: $chk (run the stage normally first)")
        ck = deserialize(chk)
        theta2_star = collect(Float64, ck["theta2_star"])
        length(theta2_star) == n_params || error("--se-only: checkpoint θ₂ has " *
            "$(length(theta2_star)) params but stage '$(args["stage"])' expects $n_params — the θ₂ " *
            "structure changed since that checkpoint; re-run this stage normally.")
        d0 = load_delta_bin(replace(chk, ".jls" => ".bin"))
        d0 === nothing && haskey(ck, "delta_star") && (d0 = collect(Float64, ck["delta_star"]))
        if d0 !== nothing && length(d0) == length(delta_work)
            copyto!(delta_work, d0)
        else
            println("  [SE-ONLY] no usable δ* — contracting from the existing warm start.")
        end
        println("  [SE-ONLY] θ₂* loaded from $(basename(chk)); outer optimisation SKIPPED.")
        ift_compute_gpu!(theta2_star)      # one evaluation: converges δ + fills the GPU buffers
        Q_min = cache_Q[]; n_outer = 0; conv_outer = true
    else
        theta2_star, Q_min, n_outer, conv_outer = solve_box_lbfgsb(
            f_obj, g_obj!, theta2_0, lo, hi;
            tol_outer = args["tol_outer"], maxiter = 500)
    end

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
    theta2_se   = fill(NaN, length(theta2_star))   # NaN sentinel = "not computed" (distinct from a genuine 0)
    theta1_pval = Float64[]; theta2_pval = Float64[]

    # ── θ₁+θ₂ SEs "in the same manner" via BLP_SE_METHOD ("wcb" default | "sandwich") ──────
    # Joint GMM Jacobian D=[−Z'X, Z'∂δ/∂θ₂]; ∂δ/∂θ₂ from the diagonal IFT (same approximation the
    # gradient uses). Overwrites the linear-IV θ₁ SE so θ₁ accounts for θ₂ estimation uncertainty.
    _sem = se_method()
    if _sem in ("sandwich", "wcb")
        try
            valid  = pc.theta1_valid .& isfinite.(delta_final)
            compute_model_shares_gpu!(buf, gbuf, delta_final, pc, R)          # shares at (δ*, θ₂*)
            s_base = collect_model_shares(buf, pc, N_obs)
            Ddelta = compute_delta_jacobian_gpu(delta_final, s_base, theta2_star, buf, gbuf,
                        prod_vec, nu_draws, sigma_indices, pi_interactions, R, coef_dim, pc)
            se_all, pval_all = gmm_cluster_ses(_sem, vcat(theta1_star, theta2_star),
                        pc.Z_moments[valid, :], pc.X_full[valid, :], Ddelta[valid, :],
                        delta_final[valid] .- pc.X_full[valid, :] * theta1_star,
                        W, pc.clusters[valid];
                        n_sigma=length(sigma_indices))   # profile out on-bound σ's from the covariance
            K1 = length(theta1_star)
            theta1_se   = se_all[1:K1];   theta2_se   = se_all[K1+1:end]
            theta1_pval = pval_all[1:K1]; theta2_pval = pval_all[K1+1:end]
            println("  SE method: $_sem  |  θ₂ SE: $(round.(theta2_se, sigdigits=3))")
        catch e
            println("  [!] SE ($_sem) failed: $e — keeping linear θ₁ SE, θ₂_se=NaN (not computed).")
        end
    end

    results["theta1"]             = theta1_star
    results["theta1_se"]          = theta1_se
    results["theta2"]             = theta2_star
    results["theta2_se"]          = theta2_se
    results["theta1_pval"]        = theta1_pval
    results["theta2_pval"]        = theta2_pval
    results["se_method"]          = _sem
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

function main_gpu_ift()
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
                    "theta1_pval"        => replace(round.(get(res, "theta1_pval", Float64[]), sigdigits=6), NaN=>1.0),
                    "theta2_pval"        => replace(round.(get(res, "theta2_pval", Float64[]), sigdigits=6), NaN=>1.0),
                    "se_method"          => get(res, "se_method", "none"),
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

# [merged: per-file autorun removed — see dispatch at end]

# ── Single entrypoint for DIRECT execution ──────────────────────────────────────
# Inert when this file is include()d (PROGRAM_FILE != @__FILE__), e.g. by the blp_2_rc.jl
# driver, which calls main_gpu_ift()/main_gpu_numerical() itself.
if abspath(PROGRAM_FILE) == @__FILE__
    let _eng = lowercase(get(ENV, "BLP_ENGINE", "ift"))
        # An unrecognised engine used to fall through to IFT and OVERWRITE the production results.
        _eng in ("ift", "numerical", "cue") ||
            error("BLP_ENGINE='$(_eng)' is not a recognised engine (ift | numerical | cue).")
        # Likewise, a non-ift engine with an empty suffix writes to the ift filenames.
        if _eng != "ift" && isempty(output_suffix())
            error("BLP_ENGINE=$(_eng) with an empty BLP_OUTPUT_SUFFIX would overwrite the " *
                  "production (IFT) artifacts. Run via blp_2_rc.jl, which sets the suffix, or " *
                  "export BLP_OUTPUT_SUFFIX explicitly.")
        end
        _eng == "ift" ? main_gpu_ift() : main_gpu_numerical()
    end
end
