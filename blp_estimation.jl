"""
blp_estimation.jl
=================
Self-contained BLP demand estimation using pre-computed draws from blp_draws.jl.

All BLP core functions are inlined here — blp_loop.jl is NOT loaded at runtime.
Pre-computed ν-draws and demographic draws are loaded from BLP_DRAWS/ once and
shared across all specs/stages, ensuring cross-strategy comparability.

Differences from blp_loop.jl:
  1. Draws are LOADED (not generated) — blp_draws.jl must run first
  2. Tighter outer bounds [-5, 5] per CG2020
  3. Clustering on CodConglomeradoPrudencial only (matches sleep estimation)

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

function input_filename(estim::Int, spec_id::Int)::String
    prefix = estim == 5 ? "demand_5_alt2logistic" : "demand_$(estim)"
    return "$(prefix)_final_spec_$(spec_id).parquet"
end

function log_status(msg::String)
    stamped = "[$(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))] $msg"
    lock(_log_lock) do; push!(_log_buf, stamped); end
    println(stamped); flush(stdout)
end

# ==========================================================================
# 0b. Paths (3-tuple: input, draws, output)
# ==========================================================================
function get_paths(is_hpc::Bool; local_dir::Union{String,Nothing}=nothing)
    if is_hpc
        input_dir  = joinpath(@__DIR__, "..", "data", "input")
        draws_dir  = joinpath(@__DIR__, "..", "data", "output", "BLP_DRAWS")
        output_dir = joinpath(@__DIR__, "..", "data", "output")
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
# 1. Theta2 Structure
# ==========================================================================
function build_theta2_structure(stage::String)
    if stage == "logit"
        return Int[], Tuple{Int,Int}[], 0
    elseif stage == "sigma"
        return [1], Tuple{Int,Int}[], 1
    elseif stage == "full"
        sigma_idx = [1, 1 + findfirst(==("log_total_assets_lag"), X_COLS)]
        pi_inter  = [(1, findfirst(==("gdp_per_capita"),     D_COLS)),
                     (1, findfirst(==("fraction_65plus"),    D_COLS)),
                     (1, findfirst(==("connections_per100"), D_COLS))]
        return sigma_idx, pi_inter, length(sigma_idx) + length(pi_inter)
    elseif stage == "extended"
        sigma_idx = [1, 1 + findfirst(==("log_total_assets_lag"), X_COLS)]
        pi_inter  = [
            (1,                                                 findfirst(==("gdp_per_capita"),              D_COLS)),
            (1,                                                 findfirst(==("fraction_65plus"),             D_COLS)),
            (1,                                                 findfirst(==("connections_per100"),          D_COLS)),
            (1 + findfirst(==("log_total_assets_lag"), X_COLS), findfirst(==("gdp_per_capita"),             D_COLS)),
            (1 + findfirst(==("fgc_covered"),          X_COLS), findfirst(==("fraction_65plus"),            D_COLS)),
            (1 + findfirst(==("equity_ratio_lag"),     X_COLS), findfirst(==("cadunico_families_per1000"),  D_COLS)),
        ]
        return sigma_idx, pi_inter, length(sigma_idx) + length(pi_inter)
    else
        error("Unknown stage: $stage")
    end
end

function unpack_theta2(theta2::Vector{Float64}, sigma_indices, pi_interactions)
    n_s = length(sigma_indices)
    n_p = length(pi_interactions)
    return theta2[1:n_s], theta2[n_s+1:n_s+n_p]
end

# ==========================================================================
# 2. HotBuffers — pre-allocated workspace for the inner loop
# ==========================================================================
mutable struct HotBuffers
    mu             ::Matrix{Float64}
    sigma_nu       ::Matrix{Float64}
    sigma_diag     ::Vector{Float64}
    delta_B        ::Vector{Float64}
    delta_D        ::Vector{Float64}
    V_B            ::Matrix{Float64}
    V_D            ::Matrix{Float64}
    q_B            ::Matrix{Float64}
    s_B            ::Vector{Float64}
    s_D            ::Vector{Float64}
    log_sum_D_time ::Matrix{Float64}
    log_D_sum_pair ::Matrix{Float64}
    log_sum_B_mkt  ::Matrix{Float64}
    joint_max      ::Matrix{Float64}
    log_denom      ::Matrix{Float64}
    log_denom_B    ::Matrix{Float64}
    neg_log_denom  ::Matrix{Float64}
    max_neg_ld     ::Matrix{Float64}
    shifted_inv    ::Matrix{Float64}
    sum_wtd        ::Matrix{Float64}
    log_inv_wtd    ::Matrix{Float64}
    log_s_D_r      ::Matrix{Float64}
    row_max        ::Matrix{Float64}
    delta_new      ::Vector{Float64}
    f              ::Vector{Float64}
    pi_products    ::Vector{Matrix{Float64}}
end

function allocate_hot_buffers(N::Int, N_B::Int, N_D::Int, R::Int,
                               n_pairs::Int, n_times::Int, coef_dim::Int,
                               n_pi::Int)::HotBuffers
    println("  [HPC] Pre-allocating hot buffers...")
    total_gb = (
        N*R + R*coef_dim + coef_dim +
        N_B + N_D +
        N_B*R + N_D*R + N_B*R +
        N_B + N_D +
        n_times*R + n_pairs*R*2 +
        n_pairs*R*3 +
        N_B*R +
        n_times*R*3 +
        n_pairs*R +
        N_D*R + N_D +
        N*2 +
        n_pi*N*R
    ) * 8 / 1e9
    println("  [HPC] Total buffer allocation: $(round(total_gb, digits=2)) GB")
    return HotBuffers(
        zeros(N, R),
        zeros(R, coef_dim),
        zeros(coef_dim),
        zeros(N_B),
        zeros(N_D),
        zeros(N_B, R),
        zeros(N_D, R),
        zeros(N_B, R),
        zeros(N_B),
        zeros(N_D),
        zeros(n_times, R),
        zeros(n_pairs, R),
        zeros(n_pairs, R),
        zeros(n_pairs, R),
        zeros(n_pairs, R),
        zeros(N_B, R),
        zeros(n_pairs, R),
        zeros(n_times, R),
        zeros(n_pairs, R),
        zeros(n_times, R),
        zeros(n_times, R),
        zeros(N_D, R),
        zeros(N_D, 1),
        zeros(N),
        zeros(N),
        Matrix{Float64}[],
    )
end

function precompute_pi_products!(buf::HotBuffers,
                                  prod_vec::Matrix{Float64},
                                  stacked_draws::Array{Float64,3},
                                  obs_key_idx::Vector{Int},
                                  pi_interactions::Vector{Tuple{Int,Int}},
                                  coef_dim::Int)
    D_dim = size(stacked_draws, 3)
    N     = size(prod_vec, 1)
    buf.pi_products = Matrix{Float64}[]
    for (cidx, didx) in pi_interactions
        if cidx <= coef_dim && didx <= D_dim
            prod = @view(prod_vec[:, cidx]) .* stacked_draws[obs_key_idx, :, didx]
            push!(buf.pi_products, prod)
            println("  [HPC] Pre-computed Pi product $(length(buf.pi_products)): $(round(sizeof(prod)/1e9, digits=2)) GB")
        else
            push!(buf.pi_products, zeros(0, 0))
        end
    end
end

function compute_mu!(buf::HotBuffers,
                     prod_vec::Matrix{Float64},
                     nu_draws::Matrix{Float64},
                     sigma_vals::Vector{Float64},
                     sigma_indices::Vector{Int},
                     pi_vals::Vector{Float64},
                     R::Int, coef_dim::Int)
    fill!(buf.sigma_diag, 0.0)
    @inbounds for (pos, cidx) in enumerate(sigma_indices)
        cidx <= coef_dim && (buf.sigma_diag[cidx] = sigma_vals[pos])
    end
    @inbounds for j in 1:coef_dim
        sd = buf.sigma_diag[j]
        for r in 1:R
            buf.sigma_nu[r, j] = nu_draws[r, j] * sd
        end
    end
    mul!(buf.mu, prod_vec, buf.sigma_nu')
    @inbounds for (pi_idx, pi_prod) in enumerate(buf.pi_products)
        size(pi_prod, 1) == 0 && continue
        pv = pi_vals[pi_idx]
        buf.mu .+= pv .* pi_prod
    end
end

# ==========================================================================
# 3. Market Index Construction
# ==========================================================================
struct Precomp
    b_mkt_idx       ::Vector{Int}
    d_time_enc      ::Vector{Int}
    unique_pairs    ::Vector{Tuple{String,String}}
    unique_times    ::Vector{String}
    pair_time_enc   ::Vector{Int}
    pop_weights     ::Vector{Float64}
    B_agg           ::SparseMatrixCSC{Float64,Int}
    D_agg           ::SparseMatrixCSC{Float64,Int}
    PT_agg          ::SparseMatrixCSC{Float64,Int}
    sort_b          ::Vector{Int}
    b_grp_start     ::Vector{Int}
    sort_d          ::Vector{Int}
    d_uval          ::Vector{Int}
    d_grp_start     ::Vector{Int}
    sort_pt         ::Vector{Int}
    pt_uval         ::Vector{Int}
    pt_grp_start    ::Vector{Int}
    b_mask          ::BitVector
    d_mask          ::BitVector
    ln_s_data_D     ::Vector{Float64}
    ln_s_data_B_cond::Vector{Float64}
    Z_moments       ::Matrix{Float64}
    X_full          ::Matrix{Float64}
    X_hat           ::Matrix{Float64}
    theta1_valid    ::BitVector
    clusters        ::Vector{String}
end

function _unique_with_starts(sorted_vec::Vector{Int})
    isempty(sorted_vec) && return Int[], Int[]
    uval   = Int[sorted_vec[1]]
    gstart = Int[1]
    @inbounds for i in 2:length(sorted_vec)
        if sorted_vec[i] != sorted_vec[i-1]
            push!(uval, sorted_vec[i])
            push!(gstart, i)
        end
    end
    return uval, gstart
end

function build_precomp(df::DataFrame, Z::Matrix{Float64},
                       X_full::Matrix{Float64}, X_hat::Matrix{Float64},
                       valid::BitVector, clusters::Vector{String})::Precomp
    N        = nrow(df)
    b_mask   = BitVector(Bool.(coalesce.(df.is_B, false)))
    d_mask   = .!b_mask
    N_B      = sum(b_mask)
    N_D      = sum(d_mask)

    mca_codes = string.(df.mca_code)
    time_ids  = string.(df.time_id)
    pop_total = coalesce.(df.pop_total, 0.0)

    raw_pairs     = collect(zip(mca_codes[b_mask], time_ids[b_mask]))
    unique_pairs  = sort(unique(raw_pairs))
    pair_to_idx   = Dict(p => i for (i, p) in enumerate(unique_pairs))
    b_mkt_idx     = [pair_to_idx[(m, t)] for (m, t) in raw_pairs]

    unique_times  = sort(unique(time_ids))
    time_to_idx   = Dict(t => i for (i, t) in enumerate(unique_times))
    d_time_enc    = [time_to_idx[t] for t in time_ids[d_mask]]

    n_pairs       = length(unique_pairs)
    n_times       = length(unique_times)
    pair_time_enc = [time_to_idx[p[2]] for p in unique_pairs]

    pop_b       = pop_total[b_mask]
    pair_pop    = zeros(n_pairs)
    for (i, w) in zip(b_mkt_idx, pop_b); pair_pop[i] += w; end
    time_pop    = zeros(n_times)
    for (i, w) in zip(pair_time_enc, pair_pop); time_pop[i] += w; end
    pop_weights = pair_pop ./ max.(time_pop[pair_time_enc], 1e-30)

    B_agg  = sparse(b_mkt_idx,     1:N_B,     ones(N_B),     n_pairs, N_B)
    D_agg  = sparse(d_time_enc,    1:N_D,     ones(N_D),     n_times, N_D)
    PT_agg = sparse(pair_time_enc, 1:n_pairs, pop_weights,   n_times, n_pairs)

    sort_b            = sortperm(b_mkt_idx)
    _, b_grp_s        = _unique_with_starts(b_mkt_idx[sort_b])
    sort_d            = sortperm(d_time_enc)
    d_uval, d_grp_s   = _unique_with_starts(d_time_enc[sort_d])
    sort_pt           = sortperm(pair_time_enc)
    pt_uval, pt_grp_s = _unique_with_starts(pair_time_enc[sort_pt])

    ln_s_D = log.(clamp.(coalesce.(df.share_D,      0.0), 1e-15, Inf))
    ln_s_B = log.(clamp.(coalesce.(df.share_B_cond, 0.0), 1e-15, Inf))

    return Precomp(b_mkt_idx, d_time_enc, unique_pairs, unique_times,
                   pair_time_enc, pop_weights,
                   B_agg, D_agg, PT_agg,
                   sort_b, b_grp_s,
                   sort_d, d_uval, d_grp_s,
                   sort_pt, pt_uval, pt_grp_s,
                   b_mask, d_mask, ln_s_D, ln_s_B,
                   Z, X_full, X_hat, valid, clusters)
end

# ==========================================================================
# 4. Model Shares — in-place
# ==========================================================================
function logsumexp_groups!(result::Matrix{Float64},
                            V, sort_idx, grp_start, uval, R)
    fill!(result, -Inf)
    n_uniq = length(grp_start)
    @inbounds for g in 1:n_uniq
        gidx       = grp_start[g]
        gend       = g < n_uniq ? grp_start[g+1] - 1 : length(sort_idx)
        target_row = uval[g]
        for c in 1:R
            mx = -Inf
            for ii in gidx:gend
                v = V[sort_idx[ii], c]; v > mx && (mx = v)
            end
            s = 0.0
            for ii in gidx:gend
                s += exp(V[sort_idx[ii], c] - mx)
            end
            result[target_row, c] = mx + log(max(s, 1e-300))
        end
    end
end

function compute_model_shares!(buf::HotBuffers, delta::Vector{Float64},
                                pc::Precomp, R::Int)
    N_B     = length(buf.delta_B)
    N_D     = length(buf.delta_D)
    n_pairs = length(pc.unique_pairs)
    n_times = length(pc.unique_times)

    b_idx = 0; d_idx = 0
    @inbounds for i in eachindex(pc.b_mask)
        if pc.b_mask[i]
            b_idx += 1
            buf.delta_B[b_idx] = delta[i]
            for r in 1:R
                buf.V_B[b_idx, r] = clamp(delta[i] + buf.mu[i, r], -500.0, 500.0)
            end
        else
            d_idx += 1
            buf.delta_D[d_idx] = delta[i]
            for r in 1:R
                buf.V_D[d_idx, r] = clamp(delta[i] + buf.mu[i, r], -500.0, 500.0)
            end
        end
    end

    logsumexp_groups!(buf.log_sum_D_time, buf.V_D, pc.sort_d, pc.d_grp_start, pc.d_uval, R)
    @inbounds for i in 1:n_pairs
        for r in 1:R
            buf.log_D_sum_pair[i, r] = buf.log_sum_D_time[pc.pair_time_enc[i], r]
        end
    end

    logsumexp_groups!(buf.log_sum_B_mkt, buf.V_B, pc.sort_b, pc.b_grp_start, collect(1:n_pairs), R)

    log_outside = 0.0
    @inbounds for i in 1:n_pairs
        for r in 1:R
            jm = max(buf.log_sum_B_mkt[i, r], buf.log_D_sum_pair[i, r], log_outside)
            buf.joint_max[i, r] = jm
            buf.log_denom[i, r] = jm + log(max(
                exp(log_outside - jm) +
                exp(buf.log_sum_B_mkt[i, r] - jm) +
                exp(buf.log_D_sum_pair[i, r] - jm), 1e-300))
        end
    end

    @inbounds for i in 1:N_B
        mkt = pc.b_mkt_idx[i]
        for r in 1:R
            buf.q_B[i, r] = exp(buf.V_B[i, r] - buf.log_denom[mkt, r])
        end
        s = 0.0
        for r in 1:R; s += buf.q_B[i, r]; end
        buf.s_B[i] = s / R
    end

    @inbounds for i in 1:n_pairs
        for r in 1:R
            buf.neg_log_denom[i, r] = -buf.log_denom[i, r]
        end
    end

    fill!(buf.max_neg_ld, -Inf)
    @inbounds for g in 1:length(pc.pt_grp_start)
        gidx   = pc.pt_grp_start[g]
        gend   = g < length(pc.pt_grp_start) ? pc.pt_grp_start[g+1] - 1 : length(pc.sort_pt)
        target = pc.pt_uval[g]
        for r in 1:R
            mx = -Inf
            for ii in gidx:gend
                v = buf.neg_log_denom[pc.sort_pt[ii], r]; v > mx && (mx = v)
            end
            buf.max_neg_ld[target, r] = mx
        end
    end

    @inbounds for i in 1:n_pairs
        for r in 1:R
            buf.shifted_inv[i, r] = exp(buf.neg_log_denom[i, r] -
                                        buf.max_neg_ld[pc.pair_time_enc[i], r])
        end
    end

    mul!(buf.sum_wtd, pc.PT_agg, buf.shifted_inv)

    @inbounds for i in 1:n_times
        for r in 1:R
            buf.log_inv_wtd[i, r] = buf.max_neg_ld[i, r] +
                                    log(max(buf.sum_wtd[i, r], 1e-300))
        end
    end

    @inbounds for i in 1:N_D
        rm = -Inf
        for r in 1:R
            v = buf.V_D[i, r] + buf.log_inv_wtd[pc.d_time_enc[i], r]
            buf.log_s_D_r[i, r] = v
            v > rm && (rm = v)
        end
        buf.row_max[i, 1] = rm
        s = 0.0
        for r in 1:R; s += exp(buf.log_s_D_r[i, r] - rm); end
        buf.s_D[i] = (s / R) * exp(rm)
    end
end

# ==========================================================================
# 5. SQUAREM BLP Contraction — in-place
# ==========================================================================
function blp_contraction!(buf::HotBuffers, delta::Vector{Float64},
                           pc::Precomp, R::Int;
                           tol::Float64=1e-12, max_iter::Int=5000)
    N            = length(delta)
    norm_history = Float64[]
    x1     = Vector{Float64}(undef, N)
    x2     = Vector{Float64}(undef, N)
    r_vec  = Vector{Float64}(undef, N)
    v_vec  = Vector{Float64}(undef, N)
    x_prop = Vector{Float64}(undef, N)

    function T_inplace!(dest::Vector{Float64}, src::Vector{Float64})
        compute_model_shares!(buf, src, pc, R)
        b_idx = 0; d_idx = 0
        @inbounds for i in 1:N
            if pc.b_mask[i]
                b_idx += 1
                dest[i] = clamp(src[i] + pc.ln_s_data_B_cond[i] -
                                log(clamp(buf.s_B[b_idx], 1e-15, Inf)), -500.0, 500.0)
            else
                d_idx += 1
                dest[i] = clamp(src[i] + pc.ln_s_data_D[i] -
                                log(clamp(buf.s_D[d_idx], 1e-15, Inf)), -500.0, 500.0)
            end
        end
    end

    fevals = 0
    while fevals + 2 <= max_iter
        T_inplace!(x1, delta); fevals += 1
        if !all(isfinite, x1)
            println("    [SQUAREM ABORT] non-finite at eval=$fevals")
            return false, fevals, norm_history
        end
        nm = 0.0
        @inbounds for i in 1:N
            r_vec[i] = x1[i] - delta[i]
            a = abs(r_vec[i]); a > nm && (nm = a)
        end
        push!(norm_history, nm)
        (fevals % 50 == 0 || nm < tol) &&
            println("    [SQUAREM eval=$fevals/$max_iter] norm=$(round(nm, sigdigits=4))")
        if nm < tol
            copyto!(delta, x1); return true, fevals, norm_history
        end

        T_inplace!(x2, x1); fevals += 1
        if !all(isfinite, x2)
            copyto!(delta, x1); return false, fevals, norm_history
        end
        nm2 = 0.0
        @inbounds for i in 1:N; a = abs(x2[i]-x1[i]); a > nm2 && (nm2 = a); end
        push!(norm_history, nm2)
        (fevals % 50 == 0 || nm2 < tol) &&
            println("    [SQUAREM eval=$fevals/$max_iter] norm=$(round(nm2, sigdigits=4))")
        if nm2 < tol
            copyto!(delta, x2); return true, fevals, norm_history
        end

        norm_r_sq = 0.0; norm_v_sq = 0.0
        @inbounds for i in 1:N
            v_vec[i]   = (x2[i] - x1[i]) - r_vec[i]
            norm_r_sq += r_vec[i]^2
            norm_v_sq += v_vec[i]^2
        end
        if norm_v_sq < 1e-28
            copyto!(delta, x2); continue
        end
        α = -sqrt(norm_r_sq / norm_v_sq)
        @inbounds for i in 1:N
            x_prop[i] = clamp(delta[i] - 2α * r_vec[i] + α^2 * v_vec[i], -500.0, 500.0)
        end
        if !all(isfinite, x_prop)
            copyto!(delta, x2)
        else
            norm_prop = 0.0
            @inbounds for i in 1:N; a = abs(x_prop[i]-x2[i]); a > norm_prop && (norm_prop = a); end
            if norm_prop > 100.0 * nm2 + 1.0
                copyto!(delta, x2)
            else
                copyto!(delta, x_prop)
            end
        end
    end
    return false, fevals, norm_history
end

# ==========================================================================
# 6. Linear IV (Eq-A5)
# ==========================================================================
function build_regressor_matrices(df::DataFrame)
    N           = nrow(df)
    spread_cols = coalesce.(df.spread_qoq, 0.0)
    x_mat       = zeros(N, L_PROD)
    for (i, col) in enumerate(X_COLS)
        col in names(df) && (x_mat[:, i] .= coalesce.(df[!, col], 0.0))
    end
    iv_cols = [c for c in vcat(IV_BLP_LOO, IV_COST, IV_CAPITAL) if c in names(df)]
    Z_mat   = zeros(N, length(iv_cols))
    for (i, col) in enumerate(iv_cols)
        v = coalesce.(df[!, col], 0.0)
        v = replace(v, Inf => 0.0, -Inf => 0.0)
        Z_mat[:, i] .= v
    end
    return spread_cols, x_mat, Z_mat, iv_cols
end

function project_endogenous_spreads(spread_cols::Vector{Float64},
                                     H::Matrix{Float64},
                                     deposit_types::Vector{Int})::Matrix{Float64}
    spread_hat = reshape(copy(spread_cols), :, 1)
    for k_val in [4, 5]
        k_mask   = deposit_types .== k_val
        any(k_mask) || continue
        H_k      = H[k_mask, :]
        spread_k = spread_cols[k_mask]
        valid    = all.(isfinite, eachrow(H_k)) .& isfinite.(spread_k)
        sum(valid) > size(H_k, 2) || continue
        try
            beta_fs           = H_k[valid, :] \ spread_k[valid]
            k_hat             = copy(spread_k)
            k_hat[valid]     .= H_k[valid, :] * beta_fs
            spread_hat[k_mask, 1] .= k_hat
        catch; end
    end
    return spread_hat
end

function estimate_theta1(delta::Vector{Float64}, pc::Precomp)
    valid  = pc.theta1_valid .& isfinite.(delta)
    X_v    = pc.X_hat[valid, :]
    d_v    = delta[valid]
    theta1 = try X_v \ d_v catch; zeros(size(pc.X_hat, 2)) end
    xi     = delta .- pc.X_full * theta1
    return theta1, xi
end

"""Cluster-robust sandwich SEs for theta1 (IV/2SLS) + IK2016 effective cluster count.
Returns (se, G_nominal, G_star) where G_star = G/(1+CV²) per Imbens & Kolesár (2016)."""
function compute_cluster_se(theta1::Vector{Float64}, delta::Vector{Float64}, pc::Precomp)
    valid    = pc.theta1_valid .& isfinite.(delta)
    N_v      = sum(valid)
    K        = length(theta1)
    X_hat_v  = pc.X_hat[valid, :]    # X̃ (first-stage projected regressors)
    X_full_v = pc.X_full[valid, :]   # X  (original regressors)
    xi_v     = delta[valid] .- X_full_v * theta1
    cl_v     = pc.clusters[valid]
    # Bread: (X̃'X)⁻¹
    XtX     = X_hat_v' * X_full_v
    XtX_inv = try inv(XtX) catch; pinv(XtX) end
    # Meat: Σ_c score_c ⊗ score_c  where score_c = X̃_c' ξ_c
    unique_cl = unique(cl_v)
    G         = length(unique_cl)
    meat      = zeros(K, K)
    for c in unique_cl
        mask    = cl_v .== c
        score_c = X_hat_v[mask, :]' * xi_v[mask]
        meat   .+= score_c * score_c'
    end
    # CR1 small-sample correction
    correction = G > 1 ? (G / (G - 1)) * (N_v / (N_v - K)) : 1.0
    V_cluster  = XtX_inv * meat * XtX_inv' .* correction
    se         = sqrt.(max.(diag(V_cluster), 0.0))
    # IK2016 effective clusters: G* = G / (1 + CV²)
    cl_sizes = [count(==(c), cl_v) for c in unique_cl]
    mu_G     = Statistics.mean(cl_sizes)
    cv_G     = mu_G > 0 ? Statistics.std(cl_sizes; corrected=false) / mu_G : 0.0
    G_star   = max(1.0, G / (1.0 + cv_G^2))
    return se, G, G_star
end

# ==========================================================================
# 7. GMM Objective
# ==========================================================================
function compute_gmm_moments(xi::Vector{Float64}, pc::Precomp)
    return vec(Statistics.mean(xi .* pc.Z_moments; dims=1))
end

function gmm_objective!(buf::HotBuffers, theta2::Vector{Float64},
                         prod_vec, nu_draws,
                         sigma_indices, pi_interactions,
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
    G          = compute_gmm_moments(xi, pc)
    return dot(G, W * G)
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

        theta1, xi              = estimate_theta1(delta, pc_logit)
        theta1_se, n_cl, G_s    = compute_cluster_se(theta1, delta, pc_logit)
        g_moments = compute_gmm_moments(xi, pc_logit)
        W  = Matrix(I(length(iv_avail)) * 1.0)
        Q  = dot(g_moments, W * g_moments)
        results["theta1"]             = theta1
        results["theta1_se"]          = theta1_se
        results["theta2"]             = Float64[]
        results["theta2_se"]          = Float64[]
        results["delta"]              = delta
        results["xi"]                 = xi
        results["Q_value"]            = Q
        results["converged"]          = true
        results["param_names_theta1"] = vcat(["alpha"], X_COLS)
        results["sigma_indices"]      = Int[]
        results["pi_interactions"]    = Tuple{Int,Int}[]
        results["n_clusters"]         = n_cl
        results["G_star"]             = G_s
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

    theta1_star, xi_star         = estimate_theta1(delta_final, pc)
    theta1_se, n_cl, G_s         = compute_cluster_se(theta1_star, delta_final, pc)

    results["theta1"]             = theta1_star
    results["theta1_se"]          = theta1_se
    results["theta2"]             = theta2_star
    results["theta2_se"]          = fill(NaN, length(theta2_star))
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
            # Per-spec JSON for Python consumption
            json_path = replace(out_path, ".jls" => ".json")
            try
                json_out = Dict{String,Any}(
                    "estim"              => get(res, "estim", estim),
                    "spec_id"            => get(res, "spec_id", sp),
                    "stage"              => get(res, "stage", current_stage),
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
        log_status("[DONE] BLP E$estim ($current_stage) complete for specs $spec_ids.")
    end
end

main()
