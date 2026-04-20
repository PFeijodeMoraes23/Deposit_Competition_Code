"""
estimation_5_demand_2_loop_ju_hpc.jl
=====================================
HPC-optimised version of estimation_5_demand_2_loop_ju.jl.

Identical to estimation_1_demand_2_loop_ju_hpc.jl except for:
  - SPEC_NUM = 5
  - Extra --alt argument with choices: alt1 alt2 alt2linear alt2logistic all
  - Input parquet: demand_5_{alt}_final_spec_{spec_id}.parquet
  - Output:  blp_results_spec_5_{alt}_{sp}_{stage}.jls
  - Summary: blp_summary_{alt}_5_{stage}.json
  - Outer loop: for each alt in alts_to_process, for each stage

HPC optimisations (requires 750 GB+ RAM):
  1. Pre-allocated HotBuffers struct — eliminates ALL GC allocation in the
     inner contraction loop (mu, V_B, V_D, delta_new, f, logsumexp scratch).
  2. In-place BLAS via mul! for the main (N,R) dgemm — zero intermediate alloc.
  3. Pre-computed Pi interaction products (N,R each, ~3 GB) — cached once since
     they depend only on fixed demographic draws × prod_vec, not on theta2/delta.
  4. @inbounds on all inner loops in logsumexp_groups and contraction.
  5. Chunking path removed — full (N,R) always fits at 750 GB.
  6. BLAS thread count set explicitly at startup.

Usage (HPC):
  julia --threads=32 estimation_5_demand_2_loop_ju_hpc.jl --spec 12 --stage sequence --R 2000 --alt all --hpc
"""

using Parquet2, DataFrames, SparseArrays, LinearAlgebra, Statistics
using Random, Optim, QuasiMonteCarlo, Distributions
using JSON3, Serialization, ArgParse, Printf, Dates

# ── Set BLAS threads to match Julia threads ──────────────────────────────────
BLAS.set_num_threads(Threads.nthreads())

# --------------------------------------------------------------------------
# 0. Constants
# --------------------------------------------------------------------------
const SPEC_NUM     = 5
const ALT_CHOICES  = ["alt1", "alt2", "alt2linear", "alt2logistic", "all"]
const ALT_LIST     = ["alt1", "alt2", "alt2linear", "alt2logistic"]

const X_COLS     = ["fgc_covered", "has_ip", "seg_S2", "seg_S3", "seg_S4", "seg_S5",
                    "log_total_assets_lag", "equity_ratio_lag"]
const L_PROD     = length(X_COLS)
const K_TYPES    = 4
const K_LIST     = [1, 2, 4, 5]
const D_COLS     = ["gdp_per_capita", "fraction_65plus", "fraction_young",
                    "pix_users_pf_per1000", "connections_per100", "frac_4g5g",
                    "branches_per1000", "cadunico_families_per1000"]
const D_DIM      = length(D_COLS)

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

# --------------------------------------------------------------------------
# 0b. Paths
# --------------------------------------------------------------------------
function get_paths(is_hpc::Bool; local_dir::Union{String,Nothing}=nothing)
    if is_hpc
        return "/home/pf382/dep_comp/data/input", "/home/pf382/dep_comp/data/output"
    else
        data_dir = local_dir !== nothing ? local_dir :
                   joinpath(dirname(dirname(dirname(abspath(@__FILE__)))), "BCB", "Egan_et_al_2025_Rep", "processed")
        return joinpath(data_dir, "ESTIMATION_OUTPUT", "DEMAND_PREP"),
               joinpath(data_dir, "ESTIMATION_OUTPUT", "BLP_RESULTS")
    end
end

function log_status(msg::String)
    stamped = "[$(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))] $msg"
    lock(_log_lock) do; push!(_log_buf, stamped); end
    println(stamped); flush(stdout)
end

# --------------------------------------------------------------------------
# 1. Data Loading (alt-aware)
# --------------------------------------------------------------------------
function load_merged_spec_data(spec_id::Int, alt::String; is_hpc::Bool=false, local_dir=nothing)
    input_dir, _ = get_paths(is_hpc; local_dir=local_dir)
    path = joinpath(input_dir, "demand_$(SPEC_NUM)_$(alt)_final_spec_$(spec_id).parquet")
    isfile(path) || error("Missing: $path")
    return DataFrame(Parquet2.Dataset(path); copycols=true)
end

function load_sigma_table(; is_hpc::Bool=false, local_dir=nothing)
    input_dir, _ = get_paths(is_hpc; local_dir=local_dir)
    path = joinpath(input_dir, "demographics_sigma.parquet")
    if !isfile(path)
        println("WARNING: demographics_sigma.parquet not found — falling back to 0.1 × σ_national")
        return nothing
    end
    df = DataFrame(Parquet2.Dataset(path); copycols=true)
    sigma_cols = [c * "_sigma" for c in D_COLS]
    avail = [c for c in sigma_cols if c in names(df)]
    tbl = Dict{Tuple{String,String}, Vector{Float64}}()
    for row in eachrow(df)
        key = (string(row.mca_code), string(row.time_id))
        tbl[key] = Float64[coalesce(row[c], 0.0) for c in avail]
    end
    println("Loaded demographics_sigma: $(length(tbl)) market-time σ entries")
    return tbl
end

# --------------------------------------------------------------------------
# 2. Simulation Draws
# --------------------------------------------------------------------------
function generate_halton_draws(R::Int, dim::Int, seed::Int)::Matrix{Float64}
    skip = seed * R
    n_total = R + skip
    pts_full = QuasiMonteCarlo.sample(n_total, zeros(dim), ones(dim), HaltonSample())
    pts = pts_full[:, (skip+1):end]
    pts = clamp.(pts, 1e-6, 1.0 - 1e-6)
    return quantile.(Normal(), pts)'
end

function generate_demographic_draws(df::DataFrame, R::Int, seed::Int;
                                    sigma_table::Union{Nothing, Dict{Tuple{String,String}, Vector{Float64}}}=nothing)
    rng = MersenneTwister(seed)
    d_cols = [c for c in D_COLS if c in names(df)]
    D = length(d_cols)
    mca_time = unique(df[:, ["mca_code", "time_id", d_cols...]])
    nat_std  = [std(skipmissing(mca_time[!, c])) for c in d_cols]
    nat_std  = [s == 0.0 ? 1.0 : s for s in nat_std]
    draws = Dict{Tuple{String,String}, Matrix{Float64}}()
    for row in eachrow(mca_time)
        key = (string(row.mca_code), string(row.time_id))
        mu  = [coalesce(row[c], 0.0) for c in d_cols]
        if sigma_table !== nothing && haskey(sigma_table, key)
            mkt_std = sigma_table[key]
            mkt_std = [s <= 0.0 ? nat_std[i] : s for (i, s) in enumerate(mkt_std)]
        else
            mkt_std = nat_std .* 0.1
        end
        draws[key] = mu .+ mkt_std .* randn(rng, D, R) |> transpose |> Matrix
    end
    return draws
end

# --------------------------------------------------------------------------
# 3. Theta2 Structure
# --------------------------------------------------------------------------
function build_theta2_structure(stage::String)
    if stage == "logit"
        return Int[], Tuple{Int,Int}[], 0
    elseif stage == "sigma"
        return [1], Tuple{Int,Int}[], 1
    elseif stage == "full"
        sigma_idx = [1, 1 + findfirst(==(("log_total_assets_lag")), X_COLS)]
        pi_inter  = [(1, findfirst(==("gdp_per_capita"),      D_COLS)),
                     (1, findfirst(==("fraction_65plus"),     D_COLS)),
                     (1, findfirst(==("connections_per100"),  D_COLS))]
        return sigma_idx, pi_inter, length(sigma_idx) + length(pi_inter)
    elseif stage == "extended"
        sigma_idx = [1, 1 + findfirst(==("log_total_assets_lag"), X_COLS)]
        pi_inter  = [(1,                                           findfirst(==("gdp_per_capita"),             D_COLS)),
                     (1,                                           findfirst(==("fraction_65plus"),            D_COLS)),
                     (1,                                           findfirst(==("connections_per100"),         D_COLS)),
                     (1 + findfirst(==("log_total_assets_lag"), X_COLS), findfirst(==("gdp_per_capita"),      D_COLS)),
                     (1 + findfirst(==("fgc_covered"),          X_COLS), findfirst(==("fraction_65plus"),     D_COLS)),
                     (1 + findfirst(==("equity_ratio_lag"),     X_COLS), findfirst(==("cadunico_families_per1000"), D_COLS))]
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

# --------------------------------------------------------------------------
# 4. HotBuffers — pre-allocated workspace for the inner loop
# --------------------------------------------------------------------------
mutable struct HotBuffers
    mu         ::Matrix{Float64}
    sigma_nu   ::Matrix{Float64}
    sigma_diag ::Vector{Float64}
    delta_B    ::Vector{Float64}
    delta_D    ::Vector{Float64}
    V_B        ::Matrix{Float64}
    V_D        ::Matrix{Float64}
    q_B        ::Matrix{Float64}
    s_B        ::Vector{Float64}
    s_D        ::Vector{Float64}
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
    delta_new  ::Vector{Float64}
    f          ::Vector{Float64}
    pi_products::Vector{Matrix{Float64}}
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
        zeros(N, R), zeros(R, coef_dim), zeros(coef_dim),
        zeros(N_B), zeros(N_D),
        zeros(N_B, R), zeros(N_D, R), zeros(N_B, R),
        zeros(N_B), zeros(N_D),
        zeros(n_times, R), zeros(n_pairs, R), zeros(n_pairs, R),
        zeros(n_pairs, R), zeros(n_pairs, R), zeros(N_B, R),
        zeros(n_pairs, R), zeros(n_times, R), zeros(n_pairs, R),
        zeros(n_times, R), zeros(n_times, R),
        zeros(N_D, R), zeros(N_D, 1),
        zeros(N), zeros(N),
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
    N = size(prod_vec, 1)
    R = size(stacked_draws, 2)
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

# --------------------------------------------------------------------------
# 4b. In-place compute_mu!
# --------------------------------------------------------------------------
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

# --------------------------------------------------------------------------
# 5. Market Index Construction
# --------------------------------------------------------------------------
struct Precomp
    b_mkt_idx    ::Vector{Int}
    d_time_enc   ::Vector{Int}
    unique_pairs ::Vector{Tuple{String,String}}
    unique_times ::Vector{String}
    pair_time_enc::Vector{Int}
    pop_weights  ::Vector{Float64}
    B_agg        ::SparseMatrixCSC{Float64,Int}
    D_agg        ::SparseMatrixCSC{Float64,Int}
    PT_agg       ::SparseMatrixCSC{Float64,Int}
    sort_b       ::Vector{Int}
    b_grp_start  ::Vector{Int}
    sort_d       ::Vector{Int}
    d_uval       ::Vector{Int}
    d_grp_start  ::Vector{Int}
    sort_pt      ::Vector{Int}
    pt_uval      ::Vector{Int}
    pt_grp_start ::Vector{Int}
    b_mask       ::BitVector
    d_mask       ::BitVector
    ln_s_data_D  ::Vector{Float64}
    ln_s_data_B_cond::Vector{Float64}
    Z_moments    ::Matrix{Float64}
    X_full       ::Matrix{Float64}
    X_hat        ::Matrix{Float64}
    theta1_valid ::BitVector
    clusters     ::Vector{String}
end

function _unique_with_starts(sorted_vec::Vector{Int})
    isempty(sorted_vec) && return Int[], Int[]
    uval = Int[sorted_vec[1]]
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
    N = nrow(df)
    b_mask = BitVector(Bool.(coalesce.(df.is_B, false)))
    d_mask = .!b_mask
    N_B = sum(b_mask); N_D = sum(d_mask)

    mca_codes = string.(df.mca_code)
    time_ids  = string.(df.time_id)
    pop_total = coalesce.(df.pop_total, 0.0)

    b_mca  = mca_codes[b_mask]; b_time = time_ids[b_mask]
    raw_pairs    = collect(zip(b_mca, b_time))
    unique_pairs = sort(unique(raw_pairs))
    pair_to_idx  = Dict(p => i for (i, p) in enumerate(unique_pairs))
    b_mkt_idx    = [pair_to_idx[(m, t)] for (m, t) in raw_pairs]

    unique_times = sort(unique(time_ids))
    time_to_idx  = Dict(t => i for (i, t) in enumerate(unique_times))
    d_time_enc   = [time_to_idx[t] for t in time_ids[d_mask]]

    n_pairs = length(unique_pairs); n_times = length(unique_times)
    pair_times    = [p[2] for p in unique_pairs]
    pair_time_enc = [time_to_idx[t] for t in pair_times]

    pop_b     = pop_total[b_mask]
    pair_pop  = zeros(n_pairs); for (i, w) in zip(b_mkt_idx, pop_b); pair_pop[i] += w; end
    time_pop  = zeros(n_times); for (i, w) in zip(pair_time_enc, pair_pop); time_pop[i] += w; end
    pop_weights = pair_pop ./ max.(time_pop[pair_time_enc], 1e-30)

    B_agg  = sparse(b_mkt_idx, 1:N_B, ones(N_B), n_pairs, N_B)
    D_agg  = sparse(d_time_enc, 1:N_D, ones(N_D), n_times, N_D)
    PT_agg = sparse(pair_time_enc, 1:n_pairs, pop_weights, n_times, n_pairs)

    sort_b = sortperm(b_mkt_idx); _, b_grp_s = _unique_with_starts(b_mkt_idx[sort_b])
    sort_d = sortperm(d_time_enc); d_uval, d_grp_s = _unique_with_starts(d_time_enc[sort_d])
    sort_pt = sortperm(pair_time_enc); pt_uval, pt_grp_s = _unique_with_starts(pair_time_enc[sort_pt])

    ln_s_D = log.(clamp.(coalesce.(df.share_D, 0.0), 1e-15, Inf))
    ln_s_B = log.(clamp.(coalesce.(df.share_B_cond, 0.0), 1e-15, Inf))

    return Precomp(b_mkt_idx, d_time_enc, unique_pairs, unique_times,
                   pair_time_enc, pop_weights, B_agg, D_agg, PT_agg,
                   sort_b, b_grp_s, sort_d, d_uval, d_grp_s,
                   sort_pt, pt_uval, pt_grp_s,
                   b_mask, d_mask, ln_s_D, ln_s_B,
                   Z, X_full, X_hat, valid, clusters)
end

# --------------------------------------------------------------------------
# 6. Model Shares — in-place
# --------------------------------------------------------------------------
function logsumexp_groups!(result::Matrix{Float64},
                            V, sort_idx, grp_start, uval, R)
    fill!(result, -Inf)
    n_uniq = length(grp_start)
    @inbounds for g in 1:n_uniq
        gidx = grp_start[g]
        gend = g < n_uniq ? grp_start[g+1] - 1 : length(sort_idx)
        target_row = uval[g]
        for c in 1:R
            mx = -Inf
            for ii in gidx:gend
                v = V[sort_idx[ii], c]
                v > mx && (mx = v)
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
    N_B = length(buf.delta_B); N_D = length(buf.delta_D)
    n_pairs = length(pc.unique_pairs); n_times = length(pc.unique_times)

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
                exp(log_outside - jm) + exp(buf.log_sum_B_mkt[i, r] - jm) + exp(buf.log_D_sum_pair[i, r] - jm),
                1e-300))
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
        gidx = pc.pt_grp_start[g]
        gend = g < length(pc.pt_grp_start) ? pc.pt_grp_start[g+1] - 1 : length(pc.sort_pt)
        target = pc.pt_uval[g]
        for r in 1:R
            mx = -Inf
            for ii in gidx:gend
                v = buf.neg_log_denom[pc.sort_pt[ii], r]
                v > mx && (mx = v)
            end
            buf.max_neg_ld[target, r] = mx
        end
    end

    @inbounds for i in 1:n_pairs
        for r in 1:R
            buf.shifted_inv[i, r] = exp(buf.neg_log_denom[i, r] - buf.max_neg_ld[pc.pair_time_enc[i], r])
        end
    end

    mul!(buf.sum_wtd, pc.PT_agg, buf.shifted_inv)

    @inbounds for i in 1:n_times
        for r in 1:R
            buf.log_inv_wtd[i, r] = buf.max_neg_ld[i, r] + log(max(buf.sum_wtd[i, r], 1e-300))
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
        for r in 1:R
            s += exp(buf.log_s_D_r[i, r] - rm)
        end
        buf.s_D[i] = (s / R) * exp(rm)
    end
end

# --------------------------------------------------------------------------
# 7. Anderson(m=20) BLP Contraction — in-place
# --------------------------------------------------------------------------
function blp_contraction!(buf::HotBuffers, delta::Vector{Float64},
                           pc::Precomp, R::Int;
                           tol::Float64=1e-12, max_iter::Int=5000)
    N = length(delta)
    norm_history = Float64[]

    # SQUAREM temporaries (5 × N; much less than old 2 × N × 20 history matrices)
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
        # Step 1: x1 = T(delta)
        T_inplace!(x1, delta);  fevals += 1
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
        (fevals % 50 == 0 || nm < tol) && println("    [SQUAREM eval=$fevals/$max_iter] norm=$(round(nm, sigdigits=4))")
        if nm < tol
            copyto!(delta, x1); return true, fevals, norm_history
        end

        # Step 2: x2 = T(x1)
        T_inplace!(x2, x1);  fevals += 1
        if !all(isfinite, x2)
            copyto!(delta, x1); return false, fevals, norm_history
        end

        nm2 = 0.0
        @inbounds for i in 1:N; a = abs(x2[i]-x1[i]); a > nm2 && (nm2 = a); end
        push!(norm_history, nm2)
        (fevals % 50 == 0 || nm2 < tol) && println("    [SQUAREM eval=$fevals/$max_iter] norm=$(round(nm2, sigdigits=4))")
        if nm2 < tol
            copyto!(delta, x2); return true, fevals, norm_history
        end

        # SQUAREM extrapolation: v = (x2 - x1) - r, α = -‖r‖₂ / ‖v‖₂
        norm_r_sq = 0.0; norm_v_sq = 0.0
        @inbounds for i in 1:N
            v_vec[i] = (x2[i] - x1[i]) - r_vec[i]
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

        # Safety: fall back to x2 if extrapolation diverges
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

# --------------------------------------------------------------------------
# 8. Linear IV
# --------------------------------------------------------------------------
function build_regressor_matrices(df::DataFrame)
    N = nrow(df)
    spread_cols = coalesce.(df.spread_qoq, 0.0)
    x_mat = zeros(N, L_PROD)
    for (i, col) in enumerate(X_COLS)
        col in names(df) && (x_mat[:, i] .= coalesce.(df[!, col], 0.0))
    end
    iv_cols = [c for c in vcat(IV_BLP_LOO, IV_COST, IV_CAPITAL) if c in names(df)]
    Z_mat = zeros(N, length(iv_cols))
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
        k_mask = deposit_types .== k_val
        any(k_mask) || continue
        H_k = H[k_mask, :]; spread_k = spread_cols[k_mask]
        valid = all.(isfinite, eachrow(H_k)) .& isfinite.(spread_k)
        sum(valid) > size(H_k, 2) || continue
        try
            beta_fs = H_k[valid, :] \ spread_k[valid]
            k_hat = copy(spread_k); k_hat[valid] .= H_k[valid, :] * beta_fs
            spread_hat[k_mask, 1] .= k_hat
        catch; end
    end
    return spread_hat
end

function estimate_theta1(delta::Vector{Float64}, pc::Precomp)
    valid = pc.theta1_valid .& isfinite.(delta)
    theta1 = try pc.X_hat[valid, :] \ delta[valid] catch; zeros(size(pc.X_hat, 2)) end
    return theta1, delta .- pc.X_full * theta1
end

# --------------------------------------------------------------------------
# 9. GMM Objective
# --------------------------------------------------------------------------
compute_gmm_moments(xi, pc::Precomp) = vec(Statistics.mean(xi .* pc.Z_moments; dims=1))

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
    G = compute_gmm_moments(xi, pc)
    return dot(G, W * G)
end

# --------------------------------------------------------------------------
# 10. Run BLP for one spec (alt-aware)
# --------------------------------------------------------------------------
function run_blp_for_spec(spec_id::Int, current_alt::String, args)
    println("\n" * "="^60)
    println("  Alt=$current_alt  Specification=$spec_id")
    println("="^60)

    df = try
        load_merged_spec_data(spec_id, current_alt; is_hpc=args["hpc"], local_dir=get(args, "local_dir", nothing))
    catch e
        println("  [!] $e"); return nothing
    end
    nrow(df) == 0 && (println("  [!] Empty."); return nothing)
    println("  Loaded $(nrow(df)) observations")

    sigma_indices, pi_interactions, n_params = build_theta2_structure(args["stage"])
    R = args["R"]; seed = args["seed"]; coef_dim = 1 + L_PROD
    nu_draws = generate_halton_draws(R, coef_dim, seed)
    results = Dict{String,Any}("spec_id" => spec_id, "stage" => args["stage"], "alt" => current_alt)

    if args["stage"] == "logit"
        is_B = Bool.(coalesce.(df.is_B, false))
        share_D_clean = Float64.(coalesce.(df.share_D, 0.0))
        share_B_cond_clean = Float64.(coalesce.(df.share_B_cond, 0.0))
        delta = zeros(nrow(df))
        delta[.!is_B] .= log.(clamp.(share_D_clean[.!is_B], 1e-15, Inf))
        delta[is_B]   .= log.(clamp.(share_B_cond_clean[is_B], 1e-15, Inf))

        spread_cols, x_mat, Z_mat, iv_cols = build_regressor_matrices(df)
        H = hcat(x_mat, Z_mat); dep_types = Int.(coalesce.(df.deposit_type, 0))
        spread_hat = project_endogenous_spreads(spread_cols, H, dep_types)
        X_full = hcat(spread_cols, x_mat); X_hat = hcat(spread_hat, x_mat)
        valid = BitVector(all.(isfinite, eachrow(X_hat)))
        clusters = string.(df.CodConglomeradoPrudencial) .* "_" .*
                   first.(split.(string.(df.time_id), "Q"))
        iv_avail = [c for c in iv_cols if std(replace(coalesce.(df[!, c], 0.0), Inf=>0.0, -Inf=>0.0)) > 1e-10]
        Z_clean = zeros(nrow(df), length(iv_avail))
        for (i, c) in enumerate(iv_avail)
            v = coalesce.(df[!, c], 0.0); replace!(v, Inf=>0.0, -Inf=>0.0)
            Z_clean[:, i] .= v
        end
        pc = build_precomp(df, Z_clean, X_full, X_hat, valid, clusters)
        theta1, xi = estimate_theta1(delta, pc)
        G = compute_gmm_moments(xi, pc); W = Matrix(I(length(iv_avail)) * 1.0); Q = dot(G, W * G)
        merge!(results, Dict("theta1"=>theta1, "theta2"=>Float64[], "delta"=>delta, "xi"=>xi,
                              "Q_value"=>Q, "converged"=>true, "param_names_theta1"=>vcat(["alpha"], X_COLS)))
        println("  alpha=$(round(theta1[1], sigdigits=6))  Q=$(round(Q, sigdigits=6))")
        return results
    end

    # ---- BLP stages ----
    println("  Stage: $(uppercase(args["stage"])) ($n_params params)")
    println("  [HPC] Precomputing arrays + allocating hot buffers...")

    sigma_tbl  = load_sigma_table(; is_hpc=args["hpc"],
                                    local_dir=get(args, "local_dir", nothing))
    demo_draws = generate_demographic_draws(df, R, seed; sigma_table=sigma_tbl)
    d_cols = [c for c in D_COLS if c in names(df)]; D_dim = length(d_cols); N_obs = nrow(df)
    mca_codes = string.(df.mca_code); time_ids = string.(df.time_id)
    unique_keys = sort(collect(keys(demo_draws))); n_keys = length(unique_keys)
    key_to_idx = Dict(k => i for (i, k) in enumerate(unique_keys))

    stacked_draws = zeros(Float64, n_keys, R, D_dim)
    for (i, k) in enumerate(unique_keys); stacked_draws[i, :, :] .= demo_draws[k]; end
    stacked_draws_padded = vcat(stacked_draws, zeros(1, R, D_dim))
    pad_idx = n_keys + 1
    obs_key_idx = [get(key_to_idx, (mca_codes[i], time_ids[i]), pad_idx) for i in 1:N_obs]

    prod_vec = zeros(N_obs, coef_dim)
    spreads = coalesce.(df.spread_qoq, 0.0)
    dep_types = Int.(coalesce.(df.deposit_type, 0))
    prod_vec[:, 1] .= spreads
    for (i, col) in enumerate(X_COLS)
        col in names(df) && (prod_vec[:, 1+i] .= coalesce.(df[!, col], 0.0))
    end

    spread_cols, x_mat, Z_mat, iv_cols = build_regressor_matrices(df)
    iv_avail = [c for c in iv_cols if std(replace(coalesce.(df[!, c], 0.0), Inf=>0.0, -Inf=>0.0)) > 1e-10]
    Z = zeros(N_obs, length(iv_avail))
    for (i, c) in enumerate(iv_avail)
        v = coalesce.(df[!, c], 0.0); replace!(v, Inf=>0.0, -Inf=>0.0); Z[:, i] .= v
    end
    W = try inv(Z' * Z ./ N_obs) catch; Matrix(I(length(iv_avail)) * 1.0) end

    H = hcat(x_mat, Z_mat); spread_hat = project_endogenous_spreads(spread_cols, H, dep_types)
    X_full = hcat(spread_cols, x_mat); X_hat = hcat(spread_hat, x_mat)
    valid = BitVector(all.(isfinite, eachrow(X_hat)))
    clusters = string.(df.CodConglomeradoPrudencial) .* "_" .*
               first.(split.(string.(df.time_id), "Q"))
    pc = build_precomp(df, Z, X_full, X_hat, valid, clusters)

    N_B = sum(pc.b_mask); N_D = sum(pc.d_mask)
    n_pairs = length(pc.unique_pairs); n_times = length(pc.unique_times)

    buf = allocate_hot_buffers(N_obs, N_B, N_D, R, n_pairs, n_times, coef_dim,
                                length(pi_interactions))
    precompute_pi_products!(buf, prod_vec, stacked_draws_padded, obs_key_idx,
                             pi_interactions, coef_dim)

    _, out_dir = get_paths(args["hpc"])
    prev_stages = Dict("full" => "sigma", "extended" => "full")
    theta2_0 = nothing
    if args["stage"] in keys(prev_stages)
        pp = joinpath(out_dir, "blp_checkpoint_spec_5_$(current_alt)_$(spec_id)_$(prev_stages[args["stage"]]).jls")
        if isfile(pp)
            try t2p = deserialize(pp)["theta2_star"]
                theta2_0 = zeros(n_params); theta2_0[1:length(t2p)] .= t2p
                println("  [WARM-START] from $pp")
            catch e; end
        end
    end
    theta2_0 === nothing && (theta2_0 = randn(MersenneTwister(seed), n_params) .* 0.01)

    delta_work = zeros(N_obs)
    delta_work[pc.d_mask] .= pc.ln_s_data_D[pc.d_mask]
    delta_work[pc.b_mask] .= pc.ln_s_data_B_cond[pc.b_mask]

    if get(args, "dry_run", false)
        sv, pv = unpack_theta2(theta2_0, sigma_indices, pi_interactions)
        compute_mu!(buf, prod_vec, nu_draws, sv, sigma_indices, pv, R, coef_dim)
        t0 = time()
        blp_contraction!(buf, copy(delta_work), pc, R; tol=args["tol_inner"], max_iter=10)
        println("  [DRY RUN] $(round(time()-t0, digits=2))s/10 iters. Exiting.")
        return Dict("dry_run" => true)
    end

    outer_iter = Ref(0)
    function obj_fn(t2)
        val = gmm_objective!(buf, t2, prod_vec, nu_draws,
                              sigma_indices, pi_interactions, R, coef_dim,
                              W, args["tol_inner"], args["max_inner"],
                              delta_work, pc)
        outer_iter[] += 1
        outer_iter[] % 10 == 0 && log_status("  [$current_alt OUTER=$(outer_iter[])] theta2=$(round.(t2, digits=4))")
        return val
    end

    r = optimize(obj_fn, fill(-15.0, n_params), fill(15.0, n_params), theta2_0,
                 Fminbox(LBFGS()), Optim.Options(iterations=500, f_reltol=args["tol_outer"], show_trace=false))
    t2s = Optim.minimizer(r)
    println("  Converged: $(Optim.converged(r))  Q=$(round(Optim.minimum(r), sigdigits=6))")

    sv, pv = unpack_theta2(t2s, sigma_indices, pi_interactions)
    compute_mu!(buf, prod_vec, nu_draws, sv, sigma_indices, pv, R, coef_dim)
    delta_final = copy(delta_work)
    blp_contraction!(buf, delta_final, pc, R; tol=args["tol_inner"], max_iter=args["max_inner"])
    t1s, xis = estimate_theta1(delta_final, pc)

    merge!(results, Dict("theta1"=>t1s, "theta2"=>t2s, "delta"=>delta_final, "xi"=>xis,
                          "Q_value"=>Optim.minimum(r), "converged"=>Optim.converged(r),
                          "param_names_theta1"=>vcat(["alpha"], X_COLS)))
    println("  alpha=$(round(t1s[1], sigdigits=6))")

    mkpath(out_dir)
    chk = joinpath(out_dir, "blp_checkpoint_spec_5_$(current_alt)_$(spec_id)_$(args["stage"]).jls")
    try serialize(chk, Dict("theta2_star"=>t2s, "delta_star"=>delta_final))
        println("  [CHECKPOINT] $(basename(chk))")
    catch e; end
    return results
end

# --------------------------------------------------------------------------
# 11. Main (with --alt outer loop)
# --------------------------------------------------------------------------
function parse_args_blp()
    s = ArgParseSettings(description="BLP Demand Estimation Loop — Spec 5 HPC (with --alt)")
    @add_arg_table! s begin
        "--spec";    arg_type=String; default="12"
        "--stage";   arg_type=String; default="logit"
        "--alt";     arg_type=String; default="all"
        "--R";       arg_type=Int;    default=2000
        "--seed";    arg_type=Int;    default=42
        "--tol-inner"; arg_type=Float64; default=1e-12; dest_name="tol_inner"
        "--max-inner"; arg_type=Int;     default=2000;  dest_name="max_inner"
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
    args = parse_args_blp()
    spec_ids = args["spec"] == "all" ? collect(1:12) : [parse(Int, args["spec"])]
    alts_to_process = args["alt"] == "all" ? ALT_LIST : [args["alt"]]
    alts_to_process ⊆ ALT_CHOICES || error("Unknown --alt value(s): $(setdiff(alts_to_process, ALT_CHOICES))")

    log_status("BLP Demand Estimation – Spec$(SPEC_NUM) HPC – START")
    log_status("  Stage: $(args["stage"]) | Alts: $alts_to_process | Specs: $spec_ids | R=$(args["R"])")
    log_status("  BLAS threads: $(BLAS.get_num_threads()) | Julia threads: $(Threads.nthreads())")
    _, out_dir = get_paths(args["hpc"]; local_dir=get(args, "local_dir", nothing)); mkpath(out_dir)
    stages = args["stage"] == "sequence" ? ["logit", "sigma", "full", "extended"] : [args["stage"]]

    for current_alt in alts_to_process
        log_status("  === Alt: $current_alt ===")
        for cs in stages
            args["stage"] = cs; all_res = Dict{Int,Any}(); lk = ReentrantLock()
            println("  Running $(length(spec_ids)) specs  alt=$current_alt  stage=$cs  threads=$(Threads.nthreads())")
            Threads.@threads for sp in spec_ids
                res = try run_blp_for_spec(sp, current_alt, args) catch e
                    println("  [!] Spec $sp alt=$current_alt: $e"); nothing
                end
                res === nothing && continue; get(res, "dry_run", false) && continue
                pkl = joinpath(out_dir, "blp_results_spec_5_$(current_alt)_$(sp)_$(cs).jls")
                serialize(pkl, res); println("  Saved: $(basename(pkl))")
                lock(lk) do
                    all_res[sp] = Dict("Q_value"=>get(res, "Q_value", 0.0),
                                       "converged"=>get(res, "converged", true),
                                       "theta1_alpha"=>isempty(get(res, "theta1", Float64[])) ? [] : [res["theta1"][1]],
                                       "theta2"=>get(res, "theta2", Float64[]),
                                       "stage"=>cs, "alt"=>current_alt)
                end
            end
            sp_path = joinpath(out_dir, "blp_summary_$(current_alt)_5_$(cs).json")
            open(sp_path, "w") do f; JSON3.write(f, all_res); end
            log_status("[DONE] alt=$current_alt  stage=$cs  Summary: $(basename(sp_path))")
        end
    end
end

main()
