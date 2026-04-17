"""
estimation_4_demand_2_loop_ju_hpc.jl
=====================================
HPC-optimised version of estimation_4_demand_2_loop_ju.jl.

Optimisations over the local version (requires 750 GB+ RAM):
  1. Pre-allocated HotBuffers struct — eliminates ALL GC allocation in the
     inner contraction loop (mu, V_B, V_D, delta_new, f, logsumexp scratch).
  2. In-place BLAS via mul! for the main (N,R) dgemm — zero intermediate alloc.
  3. Pre-computed Pi interaction products (N,R each, ~3 GB) — cached once since
     they depend only on fixed demographic draws × prod_vec, not on theta2/delta.
  4. @inbounds on all inner loops in logsumexp_groups and contraction.
  5. Chunking path removed — full (N,R) always fits at 750 GB.
  6. BLAS thread count set explicitly at startup.

Usage (HPC):
  julia --threads=32 estimation_4_demand_2_loop_ju_hpc.jl --spec 12 --stage sequence --R 2000 --hpc
"""

using Parquet2, DataFrames, SparseArrays, LinearAlgebra, Statistics
using Random, Optim, QuasiMonteCarlo, Distributions
using JSON3, Serialization, ArgParse, Printf, Dates

# ── Set BLAS threads to match Julia threads ──────────────────────────────────
BLAS.set_num_threads(Threads.nthreads())

# --------------------------------------------------------------------------
# 0. Constants
# --------------------------------------------------------------------------
const SPEC_NUM   = 4
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
        input_dir  = "/home/pf382/dep_comp/data/input"
        output_dir = "/home/pf382/dep_comp/data/output"
    else
        if local_dir !== nothing
            data_dir = local_dir
        else
            _root    = dirname(dirname(dirname(abspath(@__FILE__))))
            data_dir = joinpath(_root, "BCB", "Egan_et_al_2025_Rep", "processed")
        end
        input_dir  = joinpath(data_dir, "ESTIMATION_OUTPUT", "DEMAND_PREP")
        output_dir = joinpath(data_dir, "ESTIMATION_OUTPUT", "BLP_RESULTS")
    end
    return input_dir, output_dir
end

function log_status(msg::String)
    stamped = "[$(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))] $msg"
    lock(_log_lock) do
        push!(_log_buf, stamped)
    end
    println(stamped)
    flush(stdout)
end

# --------------------------------------------------------------------------
# 1. Data Loading
# --------------------------------------------------------------------------
function load_merged_spec_data(spec_id::Int; is_hpc::Bool=false, local_dir=nothing)
    input_dir, _ = get_paths(is_hpc; local_dir=local_dir)
    path = joinpath(input_dir, "demand_$(SPEC_NUM)_final_spec_$(spec_id).parquet")
    isfile(path) || error("Missing: $path")
    df = DataFrame(Parquet2.Dataset(path); copycols=true)
    return df
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

function generate_demographic_draws(df::DataFrame, R::Int, seed::Int)
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
        draws[key] = mu .+ (nat_std .* 0.1) .* randn(rng, D, R) |> transpose |> Matrix
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
"""
All (N,R) and (N,) sized scratch arrays pre-allocated once.
Eliminates GC pressure in the contraction hot loop.
"""
mutable struct HotBuffers
    # mu workspace
    mu         ::Matrix{Float64}   # (N, R)
    sigma_nu   ::Matrix{Float64}   # (R, coef_dim)
    sigma_diag ::Vector{Float64}   # (coef_dim,)
    # shares workspace
    delta_B    ::Vector{Float64}   # (N_B,)
    delta_D    ::Vector{Float64}   # (N_D,)
    V_B        ::Matrix{Float64}   # (N_B, R)
    V_D        ::Matrix{Float64}   # (N_D, R)
    q_B        ::Matrix{Float64}   # (N_B, R)
    s_B        ::Vector{Float64}   # (N_B,)
    s_D        ::Vector{Float64}   # (N_D,)
    log_sum_D_time ::Matrix{Float64}  # (n_times, R)
    log_D_sum_pair ::Matrix{Float64}  # (n_pairs, R)
    log_sum_B_mkt  ::Matrix{Float64}  # (n_pairs, R)
    joint_max      ::Matrix{Float64}  # (n_pairs, R)
    log_denom      ::Matrix{Float64}  # (n_pairs, R)
    log_denom_B    ::Matrix{Float64}  # (N_B, R)
    neg_log_denom  ::Matrix{Float64}  # (n_pairs, R)
    max_neg_ld     ::Matrix{Float64}  # (n_times, R)
    shifted_inv    ::Matrix{Float64}  # (n_pairs, R)
    sum_wtd        ::Matrix{Float64}  # (n_times, R)
    log_inv_wtd    ::Matrix{Float64}  # (n_times, R)
    log_s_D_r      ::Matrix{Float64}  # (N_D, R)
    row_max        ::Matrix{Float64}  # (N_D, 1)
    # contraction workspace
    delta_new  ::Vector{Float64}   # (N,)
    f          ::Vector{Float64}   # (N,)
    # pre-computed Pi interaction products (N, R) each — independent of theta2
    pi_products::Vector{Matrix{Float64}}
end

function allocate_hot_buffers(N::Int, N_B::Int, N_D::Int, R::Int,
                               n_pairs::Int, n_times::Int, coef_dim::Int,
                               n_pi::Int)::HotBuffers
    println("  [HPC] Pre-allocating hot buffers...")
    total_gb = (
        N*R + R*coef_dim + coef_dim +  # mu workspace
        N_B + N_D +                      # delta_B, delta_D
        N_B*R + N_D*R + N_B*R +         # V_B, V_D, q_B
        N_B + N_D +                      # s_B, s_D
        n_times*R + n_pairs*R*2 +        # log_sum_D_time, log_D_sum_pair, log_sum_B_mkt
        n_pairs*R*3 +                    # joint_max, log_denom, neg_log_denom
        N_B*R +                          # log_denom_B
        n_times*R*3 +                    # max_neg_ld, sum_wtd, log_inv_wtd
        n_pairs*R +                      # shifted_inv
        N_D*R + N_D +                    # log_s_D_r, row_max
        N*2 +                            # delta_new, f
        n_pi*N*R                         # pi_products
    ) * 8 / 1e9
    println("  [HPC] Total buffer allocation: $(round(total_gb, digits=2)) GB")

    return HotBuffers(
        zeros(N, R),            # mu
        zeros(R, coef_dim),     # sigma_nu
        zeros(coef_dim),        # sigma_diag
        zeros(N_B),             # delta_B
        zeros(N_D),             # delta_D
        zeros(N_B, R),          # V_B
        zeros(N_D, R),          # V_D
        zeros(N_B, R),          # q_B
        zeros(N_B),             # s_B
        zeros(N_D),             # s_D
        zeros(n_times, R),      # log_sum_D_time
        zeros(n_pairs, R),      # log_D_sum_pair
        zeros(n_pairs, R),      # log_sum_B_mkt
        zeros(n_pairs, R),      # joint_max
        zeros(n_pairs, R),      # log_denom
        zeros(N_B, R),          # log_denom_B
        zeros(n_pairs, R),      # neg_log_denom
        zeros(n_times, R),      # max_neg_ld
        zeros(n_pairs, R),      # shifted_inv
        zeros(n_times, R),      # sum_wtd
        zeros(n_times, R),      # log_inv_wtd
        zeros(N_D, R),          # log_s_D_r
        zeros(N_D, 1),          # row_max
        zeros(N),               # delta_new
        zeros(N),               # f
        Matrix{Float64}[],      # pi_products (filled later)
    )
end

"""Pre-compute Pi interaction products: prod_vec[:, cidx] .* stacked_draws[obs_key_idx, :, didx]
Each is (N, R) and fixed across theta2 iterations."""
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
            # prod_vec[:, cidx] is (N,), stacked_draws[obs_key_idx, :, didx] is (N, R)
            prod = @view(prod_vec[:, cidx]) .* stacked_draws[obs_key_idx, :, didx]  # (N, R)
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
"""Compute mu in-place using pre-allocated buffers and mul!."""
function compute_mu!(buf::HotBuffers,
                     prod_vec::Matrix{Float64},
                     nu_draws::Matrix{Float64},
                     sigma_vals::Vector{Float64},
                     sigma_indices::Vector{Int},
                     pi_vals::Vector{Float64},
                     R::Int, coef_dim::Int)
    # Build sigma_diag
    fill!(buf.sigma_diag, 0.0)
    @inbounds for (pos, cidx) in enumerate(sigma_indices)
        cidx <= coef_dim && (buf.sigma_diag[cidx] = sigma_vals[pos])
    end

    # sigma_nu = nu_draws[:, 1:coef_dim] .* sigma_diag'  (R, coef_dim)
    @inbounds for j in 1:coef_dim
        sd = buf.sigma_diag[j]
        for r in 1:R
            buf.sigma_nu[r, j] = nu_draws[r, j] * sd
        end
    end

    # mu = prod_vec * sigma_nu'  via in-place BLAS: (N, coef_dim) × (coef_dim, R) → (N, R)
    mul!(buf.mu, prod_vec, buf.sigma_nu')

    # Add pre-computed Pi products (scaled by pi_vals)
    @inbounds for (pi_idx, pi_prod) in enumerate(buf.pi_products)
        size(pi_prod, 1) == 0 && continue
        pv = pi_vals[pi_idx]
        buf.mu .+= pv .* pi_prod
    end
end

# --------------------------------------------------------------------------
# 5. Market Index Construction (unchanged)
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

function build_precomp(df::DataFrame, Z::Matrix{Float64},
                       X_full::Matrix{Float64}, X_hat::Matrix{Float64},
                       valid::BitVector, clusters::Vector{String})::Precomp
    N = nrow(df)
    is_B_raw  = df.is_B
    b_mask    = BitVector(Bool.(coalesce.(is_B_raw, false)))
    d_mask    = .!b_mask
    N_B       = sum(b_mask)
    N_D       = sum(d_mask)

    mca_codes = string.(df.mca_code)
    time_ids  = string.(df.time_id)
    pop_total = coalesce.(df.pop_total, 0.0)

    b_mca  = mca_codes[b_mask]
    b_time = time_ids[b_mask]
    raw_pairs     = collect(zip(b_mca, b_time))
    unique_pairs  = sort(unique(raw_pairs))
    pair_to_idx   = Dict(p => i for (i, p) in enumerate(unique_pairs))
    b_mkt_idx     = [pair_to_idx[(m, t)] for (m, t) in raw_pairs]

    unique_times  = sort(unique(time_ids))
    time_to_idx   = Dict(t => i for (i, t) in enumerate(unique_times))
    d_time_enc    = [time_to_idx[t] for t in time_ids[d_mask]]

    n_pairs = length(unique_pairs)
    n_times = length(unique_times)

    pair_times    = [p[2] for p in unique_pairs]
    pair_time_enc = [time_to_idx[t] for t in pair_times]

    pop_b         = pop_total[b_mask]
    pair_pop      = zeros(n_pairs)
    for (i, w) in zip(b_mkt_idx, pop_b); pair_pop[i] += w; end
    time_pop      = zeros(n_times)
    for (i, w) in zip(pair_time_enc, pair_pop); time_pop[i] += w; end
    pop_weights   = pair_pop ./ max.(time_pop[pair_time_enc], 1e-30)

    B_agg  = sparse(b_mkt_idx,  1:N_B,    ones(N_B),  n_pairs, N_B)
    D_agg  = sparse(d_time_enc, 1:N_D,    ones(N_D),  n_times, N_D)
    PT_agg = sparse(pair_time_enc, 1:n_pairs, pop_weights, n_times, n_pairs)

    sort_b      = sortperm(b_mkt_idx)
    _, b_grp_s  = _unique_with_starts(b_mkt_idx[sort_b])
    sort_d      = sortperm(d_time_enc)
    d_uval, d_grp_s = _unique_with_starts(d_time_enc[sort_d])
    sort_pt     = sortperm(pair_time_enc)
    pt_uval, pt_grp_s = _unique_with_starts(pair_time_enc[sort_pt])

    ln_s_D    = log.(clamp.(coalesce.(df.share_D,      0.0), 1e-15, Inf))
    ln_s_B    = log.(clamp.(coalesce.(df.share_B_cond, 0.0), 1e-15, Inf))

    return Precomp(b_mkt_idx, d_time_enc, unique_pairs, unique_times,
                   pair_time_enc, pop_weights,
                   B_agg, D_agg, PT_agg,
                   sort_b, b_grp_s,
                   sort_d, d_uval, d_grp_s,
                   sort_pt, pt_uval, pt_grp_s,
                   b_mask, d_mask, ln_s_D, ln_s_B,
                   Z, X_full, X_hat, valid, clusters)
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

# --------------------------------------------------------------------------
# 6. Model Shares — in-place version
# --------------------------------------------------------------------------
"""Stable log-sum-exp per group, writing into pre-allocated result matrix."""
function logsumexp_groups!(result::Matrix{Float64},
                            V, sort_idx, grp_start, uval, R)
    fill!(result, -Inf)
    n_uniq = length(grp_start)
    @inbounds for g in 1:n_uniq
        gidx = grp_start[g]
        gend = g < n_uniq ? grp_start[g+1] - 1 : length(sort_idx)
        target_row = uval[g]
        # Find max per column for this group
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
    N_B = length(buf.delta_B)
    N_D = length(buf.delta_D)
    n_pairs = length(pc.unique_pairs)
    n_times = length(pc.unique_times)

    # Extract B/D slices in-place
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

    # D-firm log-sum per time period
    logsumexp_groups!(buf.log_sum_D_time, buf.V_D, pc.sort_d, pc.d_grp_start, pc.d_uval, R)
    @inbounds for i in 1:n_pairs
        for r in 1:R
            buf.log_D_sum_pair[i, r] = buf.log_sum_D_time[pc.pair_time_enc[i], r]
        end
    end

    # B-firm log-sum per market
    logsumexp_groups!(buf.log_sum_B_mkt, buf.V_B, pc.sort_b, pc.b_grp_start, collect(1:n_pairs), R)

    # Joint log-denominator
    log_outside = 0.0  # log(1.0)
    @inbounds for i in 1:n_pairs
        for r in 1:R
            jm = max(buf.log_sum_B_mkt[i, r], buf.log_D_sum_pair[i, r], log_outside)
            buf.joint_max[i, r] = jm
            buf.log_denom[i, r] = jm + log(max(
                exp(log_outside - jm) + exp(buf.log_sum_B_mkt[i, r] - jm) + exp(buf.log_D_sum_pair[i, r] - jm),
                1e-300))
        end
    end

    # B-firm shares
    @inbounds for i in 1:N_B
        mkt = pc.b_mkt_idx[i]
        for r in 1:R
            buf.q_B[i, r] = exp(buf.V_B[i, r] - buf.log_denom[mkt, r])
        end
        s = 0.0
        for r in 1:R; s += buf.q_B[i, r]; end
        buf.s_B[i] = s / R
    end

    # D-firm national shares
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
# 8. Linear IV  (Eq-A5)
# --------------------------------------------------------------------------
function build_regressor_matrices(df::DataFrame)
    N     = nrow(df)
    spread_cols = coalesce.(df.spread_qoq, 0.0)
    x_mat = zeros(N, L_PROD)
    for (i, col) in enumerate(X_COLS)
        if col in names(df)
            x_mat[:, i] .= coalesce.(df[!, col], 0.0)
        end
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
        k_mask = deposit_types .== k_val
        any(k_mask) || continue
        H_k      = H[k_mask, :]
        spread_k = spread_cols[k_mask]
        valid    = all.(isfinite, eachrow(H_k)) .& isfinite.(spread_k)
        sum(valid) > size(H_k, 2) || continue
        try
            beta_fs   = H_k[valid, :] \ spread_k[valid]
            k_hat     = copy(spread_k)
            k_hat[valid] .= H_k[valid, :] * beta_fs
            spread_hat[k_mask, 1] .= k_hat
        catch; end
    end
    return spread_hat
end

function estimate_theta1(delta::Vector{Float64}, pc::Precomp)
    X_full = pc.X_full
    X_hat  = pc.X_hat
    valid  = pc.theta1_valid .& isfinite.(delta)
    X_v    = X_hat[valid, :]
    d_v    = delta[valid]
    theta1 = try X_v \ d_v catch; zeros(size(X_hat, 2)) end
    xi     = delta .- X_full * theta1
    return theta1, xi
end

# --------------------------------------------------------------------------
# 9. GMM Objective
# --------------------------------------------------------------------------
function compute_gmm_moments(xi::Vector{Float64}, pc::Precomp)
    Z = pc.Z_moments
    return vec(Statistics.mean(xi .* Z; dims=1))
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

    # Compute mu in-place
    compute_mu!(buf, prod_vec, nu_draws, sigma_vals, sigma_indices, pi_vals, R, coef_dim)

    # Run contraction in-place (modifies delta)
    converged, n_iter, _ = blp_contraction!(buf, delta, pc, R;
                                             tol=tol_inner, max_iter=max_inner)
    converged || println("  [!] Inner loop did not converge in $n_iter iterations")

    theta1, xi = estimate_theta1(delta, pc)
    G          = compute_gmm_moments(xi, pc)
    return dot(G, W * G)
end

# --------------------------------------------------------------------------
# 10. Run BLP for one spec
# --------------------------------------------------------------------------
function run_blp_for_spec(spec_id::Int, args)
    println("\n" * "="^60)
    println("  Specification $spec_id")
    println("="^60)

    df = try
        load_merged_spec_data(spec_id; is_hpc=args["hpc"], local_dir=get(args,"local_dir",nothing))
    catch e
        println("  [!] No data for spec $spec_id: $e. Skipping.")
        return nothing
    end
    nrow(df) == 0 && (println("  [!] Empty dataframe. Skipping."); return nothing)

    println("  Merged panel loaded: $(nrow(df)) observations")

    sigma_indices, pi_interactions, n_params = build_theta2_structure(args["stage"])
    R        = args["R"]
    seed     = args["seed"]
    coef_dim = 1 + L_PROD

    nu_draws = generate_halton_draws(R, coef_dim, seed)

    results = Dict{String,Any}("spec_id" => spec_id, "stage" => args["stage"])

    if args["stage"] == "logit"
        println("  Stage: LOGIT (theta2 = 0)")
        is_B  = Bool.(coalesce.(df.is_B, false))
        share_D_clean     = Float64.(coalesce.(df.share_D,      0.0))
        share_B_cond_clean= Float64.(coalesce.(df.share_B_cond, 0.0))
        delta = zeros(nrow(df))
        delta[.!is_B] .= log.(clamp.(share_D_clean[.!is_B], 1e-15, Inf))
        delta[is_B]   .= log.(clamp.(share_B_cond_clean[is_B], 1e-15, Inf))

        spread_cols, x_mat, Z_mat, iv_cols = build_regressor_matrices(df)
        H          = hcat(x_mat, Z_mat)
        dep_types  = Int.(coalesce.(df.deposit_type, 0))
        spread_hat = project_endogenous_spreads(spread_cols, H, dep_types)
        X_full     = hcat(spread_cols, x_mat)
        X_hat      = hcat(spread_hat,  x_mat)
        valid      = BitVector(all.(isfinite, eachrow(X_hat)))
        clusters   = string.(df.CodConglomeradoPrudencial) .* "_" .*
                     first.(split.(string.(df.time_id), "Q"))
        iv_avail   = [c for c in iv_cols if std(replace(coalesce.(df[!, c], 0.0), Inf=>0.0, -Inf=>0.0)) > 1e-10]
        Z_clean    = zeros(nrow(df), length(iv_avail))
        for (i, c) in enumerate(iv_avail)
            v = coalesce.(df[!, c], 0.0); replace!(v, Inf=>0.0, -Inf=>0.0)
            Z_clean[:, i] .= v
        end
        pc_logit = Precomp(Int[], Int[], Tuple{String,String}[], String[], Int[], Float64[],
                           sparse(zeros(0,0)), sparse(zeros(0,0)), sparse(zeros(0,0)),
                           Int[], Int[], Int[], Int[], Int[], Int[], Int[], Int[],
                           BitVector(Bool.(coalesce.(df.is_B, false))),
                           .!BitVector(Bool.(coalesce.(df.is_B, false))),
                           log.(clamp.(share_D_clean,      1e-15, Inf)),
                           log.(clamp.(share_B_cond_clean, 1e-15, Inf)),
                           Z_clean, X_full, X_hat, valid, clusters)

        theta1, xi = estimate_theta1(delta, pc_logit)
        G  = compute_gmm_moments(xi, pc_logit)
        W  = Matrix(I(length(iv_avail)) * 1.0)
        Q  = dot(G, W * G)
        results["theta1"]             = theta1
        results["theta2"]             = Float64[]
        results["delta"]              = delta
        results["xi"]                 = xi
        results["Q_value"]            = Q
        results["converged"]          = true
        results["param_names_theta1"] = vcat(["alpha"], X_COLS)
        println("  theta1 (alpha): $(round(theta1[1], sigdigits=6))")
        println("  Q(0) = $(round(Q, sigdigits=6))")
        return results
    end

    # ---- BLP stages ----
    println("  Stage: $(uppercase(args["stage"])) ($n_params parameters)")
    println("  [HPC] Precomputing arrays + allocating hot buffers...")

    demo_draws = generate_demographic_draws(df, R, seed)
    d_cols = [c for c in D_COLS if c in names(df)]
    D_dim  = length(d_cols)
    N_obs  = nrow(df)

    mca_codes = string.(df.mca_code)
    time_ids  = string.(df.time_id)
    unique_keys = sort(collect(keys(demo_draws)))
    n_keys      = length(unique_keys)
    key_to_idx  = Dict(k => i for (i, k) in enumerate(unique_keys))
    
    stacked_draws = zeros(Float64, n_keys, R, D_dim)
    for (i, k) in enumerate(unique_keys)
        stacked_draws[i, :, :] .= demo_draws[k]
    end
    
    zero_draw  = zeros(Float64, 1, R, D_dim)
    stacked_draws_padded = vcat(stacked_draws, zero_draw)
    pad_idx    = n_keys + 1

    obs_key_idx = [get(key_to_idx, (mca_codes[i], time_ids[i]), pad_idx) for i in 1:N_obs]

    prod_vec  = zeros(N_obs, coef_dim)
    spreads   = coalesce.(df.spread_qoq, 0.0)
    dep_types = Int.(coalesce.(df.deposit_type, 0))
    prod_vec[:, 1] .= spreads
    for (i, col) in enumerate(X_COLS)
        col in names(df) && (prod_vec[:, 1+i] .= coalesce.(df[!, col], 0.0))
    end

    spread_cols, x_mat, Z_mat, iv_cols = build_regressor_matrices(df)
    iv_avail = [c for c in iv_cols if std(replace(coalesce.(df[!, c], 0.0), Inf=>0.0, -Inf=>0.0)) > 1e-10]
    Z = zeros(N_obs, length(iv_avail))
    for (i, c) in enumerate(iv_avail)
        v = coalesce.(df[!, c], 0.0); replace!(v, Inf=>0.0, -Inf=>0.0)
        Z[:, i] .= v
    end
    W = try inv(Z' * Z ./ N_obs) catch; Matrix(I(length(iv_avail)) * 1.0) end

    H            = hcat(x_mat, Z_mat)
    spread_hat   = project_endogenous_spreads(spread_cols, H, dep_types)
    X_full       = hcat(spread_cols, x_mat)
    X_hat        = hcat(spread_hat,  x_mat)
    valid        = BitVector(all.(isfinite, eachrow(X_hat)))
    clusters     = string.(df.CodConglomeradoPrudencial) .* "_" .*
                   first.(split.(string.(df.time_id), "Q"))
    pc = build_precomp(df, Z, X_full, X_hat, valid, clusters)

    N_B = sum(pc.b_mask)
    N_D = sum(pc.d_mask)
    n_pairs = length(pc.unique_pairs)
    n_times = length(pc.unique_times)

    # Allocate hot buffers
    buf = allocate_hot_buffers(N_obs, N_B, N_D, R, n_pairs, n_times, coef_dim,
                                length(pi_interactions))
    precompute_pi_products!(buf, prod_vec, stacked_draws_padded, obs_key_idx,
                             pi_interactions, coef_dim)

    # Starting values
    _, out_dir = get_paths(args["hpc"])
    prev_stages = Dict("full" => "sigma", "extended" => "full")
    theta2_0    = nothing
    if args["stage"] in keys(prev_stages)
        prev_path = joinpath(out_dir, "blp_checkpoint_spec_$(spec_id)_$(prev_stages[args["stage"]]).jls")
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

    println("  Outer minimisation: $(args["method"])")
    println("  Inner tolerance: $(args["tol_inner"])")

    # Working delta vector (mutated in-place across outer iterations)
    delta_work = zeros(N_obs)
    delta_work[pc.d_mask] .= pc.ln_s_data_D[pc.d_mask]
    delta_work[pc.b_mask] .= pc.ln_s_data_B_cond[pc.b_mask]

    if get(args, "dry_run", false)
        println("  [DRY RUN] Running 10 contraction iterations for timing...")
        sv, pv = unpack_theta2(theta2_0, sigma_indices, pi_interactions)
        compute_mu!(buf, prod_vec, nu_draws, sv, sigma_indices, pv, R, coef_dim)
        t0 = time()
        blp_contraction!(buf, copy(delta_work), pc, R; tol=args["tol_inner"], max_iter=10)
        elapsed = time() - t0
        println("  [DRY RUN] 10 iterations in $(round(elapsed, digits=2))s ($(round(elapsed/10, digits=3)) s/iter).")
        println("  [DRY RUN] Exiting early.")
        return Dict("dry_run" => true)
    end

    outer_iter = Ref(0)
    function obj_fn(t2)
        # Reset delta_work to last-known good values each outer call
        val = gmm_objective!(buf, t2, prod_vec, nu_draws,
                              sigma_indices, pi_interactions, R, coef_dim,
                              W, args["tol_inner"], args["max_inner"],
                              delta_work, pc)
        outer_iter[] += 1
        outer_iter[] % 10 == 0 && log_status("  [OUTER iter=$(outer_iter[])] theta2=$(round.(t2, digits=4))")
        return val
    end

    lo = fill(-15.0, n_params)
    hi = fill( 15.0, n_params)
    result = optimize(obj_fn, lo, hi, theta2_0, Fminbox(LBFGS()),
                      Optim.Options(iterations=500, f_reltol=args["tol_outer"], show_trace=false))

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
    chk_path = joinpath(out_dir, "blp_checkpoint_spec_$(spec_id)_$(args["stage"]).jls")
    try serialize(chk_path, Dict("theta2_star" => theta2_star, "delta_star" => delta_final))
        println("  [CHECKPOINT] Saved $(basename(chk_path))")
    catch e; println("  [CHECKPOINT-WARN] $e") end

    return results
end

# --------------------------------------------------------------------------
# 11. Main
# --------------------------------------------------------------------------
function parse_args_blp()
    s = ArgParseSettings(description="BLP Demand Estimation Loop — HPC Optimised")
    @add_arg_table! s begin
        "--spec";    arg_type=String;  default="12"
        "--stage";   arg_type=String;  default="logit"
        "--R";       arg_type=Int;     default=2000
        "--seed";    arg_type=Int;     default=42
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

    log_status("BLP Demand Estimation Loop (HPC) — START")
    log_status("  Stage: $(args["stage"]) | Specs: $spec_ids")
    log_status("  R=$(args["R"]) | seed=$(args["seed"]) | method=$(args["method"]) | HPC=$(args["hpc"])")
    log_status("  tol_inner=$(args["tol_inner"]) | tol_outer=$(args["tol_outer"]) | threads=$(Threads.nthreads())")
    log_status("  BLAS threads: $(BLAS.get_num_threads())")

    _, out_dir = get_paths(args["hpc"]; local_dir=get(args, "local_dir", nothing))
    mkpath(out_dir)
    println("\n  [DIAGNOSTIC] Output Directory: $out_dir")

    stages_to_run = args["stage"] == "sequence" ? ["logit", "sigma", "full", "extended"] : [args["stage"]]

    for current_stage in stages_to_run
        args["stage"] = current_stage
        all_results   = Dict{Int,Any}()
        lk            = ReentrantLock()

        println("  Starting execution of $(length(spec_ids)) specs for stage '$current_stage' with $(Threads.nthreads()) threads...")

        Threads.@threads for sp in spec_ids
            res = try run_blp_for_spec(sp, args) catch e
                println("  [!] Spec $sp failed: $e"); nothing
            end
            res === nothing && continue
            get(res, "dry_run", false) && continue
            out_pkl = joinpath(out_dir, "blp_results_spec_$(sp)_$(current_stage).jls")
            serialize(out_pkl, res)
            println("  Saved: $(basename(out_pkl))")
            lock(lk) do
                all_results[sp] = Dict(
                    "Q_value"     => get(res, "Q_value", 0.0),
                    "converged"   => get(res, "converged", true),
                    "theta1_alpha"=> isempty(get(res, "theta1", Float64[])) ? [] : [res["theta1"][1]],
                    "theta2"      => get(res, "theta2", Float64[]),
                    "stage"       => current_stage)
            end
        end

        summary_path = joinpath(out_dir, "blp_summary_$(SPEC_NUM)_$(current_stage).json")
        open(summary_path, "w") do f; JSON3.write(f, all_results); end
        log_status("Summary saved to: $summary_path")
        log_status("[DONE] BLP Estimation ($current_stage) complete for specs $spec_ids.")
    end
end

main()
