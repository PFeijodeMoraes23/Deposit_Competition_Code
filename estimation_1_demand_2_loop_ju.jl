"""
estimation_1_demand_2_loop_ju.jl
=================================
Julia port of estimation_1_demand_2_loop.py.

BLP outer-inner demand estimation loop (Appendix-BLP, V_Main.tex).
Takes per-spec demand prep parquets from estimation_1_demand_1_prep.py and
recovers theta2* = (Pi*, Sigma*) via GMM and theta1* via linear IV.

Key differences vs Python:
  - Shared-memory multithreading via Threads.@threads (no process serialisation)
  - In-place BLAS via mul! (no intermediate allocation in hot path)
  - Full Float64 throughout (Julia default)
  - Chunked-R path: --chunk-size arg limits peak (N×R) alloc per thread

Usage
-----
  julia -t 12 estimation_1_demand_2_loop_ju.jl --spec 1 --stage logit
  julia -t 12 estimation_1_demand_2_loop_ju.jl --spec all --stage full --R 500
  julia -t 12 estimation_1_demand_2_loop_ju.jl --dry-run --spec 1 --R 10 --stage sigma --chunk-size 5

Dependencies (add to Julia environment before first run):
  Pkg.add(["Parquet2", "DataFrames", "SparseArrays", "LinearAlgebra",
           "Statistics", "Random", "Optim", "QuasiMonteCarlo",
           "Distributions", "JSON3", "Serialization", "ArgParse"])

References
----------
  Berry, Levinsohn & Pakes (1995, Econometrica)
  Conlon & Gortmaker (2020, RAND J. Econ.)
  Egan, Hortacsu & Matvos (2017, AER)
  Nevo (2001, Econometrica)
"""

using Parquet2, DataFrames, SparseArrays, LinearAlgebra, Statistics
using Random, Optim, QuasiMonteCarlo, Distributions
using JSON3, Serialization, ArgParse, Printf, Dates

# --------------------------------------------------------------------------
# 0. Constants
# --------------------------------------------------------------------------
const SPEC_NUM   = 1         # data file prefix: demand_1_final_spec_*.parquet
const X_COLS     = ["fgc_covered", "has_ip", "seg_S2", "seg_S3", "seg_S4", "seg_S5",
                    "log_total_assets_lag", "equity_ratio_lag"]
const L_PROD     = length(X_COLS)   # 8
const K_TYPES    = 4
const K_LIST     = [1, 2, 4, 5]
const D_COLS     = ["gdp_per_capita", "fraction_65plus", "fraction_young",
                    "pix_users_pf_per1000", "connections_per100", "frac_4g5g",
                    "branches_per1000", "cadunico_families_per1000"]
const D_DIM      = length(D_COLS)   # 8

const IV_BLP_LOO = ["loo_log_assets", "mean_loo_log_assets",
                    "loo_equity_ratio", "mean_loo_equity_ratio",
                    "loo_basileia", "mean_loo_basileia",
                    "loo_credit_assets", "mean_loo_credit_assets",
                    "loo_npl_provision", "mean_loo_npl_provision",
                    "n_rivals"]
const IV_COST    = ["personnel_cost_ratio_lag", "admin_cost_ratio_lag",
                    "tax_cost_ratio_lag"]
const IV_CAPITAL = ["indice_basileia_lag"]

# Global log buffer (thread-safe via ReentrantLock)
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
"""Generate R × dim Halton draws mapped to N(0,1) (Nevo 2001)."""
function generate_halton_draws(R::Int, dim::Int, seed::Int)::Matrix{Float64}
    skip = seed * R
    n_total = R + skip
    pts_full = QuasiMonteCarlo.sample(n_total, zeros(dim), ones(dim), HaltonSample())  # (dim, n_total)
    pts = pts_full[:, (skip+1):end]  # (dim, R)
    pts = clamp.(pts, 1e-6, 1.0 - 1e-6)
    return quantile.(Normal(), pts)'   # (R, dim)
end

"""Draw R demographic vectors per (mca, time) from N(mu_m, (0.1*sigma_nat)^2)."""
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
        draws[key] = mu .+ (nat_std .* 0.1) .* randn(rng, D, R) |> transpose |> Matrix  # (R, D)
    end
    return draws
end

# --------------------------------------------------------------------------
# 3. Theta2 Structure
# --------------------------------------------------------------------------
"""Returns (sigma_indices, pi_interactions, n_params).
Coefficient vector layout: [spread, x_1, ..., x_L]  (1-based Julia indexing, indices offset by 1)
"""
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
# 4. Compute mu  (N × R)
# --------------------------------------------------------------------------
"""Compute mu_rjmt using vectorised BLAS + Pi interactions."""
function compute_mu(prod_vec::Matrix{Float64},      # (N, coef_dim)
                    nu_draws::Matrix{Float64},       # (R, coef_dim)
                    stacked_draws::Array{Float64,3}, # (n_keys+1, R, D)
                    obs_key_idx::Vector{Int},        # (N,) 1-based
                    sigma_vals::Vector{Float64},
                    sigma_indices::Vector{Int},
                    pi_vals::Vector{Float64},
                    pi_interactions::Vector{Tuple{Int,Int}},
                    R::Int, coef_dim::Int)::Matrix{Float64}

    sigma_diag = zeros(Float64, coef_dim)
    for (pos, cidx) in enumerate(sigma_indices)
        cidx <= coef_dim && (sigma_diag[cidx] = sigma_vals[pos])
    end

    # sigma_nu: (R, coef_dim) .* sigma_diag  → (R, coef_dim)
    sigma_nu = nu_draws[:, 1:coef_dim] .* sigma_diag'   # broadcast row-wise

    # mu = prod_vec * sigma_nu'  → (N, R)
    mu = prod_vec * sigma_nu'    # BLAS dgemm

    # Pi term
    D_dim = size(stacked_draws, 3)
    for (pi_idx, (cidx, didx)) in enumerate(pi_interactions)
        (cidx <= coef_dim && didx <= D_dim) || continue
        p_v = @view prod_vec[:, cidx]             # (N,)
        # stacked_draws[obs_key_idx, :, didx] → (N, R)
        d_v = stacked_draws[obs_key_idx, :, didx] # (N, R)
        mu .+= pi_vals[pi_idx] .* p_v .* d_v
    end
    return mu
end

# --------------------------------------------------------------------------
# 5. Market Index Construction
# --------------------------------------------------------------------------
struct Precomp
    # market indices
    b_mkt_idx    ::Vector{Int}
    d_time_enc   ::Vector{Int}
    unique_pairs ::Vector{Tuple{String,String}}
    unique_times ::Vector{String}
    pair_time_enc::Vector{Int}
    pop_weights  ::Vector{Float64}
    # sparse aggregation
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
    # static per-spec arrays
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

    # Unique (mca, time) B-markets
    b_mca  = mca_codes[b_mask]
    b_time = time_ids[b_mask]
    raw_pairs     = collect(zip(b_mca, b_time))
    unique_pairs  = sort(unique(raw_pairs))
    pair_to_idx   = Dict(p => i for (i, p) in enumerate(unique_pairs))
    b_mkt_idx     = [pair_to_idx[(m, t)] for (m, t) in raw_pairs]   # 1-based

    # Time encoding for D-firms
    unique_times  = sort(unique(time_ids))
    time_to_idx   = Dict(t => i for (i, t) in enumerate(unique_times))
    d_time_enc    = [time_to_idx[t] for t in time_ids[d_mask]]      # 1-based

    n_pairs = length(unique_pairs)
    n_times = length(unique_times)

    pair_times    = [p[2] for p in unique_pairs]
    pair_time_enc = [time_to_idx[t] for t in pair_times]

    # Population weights
    pop_b         = pop_total[b_mask]
    pair_pop      = zeros(n_pairs)
    for (i, w) in zip(b_mkt_idx, pop_b); pair_pop[i] += w; end
    time_pop      = zeros(n_times)
    for (i, w) in zip(pair_time_enc, pair_pop); time_pop[i] += w; end
    pop_weights   = pair_pop ./ max.(time_pop[pair_time_enc], 1e-30)

    # Sparse aggregation matrices
    B_agg  = sparse(b_mkt_idx,  1:N_B,    ones(N_B),  n_pairs, N_B)
    D_agg  = sparse(d_time_enc, 1:N_D,    ones(N_D),  n_times, N_D)
    PT_agg = sparse(pair_time_enc, 1:n_pairs, pop_weights, n_times, n_pairs)

    sort_b      = sortperm(b_mkt_idx)
    _, b_grp_s  = _unique_with_starts(b_mkt_idx[sort_b])
    sort_d      = sortperm(d_time_enc)
    d_uval, d_grp_s = _unique_with_starts(d_time_enc[sort_d])
    sort_pt     = sortperm(pair_time_enc)
    pt_uval, pt_grp_s = _unique_with_starts(pair_time_enc[sort_pt])

    # Contraction static arrays
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

"""Return (unique_sorted_values, start_indices_1based) for a sorted integer vector."""
function _unique_with_starts(sorted_vec::Vector{Int})
    isempty(sorted_vec) && return Int[], Int[]
    uval = Int[sorted_vec[1]]
    gstart = Int[1]
    for i in 2:length(sorted_vec)
        if sorted_vec[i] != sorted_vec[i-1]
            push!(uval, sorted_vec[i])
            push!(gstart, i)
        end
    end
    return uval, gstart
end

# --------------------------------------------------------------------------
# 6. Model Shares  (Eq-13-B/D)
# --------------------------------------------------------------------------
"""Stable log-sum-exp per group using per-group max (reduceat equivalent)."""
function logsumexp_groups(V, sort_idx, grp_start, n_groups, uval, R; weights=nothing)
    n_uniq = length(grp_start)
    result  = fill(-Inf, n_groups, R)
    for g in 1:n_uniq
        gidx = grp_start[g]
        gend = g < n_uniq ? grp_start[g+1] - 1 : length(sort_idx)
        rows = sort_idx[gidx:gend]
        V_g  = V[rows, :]           # (n_g, R)
        mx   = maximum(V_g; dims=1)
        if weights !== nothing
            w_g = weights[rows]
            lse = mx .+ log.(max.(sum(w_g .* exp.(V_g .- mx); dims=1), 1e-300))
        else
            lse = mx .+ log.(max.(sum(exp.(V_g .- mx); dims=1), 1e-300))
        end
        result[uval[g], :] .= vec(lse)
    end
    return result
end

function compute_model_shares(delta::Vector{Float64},
                               mu::Matrix{Float64},    # (N, R)
                               pc::Precomp,
                               R::Int)
    N_B = sum(pc.b_mask)
    N_D = sum(pc.d_mask)
    n_pairs = length(pc.unique_pairs)
    n_times = length(pc.unique_times)

    delta_B = delta[pc.b_mask]
    delta_D = delta[pc.d_mask]
    mu_B    = mu[pc.b_mask, :]    # (N_B, R)
    mu_D    = mu[pc.d_mask, :]    # (N_D, R)

    V_B = clamp.(delta_B .+ mu_B, -500.0, 500.0)   # (N_B, R)
    V_D = clamp.(delta_D .+ mu_D, -500.0, 500.0)   # (N_D, R)

    # D-firm log-sum per time period
    log_sum_D_time = logsumexp_groups(V_D, pc.sort_d, pc.d_grp_start,
                                       n_times, pc.d_uval, R)  # (n_times, R)
    log_D_sum_pair = log_sum_D_time[pc.pair_time_enc, :]        # (n_pairs, R)

    # B-firm log-sum per market
    log_sum_B_mkt  = logsumexp_groups(V_B, pc.sort_b, pc.b_grp_start,
                                       n_pairs, collect(1:n_pairs), R)  # (n_pairs, R)

    # Joint log-denominator with outside-option anchor ε=1
    OUTSIDE_EPS  = 1.0
    log_outside  = log(OUTSIDE_EPS)
    joint_max    = max.(max.(log_sum_B_mkt, log_D_sum_pair), log_outside)  # (n_pairs, R)
    log_denom    = joint_max .+ log.(max.(
        exp.(log_outside .- joint_max) .+
        exp.(log_sum_B_mkt .- joint_max) .+
        exp.(log_D_sum_pair .- joint_max), 1e-300))  # (n_pairs, R)

    # B-firm shares
    log_denom_B = log_denom[pc.b_mkt_idx, :]              # (N_B, R)
    q_B         = exp.(V_B .- log_denom_B)                # (N_B, R)
    s_B         = vec(Statistics.mean(q_B; dims=2))        # (N_B,)

    # D-firm national shares: Eq-13-D
    neg_log_denom = -log_denom                            # (n_pairs, R)

    # Weighted version: PT_agg has pop_weights in its values
    # max-shifted sum via sparse multiply
    max_neg_ld    = fill(-Inf, n_times, R)
    for g in 1:length(pc.pt_grp_start)
        gidx  = pc.pt_grp_start[g]
        gend  = g < length(pc.pt_grp_start) ? pc.pt_grp_start[g+1] - 1 : length(pc.sort_pt)
        rows  = pc.sort_pt[gidx:gend]
        V_g   = neg_log_denom[rows, :]
        mx    = maximum(V_g; dims=1)
        max_neg_ld[pc.pt_uval[g], :] .= vec(mx)
    end

    shifted_inv  = exp.(neg_log_denom .- max_neg_ld[pc.pair_time_enc, :])  # (n_pairs, R)
    sum_wtd      = pc.PT_agg * shifted_inv                                   # (n_times, R) BLAS
    log_inv_wtd  = max_neg_ld .+ log.(max.(sum_wtd, 1e-300))                # (n_times, R)

    log_s_D_r    = V_D .+ log_inv_wtd[pc.d_time_enc, :]                     # (N_D, R)
    row_max      = maximum(log_s_D_r; dims=2)                               # (N_D, 1)
    s_D_nat      = vec(Statistics.mean(exp.(log_s_D_r .- row_max); dims=2)) .* vec(exp.(row_max))  # (N_D,)

    return s_B, s_D_nat
end

# --------------------------------------------------------------------------
# 7. Anderson(m=5) BLP Contraction
# --------------------------------------------------------------------------
function blp_contraction(delta_init::Union{Vector{Float64},Nothing},
                          mu::Matrix{Float64},      # (N, R) — full or chunk
                          pc::Precomp,
                          R::Int;
                          tol::Float64=1e-12,
                          max_iter::Int=1500)
    N     = length(pc.b_mask)
    delta = if delta_init !== nothing && length(delta_init) == N
        copy(delta_init)
    else
        d = zeros(N)
        d[pc.d_mask] .= pc.ln_s_data_D[pc.d_mask]
        d[pc.b_mask] .= pc.ln_s_data_B_cond[pc.b_mask]
        d
    end

    m_aa     = 5
    F_hist   = zeros(N, m_aa)
    X_hist   = zeros(N, m_aa)
    ptr      = 1
    hist_len = 0
    norm_history = Float64[]

    for h in 1:max_iter
        s_B, s_D = compute_model_shares(delta, mu, pc, R)
        delta_new = copy(delta)
        delta_new[pc.d_mask] .= delta[pc.d_mask] .+ pc.ln_s_data_D[pc.d_mask] .-
                                  log.(clamp.(s_D, 1e-15, Inf))
        delta_new[pc.b_mask] .= delta[pc.b_mask] .+ pc.ln_s_data_B_cond[pc.b_mask] .-
                                  log.(clamp.(s_B, 1e-15, Inf))
        clamp!(delta_new, -500.0, 500.0)

        if !all(isfinite, delta_new)
            n_bad = count(!isfinite, delta_new)
            println("    [contraction ABORT] $n_bad/$N non-finite entries.")
            return delta, false, h, norm_history
        end

        f    = delta_new .- delta
        norm = maximum(abs.(f))
        push!(norm_history, norm)

        (h % 50 == 0 || norm < tol) && println("    [Anderson iter=$h/$max_iter] norm=$(round(norm, sigdigits=4))")
        if norm < tol
            return delta_new, true, h, norm_history
        end

        F_hist[:, ptr] = f
        X_hist[:, ptr] = delta
        ptr      = mod1(ptr + 1, m_aa)
        hist_len = min(hist_len + 1, m_aa)

        if hist_len > 1
            F_k = F_hist[:, 1:hist_len]
            dF  = F_k[:, 2:end] .- F_k[:, 1:1]
            try
                rhs   = -(dF' * F_k[:, 1])
                A_aa  = dF' * dF + 1e-10 * I(hist_len - 1)
                c_bar = A_aa \ rhs
                c     = vcat(1.0 - sum(c_bar), c_bar)
                d_aa  = (X_hist[:, 1:hist_len] .+ F_hist[:, 1:hist_len]) * c
                delta = if all(isfinite, d_aa)
                    clamp.(d_aa, -500.0, 500.0)
                else
                    delta_new
                end
            catch
                delta = delta_new
            end
        else
            delta = delta_new
        end
    end
    return delta, false, max_iter, norm_history
end

"""Chunked-R contraction: avoids materialising full (N, R) mu."""
function blp_contraction_draws(delta_init, R::Int,
                                 prod_vec, nu_draws, stacked_draws_padded,
                                 obs_key_idx, sigma_vals, sigma_indices,
                                 pi_vals, pi_interactions, coef_dim::Int,
                                 pc::Precomp;
                                 tol::Float64=1e-12, max_iter::Int=1500,
                                 chunk_size::Union{Int,Nothing}=nothing)
    if chunk_size === nothing || chunk_size >= R
        mu = compute_mu(prod_vec, nu_draws, stacked_draws_padded, obs_key_idx,
                        sigma_vals, sigma_indices, pi_vals, pi_interactions, R, coef_dim)
        return blp_contraction(delta_init, mu, pc, R; tol=tol, max_iter=max_iter)
    end

    # ---- Chunked path ----
    N   = size(prod_vec, 1)
    N_B = sum(pc.b_mask)
    N_D = sum(pc.d_mask)

    delta = if delta_init !== nothing && length(delta_init) == N
        copy(delta_init)
    else
        d = zeros(N)
        d[pc.d_mask] .= pc.ln_s_data_D[pc.d_mask]
        d[pc.b_mask] .= pc.ln_s_data_B_cond[pc.b_mask]
        d
    end

    m_aa = 5; F_hist = zeros(N, m_aa); X_hist = zeros(N, m_aa)
    ptr = 1; hist_len = 0; norm_history = Float64[]

    for h in 1:max_iter
        s_B_acc = zeros(N_B); s_D_acc = zeros(N_D)
        r0 = 1
        while r0 <= R
            r1  = min(r0 + chunk_size - 1, R)
            rc  = r1 - r0 + 1
            nu_c  = nu_draws[r0:r1, :]
            sd_c  = stacked_draws_padded[:, r0:r1, :]
            mu_c  = compute_mu(prod_vec, nu_c, sd_c, obs_key_idx,
                               sigma_vals, sigma_indices, pi_vals, pi_interactions, rc, coef_dim)
            s_B_c, s_D_c = compute_model_shares(delta, mu_c, pc, rc)
            s_B_acc .+= s_B_c .* rc
            s_D_acc .+= s_D_c .* rc
            r0 = r1 + 1
        end
        s_B = s_B_acc ./ R; s_D = s_D_acc ./ R

        delta_new = copy(delta)
        delta_new[pc.d_mask] .= delta[pc.d_mask] .+ pc.ln_s_data_D[pc.d_mask] .-
                                  log.(clamp.(s_D, 1e-15, Inf))
        delta_new[pc.b_mask] .= delta[pc.b_mask] .+ pc.ln_s_data_B_cond[pc.b_mask] .-
                                  log.(clamp.(s_B, 1e-15, Inf))
        clamp!(delta_new, -500.0, 500.0)

        !all(isfinite, delta_new) && (println("    [ABORT] non-finite"); return delta, false, h, norm_history)
        f    = delta_new .- delta
        norm = maximum(abs.(f))
        push!(norm_history, norm)
        (h % 50 == 0 || norm < tol) && println("    [Anderson iter=$h] norm=$(round(norm, sigdigits=4))")
        norm < tol && return delta_new, true, h, norm_history

        F_hist[:, ptr] = f; X_hist[:, ptr] = delta
        ptr = mod1(ptr + 1, m_aa); hist_len = min(hist_len + 1, m_aa)
        if hist_len > 1
            F_k = F_hist[:, 1:hist_len]; dF = F_k[:, 2:end] .- F_k[:, 1:1]
            try
                c_bar = (dF' * dF + 1e-10 * I(hist_len-1)) \ (-(dF' * F_k[:, 1]))
                c     = vcat(1.0 - sum(c_bar), c_bar)
                d_aa  = (X_hist[:, 1:hist_len] .+ F_hist[:, 1:hist_len]) * c
                delta = all(isfinite, d_aa) ? clamp.(d_aa, -500.0, 500.0) : delta_new
            catch; delta = delta_new; end
        else; delta = delta_new; end
    end
    return delta, false, max_iter, norm_history
end

# --------------------------------------------------------------------------
# 8. Linear IV  (Eq-A5)
# --------------------------------------------------------------------------
function build_regressor_matrices(df::DataFrame)
    N     = nrow(df)
    # Single spread column (alpha common across deposit types)
    spread_cols = coalesce.(df.spread_qoq, 0.0)  # (N,)

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
    return spread_hat   # (N, 1)
end

function estimate_theta1(delta::Vector{Float64}, pc::Precomp)
    X_full = pc.X_full
    X_hat  = pc.X_hat
    valid  = pc.theta1_valid .& isfinite.(delta)
    X_v    = X_hat[valid, :]
    d_v    = delta[valid]
    cl     = pc.clusters[valid]

    theta1 = try X_v \ d_v catch; zeros(size(X_hat, 2)) end
    xi     = delta .- X_full * theta1
    return theta1, xi
end

# --------------------------------------------------------------------------
# 9. GMM Objective
# --------------------------------------------------------------------------
function compute_gmm_moments(xi::Vector{Float64}, pc::Precomp)
    Z = pc.Z_moments
    return vec(Statistics.mean(xi .* Z; dims=1))   # (n_iv,)
end

function gmm_objective(theta2::Vector{Float64},
                        df::DataFrame,
                        prod_vec, nu_draws, stacked_draws,
                        obs_key_idx, sigma_indices, pi_interactions,
                        R::Int, coef_dim::Int,
                        W::Matrix{Float64},
                        tol_inner::Float64, max_inner::Int,
                        delta_cache::Ref{Union{Vector{Float64},Nothing}},
                        pc::Precomp;
                        chunk_size::Union{Int,Nothing}=nothing)::Float64

    sigma_vals, pi_vals = unpack_theta2(theta2, sigma_indices, pi_interactions)
    delta_init          = delta_cache[]

    delta, converged, n_iter, _ = blp_contraction_draws(
        delta_init, R, prod_vec, nu_draws, stacked_draws,
        obs_key_idx, sigma_vals, sigma_indices, pi_vals, pi_interactions, coef_dim, pc;
        tol=tol_inner, max_iter=max_inner, chunk_size=chunk_size)

    converged || println("  [!] Inner loop did not converge in $n_iter iterations")
    delta_cache[] = copy(delta)

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
    coef_dim = 1 + L_PROD   # single spread + L_PROD x-cols

    nu_draws = generate_halton_draws(R, coef_dim, seed)  # (R, coef_dim)

    results = Dict{String,Any}("spec_id" => spec_id, "stage" => args["stage"])

    if args["stage"] == "logit"
        println("  Stage: LOGIT (theta2 = 0)")
        is_B  = Bool.(coalesce.(df.is_B, false))
        share_D_clean     = Float64.(coalesce.(df.share_D,      0.0))
        share_B_cond_clean= Float64.(coalesce.(df.share_B_cond, 0.0))
        delta = zeros(nrow(df))
        delta[.!is_B] .= log.(clamp.(share_D_clean[.!is_B], 1e-15, Inf))
        delta[is_B]   .= log.(clamp.(share_B_cond_clean[is_B], 1e-15, Inf))

        # Build minimal precomp for estimate_theta1
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
    println("  Precomputing arrays for vectorized logic...")

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
    stacked_draws_padded = vcat(stacked_draws, zero_draw)   # (n_keys+1, R, D)
    pad_idx    = n_keys + 1

    obs_key_idx = [get(key_to_idx, (mca_codes[i], time_ids[i]), pad_idx) for i in 1:N_obs]

    # prod_vec (N, coef_dim)
    prod_vec  = zeros(N_obs, coef_dim)
    spreads   = coalesce.(df.spread_qoq, 0.0)
    dep_types = Int.(coalesce.(df.deposit_type, 0))
    prod_vec[:, 1] .= spreads
    for (i, col) in enumerate(X_COLS)
        col in names(df) && (prod_vec[:, 1+i] .= coalesce.(df[!, col], 0.0))
    end

    # Weighting matrix W = (Z'Z/N)^{-1}
    spread_cols, x_mat, Z_mat, iv_cols = build_regressor_matrices(df)
    iv_avail = [c for c in iv_cols if std(replace(coalesce.(df[!, c], 0.0), Inf=>0.0, -Inf=>0.0)) > 1e-10]
    Z = zeros(N_obs, length(iv_avail))
    for (i, c) in enumerate(iv_avail)
        v = coalesce.(df[!, c], 0.0); replace!(v, Inf=>0.0, -Inf=>0.0)
        Z[:, i] .= v
    end
    W = try inv(Z' * Z ./ N_obs) catch; Matrix(I(length(iv_avail)) * 1.0) end

    # Build static per-spec arrays for SpecCache
    H            = hcat(x_mat, Z_mat)
    spread_hat   = project_endogenous_spreads(spread_cols, H, dep_types)
    X_full       = hcat(spread_cols, x_mat)
    X_hat        = hcat(spread_hat,  x_mat)
    valid        = BitVector(all.(isfinite, eachrow(X_hat)))
    clusters     = string.(df.CodConglomeradoPrudencial) .* "_" .*
                   first.(split.(string.(df.time_id), "Q"))
    pc = build_precomp(df, Z, X_full, X_hat, valid, clusters)

    chunk_size = let cs = get(args, "chunk_size", 0)
        (cs === nothing || cs == 0) ? nothing : cs
    end
    chunk_size !== nothing && println("  Chunked-R: chunk_size=$chunk_size (R=$R, $(cld(R, chunk_size)) chunks/iter)")

    # Starting values
    _, out_dir = get_paths(args["hpc"])
    prev_stages = Dict("full" => "sigma", "extended" => "full")
    theta2_0    = nothing
    if args["stage"] in keys(prev_stages)
        prev_path = joinpath(out_dir, "blp_checkpoint_spec_$(spec_id)_$(prev_stages[args["stage"]]).jld2_compat.jls")
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

    delta_cache  = Ref{Union{Vector{Float64},Nothing}}(nothing)

    if get(args, "dry_run", false)
        println("  [DRY RUN] Running 10 contraction iterations for timing...")
        sv, pv = unpack_theta2(theta2_0, sigma_indices, pi_interactions)
        t0 = time()
        blp_contraction_draws(nothing, R, prod_vec, nu_draws, stacked_draws_padded,
                               obs_key_idx, sv, sigma_indices, pv, pi_interactions, coef_dim, pc;
                               tol=args["tol_inner"], max_iter=10, chunk_size=chunk_size)
        elapsed = time() - t0
        println("  [DRY RUN] 10 iterations in $(round(elapsed, digits=2))s ($(round(elapsed/10, digits=3)) s/iter).")
        println("  [DRY RUN] Exiting early.")
        return Dict("dry_run" => true)
    end

    outer_iter = Ref(0)
    function obj_fn(t2)
        val = gmm_objective(t2, df, prod_vec, nu_draws, stacked_draws_padded,
                            obs_key_idx, sigma_indices, pi_interactions, R, coef_dim,
                            W, args["tol_inner"], args["max_inner"], delta_cache, pc;
                            chunk_size=chunk_size)
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

    sv, pv = unpack_theta2(theta2_star, sigma_indices, pi_interactions)
    delta_star, conv, n_it, _ = blp_contraction_draws(
        delta_cache[], R, prod_vec, nu_draws, stacked_draws_padded,
        obs_key_idx, sv, sigma_indices, pv, pi_interactions, coef_dim, pc;
        tol=args["tol_inner"], max_iter=args["max_inner"], chunk_size=chunk_size)

    theta1_star, xi_star = estimate_theta1(delta_star, pc)

    results["theta1"]             = theta1_star
    results["theta2"]             = theta2_star
    results["delta"]              = delta_star
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
    chk_path = joinpath(out_dir, "blp_checkpoint_spec_$(spec_id)_$(args["stage"]).jld2_compat.jls")
    try serialize(chk_path, Dict("theta2_star" => theta2_star, "delta_star" => delta_star))
        println("  [CHECKPOINT] Saved $(basename(chk_path))")
    catch e; println("  [CHECKPOINT-WARN] $e") end

    return results
end

# --------------------------------------------------------------------------
# 11. Main
# --------------------------------------------------------------------------
function parse_args_blp()
    s = ArgParseSettings(description="BLP Demand Estimation Loop (Appendix-BLP)")
    @add_arg_table! s begin
        "--spec";    arg_type=String;  default="12"
        "--stage";   arg_type=String;  default="logit"
        "--R";       arg_type=Int;     default=100
        "--seed";    arg_type=Int;     default=42
        "--tol-inner"; arg_type=Float64; default=1e-12; dest_name="tol_inner"
        "--max-inner"; arg_type=Int;     default=2000;  dest_name="max_inner"
        "--tol-outer"; arg_type=Float64; default=1e-6;  dest_name="tol_outer"
        "--method";  arg_type=String;  default="l-bfgs-b"
        "--workers"; arg_type=Int;     default=Threads.nthreads()
        "--hpc";     action=:store_true
        "--local-dir"; arg_type=String; default=nothing; dest_name="local_dir"
        "--dry-run"; action=:store_true; dest_name="dry_run"
        "--chunk-size"; arg_type=Int;  default=0; dest_name="chunk_size"
    end
    return parse_args(s)
end

function main()
    args = parse_args_blp()
    spec_ids = args["spec"] == "all" ? collect(1:12) : [parse(Int, args["spec"])]

    log_status("BLP Demand Estimation Loop — START")
    log_status("  Stage: $(args["stage"]) | Specs: $spec_ids")
    log_status("  R=$(args["R"]) | seed=$(args["seed"]) | method=$(args["method"]) | HPC=$(args["hpc"])")
    log_status("  tol_inner=$(args["tol_inner"]) | tol_outer=$(args["tol_outer"]) | threads=$(Threads.nthreads())")

    _, out_dir = get_paths(args["hpc"]; local_dir=get(args, "local_dir", nothing))
    mkpath(out_dir)

    println("\n  [DIAGNOSTIC] Output Directory: $out_dir")

    stages_to_run = args["stage"] == "sequence" ? ["logit", "sigma", "full", "extended"] : [args["stage"]]

    for current_stage in stages_to_run
        args["stage"] = current_stage
        all_results   = Dict{Int,Any}()
        lk            = ReentrantLock()

        println("  Starting parallel execution of $(length(spec_ids)) specs for stage '$current_stage' with $(Threads.nthreads()) threads...")

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
