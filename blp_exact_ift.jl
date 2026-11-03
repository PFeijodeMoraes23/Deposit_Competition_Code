# blp_exact_ift.jl — the exact implicit-function Jacobian ∂δ*/∂θ₂ of the RC-BLP share system, on the
# CPU. Included by blp_engine_gpu.jl: the search gradient and the standard errors of both IFT paths
# (compute_ift_gradient!, compute_ift_gradient_gpu!, compute_delta_jacobian_gpu) use it.
#
# The share model (compute_model_shares_gpu!, compute_model_shares!): in quarter t, a consumer of draw r
# in B market p (mca × t) chooses among the outside good (utility 0), the B products of p and every D
# product of t. With V = clamp(δ + μ, ±500),
#   P_pr = 1 / (1 + Σ_{j∈B_p} e^{V_jr} + Σ_{k∈D_t} e^{V_kr})
#   s_j  = mean_r e^{V_jr} P_pr               j ∈ B_p
#   s_k  = mean_r e^{V_kr} Σ_p w_p P_pr       k ∈ D_t, w_p = pop_weights (Σ_{p∈t} w_p = 1)
# so the B and D products of one quarter form one coupled system and quarters are independent. The
# contraction solves ln s(δ, θ₂) = ln s_data, so ∂δ*/∂θ₂ = −(∂ln s/∂δ)⁻¹ ∂ln s/∂θ₂. The log-share
# Jacobian J̃ = diag(1/s) ∂s/∂δ of a quarter has the blocks
#   J̃_BB      = blockdiag_p( I − diag(1/s_p) Q_p Q_p' / R ),   Q_p[j,r] = e^{V_jr} P_pr
#   J̃_BD[j,l] = −mean_r e^{V_jr} e^{V_lr} P_pr² / s_j          j ∈ B_p, l ∈ D_t
#   J̃_DB[k,l] = −w_{p(l)} mean_r e^{V_lr} e^{V_kr} P_pr² / s_k  l ∈ B_p
#   J̃_DD      = I − diag(1/s_D) mean_r e^{V_kr} e^{V_lr} Σ_p w_p P_pr²
# Dividing each row by its share takes the shares' scale (D shares span many orders of magnitude) out
# of the conditioning without changing the solution. The solve eliminates the B blocks market by market
# and solves the Schur complement of the D block (at most a few hundred D rows a quarter); quarters run
# on separate threads.

using LinearAlgebra

"""The rows of each quarter: B rows market by market, each market's weight, and the D rows."""
struct ExactGroups
    b_rows::Vector{Vector{Int}}                 # per quarter: full-N indices of its B rows, market by market
    b_ranges::Vector{Vector{UnitRange{Int}}}    # per quarter: each market's range within b_rows
    pair_w::Vector{Vector{Float64}}             # per quarter: each market's weight w_p
    d_rows::Vector{Vector{Int}}                 # per quarter: full-N indices of its D rows
end

"""ExactGroups from the Precomp fields (b_mkt_idx in B-row order, d_time_enc in D-row order)."""
function build_exact_groups(b_mask::AbstractVector{Bool}, b_mkt_idx::Vector{Int},
                            d_time_enc::Vector{Int}, pair_time_enc::Vector{Int},
                            pop_weights::Vector{Float64})
    n_times = maximum(vcat(pair_time_enc, d_time_enc))
    bidx = findall(b_mask); didx = findall(.!b_mask)
    rows_of_pair = [Int[] for _ in eachindex(pair_time_enc)]
    for (b, p) in enumerate(b_mkt_idx); push!(rows_of_pair[p], bidx[b]); end
    b_rows   = [Int[] for _ in 1:n_times]
    b_ranges = [UnitRange{Int}[] for _ in 1:n_times]
    pair_w   = [Float64[] for _ in 1:n_times]
    for p in eachindex(pair_time_enc)
        r = rows_of_pair[p]
        isempty(r) && error("market $p has no B row: its weight would enter the D shares unaccounted")
        t = pair_time_enc[p]
        first = length(b_rows[t]) + 1
        append!(b_rows[t], r)
        push!(b_ranges[t], first:length(b_rows[t]))
        push!(pair_w[t], pop_weights[p])
    end
    d_rows = [Int[] for _ in 1:n_times]
    for (d, t) in enumerate(d_time_enc); push!(d_rows[t], didx[d]); end
    return ExactGroups(b_rows, b_ranges, pair_w, d_rows)
end

"""log Σ_i exp(V[i, r]) for each column r; −Inf for a matrix with no rows."""
function lse_cols(V::AbstractMatrix{Float64})
    R = size(V, 2); out = fill(-Inf, R)
    size(V, 1) == 0 && return out
    @inbounds for r in 1:R
        m = -Inf
        for i in axes(V, 1); m = max(m, V[i, r]); end
        s = 0.0
        for i in axes(V, 1); s += exp(V[i, r] - m); end
        out[r] = m + log(s)
    end
    return out
end

lse3(a, b, c) = (m = max(a, b, c); m + log(exp(a - m) + exp(b - m) + exp(c - m)))

"""One quarter of `exact_delta_jacobian`: writes its rows of Jδ, s_model and s0_row; returns the
worst condition numbers met (0.0 when `diagnostics` is false) and the count of singular blocks."""
function _exact_quarter!(Jδ, s_model, s0_row, delta, mu, dlns, rb, rd, ranges, w, diagnostics)
    R = size(mu, 2); K = size(dlns, 2)
    nb, nd = length(rb), length(rd)
    VB = clamp.(delta[rb] .+ mu[rb, :], -500.0, 500.0)
    VD = clamp.(delta[rd] .+ mu[rd, :], -500.0, 500.0)
    LD = lse_cols(VD)
    np = length(ranges)
    logden = Matrix{Float64}(undef, np, R)
    pair_of = Vector{Int}(undef, nb)
    for (q, rng) in enumerate(ranges)
        LB = lse_cols(@view VB[rng, :])
        @inbounds for r in 1:R; logden[q, r] = lse3(0.0, LB[r], LD[r]); end
        pair_of[rng] .= q
    end
    logq = VB .- logden[pair_of, :]
    QB = exp.(logq)
    sB = max.(vec(sum(QB; dims = 2)) ./ R, floatmin(Float64))
    s_model[rb] .= sB
    s0_row[rb] .= (vec(sum(exp.(.-logden); dims = 2)) ./ R)[pair_of]
    QBs = QB ./ sB                                   # q_jr / s_j
    cond_bb = 0.0; cond_s = 0.0; n_singular = 0
    local JBD, JDB, JDD
    if nd > 0
        lw  = log.(w)
        LW1 = lse_cols(lw .- logden)                 # log Σ_p w_p P_pr
        LW2 = lse_cols(lw .- 2.0 .* logden)          # log Σ_p w_p P_pr²
        sD  = max.(vec(sum(exp.(VD .+ LW1'); dims = 2)) ./ R, floatmin(Float64))
        s_model[rd] .= sD
        F   = exp.(VD .+ 0.5 .* LW2')               # ≤ 1: e^{V_kr} √(Σ_p w_p P_pr²) ≤ e^{V_kr} max_p P_pr
        JDD = Matrix{Float64}(I, nd, nd) .- ((F ./ sD) * F') ./ R
        c   = vec(maximum(VD; dims = 1))             # per-draw shift keeps both factors ≤ 1
        A   = exp.(logq .- logden[pair_of, :] .+ c')
        Bm  = exp.(VD .- c')
        JBD_raw = -(A * Bm') ./ R                    # ∂s_B/∂δ_D
        JBD = JBD_raw ./ sB
        JDB = (w[pair_of] .* JBD_raw)' ./ sD
    end
    Xbd = nd > 0 ? Matrix{Float64}(undef, nb, nd) : zeros(nb, 0)
    Xb  = Matrix{Float64}(undef, nb, K)
    bB  = dlns[rb, :]
    for rng in ranges
        Jp = Matrix{Float64}(I, length(rng), length(rng)) .- ((@view QBs[rng, :]) * (@view QB[rng, :])') ./ R
        Fp = lu(Jp; check = false)
        issuccess(Fp) || (n_singular += 1)
        diagnostics && (cond_bb = max(cond_bb, all(isfinite, Jp) ? cond(Jp) : Inf))
        Xb[rng, :] = Fp \ bB[rng, :]
        nd > 0 && (Xbd[rng, :] = Fp \ JBD[rng, :])
    end
    if nd > 0
        S  = JDD .- JDB * Xbd
        FS = lu(S; check = false)
        issuccess(FS) || (n_singular += 1)
        diagnostics && (cond_s = all(isfinite, S) ? cond(S) : Inf)
        xD = FS \ (JDB * Xb .- dlns[rd, :])
        Jδ[rd, :] = xD
        Jδ[rb, :] = .-Xb .- Xbd * xD
    else
        Jδ[rb, :] = .-Xb
    end
    return cond_bb, cond_s, n_singular
end

"""
    exact_delta_jacobian(delta, mu, dlns, G; diagnostics=false) → (Jδ, info)

∂δ*/∂θ₂ (N × K) from the full log-share Jacobian, quarter by quarter on separate threads. `mu` is
buf.mu (N × R) at θ₂, `dlns` the N × K matrix ∂ln s/∂θ₂ at fixed δ. `info`: the model shares computed
here (to check against the engine's), the outside share mean_r P_pr of each B row's market, the count
of singular blocks (their rows come back non-finite), and with `diagnostics` the worst condition
numbers of the market blocks and of the quarters' Schur complements.
"""
function exact_delta_jacobian(delta::Vector{Float64}, mu::Matrix{Float64}, dlns::Matrix{Float64},
                              G::ExactGroups; diagnostics::Bool = false)
    N = size(mu, 1); K = size(dlns, 2); T = length(G.b_rows)
    Jδ = fill(NaN, N, K); s_model = fill(NaN, N); s0_row = fill(NaN, N)
    cbb = zeros(T); cs = zeros(T); nsing = zeros(Int, T)
    nblas = BLAS.get_num_threads()
    BLAS.set_num_threads(1)                          # the quarters are the parallel unit
    try
        Threads.@threads for t in 1:T
            isempty(G.b_rows[t]) && isempty(G.d_rows[t]) && continue
            cbb[t], cs[t], nsing[t] = _exact_quarter!(Jδ, s_model, s0_row, delta, mu, dlns,
                                                      G.b_rows[t], G.d_rows[t], G.b_ranges[t],
                                                      G.pair_w[t], diagnostics)
        end
    finally
        BLAS.set_num_threads(nblas)
    end
    return Jδ, Dict{String,Any}("s_model" => s_model, "s0_row" => s0_row,
                                "n_singular_blocks" => sum(nsing),
                                "max_cond_JBB" => maximum(cbb; init = 0.0),
                                "max_cond_schur" => maximum(cs; init = 0.0))
end
