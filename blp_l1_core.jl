# blp_l1_core.jl — the L1 items of an RC-BLP fit (handoff 2026-11-02, M5), computed from the share
# model of the engine at (δ*, θ₂*). Pure Julia, no engine types: blp_l1_items.jl supplies the inputs
# from a fit; diagnostics/test_l1_core.jl checks it against finite differences on synthetic data.
#
# The share model (compute_model_shares_gpu!): in quarter t, a consumer of draw r in B market p chooses
# among the outside good, the B products of p and every D product of t. With V = clamp(δ + μ, ±500),
#   P_pr = 1 / (1 + Σ_{j∈B_p} e^{V_jr} + Σ_{k∈D_t} e^{V_kr}),   q_jr = e^{V_jr} P_pr       (j ∈ B_p)
#   q_kr = e^{V_kr} Σ_p w_p P_pr                                                         (k ∈ D_t)
# and s_i = mean_r q_ir. The spread ρ_i enters V_ir with the consumer's own coefficient
#   α_ir = α + σ_ρ ν_rρ + Σ_d π_ρd D̃_{m(i) r d}
# (α the θ₁ price coefficient; only the spread's σ and π enter), so the own-spread derivative is
#   ∂s_i/∂ρ_i = mean_r α_ir ∂q_ir/∂V_ir,   ∂q_jr/∂V_jr = q_jr(1 − q_jr),
#   ∂q_kr/∂V_kr = e^{V_kr} Σ_p w_p P_pr − e^{2V_kr} Σ_p w_p P_pr²,
# and the own-spread elasticity of row i is η_i = (ρ_i / s_i) ∂s_i/∂ρ_i: the random-coefficients
# counterpart of the plug-in α̂ ρ(1 − s) that demand.elast.blp.* reports.
#
# Weights. Row: every demand row counts once. Deposit: row i counts Dep_Act_i (its active deposits,
# share × market size). Within a row, the consumers who chose it are draw r in proportion to
# q_ir / (R s_i), so "choosers" weights put weight w_i q_ir / (R s_i) on (i, r). Population: every
# (B market, draw) counts M_p / R, the market's consumers whatever they chose.

using LinearAlgebra

"""Weighted histogram on a fixed grid, for weighted percentiles of α without storing N × R values."""
mutable struct WHist
    lo::Float64; hi::Float64; w::Vector{Float64}; total::Float64; clipped::Float64
    sum_x::Float64; nonneg::Float64
end
WHist(lo, hi, nbins) = WHist(lo, hi, zeros(nbins), 0.0, 0.0, 0.0, 0.0)

function add!(h::WHist, x::Float64, w::Float64)
    (w > 0 && isfinite(x)) || return
    nb = length(h.w)
    b = floor(Int, (x - h.lo) / (h.hi - h.lo) * nb) + 1
    if b < 1 || b > nb
        h.clipped += w; b = clamp(b, 1, nb)
    end
    h.w[b] += w; h.total += w; h.sum_x += w * x
    x >= 0 && (h.nonneg += w)
    return
end

"""Weighted percentiles (bin midpoints), the weighted mean and the weighted share of x ≥ 0."""
function summary(h::WHist; ps = (0.05, 0.50, 0.95))
    c = cumsum(h.w) ./ h.total
    mid(b) = h.lo + (b - 0.5) * (h.hi - h.lo) / length(h.w)
    q = Dict{String,Any}("p$(round(Int, 100p))" => mid(something(findfirst(>=(p), c), length(c))) for p in ps)
    return merge(q, Dict{String,Any}("mean" => h.sum_x / h.total, "share_nonneg" => h.nonneg / h.total,
                                      "total_weight" => h.total, "clipped_weight" => h.clipped))
end

"""
    l1_items(delta, mu, alpha_rows, demo_rows, n_demo, rho, s_data, dep, M_pair, G)

The L1 items at (δ*, θ₂*). `mu` is N × R at θ₂*; `alpha_rows(rows)` returns α_ir for those rows
(n × R); `demo_rows(rows, d)` returns D̃ of demographic d for those rows (n × R); `rho` the spread
(the units of α), `s_data` the observed shares, `dep` active deposits, `M_pair[t][q]` the size of
quarter t's market q; `G` the ExactGroups of blp_exact_ift.jl. Returns per-row elasticities and
model shares, the largest |ln s_model − ln s_data| (≈ 0 at a converged δ*: the share model here is
the engine's), and the summaries.
"""
function l1_items(delta::Vector{Float64}, mu::Matrix{Float64}, alpha_rows, demo_rows, n_demo::Int,
                  rho::Vector{Float64}, s_data::Vector{Float64}, dep::Vector{Float64},
                  M_pair::Vector{Vector{Float64}}, G; alo = -20.0, ahi = 20.0, nbins = 400_000)
    N, R = size(mu)
    eta = fill(NaN, N); s_model = fill(NaN, N)
    hrow = WHist(alo, ahi, nbins); hdep = WHist(alo, ahi, nbins); hpop = WHist(alo, ahi, nbins)
    demo_row = zeros(n_demo); demo_dep = zeros(n_demo); demo_pop = zeros(n_demo)
    wrow = 0.0; wdep = 0.0; wpop = 0.0
    for t in eachindex(G.b_rows)
        rb, rd, ranges, w = G.b_rows[t], G.d_rows[t], G.b_ranges[t], G.pair_w[t]
        isempty(ranges) && continue
        nb, nd, np = length(rb), length(rd), length(ranges)
        VB = clamp.(delta[rb] .+ mu[rb, :], -500.0, 500.0)
        VD = clamp.(delta[rd] .+ mu[rd, :], -500.0, 500.0)
        LD = fill(-Inf, R)
        if nd > 0
            m = vec(maximum(VD; dims = 1)); LD = m .+ log.(vec(sum(exp.(VD .- m'); dims = 1)))
        end
        logden = Matrix{Float64}(undef, np, R); pair_of = Vector{Int}(undef, nb)
        for (q, rng) in enumerate(ranges)
            pair_of[rng] .= q
            for r in 1:R
                mx = max(0.0, LD[r]); for a in rng; mx = max(mx, VB[a, r]); end
                sm = exp(-mx) + exp(LD[r] - mx); for a in rng; sm += exp(VB[a, r] - mx); end
                logden[q, r] = mx + log(sm)
            end
        end
        QB = exp.(VB .- logden[pair_of, :])
        rows_all = vcat(rb, rd)
        A = alpha_rows(rows_all)                     # (nb + nd) × R
        dV = similar(A)                              # ∂q/∂V by (row, draw)
        Qall = similar(A)
        Qall[1:nb, :] = QB
        dV[1:nb, :]  = QB .* (1.0 .- QB)
        if nd > 0
            P  = exp.(.-logden)                      # np × R
            W1 = vec(sum(w .* P; dims = 1)); W2 = vec(sum(w .* P .^ 2; dims = 1))
            E  = exp.(VD)
            Qall[nb+1:end, :] = E .* W1'
            dV[nb+1:end, :]   = E .* W1' .- E .^ 2 .* W2'
        end
        s = vec(sum(Qall; dims = 2)) ./ R
        s_model[rows_all] .= s
        eta[rows_all] .= rho[rows_all] ./ s .* (vec(sum(A .* dV; dims = 2)) ./ R)
        # choosers: row i's consumers are draw r with probability q_ir / (R s_i)
        ok = (s .> 0) .& isfinite.(s)
        di = [(ok[a] && isfinite(dep[i]) && dep[i] > 0) ? dep[i] : 0.0 for (a, i) in enumerate(rows_all)]
        PR = ifelse.(ok, Qall ./ (R .* s), 0.0)     # (nb + nd) × R, each ok row sums to 1
        @inbounds for r in 1:R, a in eachindex(rows_all)
            ok[a] || continue
            add!(hrow, A[a, r], PR[a, r]); add!(hdep, A[a, r], di[a] * PR[a, r])
        end
        wrow += count(ok); wdep += sum(di)
        # population: every (B market, draw), weight M_p / R; α and D̃ depend on the market only
        first_rows = [rng[1] for rng in ranges]
        Mq = [(isfinite(M_pair[t][q]) && M_pair[t][q] > 0) ? M_pair[t][q] : 0.0 for q in 1:np]
        @inbounds for r in 1:R, q in 1:np
            Mq[q] > 0 && add!(hpop, A[first_rows[q], r], Mq[q] / R)
        end
        wpop += sum(Mq)
        for d in 1:n_demo
            Dd = demo_rows(rows_all, d)
            demo_row[d] += sum(PR .* Dd)
            demo_dep[d] += sum((di .* PR) .* Dd)
            demo_pop[d] += sum(Mq .* (vec(sum(Dd[first_rows, :]; dims = 2)) ./ R))
        end
    end
    okd = isfinite.(s_data) .& (s_data .> 0) .& isfinite.(s_model) .& (s_model .> 0)
    return Dict{String,Any}(
        "eta" => eta, "s_model" => s_model,
        "share_check_max_abs_log_diff" => any(okd) ? maximum(abs.(log.(s_model[okd]) .- log.(s_data[okd]))) : NaN,
        "alpha_choosers_row" => summary(hrow), "alpha_choosers_deposit" => summary(hdep),
        "alpha_population" => summary(hpop),
        "demo_mean_choosers_row" => demo_row ./ wrow, "demo_mean_choosers_deposit" => demo_dep ./ wdep,
        "demo_mean_population" => demo_pop ./ wpop)
end

"""Row- and deposit-weighted means of the elasticities, for B rows, D rows and all; rows with a
non-finite elasticity or weight are left out and counted."""
function eta_summary(eta, dep, is_b)
    out = Dict{String,Any}()
    for (name, m) in (("all", trues(length(eta))), ("B", is_b), ("D", .!is_b))
        ok = m .& isfinite.(eta)
        okd = ok .& isfinite.(dep) .& (dep .> 0)
        e = eta[ok]
        out[name] = Dict{String,Any}("rows" => count(ok), "rows_dropped" => count(m) - count(ok),
            "row_mean" => isempty(e) ? NaN : sum(e) / length(e),
            "row_median" => isempty(e) ? NaN : sort(e)[cld(length(e), 2)],
            "deposit_mean" => any(okd) ? sum(eta[okd] .* dep[okd]) / sum(dep[okd]) : NaN)
    end
    return out
end
