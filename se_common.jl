# se_common.jl
# ============
# Shared standard-error machinery for demand estimation: a score/no-refit WILD CLUSTER BOOTSTRAP
# (matching the sleepiness estimation's inference, utils/sleep_links.py) and the analytical
# cluster-robust GMM sandwich, selected by BLP_SE_METHOD. Included by both blp_1_logit.jl (pure
# logit θ₁) and blp_1_estimation.jl / blp_gpu_engine.jl (RC-BLP θ₁+θ₂) so every parameter's SE is
# produced "in the same manner". Requires (from the includer's `using`s): Random, Statistics,
# Distributions, LinearAlgebra.

"SE method: \"wcb\" (wild cluster bootstrap, default) or \"sandwich\" (analytical cluster-robust)."
se_method()  = lowercase(get(ENV, "BLP_SE_METHOD", "wcb"))
"WCB replications (default 999, matching the sleepiness SLEEP_BOOT_B)."
wcb_reps()   = parse(Int, get(ENV, "BLP_WCB_REPS", get(ENV, "SLEEP_BOOT_B", "999")))
"WCB weight scheme: \"webb\" (default) or \"rademacher\" (matching SLEEP_BOOT_SCHEME)."
wcb_scheme() = lowercase(get(ENV, "BLP_WCB_SCHEME", get(ENV, "SLEEP_BOOT_SCHEME", "webb")))

"""Wild bootstrap weights, one per cluster: 6-point Webb (default; robust to few/imbalanced
clusters) or 2-point Rademacher. Mirrors utils/sleep_links.py `_wild_weights`."""
function wild_weights(n::Int, scheme::AbstractString, rng::AbstractRNG)
    if scheme == "webb"
        vals = (-sqrt(1.5), -1.0, -sqrt(0.5), sqrt(0.5), 1.0, sqrt(1.5))
        return Float64[vals[rand(rng, 1:6)] for _ in 1:n]
    end
    return Float64[rand(rng) < 0.5 ? -1.0 : 1.0 for _ in 1:n]   # Rademacher
end

"""Score/no-refit wild cluster bootstrap (Kline–Santos; same primitive as the sleepiness
`cluster_wild_bootstrap`). `IF_cl` (G×K) are per-cluster influence functions for θ̂. Draws `B`
wild-weighted perturbations θ_b = θ̂ + Σ_g w_g·IF_g and returns (se, pval): the bootstrap SD and a
symmetric Wald p, 2·Φ̄(|θ̂_k|/se_k). No small-sample factor (unit-variance weights supply it)."""
function wcb_se(theta::AbstractVector{Float64}, IF_cl::AbstractMatrix{Float64};
                B::Int=wcb_reps(), scheme::AbstractString=wcb_scheme(), seed::Int=0)
    G, K = size(IF_cl)
    (G < 2 || K == 0) && return (fill(NaN, K), fill(NaN, K))
    rng   = MersenneTwister(seed)
    draws = Matrix{Float64}(undef, B, K)
    for b in 1:B
        wv = wild_weights(G, scheme, rng)          # one weight per cluster
        @views draws[b, :] .= theta .+ IF_cl' * wv  # θ_b = θ̂ + Σ_g w_g IF_g
    end
    se   = Float64[std(@view(draws[:, k]); corrected=true) for k in 1:K]
    pval = Float64[se[k] > 0 ? 2 * ccdf(Normal(), abs(theta[k]) / se[k]) : 0.0 for k in 1:K]
    return se, pval
end

"""Joint cluster-robust SEs for the GMM parameter θ = (θ₁, θ₂), by the analytical sandwich or the
wild cluster bootstrap (same weights/seed as the logit). All inputs are on the VALID rows.

  theta  = vcat(θ₁, θ₂)  (K = K₁+K₂)   Z = instruments (N×L)   X = X_full (N×K₁)
  Ddelta = ∂δ/∂θ₂ (N×K₂)   xi = δ − X·θ₁ (N)   W = (Z'Z/N)⁻¹ (L×L)   cl = per-obs cluster id (N)

Moment g = Z'ξ/N; Jacobian (×N, cancels) D = [−Z'X, Z'Ddelta]; cluster meat S = Σ_g m_g m_g',
m_g = Σ_{i∈g} Z_i ξ_i. Sandwich V = (D'WD)⁻¹ D'W S W D (D'WD)⁻¹ · corr; WCB uses IF_g = −(D'WD)⁻¹ D'W m_g.
Returns (se, pval) over all K params (θ₁ block first)."""
function gmm_cluster_ses(method::AbstractString, theta::Vector{Float64},
                         Z::Matrix{Float64}, X::Matrix{Float64}, Ddelta::Matrix{Float64},
                         xi::Vector{Float64}, W::Matrix{Float64}, cl::Vector{String};
                         B::Int=wcb_reps(), scheme::AbstractString=wcb_scheme(), seed::Int=0)
    N, L = size(Z)
    K1   = size(X, 2); K2 = size(Ddelta, 2); K = K1 + K2
    D    = hcat(-Z' * X, Z' * Ddelta)                 # (L × K)
    uc   = unique(cl); G = length(uc)
    Mcl  = zeros(G, L)                                 # cluster meats m_g (rows)
    for (gi, c) in enumerate(uc)
        cm = cl .== c
        Mcl[gi, :] = Z[cm, :]' * xi[cm]
    end
    DtW   = D' * W                                     # (K × L)
    DtWD  = DtW * D                                    # (K × K)
    bread = try inv(DtWD) catch; pinv(DtWD) end
    if method == "sandwich"
        corr = G > 1 ? (G / (G - 1)) * ((N - 1) / (N - K)) : 1.0
        V    = bread * (DtW * (Mcl' * Mcl) * DtW') * bread' .* corr
        se   = sqrt.(max.(diag(V), 0.0))
        pval = Float64[se[k] > 0 ? 2 * ccdf(Normal(), abs(theta[k]) / se[k]) : 0.0 for k in 1:K]
        return se, pval
    else  # "wcb": per-cluster GMM influence functions IF_g = −(D'WD)⁻¹ D'W m_g, row g of IF_cl
        IF_cl = -(bread * DtW * Mcl')'                 # (G × K)
        return wcb_se(theta, Matrix(IF_cl); B=B, scheme=scheme, seed=seed)
    end
end
