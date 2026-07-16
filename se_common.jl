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

"""Effective number of clusters G* = G/(1+CV²), CV = coefficient of variation of cluster sizes.
This is the few-cluster degrees-of-freedom used for the Student-t reference (matching desc_3.py's
G*/CV table and the logit tables' t(G*)). With one conglomerate holding ~17% of obs, G*≈7 ≪ G≈506."""
function effective_clusters(cl::AbstractVector)
    counts = Dict{eltype(cl),Int}()
    for c in cl
        counts[c] = get(counts, c, 0) + 1
    end
    sizes = Float64.(collect(values(counts)))
    G = length(sizes)
    G == 0 && return 0.0
    m   = sum(sizes) / G
    cv2 = m > 0 ? (sum((sizes .- m) .^ 2) / G) / m^2 : 0.0
    return G / (1 + cv2)
end

"""Score/no-refit wild cluster bootstrap (Kline–Santos; same primitive as the sleepiness
`cluster_wild_bootstrap`). `IF_cl` (G×K) are per-cluster influence functions for θ̂. Draws `B`
wild-weighted perturbations θ_b = θ̂ + Σ_g w_g·IF_g and returns (se, pval): the bootstrap SD and a
symmetric Wald p, 2·T̄_{G*}(|θ̂_k|/se_k), using a Student-t reference with `dof`=G* effective
clusters (few-cluster correction; Normal if `dof` is not supplied). A degenerate se_k (e.g. a
σ pinned at the σ≥0 bound, where the score is exactly 0) yields pval=NaN, not a spurious 0."""
function wcb_se(theta::AbstractVector{Float64}, IF_cl::AbstractMatrix{Float64};
                B::Int=wcb_reps(), scheme::AbstractString=wcb_scheme(), seed::Int=0,
                dof::Float64=NaN)
    G, K = size(IF_cl)
    (G < 2 || K == 0) && return (fill(NaN, K), fill(NaN, K))
    rng   = MersenneTwister(seed)
    draws = Matrix{Float64}(undef, B, K)
    for b in 1:B
        wv = wild_weights(G, scheme, rng)          # one weight per cluster
        @views draws[b, :] .= theta .+ IF_cl' * wv  # θ_b = θ̂ + Σ_g w_g IF_g
    end
    se   = Float64[std(@view(draws[:, k]); corrected=true) for k in 1:K]
    ref  = (isfinite(dof) && dof > 1) ? TDist(dof) : Normal()   # t(G*) few-cluster reference
    pval = Float64[se[k] > 0 ? 2 * ccdf(ref, abs(theta[k]) / se[k]) : NaN for k in 1:K]
    return se, pval
end

"""Joint cluster-robust SEs for the GMM parameter θ = (θ₁, θ₂), by the analytical sandwich or the
wild cluster bootstrap (same weights/seed as the logit). All inputs are on the VALID rows.

  theta  = vcat(θ₁, θ₂)  (K = K₁+K₂)   Z = instruments (N×L)   X = X_full (N×K₁)
  Ddelta = ∂δ/∂θ₂ (N×K₂)   xi = δ − X·θ₁ (N)   W = (Z'Z/N)⁻¹ (L×L)   cl = per-obs cluster id (N)

Moment g = Z'ξ/N; Jacobian (×N, cancels) D = [−Z'X, Z'Ddelta]; cluster meat S = Σ_g m_g m_g',
m_g = Σ_{i∈g} Z_i ξ_i. Sandwich V = (D'WD)⁻¹ D'W S W D (D'WD)⁻¹ · corr; WCB uses IF_g = −(D'WD)⁻¹ D'W m_g.
Returns (se, pval) over all K params (θ₁ block first).

BOUNDARY PROFILING (`n_sigma`, `bound_tol`): the first `n_sigma` entries of θ₂ are the random-
coefficient σ's, bounded σ≥0. A σ pinned at the bound (|σ̂|<`bound_tol`) has ∂δ/∂σ≈0 — shares are
even in σ, so the score vanishes at σ=0 — hence its `Ddelta` column ≈0. Keeping such a column makes
`DtWD` near-singular, and `inv()` then inflates EVERY parameter's SE (θ₁ included), not just the σ's
own. We therefore PROFILE those σ's out (drop their columns = condition on σ=0, the boundary-correct
treatment, Andrews 1999), compute the covariance on the identified sub-vector, and return `NaN` SE
for the profiled σ's — exactly the σ's the tables flag with a dagger. This is the fix for the
"SEs blow up when a second on-bound σ enters" pathology (4→5 RC)."""
function gmm_cluster_ses(method::AbstractString, theta::Vector{Float64},
                         Z::Matrix{Float64}, X::Matrix{Float64}, Ddelta::Matrix{Float64},
                         xi::Vector{Float64}, W::Matrix{Float64}, cl::Vector{String};
                         B::Int=wcb_reps(), scheme::AbstractString=wcb_scheme(), seed::Int=0,
                         n_sigma::Int=0, bound_tol::Float64=1e-3)
    N, L = size(Z)
    K1   = size(X, 2); K2 = size(Ddelta, 2); K = K1 + K2

    # Profile out on-bound σ's (degenerate ∂δ/∂σ≈0 columns) before forming the Jacobian.
    drop = Int[]                                       # full-param indices to profile out
    for j in 1:min(n_sigma, K2)
        abs(theta[K1 + j]) < bound_tol && push!(drop, K1 + j)
    end
    isempty(drop) || @info "  [se] profiling out $(length(drop)) on-bound σ (param idx $drop) from the covariance"
    θ2keep  = [j for j in 1:K2 if !((K1 + j) in drop)]           # kept θ₂ columns (into Ddelta)
    keepidx = vcat(collect(1:K1), [K1 + j for j in θ2keep])      # kept full-param indices, in order
    Dd_k    = K2 > 0 ? Ddelta[:, θ2keep] : Ddelta
    Kk      = length(keepidx)

    D    = hcat(-Z' * X, Z' * Dd_k)                    # (L × Kk), degenerate cols removed
    uc   = unique(cl); G = length(uc)
    Mcl  = zeros(G, L)                                 # cluster meats m_g (rows)
    for (gi, c) in enumerate(uc)
        cm = cl .== c
        Mcl[gi, :] = Z[cm, :]' * xi[cm]
    end
    DtW   = D' * W                                     # (Kk × L)
    DtWD  = DtW * D                                    # (Kk × Kk)
    bread = try inv(DtWD) catch; pinv(DtWD) end
    gstar = effective_clusters(cl)                     # few-cluster t(G*) df (≈7 here, not G≈506)

    theta_k   = theta[keepidx]
    se_full   = fill(NaN, K)                           # profiled σ's stay NaN (dagger in the tables)
    pval_full = fill(NaN, K)
    if method == "sandwich"
        corr = G > 1 ? (G / (G - 1)) * ((N - 1) / (N - Kk)) : 1.0
        V    = bread * (DtW * (Mcl' * Mcl) * DtW') * bread' .* corr
        se_k = sqrt.(max.(diag(V), 0.0))
        ref  = gstar > 1 ? TDist(gstar) : Normal()     # t(G*) few-cluster reference
        pv_k = Float64[se_k[k] > 0 ? 2 * ccdf(ref, abs(theta_k[k]) / se_k[k]) : NaN for k in 1:Kk]
    else  # "wcb": per-cluster GMM influence functions IF_g = −(D'WD)⁻¹ D'W m_g, row g of IF_cl
        IF_cl = -(bread * DtW * Mcl')'                 # (G × Kk)
        se_k, pv_k = wcb_se(theta_k, Matrix(IF_cl); B=B, scheme=scheme, seed=seed, dof=gstar)
    end
    se_full[keepidx]   = se_k
    pval_full[keepidx] = pv_k
    return se_full, pval_full
end
