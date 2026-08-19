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

DEGENERATE-DIRECTION PROFILING (`n_sigma`, `bound_tol`, `jac_tol`): a θ₂ coordinate whose
moment-Jacobian column `Z'∂δ/∂θ₂ⱼ` is ≈0 makes `DtWD` near-singular, and `inv()` then inflates EVERY
parameter's SE (θ₁ and α included), not just that coordinate's own. Two distinct causes:
  (a) ON-BOUND σ — the first `n_sigma` θ₂ entries are the σ's, bounded σ≥0. A σ pinned at the bound
      (|σ̂|<`bound_tol`) has ∂δ/∂σ≈0 because shares are even in σ, so the score vanishes at σ=0.
  (b) FLAT direction (any σ OR π) — relative Jacobian column norm < `jac_tol`. The objective does
      not respond to the parameter at all; the ladder signature is Q frozen across consecutive
      stages while the newly freed π optimizes to ~0.
Both are PROFILED OUT (drop the column = condition on that coordinate, the boundary-correct
treatment for (a), Andrews 1999); the covariance is computed on the identified sub-vector and `NaN`
SE returned for the profiled coordinates — exactly the ones the tables flag with a dagger. (a) fixes
the "SEs blow up when a second on-bound σ enters" pathology (4→5 RC); (b) fixes the same blow-up
driven by unidentified π's in the ext1/ext2 stages. NOTE: this addresses NUMERICAL degeneracy only —
a parameter with a genuinely non-zero but small Jacobian is weakly (not un-) identified, and its
large SE is real and must be reported, not profiled away."""
function gmm_cluster_ses(method::AbstractString, theta::Vector{Float64},
                         Z::Matrix{Float64}, X::Matrix{Float64}, Ddelta::Matrix{Float64},
                         xi::Vector{Float64}, W::Matrix{Float64}, cl::Vector{String};
                         B::Int=wcb_reps(), scheme::AbstractString=wcb_scheme(), seed::Int=0,
                         n_sigma::Int=0, bound_tol::Float64=1e-3, jac_tol::Float64=1e-8)
    N, L = size(Z)
    K1   = size(X, 2); K2 = size(Ddelta, 2); K = K1 + K2

    # Profile out degenerate θ₂ directions before forming the Jacobian. Two ways a direction dies:
    #  (a) ON-BOUND σ (|σ̂| < bound_tol): shares are even in σ, so ∂δ/∂σ ≈ 0 at σ=0; conditioning on
    #      σ=0 is the boundary-correct treatment (Andrews 1999).
    #  (b) FLAT direction (any σ OR π): its moment-Jacobian column Z'∂δ/∂θ₂ⱼ is numerically
    #      negligible relative to the largest column of D — the objective simply does not respond
    #      to it. Signature in the stage ladder: Q frozen across stages while the added π optimizes
    #      to ~0 (e.g. the rc4→ext1→ext2 ladder with π(fgc×65+) ≡ 0.0000).
    # Either way the column is ~0, DtWD is near-singular, and inv() inflates EVERY parameter's SE —
    # θ₁'s (incl. α) included — not just the offending parameter's own.
    ZtX  = Z' * X                                                 # (L × K1)
    ZtDd = K2 > 0 ? Z' * Ddelta : Matrix{Float64}(undef, L, 0)    # (L × K2)
    _cn(A, j) = sqrt(sum(abs2, view(A, :, j)))                    # column norm (no LinearAlgebra dep)
    scale = 0.0
    for k in 1:K1; scale = max(scale, _cn(ZtX, k)); end
    for j in 1:K2; scale = max(scale, _cn(ZtDd, j)); end
    drop = Int[]; why = String[]                       # full-param indices to profile out
    for j in 1:K2
        if j <= n_sigma && abs(theta[K1 + j]) < bound_tol
            push!(drop, K1 + j); push!(why, "θ₂[$j] σ on-bound")
        elseif scale > 0 && _cn(ZtDd, j) / scale < jac_tol
            push!(drop, K1 + j); push!(why, "θ₂[$j] flat (rel. Jacobian norm < $jac_tol)")
        end
    end
    isempty(drop) || @info "  [se] profiling out $(length(drop)) degenerate θ₂ direction(s): $(join(why, "; "))"
    θ2keep  = [j for j in 1:K2 if !((K1 + j) in drop)]           # kept θ₂ columns (into Ddelta)
    keepidx = vcat(collect(1:K1), [K1 + j for j in θ2keep])      # kept full-param indices, in order
    Kk      = length(keepidx)

    D    = hcat(-ZtX, K2 > 0 ? ZtDd[:, θ2keep] : ZtDd)  # (L × Kk), degenerate cols removed
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
