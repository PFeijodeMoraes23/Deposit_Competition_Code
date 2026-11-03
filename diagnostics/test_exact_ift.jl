# Local test of blp_exact_ift.jl on a synthetic share system: shares and ∂δ*/∂θ against brute force,
# and the conditioning of the log-share system against the share system.
#   julia --threads=4 test_exact_ift.jl <repo root>
using LinearAlgebra, Random, Printf
include(joinpath(ARGS[1], "blp_exact_ift.jl"))

"""Reference shares, written independently of blp_exact_ift.jl: plain loops over quarters, markets, draws."""
function ref_shares(delta, mu, b_mask, mkt, tim, w_of_mkt)
    N, R = size(mu); s = zeros(N)
    V = clamp.(delta .+ mu, -500.0, 500.0)
    for t in unique(tim)
        D = [i for i in 1:N if !b_mask[i] && tim[i] == t]
        mkts = unique([mkt[i] for i in 1:N if b_mask[i] && tim[i] == t])
        for p in mkts
            B = [i for i in 1:N if b_mask[i] && mkt[i] == p]
            for r in 1:R
                den = 1.0 + sum(exp(V[i, r]) for i in B) + (isempty(D) ? 0.0 : sum(exp(V[k, r]) for k in D))
                for i in B; s[i] += exp(V[i, r]) / den / R; end
                for k in D; s[k] += w_of_mkt[p] * exp(V[k, r]) / den / R; end
            end
        end
    end
    return s
end

function case(name, shift; seed = 1, R = 40, tiny = 0.0)
    rng = MersenneTwister(seed)
    # quarter 1: 4 markets of sizes 1, 2, 3, 5 and 3 D rows; quarter 2: 2 markets of sizes 1, 4, no D row
    sizes = [(1, [1, 2, 3, 5], 3), (2, [1, 4], 0)]
    b_mask = Bool[]; mkt = Int[]; tim = Int[]; pair_time = Int[]; m = 0
    for (t, ms, nd) in sizes
        for n in ms
            m += 1; push!(pair_time, t)
            append!(b_mask, fill(true, n)); append!(mkt, fill(m, n)); append!(tim, fill(t, n))
        end
        append!(b_mask, fill(false, nd)); append!(mkt, fill(0, nd)); append!(tim, fill(t, nd))
    end
    perm = randperm(rng, length(b_mask))                 # interleave B and D rows
    b_mask, mkt, tim = b_mask[perm], mkt[perm], tim[perm]
    N = length(b_mask)
    w = rand(rng, m) .+ 0.2
    for t in unique(pair_time); idx = findall(==(t), pair_time); w[idx] ./= sum(w[idx]); end
    delta = randn(rng, N) .+ shift; mu = randn(rng, N, R)
    # `tiny`: push one D row and one B row of each quarter far down, so shares span many orders
    tiny != 0 && for t in unique(tim)
        d = findfirst(i -> !b_mask[i] && tim[i] == t, 1:N); d === nothing || (delta[d] -= tiny)
        b = findfirst(i -> b_mask[i] && tim[i] == t, 1:N); delta[b] -= tiny
    end
    b_mkt_idx = mkt[b_mask]; d_time_enc = tim[.!b_mask]
    G = build_exact_groups(b_mask, b_mkt_idx, d_time_enc, pair_time, w)
    s = ref_shares(delta, mu, b_mask, mkt, tim, w)
    K = 3; dsdθ = randn(rng, N, K) .* 1e-2 .* s           # ∂s/∂θ scales with s
    Jδ, info = exact_delta_jacobian(delta, mu, dsdθ ./ s, G; diagnostics = true)
    h = 1e-5; J = zeros(N, N)
    for l in 1:N
        e = zeros(N); e[l] = h
        J[:, l] = (ref_shares(delta .+ e, mu, b_mask, mkt, tim, w) .- ref_shares(delta .- e, mu, b_mask, mkt, tim, w)) ./ (2h)
    end
    Jl = J ./ s                                          # the log-share Jacobian, by rows
    x_ref = -(Jl \ (dsdθ ./ s))
    x_diag = -(dsdθ ./ s) ./ max.(1.0 .- s, 1e-4)        # the diagonal approximation
    rel(a, b) = norm(a .- b) / norm(b)
    s0 = filter(isfinite, info["s0_row"])
    @printf("%-36s shares %.0e..%.0e  outside [%.3f, %.3f]  cond J %.1e, log-share J %.1e  exact vs brute %.1e  diagonal vs brute %.2f  singular %d\n",
            name, minimum(s), maximum(s), minimum(s0), maximum(s0), cond(J), cond(Jl),
            rel(Jδ, x_ref), rel(x_diag, x_ref), info["n_singular_blocks"])
    return rel(Jδ, x_ref)
end

println("threads: ", Threads.nthreads())
errs = [case("large outside share (shift -3)", -3.0), case("moderate (shift 0)", 0.0),
        case("small outside share (shift +3)", 3.0), case("tiny outside share (shift +6)", 6.0, seed = 2),
        case("tiny shares (one B, one D row -18)", 0.0, seed = 3, tiny = 18.0)]
println(maximum(errs) < 1e-6 ? "PASS" : "FAIL")
