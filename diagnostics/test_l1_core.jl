# Local test of blp_l1_core.jl on a synthetic share system (the model of blp_exact_ift.jl):
# elasticities against finite differences of the shares, the α percentiles and means against exact
# weighted quantiles, the demographic means against a direct sum, and the share check.
#   julia --threads=4 diagnostics/test_l1_core.jl <repo root>
using LinearAlgebra, Random, Printf, Statistics
include(joinpath(ARGS[1], "blp_exact_ift.jl"))
include(joinpath(ARGS[1], "blp_l1_core.jl"))

"""Reference shares: plain loops over quarters, markets, draws."""
function ref_shares(V, b_mask, mkt, tim, w_of_mkt)
    N, R = size(V); s = zeros(N)
    for t in unique(tim)
        D = [i for i in 1:N if !b_mask[i] && tim[i] == t]
        for p in unique([mkt[i] for i in 1:N if b_mask[i] && tim[i] == t])
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

"""Weighted quantile by sorting."""
function wquant(x, w, p)
    o = sortperm(x); c = cumsum(w[o]) ./ sum(w)
    return x[o][something(findfirst(>=(p), c), length(o))]
end

rng = MersenneTwister(5)
sizes = [(1, [1, 2, 3, 5], 3), (2, [1, 4], 0), (3, [2, 2, 6], 4)]
b_mask = Bool[]; mkt = Int[]; tim = Int[]; pair_time = Int[]; m = 0
for (t, ms, nd) in sizes
    for n in ms
        global m += 1; push!(pair_time, t)
        append!(b_mask, fill(true, n)); append!(mkt, fill(m, n)); append!(tim, fill(t, n))
    end
    append!(b_mask, fill(false, nd)); append!(mkt, fill(0, nd)); append!(tim, fill(t, nd))
end
perm = randperm(rng, length(b_mask)); b_mask, mkt, tim = b_mask[perm], mkt[perm], tim[perm]
N = length(b_mask); R = 200
w = rand(rng, m) .+ 0.2
for t in unique(pair_time); idx = findall(==(t), pair_time); w[idx] ./= sum(w[idx]); end
G = build_exact_groups(b_mask, mkt[b_mask], tim[.!b_mask], pair_time, w)
delta = randn(rng, N) .- 1.0
mu = randn(rng, N, R) .* 0.7
# α and two demographics vary by market and draw (D rows: their own key), as in the engine
key = [b_mask[i] ? mkt[i] : 100 + i for i in 1:N]
demo = Dict(k => randn(rng, R, 2) for k in unique(key))
alpha = vcat([(-0.3 .+ 0.8 .* demo[key[i]][:, 1]')  for i in 1:N]...)       # N × R
alpha_rows(rows) = alpha[rows, :]
demo_rows(rows, d) = vcat([demo[key[i]][:, d]' for i in rows]...)
rho = rand(rng, N) .* 3 .+ 0.5
dep = rand(rng, N) .* 10
V = clamp.(delta .+ mu, -500.0, 500.0)
s_ref = ref_shares(V, b_mask, mkt, tim, w)
M_pair = [[1.0 + 0.5q for q in 1:length(G.b_ranges[t])] for t in eachindex(G.b_rows)]

out = l1_items(delta, mu, alpha_rows, demo_rows, 2, rho, s_ref, dep, M_pair, G; alo = -10.0, ahi = 10.0, nbins = 200_000)

# 1. elasticities against finite differences: ρ_j moves V_jr by α_jr per unit
h = 1e-6; errs = Float64[]
for j in 1:N
    Vp = copy(V); Vp[j, :] .+= h .* alpha[j, :]
    Vm = copy(V); Vm[j, :] .-= h .* alpha[j, :]
    e_fd = (log(ref_shares(Vp, b_mask, mkt, tim, w)[j]) - log(ref_shares(Vm, b_mask, mkt, tim, w)[j])) / (2h) * rho[j]
    push!(errs, abs(out["eta"][j] - e_fd) / max(abs(e_fd), 1e-12))
end
@printf("elasticity vs fd: max rel err %.1e over %d rows (B and D)\n", maximum(errs), N)

# 2. α among choosers, exact: weight q_ir / (R s_i) per (i, r), times dep_i for the deposit weight
q = zeros(N, R)
# per-draw choice probabilities: the reference model on one draw at a time
for r in 1:R
    q[:, r] = ref_shares(V[:, r:r], b_mask, mkt, tim, w)
end
x = vec(alpha); wr = vec(q ./ (R .* s_ref)); wd = vec(dep .* q ./ (R .* s_ref))
for (name, ww, key2) in (("row", wr, "alpha_choosers_row"), ("deposit", wd, "alpha_choosers_deposit"))
    sm = out[key2]
    @printf("α choosers (%s): mean %.6f vs %.6f; share ≥ 0 %.6f vs %.6f; p5 %.4f vs %.4f; p50 %.4f vs %.4f; p95 %.4f vs %.4f\n",
            name, sm["mean"], sum(x .* ww) / sum(ww), sm["share_nonneg"], sum(ww[x .>= 0]) / sum(ww),
            sm["p5"], wquant(x, ww, 0.05), sm["p50"], wquant(x, ww, 0.50), sm["p95"], wquant(x, ww, 0.95))
end
# 3. demographic means among choosers (row weight)
for d in 1:2
    Dd = vcat([demo[key[i]][:, d]' for i in 1:N]...)
    @printf("D̃%d choosers (row): %.6f vs %.6f\n", d, out["demo_mean_choosers_row"][d], sum(vec(Dd) .* wr) / sum(wr))
end
@printf("share check: %.1e\n", out["share_check_max_abs_log_diff"])
ok = maximum(errs) < 1e-6 && out["share_check_max_abs_log_diff"] < 1e-12 &&
     abs(out["alpha_choosers_row"]["mean"] - sum(x .* wr) / sum(wr)) < 1e-10
println(ok ? "PASS" : "FAIL")
