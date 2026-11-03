# Local CPU smoke of the edited engine (blp_engine_gpu.jl, CPU IFT path): the exact IFT gradient of
# gmm_fg! against central finite differences of Q, the engine's δ-Jacobian against finite differences
# of δ*, and fd_check_report, on two whole quarters of the E3 demand parquet with synthetic draws.
#   julia --project=<repo> --threads=4 smoke_exact_engine.jl <repo> <demand parquet>
const REPO = ARGS[1]
include(joinpath(REPO, "blp_engine_gpu.jl"))
using Random, Statistics, LinearAlgebra, DataFrames, Parquet2, Printf, Dates
say(msg) = (println("[", Dates.format(Dates.now(), "HH:MM:SS"), "] ", msg); flush(stdout))

df_all = DataFrame(Parquet2.Dataset(ARGS[2]); copycols=true)
qs = sort(unique(string.(df_all.time_id)))
keep = Set(qs[[10, 30]])
df = df_all[in.(string.(df_all.time_id), Ref(keep)), :]
N = nrow(df)
say("subsample: quarters $(sort(collect(keep))), $N rows")

R = 50; rng = MersenneTwister(7)
coef_dim = 1 + L_PROD
nu_draws = randn(rng, R, coef_dim)
keys_ = unique(collect(zip(string.(df.mca_code), string.(df.time_id))))
key_index = Dict(k => i + 1 for (i, k) in enumerate(keys_))
draws_3d = randn(rng, length(keys_) + 1, R, length(D_COLS)); draws_3d[1, :, :] .= 0.0
obs_key_idx = [key_index[(string(df.mca_code[i]), string(df.time_id[i]))] for i in 1:N]

sigma_indices, pi_interactions, n_params = build_theta2_structure("ext1")
dep_types = Int.(coalesce.(df.deposit_type, 0))
spread_cols, x_mat, Z_mat, iv_cols = build_regressor_matrices(df)
prod_vec = rc_design(spread_cols, x_mat)
Z, iv_avail = moment_instruments(Z_mat, iv_cols)
W = try inv(Z' * Z ./ N) catch; Matrix(I(length(iv_avail)) * 1.0) end
spread_hat = project_endogenous_spreads(spread_cols, hcat(x_mat, Z_mat), dep_types)
X_full = hcat(spread_cols, x_mat); X_hat = hcat(spread_hat, x_mat)
valid = BitVector(all.(isfinite, eachrow(X_hat)))
pc = build_precomp(df, Z, X_full, X_hat, valid, string.(df.CodConglomeradoPrudencial))
buf = allocate_hot_buffers(N, sum(pc.b_mask), sum(pc.d_mask), R, length(pc.unique_pairs),
                           length(pc.unique_times), coef_dim, length(pi_interactions))
precompute_pi_products!(buf, prod_vec, draws_3d, obs_key_idx, pi_interactions, coef_dim)

θ  = [0.05, 0.4, -0.2, 0.3, 0.1]
lo = fill(-5.0, n_params); hi = fill(5.0, n_params); lo[1:length(sigma_indices)] .= 0.0
δ0 = zeros(N); δ0[pc.b_mask] .= pc.ln_s_data_B_cond[pc.b_mask]; δ0[pc.d_mask] .= pc.ln_s_data_D[pc.d_mask]
function contract_at(t, dstart; tol = 1e-13)
    mu_at!(buf, prod_vec, nu_draws, t, sigma_indices, pi_interactions, R, coef_dim)
    d = copy(dstart)
    conv, it, _ = blp_contraction!(buf, d, pc, R; tol = tol, max_iter = 100000)
    return d, conv, it
end
δstar, conv0, it0 = contract_at(θ, δ0)
say("δ* at θ: conv $conv0 in $it0 evals")
function Qfun(t)
    d = contract_at(t, δstar)[1]
    xi = estimate_theta1(d, pc)[2]
    g = compute_gmm_moments(xi, pc)
    return dot(g, W * g)
end

# 1. gmm_fg!'s exact gradient against central finite differences of Q
G = zeros(n_params)
t0 = time()
Q0 = gmm_fg!(G, θ, buf, prod_vec, nu_draws, sigma_indices, pi_interactions, R, coef_dim, W, 1e-13, 100000,
             copy(δstar), pc)
t_grad = time() - t0
h = 1e-4
Gfd = [(Qfun(θ .+ h .* (1:n_params .== k)) - Qfun(θ .- h .* (1:n_params .== k))) / (2h) for k in 1:n_params]
cosg = dot(G, Gfd) / (norm(G) * norm(Gfd)); relg = norm(G .- Gfd) / norm(Gfd)
@printf("gradient: exact %s\n          fd    %s\n          cos %.6f, rel %.2e (gmm_fg! with gradient %.1f s)\n",
        string(round.(G, sigdigits = 6)), string(round.(Gfd, sigdigits = 6)), cosg, relg, t_grad)

flush(stdout); say("FD gradient done")
# 2. the engine's δ-Jacobian against finite differences of δ*
mu_at!(buf, prod_vec, nu_draws, θ, sigma_indices, pi_interactions, R, coef_dim)
dlns = dlns_dtheta(() -> compute_model_shares!(buf, δstar, pc, R), θ, buf, prod_vec, nu_draws,
                   sigma_indices, pi_interactions, R, coef_dim, pc, N)
t0 = time(); Jδ = exact_jacobian_from_dlns(δstar, buf, dlns, pc); t_jac = time() - t0
Jfd = hcat([(contract_at(θ .+ h .* (1:n_params .== k), δstar)[1] .- contract_at(θ .- h .* (1:n_params .== k), δstar)[1]) ./ (2h)
            for k in 1:n_params]...)
colrel = [norm(Jδ[:, k] .- Jfd[:, k]) / norm(Jfd[:, k]) for k in 1:n_params]
@printf("δ-Jacobian: column rel err vs fd %s (exact solve %.2f s, %d threads)\n",
        string(round.(colrel, sigdigits = 3)), t_jac, Threads.nthreads())

flush(stdout); say("δ-Jacobian check done")
# 3. fd_check_report (the convergence report every fit now writes)
rep = fd_check_report(Qfun, () -> (g = zeros(n_params);
                                   gmm_fg!(g, θ, buf, prod_vec, nu_draws, sigma_indices, pi_interactions, R,
                                           coef_dim, W, 1e-13, 100000, copy(δstar), pc); g), θ, lo, hi)
println("fd_check_report keys: ", sort(collect(keys(rep))))
@printf("  projected FD norm %.4e, exact projected norm %.4e, best dQ %.3e\n",
        rep["grad_fd_projected_norm"], rep["grad_exact_projected_norm"], rep["best_dQ"])
println("JSON-ready: ", json_finite(rep) isa Dict)
ok = cosg > 0.99999 && relg < 1e-3 && maximum(colrel) < 1e-3 && !haskey(rep, "error")
say(ok ? "SMOKE PASS" : "SMOKE FAIL")
