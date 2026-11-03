# Local check of blp_l1_items.jl's price-coefficient decomposition against the engine's compute_mu!:
# with only the spread's σ and π nonzero (π(ln assets × gdp) = 0), μ_ir must equal
# ρ_i (σ_ρ ν_rρ + Σ_d π_ρd D̃_{m(i) r d}) = ρ_i (α_ir − α). Two quarters of the E3 parquet, synthetic draws.
#   julia --project=<repo> diagnostics/check_l1_alpha.jl <repo> <demand parquet>
const REPO = ARGS[1]
include(joinpath(REPO, "blp_engine_gpu.jl"))
using Random, DataFrames, Parquet2, Printf

df_all = DataFrame(Parquet2.Dataset(ARGS[2]); copycols=true)
qs = sort(unique(string.(df_all.time_id)))
df = df_all[in.(string.(df_all.time_id), Ref(Set(qs[[10, 30]]))), :]
N = nrow(df); R = 30; rng = MersenneTwister(3)
coef_dim = 1 + L_PROD
nu_draws = randn(rng, R, coef_dim)
keys_ = unique(collect(zip(string.(df.mca_code), string.(df.time_id))))
key_index = Dict(k => i + 1 for (i, k) in enumerate(keys_))
draws_3d = randn(rng, length(keys_) + 1, R, length(D_COLS)); draws_3d[1, :, :] .= 0.0
obs_key_idx = [key_index[(string(df.mca_code[i]), string(df.time_id[i]))] for i in 1:N]
sigma_indices, pi_interactions, n_params = build_theta2_structure("ext1")
spread_cols, x_mat, Z_mat, iv_cols = build_regressor_matrices(df)
prod_vec = rc_design(spread_cols, x_mat)
X_hat = hcat(spread_cols, x_mat); valid = BitVector(all.(isfinite, eachrow(X_hat)))
Z, _ = moment_instruments(Z_mat, iv_cols)
pc = build_precomp(df, Z, hcat(spread_cols, x_mat), X_hat, valid, string.(df.CodConglomeradoPrudencial))
buf = allocate_hot_buffers(N, sum(pc.b_mask), sum(pc.d_mask), R, length(pc.unique_pairs),
                           length(pc.unique_times), coef_dim, length(pi_interactions))
precompute_pi_products!(buf, prod_vec, draws_3d, obs_key_idx, pi_interactions, coef_dim)
println("sigma_indices $sigma_indices; pi_interactions $pi_interactions")

θ = zeros(n_params); θ[1] = 0.3
for (j, (c, d)) in enumerate(pi_interactions); c == 1 && (θ[length(sigma_indices) + j] = 0.2 * j); end
sv, pv = unpack_theta2(θ, sigma_indices, pi_interactions)
compute_mu!(buf, prod_vec, nu_draws, sv, sigma_indices, pv, R, coef_dim)

# the decomposition blp_l1_items.jl uses, with α = 0
s_rho  = (pos = findfirst(==(1), sigma_indices); pos === nothing ? 0.0 : sv[pos])
pi_rho = [(d, pv[j]) for (j, (c, d)) in enumerate(pi_interactions) if c == 1]
A = fill(0.0, N, R) .+ s_rho .* nu_draws[:, 1]'
for (d, p) in pi_rho; A .+= p .* draws_3d[obs_key_idx, :, d]; end
err = maximum(abs.(buf.mu .- spread_cols .* A))
@printf("max |μ − ρ (α_ir − α)| = %.2e (scale %.2e)\n", err, maximum(abs.(buf.mu)))
println(err < 1e-10 ? "ALPHA CHECK PASS" : "ALPHA CHECK FAIL")
