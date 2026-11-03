# blp_l1_items.jl — read-only: the L1 items (handoff 2026-11-02, M5) of an ext1 fit, on the production
# data with the engine's setup. At the fit's (δ*, θ₂*) it reports, from blp_l1_core.jl:
#   the random-coefficients own-spread elasticity of every demand row, η_i = (ρ_i/s_i) mean_r α_ir ∂q_ir/∂V_ir,
#     averaged over rows and weighted by active deposits (B rows, D rows, all), beside the plug-in
#     α̂ · mean ρ(1 − s) that demand.elast.blp.* reports;
#   the distribution of the consumer's own price coefficient α_ir = α + σ_ρ ν_rρ + Σ_d π_ρd D̃_rd among
#     the depositors who chose each row (row- and deposit-weighted) and in the markets' population:
#     mean, share with α_ir ≥ 0, 5th, 50th and 95th percentiles;
#   the means of each demographic D̃ under the same weights;
#   the plug-in at both weightings, α̂ · mean ρ(1 − s) over rows and weighted by active deposits, so
#     the change from the plug-in to the RC elasticity splits into weighting and heterogeneity;
#   with L1_MEAN_PHI (the mean φ̂ as a fraction, sleep.meanphi_frac.E<k>), every elasticity's
#     on-impact counterpart, × (1 − mean φ̂), as demand.impact.blp.* is built.
# Checks printed with it: the largest |ln s_model − ln s_data| (≈ 0 at a converged δ*) and the
# contraction's convergence.
#
# Writes only DIAG_OUT/l1_items_E<k>[_<label>].json and its log. Environment: L1_ROUTINE (3 | 4),
# L1_CHECKPOINT (an ext1 checkpoint .jls with its .bin; default the production one), L1_LABEL (a tag
# for the output name, default "production"), L1_MEAN_PHI (optional), DIAG_OUT. Run through
# blp_l1_items_job.sh.

isdefined(Main, :main_gpu_ift) || include(joinpath(@__DIR__, "blp_rc.jl"))
isdefined(Main, :ExactGroups) || include(joinpath(@__DIR__, "blp_exact_ift.jl"))
include(joinpath(@__DIR__, "blp_l1_core.jl"))
using Serialization, JSON3, LinearAlgebra, Statistics, CUDA, Dates

const K     = parse(Int, ENV["L1_ROUTINE"])
const OUT   = ENV["DIAG_OUT"]
const LABEL = get(ENV, "L1_LABEL", "production")
const SPEC  = 12
const RD    = 2000
const SEED  = 42
const TOL   = 1e-12
const CAP   = 20000
const MEAN_PHI = parse(Float64, get(ENV, "L1_MEAN_PHI", "NaN"))

finite_json(x::AbstractFloat) = isfinite(x) ? x : nothing
finite_json(x::AbstractVector) = Any[finite_json(v) for v in x]
finite_json(x::AbstractDict) = Dict{String,Any}(string(k) => finite_json(v) for (k, v) in x)
finite_json(x) = x

mkpath(OUT)
println("=" ^ 72)
println("blp_l1_items: E$K, spec $SPEC, R $RD, label $LABEL")
println("=" ^ 72)

# ── The engine's setup for ext1, as in blp_diag_ext1_exact_ift.jl ─────────────────────────────────
in_dir, drw, out_dir = get_paths(true)
in_dirs = demand_search_dirs(in_dir, out_dir)
prefix  = demand_prefix(K, in_dirs)
prefix === nothing && error("no demand parquet for E$K in $in_dirs")
ENV["BLP_DELTA_SUFFIX"] = ""; ENV["BLP_OUTPUT_SUFFIX"] = ""; ENV["BLP_DEMAND_PREFIX"] = prefix
args = Dict{String,Any}("stage" => "ext1", "R" => RD, "seed" => SEED, "hpc" => true,
                        "local_dir" => nothing, "tol_inner" => 1e-10, "max_inner" => 5000)

nu_draws, draws_3d, key_index = load_precomputed_draws(drw, RD, SEED)
path = demand_parquet_path(K, SPEC, args)
df   = DataFrame(Parquet2.Dataset(path); copycols=true)
println("  parquet: $(basename(path)), $(nrow(df)) rows")

sigma_indices, pi_interactions, n_params = build_theta2_structure("ext1")
coef_dim = 1 + L_PROD
N_obs    = nrow(df)
obs_key_idx = [get(key_index, (string(df.mca_code[i]), string(df.time_id[i])), 1) for i in 1:N_obs]
dep_types = Int.(coalesce.(df.deposit_type, 0))
spread_cols, x_mat, Z_mat, iv_cols = build_regressor_matrices(df)
prod_vec = rc_design(spread_cols, x_mat)
Z, iv_avail = moment_instruments(Z_mat, iv_cols)
spread_hat = project_endogenous_spreads(spread_cols, hcat(x_mat, Z_mat), dep_types)
X_full = hcat(spread_cols, x_mat)
X_hat  = hcat(spread_hat,  x_mat)
valid  = BitVector(all.(isfinite, eachrow(X_hat)))
pc     = build_precomp(df, Z, X_full, X_hat, valid, string.(df.CodConglomeradoPrudencial))
check_design_rank(pc)
N_B = sum(pc.b_mask); N_D = sum(pc.d_mask)
buf  = allocate_hot_buffers(N_obs, N_B, N_D, RD, length(pc.unique_pairs), length(pc.unique_times),
                            coef_dim, length(pi_interactions))
precompute_pi_products!(buf, prod_vec, draws_3d, obs_key_idx, pi_interactions, coef_dim)
gbuf = allocate_gpu_buffers(buf, pc, N_obs, N_B, N_D, RD, length(pc.unique_pairs), length(pc.unique_times))
GRP  = build_exact_groups(pc.b_mask, pc.b_mkt_idx, pc.d_time_enc, pc.pair_time_enc, pc.pop_weights)

# ── The fit ────────────────────────────────────────────────────────────────────────────────────
chk = get(ENV, "L1_CHECKPOINT", joinpath(blp_dir(out_dir), "blp_checkpoint_E$(K)_spec_$(SPEC)_ext1.jls"))
ck  = deserialize(chk)
t2  = collect(Float64, ck["theta2_star"])
length(t2) == n_params || error("$chk: theta2 of length $(length(t2)), ext1 has $n_params")
d0  = load_delta_bin(replace(chk, r"\.jls$" => ".bin"))
(d0 === nothing || length(d0) != N_obs) && (d0 = collect(Float64, ck["delta_star"]))
sv, pv = unpack_theta2(t2, sigma_indices, pi_interactions)
compute_mu!(buf, prod_vec, nu_draws, sv, sigma_indices, pv, RD, coef_dim)
dstar = copy(d0)
conv, it, _ = blp_contraction_gpu!(buf, gbuf, dstar, pc, RD; tol = TOL, max_iter = CAP)
theta1, _ = estimate_theta1(dstar, pc)
alpha = theta1[1]
println("  checkpoint $(basename(chk)): theta2 $(round.(t2, sigdigits = 6)); contraction conv $conv in $it evals; alpha $(round(alpha, sigdigits = 6))")

# α_ir = α + σ_ρ ν_rρ + Σ_d π_ρd D̃_rd: the spread is column 1 of the random-coefficient design
s_rho  = (pos = findfirst(==(1), sigma_indices); pos === nothing ? 0.0 : sv[pos])
pi_rho = [(d, pv[j]) for (j, (c, d)) in enumerate(pi_interactions) if c == 1]
nu_rho = nu_draws[:, 1]
function alpha_rows(rows)
    A = fill(alpha, length(rows), RD) .+ s_rho .* nu_rho'
    for (d, p) in pi_rho
        A .+= p .* draws_3d[obs_key_idx[rows], :, d]
    end
    return A
end
demo_rows(rows, d) = draws_3d[obs_key_idx[rows], :, d]
n_demo = size(draws_3d, 3)
demo_names = [d <= length(D_COLS) ? D_COLS[d] : "demo$(d)" for d in 1:n_demo]

rho    = spread_cols
s_data = [pc.b_mask[i] ? exp(pc.ln_s_data_B_cond[i]) : exp(pc.ln_s_data_D[i]) for i in 1:N_obs]
dep    = Float64.(coalesce.(df.Dep_Act, NaN))
Mrow   = Float64.(coalesce.(df.M_mt, NaN))
M_pair = [[Mrow[GRP.b_rows[t][rng[1]]] for rng in GRP.b_ranges[t]] for t in eachindex(GRP.b_rows)]

t0 = time()
res = l1_items(dstar, buf.mu, alpha_rows, demo_rows, n_demo, rho, s_data, dep, M_pair, GRP)
println("  L1 items in $(round(time() - t0, digits = 1)) s")
eta_sum = eta_summary(res["eta"], dep, pc.b_mask)
m = isfinite.(rho) .& isfinite.(s_data) .& (s_data .>= 0) .& (s_data .< 1)
plugin = alpha * sum(rho[m] .* (1 .- s_data[m])) / count(m)
md = m .& isfinite.(dep) .& (dep .> 0)
plugin_dep = alpha * sum(dep[md] .* rho[md] .* (1 .- s_data[md])) / sum(dep[md])
# on impact: × (1 − mean φ̂), the factor of demand.impact.blp.*; NaN when L1_MEAN_PHI is unset
awake = 1.0 - MEAN_PHI
impact = Dict{String,Any}("mean_phi" => MEAN_PHI, "awake_factor" => awake,
    "plugin_row" => plugin * awake, "plugin_deposit" => plugin_dep * awake,
    "rc" => Dict{String,Any}(g => Dict{String,Any}("row_mean" => eta_sum[g]["row_mean"] * awake,
                                                   "deposit_mean" => eta_sum[g]["deposit_mean"] * awake)
                             for g in ("all", "B", "D")))

rec = Dict{String,Any}("routine" => K, "label" => LABEL, "checkpoint" => chk, "theta2" => t2,
    "alpha" => alpha, "theta1" => theta1, "contraction_conv" => conv, "contraction_evals" => it,
    "share_check_max_abs_log_diff" => res["share_check_max_abs_log_diff"],
    "eta" => eta_sum, "eta_plugin_alpha_mean_rho_1ms" => plugin,
    "eta_plugin_alpha_depmean_rho_1ms" => plugin_dep, "on_impact" => impact,
    "alpha_choosers_row" => res["alpha_choosers_row"], "alpha_choosers_deposit" => res["alpha_choosers_deposit"],
    "alpha_population" => res["alpha_population"],
    "demo_names" => demo_names, "demo_mean_choosers_row" => res["demo_mean_choosers_row"],
    "demo_mean_choosers_deposit" => res["demo_mean_choosers_deposit"],
    "demo_mean_population" => res["demo_mean_population"])
open(joinpath(OUT, "l1_items_E$(K)_$(LABEL).json"), "w") do f
    JSON3.write(f, finite_json(rec))
end

# ── Summary (what to paste back) ───────────────────────────────────────────────────────────────
r4(x) = round(x, sigdigits = 4)
println("\n=== L1 ITEMS SUMMARY E$K $LABEL ===")
println("theta2 $(r4.(t2)); alpha $(r4(alpha)); contraction conv $conv; share check $(r4(rec["share_check_max_abs_log_diff"]))")
for g in ("all", "B", "D")
    e = eta_sum[g]
    println("eta $g: row mean $(r4(e["row_mean"])), row median $(r4(e["row_median"])), deposit-weighted $(r4(e["deposit_mean"])) (rows $(e["rows"]), dropped $(e["rows_dropped"]))")
end
println("eta plug-in alpha * mean rho(1-s): rows $(r4(plugin)), deposit-weighted $(r4(plugin_dep))")
if isfinite(MEAN_PHI)
    println("on impact (x (1 - mean phi) = $(r4(awake)), mean phi $(r4(MEAN_PHI))): RC all row $(r4(impact["rc"]["all"]["row_mean"])), " *
            "deposit $(r4(impact["rc"]["all"]["deposit_mean"])); plug-in rows $(r4(impact["plugin_row"])), deposit $(r4(impact["plugin_deposit"]))")
else
    println("on impact: L1_MEAN_PHI not set")
end
for k in ("alpha_choosers_row", "alpha_choosers_deposit", "alpha_population")
    a = rec[k]
    println("$k: mean $(r4(a["mean"])), share >= 0 $(r4(a["share_nonneg"])), p5 $(r4(a["p5"])), p50 $(r4(a["p50"])), p95 $(r4(a["p95"])), clipped weight $(r4(a["clipped_weight"] / a["total_weight"]))")
end
for (j, nm) in enumerate(demo_names)
    println("D~ $nm: choosers row $(r4(rec["demo_mean_choosers_row"][j])), choosers deposit $(r4(rec["demo_mean_choosers_deposit"][j])), population $(r4(rec["demo_mean_population"][j]))")
end
println("=== END L1 ITEMS E$K ===")
