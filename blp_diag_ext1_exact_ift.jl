# blp_diag_ext1_exact_ift.jl — read-only check of the engine's IFT Jacobian ∂δ*/∂θ₂ against finite
# differences of δ* itself, on the production data. It was stage 1 of L1 (handoff 2026-10-31b), run
# when the engine still used the diagonal approximation; with the engine on the exact Jacobian
# (blp_exact_ift.jl) it is the cluster check of that engine.
#
# At the two points of blp_diag_ext1_grad.jl, A (rc4's θ₂ with the added π at 0) and B (the adopted
# ext1 fit), with the engine's own functions, it builds three N × 5 Jacobians of δ* in θ₂:
#   engine the engine's (compute_delta_jacobian_gpu), the one its gradient and standard errors use
#   exact  exact_delta_jacobian called here directly, with its condition-number diagnostics
#   fd     central finite differences of δ* itself, each side contracted to TOL from δ*
# and from each the gradient of Q (the exact directional derivative of the quadratic δ ↦ Q(δ)) and
# the joint standard errors (gmm_cluster_ses, the engine's method). It reports the engine's own
# gradient (gmm_fg_gpu!), each Jacobian's error against fd column by column and by the outside share
# of the B row's market, the shares check (the exact code's own shares against the engine's), the
# condition numbers met and the time the engine's Jacobian takes.
#
# Writes only DIAG_OUT/diag_ext1_exact_ift_E<k>.json and its log. Environment: DIAG_ROUTINE (3 | 4),
# DIAG_ADOPTED (the adopted ext1 checkpoint .jls), DIAG_OUT. Run through blp_diag_ext1_exact_ift_job.sh.

isdefined(Main, :main_gpu_ift) || include(joinpath(@__DIR__, "blp_rc.jl"))
isdefined(Main, :ExactGroups) || include(joinpath(@__DIR__, "blp_exact_ift.jl"))
using Serialization, JSON3, LinearAlgebra, Statistics, CUDA, Dates

const K        = parse(Int, ENV["DIAG_ROUTINE"])
const OUT      = ENV["DIAG_OUT"]
const ADOPTED  = ENV["DIAG_ADOPTED"]
const SPEC     = 12
const RD       = 2000
const SEED     = 42
const TOL      = 1e-12        # every contraction below (production 1e-10)
const CAP      = 20000
const H_FD     = 1e-4         # step of the finite differences of δ*
const EPS_S    = 1e-5         # step of the central ∂s/∂θ₂ at fixed δ
const S0_EDGES = [0.0, 0.2, 0.5, 0.8, 1.0 + 1e-12]

stamp() = Dates.format(Dates.now(), "HH:MM:SS")

"""A copy of a nested result with every non-finite float as nothing (JSON has no NaN)."""
finite_json(x::AbstractFloat) = isfinite(x) ? x : nothing
finite_json(x::AbstractVector) = Any[finite_json(v) for v in x]
finite_json(x::AbstractDict) = Dict{String,Any}(string(k) => finite_json(v) for (k, v) in x)
finite_json(x) = x
mkpath(OUT)
println("=" ^ 72)
println("blp_diag_ext1_exact_ift: E$K, spec $SPEC, R $RD, inner tol $TOL, adopted = $ADOPTED")
println("=" ^ 72)

# ── The engine's setup for ext1, as in blp_diag_ext1_grad.jl ─────────────────────────────────────
in_dir, drw, out_dir = get_paths(true)
in_dirs = demand_search_dirs(in_dir, out_dir)
prefix  = demand_prefix(K, in_dirs)
prefix === nothing && error("no demand parquet for E$K in $in_dirs")
ENV["BLP_DELTA_SUFFIX"] = ""; ENV["BLP_OUTPUT_SUFFIX"] = ""; ENV["BLP_DEMAND_PREFIX"] = prefix
ENV["BLP_SIGMA_UB"] = get(ENV, "BLP_SIGMA_UB", "5.0"); ENV["BLP_PI_BOUND"] = get(ENV, "BLP_PI_BOUND", "5.0")
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
W = try inv(Z' * Z ./ N_obs) catch; Matrix(I(length(iv_avail)) * 1.0) end
H = hcat(x_mat, Z_mat)
spread_hat = project_endogenous_spreads(spread_cols, H, dep_types)
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
println("  GPU: $(CUDA.name(CUDA.device()))")

lo = fill(-parse(Float64, ENV["BLP_PI_BOUND"]), n_params); hi = fill(parse(Float64, ENV["BLP_PI_BOUND"]), n_params)
lo[1:length(sigma_indices)] .= 0.0; hi[1:length(sigma_indices)] .= parse(Float64, ENV["BLP_SIGMA_UB"])
bidx = findall(pc.b_mask); didx = findall(pc.d_mask)
GRP  = build_exact_groups(pc.b_mask, pc.b_mkt_idx, pc.d_time_enc, pc.pair_time_enc, pc.pop_weights)
println("  quarters $(length(GRP.b_rows)); largest quarter: $(maximum(length.(GRP.b_rows))) B rows, " *
        "$(maximum(length.(GRP.d_rows))) D rows; largest market $(maximum(maximum(length.(r)) for r in GRP.b_ranges)) rows")

# ── Helpers ────────────────────────────────────────────────────────────────────────────────────
function set_mu!(t)
    sv, pv = unpack_theta2(t, sigma_indices, pi_interactions)
    compute_mu!(buf, prod_vec, nu_draws, sv, sigma_indices, pv, RD, coef_dim)
end

"""δ* at θ₂ from a copy of δ0, contracted to TOL; leaves buf.mu at θ₂."""
function contract(t, delta0)
    set_mu!(t)
    d = copy(delta0)
    conv, it, _ = blp_contraction_gpu!(buf, gbuf, d, pc, RD; tol = TOL, max_iter = CAP)
    return d, conv, it
end

"""Model shares at δ for the θ₂ whose μ is in buf.mu, in full-N order, without the 1e-15 floor."""
function shares_raw(delta)
    compute_model_shares_gpu!(buf, gbuf, delta, pc, RD)
    s = Vector{Float64}(undef, N_obs); s[bidx] .= buf.s_B; s[didx] .= buf.s_D
    return s
end

"""Q as a function of δ alone (θ₁ by the engine's 2SLS, fixed W): a quadratic in δ."""
function Qdelta(d)
    _, xi = estimate_theta1(d, pc)
    g = compute_gmm_moments(xi, pc)
    return dot(g, W * g)
end

"""∇Q implied by a Jacobian of δ*: the central difference of the quadratic δ ↦ Q(δ) along each
column, exact up to rounding for any step."""
function grad_along(J, dstar)
    G = fill(NaN, n_params)
    for k in 1:n_params
        col = J[:, k]
        all(isfinite, col) || continue
        m = maximum(abs, col); eta = m > 0 ? 1e-3 / m : 1.0
        G[k] = (Qdelta(dstar .+ eta .* col) - Qdelta(dstar .- eta .* col)) / (2eta)
    end
    return G
end

"""∂ln s/∂θ₂ at fixed δ* (central, EPS_S); restores buf.mu to θ₂."""
function dlnsdtheta(t, dstar)
    D = Matrix{Float64}(undef, N_obs, n_params)
    for k in 1:n_params
        tp = copy(t); tp[k] += EPS_S; tm = copy(t); tm[k] -= EPS_S
        set_mu!(tp); sp = shares_raw(dstar)
        set_mu!(tm); sm = shares_raw(dstar)
        D[:, k] = (log.(max.(sp, 1e-15)) .- log.(max.(sm, 1e-15))) ./ (2EPS_S)
    end
    set_mu!(t)
    return D
end

rel(a, b) = (nb = norm(b); nb > 0 ? norm(a .- b) / nb : NaN)
cosv(a, b) = (norm(a) > 0 && norm(b) > 0) ? dot(a, b) / (norm(a) * norm(b)) : NaN

function load_point(chk)
    ck = deserialize(chk)
    t2 = collect(Float64, ck["theta2_star"])
    t  = zeros(n_params); t[1:length(t2)] .= t2
    d  = load_delta_bin(replace(chk, r"\.jls$" => ".bin"))
    (d === nothing || length(d) != N_obs) && (d = collect(Float64, ck["delta_star"]))
    length(d) == N_obs || error("$chk: delta of length $(length(d)), data $N_obs")
    return clamp.(t, lo, hi), d
end

function ses(J, t, dstar)
    try
        th1, _ = estimate_theta1(dstar, pc)
        v = pc.theta1_valid .& isfinite.(dstar)
        se, _ = gmm_cluster_ses(se_method(), vcat(th1, t), pc.Z_moments[v, :], pc.X_full[v, :], J[v, :],
                                dstar[v] .- pc.X_full[v, :] * th1, W, pc.clusters[v];
                                n_sigma = length(sigma_indices))
        K1 = length(th1)
        return Dict("status" => SE_LAST_STATUS[], "alpha_se" => se[1], "theta2_se" => se[K1+1:end])
    catch e
        return Dict("status" => "failed: " * sprint(showerror, e))
    end
end

function point(name, t, d0)
    println("\n── point $name: theta2 = $(round.(t, digits = 5)) [$(stamp())] ──")
    dstar, conv, it = contract(t, d0)
    Q0 = Qdelta(dstar); th1, _ = estimate_theta1(dstar, pc)
    println("  Q = $(round(Q0, sigdigits = 10)) (conv $conv, $it evals), alpha = $(round(th1[1], sigdigits = 6))")
    Ge = zeros(n_params)
    gmm_fg_gpu!(Ge, t, buf, gbuf, prod_vec, nu_draws, sigma_indices, pi_interactions, RD, coef_dim,
                W, 1e-10, 5000, copy(dstar), pc)
    set_mu!(t)
    s_eng = shares_raw(dstar)
    println("  engine gradient: $(round.(Ge, sigdigits = 5)) [$(stamp())]")

    dlns = dlnsdtheta(t, dstar)
    t0 = time()
    Jx, info = exact_delta_jacobian(dstar, buf.mu, dlns, GRP; diagnostics = true)
    share_err = maximum(abs.(info["s_model"] .- s_eng) ./ max.(s_eng, 1e-300))
    println("  exact Jacobian (with diagnostics) in $(round(time() - t0, digits = 1)) s; shares check max rel diff " *
            "$(round(share_err, sigdigits = 3)); max cond J_BB $(round(info["max_cond_JBB"], sigdigits = 3)); " *
            "max cond Schur $(round(info["max_cond_schur"], sigdigits = 3)); singular blocks $(info["n_singular_blocks"])")

    t0 = time()
    Jd = compute_delta_jacobian_gpu(dstar, t, buf, gbuf, prod_vec, nu_draws,
                                    sigma_indices, pi_interactions, RD, coef_dim, pc)
    t_engine = time() - t0
    println("  engine Jacobian in $(round(t_engine, digits = 1)) s ($(Threads.nthreads()) threads)")

    Jf = Matrix{Float64}(undef, N_obs, n_params); fd_runs = Any[]
    for k in 1:n_params
        tp = copy(t); tp[k] += H_FD; tm = copy(t); tm[k] -= H_FD
        t1 = time(); dp, cp, ip = contract(tp, dstar)
        t2 = time(); dm, cm, im = contract(tm, dstar)
        t3 = time()
        Jf[:, k] = (dp .- dm) ./ (2H_FD)
        push!(fd_runs, Dict("k" => k, "conv" => [cp, cm], "evals" => [ip, im], "seconds" => [t2 - t1, t3 - t2]))
        println("    fd k=$k: conv $cp/$cm, evals $ip/$im, $(round(t2 - t1, digits = 1))/$(round(t3 - t2, digits = 1)) s")
    end
    set_mu!(t)

    Gd, Gx, Gf = grad_along(Jd, dstar), grad_along(Jx, dstar), grad_along(Jf, dstar)
    s0 = info["s0_row"]
    bins = Any[]
    for b in 1:length(S0_EDGES) - 1
        rows = [i for i in bidx if S0_EDGES[b] <= s0[i] < S0_EDGES[b+1]]
        push!(bins, Dict("bin" => "[$(S0_EDGES[b]), $(min(S0_EDGES[b+1], 1.0)))", "rows" => length(rows),
                         "engine" => isempty(rows) ? NaN : rel(Jd[rows, :], Jf[rows, :]),
                         "exact" => isempty(rows) ? NaN : rel(Jx[rows, :], Jf[rows, :])))
    end
    push!(bins, Dict("bin" => "D rows", "rows" => length(didx),
                     "engine" => rel(Jd[didx, :], Jf[didx, :]), "exact" => rel(Jx[didx, :], Jf[didx, :])))
    rec = Dict{String,Any}("theta2" => t, "Q" => Q0, "conv" => conv, "evals" => it, "alpha" => th1[1],
        "grad_engine" => Ge, "grad_engine_jac" => Gd, "grad_exact" => Gx, "grad_fd_delta" => Gf,
        "cos_engine_fd" => cosv(Ge, Gf), "rel_engine_fd" => rel(Ge, Gf),
        "cos_exact_fd" => cosv(Gx, Gf), "rel_exact_fd" => rel(Gx, Gf),
        "col_rel_engine" => [rel(Jd[:, k], Jf[:, k]) for k in 1:n_params],
        "col_rel_exact" => [rel(Jx[:, k], Jf[:, k]) for k in 1:n_params],
        "by_outside_share" => bins, "share_check_max_rel" => share_err,
        "max_cond_JBB" => info["max_cond_JBB"], "max_cond_schur" => info["max_cond_schur"],
        "n_singular_blocks" => info["n_singular_blocks"], "seconds_engine_jacobian" => t_engine,
        "fd_runs" => fd_runs,
        "se_engine" => ses(Jd, t, dstar), "se_exact" => ses(Jx, t, dstar), "se_fd" => ses(Jf, t, dstar))
    println("  gradients: engine $(round.(Ge, sigdigits = 5)), exact $(round.(Gx, sigdigits = 5)), " *
            "fd $(round.(Gf, sigdigits = 5)) [$(stamp())]")
    return rec
end

# ── Run ────────────────────────────────────────────────────────────────────────────────────────
res = Dict{String,Any}("routine" => K, "parquet" => basename(path), "n_obs" => N_obs, "tol" => TOL,
                       "h_fd" => H_FD, "eps_s" => EPS_S, "adopted_checkpoint" => ADOPTED)
rc4_chk = joinpath(blp_dir(out_dir), "blp_checkpoint_E$(K)_spec_$(SPEC)_rc4.jls")
tA, dA = load_point(rc4_chk)
res["A_warm_rc4"] = point("A (rc4's theta2, added pi = 0)", tA, dA)
tB, dB = load_point(ADOPTED)
res["B_adopted"] = point("B (adopted ext1)", tB, dB)

# The finite-difference gradient of Q itself, from blp_diag_ext1_grad.jl, when its output is here.
prev = joinpath(OUT, "diag_ext1_grad_E$(K).json")
if isfile(prev)
    p = JSON3.read(read(prev, String))
    for key in ("A_warm_rc4", "B_adopted")
        haskey(p, Symbol(key)) &&
            (res[key]["grad_fd_Q_diag_job"] = [x === nothing ? NaN : Float64(x) for x in p[Symbol(key)][:grad_fd]])
    end
end

open(joinpath(OUT, "diag_ext1_exact_ift_E$(K).json"), "w") do f
    JSON3.write(f, finite_json(res))
end

# ── Summary (what to paste back) ───────────────────────────────────────────────────────────────
r4(x) = round(x, sigdigits = 4)
r4v(v) = [r4(x) for x in v]
println("\n=== EXACT IFT SUMMARY E$K ===")
for key in ("A_warm_rc4", "B_adopted")
    p = res[key]
    println("$key: Q $(round(p["Q"], sigdigits = 8)), alpha $(r4(p["alpha"])), conv $(p["conv"]); shares check $(r4(p["share_check_max_rel"])); " *
            "cond J_BB max $(r4(p["max_cond_JBB"])), Schur max $(r4(p["max_cond_schur"])), singular blocks $(p["n_singular_blocks"]); " *
            "engine Jacobian $(round(p["seconds_engine_jacobian"], digits = 1)) s")
    println("  grad engine $(r4v(p["grad_engine"]))  cos vs fd $(r4(p["cos_engine_fd"]))  rel $(r4(p["rel_engine_fd"]))")
    println("  grad exact  $(r4v(p["grad_exact"]))  cos vs fd $(r4(p["cos_exact_fd"]))  rel $(r4(p["rel_exact_fd"]))")
    println("  grad fd(δ)  $(r4v(p["grad_fd_delta"]))  fd contractions converged $(sum(sum(r["conv"]) for r in p["fd_runs"]))/$(2n_params), " *
            "evals $(sum(sum(r["evals"]) for r in p["fd_runs"])), seconds $(round(sum(sum(r["seconds"]) for r in p["fd_runs"]), digits = 0))")
    haskey(p, "grad_fd_Q_diag_job") && println("  grad fd(Q)  $(r4v(p["grad_fd_Q_diag_job"]))  (blp_diag_ext1_grad.jl)")
    println("  column rel err vs fd(δ): engine $(r4v(p["col_rel_engine"]))  exact $(r4v(p["col_rel_exact"]))")
    println("  by B-market outside share: " * join(["$(b["bin"]) n=$(b["rows"]) engine $(r4(b["engine"])) exact $(r4(b["exact"]))"
                                                    for b in p["by_outside_share"]], "; "))
    for j in ("se_engine", "se_exact", "se_fd")
        s = p[j]
        println("  $j: " * (haskey(s, "alpha_se") ? "alpha $(r4(s["alpha_se"])), theta2 $(r4v(s["theta2_se"])) ($(s["status"]))" : s["status"]))
    end
end
println("=== END EXACT IFT E$K ===")
