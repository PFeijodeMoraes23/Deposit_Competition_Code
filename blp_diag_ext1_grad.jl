# blp_diag_ext1_grad.jl — read-only diagnostic of the ext1 rung of one routine (E3 or E4).
#
# Two questions, on the production data (data/output/demand_prep, the R = 2000 draws, the
# production rc4 checkpoint) and with the engine's own functions:
#
#  1. Why does ext1's warm start from rc4 never leave its starting point (the added
#     pi(ln assets x gdp) stays at exactly 0 and Q at rc4's)? At that point (A) and at the adopted
#     ext1 fit (B) it compares the engine's diagonal-IFT gradient (gmm_fg_gpu!) with a central
#     finite-difference gradient of the true Q, then evaluates Q along L-BFGS-B's first search
#     direction (minus the projected IFT gradient, unit length) and along the finite-difference
#     one. At A it also scans the added pi alone. If Q rises along the IFT direction at every step
#     while the finite-difference direction lowers it, the line search fails on an inaccurate
#     gradient and the start ends where it began.
#  2. Which observations hold the inner contraction's stalled norm (0.05129 = |ln 0.95| in every
#     TR5 log)? It re-runs the first contraction of the rand1 start (from the logit delta) and of each
#     directional trial point, stops after STALL_PROBE evaluations if not converged, and lists the
#     observations with the largest |ln s_data - ln s_model|.
#
# Writes only DIAG_OUT/diag_ext1_grad_E<k>.json and its log; no checkpoint, result or delta file.
#
# Environment: DIAG_ROUTINE (3 | 4), DIAG_ADOPTED (the adopted ext1 checkpoint .jls, with its .bin),
# DIAG_OUT (output folder). Run through blp_diag_ext1_grad_job.sh.

isdefined(Main, :main_gpu_ift) || include(joinpath(@__DIR__, "blp_rc.jl"))
using Serialization, JSON3, LinearAlgebra, Statistics, CUDA

const K        = parse(Int, ENV["DIAG_ROUTINE"])
const OUT      = ENV["DIAG_OUT"]
const ADOPTED  = ENV["DIAG_ADOPTED"]
const SPEC     = 12
const RD       = 2000
const SEED     = 42
const TOL      = 1e-12        # every Q below: inner contraction to 1e-12 (production 1e-10, re-evaluation 1e-11)
const CAP      = 20000
const STALL_PROBE = 300       # evaluations before a not-yet-converged contraction is probed
const H_FD     = 1e-4
const T_DIR    = [1.0, 0.3, 0.1, 0.03, 0.01, 1e-3, 1e-4]
const PI5_GRID = [-0.2, -0.1, -0.05, -0.01, 0.01, 0.05, 0.1]
# The first extra start of config/blp_ext1_extra_starts.csv (rand1), whose first contraction from
# the logit delta stalled at norm 0.05129 in every TR5 run. Written out so this script does not
# depend on that file being on this tree.
const RAND1    = [0.0437, -0.0824, 0.1943, 2.4934, 0.5764]

mkpath(OUT)
println("=" ^ 72)
println("blp_diag_ext1_grad: E$K, spec $SPEC, R $RD, inner tol $TOL, adopted = $ADOPTED")
println("=" ^ 72)

# ── The engine's setup for ext1 (run_blp_estimation_ift_gpu, up to the GPU buffers) ───────────────
# The draws slot is `drw`, not `draws_dir`: of_root.jl defines a function of that name in Main.
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
ln_s_data = [pc.b_mask[i] ? pc.ln_s_data_B_cond[i] : pc.ln_s_data_D[i] for i in 1:N_obs]

# ── Helpers ────────────────────────────────────────────────────────────────────────────────────
function set_mu!(t)
    sv, pv = unpack_theta2(t, sigma_indices, pi_interactions)
    compute_mu!(buf, prod_vec, nu_draws, sv, sigma_indices, pv, RD, coef_dim)
end

"""A copy of a nested result with every non-finite float as nothing (JSON has no NaN)."""
finite_json(x::AbstractFloat) = isfinite(x) ? x : nothing
finite_json(x::AbstractVector) = Any[finite_json(v) for v in x]
finite_json(x::AbstractDict) = Dict{String,Any}(string(k) => finite_json(v) for (k, v) in x)
finite_json(x) = x

"""The observations with the largest |ln s_data - ln s_model| at δ (μ already set for its θ₂)."""
function residual_top(delta; n = 10)
    compute_model_shares_gpu!(buf, gbuf, delta, pc, RD)
    s = collect_model_shares(buf, pc, N_obs)
    r = ln_s_data .- log.(s)
    idx = partialsortperm(abs.(r), 1:n; rev = true)
    rows = Any[]
    for i in idx
        mk = (df.mca_code[i], df.time_id[i])
        nB = count(j -> pc.b_mask[j] && df.mca_code[j] == mk[1] && df.time_id[j] == mk[2], 1:N_obs)
        push!(rows, Dict("row" => i, "bank" => string(df.CodConglomeradoPrudencial[i]),
            "mca" => string(df.mca_code[i]), "time_id" => string(df.time_id[i]),
            "deposit_type" => dep_types[i], "is_B" => pc.b_mask[i], "n_B_in_market" => nB,
            "s_data" => exp(ln_s_data[i]), "s_model" => s[i], "delta" => delta[i], "resid" => r[i]))
    end
    n_at = count(x -> abs(abs(x) - abs(log(0.95))) < 1e-4, r)
    n_cap = count(i -> abs(exp(ln_s_data[i]) - 0.95) < 1e-9, 1:N_obs)
    return rows, n_at, n_cap
end

"""Q at θ₂ from a copy of δ0, contracted to TOL. A contraction still running after STALL_PROBE
evaluations is probed (residual_top) and then continued to TOL."""
function Q_at(t, delta0; probe = false)
    set_mu!(t)
    d = copy(delta0)
    probe_rec = nothing
    conv, it, _ = blp_contraction_gpu!(buf, gbuf, d, pc, RD; tol = TOL, max_iter = probe ? STALL_PROBE : CAP)
    if probe && !conv
        rows, n_at, n_cap = residual_top(d)
        probe_rec = Dict("after_evals" => it, "top" => rows, "n_resid_at_ln095" => n_at,
                         "n_obs_at_share_cap" => n_cap)
        conv, it2, _ = blp_contraction_gpu!(buf, gbuf, d, pc, RD; tol = TOL, max_iter = CAP)
        it += it2
    end
    theta1, xi = estimate_theta1(d, pc)
    g = compute_gmm_moments(xi, pc)
    return dot(g, W * g), conv, it, theta1[1], d, probe_rec
end

"""The engine's own objective and diagonal-IFT gradient (gmm_fg_gpu!, production tolerances)."""
function ift_grad(t, delta0)
    G = zeros(n_params); d = copy(delta0)
    Q = gmm_fg_gpu!(G, t, buf, gbuf, prod_vec, nu_draws, sigma_indices, pi_interactions, RD, coef_dim,
                    W, 1e-10, 5000, d, pc)
    return Q, G
end

"""Central finite-difference gradient of the true Q (one-sided forward where the box binds)."""
function fd_grad(t, dstar; h = H_FD)
    G = zeros(n_params); side = String[]
    for k in 1:n_params
        tp = copy(t); tp[k] += h
        tm = copy(t); tm[k] -= h
        Qp = Q_at(clamp.(tp, lo, hi), dstar)[1]
        if tm[k] < lo[k]
            Q0 = Q_at(t, dstar)[1]
            G[k] = (Qp - Q0) / h; push!(side, "forward")
        else
            Qm = Q_at(tm, dstar)[1]
            G[k] = (Qp - Qm) / (2h); push!(side, "central")
        end
        println("    fd k=$k: $(round(G[k], sigdigits = 6)) ($(side[end]))")
    end
    return G, side
end

"""L-BFGS-B's first direction from θ₂: minus the gradient, zeroed where it pushes into a binding bound."""
function proj_dir(t, G)
    d = -copy(G)
    for k in 1:n_params
        (t[k] <= lo[k] && d[k] < 0) && (d[k] = 0.0)
        (t[k] >= hi[k] && d[k] > 0) && (d[k] = 0.0)
    end
    nd = norm(d)
    return nd > 0 ? d ./ nd : d
end

function dir_scan(t, dstar, Q0, d, G; probe = true)
    out = Any[]
    for s in T_DIR
        tt = clamp.(t .+ s .* d, lo, hi)
        Q, conv, it, a, _, pr = Q_at(tt, dstar; probe = probe)
        push!(out, Dict("t" => s, "theta2" => tt, "dQ" => Q - Q0, "linear_pred" => s * dot(G, d),
                        "conv" => conv, "evals" => it, "alpha" => a, "probe" => pr))
        println("    t=$(s): dQ=$(round(Q - Q0, sigdigits = 5)) (linear $(round(s * dot(G, d), sigdigits = 5)))" *
                " evals=$it conv=$conv" * (pr === nothing ? "" : " [stall probe at $(pr["after_evals"])]"))
    end
    return out
end

function load_point(chk)
    ck = deserialize(chk)
    t2 = collect(Float64, ck["theta2_star"])
    t  = zeros(n_params); t[1:length(t2)] .= t2
    d  = load_delta_bin(replace(chk, r"\.jls$" => ".bin"))
    (d === nothing || length(d) != N_obs) && (d = collect(Float64, ck["delta_star"]))
    length(d) == N_obs || error("$chk: delta of length $(length(d)), data $N_obs")
    return clamp.(t, lo, hi), d
end

function point_report(name, t, d0; pi5_scan = false)
    println("\n── point $name: theta2 = $(round.(t, digits = 5)) ──")
    Q0, conv0, it0, a0, dstar, _ = Q_at(t, d0)
    println("  Q = $(round(Q0, sigdigits = 10)) (conv $conv0, $it0 evals), alpha = $(round(a0, sigdigits = 6))")
    Qi, Gi = ift_grad(t, dstar)
    println("  IFT gradient: $(round.(Gi, sigdigits = 5))  (its Q $(round(Qi, sigdigits = 10)))")
    Gf, side = fd_grad(t, dstar)
    di, df_ = proj_dir(t, Gi), proj_dir(t, Gf)
    cosang = (norm(Gi) > 0 && norm(Gf) > 0) ? dot(Gi, Gf) / (norm(Gi) * norm(Gf)) : NaN
    println("  FD gradient:  $(round.(Gf, sigdigits = 5))   cos(IFT, FD) = $(round(cosang, digits = 4))")
    println("  along the IFT direction (L-BFGS-B's first step is t = 1):")
    scan_i = dir_scan(t, dstar, Q0, di, Gf)
    println("  along the FD direction:")
    scan_f = dir_scan(t, dstar, Q0, df_, Gf; probe = false)
    rec = Dict{String,Any}("theta2" => t, "Q" => Q0, "conv" => conv0, "alpha" => a0,
        "grad_ift" => Gi, "Q_ift" => Qi, "grad_fd" => Gf, "fd_side" => side, "h" => H_FD,
        "cos_ift_fd" => cosang, "dir_ift" => di, "dir_fd" => df_, "scan_ift" => scan_i, "scan_fd" => scan_f)
    if pi5_scan
        println("  the added pi alone:")
        sc = Any[]
        for v in PI5_GRID
            tt = copy(t); tt[end] = v
            Q, conv, it, a = Q_at(tt, dstar)
            push!(sc, Dict("pi5" => v, "dQ" => Q - Q0, "conv" => conv, "alpha" => a))
            println("    pi5=$(v): dQ=$(round(Q - Q0, sigdigits = 5)) alpha=$(round(a, sigdigits = 5))")
        end
        rec["pi5_scan"] = sc
    end
    return rec
end

# ── Run ────────────────────────────────────────────────────────────────────────────────────────
res = Dict{String,Any}("routine" => K, "parquet" => basename(path), "n_obs" => N_obs, "tol" => TOL,
                       "adopted_checkpoint" => ADOPTED)
rc4_chk = joinpath(blp_dir(out_dir), "blp_checkpoint_E$(K)_spec_$(SPEC)_rc4.jls")
tA, dA = load_point(rc4_chk)
res["A_warm_rc4"] = point_report("A (warm_rc4: rc4's theta2, added pi = 0, rc4's delta)", tA, dA; pi5_scan = true)
tB, dB = load_point(ADOPTED)
res["B_adopted"]  = point_report("B (adopted ext1)", tB, dB)

# Stall probe: the rand1 start's first contraction, from the stage's delta (the logit delta).
dlog = nothing
for c in (joinpath(in_dir, "logit_delta_E$(K)_spec_$(SPEC).bin"), joinpath(logit_dir(out_dir), "logit_delta_E$(K)_spec_$(SPEC).bin"))
    global dlog
    dlog === nothing && isfile(c) && (dlog = load_delta_bin(c))
end
if dlog !== nothing && length(dlog) == N_obs
    println("\n── stall probe: rand1 start $(RAND1), logit delta ──")
    Q, conv, it, a, _, pr = Q_at(clamp.(RAND1, lo, hi), dlog; probe = true)
    res["stall_probe_rand1"] = Dict("Q" => Q, "conv" => conv, "evals" => it, "probe" => pr)
    println("  evals $it, conv $conv" * (pr === nothing ? ", no stall at $STALL_PROBE" : ""))
else
    res["stall_probe_rand1"] = "logit delta not found"
end

open(joinpath(OUT, "diag_ext1_grad_E$(K).json"), "w") do f; JSON3.write(f, finite_json(res)); end

# ── Summary (what to paste back) ───────────────────────────────────────────────────────────────
r5(x) = round(x, sigdigits = 5)
println("\n=== DIAG SUMMARY E$K ===")
for key in ("A_warm_rc4", "B_adopted")
    p = res[key]
    best_i = minimum(s["dQ"] for s in p["scan_ift"]); best_f = minimum(s["dQ"] for s in p["scan_fd"])
    println("$key: Q $(round(p["Q"], sigdigits = 8)), alpha $(r5(p["alpha"]))")
    println("  grad IFT $(r5.(p["grad_ift"]))")
    println("  grad FD  $(r5.(p["grad_fd"]))  cos $(round(p["cos_ift_fd"], digits = 3))  |FD| $(r5(norm(p["grad_fd"])))")
    println("  dQ along IFT dir, t=$(T_DIR): $(r5.([s["dQ"] for s in p["scan_ift"]]))  best $(r5(best_i))")
    println("  dQ along FD dir,  t=$(T_DIR): $(r5.([s["dQ"] for s in p["scan_fd"]]))  best $(r5(best_f))")
    haskey(p, "pi5_scan") && println("  added pi alone $(PI5_GRID): $(r5.([s["dQ"] for s in p["pi5_scan"]]))")
    for s in p["scan_ift"]
        pr = s["probe"]; pr === nothing && continue
        tp = pr["top"][1]
        println("  stall at t=$(s["t"]) after $(pr["after_evals"]) evals: top |resid| $(r5(abs(tp["resid"]))) at bank $(tp["bank"]) " *
                "mca $(tp["mca"]) $(tp["time_id"]) type $(tp["deposit_type"]) B=$(tp["is_B"]) nB=$(tp["n_B_in_market"]) " *
                "s_data $(r5(tp["s_data"])) s_model $(r5(tp["s_model"])); obs at |ln0.95|: $(pr["n_resid_at_ln095"]); at share cap: $(pr["n_obs_at_share_cap"])")
    end
end
sp = res["stall_probe_rand1"]
if sp isa Dict && sp["probe"] !== nothing
    pr = sp["probe"]
    println("stall probe rand1: after $(pr["after_evals"]) evals; obs at |ln0.95|: $(pr["n_resid_at_ln095"]); at share cap: $(pr["n_obs_at_share_cap"])")
    for tp in pr["top"][1:min(5, length(pr["top"]))]
        println("  |resid| $(r5(abs(tp["resid"]))) bank $(tp["bank"]) mca $(tp["mca"]) $(tp["time_id"]) type $(tp["deposit_type"]) " *
                "B=$(tp["is_B"]) nB=$(tp["n_B_in_market"]) s_data $(r5(tp["s_data"])) s_model $(r5(tp["s_model"])) delta $(r5(tp["delta"]))")
    end
else
    println("stall probe rand1: $(sp isa Dict ? "converged within $STALL_PROBE evals ($(sp["evals"]))" : sp)")
end
println("=== END DIAG E$K ===")
