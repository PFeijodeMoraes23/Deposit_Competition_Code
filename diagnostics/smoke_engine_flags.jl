# Local CPU smoke of engine edit (b) (2026-11-11) on real data: two whole quarters of the E3 demand
# parquet, synthetic draws (R = 50), the ext1 structure, the CPU twins of the engine's functions
# (blp_contraction!, gmm_fg!, compute_ift_gradient!), as diagnostics/smoke_exact_engine.jl.
#   julia --project=<repo> --threads=1 smoke_engine_flags.jl <repo> <demand parquet>
# One thread, so the BLAS/threaded sums run in one fixed order and two runs can be compared bit for bit.
#  A. b1: a short L-BFGS-B solve of the real objective, today's solver (HEAD copy) twice (control),
#     then BLP_OUTER_STOP_REASON 0 and 1: the same points evaluated, minimizer, minimum, count, bits.
#  B. b3: guarded_fg! against gmm_fg! when every contraction converges (Q, ∇Q, δ bit for bit), then a
#     forced failure (an inner cap of 2): Q_fail = Q_ref + 2·abs(g_refᵀΔθ) + 1e-8, zero gradient, δ restored.
#  C. b4: the contraction with the clamp test off and on, with nothing at the clamp (bit for bit), and
#     with one observation's μ forced to −2000 (its δ runs to +500 and stays): today "converged", b4
#     "clamp_bound", with the counts.
#  D. b2 lite and b2 full + b3 from a far start on the real objective: they run and record their rounds.
const REPO = ARGS[1]
include(joinpath(REPO, "blp_engine_gpu.jl"))
using Random, Statistics, LinearAlgebra, DataFrames, Parquet2, Printf, Dates
say(msg) = (println("[", Dates.format(Dates.now(), "HH:MM:SS"), "] ", msg); flush(stdout))
const FAILS = String[]
check(name, ok, detail = "") = (println((ok ? "PASS " : "FAIL ") * name * (isempty(detail) ? "" : ": " * detail));
                               ok || push!(FAILS, name); ok)
bits(v) = reinterpret(UInt64, collect(Float64, v))
same_bits(a, b) = length(a) == length(b) && all(bits(a) .== bits(b))
same_trace(a, b) = length(a) == length(b) && all(same_bits(x, y) for (x, y) in zip(a, b))

df_all = DataFrame(Parquet2.Dataset(ARGS[2]); copycols=true)
qs = sort(unique(string.(df_all.time_id)))
keep = Set(qs[[10, 30]])
df = df_all[in.(string.(df_all.time_id), Ref(keep)), :]
N = nrow(df)
say("subsample: quarters $(sort(collect(keep))), $N rows; Julia threads $(Threads.nthreads()), BLAS threads $(BLAS.get_num_threads())")

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
lo = fill(-5.0, n_params); hi = fill(5.0, n_params); lo[1:length(sigma_indices)] .= 0.0
δ_logshare = zeros(N); δ_logshare[pc.b_mask] .= pc.ln_s_data_B_cond[pc.b_mask]; δ_logshare[pc.d_mask] .= pc.ln_s_data_D[pc.d_mask]
TOL = 1e-10; CAP = 5000
setmu!(t) = mu_at!(buf, prod_vec, nu_draws, t, sigma_indices, pi_interactions, R, coef_dim)

# The engine's CPU objective as run_blp_estimation_ift_gpu wires it: f and g! over one δ (delta_work)
# with a one-point cache; `guard` switches it to guarded_fg! (b3), as ift_compute_gpu! does.
function engine_objective(delta_start; guard::Bool = false, cap::Int = CAP)
    delta_work = copy(delta_start)
    cx = fill(NaN, n_params); cQ = Ref(0.0); cg = zeros(n_params); tr = Vector{Vector{Float64}}()
    gd = guard ? guard_reset!(InnerGuard(n_params, N), delta_work) : nothing
    function compute!(t2)
        isequal(t2, cx) && return
        push!(tr, copy(t2))
        cQ[] = if gd === nothing
            gmm_fg!(cg, t2, buf, prod_vec, nu_draws, sigma_indices, pi_interactions, R, coef_dim, W,
                    TOL, cap, delta_work, pc)
        else
            guarded_fg!(cg, t2, gd, delta_work,
                (d, c) -> (setmu!(t2); cr = blp_contraction!(buf, d, pc, R; tol = TOL, max_iter = c); (cr[1], cr[2])),
                d -> (gm = compute_gmm_moments(estimate_theta1(d, pc)[2], pc); (dot(gm, W * gm), gm)),
                (G, gm, d) -> (compute_ift_gradient!(G, gm, d, t2, buf, prod_vec, nu_draws, sigma_indices,
                                                     pi_interactions, R, coef_dim, W, pc); LAST_JACOBIAN_SINGULAR[]);
                cap = cap, cap_first = 10 * cap)
        end
        copyto!(cx, t2)
        flush(stdout)
    end
    return (t -> (compute!(t); cQ[])), ((G, t) -> (compute!(t); copyto!(G, cg); G)), tr, delta_work, gd
end

# Today's solver: the L-BFGS-B branch of blp_engine_cpu.jl at HEAD 24f5d096, l.77–114 (attempt 1;
# the verbatim function is in diagnostics/test_solve_box.jl).
function solve_box_lbfgsb_today(f, g!, x0::Vector{Float64}, lo::Vector{Float64}, hi::Vector{Float64};
                                maxiter::Int=500)
    fcount = Ref(0)
    f_c(x)     = (fcount[] += 1; f(x))
    g_c!(G, x) = (g!(G, x); G)
    fmin, xmin = Base.invokelatest(LBFGSB.lbfgsb, f_c, g_c!, x0; lb=lo, ub=hi, maxiter=maxiter)
    return (collect(Float64, xmin), Float64(fmin), fcount[], fcount[] < maxiter)
end

# ── A. b1 on the real objective ─────────────────────────────────────────────────────────────────
θ_start = [0.02, 0.3, -0.2, 0.25, 0.1]          # a start the toy box and these data move from
say("A: δ at θ_start")
setmu!(θ_start); δA = copy(δ_logshare); cA, iA, _ = blp_contraction!(buf, δA, pc, R; tol = TOL, max_iter = 50000)
say("   contraction from log shares: conv $cA in $iA evals")
pts = [θ_start, θ_start .+ [0.0, 0.02, -0.01, 0.01, 0.0], θ_start .+ [0.01, -0.03, 0.02, 0.0, 0.02]]
# SMOKE_PARTS (default ABCD) runs a subset of the sections.
const SMOKE_PARTS = get(ENV, "SMOKE_PARTS", "ABCD")
if occursin('A', SMOKE_PARTS)
# The solver comparison does not depend on the inner cap (both solvers see the same objective), so
# it is kept small: a far trial costs CAP_A iterations, not 5000.
MAXIT = 6; CAP_A = 300
runs = Dict{String,Any}()
for (tag, solver) in (("today1", (f, g) -> solve_box_lbfgsb_today(f, g, copy(θ_start), lo, hi; maxiter = MAXIT)),
                      ("today2", (f, g) -> solve_box_lbfgsb_today(f, g, copy(θ_start), lo, hi; maxiter = MAXIT)),
                      ("b1off",  (f, g) -> solve_box_lbfgsb(f, g, copy(θ_start), lo, hi; maxiter = MAXIT, stop_reason = false, step = :off)),
                      ("b1on",   (f, g) -> solve_box_lbfgsb(f, g, copy(θ_start), lo, hi; maxiter = MAXIT, step = :off)))
    f, g!, tr, dw, _ = engine_objective(δA; cap = CAP_A)
    t0 = time(); r = solver(f, g!); runs[tag] = (r = r, tr = tr, dw = copy(dw), sec = time() - t0)
    say("   $tag: $(length(tr)) evaluations in $(round(time() - t0, digits = 1)) s, Q $(r[2]), x $(round.(r[1], digits = 5))" *
        (length(r) >= 6 ? ", stop '$(get(r[6], "stop_reason", "-"))', line search '$(get(r[6], "line_search", "-"))', " *
                          "$(get(r[6], "n_iter", "-")) iterations, f_start $(get(r[6], "f_start", "-")), " *
                          "f_at_x $(get(r[6], "f_at_x", "-")), moved $(get(r[6], "moved", "-"))" : ""))
end
eq(a, b) = same_trace(a.tr, b.tr) && same_bits(a.r[1], b.r[1]) && isequal(a.r[2], b.r[2]) && a.r[3] == b.r[3] &&
           a.r[4] == b.r[4] && same_bits(a.dw, b.dw)
ctrl = eq(runs["today1"], runs["today2"])
check("A control: today's solver twice, same bits (the objective is reproducible here)", ctrl,
      "$(length(runs["today1"].tr)) evaluations")
check("A b1 off == today (points, minimizer, minimum, count, converged, final δ; bits)", eq(runs["today1"], runs["b1off"]))
check("A b1 on (default) == today (bits)", eq(runs["today1"], runs["b1on"]),
      "stop '$(runs["b1on"].r[6]["stop_reason"])', moved $(runs["b1on"].r[6]["moved"])")
end

# ── B. b3 ─────────────────────────────────────────────────────────────────────────────────────────
if occursin('B', SMOKE_PARTS)
say("B: guarded_fg! against gmm_fg!")
fu, gu, tru, dwu, _  = engine_objective(δA)
fg, gg, trg, dwg, gd = engine_objective(δA; guard = true)
okB = true; detB = String[]
for (k, t) in enumerate(pts)
    Gu = zeros(n_params); Gg = zeros(n_params)
    Qu = fu(t); gu(Gu, t); Qg = fg(t); gg(Gg, t)
    s = isequal(Qu, Qg) && same_bits(Gu, Gg) && same_bits(dwu, dwg)
    global okB &= s
    push!(detB, "point $k Q $(Qu) $(s ? "==" : "!=") $(Qg)")
end
check("B b3, every contraction converged: Q, ∇Q, δ bit for bit at 3 points", okB && gd.n_fail == 0, join(detB, "; "))
# A forced failure: the next evaluation gets an inner cap of 2 iterations.
t_far = θ_start .+ [0.0, 0.8, -0.6, 0.5, 0.4]
δ_before = copy(dwg); G = zeros(n_params)
Qf = guarded_fg!(G, t_far, gd, dwg,
        (d, cap) -> (setmu!(t_far); cr = blp_contraction!(buf, d, pc, R; tol = TOL, max_iter = 2); (cr[1], cr[2])),
        d -> (gm = compute_gmm_moments(estimate_theta1(d, pc)[2], pc); (dot(gm, W * gm), gm)),
        (G_, gm, d) -> (compute_ift_gradient!(G_, gm, d, t_far, buf, prod_vec, nu_draws, sigma_indices,
                                              pi_interactions, R, coef_dim, W, pc); LAST_JACOBIAN_SINGULAR[]);
        cap = 2, cap_first = 2)
expect = gd.Q_ref + 2 * abs(dot(gd.g_ref, t_far .- gd.theta_ref)) + 1e-8 * max(1.0, abs(gd.Q_ref))
check("B b3 forced failure: Q_fail as specified, zero gradient, δ restored, counted",
      isequal(Qf, expect) && all(G .== 0) && same_bits(dwg, δ_before) && gd.n_fail == 1,
      "Q_fail $(Qf) (Q_ref $(gd.Q_ref)), n_fail $(gd.n_fail)")
end

# ── C. b4 ─────────────────────────────────────────────────────────────────────────────────────────
if occursin('C', SMOKE_PARTS)
say("C: the clamp test")
θc = pts[2]
function contract_at(t, d0; ctest::Bool, force_row::Int = 0)
    setmu!(t)
    force_row > 0 && (buf.mu[force_row, :] .= -2000.0)
    d = copy(d0)
    c, it, _ = withenv(() -> blp_contraction!(buf, d, pc, R; tol = TOL, max_iter = 50000),
                       "BLP_INNER_CLAMP_TEST" => ctest ? "1" : "0")
    return d, c, it, copy(CLAMP_STATS[])
end
d0, c0, i0, _  = contract_at(θc, δA; ctest = false)
d1, c1, i1, s1 = contract_at(θc, δA; ctest = true)
check("C b4 with nothing at the clamp == off (δ bits, converged, iterations)",
      same_bits(d0, d1) && c0 == c1 && i0 == i1 && get(s1, "exit", "") == "converged" &&
      get(s1, "n_delta_at_clamp", -1) == 0, "conv $c0 in $i0 evals; stats $(s1)")
row = findfirst(pc.b_mask)
d2, c2, i2, _  = contract_at(θc, δA; ctest = false, force_row = row)
d3, c3, i3, s3 = contract_at(θc, δA; ctest = true, force_row = row)
check("C b4 forced clamp: today converged with δ at +500, b4 clamp_bound with the counts",
      c2 && d2[row] == 500.0 && !c3 && get(s3, "exit", "") == "clamp_bound" && get(s3, "n_bound", 0) >= 1 &&
      get(s3, "n_V_at_clamp", 0) >= R && i3 == i2,
      "today conv $c2 in $i2 evals, δ[row] $(d2[row]); b4 conv $c3 in $i3 evals, stats $(s3)")
setmu!(θc)
end

# ── D. b2 on the real objective ───────────────────────────────────────────────────────────────────
if occursin('D', SMOKE_PARTS)
# From the start where section A's solvers froze (same objective: δA, inner cap 300), 15 iterations.
say("D: b2 lite, b2 lite + b3, b2 full + b3 from section A's frozen start")
for (tag, kw, guard) in (("lite", (step = :lite, r0 = 0.02), false),
                         ("lite+guard", (step = :lite, r0 = 0.02), true),
                         ("full+guard", (step = :full, r0 = 0.02, rmax = 1.0, rmin = 1e-4), true))
    f, g!, tr, _, gdd = engine_objective(δA; guard = guard, cap = 300)
    t0 = time(); r = solve_box_lbfgsb(f, g!, copy(θ_start), lo, hi; maxiter = 15, kw...)
    i = r[6]
    check("D b2 $tag leaves the start where today's solver froze and moves downhill", r[1] != θ_start && i["f_at_x"] < i["f_start"],
          "Q $(round(i["f_start"], sigdigits = 6)) -> $(round(i["f_at_x"], sigdigits = 6)), moved $(round(i["moved"], sigdigits = 3)), " *
          "$(r[3]) evaluations in $(round(time() - t0, digits = 1)) s, stop '$(i["stop_reason"])'" *
          (haskey(i, "rounds") ? ", $(length(i["rounds"])) rounds, end '$(i["end_note"])'" : "") *
          (gdd === nothing ? "" : ", guard failures $(gdd.n_fail)/$(gdd.n_eval)"))
end
end

say(isempty(FAILS) ? "SMOKE ALL PASS" : "SMOKE FAIL: " * join(FAILS, "; "))
isempty(FAILS) || exit(1)
