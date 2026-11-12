# Local CPU test of the outer solver of blp_engine_cpu.jl (engine edit (b), 2026-11-11): b1 (the real
# stop reason), b2 (first-step control: lite, full) and b3 (the inner-loop guard, `guarded_fg!`), on toy
# problems, against today's solver (`solve_box_lbfgsb` as it stood at HEAD 24f5d096, copied below).
#   julia --project=<repo> --threads=1 test_solve_box.jl <repo>
# Prints one PASS/FAIL line per check and "ALL PASS" or "SOME FAIL" at the end; exits 1 on a failure.
#
# 1. Smooth problems, flags off: b1 off and b1 on (the default) reproduce today's solver bit for bit:
#    every point evaluated, the minimizer, the minimum, the evaluation count and `converged`.
# 2. The defect: a quadratic whose minimum is 0.05 from θ₀, with a gradient of about 4 there, and
#    whose objective is garbage farther than 0.1 from θ₀: a large value there, and an inner state
#    ("δ") left polluted, so every later evaluation reads 1.0 too high, as a δ held at the ±500
#    clamp does. Two kinds of garbage (huge; moderate). Today's solver returns θ₀ with converged =
#    true; b1 returns the same bit for bit and its record shows the freeze (moved 0, the line search's
#    "WARNING: STP = STPMIN", f at θ₀ above f at the start, stop "CONVERGENCE: REL_REDUCTION_OF_F…");
#    b2 lite and b2 full reach the minimum; b3 alone (the guard, no step control) reaches it too.
# 3. b3 is bit-identical to the unguarded evaluation when every "contraction" converges.
const REPO = ARGS[1]
include(joinpath(REPO, "blp_engine_cpu.jl"))
using LinearAlgebra, Printf

# ── Today's solver (blp_engine_cpu.jl at HEAD 24f5d096, l.77–114), verbatim but for its name ──────
function solve_box_lbfgsb_today(f, g!, x0::Vector{Float64},
                                lo::Vector{Float64}, hi::Vector{Float64};
                                tol_outer::Float64=1e-6, maxiter::Int=500, m::Int=10)
    n = length(x0)
    if HAVE_LBFGSB
        fcount = Ref(0)
        f_c(x)     = (fcount[] += 1; f(x))
        g_c!(G, x) = (g!(G, x); G)
        for attempt in 1:2
            try
                fmin, xmin = if attempt == 1
                    Base.invokelatest(LBFGSB.lbfgsb, f_c, g_c!, x0;
                        lb=lo, ub=hi, maxiter=maxiter)
                else
                    bounds = zeros(3, n)
                    @inbounds for i in 1:n
                        bounds[1, i] = 2.0; bounds[2, i] = lo[i]; bounds[3, i] = hi[i]
                    end
                    opt = Base.invokelatest(LBFGSB.L_BFGS_B, n, m)
                    Base.invokelatest(opt, f_c, g_c!, x0, bounds; maxiter=maxiter)
                end
                if xmin isa AbstractVector && length(xmin) == n && isfinite(Float64(fmin))
                    return (collect(Float64, xmin), Float64(fmin),
                            fcount[], fcount[] < maxiter)
                end
            catch e
                attempt == 2 && @warn "L-BFGS-B (LBFGSB.jl) call failed — falling back " *
                    "to Fminbox(LBFGS()). Check the LBFGSB API/version." exception=(e, catch_backtrace())
            end
        end
    end
    res = optimize(f, g!, lo, hi, x0, Fminbox(LBFGS()),
                   Optim.Options(iterations=maxiter, f_reltol=tol_outer, show_trace=false))
    return (Optim.minimizer(res), Optim.minimum(res),
            Optim.iterations(res), Optim.converged(res))
end

const FAILS = String[]
function check(name, ok, detail = "")
    println((ok ? "PASS " : "FAIL ") * name * (isempty(detail) ? "" : ": " * detail))
    ok || push!(FAILS, name)
    return ok
end
bits(v::AbstractVector{Float64}) = reinterpret(UInt64, v)
same_bits(a, b) = length(a) == length(b) && all(bits(a) .== bits(b))
same_trace(a, b) = length(a) == length(b) && all(same_bits(x, y) for (x, y) in zip(a, b))
vfmt(v) = "[" * join([@sprintf("%.6g", x) for x in v], ", ") * "]"

# ── 1. Smooth problems: b1 off and on against today's solver ─────────────────────────────────────
"""f and g! that record every point the solver evaluates (f is called first at each point)."""
function traced(fun, grad!)
    tr = Vector{Vector{Float64}}()
    f(x) = (push!(tr, copy(x)); fun(x))
    g!(G, x) = (grad!(G, x); G)
    return f, g!, tr
end

A5 = [2.0 0.3 0 0 0; 0.3 5.0 0.4 0 0; 0 0.4 9.0 0.2 0; 0 0 0.2 14.0 0.5; 0 0 0 0.5 25.0]
c5 = [0.4, -0.7, 1.3, 0.2, -0.9]
rosen(x) = sum(100 * (x[i+1] - x[i]^2)^2 + (1 - x[i])^2 for i in 1:length(x)-1)
function rosen_g!(G, x)
    n = length(x); fill!(G, 0.0)
    for i in 1:n-1
        G[i]   += -400 * x[i] * (x[i+1] - x[i]^2) - 2 * (1 - x[i])
        G[i+1] += 200 * (x[i+1] - x[i]^2)
    end
    return G
end
blp_like(x) = log(1 + sum(exp.(x .- [0.3, -0.2, 0.1, 0.0, 0.4]))) + 0.5 * sum(abs2, x) - 0.2 * x[1]
function blp_like_g!(G, x)
    e = exp.(x .- [0.3, -0.2, 0.1, 0.0, 0.4]); s = e ./ (1 + sum(e))
    G .= s .+ x; G[1] -= 0.2; return G
end
smooth = [
    ("quadratic 5-D, interior minimum", x -> 0.5 * dot(x .- c5, A5 * (x .- c5)) + 0.03,
         (G, x) -> (G .= A5 * (x .- c5); G), [0.3, 0.1, -0.2, 0.05, 0.4], [0.0, -5, -5, -5, -5], fill(5.0, 5)),
    ("Rosenbrock 4-D, box [-2, 2]", rosen, rosen_g!, [-1.2, 1.0, -1.2, 1.0], fill(-2.0, 4), fill(2.0, 4)),
    ("quadratic, minimum outside the box (two bounds bind)",
         x -> sum(abs2, x .- [3.0, -0.5, -4.0]), (G, x) -> (G .= 2 .* (x .- [3.0, -0.5, -4.0]); G),
         [0.1, 0.2, 0.3], fill(-2.0, 3), fill(2.0, 3)),
    ("logit-like 5-D, sigma-type bound at 0", blp_like, blp_like_g!, [0.5, 0.1, -0.3, 0.2, 0.0],
         [0.0, -5, -5, -5, -5], fill(5.0, 5)),
]
for (name, fun, grad!, x0, lo, hi) in smooth
    f0, g0, tr0 = traced(fun, grad!)
    r0 = solve_box_lbfgsb_today(f0, g0, copy(x0), lo, hi)
    f1, g1, tr1 = traced(fun, grad!)
    r1 = solve_box_lbfgsb(f1, g1, copy(x0), lo, hi; stop_reason = false, step = :off)
    f2, g2, tr2 = traced(fun, grad!)
    r2 = solve_box_lbfgsb(f2, g2, copy(x0), lo, hi)           # the defaults: b1 on, step off
    same1 = same_trace(tr0, tr1) && same_bits(r0[1], r1[1]) && isequal(r0[2], r1[2]) && r0[3] == r1[3] && r0[4] == r1[4]
    same2 = same_trace(tr0, tr2) && same_bits(r0[1], r2[1]) && isequal(r0[2], r2[2]) && r0[3] == r2[3] && r0[4] == r2[4]
    check("1 $name: b1 off == today (bits)", same1, "$(length(tr0)) evaluations")
    check("1 $name: b1 on (default) == today (bits)", same2,
          "$(length(tr2)) evaluations, stop '$(r2[6]["stop_reason"])', $(r2[6]["n_iter"]) iterations, " *
          "f_at_x $(r2[6]["f_at_x"]) vs minimum $(r2[2])")
    check("1 $name: n_evals == n_feval, 6 values", length(r2) == 6 && r2[5] == r2[3])
end

# ── 2. The defect toy ─────────────────────────────────────────────────────────────────────────────
const θ0 = [0.01, -0.02, 0.03, 0.0, 0.01]
const XS = θ0 .+ [0.04, -0.03, 0.05, -0.02, 0.03]          # the minimum, 0.05 from θ0
const AQ = [100.0, 60.0, 80.0, 120.0, 90.0]                  # curvature: the gradient at θ0 is about 4
const RAD = 0.1                                              # garbage beyond this sup-norm radius
const LO = fill(-5.0, 5); const HI = fill(5.0, 5)
q_true(x) = 0.5 * sum(AQ .* (x .- XS) .^ 2) + 0.1
q_grad!(G, x) = (G .= AQ .* (x .- XS); G)
far(x) = maximum(abs.(x .- θ0)) > RAD

# The toy's inner state d = [pollution]: a far point fails (not converged) and pollutes d; a near point
# converges but cannot clear a pollution it inherits (a δ coordinate held at the clamp), unless the
# caller restarts it from a clean d. Two kinds of garbage: (A) a huge value far away and a pollution
# of 1.0; (B) a moderate one (Q(θ0) + 2) and a pollution of 0.5.
const KINDS = Dict("A" => (pollution = 1.0, garbage = x -> 50.0 + 10 * sum(abs2, x .- θ0)),
                   "B" => (pollution = 0.5, garbage = x -> q_true(θ0) + 2.0))
toy_contract!(k, x, d, cap) = far(x) ? (d[1] = KINDS[k].pollution; (false, cap)) : (true, 1)
toy_objective(k, x, d) = far(x) ? (KINDS[k].garbage(x), nothing) : (q_true(x) + d[1], nothing)
toy_gradient!(x, G, d) = (far(x) ? fill!(G, 0.0) : q_grad!(G, x); 0)

"""Today's engine objective (gmm_fg_gpu!-like): contract from the current d, then Q and ∇Q whatever
the contraction did, with a one-point cache shared by f and g!."""
function engine_today(k)
    d = [0.0]; cx = fill(NaN, 5); cf = Ref(0.0); cg = zeros(5); tr = Vector{Vector{Float64}}()
    function compute!(x)
        isequal(x, cx) && return
        push!(tr, copy(x))
        toy_contract!(k, x, d, 5000)
        cf[] = toy_objective(k, x, d)[1]
        toy_gradient!(x, cg, d)
        copyto!(cx, x)
    end
    return (x -> (compute!(x); cf[])), ((G, x) -> (compute!(x); copyto!(G, cg); G)), tr
end
"""The same objective through guarded_fg! (b3)."""
function engine_guarded(k)
    d = [0.0]; cx = fill(NaN, 5); cf = Ref(0.0); cg = zeros(5); tr = Vector{Vector{Float64}}()
    gd = guard_reset!(InnerGuard(5, 1), d)
    function compute!(x)
        isequal(x, cx) && return
        push!(tr, copy(x))
        cf[] = guarded_fg!(cg, x, gd, d, (dd, cap) -> toy_contract!(k, x, dd, cap),
                           dd -> toy_objective(k, x, dd), (G, gm, dd) -> toy_gradient!(x, G, dd);
                           cap = 5000, cap_first = 50000)
        copyto!(cx, x)
    end
    return (x -> (compute!(x); cf[])), ((G, x) -> (compute!(x); copyto!(G, cg); G)), tr, gd
end
dist(x) = maximum(abs.(x .- XS))
info_note(i) = "stop '$(i["stop_reason"])', line search '$(i["line_search"])', $(i["n_iter"]) iteration(s), " *
               "moved $(i["moved"]), f_start $(round(i["f_start"], sigdigits = 6)), f_at_x " *
               "$(round(i["f_at_x"], sigdigits = 6))"

for k in ("A", "B")
    fT, gT, trT = engine_today(k)
    rT = solve_box_lbfgsb_today(fT, gT, copy(θ0), LO, HI)
    check("2$k defect: today's solver returns θ0 with converged = true", same_bits(rT[1], θ0) && rT[4] == true,
          "x $(vfmt(rT[1])), Q $(rT[2]) (Q at θ0 $(q_true(θ0))), $(rT[3]) evaluations, converged $(rT[4]); " *
          "first trial $(vfmt(trT[min(2, end)]))")

    fB, gB, trB = engine_today(k)
    rB = solve_box_lbfgsb(fB, gB, copy(θ0), LO, HI)             # b1 on (default), step off
    check("2$k defect: b1 returns the same, bit for bit", same_bits(rB[1], rT[1]) && isequal(rB[2], rT[2]) &&
          rB[3] == rT[3] && rB[4] == rT[4] && same_trace(trB, trT))
    # b1's record exposes the freeze: no move, Q at the returned θ0 above Q at the start, and the
    # messages. Both kinds end the same way, not with ABNORMAL_TERMINATION_IN_LNSRCH: the line search
    # shrinks its step to 0 (dcsrch "WARNING: STP = STPMIN"), L-BFGS-B accepts θ0 re-evaluated from the
    # polluted state as its new iterate, and stops on CONVERGENCE: REL_REDUCTION_OF_F because f rose.
    want = startswith(rB[6]["stop_reason"], "CONVERGENCE: REL_REDUCTION") &&
           startswith(rB[6]["line_search"], "WARNING") && rB[6]["f_at_x"] > rB[6]["f_start"]
    check("2$k defect: b1 records the freeze (moved 0, line-search WARNING, f rose)",
          rB[6]["moved"] == 0.0 && want, info_note(rB[6]))

    fL, gL, trL = engine_today(k)
    rL = solve_box_lbfgsb(fL, gL, copy(θ0), LO, HI; step = :lite, r0 = 0.02)
    check("2$k defect: b2 lite reaches the minimum", dist(rL[1]) < 1e-5 && abs(rL[2] - 0.1) < 1e-8 && rL[4],
          "max|x - x*| $(@sprintf("%.2e", dist(rL[1]))), Q $(rL[2]), stop '$(rL[6]["stop_reason"])', " *
          "scale c $(round(rL[6]["scale"], sigdigits = 4)), $(rL[3]) evaluations, max|trial - θ0| " *
          "$(@sprintf("%.3f", maximum(maximum(abs.(t .- θ0)) for t in trL)))")

    fF, gF, trF = engine_today(k)
    rF = solve_box_lbfgsb(fF, gF, copy(θ0), LO, HI; step = :full, r0 = 0.02, rmax = 1.0, rmin = 1e-4)
    check("2$k defect: b2 full reaches the minimum", dist(rF[1]) < 1e-5 && abs(rF[2] - 0.1) < 1e-8 && rF[4],
          "max|x - x*| $(@sprintf("%.2e", dist(rF[1]))), Q $(rF[2]), $(length(rF[6]["rounds"])) round(s), " *
          "end '$(rF[6]["end_note"])', $(rF[3]) evaluations")

    if k == "A"
        # The first moving box (0.5) is wider than the garbage radius: round 1 fails (its line search
        # gives up, no progress), r halves, and the minimizer is still reached. Q stays 1.0 too high
        # there: the toy's pollution persists without b3.
        fW, gW, trW = engine_today(k)
        rW = solve_box_lbfgsb(fW, gW, copy(θ0), LO, HI; step = :full, r0 = 0.5, rmax = 1.0, rmin = 1e-4)
        rds = rW[6]["rounds"]
        check("2A defect: b2 full from r0 0.5: a failed round halves r, then the minimizer",
              rds[1]["failed"] && rds[2]["r"] == 0.25 && dist(rW[1]) < 1e-5,
              "rounds: " * join(["r $(d["r"]) $(d["failed"] ? "failed" : d["on_moving_face"] ? "face" : "inside")"
                                 for d in rds], ", ") * "; max|x - x*| $(@sprintf("%.2e", dist(rW[1]))), Q $(rW[2])")
    end

    fG, gG, trG, gdG = engine_guarded(k)
    rG = solve_box_lbfgsb(fG, gG, copy(θ0), LO, HI)             # b3 alone (b1 on, step off)
    check("2$k defect: b3 alone (guard, no step control) reaches the minimum",
          dist(rG[1]) < 1e-5 && abs(rG[2] - 0.1) < 1e-8 && rG[4] && gdG.n_fail >= 1,
          "max|x - x*| $(@sprintf("%.2e", dist(rG[1]))), Q $(rG[2]), stop '$(rG[6]["stop_reason"])', " *
          "$(rG[3]) evaluations, guard failures $(gdG.n_fail) of $(gdG.n_eval)")

    fLG, gLG, trLG, gdLG = engine_guarded(k)
    rLG = solve_box_lbfgsb(fLG, gLG, copy(θ0), LO, HI; step = :lite, r0 = 0.02)
    check("2$k defect: b2 lite + b3 reaches the minimum", dist(rLG[1]) < 1e-5 && rLG[4],
          "max|x - x*| $(@sprintf("%.2e", dist(rLG[1]))), $(rLG[3]) evaluations, guard failures $(gdLG.n_fail)")
end

# ── 3. b3 when every contraction converges: bit-identical to the unguarded evaluation ─────────────
# A near-only problem (no point is far): the guarded and unguarded objectives give the same trace,
# minimizer, minimum and evaluation count.
function engine_pair(guarded::Bool)
    d = [0.0]; cx = fill(NaN, 5); cf = Ref(0.0); cg = zeros(5); tr = Vector{Vector{Float64}}()
    gd = guard_reset!(InnerGuard(5, 1), d)
    con!(x, dd, cap) = (dd[1] = dd[1] * 0.5; (true, 1))     # a "contraction" that converges and moves d
    obj(x, dd) = (q_true(x) + dd[1], nothing)
    gra!(x, G, dd) = (q_grad!(G, x); 0)
    function compute!(x)
        isequal(x, cx) && return
        push!(tr, copy(x))
        if guarded
            cf[] = guarded_fg!(cg, x, gd, d, (dd, cap) -> con!(x, dd, cap), dd -> obj(x, dd),
                               (G, gm, dd) -> gra!(x, G, dd); cap = 5000, cap_first = 50000)
        else
            con!(x, d, 5000); cf[] = obj(x, d)[1]; gra!(x, cg, d)
        end
        copyto!(cx, x)
    end
    return (x -> (compute!(x); cf[])), ((G, x) -> (compute!(x); copyto!(G, cg); G)), tr
end
fu, gu, tru = engine_pair(false); ru = solve_box_lbfgsb(fu, gu, copy(θ0), LO, HI; step = :lite, r0 = 0.02)
fg, gg, trg = engine_pair(true);  rg = solve_box_lbfgsb(fg, gg, copy(θ0), LO, HI; step = :lite, r0 = 0.02)
check("3 b3 with every contraction converged == unguarded (bits)",
      same_trace(tru, trg) && same_bits(ru[1], rg[1]) && isequal(ru[2], rg[2]) && ru[3] == rg[3],
      "$(length(tru)) evaluations")

# ── 4. The knobs: env parsing and refusals ────────────────────────────────────────────────────────
withenv("BLP_OUTER_STEP" => "lite", "BLP_OUTER_R0" => "0.05", "BLP_OUTER_STOP_REASON" => "0",
        "BLP_INNER_GUARD" => "1", "BLP_INNER_CLAMP_TEST" => "1") do
    check("4 env knobs read", outer_step_mode() == :lite && outer_r0() == 0.05 && !outer_stop_reason() &&
          inner_guard_on() && inner_clamp_test(), outer_knobs_note())
end
withenv("BLP_OUTER_STEP" => nothing, "BLP_OUTER_STOP_REASON" => nothing, "BLP_INNER_GUARD" => nothing,
        "BLP_INNER_CLAMP_TEST" => nothing) do
    check("4 defaults: stop reason on, step off, guard off, clamp test off",
          outer_stop_reason() && outer_step_mode() == :off && !inner_guard_on() && !inner_clamp_test(),
          outer_knobs_note())
end
withenv("BLP_OUTER_STEP" => "fast") do
    check("4 a wrong BLP_OUTER_STEP is refused", try outer_step_mode(); false catch; true end)
end

println(isempty(FAILS) ? "ALL PASS" : "SOME FAIL: " * join(FAILS, "; "))
isempty(FAILS) || exit(1)
