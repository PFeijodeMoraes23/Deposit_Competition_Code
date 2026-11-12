"""
blp_estimation.jl
=================
Self-contained BLP demand estimation using pre-computed draws from blp_draws.jl.

All BLP core functions are inlined here — blp_loop.jl is NOT loaded at runtime.
Pre-computed ν-draws and demographic draws are loaded once from the draws directory
(`draws_dir`, of_root.jl: data/output/blp/draws on the cluster, ESTIMATION_OUTPUT/BLP_DRAWS
locally) and shared across all specs/stages, ensuring cross-strategy comparability.

Differences from blp_loop.jl:
  1. Draws are LOADED (not generated) — blp_draws.jl must run first
  2. Tighter outer bounds [-5, 5] per CG2020
  3. Clustering on CodConglomeradoPrudencial only (matches sleep estimation)

Usage (cluster):
  julia --project=\${PROJECT_DIR} --threads=32 \\
      blp_estimation.jl --estim 1 --spec 12 --stage sequence \\
      --R 2000 --seed 42 --hpc

Local test:
  julia --project=. --threads=4 blp_estimation.jl \\
      --estim 1 --spec 12 --stage sigma --R 50 --seed 42

References
----------
  Berry, Levinsohn & Pakes (1995, Econometrica)
  Conlon & Gortmaker (2020, RAND J. Econ.)
"""

using MKL                       # must precede LinearAlgebra — replaces OpenBLAS with Intel MKL
using LoopVectorization         # @turbo SIMD annotations
using Parquet2, DataFrames, SparseArrays, LinearAlgebra, Statistics
using Random, Optim, QuasiMonteCarlo, Distributions
using JSON3, Serialization, ArgParse, Printf, Dates

# Shared SE machinery (wild cluster bootstrap + analytical GMM sandwich, BLP_SE_METHOD).
include(joinpath(@__DIR__, "blp_se_common.jl"))
include(joinpath(@__DIR__, "of_root.jl"))
# The input rule (missing / non-finite values) and the D-type dummy, shared with blp_logit.jl.
isdefined(Main, :DemandInputError) || include(joinpath(@__DIR__, "demand_inputs.jl"))
# The lineup and the demand prefixes come from config/routines.toml, which utils/routines.py
# reads too. Guarded because several files in one session want the consts.
isdefined(Main, :ROUTINE_REGISTRY) || include(joinpath(@__DIR__, "routines.jl"))

# ── Optional: true L-BFGS-B outer solver (Byrd–Lu–Nocedal–Zhu) ──────────────────
# LBFGSB.jl handles the box directly (active set + projected line search), whereas
# Optim's Fminbox(LBFGS()) solves a SEQUENCE of barrier subproblems and so burns
# many extra objective/gradient evaluations to reach the same KKT point. Loaded
# OPTIONALLY: if absent (or its API drifts), `solve_box_lbfgsb` falls back loudly to
# Fminbox, so a job never fails for lack of it. Enable once with:
#     julia --project=. -e 'using Pkg; Pkg.add("LBFGSB")'
const HAVE_LBFGSB = try
    @eval import LBFGSB
    true
catch
    false
end
HAVE_LBFGSB || @warn "LBFGSB.jl not installed — box-constrained outer optimisation " *
    "will use Optim's Fminbox(LBFGS()). Run `Pkg.add(\"LBFGSB\")` for true L-BFGS-B."

# ── Outer-solver knobs (engine edit (b), 2026-11-11) ────────────────────────────
# Read at call time, like BLP_FD_CHECK, so a job gets them through `sbatch --export=ALL`; each is
# also a keyword of `solve_box_lbfgsb`. The defaults reproduce the runs made before the edit.
"""BLP_OUTER_STOP_REASON (default 1; 0 = the `LBFGSB.lbfgsb` call): `solve_box_lbfgsb` drives L-BFGS-B
through `_lbfgsb_loop`, which keeps the stop message, the iteration count and f at the returned x."""
outer_stop_reason()::Bool = get(ENV, "BLP_OUTER_STOP_REASON", "1") != "0"

"""BLP_OUTER_STEP: `off` (default), `lite` or `full`, the first-step control of `solve_box_lbfgsb`."""
function outer_step_mode()::Symbol
    v = lowercase(strip(get(ENV, "BLP_OUTER_STEP", "off")))
    v in ("off", "lite", "full") || error("BLP_OUTER_STEP='$v' is not off, lite or full")
    return Symbol(v)
end
"""BLP_OUTER_R0 (default 0.02): under `lite` the largest move of any coordinate at the first trial;
under `full` the first radius of the moving box."""
outer_r0()::Float64   = parse(Float64, get(ENV, "BLP_OUTER_R0",   "0.02"))
"""BLP_OUTER_RMAX (default 1.0): the largest radius of the moving box (`full`)."""
outer_rmax()::Float64 = parse(Float64, get(ENV, "BLP_OUTER_RMAX", "1.0"))
"""BLP_OUTER_RMIN (default 1e-4): a `full` solve ends when its radius halves below this."""
outer_rmin()::Float64 = parse(Float64, get(ENV, "BLP_OUTER_RMIN", "1e-4"))

"""The outer and inner knobs in force, one line for the log."""
outer_knobs_note() = "stop reason $(outer_stop_reason() ? "on" : "off") | step $(outer_step_mode())" *
    (outer_step_mode() == :off ? "" : " (r0 $(outer_r0())" *
        (outer_step_mode() == :full ? ", rmax $(outer_rmax()), rmin $(outer_rmin()))" : ")")) *
    " | inner guard $(inner_guard_on() ? "on" : "off") | clamp test $(inner_clamp_test() ? "on" : "off")"

# L-BFGS-B's stopping tolerances and evaluation cap: the defaults of LBFGSB.jl 0.4.1 (wrapper.jl l.32),
# passed explicitly.
const LBFGSB_FACTR  = 1e7
const LBFGSB_PGTOL  = 1e-5
const LBFGSB_MAXFUN = 15000

"""Bound type of one coordinate as LBFGSB.jl's `typ_bnd`: 0 free, 1 lower only, 2 both, 3 upper only."""
function _lbfgsb_bound_type(l::Float64, u::Float64)::Int
    l > u && error("Inconsistent bounds")
    lbinf = l == -Inf
    ubinf = u == +Inf
    lbinf && ubinf && return 0
    !lbinf && ubinf && return 1
    !lbinf && !ubinf && return 2
    return 3
end

"""
    _lbfgsb_loop(f, g!, x0, lo, hi; m, factr, pgtol, maxiter, maxfun, scale_r0 = nothing) -> NamedTuple

The loop of LBFGSB.jl 0.4.1's `L_BFGS_B` call operator (wrapper.jl l.31–73), written out so that the
stop message survives. It makes the same `setulb` calls with the same arguments in the same order, so
with `scale_r0 = nothing` its iterates are those of `LBFGSB.lbfgsb(f, g!, x0; lb = lo, ub = hi, m,
factr, pgtol, maxiter, maxfun)` bit for bit (diagnostics/test_solve_box.jl checks it).

Returns
  x        L-BFGS-B's x on return: the last accepted iterate (after `ABNORMAL_TERMINATION_IN_LNSRCH`
           it restores the iterate the failed line search started from);
  f_last   the last f requested, what `LBFGSB.lbfgsb` returns (after an abnormal stop it belongs to the
           last rejected trial, not to x);
  f_at_x, g_at_x   f and ∇f at x, from the evaluation that produced x (NaN if none matches x);
  f_start, g_start f and ∇f at the first point evaluated;
  task     the stop message; n_iter (`isave[30]`); n_fg (`isave[34]`); scale (c, 1 unscaled).

`scale_r0 = r` (BLP_OUTER_STEP=lite): f and ∇f at x0 projected on the box are evaluated first and
reused as L-BFGS-B's first evaluation. L-BFGS-B then minimises c·f, c = min(1, r / max_i abs(∇f_i))
over the coordinates not held at a bound by their derivative, so its first trial point
P(x0 − c∇f) moves no coordinate by more than r; pgtol and factr are multiplied by c, which keeps
the stop tests in f units (identical when abs(f) ≤ 1, stricter otherwise). From the first curvature
pair on, the quasi-Newton steps do not depend on c. All values returned are in f units.
"""
function _lbfgsb_loop(f, g!, x0::Vector{Float64}, lo::Vector{Float64}, hi::Vector{Float64};
                      m::Int = 10, factr::Float64 = LBFGSB_FACTR, pgtol::Float64 = LBFGSB_PGTOL,
                      maxiter::Int = 500, maxfun::Int = LBFGSB_MAXFUN,
                      scale_r0::Union{Nothing,Float64} = nothing)
    n   = length(x0)
    opt = Base.invokelatest(LBFGSB.L_BFGS_B, n, m)
    x   = copy(x0)
    fl  = 0.0                                   # the f handed to setulb (scaled), as the wrapper's `f`
    fill!(opt.task, Cuchar(' ')); fill!(opt.csave, Cuchar(' ')); fill!(opt.lsave, zero(Cint))
    fill!(opt.isave, zero(Cint)); fill!(opt.dsave, zero(Cdouble)); fill!(opt.wa, zero(Cdouble))
    fill!(opt.iwa, zero(Cint)); fill!(opt.g, zero(Cdouble)); fill!(opt.nbd, zero(Cint))
    fill!(opt.l, zero(Cdouble)); fill!(opt.u, zero(Cdouble))
    @inbounds for i in 1:n
        opt.nbd[i] = _lbfgsb_bound_type(lo[i], hi[i])
        opt.l[i]   = lo[i]
        opt.u[i]   = hi[i]
    end
    c = 1.0; pre = nothing
    f_start = NaN; g_start = fill(NaN, n)
    if scale_r0 !== nothing
        xp = clamp.(x0, lo, hi)                 # where L-BFGS-B evaluates first (its `active` projection)
        fp = Float64(f(xp)); gp = zeros(n); g!(gp, xp)
        gmax = 0.0
        @inbounds for i in 1:n
            held = (xp[i] <= lo[i] && gp[i] > 0) || (xp[i] >= hi[i] && gp[i] < 0)
            held || (gmax = max(gmax, abs(gp[i])))
        end
        (isfinite(gmax) && gmax > scale_r0) && (c = scale_r0 / gmax)
        pre = (xp, fp, gp)
    end
    factr_c = c == 1.0 ? factr : c * factr
    pgtol_c = c == 1.0 ? pgtol : c * pgtol
    x_last = fill(NaN, n); f_lastv = NaN; g_last = fill(NaN, n)     # the last evaluation (f units)
    x_acc  = fill(NaN, n); f_acc   = NaN; g_acc  = fill(NaN, n)     # the last accepted iterate
    n_fg_calls = 0
    opt.task[1:5] = b"START"
    while true
        Base.invokelatest(LBFGSB.setulb, n, m, x, opt.l, opt.u, opt.nbd, fl, opt.g, factr_c, pgtol_c,
                          opt.wa, opt.iwa, opt.task, -1, opt.csave, opt.lsave, opt.isave, opt.dsave)
        if opt.task[1:2] == b"FG"
            n_fg_calls += 1
            if n_fg_calls == 1 && pre !== nothing && isequal(x, pre[1])
                fv = pre[2]; copyto!(opt.g, pre[3])
            else
                fv = f(x)
                g!(opt.g, x)
            end
            copyto!(x_last, x); f_lastv = Float64(fv); copyto!(g_last, opt.g)
            if n_fg_calls == 1
                f_start = f_lastv; copyto!(g_start, g_last)
                copyto!(x_acc, x); f_acc = f_lastv; copyto!(g_acc, g_last)   # the initial iterate
            end
            if c == 1.0
                fl = fv
            else
                fl = c * fv
                opt.g .*= c
            end
        elseif opt.task[1:5] == b"NEW_X"
            copyto!(x_acc, x)
            if isequal(x, x_last)
                f_acc = f_lastv; copyto!(g_acc, g_last)
            else
                f_acc = NaN; fill!(g_acc, NaN)
            end
            if opt.isave[30] ≥ maxiter
                opt.task[1:43] = b"STOP: TOTAL NO. of ITERATIONS REACHED LIMIT"
            elseif opt.isave[34] ≥ maxfun
                opt.task[1:52] = b"STOP: TOTAL NO. of f AND g EVALUATIONS EXCEEDS LIMIT"
            end
        else
            break
        end
    end
    f_at_x, g_at_x = if isequal(x, x_acc)
        f_acc, copy(g_acc)
    elseif isequal(x, x_last)
        f_lastv, copy(g_last)
    else
        NaN, fill(NaN, n)
    end
    return (x = x, f_last = f_lastv, f_at_x = f_at_x, g_at_x = g_at_x,
            f_start = f_start, g_start = g_start,
            task = String(strip(String(copy(opt.task)))),
            line_search = String(strip(String(copy(opt.csave)))),
            n_iter = Int(opt.isave[30]), n_fg = Int(opt.isave[34]), scale = c)
end

"""True when x sits on a face of the moving box [lo_k, hi_k] that is not a face of the box [lo, hi]
and the derivative there points out of the moving box (a non-finite derivative counts as outward)."""
function _on_moving_face(x::Vector{Float64}, g::Vector{Float64}, lo_k::Vector{Float64},
                         hi_k::Vector{Float64}, lo::Vector{Float64}, hi::Vector{Float64})::Bool
    @inbounds for i in eachindex(x)
        out_lo = !isfinite(g[i]) || g[i] > 0
        out_hi = !isfinite(g[i]) || g[i] < 0
        (x[i] <= lo_k[i] && lo_k[i] > lo[i] && out_lo) && return true
        (x[i] >= hi_k[i] && hi_k[i] < hi[i] && out_hi) && return true
    end
    return false
end

"""
    _lbfgsb_moving_box(f, g!, x0, lo, hi, fcount; m, factr, pgtol, maxiter, r0, rmax, rmin) -> NamedTuple

BLP_OUTER_STEP=full: L-BFGS-B in a moving box [θ_k − r, θ_k + r] ∩ [lo, hi], each round a
`_lbfgsb_loop` lite-scaled with r0 = r, so no round's first trial moves a coordinate by more than r.
After a round:
  - a failed round — no progress (f not below its start by L-BFGS-B's own factr·ε test) and either an
    ABNORMAL stop or a line search that gave up (dcsrch WARNING: L-BFGS-B then re-takes the round's
    start as its iterate and stops on the f test, f having risen): θ stays, r halves; the solve ends
    after two such rounds in a row or when r < rmin;
  - progress, but an ABNORMAL stop or a line search that gave up: restart at its point, same r;
  - converged on a face of the moving box whose derivative points out of it: restart there with
    r ← min(2r, rmax);
  - converged inside the moving box (every binding face is a face of the box): the end;
  - L-BFGS-B's STOP/ERROR, or `fcount[] ≥ maxiter` (the caller's evaluation budget): the end.
Returns x and f, ∇f at x (f units), the last round's stop message (`task`), the iterations summed
over rounds, one Dict per round, and why the solve ended (`end_note`).
"""
function _lbfgsb_moving_box(f, g!, x0::Vector{Float64}, lo::Vector{Float64}, hi::Vector{Float64},
                            fcount::Base.RefValue{Int};
                            m::Int = 10, factr::Float64 = LBFGSB_FACTR, pgtol::Float64 = LBFGSB_PGTOL,
                            maxiter::Int = 500, r0::Float64 = 0.02, rmax::Float64 = 1.0,
                            rmin::Float64 = 1e-4)
    n  = length(x0)
    θ  = clamp.(x0, lo, hi)
    fθ = NaN; gθ = fill(NaN, n); f_first = NaN
    r  = r0
    rounds = Dict{String,Any}[]
    iters = 0; noprog = 0; task = ""; line_search = ""; end_note = ""
    while true
        lo_k = max.(lo, θ .- r)
        hi_k = min.(hi, θ .+ r)
        res = _lbfgsb_loop(f, g!, θ, lo_k, hi_k; m = m, factr = factr, pgtol = pgtol,
                           maxiter = max(1, maxiter - iters),
                           maxfun = max(1, LBFGSB_MAXFUN - fcount[]), scale_r0 = r)
        iters += res.n_iter
        task   = res.task
        line_search = res.line_search
        isfinite(f_first) || (f_first = res.f_start)
        isfinite(fθ) || (fθ = res.f_start; copyto!(gθ, res.g_start))
        f0 = res.f_start; f1 = res.f_at_x
        tolp = factr * eps(Float64) * max(abs(f0), abs(f1), 1.0)
        progress = isfinite(f0) && isfinite(f1) && f1 < f0 - tolp
        gave_up  = startswith(task, "ABNORMAL") || startswith(line_search, "WARNING")
        failed   = !progress && (gave_up || !(f1 <= f0))
        face = !failed && !gave_up && _on_moving_face(res.x, res.g_at_x, lo_k, hi_k, lo, hi)
        push!(rounds, Dict{String,Any}("r" => r, "stop_reason" => task, "line_search" => line_search,
                                       "n_iter" => res.n_iter, "f_start" => f0, "f_end" => f1,
                                       "progress" => progress, "failed" => failed, "on_moving_face" => face,
                                       "scale" => res.scale, "evals_after" => fcount[]))
        log_status("  [OUTER full] round $(length(rounds)): r $(round(r, sigdigits = 3)) | $task" *
                   (isempty(line_search) ? "" : " (line search: $line_search)") * " | " *
                   "Q $(round(f0, sigdigits = 8)) -> $(round(f1, sigdigits = 8)) | " *
                   (failed ? "failed" : face ? "on the moving box" : "inside") * " | evals $(fcount[])")
        if !failed && isfinite(f1)
            θ = copy(res.x); fθ = f1; gθ = copy(res.g_at_x)
        end
        if fcount[] >= maxiter
            end_note = "evaluation budget ($maxiter) spent"; break
        elseif startswith(task, "STOP") || startswith(task, "ERROR")
            end_note = "L-BFGS-B: $task"; break
        elseif failed
            noprog += 1; r /= 2
            noprog >= 2 && (end_note = "two rounds in a row without progress"; break)
            r < rmin && (end_note = "radius below rmin ($rmin)"; break)
            continue
        end
        noprog = 0
        gave_up && continue          # progress, but the last line search gave up: restart there, same r
        if face
            r = min(2r, rmax)
            continue
        end
        end_note = "converged inside the moving box"
        break
    end
    return (x = θ, f_at_x = fθ, g_at_x = gθ, f_start = f_first, task = task,
            line_search = line_search, n_iter = iters, rounds = rounds, end_note = end_note)
end

"""
    solve_box_lbfgsb(f, g!, x0, lo, hi; tol_outer=1e-6, maxiter=500, m=10, factr, pgtol,
                     stop_reason, step, r0, rmax, rmin)
        -> (minimizer, minimum, n_feval, converged, n_evals, info)

Bound-constrained outer minimisation. Prefers true L-BFGS-B (LBFGSB.jl); falls back
LOUDLY to `Fminbox(LBFGS())` if LBFGSB is unavailable or its call errors / returns an
unexpected shape. Both target the same KKT point — L-BFGS-B just reaches it with far
fewer f/∇f evaluations by handling the box natively instead of via barrier subproblems.

`f(x)::Float64` returns the objective; `g!(G, x)` writes ∇f into `G`. Callers should
share a one-point cache between `f` and `g!` so the inner BLP contraction is solved
once per unique `x` (the IFT and numerical engines both do).

`converged` is a conservative proxy: `true` only if it stopped in < maxiter f-evals. It does not say
why L-BFGS-B stopped: `info["stop_reason"]` does (`stop_reason = true`, BLP_OUTER_STOP_REASON, the
default), e.g. `ABNORMAL_TERMINATION_IN_LNSRCH` for a line search that failed. With `step = :off` the
first four values are those of the `LBFGSB.lbfgsb` call; `minimum` is the last f L-BFGS-B requested,
which after an abnormal stop belongs to a rejected trial (`info["f_at_x"]` is f at the minimizer).
`step = :lite` / `:full` (BLP_OUTER_STEP, `_lbfgsb_loop`, `_lbfgsb_moving_box`) bound the first
trial's length and return f at the minimizer. `n_evals` equals `n_feval` (the count of f calls), as
callers that read a fifth value expect. `factr`/`pgtol` are L-BFGS-B's own tolerances; `tol_outer`
is read only by the Fminbox fallback.
"""
function solve_box_lbfgsb(f, g!, x0::Vector{Float64},
                          lo::Vector{Float64}, hi::Vector{Float64};
                          tol_outer::Float64=1e-6, maxiter::Int=500, m::Int=10,
                          factr::Float64=LBFGSB_FACTR, pgtol::Float64=LBFGSB_PGTOL,
                          stop_reason::Bool=outer_stop_reason(), step::Symbol=outer_step_mode(),
                          r0::Float64=outer_r0(), rmax::Float64=outer_rmax(), rmin::Float64=outer_rmin())
    n = length(x0)
    step in (:off, :lite, :full) || error("solve_box_lbfgsb: step = $step is not :off, :lite or :full")
    use_loop = stop_reason || step != :off
    if HAVE_LBFGSB
        fcount = Ref(0)
        f_c(x)     = (fcount[] += 1; f(x))
        g_c!(G, x) = (g!(G, x); G)
        # Try two known LBFGSB.jl call conventions before giving up. Validate the
        # returned shape so a wrong return-order can't silently corrupt the result.
        for attempt in 1:2
            try
                if use_loop
                    # b1/b2: the setulb loop, which keeps L-BFGS-B's stop message.
                    r = if step == :full
                        _lbfgsb_moving_box(f_c, g_c!, x0, lo, hi, fcount; m = m, factr = factr,
                                           pgtol = pgtol, maxiter = maxiter, r0 = r0, rmax = rmax,
                                           rmin = rmin)
                    else
                        _lbfgsb_loop(f_c, g_c!, x0, lo, hi; m = m, factr = factr, pgtol = pgtol,
                                     maxiter = maxiter, scale_r0 = step == :lite ? r0 : nothing)
                    end
                    fret = step == :off ? r.f_last : r.f_at_x
                    if r.x isa AbstractVector && length(r.x) == n && isfinite(Float64(fret))
                        # A start that "froze" shows here whatever the stop message says: moved 0
                        # (x is the start, bit for bit) and, when a line search gave up after
                        # setting its step back to 0, f_at_x above f_start.
                        info = Dict{String,Any}(
                            "solver" => "L-BFGS-B (setulb loop)", "attempt" => attempt,
                            "stop_reason" => r.task, "line_search" => r.line_search,
                            "n_iter" => r.n_iter, "n_evals" => fcount[],
                            "f_start" => r.f_start, "f_at_x" => r.f_at_x,
                            "moved" => maximum(abs.(r.x .- clamp.(x0, lo, hi)); init = 0.0),
                            "step" => String(step),
                            "factr" => factr, "pgtol" => pgtol, "maxiter" => maxiter)
                        if step == :full
                            info["rounds"] = r.rounds; info["end_note"] = r.end_note
                            info["r0"] = r0; info["rmax"] = rmax; info["rmin"] = rmin
                        else
                            info["f_last"] = r.f_last; info["n_fg_lbfgsb"] = r.n_fg
                            step == :lite && (info["scale"] = r.scale; info["r0"] = r0)
                        end
                        return (collect(Float64, r.x), Float64(fret),
                                fcount[], fcount[] < maxiter, fcount[], info)
                    end
                else
                    fmin, xmin = if attempt == 1                      # high-level convenience
                        Base.invokelatest(LBFGSB.lbfgsb, f_c, g_c!, x0;
                            lb=lo, ub=hi, maxiter=maxiter, factr=factr, pgtol=pgtol)
                    else                                             # low-level callable + bounds matrix
                        bounds = zeros(3, n)
                        @inbounds for i in 1:n
                            bounds[1, i] = 2.0; bounds[2, i] = lo[i]; bounds[3, i] = hi[i]
                        end
                        opt = Base.invokelatest(LBFGSB.L_BFGS_B, n, m)
                        Base.invokelatest(opt, f_c, g_c!, x0, bounds; maxiter=maxiter,
                                          factr=factr, pgtol=pgtol)
                    end
                    if xmin isa AbstractVector && length(xmin) == n && isfinite(Float64(fmin))
                        return (collect(Float64, xmin), Float64(fmin),
                                fcount[], fcount[] < maxiter, fcount[],
                                Dict{String,Any}("solver" => "L-BFGS-B (LBFGSB.lbfgsb)",
                                    "attempt" => attempt, "n_evals" => fcount[], "step" => "off",
                                    "stop_reason" => "not recorded (BLP_OUTER_STOP_REASON=0)",
                                    "n_iter" => nothing, "factr" => factr, "pgtol" => pgtol,
                                    "maxiter" => maxiter))
                    end
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
            Optim.iterations(res), Optim.converged(res), Optim.iterations(res),
            Dict{String,Any}("solver" => "Fminbox(LBFGS()) fallback",
                             "stop_reason" => "Fminbox fallback (Optim converged = $(Optim.converged(res)))",
                             "n_iter" => Optim.iterations(res), "n_evals" => Optim.f_calls(res),
                             "f_at_x" => Optim.minimum(res), "step" => "off",
                             "factr" => nothing, "pgtol" => nothing, "maxiter" => maxiter))
end

# ── Set BLAS threads ────────────────────────────────────────────────────────
# Reserve most threads for Julia-level parallelism (logsumexp, q_B loops).
# The main BLAS call here is (N×coef_dim)×(coef_dim×R) — small k-dimension;
# it's memory-bandwidth-bound and doesn't scale past 4–8 threads anyway.
BLAS.set_num_threads(min(8, Threads.nthreads()))

# ==========================================================================
# 0. Constants (must match blp_loop.jl and blp_draws.jl)
# ==========================================================================
# X_COLS: the product characteristics that carry random coefficients. With the spread they form
# `prod_vec` (coef_dim = 1 + L_PROD columns), the matrix the ν draws (2000 × 9) multiply, and the
# π interactions index into it. Adding a column here changes the random-coefficient dimension.
const X_COLS = ["fgc_covered", "has_ip", "seg_S2", "seg_S3", "seg_S4", "seg_S5",
                "log_total_assets_lag", "is_state_owned"]
const L_PROD = length(X_COLS)
# X1_COLS: the LINEAR design (θ₁ after α), X_COLS then the D-type dummy, built from `is_B`
# (demand_inputs.jl `dtype_column`). The dummy enters the linear part only: it is not in prod_vec,
# the draws or the moments. The order is blp_logit.jl's `full_dtype` sub-model, which is therefore
# the θ₂ = 0 nest of this model.
const X1_COLS = vcat(X_COLS, [DTYPE_COL])
const L_X1    = length(X1_COLS)
const K_TYPES = 4
const K_LIST  = [1, 2, 4, 5]
const D_COLS  = ["gdp_per_capita", "fraction_65plus", "fraction_young",
                 "pix_users_pf_per1000", "connections_per100", "frac_4g5g",
                 "branches_per1000", "cadunico_families_per1000"]
const D_DIM   = length(D_COLS)

# Full instrument set (15 = these 11 + IV_COST + IV_CAPITAL). A trim (dropping the near-singular
# duplicate `mean_loo_basileia`, VIF ≈ 707) was considered 2026-07-09 but NOT adopted — the GMM
# `pinv` weight matrix already handles the collinearity, and the linear-logit trim was rejected
# for weakening α (see blp_logit.jl). Kept consistent with the logit; trimming is a robustness
# discussion only (review §4).
# `loo_borrow_assets` / `mean_loo_borrow_assets`: rivals' borrowings and onlendings (IF.data Passivo
# line 78295) over total assets, the ratio of utils/demand_chars.py. Their column names in the stored
# parquets, loo_credit_assets / mean_loo_credit_assets, are read only under BLP_ZERO_FILL_MISSING
# (demand_inputs.jl).
const IV_BLP_LOO = ["loo_log_assets", "mean_loo_log_assets",
                    "loo_equity_ratio", "mean_loo_equity_ratio",
                    "loo_basileia", "mean_loo_basileia",
                    "loo_borrow_assets", "mean_loo_borrow_assets",
                    "loo_npl_provision", "mean_loo_npl_provision",
                    "n_rivals"]
const IV_COST    = ["personnel_cost_ratio_lag", "admin_cost_ratio_lag",
                    "tax_cost_ratio_lag"]
const IV_CAPITAL = ["indice_basileia_lag"]
# ESTBAN branch-competition IV (panel_10): log1p lagged RIVAL-branch count in the MCA — the only IV
# with within-conglomerate (cross-municipality) variation; independent + relevant for demand deposits.
const IV_ESTBAN  = ["estban_rival_branches_lag"]

const _log_buf  = String[]
const _log_lock = ReentrantLock()

function input_filename(estim::Int, spec_id::Int)::String
    # Prefer the prefix auto-discovered + passed by blp_rc.jl (handles new routines with no
    # registry edit); fall back to DEMAND_PREFIXES (routines.jl, from config/routines.toml,
    # where E1/E2 carry the bare demand_{k} and E3/E4 the link tags that
    # estimation_demand_link_common.DEMAND_CFG writes), then the bare demand_{k}.
    pfx = get(ENV, "BLP_DEMAND_PREFIX", "")
    prefix = isempty(pfx) ? routine_demand_prefix(estim) : pfx
    return "$(prefix)_spec_$(spec_id).parquet"
end

"""Suffix appended to all written result/checkpoint/summary filenames, from
ENV["BLP_OUTPUT_SUFFIX"] (default ""). The numerical engine sets it to "_num" so its
outputs (and inter-stage warm-starts) stay self-consistent and never clobber the IFT
results."""
output_suffix()::String = get(ENV, "BLP_OUTPUT_SUFFIX", "")

function log_status(msg::String)
    stamped = "[$(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))] $msg"
    lock(_log_lock) do; push!(_log_buf, stamped); end
    println(stamped); flush(stdout)
end

# ── Version-agnostic δ persistence (raw binary; immune to Julia serialization version) ──
"""Save a Float64 vector to a raw binary file: [Int64 length][Float64... values]"""
function save_delta_bin(path::String, delta::Vector{Float64})
    open(path, "w") do io
        write(io, Int64(length(delta)))
        write(io, delta)
    end
end

"""
    json_se_values(v, se_status, fill; sigdigits=8) -> Vector

Standard errors or p-values as a result JSON stores them. A stage whose joint SEs exist
(`se_status` "ok", or "" for a result that records no status) keeps the encoding every reader
knows: a NaN entry, a σ profiled out at its bound, is written as `fill` (0.0 for an SE, 1.0 for a
p-value) and the tables print it with a dagger. A stage WITHOUT joint SEs (any other status:
refused by the SE guard of blp_se_common.jl, failed, or not computed) writes every non-finite
entry as null, never as a number, so no reader can take a missing SE for a zero one.
"""
function json_se_values(v::AbstractVector, se_status::AbstractString, fill; sigdigits::Int=8)
    se_status in ("", "ok") && return replace(round.(Float64.(v), sigdigits=sigdigits), NaN => fill)
    return Any[isfinite(x) ? round(x, sigdigits=sigdigits) : nothing for x in Float64.(v)]
end

"""Load a Float64 vector from a raw binary file saved by save_delta_bin.
Returns nothing if the file is missing or corrupt."""
function load_delta_bin(path::String)::Union{Vector{Float64},Nothing}
    isfile(path) || return nothing
    try
        open(path, "r") do io
            n     = read(io, Int64)
            delta = Vector{Float64}(undef, n)
            read!(io, delta)
            return delta
        end
    catch _e
        println("  [δ BIN] Load failed: $(_e)")
        return nothing
    end
end

# ==========================================================================
# 0b. Paths (3-tuple: input, draws, output)
# ==========================================================================
# The draws slot is bound to `draws` rather than `draws_dir`: the of_root helper of that name
# is called below, and a local binding of the same name would shadow the function everywhere in
# this scope.
function get_paths(is_hpc::Bool; local_dir=nothing)
    if is_hpc
        input_dir  = joinpath(@__DIR__, "..", "data", "input")
        output_dir = joinpath(@__DIR__, "..", "data", "output")
        # The draws belong to the BLP step folder (data/output/blp/draws) — the draws job is
        # their only producer and they travel in the BLP download alongside the results.
        draws      = draws_dir(output_dir)
    else
        if local_dir !== nothing
            data_dir = local_dir
        else
            data_dir = joinpath(resolve_of_root(), "BCB", "Egan_et_al_2025_Rep", "processed")
        end
        input_dir  = joinpath(data_dir, "ESTIMATION_OUTPUT", "DEMAND_PREP")
        draws      = joinpath(data_dir, "ESTIMATION_OUTPUT", "BLP_DRAWS")
        output_dir = joinpath(data_dir, "ESTIMATION_OUTPUT", "BLP_RESULTS")
    end
    return input_dir, draws, output_dir
end

"""
    demand_parquet_path(estim, spec_id, args) -> Union{String,Nothing}

Locate routine `estim`'s demand-prep parquet for `spec_id`. The name comes from
`input_filename` (BLP_DEMAND_PREFIX, set by blp_rc.jl from the prefix it discovered); the
directory from `demand_search_dirs` (of_root.jl) — on the cluster the single directory the
prep step writes, `data/output/demand_prep`, off the cluster the local DEMAND_PREP.

Returns `nothing` after printing the `[!] Missing` line, naming every searched location, so
each caller keeps its own "missing input ⇒ skip this spec" control flow.

Lives here, in the CPU baseline, rather than in blp_engine_gpu.jl: `run_blp_estimation`
below needs it, and cf_demand_eval.jl loads this file WITHOUT the engine when
CF_GPU=0. The engine include()s this file, so its three entry points see it too.
"""
function demand_parquet_path(estim::Int, spec_id::Int, args)
    in_dir, _, out_dir = get_paths(args["hpc"]; local_dir=get(args, "local_dir", nothing))
    fname = input_filename(estim, spec_id)
    dirs  = demand_search_dirs(in_dir, out_dir)
    path  = resolve_in_then_out(fname, dirs...)
    path === nothing && println("  [!] Missing: $fname. Searched:\n" *
                                describe_search_dirs(dirs...))
    return path
end

# ==========================================================================
# 1. Theta2 Structure
# ==========================================================================
function build_theta2_structure(stage::String)
    if stage == "logit"
        return Int[], Tuple{Int,Int}[], 0
    elseif stage == "sigma"
        # RC1: σ_spread only
        return [1], Tuple{Int,Int}[], 1
    elseif stage == "rc2"
        # RC2: σ_spread + π(spread × gdp_per_capita)
        return [1], [(1, findfirst(==("gdp_per_capita"), D_COLS))], 2
    elseif stage == "rc3"
        # RC3: σ_spread + π(spread × gdp_per_capita) + π(spread × fraction_65plus)
        pi_inter = [(1, findfirst(==("gdp_per_capita"),  D_COLS)),
                    (1, findfirst(==("fraction_65plus"), D_COLS))]
        return [1], pi_inter, 3
    elseif stage == "rc4"
        # RC4: σ_spread + 3 demographic π interactions (no σ_log_assets yet)
        pi_inter = [(1, findfirst(==("gdp_per_capita"),     D_COLS)),
                    (1, findfirst(==("fraction_65plus"),    D_COLS)),
                    (1, findfirst(==("connections_per100"), D_COLS))]
        return [1], pi_inter, 4
    elseif stage == "full"
        # RC5 (now 4 params: σ_spread + 3 demographic π). σ_log_assets was dropped, so this stage is
        # currently IDENTICAL to rc4 — a redundant rung that reconverges to rc4 in seconds. Kept as a
        # placeholder until the ladder/table labels are renumbered (deferred). No longer the slow
        # stage: without the on-bound σ_log_assets to grind, `full` no longer walks a flat objective.
        sigma_idx = [1]   # σ_spread only — σ_log_assets dropped (2026-07-15: at the σ≥0 bound in every link routine)
        pi_inter  = [(1, findfirst(==("gdp_per_capita"),     D_COLS)),
                     (1, findfirst(==("fraction_65plus"),    D_COLS)),
                     (1, findfirst(==("connections_per100"), D_COLS))]
        return sigma_idx, pi_inter, length(sigma_idx) + length(pi_inter)
    elseif stage == "ext1"
        # RC6: full + π(log_assets × gdp_per_capita)
        sigma_idx = [1]   # σ_spread only — σ_log_assets dropped (2026-07-15: at the σ≥0 bound in every link routine)
        pi_inter  = [
            (1,                                                  findfirst(==("gdp_per_capita"),     D_COLS)),
            (1,                                                  findfirst(==("fraction_65plus"),    D_COLS)),
            (1,                                                  findfirst(==("connections_per100"), D_COLS)),
            (1 + findfirst(==("log_total_assets_lag"), X_COLS),  findfirst(==("gdp_per_capita"),     D_COLS)),
        ]
        return sigma_idx, pi_inter, length(sigma_idx) + length(pi_inter)
    elseif stage == "ext2"
        # RC7: ext1 + π(fgc_covered × fraction_65plus)
        sigma_idx = [1]   # σ_spread only — σ_log_assets dropped (2026-07-15: at the σ≥0 bound in every link routine)
        pi_inter  = [
            (1,                                                  findfirst(==("gdp_per_capita"),     D_COLS)),
            (1,                                                  findfirst(==("fraction_65plus"),    D_COLS)),
            (1,                                                  findfirst(==("connections_per100"), D_COLS)),
            (1 + findfirst(==("log_total_assets_lag"), X_COLS),  findfirst(==("gdp_per_capita"),     D_COLS)),
            (1 + findfirst(==("fgc_covered"),          X_COLS),  findfirst(==("fraction_65plus"),    D_COLS)),
        ]
        return sigma_idx, pi_inter, length(sigma_idx) + length(pi_inter)
    elseif stage == "extended"
        # RC8: ext2 + π(fgc_covered × cadunico_families_per1000)
        sigma_idx = [1]   # σ_spread only — σ_log_assets dropped (2026-07-15: at the σ≥0 bound in every link routine)
        pi_inter  = [
            (1,                                                  findfirst(==("gdp_per_capita"),              D_COLS)),
            (1,                                                  findfirst(==("fraction_65plus"),             D_COLS)),
            (1,                                                  findfirst(==("connections_per100"),          D_COLS)),
            (1 + findfirst(==("log_total_assets_lag"), X_COLS),  findfirst(==("gdp_per_capita"),             D_COLS)),
            (1 + findfirst(==("fgc_covered"),          X_COLS),  findfirst(==("fraction_65plus"),            D_COLS)),
            (1 + findfirst(==("fgc_covered"),          X_COLS),  findfirst(==("cadunico_families_per1000"),  D_COLS)),
        ]
        return sigma_idx, pi_inter, length(sigma_idx) + length(pi_inter)
    else
        error("Unknown stage: $stage")
    end
end

function unpack_theta2(theta2::Vector{Float64}, sigma_indices, pi_interactions)
    n_s = length(sigma_indices)
    n_p = length(pi_interactions)
    return theta2[1:n_s], theta2[n_s+1:n_s+n_p]
end

# ==========================================================================
# 1b. Two-start stages
# ==========================================================================
"""
Stages solved from two starting points, keeping the lower re-evaluated Q (rule in `solve_two_starts`).
Per stage: which start runs first (the start the stage used before; it wins a tie), where the warm
start's δ comes from, and the offset added to the run's `--seed` for the cold start.

  ext1      first: cold = the θ₂₀ the engine builds, `randn(MersenneTwister(seed), 5) .* 0.01`
            (seed 42 by default). Second: warm_rc4 = rc4's θ₂* padded with a zero for
            π(ln assets × gdp), with rc4's δ*. rc4's θ₂ is the first four entries of ext1's, so the
            ext1 objective there equals rc4's minimum.
  extended  first: warm_ext2 = the θ₂₀ the engine builds, ext2's θ₂* padded with a zero, from the
            stage's δ start (the logit δ̂). Second: cold = `randn(MersenneTwister(seed + 1), 7) .* 0.01`
            (seed 43 by default), clamped to the box.

The warm start's checkpoint is the stage's entry in each engine's `prev_stages`. An explicit
BLP_THETA2_INIT_FILE seed, in the engines that read one, replaces every start.

ext1 also runs the extra starts of `EXTRA_STARTS_CSV` after these two (`extra_starts`).
"""
const TWO_START_STAGES = Dict(
    "ext1"     => (first = :cold, warm_delta = :checkpoint, cold_seed_offset = 0),
    "extended" => (first = :warm, warm_delta = :stage,      cold_seed_offset = 1),
)

"""
Extra starts per stage, run after the two of `TWO_START_STAGES`: one per row of the file (a header,
then the start's name and the stage's θ₂ values). ext1's two built-in starts can both end at rc4's
solution (the added π at zero, Q equal to rc4's), a local optimum: so in the 10-16 and 10-29 runs,
while the sandboxed multi-start of 10-19 reached a lower Q from the random starts rand1-rand3, which
this file holds.
"""
const EXTRA_STARTS_CSV = Dict("ext1" => joinpath(@__DIR__, "config", "blp_ext1_extra_starts.csv"))

"""
    extra_starts(stage, n_params, lo, hi) -> (starts, source)

The extra starts of `stage`, each `(name, θ₀ clamped to [lo, hi], nothing, "the stage's δ start")`,
and a note of where they came from. A stage without a file, or a file that is missing, adds none and
the note says so. A row with the wrong number of values or a repeated name throws a `TwoStartError`.
"""
function extra_starts(stage::AbstractString, n_params::Int, lo::Vector{Float64}, hi::Vector{Float64})
    path = get(EXTRA_STARTS_CSV, stage, "")
    isempty(path) && return Any[], "none for this stage"
    isfile(path) || return Any[], "none: $path missing"
    starts = Any[]
    names  = String[]
    for (i, ln) in enumerate(eachline(path))
        s = strip(ln)
        (i == 1 || isempty(s)) && continue
        f = strip.(split(s, ","))
        name = String(f[1])
        v = parse.(Float64, f[2:end])
        length(v) == n_params || throw(TwoStartError("$(basename(path)) row '$name' has " *
            "$(length(v)) values; stage $stage has $n_params"))
        name in names && throw(TwoStartError("$(basename(path)) repeats the start '$name'"))
        push!(names, name)
        push!(starts, (name, clamp.(v, lo, hi), nothing, "the stage's δ start"))
    end
    return starts, basename(path)
end

"""Key under which a two-start stage records its starts: in the result Dict, the .json and the checkpoint."""
two_start_key(stage::AbstractString) = "$(stage)_start"

"""
BLP_TWO_START_ALLOW_SINGLE=1 lets a two-start stage run its cold start alone when the warm start's
checkpoint is missing or unusable, and lets ext2 start cold without the ext1 checkpoint
(`require_prev_checkpoint`); the log says so loudly and the record carries the reason. Unset (the
default), the stage throws a `TwoStartError`, which every stage loop rethrows: the job ends nonzero and
no later stage starts from a one-start estimate.
"""
two_start_allow_single()::Bool = get(ENV, "BLP_TWO_START_ALLOW_SINGLE", "0") == "1"

"""A two-start stage could not run both of its starts (see `two_start_allow_single`)."""
struct TwoStartError <: Exception
    msg::String
end
Base.showerror(io::IO, e::TwoStartError) = print(io, "TwoStartError: ", e.msg)

"""
    two_start_list(stage, prev_stage, theta2_0, warm_path, n_params, N_obs, lo, hi, seed)
        -> (starts, record)

The starts of a two-start stage in the order they run, each `(name, θ₀, δ₀, δ₀ description)`, δ₀ =
nothing meaning the δ the stage loaded. `theta2_0` is the θ₂₀ the engine built, the stage's first start
(cold for ext1, warm for extended), and is used unchanged. The other start is built here: the cold one
as `clamp.(randn(MersenneTwister(seed + cold_seed_offset), n_params) .* 0.01, lo, hi)`, the warm one as
the checkpoint's θ₂* padded with zeros and clamped to [lo, hi], with the checkpoint's δ* (its .bin,
else `delta_star`) when the stage takes the warm δ from there. The checkpoint at `warm_path` is read in
both cases: when it is missing or unusable this throws a `TwoStartError`, unless
`two_start_allow_single()`, in which case the cold start runs alone and `record` says why. The stage's
`extra_starts` follow, and `record["extra_starts"]` names their source.
"""
function two_start_list(stage::AbstractString, prev_stage::AbstractString,
                        theta2_0::Vector{Float64}, warm_path::AbstractString,
                        n_params::Int, N_obs::Int, lo::Vector{Float64}, hi::Vector{Float64},
                        seed::Int)
    cfg       = TWO_START_STAGES[stage]
    warm_name = "warm_$(prev_stage)"
    cold_seed = seed + cfg.cold_seed_offset
    record    = Dict{String,Any}("cold_seed" => cold_seed, "warm_checkpoint" => basename(warm_path))
    t2w = nothing; dw = nothing; why = ""
    if !isfile(warm_path)
        why = "checkpoint missing: $warm_path"
    else
        try
            ck  = deserialize(warm_path)
            t2p = get(ck, "theta2_star", nothing)
            if t2p === nothing || length(t2p) > n_params
                why = "checkpoint has no theta2_star of length <= $n_params: $warm_path"
            else
                t2w = zeros(n_params)
                t2w[1:length(t2p)] .= t2p
                t2w .= clamp.(t2w, lo, hi)
                if cfg.warm_delta == :checkpoint
                    d = load_delta_bin(replace(warm_path, r"\.jls$" => ".bin"))
                    (d === nothing || length(d) != N_obs) && (d = get(ck, "delta_star", nothing))
                    dw = (d !== nothing && length(d) == N_obs) ? collect(Float64, d) : nothing
                end
            end
        catch e
            t2w = nothing
            why = "checkpoint unreadable: $warm_path ($(sprint(showerror, e)))"
        end
    end
    cold = ("cold",
            cfg.first == :cold ? copy(theta2_0) :
                                 clamp.(randn(MersenneTwister(cold_seed), n_params) .* 0.01, lo, hi),
            nothing, "the stage's δ start")
    extra, extra_src = extra_starts(stage, n_params, lo, hi)
    record["extra_starts"] = extra_src
    if t2w === nothing
        msg = "stage $stage: the $warm_name start is unavailable ($why)"
        two_start_allow_single() || throw(TwoStartError(msg * ". Run $prev_stage first, or set " *
            "BLP_TWO_START_ALLOW_SINGLE=1 to run the cold start alone."))
        log_status("  [!!] [TWO-START] $msg. BLP_TWO_START_ALLOW_SINGLE=1: the COLD START RUNS " *
                   "without it" * (isempty(extra) ? ", ALONE." : ", with $(length(extra)) extra start(s)."))
        record[warm_name] = Dict{String,Any}("status" => "not run: $why")
        return vcat(Any[cold], extra), record
    end
    warm = cfg.first == :warm ?
        (warm_name, copy(theta2_0), nothing, "the stage's δ start") :
        (warm_name, t2w, dw, dw === nothing ?
            "the stage's δ start (no δ* of length $N_obs in the checkpoint)" :
            "δ* of $(basename(warm_path))")
    return vcat(cfg.first == :cold ? Any[cold, warm] : Any[warm, cold], extra), record
end

"""
    solve_two_starts(solve_from, starts, record, delta_work; contract, objective, reeval_note = "")
        -> (theta2_star, Q_min, n_outer, converged, record)

Runs `solve_from(name, θ₀)` from each start of `two_start_list`, in order. `solve_from` is one bounded
minimisation that returns (θ₂*, Q, n_outer, converged[, n_evals[, info]]) and leaves its last δ in
`delta_work`; n_evals defaults to n_outer (on the L-BFGS-B path n_outer already counts objective
evaluations). `info`, the Dict `solve_box_lbfgsb` returns sixth, goes into the start's record as
"outer", with its "stop_reason" and "n_iter" also at the record's top level. Each start's δ₀, or the
δ the stage loaded, is copied into `delta_work` before it runs.

After each start its Q is re-evaluated at its θ₂*: `contract(θ₂*)` sets μ and runs the inner
contraction from that start's last δ to the re-evaluation tolerance (`reeval_tol`, `reeval_cap`) and
returns whether it converged; `objective(θ₂*)` then returns the engine's own objective from that δ
without touching the optimiser's one-point caches. The choice uses this re-evaluated Q ("Q_reeval"),
not the optimiser's ("Q_opt"):
  - eligible: the starts whose re-evaluation converged with a finite Q_reeval. The lowest wins, the
    earlier start in `order` on a tie;
  - if no re-evaluation converged, the lowest finite Q_reeval wins and the record says so;
  - if no start has a finite Q_reeval, this throws a `TwoStartError`.
Q_min is the kept start's Q_reeval. `delta_work` is left holding the kept start's re-evaluated δ, so the
final contraction, the SE step and the checkpoint see the chosen solution only. `record` gains one
entry per start that ran (status, θ₀, θ₂*, Q_opt, Q_reeval, Q (= Q_reeval), reeval_converged, n_outer,
n_evals, converged, δ₀) and "reeval", "order", "tie_rule", "choice_basis", "chosen", "n_starts" and,
when the cold and the warm start both ran, "dQ_warm_minus_cold" (on Q_reeval).
"""
function solve_two_starts(solve_from::Function, starts, record::Dict{String,Any},
                          delta_work::Vector{Float64}; contract::Function, objective::Function,
                          reeval_note::AbstractString = "")
    delta_stage = copy(delta_work)
    sols = Dict{String,Any}()
    ends = Dict{String,Vector{Float64}}()
    for (name, th0, d0, d0_note) in starts
        copyto!(delta_work, d0 === nothing ? delta_stage : d0)
        log_status("  [TWO-START] start '$name': theta2_0 = $(round.(th0, digits=4)) | delta_0 = $d0_note")
        sol = solve_from(name, copy(th0))
        t, Q_opt, n_outer, conv = sol[1], sol[2], sol[3], sol[4]
        n_evals = length(sol) >= 5 ? sol[5] : n_outer
        info    = length(sol) >= 6 ? sol[6] : nothing
        reeval_conv = contract(t)
        Q = objective(t)
        ends[name] = copy(delta_work)
        sols[name] = (t, Q, n_outer, conv, reeval_conv)
        record[name] = Dict{String,Any}("status" => "ran", "theta2_start" => copy(th0),
                                        "theta2" => copy(t), "Q_opt" => Q_opt, "Q_reeval" => Q,
                                        "Q" => Q, "reeval_converged" => reeval_conv,
                                        "n_outer" => n_outer, "n_evals" => n_evals,
                                        "converged" => conv, "delta_start" => d0_note)
        if info isa AbstractDict
            record[name]["stop_reason"] = get(info, "stop_reason", nothing)
            record[name]["n_iter"]      = get(info, "n_iter", nothing)
            record[name]["outer"]       = info
        end
        log_status("  [TWO-START] start '$name' done: Q_opt = $(round(Q_opt, sigdigits=8)) | " *
                   "Q_reeval = $(round(Q, sigdigits=8)) (inner converged = $reeval_conv) | " *
                   "evals = $n_evals | converged = $conv | theta2* = $(round.(t, digits=4))" *
                   (info isa AbstractDict ? " | stop = $(get(info, "stop_reason", "?")), " *
                                            "iterations $(get(info, "n_iter", "?"))" : ""))
    end
    order = String[s[1] for s in starts]
    pool  = String[n for n in order if isfinite(sols[n][2]) && sols[n][5]]
    basis = "converged re-evaluated Q"
    if isempty(pool)
        pool  = String[n for n in order if isfinite(sols[n][2])]
        basis = "re-evaluated Q; NO re-evaluation converged"
        isempty(pool) || log_status("  [!!] [TWO-START] no re-evaluation converged: the choice uses " *
                                    "the unconverged re-evaluated Q (see the record).")
    end
    isempty(pool) && throw(TwoStartError("no start reached a finite Q (" * join(order, ", ") * ")"))
    chosen = pool[1]
    for n in pool
        sols[n][2] < sols[chosen][2] && (chosen = n)
    end
    copyto!(delta_work, ends[chosen])
    record["reeval"]       = reeval_note
    record["order"]        = order
    record["tie_rule"]     = "lowest eligible Q_reeval; on a tie the first start in `order`"
    record["choice_basis"] = basis
    record["chosen"]       = chosen
    record["n_starts"]     = length(starts)
    w = findfirst(n -> startswith(n, "warm_"), order)
    if w !== nothing && "cold" in order
        record["dQ_warm_minus_cold"] = sols[order[w]][2] - sols["cold"][2]
    end
    log_status("  [TWO-START] chosen: '$chosen' (Q_reeval = $(round(sols[chosen][2], sigdigits=8)); " *
               "basis: $basis) | $(length(order)) start(s)" *
               (haskey(record, "dQ_warm_minus_cold") ?
                    " | Q_warm - Q_cold = $(round(record["dQ_warm_minus_cold"], sigdigits=4))" : "") *
               (length(order) == 1 ? " | ONE start only, see the record" : ""))
    t, Q, n_outer, conv, _ = sols[chosen]
    return t, Q, n_outer, conv, record
end

"""(θ₂*, Q, iterations, converged, objective evaluations) of an Optim result, as `solve_two_starts` takes it."""
optim_tuple(r) = (Optim.minimizer(r), Optim.minimum(r), Optim.iterations(r), Optim.converged(r),
                  Optim.f_calls(r))

"""Copy of a nested Dict/Vector with every non-finite float as `nothing` (JSON has no NaN); the .jls keeps the floats."""
json_finite(x::AbstractFloat) = isfinite(x) ? x : nothing
json_finite(x::AbstractVector) = Any[json_finite(v) for v in x]
json_finite(x::AbstractDict) = Dict{String,Any}(string(k) => json_finite(v) for (k, v) in x)
json_finite(x) = x

"""μ at θ₂ (`compute_mu!` after `unpack_theta2`), so a closure can set it without assigning locals."""
function mu_at!(buf, prod_vec, nu_draws, theta2::Vector{Float64}, sigma_indices, pi_interactions,
                R::Int, coef_dim::Int)
    sigma_vals, pi_vals = unpack_theta2(theta2, sigma_indices, pi_interactions)
    compute_mu!(buf, prod_vec, nu_draws, sigma_vals, sigma_indices, pi_vals, R, coef_dim)
    return nothing
end

"""
The re-evaluation of each start's Q in `solve_two_starts`: the inner contraction runs from that start's
last δ to `reeval_tol(args)` = tol_inner / 10 with at most `reeval_cap(args)` = 10 × max_inner
iterations. The contraction's norm is a max-abs norm, so tol_inner / 10 (1e-11 in production) is still
about three orders of magnitude above the Float64 rounding of δ.
"""
reeval_tol(args) = args["tol_inner"] / 10
reeval_cap(args) = 10 * args["max_inner"]
reeval_rule(args) = "inner contraction from each start's last δ to tol $(reeval_tol(args)) with at " *
                    "most $(reeval_cap(args)) iterations"

# ── Inner-loop guard (engine edit (b), piece b3) ─────────────────────────────────
"""BLP_INNER_GUARD (default 0): the GPU IFT objective runs through `guarded_fg!`."""
inner_guard_on()::Bool = get(ENV, "BLP_INNER_GUARD", "0") == "1"

"""
State of `guarded_fg!` for one start: the last converged δ (`delta_good`), the last good evaluation
(θ, Q, ∇Q) that a failed one is scored against, whether the next evaluation is the start's first, and
counts: evaluations, failures, failures that were b4's clamp-bound stop, failures from a singular
block, and whether the first evaluation failed.
"""
mutable struct InnerGuard
    delta_good   ::Vector{Float64}
    theta_ref    ::Vector{Float64}
    g_ref        ::Vector{Float64}
    Q_ref        ::Float64
    have_ref     ::Bool
    first        ::Bool
    n_eval       ::Int
    n_fail       ::Int
    n_clamp_bound::Int
    n_singular   ::Int
    first_failed ::Bool
end
InnerGuard(n_params::Int, N::Int) = InnerGuard(zeros(N), zeros(n_params), zeros(n_params), NaN,
                                               false, true, 0, 0, 0, 0, false)

"""A fresh guard state for a start whose δ₀ is `delta_start` (copied)."""
function guard_reset!(gd::InnerGuard, delta_start::AbstractVector{Float64})
    copyto!(gd.delta_good, delta_start)
    gd.have_ref = false; gd.Q_ref = NaN; gd.first = true
    gd.n_eval = 0; gd.n_fail = 0; gd.n_clamp_bound = 0; gd.n_singular = 0; gd.first_failed = false
    return gd
end

"""The guard's counts, as a start record carries them."""
guard_record(gd::InnerGuard) = Dict{String,Any}("n_eval" => gd.n_eval, "n_fail" => gd.n_fail,
    "n_clamp_bound" => gd.n_clamp_bound, "n_singular" => gd.n_singular,
    "first_eval_failed" => gd.first_failed)

"""`sol`, a `solve_box_lbfgsb` return, with the guard's counts added to its info Dict as "inner_guard"
(unchanged when there is no guard)."""
function with_guard_record(sol, gd)
    (gd === nothing || length(sol) < 6 || !(sol[6] isa AbstractDict)) && return sol
    sol[6]["inner_guard"] = guard_record(gd)
    return sol
end

"""The value a failed evaluation at θ returns: the last good Q plus twice the change its gradient
predicts for the step from its point, plus 1e-8·max(1, abs(Q)); 1e10 before any good evaluation."""
function guard_fail_value(gd::InnerGuard, theta2::AbstractVector{Float64})::Float64
    gd.have_ref || return 1e10
    return gd.Q_ref + 2 * abs(dot(gd.g_ref, theta2 .- gd.theta_ref)) + 1e-8 * max(1.0, abs(gd.Q_ref))
end

"""
    guarded_fg!(G, theta2, gd, delta, contract!, objective, gradient!; cap, cap_first) -> Q

One objective-and-gradient evaluation under the inner-loop guard (BLP_INNER_GUARD=1):
  1. δ starts from `gd.delta_good`, the last converged δ, not from the previous trial's δ;
  2. `contract!(delta, max_iter) -> (converged, iterations)` sets μ at θ₂ and runs the inner loop in
     place, with `cap_first` iterations at the start's first evaluation and `cap` after it;
  3. if it converged with a finite δ: `objective(delta) -> (Q, g_moments)`, then
     `gradient!(G, g_moments, delta) -> n_singular_blocks`;
  4. the evaluation fails when the contraction did not converge (b4's clamp-bound stop included),
     δ, Q or ∇Q is not finite, or the exact Jacobian met a singular block. A failure skips what is
     left of 3, restores δ to `delta_good`, fills G with zeros and returns `guard_fail_value`.
A good evaluation becomes the reference and its δ the new `delta_good`. When every contraction
converges without a singular block, δ, Q and ∇Q are those of the unguarded evaluation bit for bit.
"""
function guarded_fg!(G::AbstractVector{Float64}, theta2::Vector{Float64}, gd::InnerGuard,
                     delta::Vector{Float64}, contract!::Function, objective::Function,
                     gradient!::Function; cap::Int, cap_first::Int)::Float64
    copyto!(delta, gd.delta_good)
    first = gd.first
    gd.first = false
    gd.n_eval += 1
    conv, it = contract!(delta, first ? cap_first : cap)
    clamp_bound = inner_clamp_test() && get(CLAMP_STATS[], "exit", "") == "clamp_bound"
    ok = conv && all(isfinite, delta)
    Q = NaN; nsing = 0; why = ok ? "" : (clamp_bound ? "clamp-bound" : "not converged in $it")
    if ok
        Q, gm = objective(delta)
        ok = isfinite(Q)
        ok || (why = "Q not finite")
        if ok
            nsing = gradient!(G, gm, delta)
            ok = nsing == 0 && all(isfinite, G)
            ok || (why = nsing > 0 ? "$nsing singular block(s)" : "gradient not finite")
        end
    end
    if ok
        copyto!(gd.delta_good, delta)
        copyto!(gd.theta_ref, theta2); copyto!(gd.g_ref, G); gd.Q_ref = Q; gd.have_ref = true
        return Q
    end
    gd.n_fail += 1
    clamp_bound && (gd.n_clamp_bound += 1)
    nsing > 0 && (gd.n_singular += 1)
    first && (gd.first_failed = true)
    copyto!(delta, gd.delta_good)
    fill!(G, 0.0)
    Qf = guard_fail_value(gd, theta2)
    log_status("  [INNER-GUARD] evaluation $(gd.n_eval) failed ($why) at theta2 = " *
               "$(round.(theta2, digits = 4)): Q = $(round(Qf, sigdigits = 8)) returned, gradient " *
               "zero, δ back to the last converged one")
    return Qf
end

"""
    require_prev_checkpoint(stage, prev_stage, prev_path) -> Dict

A stage that follows a two-start stage (ext2 after ext1) starts only from that stage's checkpoint. The
engines call this, in the optimisation block, when that checkpoint was not loaded: it throws a
`TwoStartError` unless `two_start_allow_single()`, in which case it logs loudly and returns the record
of the cold start, which the engine stores under `two_start_key(stage)`.
"""
function require_prev_checkpoint(stage::AbstractString, prev_stage::AbstractString,
                                 prev_path::AbstractString)
    msg = "stage $stage: the $prev_stage checkpoint it starts from is unavailable ($prev_path)"
    two_start_allow_single() || throw(TwoStartError(msg * ". Run $prev_stage first, or set " *
        "BLP_TWO_START_ALLOW_SINGLE=1 to start $stage cold."))
    log_status("  [!!] [TWO-START] $msg. BLP_TWO_START_ALLOW_SINGLE=1: $stage STARTS COLD.")
    return Dict{String,Any}("status" => "cold start: the $prev_stage checkpoint is unavailable",
                            "prev_checkpoint" => prev_path, "chosen" => "cold", "n_starts" => 1)
end

"""Keys of a result that hold a start record (`two_start_key`): the two-start stages, and ext2 when it starts cold under the override."""
start_record_keys(res) = [k for k in keys(res) if k isa AbstractString && endswith(k, "_start")]

# ==========================================================================
# 2. HotBuffers — pre-allocated workspace for the inner loop
# ==========================================================================
mutable struct HotBuffers
    mu             ::Matrix{Float64}
    sigma_nu       ::Matrix{Float64}
    sigma_diag     ::Vector{Float64}
    delta_B        ::Vector{Float64}
    delta_D        ::Vector{Float64}
    V_B            ::Matrix{Float64}
    V_D            ::Matrix{Float64}
    q_B            ::Matrix{Float64}
    s_B            ::Vector{Float64}
    s_D            ::Vector{Float64}
    log_sum_D_time ::Matrix{Float64}
    log_D_sum_pair ::Matrix{Float64}
    log_sum_B_mkt  ::Matrix{Float64}
    joint_max      ::Matrix{Float64}
    log_denom      ::Matrix{Float64}
    log_denom_B    ::Matrix{Float64}
    neg_log_denom  ::Matrix{Float64}
    max_neg_ld     ::Matrix{Float64}
    shifted_inv    ::Matrix{Float64}
    sum_wtd        ::Matrix{Float64}
    log_inv_wtd    ::Matrix{Float64}
    log_s_D_r      ::Matrix{Float64}
    row_max        ::Matrix{Float64}
    delta_new      ::Vector{Float64}
    f              ::Vector{Float64}
    pi_products    ::Vector{Matrix{Float64}}
    pi_base        ::Vector{Matrix{Float64}}   # saved ρ̂ spread-products → cf_shares_at restore-by-copy (B.1)
end

function allocate_hot_buffers(N::Int, N_B::Int, N_D::Int, R::Int,
                               n_pairs::Int, n_times::Int, coef_dim::Int,
                               n_pi::Int)::HotBuffers
    println("  [HPC] Pre-allocating hot buffers...")
    total_gb = (
        N*R + R*coef_dim + coef_dim +
        N_B + N_D +
        N_B*R + N_D*R + N_B*R +
        N_B + N_D +
        n_times*R + n_pairs*R*2 +
        n_pairs*R*3 +
        N_B*R +
        n_times*R*3 +
        n_pairs*R +
        N_D*R + N_D +
        N*2 +
        n_pi*N*R
    ) * 8 / 1e9
    println("  [HPC] Total buffer allocation: $(round(total_gb, digits=2)) GB")
    return HotBuffers(
        zeros(N, R),
        zeros(R, coef_dim),
        zeros(coef_dim),
        zeros(N_B),
        zeros(N_D),
        zeros(N_B, R),
        zeros(N_D, R),
        zeros(N_B, R),
        zeros(N_B),
        zeros(N_D),
        zeros(n_times, R),
        zeros(n_pairs, R),
        zeros(n_pairs, R),
        zeros(n_pairs, R),
        zeros(n_pairs, R),
        zeros(N_B, R),
        zeros(n_pairs, R),
        zeros(n_times, R),
        zeros(n_pairs, R),
        zeros(n_times, R),
        zeros(n_times, R),
        zeros(N_D, R),
        zeros(N_D, 1),
        zeros(N),
        zeros(N),
        Matrix{Float64}[],
        Matrix{Float64}[],       # pi_base (B.1) — filled by save_pi_base! after the initial precompute
    )
end

function precompute_pi_products!(buf::HotBuffers,
                                  prod_vec::Matrix{Float64},
                                  stacked_draws::Array{Float64,3},
                                  obs_key_idx::Vector{Int},
                                  pi_interactions::Vector{Tuple{Int,Int}},
                                  coef_dim::Int)
    D_dim = size(stacked_draws, 3)
    N     = size(prod_vec, 1)
    buf.pi_products = Matrix{Float64}[]
    for (cidx, didx) in pi_interactions
        if cidx <= coef_dim && didx <= D_dim
            prod = @view(prod_vec[:, cidx]) .* stacked_draws[obs_key_idx, :, didx]
            push!(buf.pi_products, prod)
            println("  [HPC] Pre-computed Pi product $(length(buf.pi_products)): $(round(sizeof(prod)/1e9, digits=2)) GB")
        else
            push!(buf.pi_products, zeros(0, 0))
        end
    end
end

function compute_mu!(buf::HotBuffers,
                     prod_vec::Matrix{Float64},
                     nu_draws::Matrix{Float64},
                     sigma_vals::Vector{Float64},
                     sigma_indices::Vector{Int},
                     pi_vals::Vector{Float64},
                     R::Int, coef_dim::Int)
    fill!(buf.sigma_diag, 0.0)
    @inbounds for (pos, cidx) in enumerate(sigma_indices)
        cidx <= coef_dim && (buf.sigma_diag[cidx] = sigma_vals[pos])
    end
    # sigma_nu[r,j] = nu_draws[r,j] * sigma_diag[j]: stride-1 in r → vectorizes with @turbo
    @turbo for j in 1:coef_dim
        sd = buf.sigma_diag[j]
        for r in 1:R
            buf.sigma_nu[r, j] = nu_draws[r, j] * sd
        end
    end
    mul!(buf.mu, prod_vec, buf.sigma_nu')
    @inbounds for (pi_idx, pi_prod) in enumerate(buf.pi_products)
        size(pi_prod, 1) == 0 && continue
        pv = pi_vals[pi_idx]
        buf.mu .+= pv .* pi_prod
    end
end

# ==========================================================================
# 3. Market Index Construction
# ==========================================================================
struct Precomp
    b_mkt_idx       ::Vector{Int}
    d_time_enc      ::Vector{Int}
    unique_pairs    ::Vector{Tuple{String,String}}
    unique_times    ::Vector{String}
    pair_time_enc   ::Vector{Int}
    pop_weights     ::Vector{Float64}
    B_agg           ::SparseMatrixCSC{Float64,Int}
    D_agg           ::SparseMatrixCSC{Float64,Int}
    PT_agg          ::SparseMatrixCSC{Float64,Int}
    sort_b          ::Vector{Int}
    b_grp_start     ::Vector{Int}
    sort_d          ::Vector{Int}
    d_uval          ::Vector{Int}
    d_grp_start     ::Vector{Int}
    sort_pt         ::Vector{Int}
    pt_uval         ::Vector{Int}
    pt_grp_start    ::Vector{Int}
    b_mask          ::BitVector
    d_mask          ::BitVector
    ln_s_data_D     ::Vector{Float64}
    ln_s_data_B_cond::Vector{Float64}
    Z_moments       ::Matrix{Float64}
    X_full          ::Matrix{Float64}
    X_hat           ::Matrix{Float64}
    theta1_valid    ::BitVector
    clusters        ::Vector{String}
end

function _unique_with_starts(sorted_vec::Vector{Int})
    isempty(sorted_vec) && return Int[], Int[]
    uval   = Int[sorted_vec[1]]
    gstart = Int[1]
    @inbounds for i in 2:length(sorted_vec)
        if sorted_vec[i] != sorted_vec[i-1]
            push!(uval, sorted_vec[i])
            push!(gstart, i)
        end
    end
    return uval, gstart
end

function build_precomp(df::DataFrame, Z::Matrix{Float64},
                       X_full::Matrix{Float64}, X_hat::Matrix{Float64},
                       valid::BitVector, clusters::Vector{String})::Precomp
    N        = nrow(df)
    b_mask   = BitVector(Bool.(coalesce.(df.is_B, false)))
    d_mask   = .!b_mask
    N_B      = sum(b_mask)
    N_D      = sum(d_mask)

    mca_codes = string.(df.mca_code)
    time_ids  = string.(df.time_id)
    pop_total = coalesce.(df.pop_total, 0.0)

    raw_pairs     = collect(zip(mca_codes[b_mask], time_ids[b_mask]))
    unique_pairs  = sort(unique(raw_pairs))
    pair_to_idx   = Dict(p => i for (i, p) in enumerate(unique_pairs))
    b_mkt_idx     = [pair_to_idx[(m, t)] for (m, t) in raw_pairs]

    unique_times  = sort(unique(time_ids))
    time_to_idx   = Dict(t => i for (i, t) in enumerate(unique_times))
    d_time_enc    = [time_to_idx[t] for t in time_ids[d_mask]]

    n_pairs       = length(unique_pairs)
    n_times       = length(unique_times)
    pair_time_enc = [time_to_idx[p[2]] for p in unique_pairs]

    # Weight for D-type share aggregation: bc_mt * pop_mt (eq:a3-D)
    # bc_mt * pop_mt is constant within each (mca, time) pair; use per-pair mean
    # to avoid accumulating n_firms_mt * bc_mt * pop_mt.
    bc_vals      = Float64.(coalesce.(df.banked_correction, 1.1))
    bc_pop_b     = (bc_vals .* pop_total)[b_mask]
    pair_bc_pop  = zeros(n_pairs)
    pair_n_firms = zeros(Int, n_pairs)
    for (i, w) in zip(b_mkt_idx, bc_pop_b)
        pair_bc_pop[i]  += w
        pair_n_firms[i] += 1
    end
    pair_bc_pop  ./= max.(pair_n_firms, 1)   # bc_mt * pop_mt per pair
    time_bc_pop  = zeros(n_times)
    for (i, w) in zip(pair_time_enc, pair_bc_pop); time_bc_pop[i] += w; end
    pop_weights  = pair_bc_pop ./ max.(time_bc_pop[pair_time_enc], 1e-30)

    B_agg  = sparse(b_mkt_idx,     1:N_B,     ones(N_B),     n_pairs, N_B)
    D_agg  = sparse(d_time_enc,    1:N_D,     ones(N_D),     n_times, N_D)
    PT_agg = sparse(pair_time_enc, 1:n_pairs, pop_weights,   n_times, n_pairs)

    sort_b            = sortperm(b_mkt_idx)
    _, b_grp_s        = _unique_with_starts(b_mkt_idx[sort_b])
    sort_d            = sortperm(d_time_enc)
    d_uval, d_grp_s   = _unique_with_starts(d_time_enc[sort_d])
    sort_pt           = sortperm(pair_time_enc)
    pt_uval, pt_grp_s = _unique_with_starts(pair_time_enc[sort_pt])

    ln_s_D = log.(clamp.(coalesce.(df.share_D,      0.0), 1e-15, Inf))
    ln_s_B = log.(clamp.(coalesce.(df.share_B_cond, 0.0), 1e-15, Inf))

    return Precomp(b_mkt_idx, d_time_enc, unique_pairs, unique_times,
                   pair_time_enc, pop_weights,
                   B_agg, D_agg, PT_agg,
                   sort_b, b_grp_s,
                   sort_d, d_uval, d_grp_s,
                   sort_pt, pt_uval, pt_grp_s,
                   b_mask, d_mask, ln_s_D, ln_s_B,
                   Z, X_full, X_hat, valid, clusters)
end

# ==========================================================================
# 4. Model Shares — in-place
# ==========================================================================
function logsumexp_groups!(result::Matrix{Float64},
                            V, sort_idx, grp_start, uval, R)
    fill!(result, -Inf)
    n_uniq = length(grp_start)
    # Each group writes to a distinct row of result → no data race.
    # Threads.@threads partitions groups across all Julia threads (32 on Bouchet).
    Threads.@threads for g in 1:n_uniq
        gidx       = grp_start[g]
        gend       = g < n_uniq ? grp_start[g+1] - 1 : length(sort_idx)
        target_row = uval[g]
        @inbounds for c in 1:R
            mx = -Inf
            for ii in gidx:gend
                v = V[sort_idx[ii], c]; v > mx && (mx = v)
            end
            s = 0.0
            for ii in gidx:gend
                s += exp(V[sort_idx[ii], c] - mx)
            end
            result[target_row, c] = mx + log(max(s, 1e-300))
        end
    end
end

function compute_model_shares!(buf::HotBuffers, delta::Vector{Float64},
                                pc::Precomp, R::Int)
    N_B     = length(buf.delta_B)
    N_D     = length(buf.delta_D)
    n_pairs = length(pc.unique_pairs)
    n_times = length(pc.unique_times)

    b_idx = 0; d_idx = 0
    @inbounds for i in eachindex(pc.b_mask)
        if pc.b_mask[i]
            b_idx += 1
            buf.delta_B[b_idx] = delta[i]
            for r in 1:R
                buf.V_B[b_idx, r] = clamp(delta[i] + buf.mu[i, r], -500.0, 500.0)
            end
        else
            d_idx += 1
            buf.delta_D[d_idx] = delta[i]
            for r in 1:R
                buf.V_D[d_idx, r] = clamp(delta[i] + buf.mu[i, r], -500.0, 500.0)
            end
        end
    end

    logsumexp_groups!(buf.log_sum_D_time, buf.V_D, pc.sort_d, pc.d_grp_start, pc.d_uval, R)
    @inbounds for i in 1:n_pairs
        for r in 1:R
            buf.log_D_sum_pair[i, r] = buf.log_sum_D_time[pc.pair_time_enc[i], r]
        end
    end

    logsumexp_groups!(buf.log_sum_B_mkt, buf.V_B, pc.sort_b, pc.b_grp_start, collect(1:n_pairs), R)

    log_outside = 0.0
    @inbounds for i in 1:n_pairs
        for r in 1:R
            jm = max(buf.log_sum_B_mkt[i, r], buf.log_D_sum_pair[i, r], log_outside)
            buf.joint_max[i, r] = jm
            buf.log_denom[i, r] = jm + log(max(
                exp(log_outside - jm) +
                exp(buf.log_sum_B_mkt[i, r] - jm) +
                exp(buf.log_D_sum_pair[i, r] - jm), 1e-300))
        end
    end

    # Each i writes only to buf.q_B[i,:] and buf.s_B[i] → no data race.
    Threads.@threads for i in 1:N_B
        mkt = pc.b_mkt_idx[i]
        @inbounds for r in 1:R
            buf.q_B[i, r] = exp(buf.V_B[i, r] - buf.log_denom[mkt, r])
        end
        s = 0.0
        @inbounds for r in 1:R; s += buf.q_B[i, r]; end
        buf.s_B[i] = s / R
    end

    @inbounds for i in 1:n_pairs
        for r in 1:R
            buf.neg_log_denom[i, r] = -buf.log_denom[i, r]
        end
    end

    fill!(buf.max_neg_ld, -Inf)
    @inbounds for g in 1:length(pc.pt_grp_start)
        gidx   = pc.pt_grp_start[g]
        gend   = g < length(pc.pt_grp_start) ? pc.pt_grp_start[g+1] - 1 : length(pc.sort_pt)
        target = pc.pt_uval[g]
        for r in 1:R
            mx = -Inf
            for ii in gidx:gend
                v = buf.neg_log_denom[pc.sort_pt[ii], r]; v > mx && (mx = v)
            end
            buf.max_neg_ld[target, r] = mx
        end
    end

    @inbounds for i in 1:n_pairs
        for r in 1:R
            buf.shifted_inv[i, r] = exp(buf.neg_log_denom[i, r] -
                                        buf.max_neg_ld[pc.pair_time_enc[i], r])
        end
    end

    mul!(buf.sum_wtd, pc.PT_agg, buf.shifted_inv)

    @inbounds for i in 1:n_times
        for r in 1:R
            buf.log_inv_wtd[i, r] = buf.max_neg_ld[i, r] +
                                    log(max(buf.sum_wtd[i, r], 1e-300))
        end
    end

    # Each i writes only to buf.log_s_D_r[i,:], buf.row_max[i,1], buf.s_D[i] → no data race.
    Threads.@threads for i in 1:N_D
        tenc = pc.d_time_enc[i]
        rm = -Inf
        @inbounds for r in 1:R
            v = buf.V_D[i, r] + buf.log_inv_wtd[tenc, r]
            buf.log_s_D_r[i, r] = v
            v > rm && (rm = v)
        end
        buf.row_max[i, 1] = rm
        s = 0.0
        @inbounds for r in 1:R; s += exp(buf.log_s_D_r[i, r] - rm); end
        buf.s_D[i] = (s / R) * exp(rm)
    end
end

# ==========================================================================
# 5. SQUAREM BLP Contraction — in-place
# ==========================================================================
# ── Clamp-aware convergence test (engine edit (b), piece b4) ─────────────────────
"""BLP_INNER_CLAMP_TEST (default 0): the contractions also test the coordinates held at the ±500 clamp."""
inner_clamp_test()::Bool = get(ENV, "BLP_INNER_CLAMP_TEST", "0") == "1"

"""What the last contraction met at the ±500 clamp, set only under BLP_INNER_CLAMP_TEST=1: "exit"
(converged | clamp_bound | cap | nonfinite), "n_delta_at_clamp" (δ coordinates at ±500 on exit),
"n_bound" (those whose pre-clamp residual is ≥ tol; -1 when not computed), "max_resid_at_clamp",
"n_V_at_clamp" (utility entries V = δ + μ at ±500 in the last share computation) and "iterations"."""
const CLAMP_STATS = Ref{Dict{String,Any}}(Dict{String,Any}())

"""
    clamp_residual_stats(dest, s_model, ln_s_data, floor_s, tol) -> (n_at, n_bound, max_resid)

Over the coordinates of `dest` = T(src) that the ±500 clamp holds (abs(dest_i) ≥ 500): how many there
are, how many have a pre-clamp residual abs(ln s_data_i − ln max(s_model_i, floor_s)) ≥ tol (s_model
the shares at src), and the largest such residual. The same code runs on CPU and GPU vectors.
"""
function clamp_residual_stats(dest::AbstractVector, s_model::AbstractVector, ln_s_data::AbstractVector,
                              floor_s, tol::Float64)
    n_at = mapreduce(x -> abs(x) >= 500 ? 1 : 0, +, dest; init = 0)
    n_at == 0 && return (0, 0, 0.0)
    r = ifelse.(abs.(dest) .>= 500, abs.(ln_s_data .- log.(max.(s_model, floor_s))), zero(eltype(dest)))
    return (n_at, mapreduce(x -> x >= tol ? 1 : 0, +, r; init = 0), Float64(maximum(r)))
end

"""Count of entries at ±500 in a CPU or GPU array (the share kernels' clamp of V = δ + μ)."""
n_at_clamp(A::AbstractArray) = mapreduce(x -> abs(x) >= 500 ? 1 : 0, +, A; init = 0)

"""Store one contraction exit in CLAMP_STATS; print a [CLAMP] line when anything sits at ±500.
Returns true when the exit is clamp-bound (a converged norm with a clamped residual ≥ tol)."""
function record_clamp_exit!(exit_kind::AbstractString, n_at::Int, n_bound::Int, max_r::Float64,
                            n_V::Int, iterations::Int, tol::Float64)::Bool
    kind = (exit_kind == "converged" && n_bound > 0) ? "clamp_bound" : String(exit_kind)
    CLAMP_STATS[] = Dict{String,Any}("exit" => kind, "n_delta_at_clamp" => n_at, "n_bound" => n_bound,
                                     "max_resid_at_clamp" => max_r, "n_V_at_clamp" => n_V,
                                     "iterations" => iterations, "tol" => tol)
    (n_at > 0 || n_V > 0) &&
        println("    [CLAMP] exit $kind after $iterations evals: δ at ±500 $n_at" *
                (n_bound >= 0 ? " ($n_bound with pre-clamp residual ≥ $tol, max $(round(max_r, sigdigits = 4)))" : "") *
                " | V at ±500 $n_V")
    return kind == "clamp_bound"
end

function blp_contraction!(buf::HotBuffers, delta::Vector{Float64},
                           pc::Precomp, R::Int;
                           tol::Float64=1e-12, max_iter::Int=5000)
    N            = length(delta)
    norm_history = Float64[]
    x1     = Vector{Float64}(undef, N)
    x2     = Vector{Float64}(undef, N)
    r_vec  = Vector{Float64}(undef, N)
    v_vec  = Vector{Float64}(undef, N)
    x_prop = Vector{Float64}(undef, N)
    # b4 (BLP_INNER_CLAMP_TEST=1): at an exit, the coordinates the clamp holds are checked on their
    # pre-clamp residual, from the shares of the last T step (buf.s_B / buf.s_D, at that step's src).
    ctest = inner_clamp_test()
    ln_s_full = ctest ? [pc.b_mask[i] ? pc.ln_s_data_B_cond[i] : pc.ln_s_data_D[i] for i in 1:N] : Float64[]
    function clamp_exit!(dest::Vector{Float64}, kind::String, fevals::Int)::Bool
        n_V = n_at_clamp(buf.V_B) + n_at_clamp(buf.V_D)
        if kind == "converged"
            s_full = Vector{Float64}(undef, N); b_idx = 0; d_idx = 0
            @inbounds for i in 1:N
                if pc.b_mask[i]; b_idx += 1; s_full[i] = buf.s_B[b_idx]
                else;            d_idx += 1; s_full[i] = buf.s_D[d_idx]; end
            end
            n_at, n_bound, max_r = clamp_residual_stats(dest, s_full, ln_s_full, 1e-15, tol)
            return record_clamp_exit!(kind, n_at, n_bound, max_r, n_V, fevals, tol)
        end
        return record_clamp_exit!(kind, n_at_clamp(dest), -1, NaN, n_V, fevals, tol)
    end

    function T_inplace!(dest::Vector{Float64}, src::Vector{Float64})
        compute_model_shares!(buf, src, pc, R)
        b_idx = 0; d_idx = 0
        @inbounds for i in 1:N
            if pc.b_mask[i]
                b_idx += 1
                dest[i] = clamp(src[i] + pc.ln_s_data_B_cond[i] -
                                log(clamp(buf.s_B[b_idx], 1e-15, Inf)), -500.0, 500.0)
            else
                d_idx += 1
                dest[i] = clamp(src[i] + pc.ln_s_data_D[i] -
                                log(clamp(buf.s_D[d_idx], 1e-15, Inf)), -500.0, 500.0)
            end
        end
    end

    fevals = 0
    while fevals + 2 <= max_iter
        T_inplace!(x1, delta); fevals += 1
        if !all(isfinite, x1)
            println("    [SQUAREM ABORT] non-finite at eval=$fevals")
            ctest && clamp_exit!(delta, "nonfinite", fevals)
            return false, fevals, norm_history
        end
        nm = 0.0
        @inbounds for i in 1:N
            r_vec[i] = x1[i] - delta[i]
            a = abs(r_vec[i]); a > nm && (nm = a)
        end
        push!(norm_history, nm)
        (fevals % 50 == 0 || nm < tol) &&
            println("    [SQUAREM eval=$fevals/$max_iter] norm=$(round(nm, sigdigits=4))")
        if nm < tol
            if ctest && clamp_exit!(x1, "converged", fevals)
                copyto!(delta, x1); return false, fevals, norm_history
            end
            copyto!(delta, x1); return true, fevals, norm_history
        end

        T_inplace!(x2, x1); fevals += 1
        if !all(isfinite, x2)
            ctest && clamp_exit!(x1, "nonfinite", fevals)
            copyto!(delta, x1); return false, fevals, norm_history
        end
        nm2 = 0.0
        @inbounds for i in 1:N; a = abs(x2[i]-x1[i]); a > nm2 && (nm2 = a); end
        push!(norm_history, nm2)
        (fevals % 50 == 0 || nm2 < tol) &&
            println("    [SQUAREM eval=$fevals/$max_iter] norm=$(round(nm2, sigdigits=4))")
        if nm2 < tol
            if ctest && clamp_exit!(x2, "converged", fevals)
                copyto!(delta, x2); return false, fevals, norm_history
            end
            copyto!(delta, x2); return true, fevals, norm_history
        end

        norm_r_sq = 0.0; norm_v_sq = 0.0
        @inbounds for i in 1:N
            v_vec[i]   = (x2[i] - x1[i]) - r_vec[i]
            norm_r_sq += r_vec[i]^2
            norm_v_sq += v_vec[i]^2
        end
        if norm_v_sq < 1e-28
            copyto!(delta, x2); continue
        end
        α = -sqrt(norm_r_sq / norm_v_sq)
        @inbounds for i in 1:N
            x_prop[i] = clamp(delta[i] - 2α * r_vec[i] + α^2 * v_vec[i], -500.0, 500.0)
        end
        if !all(isfinite, x_prop)
            copyto!(delta, x2)
        else
            norm_prop = 0.0
            @inbounds for i in 1:N; a = abs(x_prop[i]-x2[i]); a > norm_prop && (norm_prop = a); end
            if norm_prop > 100.0 * nm2 + 1.0
                copyto!(delta, x2)
            else
                copyto!(delta, x_prop)
            end
        end
    end
    ctest && clamp_exit!(delta, "cap", fevals)
    return false, fevals, norm_history
end

# ==========================================================================
# 6. Linear IV (Eq-A5)
# ==========================================================================
"""
    build_regressor_matrices(df) -> (spread_cols, x_mat, Z_mat, iv_cols)

The price (spread in pp), the linear design `x_mat` (N × L_X1: the X_COLS columns, then the D-type
dummy) and every instrument of `iv_cols` (N × L, before the zero-variance filter), read under the
input rule of demand_inputs.jl: a missing or non-finite value in any of them stops the run with a
`DemandInputError` that counts it per column and per row, unless BLP_ZERO_FILL_MISSING=1 (then 0.0).

A column ABSENT from the parquet is an error in every case. It used to vanish silently: X_COLS fell
back to a column of zeros and a missing instrument was dropped from Z, which is how the whole
IV_BLP_LOO block once disappeared after a market_panel rebuild skipped panel_loo_instruments.py,
leaving the BLP identified off the cost shifters alone and α̂ positive (plan §0B).
"""
function build_regressor_matrices(df::DataFrame)
    iv_cols = vcat(IV_BLP_LOO, IV_ESTBAN, IV_COST, IV_CAPITAL)
    s_blk = read_input_block(df, ["spread_ann"], "price")
    x_blk = read_input_block(df, X_COLS, "X")
    z_blk = read_input_block(df, iv_cols, "instrument")
    enforce_inputs!([s_blk, x_blk, z_blk]; context="demand parquet")
    check_shares_and_types(df)                                 # shares and deposit type (same rule)
    spread_cols = s_blk[1][:, 1] ./ 100.0                      # bps → percentage points (÷100)
    x_mat = hcat(x_blk[1], dtype_column(df))                   # X1_COLS order: X_COLS, then D
    size(x_mat, 2) == L_X1 || error("linear design has $(size(x_mat, 2)) columns, X1_COLS $(L_X1)")
    return spread_cols, x_mat, z_blk[1], iv_cols
end

"""
    rc_design(spread_cols, x_mat) -> prod_vec

The random-coefficient design, N × coef_dim: the spread, then the X_COLS columns of the linear
design. The D-type dummy (the last column of `x_mat`) is not part of it.
"""
function rc_design(spread_cols::Vector{Float64}, x_mat::Matrix{Float64})::Matrix{Float64}
    X1_COLS[1:L_PROD] == X_COLS || error("X1_COLS must start with X_COLS")
    size(x_mat, 2) == L_X1 || error("rc_design: x_mat has $(size(x_mat, 2)) columns, expected $(L_X1)")
    return hcat(spread_cols, x_mat[:, 1:L_PROD])
end

"""
    moment_instruments(Z_mat, iv_cols) -> (Z, iv_avail)

The instruments that enter the GMM moments: the columns of `Z_mat` with non-zero variance, named in
`iv_avail`. A constant column carries no identifying information and would make Z'Z singular, so it
is dropped, with a warning naming it.
"""
function moment_instruments(Z_mat::Matrix{Float64}, iv_cols::Vector{String})
    keep = [std(view(Z_mat, :, j)) > 1e-10 for j in 1:size(Z_mat, 2)]
    all(keep) || @warn "Dropping zero-variance instrument(s): " * join(iv_cols[.!keep], ", ")
    return Z_mat[:, keep], iv_cols[keep]
end

function project_endogenous_spreads(spread_cols::Vector{Float64},
                                     H::Matrix{Float64},
                                     deposit_types::Vector{Int})::Matrix{Float64}
    spread_hat = reshape(copy(spread_cols), :, 1)
    for k_val in [4, 5]
        k_mask   = deposit_types .== k_val
        any(k_mask) || continue
        H_k      = H[k_mask, :]
        spread_k = spread_cols[k_mask]
        valid    = all.(isfinite, eachrow(H_k)) .& isfinite.(spread_k)
        sum(valid) > size(H_k, 2) || continue
        try
            beta_fs           = H_k[valid, :] \ spread_k[valid]
            k_hat             = copy(spread_k)
            k_hat[valid]     .= H_k[valid, :] * beta_fs
            spread_hat[k_mask, 1] .= k_hat
        catch; end
    end
    return spread_hat
end

function estimate_theta1(delta::Vector{Float64}, pc::Precomp)
    valid  = pc.theta1_valid .& isfinite.(delta)
    X_v    = pc.X_hat[valid, :]
    d_v    = delta[valid]
    theta1 = try X_v \ d_v catch; zeros(size(pc.X_hat, 2)) end
    xi     = delta .- pc.X_full * theta1
    return theta1, xi
end

"""
    estimate_theta1_gmm(delta, pc, W) -> (theta1, xi)

W-weighted linear-GMM θ₁ for the CUE engine: θ₁(W) = (X'Z W Z'X)⁻¹ X'Z W Z'δ with X = `pc.X_full`
and Z = `pc.Z_moments`, solved on `pc.theta1_valid ∧ isfinite.(δ)` rows with ξ on ALL rows —
mirroring `estimate_theta1`'s masking asymmetry. NOT identical to `estimate_theta1` even at
W = (Z'Z/N)⁻¹: production 2SLS projects only the type-4/5 spread on H = [X, Z] and keeps x_mat as
its own instrument, while this instruments all X columns with the excluded Z. Iteration 0 of the CUE
fixed point therefore calls `estimate_theta1` itself; this function is used from iteration 1 on
(CUE/GPU-numerical path only — no production caller).
"""
function estimate_theta1_gmm(delta::Vector{Float64}, pc::Precomp, W::Matrix{Float64})
    valid = pc.theta1_valid .& isfinite.(delta)
    Zv  = pc.Z_moments[valid, :]
    Xv  = pc.X_full[valid, :]
    ZtX = Zv' * Xv                       # L×K
    Ztd = Zv' * delta[valid]             # L
    A   = ZtX' * W * ZtX
    b   = ZtX' * W * Ztd
    theta1 = try A \ b catch; pinv(A) * b end
    xi = delta .- pc.X_full * theta1     # ALL rows (mirrors estimate_theta1)
    return theta1, xi
end

"""
    check_design_rank(pc; colnames, abort=true) -> Int

Up-front guard against a rank-deficient projected design `X_hat` over the rows used
for the θ₁ solve (`pc.theta1_valid`). A degenerate/collinear column makes θ₁
unidentified, and the two estimators hide it differently — a silent false positive:

* `estimate_theta1` uses `X \\ d` (rank-revealing) + `inv(XtX) catch pinv`, so blp_1
  *absorbs* a zero/constant column and still reports `converged: true`.
* the IFT gradient uses an unpivoted `qr(X_hat_v) \\ …`, so blp_2 throws
  `SingularException(j)` at the deficient column j.

This surfaces the offending column **by name** before any estimation runs. Returns the
numerical rank; if `abort` and rank < ncols it errors (default), else warns.
`colnames` must align with the columns of `pc.X_hat` (i.e. `["spread_hat"; X1_COLS...]`).
"""
function check_design_rank(pc::Precomp;
                           colnames::Vector{String}=vcat(["spread_hat"], X1_COLS),
                           abort::Bool=true)::Int
    valid = pc.theta1_valid
    Xv    = pc.X_hat[valid, :]
    n, k  = size(Xv)
    name(j) = 1 <= j <= length(colnames) ? colnames[j] : "col$j"

    # Per-column degeneracy (named for a human-readable message)
    zerocols  = String[]; constcols = String[]
    for j in 1:k
        col = @view Xv[:, j]
        if all(iszero, col)
            push!(zerocols, name(j))
        else
            mn, mx = extrema(col)
            mx == mn && push!(constcols, name(j))
        end
    end

    # Rank-revealing pivoted QR; rank = pivots above a relative tolerance.
    F      = qr(Xv, ColumnNorm())
    rdiag  = abs.(diag(F.R))
    tol    = (isempty(rdiag) ? 0.0 : maximum(rdiag)) * eps(Float64) * max(n, k) * 10
    rnk    = count(>(tol), rdiag)
    deficient = rnk < k ? String[name(F.p[i]) for i in (rnk+1):k] : String[]

    log_status("  [DESIGN CHECK] X_hat: $(n)×$(k) | rank=$(rnk)" *
               (isempty(zerocols)  ? "" : " | all-zero: $(zerocols)") *
               (isempty(constcols) ? "" : " | constant: $(constcols)"))
    if rnk < k
        msg = "[DESIGN CHECK FAILED] X_hat is rank-deficient (rank $(rnk) < $(k) cols). " *
              "Degenerate column(s): $(unique(vcat(zerocols, deficient)))" *
              (isempty(constcols) ? "" : "; also constant (unidentified): $(constcols)") *
              ". θ₁ is not identified — blp_1 would solve it silently (pivoted QR/pinv), " *
              "blp_2 throws SingularException in the IFT QR. Fix the input data " *
              "(missing/constant regressor) before trusting any estimates."
        abort ? error(msg) : @warn msg
    else
        log_status("  [DESIGN CHECK] ✓ X_hat full rank ($(rnk)=$(k))" *
                   (isempty(constcols) ? "" : " — but constant col(s) $(constcols) are unidentified"))
    end
    return rnk
end

"""Cluster-robust sandwich SEs for theta1 (IV/2SLS) + IK2016 effective cluster count.
Returns (se, G_nominal, G_star) where G_star = G/(1+CV²) per Imbens & Kolesár (2016)."""
function compute_cluster_se(theta1::Vector{Float64}, delta::Vector{Float64}, pc::Precomp)
    valid    = pc.theta1_valid .& isfinite.(delta)
    N_v      = sum(valid)
    K        = length(theta1)
    X_hat_v  = pc.X_hat[valid, :]    # X̃ (first-stage projected regressors)
    X_full_v = pc.X_full[valid, :]   # X  (original regressors)
    xi_v     = delta[valid] .- X_full_v * theta1
    cl_v     = pc.clusters[valid]
    # Bread: (X̃'X)⁻¹
    XtX     = X_hat_v' * X_full_v
    XtX_inv = try inv(XtX) catch; pinv(XtX) end
    # Meat: Σ_c score_c ⊗ score_c  where score_c = X̃_c' ξ_c
    unique_cl = unique(cl_v)
    G         = length(unique_cl)
    meat      = zeros(K, K)
    for c in unique_cl
        mask    = cl_v .== c
        score_c = X_hat_v[mask, :]' * xi_v[mask]
        meat   .+= score_c * score_c'
    end
    # CR1 small-sample correction
    correction = G > 1 ? (G / (G - 1)) * (N_v / (N_v - K)) : 1.0
    V_cluster  = XtX_inv * meat * XtX_inv' .* correction
    se         = sqrt.(max.(diag(V_cluster), 0.0))
    # IK2016 effective clusters: G* = G / (1 + CV²)
    cl_sizes = [count(==(c), cl_v) for c in unique_cl]
    mu_G     = Statistics.mean(cl_sizes)
    cv_G     = mu_G > 0 ? Statistics.std(cl_sizes; corrected=false) / mu_G : 0.0
    G_star   = max(1.0, G / (1.0 + cv_G^2))
    return se, G, G_star
end

# ==========================================================================
# 7. GMM Objective
# ==========================================================================
function compute_gmm_moments(xi::Vector{Float64}, pc::Precomp)
    return vec(Statistics.mean(xi .* pc.Z_moments; dims=1))
end

function gmm_objective!(buf::HotBuffers, theta2::Vector{Float64},
                         prod_vec, nu_draws,
                         sigma_indices, pi_interactions,
                         R::Int, coef_dim::Int,
                         W::Matrix{Float64},
                         tol_inner::Float64, max_inner::Int,
                         delta::Vector{Float64},
                         pc::Precomp)::Float64
    sigma_vals, pi_vals = unpack_theta2(theta2, sigma_indices, pi_interactions)
    compute_mu!(buf, prod_vec, nu_draws, sigma_vals, sigma_indices, pi_vals, R, coef_dim)
    converged, n_iter, _ = blp_contraction!(buf, delta, pc, R;
                                             tol=tol_inner, max_iter=max_inner)
    converged || println("  [!] Inner loop did not converge in $n_iter iterations")
    theta1, xi = estimate_theta1(delta, pc)
    G          = compute_gmm_moments(xi, pc)
    return dot(G, W * G)
end

# ==========================================================================
# 7b. CUE (continuously-updated GMM) — the LIML analogue. GPU-numerical engine only.
# ==========================================================================
# The production estimator is one-step GMM: W = (Z'Z/N)⁻¹ FIXED, θ₁ concentrated out by 2SLS.
# CUE replaces the weight with W(θ) = pinv(Ω̂(θ)), Ω̂ clustered on conglomerate, recomputed at EVERY
# objective evaluation, and solves the concentrated θ₁ by W-weighted linear GMM iterated to a fixed
# point — so α inherits LIML's weak-identification behaviour (no bias toward OLS). Selected via
# BLP_ENGINE=cue (suffix "_cue" on every artifact; see blp_rc.jl). It rides the NUMERICAL engine:
# its gradient is pure forward finite differences of the objective, so changing only the objective
# keeps objective/gradient consistent automatically. The IFT analytic gradient assumes ∂W/∂θ₂ = 0 and
# is NOT valid for CUE.
#
# Q_cue is returned in Hansen-J units, N·ḡ'Wḡ — NOT comparable to the fixed-W Q = ḡ'(Z'Z/N)⁻¹ḡ.

"""Per-run CUE state: integer cluster codes precomputed ONCE (a naive `clusters .== c` scan per
evaluation would be O(G·N) ≈ 350M comparisons), env-tunable knobs, and eval/non-convergence counters."""
struct CueOpts
    codes::Vector{Int}                 # cluster code 1..G per row
    G::Int
    max_witer::Int                     # BLP_CUE_MAX_WITER  (default 10)
    wtol::Float64                      # BLP_CUE_WTOL       (default 1e-10; FD-gradient stability)
    pinv_rtol::Float64                 # BLP_CUE_PINV_RTOL  (default 1e-10)
    ridge::Float64                     # BLP_CUE_RIDGE      (default 0 = off; ·tr(Ω̂)/L·I stabiliser)
    n_evals::Base.RefValue{Int}
    n_nonconv::Base.RefValue{Int}
    itmax_seen::Base.RefValue{Int}
end

function CueOpts(clusters::Vector{String})
    uc  = unique(clusters)
    idx = Dict(c => i for (i, c) in enumerate(uc))
    return CueOpts([idx[c] for c in clusters], length(uc),
                   parse(Int,     get(ENV, "BLP_CUE_MAX_WITER", "10")),
                   parse(Float64, get(ENV, "BLP_CUE_WTOL",      "1e-10")),
                   parse(Float64, get(ENV, "BLP_CUE_PINV_RTOL", "1e-10")),
                   parse(Float64, get(ENV, "BLP_CUE_RIDGE",     "0.0")),
                   Ref(0), Ref(0), Ref(0))
end

"""Clustered moment covariance Ω̂ = (1/N)Σ_g (Z_g'ξ_g)(Z_g'ξ_g)' — the blp_se_common.jl:128 meat pattern,
code-indexed. ALL rows, unmasked (mirrors `compute_gmm_moments`)."""
function cue_weight(xi::Vector{Float64}, Z::Matrix{Float64}, opts::CueOpts)
    N, L = size(Z)
    MclT = zeros(L, opts.G)                       # L×G so the inner loop is stride-1 in Z's rows
    @inbounds for i in 1:N
        g = opts.codes[i]; x = xi[i]
        for l in 1:L
            MclT[l, g] += x * Z[i, l]
        end
    end
    Omega = (MclT * MclT') ./ N
    if opts.ridge > 0
        Omega .+= (opts.ridge * tr(Omega) / L) .* Matrix{Float64}(I, L, L)
    end
    return Omega
end

cue_pinv(Omega::Matrix{Float64}, rtol::Float64) =
    Matrix(pinv(Symmetric((Omega .+ Omega') ./ 2); rtol=rtol))

"""
    cue_fixed_point(delta, pc, opts) -> (theta1, xi, W_final, Q, iters, converged)

The CUE inner fixed point at a given δ(θ₂). Iteration 0 is the VERBATIM production 2SLS
(`estimate_theta1`) — the self-check anchor; iterations 1.. alternate W = pinv(Ω̂(ξ)) and
θ₁ = W-weighted linear GMM (`estimate_theta1_gmm`) until relΔθ₁ < wtol or max_witer. A non-finite
update keeps the last finite iterate. Q = N·ḡ'W_final ḡ with W_final re-formed from the FINAL ξ
(the CU criterion / Hansen J at the final iterate). All linear algebra is O(N·L) + a 16×16 pinv —
negligible beside the GPU contraction that precedes it.
"""
function cue_fixed_point(delta::Vector{Float64}, pc::Precomp, opts::CueOpts)
    theta1, xi = estimate_theta1(delta, pc)       # iteration 0 ≡ production 2SLS
    L = size(pc.Z_moments, 2)
    all(isfinite, xi) || return theta1, xi, Matrix{Float64}(I, L, L), 1e12, 0, false
    iters = 0; converged = false
    for it in 1:opts.max_witer
        Wn = cue_pinv(cue_weight(xi, pc.Z_moments, opts), opts.pinv_rtol)
        t1n, xin = estimate_theta1_gmm(delta, pc, Wn)
        all(isfinite, t1n) || break               # keep last finite iterate
        drel = maximum(abs, t1n .- theta1) / max(1.0, maximum(abs, theta1))
        theta1 = t1n; xi = xin; iters = it
        if drel < opts.wtol
            converged = true; break
        end
    end
    Wf = cue_pinv(cue_weight(xi, pc.Z_moments, opts), opts.pinv_rtol)   # W from the FINAL ξ
    g  = compute_gmm_moments(xi, pc)              # ALL rows, unmasked
    Q  = all(isfinite, g) ? length(delta) * dot(g, Wf * g) : 1e12
    return theta1, xi, Wf, Q, iters, converged
end

"""
    cue_fixed_point_alpha0(delta, pc, opts, alpha0) -> (beta, xi, W_final, S, iters, converged)

Stock–Wright building block: the SAME CUE fixed point with α held at `alpha0`. δ̃ = δ − α₀·spread
(X_full column 1 IS the spread), β-only W-weighted GMM on X̃ = X_full[:, 2:end], ξ = δ − α₀·p − X̃β.
Minimised over θ₂ this gives S(α₀) = min N·ḡ'Wḡ, inverted into the identification-robust set for α.
"""
function cue_fixed_point_alpha0(delta::Vector{Float64}, pc::Precomp, opts::CueOpts, alpha0::Float64)
    spread = view(pc.X_full, :, 1)
    Xt     = pc.X_full[:, 2:end]                  # K−1 exogenous columns
    dtil   = delta .- alpha0 .* spread
    valid  = pc.theta1_valid .& isfinite.(delta)
    L      = size(pc.Z_moments, 2)
    # iteration 0: β by OLS-style 2SLS analogue on the projected design (X_hat cols 2:end = x_mat,
    # which are exogenous ⇒ plain least squares of δ̃ on X̃ over valid rows)
    beta = try Xt[valid, :] \ dtil[valid] catch; zeros(size(Xt, 2)) end
    xi   = dtil .- Xt * beta
    all(isfinite, xi) || return beta, xi, Matrix{Float64}(I, L, L), 1e12, 0, false
    Zv   = pc.Z_moments[valid, :]
    ZtX  = Zv' * Xt[valid, :]
    Ztd  = Zv' * dtil[valid]
    iters = 0; converged = false
    for it in 1:opts.max_witer
        Wn = cue_pinv(cue_weight(xi, pc.Z_moments, opts), opts.pinv_rtol)
        A  = ZtX' * Wn * ZtX
        b  = ZtX' * Wn * Ztd
        bn = try A \ b catch; pinv(A) * b end
        all(isfinite, bn) || break
        drel = maximum(abs, bn .- beta) / max(1.0, maximum(abs, beta))
        beta = bn; xi = dtil .- Xt * beta; iters = it
        if drel < opts.wtol
            converged = true; break
        end
    end
    Wf = cue_pinv(cue_weight(xi, pc.Z_moments, opts), opts.pinv_rtol)
    g  = compute_gmm_moments(xi, pc)
    S  = all(isfinite, g) ? length(delta) * dot(g, Wf * g) : 1e12
    return beta, xi, Wf, S, iters, converged
end

"""Efficient-GMM sandwich SEs for the CUE θ₁, with θ₂ and W held fixed (labelled as such in the
results): V = (A'WA)⁻¹/N_v with A = Z'X/N_v on valid rows, CR1 small-cluster correction."""
function cue_linear_se(delta::Vector{Float64}, pc::Precomp, W::Matrix{Float64}, G::Int)
    valid = pc.theta1_valid .& isfinite.(delta)
    Nv    = sum(valid)
    K     = size(pc.X_full, 2)
    A     = (pc.Z_moments[valid, :]' * pc.X_full[valid, :]) ./ Nv
    V     = pinv(Symmetric(A' * W * A)) ./ Nv
    corr  = G > 1 ? (G / (G - 1)) * ((Nv - 1) / (Nv - K)) : 1.0
    return sqrt.(max.(diag(V) .* corr, 0.0))
end

# ==========================================================================
# 1. Load Pre-Computed Draws
# ==========================================================================
function load_precomputed_draws(draws_dir::String, R::Int, seed::Int)
    nu_path  = joinpath(draws_dir, "halton_nu_R$(R)_seed$(seed).jls")
    demo_path = joinpath(draws_dir, "demo_draws_R$(R)_seed$(seed).jls")
    key_path  = joinpath(draws_dir, "demo_key_index_R$(R)_seed$(seed).jls")

    isfile(nu_path) || error("Missing ν-draws: $nu_path")
    isfile(demo_path) || error("Missing demo draws: $demo_path")
    isfile(key_path) || error("Missing key index: $key_path")

    log_status("Loading pre-computed draws from $draws_dir")

    nu_draws   = deserialize(nu_path)::Matrix{Float64}
    draws_3d   = deserialize(demo_path)::Array{Float64,3}
    key_index  = deserialize(key_path)::Dict{Tuple{String,String},Int}
    # compute_mu! reads ν columns 1:coef_dim inside an @turbo loop, which does not check bounds.
    size(nu_draws, 2) >= 1 + L_PROD || error(
        "ν draws have $(size(nu_draws, 2)) columns but the random-coefficient design has " *
        "$(1 + L_PROD) (spread + X_COLS). Rebuild the draws (blp_draws.jl) for this X_COLS.")

    # Normalize each demographic dimension to unit std so that π bounds of ±2
    # are meaningful for all demographics regardless of their natural scale
    # (gdp_per_capita ~50,000 vs fraction_65plus ~0.15 differ by ~300,000×).
    # π is then in "utility per 1-std unit of demographic × 1 pp of spread".
    D = size(draws_3d, 3)
    demo_scale  = ones(D)
    demo_center = zeros(D)
    for d in 1:D
        vals = vec(draws_3d[2:end, :, d])   # skip padding row (index 1)
        σ = std(vals)
        if σ > 1e-10
            demo_scale[d] = σ
            draws_3d[:, :, d] ./= σ
        end
        # Center the real-market rows to zero mean → D̃ = (D − D̄)/σ, so the reported
        # θ₁ is the AVERAGE-MARKET coefficient (the value at the cross-market mean),
        # NOT the value at demographics = 0 (a market that does not exist). This is a
        # pure reparametrization: the constant Σ_k x_k Σ_d π_kd·D̄_d/σ_d is linear in the
        # product characteristics, so it is absorbed entirely into θ₁ (δ̂ shifts by the
        # matching X·b); ξ̂, simulated shares, Q, and θ₂ (σ, π) + their SEs are unchanged.
        # The padding row 1 is left at 0 = the average market, so unmatched observations
        # carry no π deviation.
        μ_d = mean(vec(draws_3d[2:end, :, d]))
        demo_center[d] = μ_d
        draws_3d[2:end, :, d] .-= μ_d
    end

    log_status("  ν-draws: $(size(nu_draws)) | demo_draws: $(size(draws_3d)) | keys: $(length(key_index))")
    log_status("  Demo σ-scale (for π re-interpretation): $(round.(demo_scale, sigdigits=4))")
    log_status("  Demo center (subtracted D̄/σ → θ₁ is average-market): $(round.(demo_center, sigdigits=4))")
    return nu_draws, draws_3d, key_index
end

# ==========================================================================
# 2. Run BLP for one spec — using pre-computed draws
# ==========================================================================
function run_blp_estimation(estim::Int, spec_id::Int, args,
                             nu_draws::Matrix{Float64},
                             draws_3d::Array{Float64,3},
                             key_index::Dict{Tuple{String,String},Int})
    println("\n" * "=" ^ 60)
    println("  Estimation $estim — Specification $spec_id")
    println("=" ^ 60)

    path = demand_parquet_path(estim, spec_id, args)
    path === nothing && return nothing

    df = DataFrame(Parquet2.Dataset(path); copycols=true)
    nrow(df) == 0 && (println("  [!] Empty dataframe."); return nothing)
    println("  Loaded: $(nrow(df)) observations")

    sigma_indices, pi_interactions, n_params = build_theta2_structure(args["stage"])
    R        = args["R"]
    seed     = args["seed"]
    coef_dim = 1 + L_PROD

    results = Dict{String,Any}("spec_id" => spec_id, "estim" => estim,
                                "stage" => args["stage"])

    if args["stage"] == "logit"
        println("  Stage: LOGIT — delegating to blp_logit.jl logic")
        is_B = Bool.(coalesce.(df.is_B, false))
        share_D      = Float64.(coalesce.(df.share_D,      0.0))
        share_B_cond = Float64.(coalesce.(df.share_B_cond, 0.0))
        delta = zeros(nrow(df))
        delta[.!is_B] .= log.(clamp.(share_D[.!is_B],      1e-15, Inf))
        delta[is_B]   .= log.(clamp.(share_B_cond[is_B],    1e-15, Inf))

        # Read the design first: an input gap stops the run before anything is written.
        spread_cols, x_mat, Z_mat, iv_cols = build_regressor_matrices(df)

        # Save δ as version-agnostic binary — survives Julia version upgrades on cluster.
        # Into the logit step directory, the one place the δ readers (this file's warm start,
        # cf_demand_eval.jl's logit stage) look for it.
        let _logit_out = logit_dir(get_paths(args["hpc"]; local_dir=args["local_dir"])[3])
            mkpath(_logit_out)
            _bin = joinpath(_logit_out, "logit_delta_E$(estim)_spec_$(spec_id).bin")
            save_delta_bin(_bin, delta)
            log_status("  [δ BIN] Saved $(basename(_bin)) ($(length(delta)) obs)")
        end

        H          = hcat(x_mat, Z_mat)
        dep_types  = Int.(coalesce.(df.deposit_type, 0))
        spread_hat = project_endogenous_spreads(spread_cols, H, dep_types)
        X_full     = hcat(spread_cols, x_mat)
        X_hat      = hcat(spread_hat,  x_mat)
        valid      = BitVector(all.(isfinite, eachrow(X_hat)))
        # Clustering: CodConglomeradoPrudencial only (matches sleep estimation)
        clusters   = string.(df.CodConglomeradoPrudencial)
        Z_clean, iv_avail = moment_instruments(Z_mat, iv_cols)
        pc_logit = Precomp(Int[], Int[], Tuple{String,String}[], String[], Int[],
                           Float64[],
                           sparse(zeros(0,0)), sparse(zeros(0,0)), sparse(zeros(0,0)),
                           Int[], Int[], Int[], Int[], Int[], Int[], Int[], Int[],
                           BitVector(Bool.(coalesce.(df.is_B, false))),
                           .!BitVector(Bool.(coalesce.(df.is_B, false))),
                           log.(clamp.(share_D,      1e-15, Inf)),
                           log.(clamp.(share_B_cond, 1e-15, Inf)),
                           Z_clean, X_full, X_hat, valid, clusters)

        theta1, xi              = estimate_theta1(delta, pc_logit)
        theta1_se, n_cl, G_s    = compute_cluster_se(theta1, delta, pc_logit)
        g_moments = compute_gmm_moments(xi, pc_logit)
        W  = Matrix(I(length(iv_avail)) * 1.0)
        Q  = dot(g_moments, W * g_moments)
        results["theta1"]             = theta1
        results["theta1_se"]          = theta1_se
        results["theta2"]             = Float64[]
        results["theta2_se"]          = Float64[]
        results["delta"]              = delta
        results["xi"]                 = xi
        results["Q_value"]            = Q
        results["converged"]          = true
        results["param_names_theta1"] = vcat(["alpha"], X1_COLS)
        results["sigma_indices"]      = Int[]
        results["pi_interactions"]    = Tuple{Int,Int}[]
        results["n_clusters"]         = n_cl
        results["G_star"]             = G_s
        println("  theta1 (alpha): $(round(theta1[1], sigdigits=6))")
        println("  Q(0) = $(round(Q, sigdigits=6))")
        return results
    end

    # ── BLP stages (sigma / full / extended) ─────────────────────────────
    println("  Stage: $(uppercase(args["stage"])) ($n_params parameters)")

    # Map observations to pre-computed draw indices
    N_obs     = nrow(df)
    mca_codes = string.(df.mca_code)
    time_ids  = string.(df.time_id)
    pad_idx   = 1  # the zero padding row: blp_draws.jl writes it at index 1, markets at 2..n+1

    obs_key_idx = [get(key_index, (mca_codes[i], time_ids[i]), pad_idx)
                   for i in 1:N_obs]

    n_mapped  = count(i -> i != pad_idx, obs_key_idx)
    log_status("  Obs mapped to draws: $n_mapped / $N_obs ($(round(100*n_mapped/N_obs, digits=1))%)")

    dep_types = Int.(coalesce.(df.deposit_type, 0))
    spread_cols, x_mat, Z_mat, iv_cols = build_regressor_matrices(df)
    prod_vec = rc_design(spread_cols, x_mat)           # spread + X_COLS (N × coef_dim); D is not in it
    Z, iv_avail = moment_instruments(Z_mat, iv_cols)
    W = try inv(Z' * Z ./ N_obs) catch; Matrix(I(length(iv_avail)) * 1.0) end

    H          = hcat(x_mat, Z_mat)
    spread_hat = project_endogenous_spreads(spread_cols, H, dep_types)
    X_full     = hcat(spread_cols, x_mat)
    X_hat      = hcat(spread_hat,  x_mat)
    valid      = BitVector(all.(isfinite, eachrow(X_hat)))
    # Clustering: CodConglomeradoPrudencial only
    clusters   = string.(df.CodConglomeradoPrudencial)
    pc = build_precomp(df, Z, X_full, X_hat, valid, clusters)

    N_B     = sum(pc.b_mask)
    N_D     = sum(pc.d_mask)
    n_pairs = length(pc.unique_pairs)
    n_times = length(pc.unique_times)

    buf = allocate_hot_buffers(N_obs, N_B, N_D, R, n_pairs, n_times, coef_dim,
                                length(pi_interactions))
    precompute_pi_products!(buf, prod_vec, draws_3d, obs_key_idx,
                             pi_interactions, coef_dim)

    # ── Warm-start from previous stage checkpoint ────────────────────────
    # in_dir is the FIRST get_paths return (on HPC: data/input, where an uploaded
    # logit_delta warm-start would live); out_dir is the output ROOT — the checkpoints reached
    # through blp_dir, the logit δ̂ through logit_dir, each family in the one directory its
    # producer writes.
    in_dir, _, out_dir = get_paths(args["hpc"])
    # Warm-start chain. `ext1` reads `rc4`, not `full`: since σ(ln assets) was dropped from θ₂,
    # `full` has the SAME θ₂ structure as `rc4` and reproduces it exactly (verified in job 18689035 —
    # identical Q, α, SE and every θ₂ across all link routines), so it is a redundant rung and is no
    # longer part of the default sequence. Its entry is kept so `--stage full` still works if
    # requested explicitly. ext1 and extended are two-start stages (TWO_START_STAGES): ext1 keeps the
    # cold θ₂₀ below as its first start and takes the rc4 checkpoint as its second; extended keeps
    # this ext2 warm start as its first and adds a cold one. Both run in the optimisation block.
    prev_stages = Dict("rc2" => "sigma", "rc3" => "rc2", "rc4" => "rc3",
                       "full" => "rc4", "ext1" => "rc4", "ext2" => "ext1",
                       "extended" => "ext2")
    theta2_0 = nothing
    two_start = haskey(TWO_START_STAGES, args["stage"])
    if args["stage"] in keys(prev_stages) &&
       !(two_start && TWO_START_STAGES[args["stage"]].first == :cold)
        prev_path = joinpath(blp_dir(out_dir),
            "blp_checkpoint_E$(estim)_spec_$(spec_id)_$(prev_stages[args["stage"]]).jls")
        if isfile(prev_path)
            try
                prev = deserialize(prev_path)
                t2p  = get(prev, "theta2_star", nothing)
                if t2p !== nothing && length(t2p) <= n_params
                    theta2_0 = zeros(n_params)
                    theta2_0[1:length(t2p)] .= t2p
                    println("  [WARM-START] theta2_0 from $prev_path")
                end
            catch e
                println("  [WARM-START] Could not load checkpoint: $e")
            end
        end
    end
    # A stage that follows a two-start stage (ext2 after ext1) starts only from that stage's result:
    # without its checkpoint the optimisation block refuses it (require_prev_checkpoint).
    prev_missing = theta2_0 === nothing && haskey(prev_stages, args["stage"]) &&
                   haskey(TWO_START_STAGES, prev_stages[args["stage"]])
    theta2_0 === nothing && (theta2_0 = randn(MersenneTwister(seed), n_params) .* 0.01)

    # ── Bounds: ±2 in pp units keeps σ×ν×spread O(1) for typical spreads ──
    lo = fill(-2.0, n_params)
    hi = fill( 2.0, n_params)

    println("  Outer minimisation: $(args["method"])")
    println("  Inner tolerance: $(args["tol_inner"])")
    println("  Bounds: [$(lo[1]), $(hi[1])]")

    # ── δ warm-start from logit checkpoint ──────────────────────────────────
    # blp_logit.jl saves logit_delta_E{id}_spec_{sp}.jls locally.
    # Rsync that file to the cluster output dir before submitting sigma jobs.
    # Starting from logit δ* saves 50–200 SQUAREM iters per outer GMM call.
    delta_work = zeros(N_obs)
    let _loaded = false
        # 1. Try version-agnostic binary first (no Julia serialization version dependency)
        #    Generated by blp_logit.jl or blp_engine_cpu.jl --stage logit
        #    in_dir (data/input on HPC) is checked first so a δ̂ supplied as an upload wins over
        #    one the logit step produced in this run; otherwise the logit step directory.
        for _bin_cand in [
                joinpath(in_dir,             "logit_delta_E$(estim)_spec_$(spec_id).bin"),
                joinpath(logit_dir(out_dir), "logit_delta_E$(estim)_spec_$(spec_id).bin"),
            ]
            isfile(_bin_cand) || continue
            _d = load_delta_bin(_bin_cand)
            if _d !== nothing && length(_d) == N_obs
                copyto!(delta_work, _d)
                log_status("  [δ WARM-START] Loaded $(basename(_bin_cand)) [binary] (n=$(N_obs))")
                _loaded = true
                break
            else
                log_status("  [δ WARM-START] Skipped .bin: " *
                           "size $(length(something(_d, []))) ≠ N_obs=$(N_obs)")
            end
        end
        # 2. Fall back to .jls checkpoints (may fail on Julia version mismatch)
        if !_loaded
            for _cand in [
                joinpath(in_dir,             "logit_delta_E$(estim)_spec_$(spec_id).jls"),
                joinpath(logit_dir(out_dir), "logit_delta_E$(estim)_spec_$(spec_id).jls"),
                joinpath(blp_dir(out_dir),   "blp_checkpoint_E$(estim)_spec_$(spec_id)_logit.jls"),
                joinpath(blp_dir(out_dir),   "blp_results_E$(estim)_spec_$(spec_id)_logit.jls"),
            ]
                isfile(_cand) || continue
                try
                    _ck = deserialize(_cand)
                    _d  = get(_ck, "delta", nothing)
                    if _d !== nothing && length(_d) == N_obs
                        copyto!(delta_work, _d)
                        log_status("  [δ WARM-START] Loaded $(basename(_cand)) (n=$(N_obs))")
                        _loaded = true
                        break
                    else
                        log_status("  [δ WARM-START] Skipped $(basename(_cand)): " *
                                   "delta size $(length(something(_d, []))) ≠ N_obs=$(N_obs)")
                    end
                catch _e
                    log_status("  [δ WARM-START] Skipped $(basename(_cand)): $(_e)")
                end
            end
        end
        if !_loaded
            delta_work[pc.d_mask] .= pc.ln_s_data_D[pc.d_mask]
            delta_work[pc.b_mask] .= pc.ln_s_data_B_cond[pc.b_mask]
            log_status("  [δ WARM-START] No logit checkpoint found — log-share init")
        end
    end

    if get(args, "dry_run", false)
        println("  [DRY RUN] Running 10 contraction iterations for timing...")
        sv, pv = unpack_theta2(theta2_0, sigma_indices, pi_interactions)
        compute_mu!(buf, prod_vec, nu_draws, sv, sigma_indices, pv, R, coef_dim)
        t0 = time()
        blp_contraction!(buf, copy(delta_work), pc, R;
                          tol=args["tol_inner"], max_iter=10)
        elapsed = time() - t0
        println("  [DRY RUN] 10 iterations in $(round(elapsed, digits=2))s " *
                "($(round(elapsed/10, digits=3)) s/iter).")
        return Dict("dry_run" => true)
    end

    outer_iter = Ref(0)
    function obj_fn(t2)
        val = gmm_objective!(buf, t2, prod_vec, nu_draws,
                              sigma_indices, pi_interactions, R, coef_dim,
                              W, args["tol_inner"], args["max_inner"],
                              delta_work, pc)
        outer_iter[] += 1
        outer_iter[] % 10 == 0 &&
            log_status("  [OUTER iter=$(outer_iter[])] Q=$(round(val, sigdigits=6)) " *
                       "theta2=$(round.(t2, digits=4))")
        return val
    end

    # One bounded minimisation from θ₀; a two-start stage runs it once per start.
    solve_from = (name, theta2_start) -> begin
        outer_iter[] = 0
        optim_tuple(optimize(obj_fn, lo, hi, theta2_start, Fminbox(LBFGS()),
                             Optim.Options(iterations=500, f_reltol=args["tol_outer"],
                                           show_trace=false)))
    end
    if two_start
        starts, start_rec = two_start_list(args["stage"], prev_stages[args["stage"]], theta2_0,
            joinpath(blp_dir(out_dir),
                     "blp_checkpoint_E$(estim)_spec_$(spec_id)_$(prev_stages[args["stage"]]).jls"),
            n_params, N_obs, lo, hi, seed)
        theta2_star, Q_min, n_outer, conv_outer, start_rec =
            solve_two_starts(solve_from, starts, start_rec, delta_work;
                # Q is re-evaluated at each start's θ₂* (reeval_rule): the contraction, then the objective.
                contract  = t -> (mu_at!(buf, prod_vec, nu_draws, t, sigma_indices, pi_interactions, R, coef_dim);
                                  blp_contraction!(buf, delta_work, pc, R; tol = reeval_tol(args),
                                                   max_iter = reeval_cap(args))[1]),
                objective = t -> gmm_objective!(buf, t, prod_vec, nu_draws, sigma_indices, pi_interactions,
                                                R, coef_dim, W, args["tol_inner"], args["max_inner"],
                                                delta_work, pc),
                reeval_note = reeval_rule(args))
        results[two_start_key(args["stage"])] = start_rec
    else
        prev_missing && (results[two_start_key(args["stage"])] = require_prev_checkpoint(
            args["stage"], prev_stages[args["stage"]],
            joinpath(blp_dir(out_dir),
                     "blp_checkpoint_E$(estim)_spec_$(spec_id)_$(prev_stages[args["stage"]]).jls")))
        theta2_star, Q_min, n_outer, conv_outer, _ = solve_from("single", theta2_0)
    end
    println("  Optimiser converged: $(conv_outer)")
    println("  Q(theta2*) = $(round(Q_min, sigdigits=6))")
    println("  theta2* = $theta2_star")

    # Final contraction at optimum
    sv, pv = unpack_theta2(theta2_star, sigma_indices, pi_interactions)
    compute_mu!(buf, prod_vec, nu_draws, sv, sigma_indices, pv, R, coef_dim)
    delta_final = copy(delta_work)
    conv, n_it, _ = blp_contraction!(buf, delta_final, pc, R;
                                      tol=args["tol_inner"], max_iter=args["max_inner"])

    theta1_star, xi_star         = estimate_theta1(delta_final, pc)
    theta1_se, n_cl, G_s         = compute_cluster_se(theta1_star, delta_final, pc)

    results["theta1"]             = theta1_star
    results["theta1_se"]          = theta1_se
    results["theta2"]             = theta2_star
    results["theta2_se"]          = fill(NaN, length(theta2_star))
    # This engine computes no joint (θ₁, θ₂) SEs: θ₁'s are the linear-IV ones, θ₂'s are not computed.
    results["se_status"]          = "no joint SEs (CPU engine): theta1_se linear-IV, theta2_se not computed"
    results["delta"]              = delta_final
    results["xi"]                 = xi_star
    results["Q_value"]            = Q_min
    results["converged"]          = conv_outer
    results["n_outer_iter"]       = n_outer
    results["param_names_theta1"] = vcat(["alpha"], X1_COLS)
    results["sigma_indices"]      = sigma_indices
    results["pi_interactions"]    = pi_interactions
    results["n_clusters"]         = n_cl
    results["G_star"]             = G_s
    println("  theta1 (alpha): $(round(theta1_star[1], sigdigits=6))")

    # Save checkpoint into the BLP step directory — the same place the warm-start reader above
    # and _result_path (cf_demand_eval.jl) look, so a checkpoint written here is the one
    # the next stage and the CF stack pick up.
    _res_dir = blp_dir(out_dir)
    mkpath(_res_dir)
    chk_path = joinpath(_res_dir,
        "blp_checkpoint_E$(estim)_spec_$(spec_id)_$(args["stage"]).jls")
    try
        # A two-start stage's record travels with its checkpoint (the GPU IFT --se-only reads it back).
        _ts_ck = Dict("theta2_star" => theta2_star, "delta_star" => delta_final)
        _ts_k  = two_start_key(args["stage"])
        haskey(results, _ts_k) && (_ts_ck = Dict{String,Any}(_ts_ck..., _ts_k => results[_ts_k]))
        serialize(chk_path, _ts_ck)
        println("  [CHECKPOINT] Saved $(basename(chk_path))")
    catch e
        println("  [CHECKPOINT-WARN] $e")
    end
    # Save delta as version-agnostic binary for next-stage warm-start
    chk_bin_path = replace(chk_path, ".jls" => ".bin")
    save_delta_bin(chk_bin_path, delta_final)
    println("  [CHECKPOINT BIN] Saved $(basename(chk_bin_path))")

    return results
end

# ==========================================================================
# 3. CLI + Main
# ==========================================================================
function parse_args_est()
    s = ArgParseSettings(description="BLP Demand Estimation — Pre-Computed Draws")
    @add_arg_table! s begin
        "--estim";   arg_type=Int;     required=true;  help="Estimation strategy (1–5)"
        "--spec";    arg_type=String;  default="12"
        "--stage";   arg_type=String;  default="sigma"
        "--R";       arg_type=Int;     default=2000
        "--seed";    arg_type=Int;     default=42
        "--tol-inner"; arg_type=Float64; default=1e-12; dest_name="tol_inner"
        "--max-inner"; arg_type=Int;     default=5000;  dest_name="max_inner"
        "--tol-outer"; arg_type=Float64; default=1e-6;  dest_name="tol_outer"
        "--method";  arg_type=String;  default="l-bfgs-b"
        "--workers"; arg_type=Int;     default=Threads.nthreads()
        "--hpc";     action=:store_true
        "--local-dir"; arg_type=String; default=nothing; dest_name="local_dir"
        "--dry-run"; action=:store_true; dest_name="dry_run"
        # Recompute SEs from this stage's existing checkpoint instead of re-optimising. Valid ONLY
        # when the change is confined to the SE routine (blp_se_common.jl / BLP_SE_METHOD / WCB knobs),
        # which runs after optimisation and cannot move the point estimates. Requires
        # blp_checkpoint_E{estim}_spec_{spec}_{stage}.jls from a prior normal run.
        "--se-only"; action=:store_true; dest_name="se_only"
    end
    return parse_args(s)
end

function main()
    args     = parse_args_est()
    estim    = args["estim"]
    spec_ids = args["spec"] == "all" ? collect(1:12) : [parse(Int, args["spec"])]

    log_status("BLP Estimation (Pre-Computed Draws) — START")
    log_status("  E$estim | Stage: $(args["stage"]) | Specs: $spec_ids")
    log_status("  R=$(args["R"]) | seed=$(args["seed"]) | HPC=$(args["hpc"])")
    log_status("  tol_inner=$(args["tol_inner"]) | tol_outer=$(args["tol_outer"]) | threads=$(Threads.nthreads())")

    _, draws_dir, out_dir = get_paths(args["hpc"]; local_dir=args["local_dir"])
    # Results, per-spec JSON and the stage summary all belong to the BLP step directory, which is
    # what the archiver packages and what _result_path resolves; created up front because the
    # spec loop below writes into it from several threads.
    res_dir = blp_dir(out_dir)
    mkpath(res_dir)

    # Load pre-computed draws once (shared across all specs and stages)
    nu_draws, draws_3d, key_index = load_precomputed_draws(
        draws_dir, args["R"], args["seed"])

    stages_to_run = args["stage"] == "sequence" ?
                    ["sigma", "rc2", "rc3", "rc4", "ext1", "ext2", "extended"] : [args["stage"]]

    for current_stage in stages_to_run
        args["stage"] = current_stage

        # Stage-skipping
        all_done = true
        for sp in spec_ids
            rpath = joinpath(res_dir,
                "blp_results_E$(estim)_spec_$(sp)_$(current_stage).jls")
            if !isfile(rpath); all_done = false; break; end
        end
        if all_done
            log_status("[SKIP] Stage '$current_stage' already complete.")
            continue
        end

        all_results = Dict{Int,Any}()
        lk          = ReentrantLock()

        println("\n  Starting stage '$current_stage' for $(length(spec_ids)) specs " *
                "with $(Threads.nthreads()) threads...")

        Threads.@threads for sp in spec_ids
            out_path = joinpath(res_dir,
                "blp_results_E$(estim)_spec_$(sp)_$(current_stage).jls")
            if isfile(out_path)
                log_status("  [SKIP] Spec $sp already done for '$current_stage'")
                continue
            end

            res = try
                run_blp_estimation(estim, sp, args, nu_draws, draws_3d, key_index)
            catch e
                # A gap in the demand inputs stops the run: no later stage may start without it.
                e isa DemandInputError && rethrow()
                # So does a two-start stage that cannot run both of its starts, and any failure while a
                # two-start stage (ext1, extended) runs: the stages after it must not start without it.
                if e isa TwoStartError || haskey(TWO_START_STAGES, current_stage)
                    println("  [!] Spec $sp failed in stage '$current_stage': $e. Stopping the job.")
                    rethrow()
                end
                println("  [!] Spec $sp failed: $e"); nothing
            end
            res === nothing && continue
            get(res, "dry_run", false) && continue

            serialize(out_path, res)
            println("  Saved: $(basename(out_path))")
            # Per-spec JSON for Python consumption
            json_path = replace(out_path, ".jls" => ".json")
            try
                json_out = Dict{String,Any}(
                    "estim"              => get(res, "estim", estim),
                    "spec_id"            => get(res, "spec_id", sp),
                    "stage"              => get(res, "stage", current_stage),
                    "theta1"             => round.(get(res, "theta1",    Float64[]), sigdigits=8),
                    "theta1_se"          => json_se_values(get(res, "theta1_se", Float64[]), get(res, "se_status", ""), 0.0),
                    "theta2"             => round.(get(res, "theta2",    Float64[]), sigdigits=8),
                    "theta2_se"          => json_se_values(get(res, "theta2_se", Float64[]), get(res, "se_status", ""), 0.0),
                    "se_status"          => get(res, "se_status", ""),
                    "Q_value"            => get(res, "Q_value",   0.0),
                    "converged"          => get(res, "converged", false),
                    "param_names_theta1" => get(res, "param_names_theta1", String[]),
                    "sigma_indices"      => get(res, "sigma_indices",     Int[]),
                    "pi_interactions"    => get(res, "pi_interactions",   []),
                    "n_clusters"         => get(res, "n_clusters",        0),
                    "G_star"             => get(res, "G_star",            0.0),
                )
                # A start record (two-start stages; ext2 when it starts cold under the override),
                # non-finite values as null.
                for _ts_k in start_record_keys(res)
                    haskey(res, _ts_k) && (json_out[_ts_k] = json_finite(res[_ts_k]))
                end
                open(json_path, "w") do f; JSON3.write(f, json_out); end
            catch e
                println("  [JSON-WARN] $e")
            end
            lock(lk) do
                all_results[sp] = Dict(
                    "Q_value"      => get(res, "Q_value", 0.0),
                    "converged"    => get(res, "converged", true),
                    "theta1_alpha" => isempty(get(res, "theta1", Float64[])) ?
                                     [] : [res["theta1"][1]],
                    "theta2"       => get(res, "theta2", Float64[]),
                    "stage"        => current_stage)
            end
        end

        summary_path = joinpath(res_dir, "blp_summary_E$(estim)_$(current_stage).json")
        open(summary_path, "w") do f; JSON3.write(f, all_results); end
        log_status("Summary saved to: $summary_path")
        log_status("[DONE] BLP E$estim ($current_stage) complete for specs $spec_ids.")
    end
end

# Guard allows safe include() from blp_estimation_gpu.jl without auto-running main
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
