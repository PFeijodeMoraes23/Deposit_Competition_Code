#!/usr/bin/env julia
# ============================================================================
# sleep_joint_sieve.jl — Julia hot-path for the joint single-index sleepiness
# estimator (Est7: Ichimura SLS with a monotone I-spline sieve link).
#
# Scope: ONLY the expensive theta-search — argmin over the unit sphere of the
# entity[/+time]-demeaned SSR with the monotone sieve link profiled out. Python
# does data prep, the CF first stage, the FINAL link refit at theta_hat (with an
# EXACT two-way demean), phi, AMEs and the wild bootstrap; it pre-demeans y and cf
# (passed in) so Julia demeans only the per-theta design R.*Z.
#
# PERF: the hot loop is ALLOCATION-FREE. The profiled BVLS link is solved from
# K x K cross-products (Gram matrices) via FWL (partial out the free CF coef) +
# Lawson-Hanson NNLS on the NORMAL equations — no N-sized temporaries per eval.
# Two-way demean is in-place by column with a fixed sweep count (the search
# tolerates an approximate FE projection; Python's final refit is exact).
# Threaded multistart. Mirrors utils/sleep_links.py:_fit_link_sieve.
# ============================================================================
using Optim
using Base.Threads: @threads, nthreads
using LinearAlgebra: dot
using Statistics: median, quantile
using Printf

const DEMEAN_ITERS = 8           # fixed alternating-projection sweeps (two-way search;
                                 # the FINAL Python link refit uses an exact tol-based demean)

read_f64(path, n) = (a = Vector{Float64}(undef, n); read!(path, a); a)
read_i32(path, n) = (a = Vector{Int32}(undef, n);  read!(path, a); a)

function read_manifest(path)
    d = Dict{String,String}()
    for ln in eachline(path)
        isempty(strip(ln)) && continue
        kv = split(ln, '='; limit=2); length(kv) == 2 || continue
        d[strip(kv[1])] = strip(kv[2])
    end
    return d
end

counts_of(code::Vector{Int32}, ng::Int) =
    (c = zeros(Float64, ng); @inbounds for i in eachindex(code); c[code[i]+1] += 1.0; end; c)

# ---- in-place within-group demean of ONE matrix column (by index) --------
function demean1_col!(M::Matrix{Float64}, j::Int, code::Vector{Int32}, ng::Int,
                      cnt::Vector{Float64}, s::Vector{Float64})
    N = size(M, 1)
    fill!(s, 0.0)
    @inbounds for i in 1:N; s[code[i]+1] += M[i, j]; end
    @inbounds for g in 1:ng; s[g] = cnt[g] > 0 ? s[g]/cnt[g] : 0.0; end
    @inbounds for i in 1:N; M[i, j] -= s[code[i]+1]; end
end

# one-way (tcode=nothing) or two-way (fixed DEMEAN_ITERS sweeps) demean of col j
function demean_col!(M, j, ecode, nE, ec, tcode, nT, tc, s_e, s_t)
    if tcode === nothing
        demean1_col!(M, j, ecode, nE, ec, s_e)
    else
        for _ in 1:DEMEAN_ITERS
            demean1_col!(M, j, ecode, nE, ec, s_e)
            demean1_col!(M, j, tcode, nT, tc, s_t)
        end
    end
end

# ---- Cox-de Boor B-spline basis at scalar x (writes K vals) --------------
function bspline_basis!(B::Vector{Float64}, x::Float64, t::Vector{Float64}, deg::Int, K::Int)
    fill!(B, 0.0)
    lo = t[1]; hi = t[end]; x < lo && (x = lo); x > hi && (x = hi)
    nt = length(t); sp = deg + 1
    @inbounds for i in (deg+1):(nt-deg-1)
        if (x >= t[i] && x < t[i+1]) || i == nt - deg - 1
            sp = i; (x >= t[i] && x < t[i+1]) && break
        end
    end
    N = zeros(Float64, deg + 1); N[1] = 1.0
    @inbounds for d in 1:deg
        saved = 0.0
        for r in 1:d
            tr = t[sp+r]; tl = t[sp+r-d]; denom = tr - tl
            temp = denom > 0 ? N[r]/denom : 0.0
            N[r] = saved + (tr - x)*temp; saved = (x - tl)*temp
        end
        N[d+1] = saved
    end
    @inbounds for r in 0:deg
        col = sp - deg + r; (1 <= col <= K) && (B[col] = N[r+1])
    end
    return B
end

function ramp_design!(R::Matrix{Float64}, v::Vector{Float64}, t::Vector{Float64},
                      deg::Int, K::Int, Bbuf::Vector{Float64})
    @inbounds for i in eachindex(v)
        bspline_basis!(Bbuf, v[i], t, deg, K)
        acc = 0.0
        for j in K:-1:1; acc += Bbuf[j]; R[i, j] = acc; end
    end
    return R
end

# Opt 5: binned (Fan-Marron 1994) ramp design. Evaluate Cox-de-Boor once per bin
# (nbins+1 grid points) and linearly interpolate per row — cuts N basis calls to
# ~nbins. The ramp is smooth, so the interpolation bias is O(1/nbins^2), negligible
# for the theta-SEARCH (Python's final refit is exact, unbinned).
function ramp_design_binned!(R::Matrix{Float64}, v::Vector{Float64}, t::Vector{Float64},
                             deg::Int, K::Int, Bbuf::Vector{Float64},
                             nbins::Int, gridR::Matrix{Float64})
    lo = t[1]; hi = t[end]; (hi <= lo) && (hi = lo + 1.0)
    dv = (hi - lo) / nbins; invdv = 1.0 / dv
    @inbounds for b in 0:nbins
        bspline_basis!(Bbuf, lo + b * dv, t, deg, K)
        acc = 0.0
        for j in K:-1:1; acc += Bbuf[j]; gridR[b+1, j] = acc; end
    end
    @inbounds for i in eachindex(v)
        pos = (v[i] - lo) * invdv
        b = floor(Int, pos); b < 0 && (b = 0); b > nbins - 1 && (b = nbins - 1)
        fr = pos - b
        for j in 1:K
            R[i, j] = gridR[b+1, j] + fr * (gridR[b+2, j] - gridR[b+1, j])
        end
    end
    return R
end

# ---- Lawson-Hanson NNLS on the NORMAL equations (no N-sized work) ---------
function nnls_normal!(x, AtA, Atb, P, w; maxiter=200, tol=1e-10)
    n = length(Atb)
    fill!(x, 0.0); fill!(P, false); it = 0
    while true
        @inbounds for j in 1:n
            s = Atb[j]
            for k in 1:n; s -= AtA[j, k]*x[k]; end
            w[j] = s
        end
        maxw = -Inf; jmax = 0
        @inbounds for j in 1:n
            if !P[j] && w[j] > maxw; maxw = w[j]; jmax = j; end
        end
        (jmax == 0 || maxw <= tol) && break
        it += 1; it > maxiter && break
        P[jmax] = true
        while true
            idx = findall(P)
            z = @views AtA[idx, idx] \ Atb[idx]
            if all(>(0), z)
                fill!(x, 0.0)
                @inbounds for (kk, j) in enumerate(idx); x[j] = z[kk]; end
                break
            end
            alpha = Inf
            @inbounds for (kk, j) in enumerate(idx)
                if z[kk] <= 0
                    dd = x[j] - z[kk]; a = dd > 0 ? x[j]/dd : 0.0; a < alpha && (alpha = a)
                end
            end
            @inbounds for (kk, j) in enumerate(idx); x[j] += alpha*(z[kk] - x[j]); end
            @inbounds for j in idx
                if x[j] <= tol; x[j] = 0.0; P[j] = false; end
            end
        end
    end
    return x
end

# allocation-free quantile/median on a preallocated sort buffer (linear interp,
# matches numpy/Julia default). buf is overwritten.
@inline function quantile_sorted(s, q)
    N = length(s); h = q*(N - 1); lo = floor(Int, h); fr = h - lo
    @inbounds return (lo + 1 < N) ? s[lo+1] + fr*(s[lo+2] - s[lo+1]) : s[N]
end
function median_abs!(buf, r)
    @inbounds for i in eachindex(r); buf[i] = abs(r[i]); end
    sort!(buf); return quantile_sorted(buf, 0.5)
end
function mad!(buf, r)
    copyto!(buf, r); sort!(buf); med = quantile_sorted(buf, 0.5)
    @inbounds for i in eachindex(r); buf[i] = abs(r[i] - med); end
    sort!(buf); return 1.4826*quantile_sorted(buf, 0.5) + 1e-12
end
function cauchy_weights!(sw, r, buf)
    s = mad!(buf, r)
    @inbounds for i in eachindex(r); sw[i] = sqrt(1.0/(1.0 + (r[i]/(2.385*s))^2)); end
    return sw
end
function cauchy_obj(r, buf)
    s = 1.4826*median_abs!(buf, r) + 1e-12
    o = 0.0
    @inbounds for i in eachindex(r); o += log1p((r[i]/(2.385*s))^2); end
    return o
end

struct Work
    R::Matrix{Float64}; Rz::Matrix{Float64}; resid::Vector{Float64}; sw::Vector{Float64}
    s_e::Vector{Float64}; s_t::Vector{Float64}; Bbuf::Vector{Float64}
    AtA::Matrix{Float64}; Atb::Vector{Float64}; mRc::Vector{Float64}
    beta::Vector{Float64}; Pset::Vector{Bool}; wvec::Vector{Float64}
    sortbuf::Vector{Float64}; absbuf::Vector{Float64}; gridR::Matrix{Float64}
end
Work(N, K, nE, nT, nbins) = Work(Matrix{Float64}(undef, N, K), Matrix{Float64}(undef, N, K),
    Vector{Float64}(undef, N), ones(N), zeros(nE), zeros(max(nT, 1)), zeros(K),
    zeros(K, K), zeros(K), zeros(K), zeros(K), zeros(Bool, K), zeros(K),
    Vector{Float64}(undef, N), Vector{Float64}(undef, N), zeros(max(nbins + 1, 1), K))

struct Problem
    Sn::Matrix{Float64}; Z::Vector{Float64}; y_dm::Vector{Float64}
    cf_dm::Union{Vector{Float64},Nothing}
    ecode::Vector{Int32}; nE::Int; ec::Vector{Float64}
    tcode::Union{Vector{Int32},Nothing}; nT::Int; tc::Vector{Float64}
    n_interior::Int; degree::Int; loss::Symbol; K::Int; d::Int; nbins::Int
end

# ---- profiled monotone sieve fit at fixed index v; writes resid, returns it.
# Solves the weighted BVLS [Rz_dm (>=0) | cf_dm (free)] via FWL on K x K Gram
# cross-products + NNLS-normal. sw === workspace of sqrt weights (ones if LS).
function link_resid!(wk::Work, prob::Problem, v::Vector{Float64}, t::Vector{Float64}, weighted::Bool)
    K = prob.K; N = length(v); Rz = wk.Rz
    if prob.nbins > 0
        ramp_design_binned!(wk.R, v, t, prob.degree, K, wk.Bbuf, prob.nbins, wk.gridR)
    else
        ramp_design!(wk.R, v, t, prob.degree, K, wk.Bbuf)
    end
    @inbounds for j in 1:K, i in 1:N; Rz[i, j] = wk.R[i, j]*prob.Z[i]; end
    @inbounds for j in 1:K
        demean_col!(Rz, j, prob.ecode, prob.nE, prob.ec, prob.tcode, prob.nT, prob.tc, wk.s_e, wk.s_t)
    end
    y = prob.y_dm; cf = prob.cf_dm; has_cf = cf !== nothing
    AtA = wk.AtA; Atb = wk.Atb; mRc = wk.mRc; sw = wk.sw
    fill!(AtA, 0.0); fill!(Atb, 0.0); fill!(mRc, 0.0)
    ccc = 0.0; ccy = 0.0
    @inbounds for i in 1:N
        wi = weighted ? sw[i]*sw[i] : 1.0
        yi = y[i]; cfi = has_cf ? cf[i] : 0.0
        for a in 1:K
            ra = wi*Rz[i, a]
            Atb[a] += ra*yi; mRc[a] += ra*cfi
            for b in a:K; AtA[a, b] += ra*Rz[i, b]; end
        end
        ccc += wi*cfi*cfi; ccy += wi*cfi*yi
    end
    @inbounds for a in 1:K, b in 1:a-1; AtA[a, b] = AtA[b, a]; end
    if has_cf && ccc > 1e-30                       # FWL: partial out the free CF coef
        f = ccy/ccc
        @inbounds for a in 1:K
            Atb[a] -= mRc[a]*f
            for b in 1:K; AtA[a, b] -= mRc[a]*mRc[b]/ccc; end
        end
    end
    beta = nnls_normal!(wk.beta, AtA, Atb, wk.Pset, wk.wvec)
    gamma = (has_cf && ccc > 1e-30) ? (ccy - dot(mRc, beta))/ccc : 0.0
    resid = wk.resid
    @inbounds for i in 1:N
        f = 0.0
        for a in 1:K; f += Rz[i, a]*beta[a]; end
        resid[i] = y[i] - f - (has_cf ? gamma*cf[i] : 0.0)
    end
    return resid
end

function knots_at!(sortbuf, v, n_interior, degree)
    copyto!(sortbuf, v); sort!(sortbuf)
    lo = quantile_sorted(sortbuf, 0.001); hi = quantile_sorted(sortbuf, 0.999)
    hi <= lo && (hi = lo + 1.0)
    qs = range(0, 1; length=n_interior + 2)[2:end-1]
    interior = sort!([clamp(quantile_sorted(sortbuf, q), lo + 1e-9, hi - 1e-9) for q in qs])
    return vcat(fill(lo, degree + 1), interior, fill(hi, degree + 1))
end

function profile_obj(prob::Problem, theta::Vector{Float64}, wk::Work, vbuf::Vector{Float64})
    Sn = prob.Sn; d = prob.d; N = size(Sn, 1)
    nrm = sqrt(sum(abs2, theta)) + 1e-12
    @inbounds for i in 1:N
        s = 0.0
        for k in 1:d; s += Sn[i, k]*theta[k]; end
        vbuf[i] = s/nrm
    end
    t = knots_at!(wk.sortbuf, vbuf, prob.n_interior, prob.degree)
    if prob.loss == :robust
        r = link_resid!(wk, prob, vbuf, t, false)
        for _ in 1:2
            cauchy_weights!(wk.sw, r, wk.absbuf)
            r = link_resid!(wk, prob, vbuf, t, true)
        end
        return cauchy_obj(r, wk.absbuf)
    else
        r = link_resid!(wk, prob, vbuf, t, false)
        return dot(r, r)
    end
end

function run_estimator(prob::Problem, starts::Vector{Vector{Float64}}, maxiter::Int)
    nst = length(starts); objs = fill(Inf, nst); xs = Vector{Vector{Float64}}(undef, nst)
    @threads for s in 1:nst
        N = size(prob.Sn, 1)
        wk = Work(N, prob.K, prob.nE, prob.nT, prob.nbins); vbuf = Vector{Float64}(undef, N)
        f = th -> profile_obj(prob, th, wk, vbuf)
        res = optimize(f, starts[s], NelderMead(),
                       Optim.Options(iterations=maxiter, x_abstol=1e-3, f_abstol=1e-5))
        objs[s] = Optim.minimum(res)
        x = Optim.minimizer(res); xs[s] = x ./ (sqrt(sum(abs2, x)) + 1e-12)
    end
    bi = argmin(objs); return xs[bi], objs[bi]
end

function main()
    mpath = ARGS[1]; man = read_manifest(mpath); dir = dirname(mpath)
    N = parse(Int, man["N"]); d = parse(Int, man["d"]); nE = parse(Int, man["nE"])
    twoway = get(man, "twoway", "0") == "1"
    nT = twoway ? parse(Int, man["nT"]) : 0
    has_cf = get(man, "has_cf", "1") == "1"
    n_interior = parse(Int, get(man, "n_interior", "5")); degree = parse(Int, get(man, "degree", "3"))
    loss = Symbol(get(man, "loss", "robust")); n_starts = parse(Int, get(man, "n_starts", "2"))
    maxiter = parse(Int, get(man, "maxiter", string(300 * d))); K = n_interior + degree + 1
    nbins = parse(Int, get(man, "nbins", "0"))           # opt 5: binned ramp (0 = exact)

    Sn = reshape(read_f64(joinpath(dir, "Sn.bin"), N * d), N, d)
    Z = read_f64(joinpath(dir, "Z.bin"), N); y_dm = read_f64(joinpath(dir, "y_dm.bin"), N)
    cf_dm = has_cf ? read_f64(joinpath(dir, "cf_dm.bin"), N) : nothing
    ecode = read_i32(joinpath(dir, "ecode.bin"), N); ec = counts_of(ecode, nE)
    tcode = twoway ? read_i32(joinpath(dir, "tcode.bin"), N) : nothing
    tc = twoway ? counts_of(tcode, nT) : zeros(1)

    prob = Problem(Sn, Z, y_dm, cf_dm, ecode, nE, ec, tcode, nT, tc, n_interior, degree, loss, K, d, nbins)

    starts = Vector{Vector{Float64}}()
    if isfile(joinpath(dir, "init.bin"))
        iv = read_f64(joinpath(dir, "init.bin"), d); nrm = sqrt(sum(abs2, iv))
        nrm > 1e-12 && push!(starts, iv ./ nrm)
    end
    ones_d = fill(1.0/sqrt(d), d)
    isempty(starts) ? push!(starts, ones_d) : (n_starts >= 2 && push!(starts, ones_d))
    rr = 1
    while length(starts) < n_starts
        r = [sin(rr*(k + 1.0)) for k in 1:d]; rr += 7; push!(starts, r ./ sqrt(sum(abs2, r)))
    end

    t0 = time(); theta, obj = run_estimator(prob, starts, maxiter); el = time() - t0
    @printf(stderr, "[julia joint-sieve] N=%d d=%d twoway=%s starts=%d obj=%.6g  %.1fs  threads=%d\n",
            N, d, twoway, length(starts), obj, el, nthreads())
    write(joinpath(dir, "theta_out.bin"), theta)
    open(joinpath(dir, "result.txt"), "w") do io
        println(io, "obj=", obj); println(io, "secs=", round(el, digits=2)); println(io, "threads=", nthreads())
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
