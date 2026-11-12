# blp_diag_ext1_d0.jl — D0 of the RC-BLP engine plan (2026-11-11): a read-only diagnostic of the ext1
# rung of one routine (E3 or E4) on the production data, measuring why the pipeline's cold and random
# starts freeze. It uses only engine functions that exist at HEAD 24f5d096 (so it runs with or without
# engine edit (b)) and the production files of data/output/blp, data/output/logit and config/.
#
#  (i)   For each start (cold = the engine's randn(MersenneTwister(42), 5) .* 0.01, and rand1–rand3
#        of config/blp_ext1_extra_starts.csv): the start's first evaluation as production ran it
#        (contraction from the logit δ̂ at tol 1e-10, cap 5000; Q; the exact IFT gradient, with its
#        singular blocks), ‖∇Q(θ₀)‖, L-BFGS-B's first trial P(θ₀ − ∇Q) (B₀ = I, a fully boxed problem:
#        step 1), and the contraction there from the δ θ₀ left (converged?, iterations, singular
#        blocks, δ and V at ±500, residuals).
#  (ii)  Q along θ₀ + t·(P(θ₀ − ∇Q) − θ₀), t ∈ {0.001, 0.01, 0.03, 0.1, 0.3, 1}, each contraction
#        from the last converged δ of the scan (first: δ at θ₀ tightened to 1e-12), tol 1e-10, cap
#        5000; ascending, stopping at the first t whose contraction fails. A start whose δ at θ₀ does
#        not converge within D0_CLEAN_CAP iterations is not scanned (reported).
#  (iii) Q at the production θ* (ext1 checkpoint) from three δ starts: δ* (its .bin), the logit δ̂
#        (tol 1e-10 cap 5000, as ext2/extended's first evaluation, then tol 1e-11 up to
#        D0_REEVAL_CAP, as the two-start re-evaluation) and a polluted δ (the δ a start's first trial
#        left, the one with the most coordinates at ±500), tol 1e-11 up to D0_REEVAL_CAP; with the
#        clamp counts and residuals of each. And whether μ at (θ*, 0) under ext2 and (θ*, 0, 0) under
#        extended is μ at θ* bit for bit (the stage explanation of the 4.6e-5 gap in E4).
#  (iv)  At a plateau (a contraction still running after STALL_PROBE evaluations, from the logit δ̂,
#        at each θ₀ and at θ*): the observations whose residual sits at |ln 0.95| = 0.051293, whether
#        their market's data inside share is at the 0.95 cap (DEMAND_TOTAL_SHARE_CAP), the model's
#        outside share there, and the clamp counts.
#
# Writes only DIAG_OUT/diag_ext1_d0_E<k>.json and its log. Environment: DIAG_ROUTINE (3 | 4), DIAG_OUT,
# DIAG_PARTS (default "i+ii+iii+iv"; '+' because sbatch --export splits on commas), D0_CLEAN_CAP
# (10000), D0_REEVAL_CAP (20000). Run through blp_diag_ext1_d0_job.sh.

isdefined(Main, :main_gpu_ift) || include(joinpath(@__DIR__, "blp_rc.jl"))
using Serialization, JSON3, LinearAlgebra, Statistics, CUDA, Random, Dates

const K        = parse(Int, ENV["DIAG_ROUTINE"])
const OUT      = ENV["DIAG_OUT"]
const PARTS    = Set(strip.(split(get(ENV, "DIAG_PARTS", "i+ii+iii+iv"), r"[+,]")))
const SPEC     = 12
const RD       = 2000
const SEED     = 42
const TOL_PROD = 1e-10        # production inner tolerance and cap (blp_stage_job.sh)
const CAP_PROD = 5000
const TOL_RE   = 1e-11        # the two-start re-evaluation's tolerance (reeval_tol)
const CAP_RE   = parse(Int, get(ENV, "D0_REEVAL_CAP", "20000"))
const TOL_CLEAN = 1e-12
const CAP_CLEAN = parse(Int, get(ENV, "D0_CLEAN_CAP", "10000"))
const STALL_PROBE = 300
const T_SCAN   = [0.001, 0.01, 0.03, 0.1, 0.3, 1.0]
const L095     = abs(log(0.95))
const CAP_SHARE = 0.95        # DEMAND_TOTAL_SHARE_CAP (sleep_demand_prep_link.py l.105)

stamp() = Dates.format(Dates.now(), "HH:MM:SS")
say(msg) = (println("[", stamp(), "] ", msg); flush(stdout))
finite_json(x::AbstractFloat) = isfinite(x) ? x : nothing
finite_json(x::AbstractVector) = Any[finite_json(v) for v in x]
finite_json(x::AbstractDict) = Dict{String,Any}(string(k) => finite_json(v) for (k, v) in x)
finite_json(x) = x
mkpath(OUT)
println("=" ^ 72)
println("blp_diag_ext1_d0: E$K, spec $SPEC, R $RD, parts $(sort(collect(PARTS)))")
println("=" ^ 72)

# ── The engine's setup for ext1, as in blp_diag_ext1_grad.jl ─────────────────────────────────────
in_dir, drw, out_dir = get_paths(true)
in_dirs = demand_search_dirs(in_dir, out_dir)
prefix  = demand_prefix(K, in_dirs)
prefix === nothing && error("no demand parquet for E$K in $in_dirs")
ENV["BLP_DELTA_SUFFIX"] = ""; ENV["BLP_OUTPUT_SUFFIX"] = ""; ENV["BLP_DEMAND_PREFIX"] = prefix
ENV["BLP_SIGMA_UB"] = get(ENV, "BLP_SIGMA_UB", "5.0"); ENV["BLP_PI_BOUND"] = get(ENV, "BLP_PI_BOUND", "5.0")
args = Dict{String,Any}("stage" => "ext1", "R" => RD, "seed" => SEED, "hpc" => true,
                        "local_dir" => nothing, "tol_inner" => TOL_PROD, "max_inner" => CAP_PROD)

nu_draws, draws_3d, key_index = load_precomputed_draws(drw, RD, SEED)
path = demand_parquet_path(K, SPEC, args)
df   = DataFrame(Parquet2.Dataset(path); copycols=true)
say("parquet: $(basename(path)), $(nrow(df)) rows")

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
n_pairs = length(pc.unique_pairs); n_times = length(pc.unique_times)
buf  = allocate_hot_buffers(N_obs, N_B, N_D, RD, n_pairs, n_times, coef_dim, length(pi_interactions))
precompute_pi_products!(buf, prod_vec, draws_3d, obs_key_idx, pi_interactions, coef_dim)
gbuf = allocate_gpu_buffers(buf, pc, N_obs, N_B, N_D, RD, n_pairs, n_times)
say("GPU: $(CUDA.name(CUDA.device()))")

lo = fill(-parse(Float64, ENV["BLP_PI_BOUND"]), n_params); hi = fill(parse(Float64, ENV["BLP_PI_BOUND"]), n_params)
lo[1:length(sigma_indices)] .= 0.0; hi[1:length(sigma_indices)] .= parse(Float64, ENV["BLP_SIGMA_UB"])
ln_s_data = [pc.b_mask[i] ? pc.ln_s_data_B_cond[i] : pc.ln_s_data_D[i] for i in 1:N_obs]
bidx = findall(pc.b_mask); didx = findall(pc.d_mask)

# Markets at the share cap: a B market's (pair's) data inside share Σ share_B_cond, a quarter's D
# inside share Σ share_D, equal to 0.95 (the floor on M binds there).
s_data = exp.(ln_s_data)
pair_in = zeros(n_pairs); for (k, i) in enumerate(bidx); pair_in[pc.b_mkt_idx[k]] += s_data[i]; end
time_in = zeros(n_times); for (k, i) in enumerate(didx); time_in[pc.d_time_enc[k]] += s_data[i]; end
pair_capped = abs.(pair_in .- CAP_SHARE) .< 1e-9
time_capped = abs.(time_in .- CAP_SHARE) .< 1e-9
obs_market = zeros(Int, N_obs); obs_capped = falses(N_obs)
for (k, i) in enumerate(bidx); obs_market[i] = pc.b_mkt_idx[k]; obs_capped[i] = pair_capped[pc.b_mkt_idx[k]]; end
for (k, i) in enumerate(didx); obs_market[i] = -pc.d_time_enc[k]; obs_capped[i] = time_capped[pc.d_time_enc[k]]; end
say("share cap: $(count(pair_capped)) of $n_pairs B markets and $(count(time_capped)) of $n_times quarters (D) " *
    "at Σ share = $CAP_SHARE; $(count(obs_capped)) observations in them")

# ── Helpers ────────────────────────────────────────────────────────────────────────────────────
set_mu!(t) = mu_at!(buf, prod_vec, nu_draws, t, sigma_indices, pi_interactions, RD, coef_dim)
nclamp(A) = mapreduce(x -> abs(x) >= 500 ? 1 : 0, +, A; init = 0)
function d0_Q(d)
    _, xi = estimate_theta1(d, pc)
    g = compute_gmm_moments(xi, pc)
    return dot(g, W * g), g
end
"""Contraction at θ (μ set here) in place on d."""
function d0_contract!(t, d; tol, cap)
    set_mu!(t)
    t0 = time()
    conv, it, hist = blp_contraction_gpu!(buf, gbuf, d, pc, RD; tol = tol, max_iter = cap)
    return conv, it, time() - t0, isempty(hist) ? NaN : hist[end]
end
"""At δ (μ already at θ): residuals ln s_data − ln max(s_model, 1e-15), δ and V at ±500, the model
outside share of each B market."""
function d0_state(d)
    compute_model_shares_gpu!(buf, gbuf, d, pc, RD)
    nV = nclamp(gbuf.V_B) + nclamp(gbuf.V_D)
    s  = collect_model_shares(buf, pc, N_obs)
    r  = ln_s_data .- log.(s)
    at = abs.(d) .>= 500
    s0 = Array(vec(sum(x -> exp(-x), gbuf.log_denom; dims = 2))) ./ RD
    rep = Dict{String,Any}("n_delta_at_clamp" => count(at),
        "n_delta_at_clamp_resid_ge_1e-9" => count(at .& (abs.(r) .>= 1e-9)),
        "max_resid_at_clamp" => any(at) ? maximum(abs.(r[at])) : 0.0,
        "n_delta_plus500" => count(d .>= 500), "n_delta_minus500" => count(d .<= -500),
        "max_resid" => maximum(abs.(r)), "n_V_at_clamp" => nV,
        "n_resid_at_ln095" => count(x -> abs(abs(x) - L095) < 1e-4, r))
    return rep, r, s, s0
end
"""The engine's exact IFT gradient at (θ, δ) (compute_ift_gradient_gpu!: ∂ln s/∂θ₂ on the GPU, the exact
Jacobian, non-finite rows to zero, the θ₁ projection), with its singular blocks. μ ends at θ."""
function d0_grad_at(t, d)
    set_mu!(t)
    t0 = time()
    dl = dlns_dtheta(() -> compute_model_shares_gpu!(buf, gbuf, d, pc, RD), t, buf, prod_vec, nu_draws,
                     sigma_indices, pi_interactions, RD, coef_dim, pc, N_obs)
    Jd, info = exact_delta_jacobian(d, buf.mu, dl, exact_groups(pc))
    bad = .!isfinite.(Jd)
    nbad = count(any(bad; dims = 2))
    Jd[bad] .= 0.0
    Q, gm = d0_Q(d)
    G = zeros(n_params)
    ift_gradient_from_jacobian!(G, Jd, d, gm, W, pc)
    return G, Q, Int(info["n_singular_blocks"]), nbad, time() - t0
end
"""max_i abs(g_i) over coordinates not held at a bound by their derivative (L-BFGS-B's projected
gradient with the distance-to-bound cap removed)."""
d0_free_sup(t, G) = maximum((t[i] <= lo[i] && G[i] > 0) || (t[i] >= hi[i] && G[i] < 0) ? 0.0 : abs(G[i])
                         for i in 1:n_params)
"""Entries of μ (N × R) that adding 0·P changes, P each extra π product (product column, demographic)
built on the fly as precompute_pi_products! builds it, and the count of non-finite P entries."""
function d0_mu_diff(mu::Matrix{Float64}, pv::Matrix{Float64}, dr::Array{Float64,3}, keyix::Vector{Int}, extra)
    ndiff = 0; nonfin = 0
    N, R = size(mu)
    for (cidx, didx_) in extra
        @inbounds for r in 1:R, i in 1:N
            p = pv[i, cidx] * dr[keyix[i], r, didx_]
            isfinite(p) || (nonfin += 1)
            m = mu[i, r]
            reinterpret(UInt64, m + 0.0 * p) == reinterpret(UInt64, m) || (ndiff += 1)
        end
    end
    return ndiff, nonfin
end
"""The plateau probe: a contraction from d0 at θ for STALL_PROBE evaluations; if still running, the
observations whose residual sits at |ln 0.95|, their market and its cap, the model outside share."""
function d0_plateau(t, d0)
    d = copy(d0)
    conv, it, _, nm = d0_contract!(t, d; tol = TOL_PROD, cap = STALL_PROBE)
    conv && return Dict{String,Any}("converged_within" => it)
    rep, r, s, s0 = d0_state(d)
    near = findall(x -> abs(abs(x) - L095) < 1e-4, r)
    mk = unique(obs_market[near])
    rows = Any[]
    for i in near[1:min(10, length(near))]
        m = obs_market[i]
        push!(rows, Dict("row" => i, "is_B" => pc.b_mask[i], "bank" => string(df.CodConglomeradoPrudencial[i]),
            "mca" => string(df.mca_code[i]), "time_id" => string(df.time_id[i]), "deposit_type" => dep_types[i],
            "s_data" => s_data[i], "s_model" => s[i], "delta" => d[i], "resid" => r[i],
            "market_capped" => obs_capped[i],
            "market_data_inside" => m > 0 ? pair_in[m] : time_in[-m],
            "market_model_outside" => m > 0 ? s0[m] : nothing))
    end
    rep["after_evals"] = it; rep["norm"] = nm
    rep["n_at_ln095_in_capped_market"] = count(obs_capped[near])
    rep["n_markets_at_ln095"] = length(mk)
    rep["n_markets_at_ln095_capped"] = count(m -> m > 0 ? pair_capped[m] : time_capped[-m], mk)
    rep["model_outside_share_there_max"] = isempty(mk) || all(<(0), mk) ? nothing : maximum(s0[m] for m in mk if m > 0)
    rep["top"] = rows
    return rep
end

# ── The production files (read only) ──────────────────────────────────────────────────────────
blp = blp_dir(out_dir)
chk = joinpath(blp, "blp_checkpoint_E$(K)_spec_$(SPEC)_ext1.jls")
ck  = deserialize(chk)
θstar = clamp.(collect(Float64, ck["theta2_star"]), lo, hi)
δstar = load_delta_bin(replace(chk, r"\.jls$" => ".bin"))
(δstar === nothing || length(δstar) != N_obs) && (δstar = collect(Float64, ck["delta_star"]))
length(δstar) == N_obs || error("$chk: δ* of length $(length(δstar)), data $N_obs")
resj = joinpath(blp, "blp_results_E$(K)_spec_$(SPEC)_ext1.json")
prodQ = isfile(resj) ? Float64(JSON3.read(read(resj, String))[:Q_value]) : NaN
δlogit = nothing; δlogit_src = ""
for c in (joinpath(in_dir, "logit_delta_E$(K)_spec_$(SPEC).bin"), joinpath(logit_dir(out_dir), "logit_delta_E$(K)_spec_$(SPEC).bin"))
    global δlogit, δlogit_src
    if δlogit === nothing && isfile(c)
        d = load_delta_bin(c)
        (d !== nothing && length(d) == N_obs) && (δlogit = d; δlogit_src = c)
    end
end
δlogit === nothing && error("no logit δ̂ of length $N_obs in data/input or data/output/logit")
say("θ* (ext1 checkpoint) = $(round.(θstar, digits = 6)); production Q $(prodQ); logit δ̂ from $δlogit_src")

cold = clamp.(randn(MersenneTwister(SEED), n_params) .* 0.01, lo, hi)
extra, extra_src = extra_starts("ext1", n_params, lo, hi)
starts = Any[("cold", cold)]
for s in extra; push!(starts, (s[1], s[2])); end
say("starts: cold + $(length(extra)) from $extra_src")

res = Dict{String,Any}("routine" => K, "parquet" => basename(path), "n_obs" => N_obs, "parts" => sort(collect(PARTS)),
    "checkpoint" => chk, "theta_star" => θstar, "Q_production" => prodQ, "logit_delta" => δlogit_src,
    "box" => Dict("lo" => lo, "hi" => hi), "tol_prod" => TOL_PROD, "cap_prod" => CAP_PROD,
    "tol_reeval" => TOL_RE, "cap_reeval" => CAP_RE, "cap_clean" => CAP_CLEAN,
    "n_markets_capped" => count(pair_capped), "n_quarters_capped" => count(time_capped),
    "n_obs_in_capped" => count(obs_capped))

# ── (i) the starts' first evaluation and first trial; (iv) the plateau at θ₀ ───────────────────────
first_eval = Dict{String,Any}()
deltas_after_trial = Dict{String,Vector{Float64}}()
deltas_at_start    = Dict{String,Vector{Float64}}()
if "i" in PARTS
    for (name, θ0) in starts
        say("(i) start $name: θ₀ = $(round.(θ0, digits = 5))")
        d0 = copy(δlogit)
        c0, it0, sec0, nm0 = d0_contract!(θ0, d0; tol = TOL_PROD, cap = CAP_PROD)
        st0, _, _, _ = d0_state(d0)
        G0, Q0, ns0, nb0, sj0 = d0_grad_at(θ0, d0)
        θ1 = clamp.(θ0 .- G0, lo, hi)
        say("    θ₀: conv $c0 in $it0 ($(round(sec0, digits = 0)) s, last norm $(round(nm0, sigdigits = 4))), Q $(round(Q0, sigdigits = 8)), " *
            "|∇Q|₂ $(round(norm(G0), sigdigits = 4)), free sup $(round(d0_free_sup(θ0, G0), sigdigits = 4)), singular $ns0; " *
            "first trial $(round.(θ1, digits = 4)) (sup move $(round(maximum(abs.(θ1 .- θ0)), sigdigits = 4)))")
        d1 = copy(d0)
        c1, it1, sec1, nm1 = d0_contract!(θ1, d1; tol = TOL_PROD, cap = CAP_PROD)
        st1, _, _, _ = d0_state(d1)
        G1, Q1, ns1, nb1, sj1 = d0_grad_at(θ1, d1)
        say("    first trial: conv $c1 in $it1 ($(round(sec1, digits = 0)) s, last norm $(round(nm1, sigdigits = 4))), Q $(round(Q1, sigdigits = 8)), " *
            "singular $ns1, δ at ±500 $(st1["n_delta_at_clamp"]) ($(st1["n_delta_at_clamp_resid_ge_1e-9"]) with residual), V at ±500 $(st1["n_V_at_clamp"])")
        rec = Dict{String,Any}("theta0" => θ0, "conv0" => c0, "iters0" => it0, "seconds0" => sec0, "last_norm0" => nm0,
            "Q0" => Q0, "grad0" => G0, "grad0_norm2" => norm(G0), "grad0_sup" => maximum(abs.(G0)),
            "grad0_free_sup" => d0_free_sup(θ0, G0), "singular0" => ns0, "nonfinite_rows0" => nb0,
            "seconds_jacobian0" => sj0, "state0" => st0,
            "trial1" => θ1, "trial1_sup_move" => maximum(abs.(θ1 .- θ0)), "trial1_norm2_move" => norm(θ1 .- θ0),
            "trial1_hits_box" => [θ1[i] == lo[i] || θ1[i] == hi[i] for i in 1:n_params],
            "conv1" => c1, "iters1" => it1, "seconds1" => sec1, "last_norm1" => nm1, "Q1" => Q1,
            "singular1" => ns1, "nonfinite_rows1" => nb1, "state1" => st1,
            "lbfgsb_r0_lite_scale" => min(1.0, 0.02 / max(d0_free_sup(θ0, G0), eps())))
        if "iv" in PARTS && !(c0 && it0 <= STALL_PROBE)
            rec["plateau0"] = d0_plateau(θ0, δlogit)
            p = rec["plateau0"]
            haskey(p, "after_evals") && say("    (iv) plateau at θ₀ after $(p["after_evals"]): $(p["n_resid_at_ln095"]) obs at |ln 0.95| " *
                "($(p["n_at_ln095_in_capped_market"]) in capped markets; $(p["n_markets_at_ln095"]) markets, $(p["n_markets_at_ln095_capped"]) capped), " *
                "δ at ±500 $(p["n_delta_at_clamp"])")
        end
        first_eval[name] = rec
        deltas_at_start[name] = d0
        deltas_after_trial[name] = d1
    end
    res["i_starts"] = first_eval
end

# ── (ii) Q along the first-step direction from a clean δ ────────────────────────────────────────────
if "ii" in PARTS && "i" in PARTS
    scans = Dict{String,Any}()
    for (name, θ0) in starts
        rec = first_eval[name]
        dθ = rec["trial1"] .- θ0
        dc = copy(deltas_at_start[name])
        cc, itc, secc, nmc = d0_contract!(θ0, dc; tol = TOL_CLEAN, cap = CAP_CLEAN)
        Qc, _ = d0_Q(dc)
        sc = Dict{String,Any}("clean_conv" => cc, "clean_iters" => itc, "clean_seconds" => secc,
                              "clean_last_norm" => nmc, "Q0_clean" => Qc,
                              "linear_slope" => dot(rec["grad0"], dθ), "points" => Any[])
        say("(ii) start $name: δ at θ₀ to $TOL_CLEAN: conv $cc in $itc; Q $(round(Qc, sigdigits = 10))")
        if cc
            dprev = copy(dc)
            for t in T_SCAN
                θt = clamp.(θ0 .+ t .* dθ, lo, hi)
                dt = copy(dprev)
                ct, itt, sect, nmt = d0_contract!(θt, dt; tol = TOL_PROD, cap = CAP_PROD)
                Qt, _ = d0_Q(dt)
                stt, _, _, _ = d0_state(dt)
                push!(sc["points"], Dict("t" => t, "theta" => θt, "sup_move" => maximum(abs.(θt .- θ0)),
                    "conv" => ct, "iters" => itt, "seconds" => sect, "Q" => Qt, "dQ" => Qt - Qc,
                    "dQ_linear" => t * sc["linear_slope"], "n_delta_at_clamp" => stt["n_delta_at_clamp"],
                    "n_V_at_clamp" => stt["n_V_at_clamp"]))
                say("    t=$t (sup move $(round(maximum(abs.(θt .- θ0)), sigdigits = 3))): conv $ct in $itt, " *
                    "dQ $(round(Qt - Qc, sigdigits = 5)) (linear $(round(t * sc["linear_slope"], sigdigits = 5)))")
                ct || break
                dprev = dt
            end
        else
            sc["skipped"] = "δ at θ₀ did not converge to $TOL_CLEAN within $CAP_CLEAN iterations from the logit δ̂"
        end
        scans[name] = sc
    end
    res["ii_scan"] = scans
end

# ── (iii) Q at θ* from three δ starts; the stage test ───────────────────────────────────────────────
if "iii" in PARTS
    th = Dict{String,Any}()
    function at_star(label, dstart; two_step::Bool)
        d = copy(dstart)
        out = Dict{String,Any}()
        if two_step
            c, it, sec, nm = d0_contract!(θstar, d; tol = TOL_PROD, cap = CAP_PROD)
            Qp, _ = d0_Q(d)
            out["prod"] = Dict("conv" => c, "iters" => it, "seconds" => sec, "last_norm" => nm, "Q" => Qp,
                               "dQ_vs_production" => Qp - prodQ, "state" => d0_state(d)[1])
            say("(iii) θ* from $label, tol $TOL_PROD cap $CAP_PROD: conv $c in $it, Q $(round(Qp, sigdigits = 10)) " *
                "(minus production $(round(Qp - prodQ, sigdigits = 4)))")
        end
        c, it, sec, nm = d0_contract!(θstar, d; tol = TOL_RE, cap = CAP_RE)
        Qr, _ = d0_Q(d)
        st = d0_state(d)[1]
        out["reeval"] = Dict("conv" => c, "iters" => it, "seconds" => sec, "last_norm" => nm, "Q" => Qr,
                             "dQ_vs_production" => Qr - prodQ, "max_abs_delta_minus_star" => maximum(abs.(d .- δstar)),
                             "state" => st)
        say("(iii) θ* from $label, tol $TOL_RE cap $CAP_RE: conv $c in $it, Q $(round(Qr, sigdigits = 10)) " *
            "(minus production $(round(Qr - prodQ, sigdigits = 4))), max|δ − δ*| $(round(maximum(abs.(d .- δstar)), sigdigits = 4)), " *
            "δ at ±500 $(st["n_delta_at_clamp"]) ($(st["n_delta_at_clamp_resid_ge_1e-9"]) with residual)")
        return out
    end
    th["from_delta_star"] = at_star("δ*", δstar; two_step = false)
    th["from_logit"]      = at_star("the logit δ̂", δlogit; two_step = true)
    if !isempty(deltas_after_trial)
        pick = first(sort(collect(keys(deltas_after_trial)),
                          by = n -> (-count(x -> abs(x) >= 500, deltas_after_trial[n]),
                                     -first_eval[n]["state1"]["max_resid"])))
        th["polluted_from"] = "the δ the first trial of '$pick' left"
        th["from_polluted"] = at_star("the δ of $pick's first trial", deltas_after_trial[pick]; two_step = true)
    else
        th["from_polluted"] = "part (i) not run: no polluted δ"
    end
    # The stage explanation: μ at (θ*, 0) under ext2 and (θ*, 0, 0) under extended against μ at θ*.
    set_mu!(θstar)
    stage_rec = Dict{String,Any}()
    for stg in ("ext2", "extended")
        _, pis, _ = build_theta2_structure(stg)
        extra_pi = pis[length(pi_interactions)+1:end]
        same_first = pis[1:length(pi_interactions)] == pi_interactions
        ndiff, nonfin = d0_mu_diff(buf.mu, prod_vec, draws_3d, obs_key_idx, extra_pi)
        stage_rec[stg] = Dict("first_products_same_order" => same_first, "extra_products" => length(extra_pi),
                              "mu_entries_differing" => ndiff, "nonfinite_extra_products" => nonfin)
        say("(iii) stage $stg: its first $(length(pi_interactions)) π products are ext1's in order: $same_first; " *
            "μ entries that adding 0·(the extra products) changes: $ndiff (non-finite products $nonfin)")
    end
    th["stage_mu"] = stage_rec
    if "iv" in PARTS
        th["plateau_from_logit"] = d0_plateau(θstar, δlogit)
    end
    res["iii_theta_star"] = th
end

open(joinpath(OUT, "diag_ext1_d0_E$(K).json"), "w") do f; JSON3.write(f, finite_json(res)); end
say("wrote $(joinpath(OUT, "diag_ext1_d0_E$(K).json"))")

# ── Summary (what to paste back) ───────────────────────────────────────────────────────────────
r5(x) = x === nothing ? "-" : round(x, sigdigits = 5)
println("\n=== D0 SUMMARY E$K ===")
println("θ* $(round.(θstar, digits = 5)); production Q $(prodQ); capped: $(res["n_markets_capped"]) B markets, " *
        "$(res["n_quarters_capped"]) quarters, $(res["n_obs_in_capped"]) obs")
if haskey(res, "i_starts")
    println("(i) start | θ₀: conv/iters, Q, |∇Q|₂, free sup, sing | trial: sup move, conv/iters, sing, δ±500 (resid), V±500 | lite c")
    for (name, _) in starts
        p = res["i_starts"][name]
        println("  $(rpad(name, 6)) | $(p["conv0"])/$(p["iters0"]), $(r5(p["Q0"])), $(r5(p["grad0_norm2"])), $(r5(p["grad0_free_sup"])), $(p["singular0"]) | " *
                "$(r5(p["trial1_sup_move"])), $(p["conv1"])/$(p["iters1"]), $(p["singular1"]), $(p["state1"]["n_delta_at_clamp"]) " *
                "($(p["state1"]["n_delta_at_clamp_resid_ge_1e-9"])), $(p["state1"]["n_V_at_clamp"]) | $(r5(p["lbfgsb_r0_lite_scale"]))")
        if haskey(p, "plateau0") && haskey(p["plateau0"], "after_evals")
            q = p["plateau0"]
            println("         (iv) plateau after $(q["after_evals"]): $(q["n_resid_at_ln095"]) obs at |ln0.95|, $(q["n_at_ln095_in_capped_market"]) in capped markets " *
                    "($(q["n_markets_at_ln095_capped"])/$(q["n_markets_at_ln095"]) markets capped), model outside share there max $(r5(q["model_outside_share_there_max"]))")
        end
    end
end
if haskey(res, "ii_scan")
    println("(ii) dQ along the first step (t: dQ [conv]) from a clean δ")
    for (name, _) in starts
        sc = res["ii_scan"][name]
        println("  $(rpad(name, 6)) clean $(sc["clean_conv"])/$(sc["clean_iters"]): " *
                (haskey(sc, "skipped") ? sc["skipped"] :
                 join(["$(p["t"]): $(r5(p["dQ"])) [$(p["conv"] ? "c" : "X")]" for p in sc["points"]], ", ")))
    end
end
if haskey(res, "iii_theta_star")
    th = res["iii_theta_star"]
    for key in ("from_delta_star", "from_logit", "from_polluted")
        v = th[key]; v isa AbstractDict || (println("(iii) $key: $v"); continue)
        rr = v["reeval"]
        println("(iii) θ* $key: " * (haskey(v, "prod") ? "prod conv $(v["prod"]["conv"])/$(v["prod"]["iters"]) dQ $(r5(v["prod"]["dQ_vs_production"])); " : "") *
                "reeval conv $(rr["conv"])/$(rr["iters"]) Q $(rr["Q"]) dQ $(r5(rr["dQ_vs_production"])) " *
                "δ±500 $(rr["state"]["n_delta_at_clamp"]) ($(rr["state"]["n_delta_at_clamp_resid_ge_1e-9"]) with residual, max $(r5(rr["state"]["max_resid_at_clamp"])))")
    end
    haskey(th, "polluted_from") && println("      polluted δ: $(th["polluted_from"])")
    for stg in ("ext2", "extended")
        s = th["stage_mu"][stg]
        println("(iii) stage $stg: μ entries differing $(s["mu_entries_differing"]) (first products same order $(s["first_products_same_order"]))")
    end
    if haskey(th, "plateau_from_logit") && haskey(th["plateau_from_logit"], "after_evals")
        q = th["plateau_from_logit"]
        println("(iv) θ* from logit δ̂ after $(q["after_evals"]): $(q["n_resid_at_ln095"]) obs at |ln0.95|, $(q["n_at_ln095_in_capped_market"]) in capped markets")
    end
end
println("=== END D0 E$K ===")
