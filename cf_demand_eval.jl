"""
cf_demand_eval.jl
===================
Foundation 0a for the counterfactual pipeline: a demand-evaluation + composite
elasticity module that REUSES the estimator's exact share kernels so that
counterfactual shares are identical to the in-sample BLP shares.

What it provides
----------------
  * `build_cf_context(...)`  — load the demand parquet, the precomputed
    draws, and the estimated (θ̂₁, θ̂₂, δ̂) for a routine/stage; assemble the
    same `Precomp`/`HotBuffers`/`prod_vec` the estimator builds.
  * `cf_model_shares(ctx)`   — in-sample active shares s^Act_jkmt (validation gate).
  * `cf_shares_at(ctx, ρ′)`  — active shares under a COUNTERFACTUAL spread vector ρ′,
    holding ξ̂ and all other characteristics fixed. δ′ = δ̂ + α̂·(ρ′ − ρ̂) and μ is
    recomputed for ρ′ (the spread enters the random coefficient, coef index 1).
    This is the building block reused by cf_deposit_sim.jl / CF1 / CF2.
  * `cf_dDep_dρ(ctx; ...)`   — composite deposit semi-elasticity ∂Dep/∂ρ via finite
    differences on `cf_shares_at`, scaled by the deposit law of motion (1−φ)·M.

Design notes
------------
  * REUSE, do not reimplement: we `include("blp_engine_cpu.jl")` (CPU baseline,
    no CUDA) and call its `compute_mu!`, `compute_model_shares!`, `build_precomp`,
    `build_regressor_matrices`, `project_endogenous_spreads`, `estimate_theta1`,
    `build_theta2_structure`, `unpack_theta2`, `load_precomputed_draws`,
    `precompute_pi_products!`. The CPU path is portable to a laptop; the GPU path is
    only needed for the heavy CF2 forward simulation (see bbl_fwd_sim.jl).
  * Spread convention (must match the estimator exactly): ρ = spread_ann / 100
    (bps → percentage points). θ̂₁[1] = α̂ is the mean spread coefficient.
  * s^Act: for B-firms `buf.s_B` is the CONDITIONAL within-market share
    (model s^Act,B); for D-firms `buf.s_D` is the unconditional national share
    (model s^Act,D). See eqs (13-B)/(13-D) and (16-B)/(16-D) in the proposal.
  * MEMORY: peak RAM is dominated by `buf.mu` (N×R) and the π-products
    (n_pi × N×R). For E4 "extended" (n_pi=6) at R=2000 this is tens of GB → cluster.
    At R≈200–500 it is a few GB → laptop-feasible, but shares are then a
    Monte-Carlo approximation of the R=2000 estimates (use only for development /
    magnitudes, NOT for the exact-reproduction validation gate).

Usage
-----
  # Local quick-look (approximate, low memory):
  julia --project=. --threads=4 cf_demand_eval.jl --estim 6 --spec 12 \\
      --stage extended --R 200 --seed 42

  # Cluster exact (matches the estimated shares):
  julia --project=\${PROJECT_DIR} --threads=8 cf_demand_eval.jl --estim 6 \\
      --spec 12 --stage extended --R 2000 --seed 42 --hpc

Outputs `shares_elas_E{estim}_spec_{spec}_{stage}{suffix}.parquet` into the counterfactual
step directory (`cf_out_dir`: cluster `data/output/counterfactuals`, local `CF_FOUNDATION/`).
"""

using Parquet2, DataFrames, Serialization, Statistics, LinearAlgebra, ArgParse
import JSON3   # bbl_transitions.json (market-state AR(1)s) — see the evolving-states block below

# Load CUDA + the GPU share engine ONLY when the CF opts in (CF_GPU!=0). CPU-only steps (cost2,
# cost_solve, cf1, the Jacobi init/merge) set CF_GPU=0 so they never import CUDA — critical on the
# cluster, where many concurrent CPU jobs sharing one NFS depot would otherwise stampede the Julia
# precompile/load lock just to load a package they don't use.
#
# blp_engine_cpu.jl (the CPU kernels & constants: X_COLS, get_paths, log_status, …) is pulled in
# exactly once on either branch — via the engine when we take the GPU path, directly otherwise. Do
# NOT also include it unconditionally above: that would load it twice on the GPU branch (the engine
# loads it too), guarded only by the engine's isdefined check. The CPU-branch load goes through
# `Base.include(Main, …)` — the same thing a top-level `include` expands to — so the two branches
# don't read as two static edges to the same file.
const _CF_GPU_REQUESTED = get(ENV, "CF_GPU", "1") != "0"
if _CF_GPU_REQUESTED
    # blp_gpu_engine brings in CUDA + the GPU share kernels AND the CPU baseline. It LOADS on a
    # no-GPU machine (verified): CUDA.functional()==false and we fall back to the CPU share path.
    include(joinpath(@__DIR__, "blp_engine_gpu.jl"))
    import CUDA
else
    Base.include(Main, joinpath(@__DIR__, "blp_engine_cpu.jl"))
end
# Use the GPU share kernel only when opted in AND a GPU is actually present. The `&&` short-circuits,
# so CUDA is never referenced when CF_GPU=0 (and thus need not be imported).
const _CF_USE_GPU = _CF_GPU_REQUESTED && (try CUDA.functional() catch; false end)

# The data-layout authority for the CF/BBL stack (`cf_out_dir`, `cf_in_dir`, `blp_dir`,
# `logit_dir`, `demand_search_dirs`, `is_cluster_out`) is of_root.jl, reached here through
# blp_engine_cpu.jl.

# ==========================================================================
# Evolving market states: the demographics the forward simulation walks through
# ==========================================================================
"""
    StateEvolution

The per-(market key, demographic) deviation from that MCA's own long-run level, in the SAME
scaled space the draws live in, plus the per-demographic AR(1) persistence. Together they
give the horizon-h demographic of a market as a pure SHIFT of its h=0 value:

    d_{m,h} = dbar_m + rho_d^h (d_{m,0} - dbar_m)
    shift_{m,d}(h) = d_{m,h} - d_{m,0} = (rho_d^h - 1) * dev_{m,d},   dev = d_{m,0} - dbar_m

`dev` is indexed by the DRAW-ROW index (the `obs_key_idx` a panel row gathers with), one row
per (mca_code, time_id) market key, so a horizon-h shift is a lookup into a matrix that is
built ONCE — it is identical across every deviation, every rate path and every period, which
is why nothing in this struct is recomputed inside the BBL deviation loop. Draw rows the
panel never gathers stay 0.

WHY A SHIFT AND NOT A REBUILD. The draws are market-specific draws around the market mean
(`generate_demographic_draws`: d_r = mu_m + sigma_m * eps_r), so moving the mean moves all R
draws of that market by the same amount. The r-dimension is untouched, which is what makes
the evaluator below exact and nearly free (see `cf_state_dmu!`).

SCALING. `load_precomputed_draws` divides each demographic dimension by its cross-market sd
and centres it, so pi is "utility per 1-sd unit of demographic". A deviation taken from the
demand parquet is in the parquet's own units, so it must be divided by that same sd or the
shift is silently sd_d times too large. `inv_sd[d]` is that 1/sd_d. It is CALIBRATED against
the loaded draws rather than recomputed: the pre-normalisation draws are a 14 GB array and
the load-time sd is not returned, while one regression of the draws' per-key mean on the
parquet column recovers 1/sd_d to ~1e-5 relative from a subsample.

`rho[d] == 1.0` marks a dimension with no usable transition: (rho^h - 1) == 0, so that
demographic stays frozen at its h=0 value and costs nothing.
"""
struct StateEvolution
    dev     ::Matrix{Float64}   # (n_draw_rows x D) scaled deviation from the MCA's own long-run level
    rho     ::Vector{Float64}   # (D) within-MCA AR(1) persistence; 1.0 = frozen dimension
    inv_sd  ::Vector{Float64}   # (D) parquet units -> scaled draw units
    used    ::Vector{Int}       # demographic dims that actually enter mu through a pi interaction
    src     ::String            # transitions file the rho's came from (logged, not re-read)
end

const CF_STATE_JSON = "bbl_transitions.json"

# Evolving states are ON by default: the user chose full consistency between the demand block
# and the state block over the cheap frozen-state approximation. `CF_EVOLVING_STATES=0` gets the
# frozen run back for the cost comparison and to bisect a regression.
_cf_states_on() = get(ENV, "CF_EVOLVING_STATES", "1") != "0"

"""
    _state_transitions_path(out_dir) -> String

Where `bbl_transitions.json` lives. It is bbl_transitions.py's ~8 kB output, so on the cluster
it travels as an UPLOAD into `data/input` (`cf_in_dir`) rather than being reproduced there;
locally it sits in the COST_FWD step folder that produced it. The upload wins, matching
`load_forward_rf`. The `cf_in_dir` spelling is returned when neither exists so the error names
the location the file has to be staged to.
"""
function _state_transitions_path(out_dir)
    for d in (cf_in_dir(out_dir, "COST_FWD"), cf_out_dir(out_dir, "COST_FWD"))
        p = joinpath(d, CF_STATE_JSON)
        isfile(p) && return p
    end
    return joinpath(cf_in_dir(out_dir, "COST_FWD"), CF_STATE_JSON)
end

"""
    _calibrate_inv_sd(draws_3d, keys, mu, d, cap) -> (slope, r2)

OLS slope of the draws' per-key mean on the demand parquet's column for demographic `d`,
over an evenly spaced subsample of at most `cap` market keys.

The loaded draws are (mu_k + sigma_k*eps)/sd_d - centre_d, so the mean over r of key k is
mu_k/sd_d - centre_d plus a Monte-Carlo error of sigma_k/(sd_d*sqrt(R)). That error is
uncorrelated with mu_k, so the slope is 1/sd_d and the intercept -centre_d; only the slope is
needed because a DEVIATION kills the centre. Subsampling is enough: at R=2000 and 20k keys the
slope's relative error is ~1e-5, four orders below the shift it scales.

`r2` is the guard, not a diagnostic. Any affine re-scaling of the parquet column between draw
generation and now is absorbed by the slope and leaves r2 at 1, so an r2 that is NOT near 1
means the column is not the mean the draws were built from and the caller falls back to the
draws' own per-key mean rather than scaling the wrong series. "Near 1" has to be read against
R: the within-market draw noise costs r2 exactly mean(sigma_k^2)/(R*var(mu)), which at R=2000
is 4e-5 for fraction_65plus (the noisiest demographic that enters a pi interaction) but 1.3e-3
at R=64, so a fixed 0.999 would reject a perfectly good column on a low-R development run. The
caller's threshold scales with R for that reason.
"""
function _calibrate_inv_sd(draws_3d::Array{Float64,3}, keys::Vector{Int},
                           mu::Vector{Float64}, d::Int, cap::Int)
    n    = length(keys)
    step = max(1, cld(n, max(cap, 1)))
    sel  = keys[1:step:n]
    R    = size(draws_3d, 2)
    xb = 0.0; zb = 0.0
    z  = Vector{Float64}(undef, length(sel))
    @inbounds for (j, k) in enumerate(sel)
        s = 0.0
        for r in 1:R; s += draws_3d[k, r, d]; end
        z[j] = s / R
        zb += z[j]; xb += mu[k]
    end
    m = length(sel)
    m < 3 && return (NaN, NaN)
    xb /= m; zb /= m
    sxx = 0.0; sxz = 0.0; szz = 0.0
    @inbounds for (j, k) in enumerate(sel)
        dx = mu[k] - xb; dz = z[j] - zb
        sxx += dx * dx; sxz += dx * dz; szz += dz * dz
    end
    (sxx > 0.0 && szz > 0.0) || return (NaN, NaN)
    return (sxz / sxx, (sxz * sxz) / (sxx * szz))
end

"""
    build_state_evolution(df, draws_3d, obs_key_idx, pi_interactions, coef_dim; kwargs...)
        -> Union{Nothing,StateEvolution}

Assemble the horizon shift for the market states from `bbl_transitions.json`. Returns
`nothing` — meaning "demographics frozen at the launch quarter", the pre-existing behaviour —
when the flag is off, when the stage has no demographic pi interaction (no share can move with
a state, so there is nothing to do), or when the transitions file is absent and `require` is
false.

Keywords:
  * `out_dir` / `transitions_path` : where to find the JSON (`_state_transitions_path`).
  * `enabled`  : the OFF switch, defaulting to `CF_EVOLVING_STATES != 0`.
  * `require`  : error instead of falling back when the JSON is missing. `build_cf_context`
    passes `hpc`, so a cluster run cannot quietly become a frozen-state run — the fallback
    would change the model without changing anything visible in the job log.
  * `rho_field`: which persistence to read, `CF_STATE_RHO_FIELD`. The default `rho` is the raw
    within-MCA slope. `rho_nickell_corrected` is available for a robustness pass but is >= 1
    for three of the five states (cadunico 1.021, fraction_65plus 1.018, fraction_young 1.018),
    which over a 50-quarter horizon is an explosive demographic; the 0 < rho < 1 guard below
    freezes any such dimension rather than letting it diverge.

ONLY the demographics that enter mu are touched: `used` is the set of pi-interaction
demographic indices, at most 4 of the 8 D_COLS for the `extended` stage. The rest cannot move
a share whatever their transition says, so they are neither calibrated nor stored.
"""
function build_state_evolution(df::DataFrame, draws_3d::Array{Float64,3},
                               obs_key_idx::Vector{Int},
                               pi_interactions::Vector{Tuple{Int,Int}},
                               coef_dim::Int;
                               out_dir=nothing,
                               transitions_path::Union{Nothing,String}=nothing,
                               enabled::Bool=_cf_states_on(),
                               require::Bool=false,
                               rho_field::String=get(ENV, "CF_STATE_RHO_FIELD", "rho"),
                               calib_keys::Int=20_000)
    if !enabled
        log_status("  [CF-STATE] OFF (CF_EVOLVING_STATES=0): demographics frozen at each row's launch quarter")
        return nothing
    end
    D_dim = size(draws_3d, 3)
    used  = sort(unique([didx for (cidx, didx) in pi_interactions
                         if cidx <= coef_dim && didx <= D_dim]))
    if isempty(used)
        log_status("  [CF-STATE] OFF: no demographic pi interaction in this stage, so no share responds to a state")
        return nothing
    end

    path = transitions_path !== nothing ? transitions_path :
           (out_dir === nothing ?
            error("build_state_evolution needs `out_dir` or `transitions_path`") :
            _state_transitions_path(out_dir))
    if !isfile(path)
        msg = "Market-state transitions not found at:\n    $path\n" *
              "This is bbl_transitions.py's ~8 kB JSON; on the cluster it is an UPLOAD into data/input."
        require && error(msg * "\nEvolving states are ON, so this is fatal: falling back to frozen " *
                               "demographics would change the model without changing the log.")
        @warn "$msg\n  -> FALLING BACK to FROZEN demographics (the h=0 state held over the horizon)."
        return nothing
    end
    ms = get(JSON3.read(read(path, String)), :market_states, nothing)
    if ms === nothing
        require && error("$path has no `market_states` block.")
        @warn "$(basename(path)) has no `market_states` block -> FROZEN demographics."
        return nothing
    end

    rho = ones(D_dim)
    for d in used
        e = get(ms, Symbol(D_COLS[d]), nothing)
        e === nothing && continue
        r = get(e, Symbol(rho_field), nothing)
        (r isa Real && isfinite(r) && 0.0 < Float64(r) < 1.0) || continue
        rho[d] = Float64(r)
    end

    # ── One representative panel row and one MCA per DRAW row ────────────────────────────────
    # The draws are keyed (mca_code, time_id); the deviation is against the MCA's own level, so
    # the draw rows have to be grouped back up to the MCA. Both maps come from the panel itself,
    # which is also what guarantees the parquet value read below is the mu that key's draws were
    # generated from.
    n_rows = size(draws_3d, 1)
    N      = nrow(df)
    mca    = string.(df.mca_code)
    seen   = falses(n_rows)
    row_of = zeros(Int, n_rows)
    gid    = zeros(Int, n_rows)
    mca_id = Dict{String,Int}()
    @inbounds for i in 1:N
        k = obs_key_idx[i]
        seen[k] && continue
        seen[k] = true; row_of[k] = i
        gid[k]  = get!(mca_id, mca[i], length(mca_id) + 1)
    end
    kept = findall(seen)
    G    = length(mca_id)

    dev    = zeros(n_rows, D_dim)
    inv_sd = ones(D_dim)
    r2s    = fill(NaN, D_dim)
    from_draws = Int[]
    mu = zeros(n_rows)
    for d in used
        name  = D_COLS[d]
        use_parquet = name in names(df)
        if use_parquet
            v = Float64.(coalesce.(df[!, name], NaN))
            @inbounds for k in kept; mu[k] = v[row_of[k]]; end
            use_parquet = all(isfinite, @view(mu[kept]))
        end
        if use_parquet
            s, r2 = _calibrate_inv_sd(draws_3d, kept, mu, d, calib_keys)
            r2s[d] = r2
            # Scale-aware acceptance: reject a column that is not an affine image of the draw
            # mean, without rejecting one whose r2 is only dented by the R draws' own noise.
            r2_min = max(0.90, 1.0 - 25.0 / size(draws_3d, 2))
            use_parquet = isfinite(s) && s != 0.0 && isfinite(r2) && r2 > r2_min
            use_parquet && (inv_sd[d] = s)
        end
        if !use_parquet
            # No usable parquet column for this demographic: take the market level straight from
            # the draws' own per-key mean, which is ALREADY in the scaled space (inv_sd = 1). It
            # carries the sigma_k/sqrt(R) Monte-Carlo error of the within-market draws, which is a
            # few percent of the across-quarter movement for gdp_per_capita, so it is the fallback
            # and not the default.
            R = size(draws_3d, 2)
            @inbounds for k in kept
                s = 0.0
                for r in 1:R; s += draws_3d[k, r, d]; end
                mu[k] = s / R
            end
            inv_sd[d] = 1.0
            push!(from_draws, d)
        end
        # dbar_m = the MCA's mean over the quarters the panel observes — the same object the
        # within transform in bbl_transitions.py demeans by, so rho and dbar refer to one level.
        gs = zeros(G); gn = zeros(Int, G)
        @inbounds for k in kept; g = gid[k]; gs[g] += mu[k]; gn[g] += 1; end
        @inbounds for k in kept
            dev[k, d] = (mu[k] - gs[gid[k]] / gn[gid[k]]) * inv_sd[d]
        end
    end

    log_status("  [CF-STATE] evolving demographics ON <- $(basename(path)) (rho_field=$rho_field, " *
               "$(length(kept)) market keys, $G MCAs)")
    for d in used
        hl  = rho[d] < 1.0 ? string(round(log(0.5) / log(rho[d]), digits=1)) : "frozen"
        dv  = @view dev[kept, d]
        s50 = abs.((rho[d]^50 - 1.0) .* dv)
        log_status("    d=$d $(rpad(D_COLS[d], 26)) rho=$(rpad(round(rho[d], digits=4), 6)) " *
                   "half-life=$(rpad(hl, 6))q  1/sd=$(round(inv_sd[d], sigdigits=5))" *
                   (d in from_draws ? " [from draws]" : " [r2=$(round(r2s[d], digits=6))]") *
                   "  dev sd=$(round(std(dv), sigdigits=4))  |shift| h=50: " *
                   "mean=$(round(mean(s50), sigdigits=4)) p95=$(round(quantile(s50, 0.95), sigdigits=4))")
    end
    return StateEvolution(dev, rho, inv_sd, used, path)
end

"""
    state_shift(ev, h) -> Matrix (n_draw_rows x D)

The explicit per-(market key, demographic) shift at horizon `h`, in scaled draw units:
`(rho_d^h - 1) * dev`. Zero at `h == 0` by construction, in every column.

This is the shift tensor in materialised form — it allocates, so it is for inspection and for
the numerical gates, not for the deviation loop. `cf_state_dmu!` consumes the same `(rho, dev)`
without ever forming it.
"""
function state_shift(ev::StateEvolution, h::Int)::Matrix{Float64}
    S = zeros(size(ev.dev))
    h == 0 && return S
    @inbounds for d in ev.used
        f = ev.rho[d]^h - 1.0
        f == 0.0 && continue
        @views S[:, d] .= f .* ev.dev[:, d]
    end
    return S
end

# ==========================================================================
# Context: everything needed to evaluate (counterfactual) shares
# ==========================================================================
struct CFDemandCtx
    estim          ::Int
    spec_id        ::Int
    stage          ::String
    R              ::Int
    coef_dim       ::Int
    df             ::DataFrame
    pc             ::Precomp
    buf            ::HotBuffers
    prod_vec       ::Matrix{Float64}      # col 1 = ρ̂ (spread_ann/100), cols 2:end = X
    nu_draws       ::Matrix{Float64}
    draws_3d       ::Array{Float64,3}
    obs_key_idx    ::Vector{Int}
    sigma_indices  ::Vector{Int}
    pi_interactions::Vector{Tuple{Int,Int}}
    theta1         ::Vector{Float64}
    theta2         ::Vector{Float64}
    delta_hat      ::Vector{Float64}
    rho_hat        ::Vector{Float64}      # in-sample spread (= prod_vec[:,1])
    alpha          ::Float64              # θ̂₁[1], mean spread coefficient
    gbuf           ::Any                          # GPU share buffers (nothing ⇒ CPU share path; typed Any so the struct loads without CUDA when CF_GPU=0)
    state_ev       ::Union{Nothing,StateEvolution} # evolving market states; nothing ⇒ demographics frozen at each row's launch quarter
end

"""
    _result_path(out_dir, estim, spec, stage, suffix) -> String

Path of the serialized RC result Dict the counterfactuals load.

On the cluster the RC step is the sole producer and writes one file per (routine, spec,
stage) into the BLP step folder, so the path is fully determined:
`blp_dir(out_dir)/blp_results_E{estim}_spec_{spec}_{stage}{suffix}.jls`. Resolving through a
single location is the point — a second candidate is exactly how the CF stack and the
exporter end up reading two different vintages of the same routine.

Locally `out_dir` is the `BLP_RESULTS/` root, and for the final `extended` stage the system
of record is the consolidated `cluster_processed/blp_E{estim}_spec_{spec}.jls` that the
ingest writes: a byte-identical copy of the cluster's `extended` result (same Julia Dict
schema — δ̂, θ̂₁, θ̂₂, Q), with the per-routine metadata `.json` and `INDEX.json` alongside it.
The three candidates cover the layouts an ingest may have produced under `BLP_RESULTS/`;
first existing wins, and the flat name is the fallback. Any other stage — an intermediate
kept in `cluster_raw/` — resolves to the flat name directly.

The logit stage has no RC result Dict; `build_cf_context` loads it in a separate branch from
`logit_dir(out_dir)`.
"""
function _result_path(out_dir, estim, spec_id, stage, suffix)
    flat = joinpath(blp_dir(out_dir),
                    "blp_results_E$(estim)_spec_$(spec_id)_$(stage)$(suffix).jls")
    if !is_cluster_out(out_dir) && stage == "extended"
        leaf = "blp_E$(estim)_spec_$(spec_id).jls"
        for c in (joinpath(out_dir, "cluster_processed", leaf),
                  joinpath(out_dir, "BLP_RESULTS", "cluster_processed", leaf),
                  joinpath(out_dir, "BLP_RESULTS", leaf))
            isfile(c) && return c
        end
    end
    return flat
end

"""
    collect_shares(buf, pc, N) -> Vector{Float64}

CPU analogue of the GPU `collect_model_shares`: gather `buf.s_B`/`buf.s_D`
(filled by `compute_model_shares!`) into an N-vector in original row order.
"""
function collect_shares(buf::HotBuffers, pc::Precomp, N::Int)::Vector{Float64}
    s = Vector{Float64}(undef, N)
    b = 0; d = 0
    @inbounds for i in 1:N
        if pc.b_mask[i]
            b += 1; s[i] = max(buf.s_B[b], 1e-15)
        else
            d += 1; s[i] = max(buf.s_D[d], 1e-15)
        end
    end
    return s
end

# ==========================================================================
# Routine → demand-parquet mapping (AUTO-DISCOVERED, new 8-routine scheme)
# ==========================================================================
"""
    discover_demand_parquet(input_dirs, estim, spec_id) -> String

Find the demand parquet for routine `estim`, spec `spec_id`, by scanning
`demand_<estim>[_<tag>]_spec_<spec>.parquet` (tag optional; excludes `_final`), newest mtime
wins.

The tag is optional because E1/E2 write the bare `demand_{estim}_spec_{spec}.parquet` while
E3/E4 carry one; the same regex is used by blp_logit.jl and blp_rc.jl, so the whole
stack sees the same set of files for a routine.

This mirrors the BLP side's auto-discovery (the note: "the routine list — id AND
prefix — is AUTO-DISCOVERED … newest mtime per id wins so leftover old-scheme files
can't shadow rebuilds"). Discovery is by GLOB, never the static `DEMAND_PREFIXES`
dict, whose entries go stale the moment a routine is relabelled. The lineup:
  E1 LocalB · E2 PooledLinear ·
  E3 Single-Index (headline) · E4 Single-Index+Time.
The cluster default set is {3, 4} (the link routines).

`input_dirs` is the search path from `demand_search_dirs` (of_root.jl): on the cluster the
single producer directory `data/output/demand_prep`, locally the single
ESTIMATION_OUTPUT/DEMAND_PREP. The FIRST directory holding a match settles the routine and
the mtime contest then runs within it; with one entry per tree the two rules coincide.
"""
function discover_demand_parquet(input_dirs::Vector{String}, estim::Int, spec_id::Int)::String
    pat = Regex("^demand_$(estim)(?:_.*)?_spec_$(spec_id)\\.parquet\$")
    for input_dir in input_dirs
        isdir(input_dir) || continue
        cands = String[]
        for f in readdir(input_dir)
            (occursin(pat, f) && !occursin("final", lowercase(f))) &&
                push!(cands, joinpath(input_dir, f))
        end
        isempty(cands) && continue
        path = cands[argmax(mtime.(cands))]
        length(cands) > 1 && log_status("  [CF] $(length(cands)) parquets for E$estim; using newest: $(basename(path))")
        return path
    end
    error(
        "No demand parquet for estim=$estim spec=$spec_id. Searched:\n" *
        describe_search_dirs(input_dirs...) * "\n" *
        "(scanned demand_$(estim)[_<tag>]_spec_$(spec_id).parquet, excluding _final). " *
        "New 8-routine scheme is auto-discovered from the parquet; ensure the routine's " *
        "demand-prep parquet exists (the sleep/BLP side produces it).")
end

# ==========================================================================
# Build the evaluation context (mirrors run_blp_estimation_ift_gpu assembly)
# ==========================================================================
"""
    build_cf_context(estim, spec_id, stage; R, seed, hpc, local_dir, suffix) -> CFDemandCtx

Replicates the estimator's data/precomp assembly and loads the saved (θ̂₁, θ̂₂, δ̂)
for `(estim, spec_id, stage)`. Requires the demand input parquet (local) and the
draws + result `.jls` (downloaded from the cluster).
"""
function build_cf_context(estim::Int, spec_id::Int, stage::String;
                          R::Int=2000, seed::Int=42, hpc::Bool=false,
                          local_dir=nothing,
                          time_filter::Union{Nothing,Vector{String}}=nothing,
                          keep::Union{Nothing,BitVector}=nothing,
                          placeholder::Bool=false,
                          draws_dir_override::Union{Nothing,String}=nothing,
                          evolving_states::Bool=_cf_states_on(),
                          suffix::String=get(ENV, "BLP_OUTPUT_SUFFIX", ""))
    input_dir, draws_dir, out_dir = get_paths(hpc; local_dir=local_dir)
    # new 8-routine scheme (auto-discover; not DEMAND_PREFIXES) over the one directory the
    # prep step writes to, so the CF stack reads the same parquet the BLP was estimated on.
    path = discover_demand_parquet(demand_search_dirs(input_dir, out_dir), estim, spec_id)

    df_full = DataFrame(Parquet2.Dataset(path); copycols=true)
    nrow(df_full) == 0 && error("Empty demand parquet: $path")
    N_full = nrow(df_full)

    # ── Optional subsample (KEEPS δ̂ aligned): build a row mask over the FULL
    # parquet order so we can subset both df and the saved δ̂ identically. Use for
    # local-dev memory relief (one quarter, or a market fraction). The share
    # aggregation is exact on the kept rows as long as whole (mca,time) markets are
    # kept together — `time_filter` does this; arbitrary masks may split markets. ──
    row_keep = keep === nothing ? trues(N_full) : copy(keep)
    length(row_keep) == N_full ||
        error("keep mask length $(length(row_keep)) ≠ parquet rows $N_full")
    if time_filter !== nothing
        tids = string.(df_full.time_id)
        row_keep .&= [t in time_filter for t in tids]
    end
    df = all(row_keep) ? df_full : df_full[row_keep, :]
    log_status("  [CF] Loaded $(nrow(df))/$N_full obs from $(basename(path))" *
               (all(row_keep) ? "" : "  [subsampled]"))

    sigma_indices, pi_interactions, _ = build_theta2_structure(stage)
    coef_dim = 1 + L_PROD
    N_obs    = nrow(df)

    # ── Draws + obs→draw mapping (identical to the estimator) ───────────────
    dd = draws_dir_override === nothing ? draws_dir : draws_dir_override
    nu_draws, draws_3d, key_index = load_precomputed_draws(dd, R, seed)
    mca_codes = string.(df.mca_code)
    time_ids  = string.(df.time_id)
    pad_idx   = size(draws_3d, 1)
    obs_key_idx = [get(key_index, (mca_codes[i], time_ids[i]), pad_idx) for i in 1:N_obs]

    # ── prod_vec: col 1 = ρ (spread_ann/100), cols 2:1+L_PROD = X_COLS ──────
    prod_vec = zeros(N_obs, coef_dim)
    prod_vec[:, 1] .= coalesce.(df.spread_ann, 0.0) ./ 100.0
    for (i, col) in enumerate(X_COLS)
        col in names(df) && (prod_vec[:, 1+i] .= coalesce.(df[!, col], 0.0))
    end

    # ── Precomp (shares aggregation structure) ──────────────────────────────
    spread_cols, x_mat, Z_mat, iv_cols = build_regressor_matrices(df)
    iv_avail = [c for c in iv_cols
                if std(replace(coalesce.(df[!, c], 0.0), Inf=>0.0, -Inf=>0.0)) > 1e-10]
    Z = zeros(N_obs, length(iv_avail))
    for (i, c) in enumerate(iv_avail)
        v = coalesce.(df[!, c], 0.0); replace!(v, Inf=>0.0, -Inf=>0.0); Z[:, i] .= v
    end
    H          = hcat(x_mat, Z_mat)
    dep_types  = Int.(coalesce.(df.deposit_type, 0))
    spread_hat = project_endogenous_spreads(spread_cols, H, dep_types)
    X_full     = hcat(spread_cols, x_mat)
    X_hat      = hcat(spread_hat,  x_mat)
    valid      = BitVector(all.(isfinite, eachrow(X_hat)))
    clusters   = string.(df.CodConglomeradoPrudencial)
    pc         = build_precomp(df, Z, X_full, X_hat, valid, clusters)

    N_B = sum(pc.b_mask); N_D = sum(pc.d_mask)
    n_pairs = length(pc.unique_pairs); n_times = length(pc.unique_times)
    buf = allocate_hot_buffers(N_obs, N_B, N_D, R, n_pairs, n_times, coef_dim,
                               length(pi_interactions))
    precompute_pi_products!(buf, prod_vec, draws_3d, obs_key_idx,
                            pi_interactions, coef_dim)
    save_pi_base!(buf, pi_interactions, 1)   # ρ̂ spread-products snapshot → cf_shares_at restore-by-copy (B.1)
    # GPU share buffers (H200 on the cluster; nothing on a CPU-only machine ⇒ CPU share path).
    gbuf = _CF_USE_GPU ? allocate_gpu_buffers(buf, pc, N_obs, N_B, N_D, R, n_pairs, n_times) : nothing
    gbuf === nothing || log_status("  [CF] GPU share kernel enabled ($(CUDA.name(CUDA.device())))")

    # ── Load estimated parameters & δ̂ for this stage ───────────────────────
    # PLACEHOLDER mode (smoke test before the BLP finishes): no RC result needed —
    # set θ₂=0, θ₁=0, and δ = log-share init (the estimator's own fallback, lines
    # 357–361 of blp_engine_cpu.jl). This exercises the WHOLE pipeline
    # (parquet→pc→buf→μ→shares→deposits→ψ→CF1/CF2) on real local data with draws,
    # catching plumbing bugs now; swap in δ̂/θ̂ once results land.
    rpath = _result_path(out_dir, estim, spec_id, stage, suffix)
    if !placeholder && stage == "logit"
        # LOGIT stage: δ̂ is saved as logit_delta_E{e}_spec_{s}{suffix}.bin
        # (there is no RC-style result dict). θ₂ is empty ⇒ μ=0 ⇒ plain logit shares.
        # θ₁ is only needed for SPREAD counterfactuals; pull α from the 'full' logit
        # sub-model if present, else 0 (in-sample share reproduction is unaffected).
        # input_dir (data/input on the cluster) comes first so a δ̂ supplied as an upload wins
        # over one the logit step produced; otherwise it is the logit step folder, which is the
        # only place blp_logit.jl writes.
        dbin  = joinpath(input_dir, "logit_delta_E$(estim)_spec_$(spec_id)$(suffix).bin")
        isfile(dbin) || (dbin = joinpath(logit_dir(out_dir), "logit_delta_E$(estim)_spec_$(spec_id)$(suffix).bin"))
        dfull = load_delta_bin(dbin)
        dfull === nothing && error("Missing logit δ̂ bin: $dbin")
        length(dfull) == N_full ||
            error("logit δ̂ length $(length(dfull)) ≠ parquet rows $N_full")
        delta_hat = all(row_keep) ? dfull : dfull[row_keep]
        theta2 = Float64[]
        theta1 = zeros(coef_dim)
        lpath  = joinpath(logit_dir(out_dir), "logit_E$(estim)_full_spec_$(spec_id)$(suffix).jls")
        if isfile(lpath)
            lr = deserialize(lpath); t1 = get(lr, "theta1", nothing)
            (t1 !== nothing && !isempty(t1)) && (theta1[1] = Float64(t1[1]))
        else
            @warn "No logit full sub-model ($(basename(lpath))) — α=0 (in-sample shares OK; spread CFs disabled)."
        end
    elseif placeholder || !isfile(rpath)
        placeholder || @warn "RC result missing ($(basename(rpath))) — using PLACEHOLDER params (θ₂=0, δ=log-share init). Results are NOT valid until the BLP finishes."
        n_params  = length(sigma_indices) + length(pi_interactions)
        theta1    = zeros(coef_dim)
        theta2    = zeros(n_params)
        delta_hat = zeros(N_obs)
        delta_hat[pc.d_mask] .= pc.ln_s_data_D[pc.d_mask]
        delta_hat[pc.b_mask] .= pc.ln_s_data_B_cond[pc.b_mask]
    else
        res = deserialize(rpath)
        theta1 = Float64.(res["theta1"])
        # The logit-stage result carries no random coefficients → θ₂ may be absent
        # or empty; treat it as the zero-RC case (shares reduce to plain logit).
        theta2 = Float64.(get(res, "theta2", Float64[]))
        haskey(res, "delta") || error("Result $(basename(rpath)) has no 'delta' field.")
        delta_full = Float64.(res["delta"])
        length(delta_full) == N_full ||
            error("δ̂ length $(length(delta_full)) ≠ parquet rows $N_full — results/parquet mismatch")
        delta_hat = all(row_keep) ? delta_full : delta_full[row_keep]
    end
    # Sanity: the saved θ₂ structure must match the LOCAL build_theta2_structure($stage), because
    # θ₂ is unpacked POSITIONALLY (unpack_theta2 slices by these index lists). If the saved result
    # was estimated under a different σ/π layout than the local code builds, every μ — and hence
    # every share, deposit path, ψ and cost — is silently wrong with nothing failing. This is a
    # hard error, not a warning: a structure mismatch means the numbers are invalid, and the fix is
    # to re-sync the code (or rebuild the sysimage) so build_theta2_structure matches the run that
    # produced the result. (Only checked when a real RC result was loaded; `res` is undefined in the
    # logit/placeholder branches, whose θ₂ is empty/zeroed by construction.)
    if @isdefined(res)
        saved_sig = get(res, "sigma_indices", nothing)
        saved_pi  = get(res, "pi_interactions", nothing)
        if saved_sig !== nothing && collect(saved_sig) != collect(sigma_indices)
            error("θ₂ σ-structure mismatch for stage '$stage': result has sigma_indices=" *
                  "$(collect(saved_sig)) but local build_theta2_structure gives $(sigma_indices). " *
                  "Re-sync blp_engine_cpu.jl / rebuild the sysimage to match the run that wrote " *
                  "$(basename(rpath)).")
        end
        if saved_pi !== nothing && collect(Tuple.(saved_pi)) != collect(Tuple.(pi_interactions))
            error("θ₂ π-structure mismatch for stage '$stage': result has pi_interactions=" *
                  "$(collect(Tuple.(saved_pi))) but local build_theta2_structure gives " *
                  "$(pi_interactions). θ₂ is unpacked positionally, so this would silently " *
                  "mis-map the random coefficients. Re-sync the code / rebuild the sysimage.")
        end
        exp_len = length(sigma_indices) + length(pi_interactions)
        length(theta2) == exp_len ||
            error("θ₂ length $(length(theta2)) ≠ $(exp_len) implied by the local structure for " *
                  "stage '$stage' — results/code mismatch in $(basename(rpath)).")
    end

    # Evolving market states. Built HERE, once per context, because the shift it carries is the
    # same for every deviation, every rate path and every period of the forward simulation — the
    # only place it can be hoisted out of all three loops at once.
    #
    # A missing transitions file is a WARNING here, not an error, because most holders of a
    # context (cf1, cf3, cf4, cost_solve) never advance a state and would be broken by a hard
    # requirement on a file they do not read. The step that DOES consume it asserts instead —
    # `assert_state_evolution`, called by bbl_fwd_sim.jl's psi_under — so the BBL forward
    # simulation still cannot quietly become a frozen-state run.
    state_ev = build_state_evolution(df, draws_3d, obs_key_idx, pi_interactions, coef_dim;
                                     out_dir=out_dir, enabled=evolving_states)

    return CFDemandCtx(estim, spec_id, stage, R, coef_dim, df, pc, buf, prod_vec,
                       nu_draws, draws_3d, obs_key_idx, sigma_indices, pi_interactions,
                       theta1, theta2, delta_hat, copy(prod_vec[:, 1]), theta1[1], gbuf,
                       state_ev)
end

# ==========================================================================
# Share evaluation
# ==========================================================================
# Dispatch the share aggregation to the GPU kernel (H200) when a GPU context is present,
# else the CPU kernel. Both consume ctx.buf.mu (filled by compute_mu! on CPU) and write
# results into ctx.buf.s_B / ctx.buf.s_D (Float64), so callers/collect_shares are unchanged.
@inline function _cf_model_shares!(ctx::CFDemandCtx, delta::Vector{Float64})
    if ctx.gbuf === nothing
        compute_model_shares!(ctx.buf, delta, ctx.pc, ctx.R)
    else
        compute_model_shares_gpu!(ctx.buf, ctx.gbuf, delta, ctx.pc, ctx.R)
    end
end

"""
    cf_model_shares(ctx) -> Vector{Float64}

In-sample active shares s^Act at (δ̂, θ̂₂). Used by the validation gate to check
that this module reproduces the estimator's shares.
"""
function cf_model_shares(ctx::CFDemandCtx)::Vector{Float64}
    sv, pv = unpack_theta2(ctx.theta2, ctx.sigma_indices, ctx.pi_interactions)
    compute_mu!(ctx.buf, ctx.prod_vec, ctx.nu_draws, sv, ctx.sigma_indices, pv,
                ctx.R, ctx.coef_dim)
    _cf_model_shares!(ctx, ctx.delta_hat)
    return collect_shares(ctx.buf, ctx.pc, nrow(ctx.df))
end

# ── B.2 + B.1 helpers: incremental Pi-product maintenance for cf_shares_at ─────────────────────────
# These operate on `HotBuffers` (defined in the included blp_engine_cpu.jl) but are pure CF/BBL
# machinery — kept here, next to their only caller cf_shares_at, so the demand-estimation engine
# carries nothing beyond the inert `pi_base` field. A spread deviation perturbs ONLY prod_vec[:,
# target_cidx] (the spread, cidx==1), so only the Pi products whose product-char index is target_cidx
# change; the rest are invariant. refresh_pi_products_cidx! rebuilds just those, IN PLACE (no realloc
# → no GC churn), instead of rebuilding the full 72 GB set on every deviation.
function refresh_pi_products_cidx!(buf::HotBuffers, prod_vec::Matrix{Float64},
                                   stacked_draws::Array{Float64,3}, obs_key_idx::Vector{Int},
                                   pi_interactions::Vector{Tuple{Int,Int}}, coef_dim::Int,
                                   target_cidx::Int)
    D_dim = size(stacked_draws, 3)
    @inbounds for (pi_idx, (cidx, didx)) in enumerate(pi_interactions)
        (cidx == target_cidx && cidx <= coef_dim && didx <= D_dim) || continue
        pp = buf.pi_products[pi_idx]
        size(pp, 1) == 0 && continue
        pp .= @view(prod_vec[:, cidx]) .* @view(stacked_draws[obs_key_idx, :, didx])   # in place
    end
end

# Save the current (ρ̂) spread-interacting Pi products so cf_shares_at can RESTORE them by copy
# instead of recomputing (B.1). Call once after the initial precompute, at ρ̂.
function save_pi_base!(buf::HotBuffers, pi_interactions::Vector{Tuple{Int,Int}}, target_cidx::Int)
    buf.pi_base = [ (c == target_cidx && length(buf.pi_products) >= i && size(buf.pi_products[i], 1) > 0) ?
                    copy(buf.pi_products[i]) : zeros(0, 0)
                    for (i, (c, _)) in enumerate(pi_interactions) ]
end

"""
    cf_shares_at(ctx, rho_new) -> Vector{Float64}

Active shares under a counterfactual spread vector `rho_new` (same units as
ρ = spread_ann/100), holding ξ̂ and all non-spread characteristics fixed.

  δ′ = δ̂ + α̂·(ρ′ − ρ̂)      (linear mean-utility channel)
  μ  recomputed with ρ′      (spread is coef index 1 in the random coefficient)

`rho_new` must be length N (one spread per jkmt row). For the deposit simulator,
only the endogenous types k=4,5 will differ from ρ̂; regulated types are passed
through unchanged.

`delta_shift` adds a per-row constant to δ′. Its one caller is the evolving-states path
(`cf_shares_at_h`), where the horizon-h demographic shift lands in δ rather than in μ; see
`cf_state_dmu!` for why the two are the same object here.
"""
function cf_shares_at(ctx::CFDemandCtx, rho_new::Vector{Float64};
                      delta_shift::Union{Nothing,AbstractVector{Float64}}=nothing)::Vector{Float64}
    length(rho_new) == nrow(ctx.df) || error("rho_new length ≠ N_obs")
    # A spread change perturbs ONLY prod_vec[:, 1], so only the cidx==1 Pi products change. Refresh
    # just those, IN PLACE (B.2) — the non-spread products are invariant, and this avoids rebuilding
    # (and reallocating) the full 72 GB set on every deviation.
    ctx.prod_vec[:, 1] .= rho_new
    refresh_pi_products_cidx!(ctx.buf, ctx.prod_vec, ctx.draws_3d, ctx.obs_key_idx,
                              ctx.pi_interactions, ctx.coef_dim, 1)
    sv, pv = unpack_theta2(ctx.theta2, ctx.sigma_indices, ctx.pi_interactions)
    compute_mu!(ctx.buf, ctx.prod_vec, ctx.nu_draws, sv, ctx.sigma_indices, pv,
                ctx.R, ctx.coef_dim)
    delta_cf = ctx.delta_hat .+ ctx.alpha .* (rho_new .- ctx.rho_hat)
    delta_shift === nothing || (delta_cf .+= delta_shift)
    _cf_model_shares!(ctx, delta_cf)
    s = collect_shares(ctx.buf, ctx.pc, nrow(ctx.df))
    # Restore the in-sample spread + its Pi products so the context is reusable. Restore the spread
    # products by COPY from the ρ̂ snapshot (B.1) — no recompute — falling back to a refresh only if
    # the snapshot is absent (a context built before save_pi_base!).
    ctx.prod_vec[:, 1] .= ctx.rho_hat
    if length(ctx.buf.pi_base) == length(ctx.pi_interactions)
        @inbounds for (i, (c, _)) in enumerate(ctx.pi_interactions)
            (c == 1 && size(ctx.buf.pi_base[i], 1) > 0) || continue
            ctx.buf.pi_products[i] .= ctx.buf.pi_base[i]
        end
    else
        refresh_pi_products_cidx!(ctx.buf, ctx.prod_vec, ctx.draws_3d, ctx.obs_key_idx,
                                  ctx.pi_interactions, ctx.coef_dim, 1)
    end
    return s
end

# ==========================================================================
# Shares at a horizon: the demand block reading the same states as the state block
# ==========================================================================
"""
    build_state_evolution(ctx; kwargs...) -> Union{Nothing,StateEvolution}

Rebuild the state evolution for an existing context — the switch for a frozen-vs-evolving
comparison (`build_state_evolution(ctx; enabled=false)`) without paying for a second context.
`out_dir` is required here because the context does not carry it.
"""
build_state_evolution(ctx::CFDemandCtx; kwargs...) =
    build_state_evolution(ctx.df, ctx.draws_3d, ctx.obs_key_idx,
                          ctx.pi_interactions, ctx.coef_dim; kwargs...)

"""
    assert_state_evolution(ctx)

Error when `CF_EVOLVING_STATES` asks for evolving market states but the context has none —
the transitions JSON was missing, or the stage carries no demographic π interaction.

`build_cf_context` only warns, because most contexts never advance a state. The steps that do
call this, so that a run whose whole point is the evolving-state path cannot finish as a
frozen-state run with nothing but a warning to show for it. The OFF switch is the environment
variable: `CF_EVOLVING_STATES=0` satisfies this check by design.
"""
function assert_state_evolution(ctx::CFDemandCtx)
    (ctx.state_ev !== nothing || !_cf_states_on()) && return nothing
    error("Evolving market states are ON (CF_EVOLVING_STATES != 0) but this context has none. " *
          "Either stage $(CF_STATE_JSON) where `_state_transitions_path` looks (on the cluster " *
          "that is data/input — see the [CF-STATE] warning above for the exact path), or set " *
          "CF_EVOLVING_STATES=0 to run the frozen-state model deliberately.")
end

"""
    cf_state_dmu!(out, ctx, ev, h, rho_row) -> out

The per-row utility shift the horizon-h demographics produce, written into `out` (length N).

WHY THIS IS EXACT AND NOT AN APPROXIMATION. Demographics reach the shares only through
μ = Σ_c prod_vec[:,c]·(Π d_mt + Σ ν_i)_c, i.e. only through the π products
`prod_vec[:,cidx] .* draws[key, :, didx]`. A horizon-h state is the h=0 draw plus a shift that
depends on (market, demographic) and NOT on the draw r, so

    μ_h[n,r] = μ_0[n,r] + Σ_i π_i · prod_vec[n,cidx_i] · shift_h[key_n, didx_i]

and the correction is constant across r. `compute_model_shares!` uses μ only as
V = δ[n] + μ[n,r], so an r-constant addition to μ is the same object as an addition to δ.
That is what this function returns. The consequences are the point of the design:

  * no π product is rebuilt and no draw array is mutated — the O(n_pi·N·R) refresh and the
    O(N·R·coef_dim) μ build both drop out of the per-period loop;
  * the cost of moving one period forward is O(N·n_pi), ~3e6 flops against the ~1e9 of the
    share kernel it feeds;
  * at h=0 the factor (ρ_d^0 − 1) is exactly 0, so the frozen result is reproduced bit for
    bit rather than to a tolerance.

`rho_row` is the period's spread vector, passed explicitly rather than read from
`ctx.prod_vec[:,1]`: the spread column is restored to ρ̂ between `cf_shares_at` calls, so
reading it here would silently price the demographic interaction at ρ̂ instead of ρ_t.
"""
function cf_state_dmu!(out::Vector{Float64}, ctx::CFDemandCtx, ev::StateEvolution,
                       h::Int, rho_row::AbstractVector{Float64})
    fill!(out, 0.0)
    h == 0 && return out
    _, pv = unpack_theta2(ctx.theta2, ctx.sigma_indices, ctx.pi_interactions)
    D_dim = size(ctx.draws_3d, 3)
    N     = length(out)
    @inbounds for (i, (cidx, didx)) in enumerate(ctx.pi_interactions)
        (cidx <= ctx.coef_dim && didx <= D_dim) || continue
        c = pv[i] * (ev.rho[didx]^h - 1.0)
        c == 0.0 && continue
        dcol = @view ev.dev[:, didx]
        if cidx == 1
            for n in 1:N; out[n] += c * rho_row[n] * dcol[ctx.obs_key_idx[n]]; end
        else
            for n in 1:N; out[n] += c * ctx.prod_vec[n, cidx] * dcol[ctx.obs_key_idx[n]]; end
        end
    end
    return out
end

"""
    cf_shares_at_h(ctx, rho_new, ev, h; work=nothing) -> Vector{Float64}

`cf_shares_at` with the market states advanced `h` quarters from each row's own launch
quarter. `ev === nothing` or `h == 0` is the frozen call, unchanged. `work` is an optional
length-N scratch vector so a per-period loop allocates nothing.
"""
function cf_shares_at_h(ctx::CFDemandCtx, rho_new::Vector{Float64},
                        ev::Union{Nothing,StateEvolution}, h::Int;
                        work::Union{Nothing,Vector{Float64}}=nothing)::Vector{Float64}
    (ev === nothing || h == 0) && return cf_shares_at(ctx, rho_new)
    dmu = work === nothing ? zeros(nrow(ctx.df)) : work
    cf_state_dmu!(dmu, ctx, ev, h, rho_new)
    return cf_shares_at(ctx, rho_new; delta_shift=dmu)
end

"""
    cf_shares_path(ctx, spreads_ann, ev; T=50) -> Vector{Float64} | Matrix{Float64}

The active share the forward simulation should use in each of periods 1…T at the STATIONARY
spread vector `spreads_ann`.

Returns the plain length-N share vector when `ev === nothing` — the frozen state is constant
over t, so there is one share and the existing hoist stands. With evolving states it returns
an N×T matrix, because the share now moves with the demographics even at a fixed spread.

THE EXPENSIVE PART IS STILL EVALUATED ONCE. The spread does not change over t, so μ is built
once for the whole path; every period after that is one share aggregation over a δ that
differs by the r-constant `cf_state_dmu!` shift. Nothing here depends on r^f either, so the
matrix is hoisted out of the rate-path loop exactly as the single share vector was — P rate
paths still cost 1× the share kernel, now T times rather than once.
"""
function cf_shares_path(ctx::CFDemandCtx, spreads_ann::Vector{Float64},
                        ev::Union{Nothing,StateEvolution}; T::Int=50)
    ev === nothing && return cf_shares_at(ctx, spreads_ann)
    N = nrow(ctx.df)
    length(spreads_ann) == N || error("spreads_ann length ≠ N_obs")

    ctx.prod_vec[:, 1] .= spreads_ann
    refresh_pi_products_cidx!(ctx.buf, ctx.prod_vec, ctx.draws_3d, ctx.obs_key_idx,
                              ctx.pi_interactions, ctx.coef_dim, 1)
    sv, pv = unpack_theta2(ctx.theta2, ctx.sigma_indices, ctx.pi_interactions)
    compute_mu!(ctx.buf, ctx.prod_vec, ctx.nu_draws, sv, ctx.sigma_indices, pv,
                ctx.R, ctx.coef_dim)

    base = ctx.delta_hat .+ ctx.alpha .* (spreads_ann .- ctx.rho_hat)
    S    = zeros(N, T)
    dmu  = zeros(N)
    dl   = zeros(N)
    for t in 1:T
        cf_state_dmu!(dmu, ctx, ev, t, spreads_ann)
        dl .= base .+ dmu
        _cf_model_shares!(ctx, dl)
        S[:, t] .= collect_shares(ctx.buf, ctx.pc, N)
    end

    # Restore ρ̂ and its π products, same contract as cf_shares_at: the context is reusable.
    ctx.prod_vec[:, 1] .= ctx.rho_hat
    if length(ctx.buf.pi_base) == length(ctx.pi_interactions)
        @inbounds for (i, (c, _)) in enumerate(ctx.pi_interactions)
            (c == 1 && size(ctx.buf.pi_base[i], 1) > 0) || continue
            ctx.buf.pi_products[i] .= ctx.buf.pi_base[i]
        end
    else
        refresh_pi_products_cidx!(ctx.buf, ctx.prod_vec, ctx.draws_3d, ctx.obs_key_idx,
                                  ctx.pi_interactions, ctx.coef_dim, 1)
    end
    return S
end

"""
    cf_dDep_dρ(ctx; eps=1e-4, market_size=nothing, phi=nothing) -> NamedTuple

Composite deposit semi-elasticity ∂Dep^Act/∂ρ at the observed equilibrium, via a
UNIFORM finite-difference perturbation of all spreads by `eps` (one extra forward
pass). Returns the share derivative `ds_dρ = (s(ρ̂+eps) − s(ρ̂))/eps` and, when
`market_size` (M_mt proxy) and `phi` (φ̂_mt) are supplied, the deposit semi-elasticity
`dDep_dρ = (1−φ)·M·ds_dρ` per the law of motion (eq 8).

NOTE: this is the response to a *uniform* spread shift, which folds in within-market
cross-substitution; it is the right object for aggregate pass-through intuition but
is NOT the pure own-spread Jacobian diagonal. The exact own/cross Jacobian is built
inside CF2 by perturbing each strategy and re-simulating (see bbl_fwd_sim.jl).
"""
function cf_dDep_dρ(ctx::CFDemandCtx; eps::Float64=1e-4,
                    market_size=nothing, phi=nothing)
    s0 = cf_model_shares(ctx)
    s1 = cf_shares_at(ctx, ctx.rho_hat .+ eps)
    ds = (s1 .- s0) ./ eps
    dDep = nothing
    if market_size !== nothing && phi !== nothing
        dDep = (1.0 .- phi) .* market_size .* ds
    end
    return (s0=s0, ds_dρ=ds, dDep_dρ=dDep)
end

# ==========================================================================
# Export (validation artifact)
# ==========================================================================
function cf_export_shares(ctx::CFDemandCtx; out_path::String)
    s = cf_model_shares(ctx)
    out = DataFrame(
        CodConglomeradoPrudencial = string.(ctx.df.CodConglomeradoPrudencial),
        mca_code      = string.(ctx.df.mca_code),
        time_id       = string.(ctx.df.time_id),
        deposit_type  = Int.(coalesce.(ctx.df.deposit_type, 0)),
        is_B          = Bool.(coalesce.(ctx.df.is_B, false)),
        spread        = ctx.rho_hat,
        s_act_model   = s,
    )
    # Carry the data-implied shares when present, for the validation comparison.
    "share_B_cond" in names(ctx.df) && (out.share_B_cond = coalesce.(ctx.df.share_B_cond, NaN))
    "share_D"      in names(ctx.df) && (out.share_D      = coalesce.(ctx.df.share_D, NaN))
    mkpath(dirname(out_path))
    Parquet2.writefile(out_path, out)
    log_status("  [CF] Wrote $(nrow(out)) rows → $(basename(out_path))")
    return out_path
end

# ==========================================================================
# CLI
# ==========================================================================
function _parse_cf_args()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--estim";     arg_type = Int;    default = 6
        "--spec";      arg_type = Int;    default = 12
        "--stage";     arg_type = String; default = "extended"
        "--R";         arg_type = Int;    default = 2000
        "--seed";      arg_type = Int;    default = 42
        "--hpc";       action   = :store_true
        "--local-dir"; arg_type = String; default = nothing
        "--draws-dir"; arg_type = String; default = nothing   # override draws location
        "--suffix";    arg_type = String; default = ""
        "--verify-gpu"; action  = :store_true                 # assert GPU shares == CPU shares, then exit
    end
    return parse_args(s)
end

"""
    verify_cf_gpu(ctx; rtol) — assert the GPU share kernel matches the CPU one at (δ̂, θ̂₂).

Run once on the cluster (`cf_demand_eval.jl … --verify-gpu`) to gate the GPU path. Errors if
the max relative / log-floor share difference exceeds `rtol`. No-op (warns) without a GPU.
"""
function verify_cf_gpu(ctx::CFDemandCtx; rtol::Float64=1e-7)
    ctx.gbuf === nothing && (@warn "  [CF] --verify-gpu: no GPU present (CUDA.functional()==false) — nothing to verify"; return)
    sv, pv = unpack_theta2(ctx.theta2, ctx.sigma_indices, ctx.pi_interactions)
    compute_mu!(ctx.buf, ctx.prod_vec, ctx.nu_draws, sv, ctx.sigma_indices, pv, ctx.R, ctx.coef_dim)
    verify_gpu_shares(ctx.buf, ctx.gbuf, ctx.delta_hat, ctx.pc, ctx.R; rtol=rtol)  # errors on mismatch
    log_status("  [CF] --verify-gpu PASSED: GPU shares == CPU shares to rtol=$rtol")
end

function main_cf_demand()
    a = _parse_cf_args()
    log_status("CF demand-eval — E$(a["estim"]) spec $(a["spec"]) stage $(a["stage"]) " *
               "| R=$(a["R"]) seed=$(a["seed"]) hpc=$(a["hpc"])")
    ctx = build_cf_context(a["estim"], a["spec"], a["stage"];
                           R=a["R"], seed=a["seed"], hpc=a["hpc"],
                           local_dir=a["local-dir"], suffix=a["suffix"],
                           draws_dir_override=a["draws-dir"])
    if a["verify-gpu"]; verify_cf_gpu(ctx); return; end
    _, _, out_dir = get_paths(a["hpc"]; local_dir=a["local-dir"])
    cf_dir = cf_out_dir(out_dir)                       # cluster: data/output/counterfactuals
    out_path = joinpath(cf_dir,
        "shares_elas_E$(a["estim"])_spec_$(a["spec"])_$(a["stage"])$(a["suffix"]).parquet")
    cf_export_shares(ctx; out_path=out_path)
    log_status("[DONE] CF demand-eval")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_cf_demand()
end
