"""
bbl_fwd_sim.jl
=================
BBL Step 2, part 1: forward-simulate the value-function basis ψ under the
EQUILIBRIUM strategy σ̂ and under a battery of DEVIATING strategies σ̃, then export
the firm-level ψ's for the eq:17 minimization (bbl_solve.py).

Shares come from the real RC demand (foundation_demand_eval), deposits from the real
law of motion (foundation_deposit_sim), and ψ from the exact basis (foundation_psi_basis)
— not the crude exp(δ-shift) share proxy with hard-coded r_f of the earlier prototype.

Pipeline
--------
  σ̂  (equilibrium)  → simulate deposits → accumulate ψ_eq         (per firm)
  σ̃₁…σ̃_S (deviations)→ simulate deposits → accumulate ψ_dev[s]     (per firm)
  export {ψ_eq, ψ_dev, firms, firm_is_B, Z_names}  → psi_bbl_*.jls in the BBL step folder
                                                    (`cf_out_dir(out_dir, "COST_FWD")`:
                                                     data/output/bbl; local COST_FWD/)
  bbl_solve.py reads these and minimizes Σ min{g,0}² (eq:17).

DEVIATING STRATEGY σ̃ (`--dev-scheme`, default `grid`):
  σ̃ shifts the CHOICE spreads (k∈{4,5}) by Δ and holds the perturbed policy for the
  whole horizon (a stationary deviation, as in BBL forward simulation). `grid`: Δ takes
  a symmetric grid over [−scale,+scale] excluding 0, so both raising AND lowering are
  probed at graduated magnitudes — the eq:17 objective Σ min{g,0}² is only informative
  where a deviation binds, so directed small deviations pin the FOC far better than the
  tiny symmetric normals (`normal`, legacy) they replace. `--perturb-scale` is the grid
  half-width in annualized-ρ units (ρ=spread_ann/100).

  UNILATERAL: Δ moves ONE firm's own k∈{4,5} rows; every rival holds σ̂. The loop runs one
  forward simulation per (firm j × Δ) pair (~300 choice firms × S ⇒ shard it), and keeps only
  the deviator's ψ. Rivals' mean utilities stay at equilibrium — their shares still move, via
  the share denominator, which IS the business stealing eq:17 prices. A common industry-wide
  shift is the collusive direction and would certify a false inequality; see the loop comment.

FORWARD r^f (`--rf-curve`, default the uploaded `forward_rf_qoq.csv` from scrape_forward_rf.py —
  data/input on the cluster, local COST_FWD/; see `load_forward_rf`):
  the market Selic curve enters ψ4. A FLAT r^f makes ψ4 collinear with ψ2, leaving ζ
  unidentified; the time-varying curve separates ζ from ω.

ASSET RETURN r^j (`--asset-return-col` / `--asset-margin`, default 0): enters ψ1 (V_Main
  eq 16, row 1). Both flags are read as the QUARTERLY NET margin (r^j − r^f); foundation_psi_basis
  adds r^f back so ψ1 carries the GROSS r^j the paper requires.
    ⚠ DO NOT pass `--asset-return-col gross_return_lag`. Despite the name, that column is
      `1 + deposit_rate_lag` (sleep_demand_prep_e1.py:167) — the rate the bank PAYS
      DEPOSITORS (liability side), not what it earns on assets. It sits BELOW r^f by
      construction (that gap IS the markdown this paper estimates); using it as r^j would
      hand the model a negative asset margin.
    ✅ USE `--asset-return-col asset_gross_return_lag`. (This docstring previously said no
      asset-return column existed; that is STALE — the column was added to the demand-prep
      parquets and verified present 2026-08-04: mean 1.0365 = 1 + quarterly asset return,
      i.e. ~3.5%/q net of r^f, vs gross_return_lag's 1.0119 = 1 + the deposit rate.
      `asset_return_imputed` flags the 0.2% of rows without a reported yield.) The stale note
      is why the 2026-08-03 cluster run was launched WITHOUT any asset margin and produced
      ω̂ < 0 in all 8 blocks — exactly the pathology predicted two lines below.
    Default 0 ⇒ r^j = r^f: deposits earn exactly the risk-free rate, so a deposit is worth
      only (ρ − c). That is a SUBSTANTIVE assumption (it says the marginal deposit funds
      reserves/govvies, not credit), and with the observed spreads it forces ω̂ < 0.
  A firm-constant r^j is collinear with the ω regressor (ψ2), so it shifts ω̂ one-for-one:
  the data cannot pin r^j down internally — it must be measured/calibrated from outside.

EQUILIBRIUM σ̂ (knob):
  Default σ̂ = observed spreads ρ̂ (the data IS the equilibrium). Pass
  `--policy-csv` to instead use the smoothed fitted policy from
  bbl_polfunc.py (polfunc_fitted.csv).

⚠ COMPUTE: each σ̃ costs one deposit simulation (≈ one share evaluation when spreads
are held flat). With S deviations on the full panel at R=2000 this is the heavy,
GPU/cluster step. Develop locally with --R small, --time-filter one quarter, and
--shocks small; run headline on Bouchet.

β AND T: --beta / --horizon; when omitted, BBL_BETA / BBL_HORIZON from bbl_discount.env (the one
registry of both, read by `bbl_discount` in cf_psi_basis.jl). The log line "[BBL] discount:" says
which source each value came from.

Usage (write-only here; run only after data is downloaded AND author authorizes):
  julia --project=. --threads=4 bbl_fwd_sim.jl --estim 6 --spec 12 \\
      --stage extended --R 300 --time-filter 2024Q4 --shocks 20
"""

include(joinpath(@__DIR__, "cf_psi_basis.jl"))

using DataFrames, Random, Serialization, Statistics

# ==========================================================================
# Spread scenarios
# ==========================================================================
# ── BBL Step-1 fitted-policy merge constants ────────────────────────────────────────────────────
# The polfunc regressand `spread_qoq` (= market_panel spread_a{k}) is a per-QUARTER FRACTION, whereas
# ρ̂ = spread_ann/100 is an ANNUALIZED percentage point. So fitted × 400 (=×4 quarter→year, ×100
# fraction→pp) converts the fitted policy to ρ̂ units. VERIFIED empirically: on the matched k∈{4,5}
# rows the OBSERVED spread_qoq×400 reproduces ρ̂ to corr≈0.999 (k4) / 1.000 (k5).
const _POLFUNC_QOQ_TO_ANN_PP = 400.0
# bbl_polfunc.py::compute_fitted_values predicts with missing regressors filled to 0
# (`fillna(0)`), which pins ~19% of early-panel (2013–2016) B rows at the ≈190pp regression intercept —
# absurd for a deposit spread (real ρ̂ never exceeds ~16pp). Any |fitted|>this cap is an upstream
# extrapolation artifact and is NOT adopted; that row keeps its observed spread. The cap is far above
# every real spread and far below the intercept plateau, so it is insensitive (same fallback set for a
# 20–50pp cap).
const _POLFUNC_SANE_CAP_PP = 40.0

# Composite row key for the (CodConglomeradoPrudencial × mca_code × deposit_type × time_id) match.
_polkey(firm, mca, k::Int, tid) = string(firm, '\x1f', mca, '\x1f', k, '\x1f', tid)

"""
    _load_policy_map(path) -> Dict{String,Float64}

Parse the BBL Step-1 fitted-policy CSV (`polfunc_fitted.csv` from
bbl_polfunc.py) into `(firm,mca,k,time) → fitted spread (annual pp)`, keeping only
k∈{4,5} rows with a finite fitted value. Per firm type we take that type's OWN Step-1 regression: B
firms → the `_B` column, D firms → the `_D_optB` column (national pop-weighted demographics — the
best-fitting D spec). Values are converted qoq-fraction → annual pp (×`_POLFUNC_QOQ_TO_ANN_PP`).
Uses a minimal comma split (no CSV.jl dependency; the file's fields never contain commas or quotes).
"""
function _load_policy_map(path::String)::Dict{String,Float64}
    lines = readlines(path)
    isempty(lines) && error("policy-csv is empty: $path")
    hdr = split(strip(lines[1]), ',')
    ci  = Dict(String(strip(String(h))) => i for (i, h) in enumerate(hdr))
    need = ["CodConglomeradoPrudencial", "mca_code", "deposit_type", "is_B", "time_id",
            "fitted_k4_Time_CDB_B", "fitted_k4_Time_CDB_D_optB",
            "fitted_k5_Prepaid_B", "fitted_k5_Prepaid_D_optB"]
    for c in need
        haskey(ci, c) || error("policy-csv missing column '$c' in $(basename(path)). " *
                               "Re-run bbl_polfunc.py --spec <spec>.")
    end
    i_firm = ci["CodConglomeradoPrudencial"]; i_mca = ci["mca_code"]; i_k = ci["deposit_type"]
    i_isB = ci["is_B"]; i_t = ci["time_id"]
    i_k4B = ci["fitted_k4_Time_CDB_B"]; i_k4D = ci["fitted_k4_Time_CDB_D_optB"]
    i_k5B = ci["fitted_k5_Prepaid_B"]; i_k5D = ci["fitted_k5_Prepaid_D_optB"]
    ncol = length(hdr)
    m = Dict{String,Float64}()
    @inbounds for line in Iterators.drop(lines, 1)   # drop(…, 1) skips the header row
        isempty(line) && continue
        f = split(line, ',')
        length(f) < ncol && continue
        k = tryparse(Int, strip(String(f[i_k])))
        (k === nothing || !(k == 4 || k == 5)) && continue
        isB = strip(String(f[i_isB])) == "True"
        col = k == 4 ? (isB ? i_k4B : i_k4D) : (isB ? i_k5B : i_k5D)
        raw = strip(String(f[col]))
        isempty(raw) && continue                       # missing fitted (NaN written blank) → skip
        v = tryparse(Float64, raw)
        (v === nothing || !isfinite(v)) && continue
        key = _polkey(String(strip(String(f[i_firm]))), String(strip(String(f[i_mca]))),
                      k, String(strip(String(f[i_t]))))
        m[key] = v * _POLFUNC_QOQ_TO_ANN_PP            # qoq-fraction → annual pp (ρ̂ units)
    end
    isempty(m) && error("policy-csv parsed 0 usable k∈{4,5} fitted rows: $path")
    return m
end

"""
    equilibrium_spreads(ctx; policy_csv=nothing) -> Vector{Float64}

The equilibrium choice-spread vector σ̂ (annualized pp, ρ = spread_ann/100). Default = the observed
spreads ρ̂ — the data IS the equilibrium, so every row passes through unchanged.

If `policy_csv` is given (the BBL Step-1 fitted policy, `polfunc_fitted.csv`), the CHOICE rows
k∈{4,5} are replaced by the FITTED policy so that Step-2 deviations perturb the smoothed policy rather
than raw noisy spreads (author decision 2026-07-16 — observed-spread deviations make frac_bind≈0.5
mechanically, since σ̂ is not then a turning point of the simulated value). Regulated types k∈{1,2}
always keep their observed (exogenous) spread. Matching is on
(CodConglomeradoPrudencial × mca_code × deposit_type × time_id); units are converted qoq-fraction →
annual pp via ×`_POLFUNC_QOQ_TO_ANN_PP` (see the const's note for the empirical verification).

ROBUSTNESS: the Step-1 CSV pins ~19% of early-panel B rows at the ≈190pp intercept (upstream
`fillna(0)` extrapolation). Those implausible rows (|fit|>`_POLFUNC_SANE_CAP_PP`) and any non-finite
or unmatched rows fall back to the observed spread, with a loud count. The units/key SANITY GATE
(match rate, |fit−obs| gap, correlation) is then evaluated on the ADOPTED rows and THROWS on a
genuine units or key bug (which would corrupt the adopted rows too).
"""
function equilibrium_spreads(ctx::CFDemandCtx; policy_csv::Union{Nothing,String}=nothing)
    σ̂ = copy(ctx.rho_hat)
    policy_csv === nothing && return σ̂
    isfile(policy_csv) || error("--policy-csv file not found: $policy_csv")

    polmap = _load_policy_map(policy_csv)              # (firm,mca,k,time) → fitted spread (annual pp)

    firm  = string.(ctx.df.CodConglomeradoPrudencial)
    mca   = string.(ctx.df.mca_code)
    tid   = string.(ctx.df.time_id)
    kvec  = Int.(coalesce.(ctx.df.deposit_type, 0))
    endog = (kvec .== 4) .| (kvec .== 5)
    n_endog = count(endog)
    n_endog == 0 && error("no k∈{4,5} rows in ctx.df — cannot apply the fitted policy.")

    n_matched = 0; n_adopted = 0
    fit_adopt = Float64[]; obs_adopt = Float64[]
    @inbounds for i in eachindex(σ̂)
        endog[i] || continue
        key = _polkey(firm[i], mca[i], kvec[i], tid[i])
        haskey(polmap, key) || continue                # unmatched → keep observed
        f = polmap[key]
        isfinite(f) || continue
        n_matched += 1
        if abs(f) <= _POLFUNC_SANE_CAP_PP
            σ̂[i] = f                                   # ADOPT the fitted policy on this choice row
            n_adopted += 1
            push!(fit_adopt, f); push!(obs_adopt, ctx.rho_hat[i])
        end                                            # else: implausible fitted → keep observed
    end

    match_rate  = n_matched / n_endog
    n_untrusted = n_matched - n_adopted
    n_unmatched = n_endog - n_matched
    log_status("  [BBL] policy-csv ← $(basename(policy_csv)); fitted qoq-fraction × " *
               "$(_POLFUNC_QOQ_TO_ANN_PP) → annual pp (ρ̂ units)")
    log_status("  [BBL] k∈{4,5}: $n_endog rows | matched $n_matched " *
               "($(round(100 * match_rate, digits=1))%) | adopted $n_adopted | " *
               "fell back to observed: $n_unmatched unmatched + $n_untrusted implausible " *
               "(|fit|>$(_POLFUNC_SANE_CAP_PP)pp)")

    # ── SANITY GATE (throws on a units or key bug) ──────────────────────────────────────────────
    match_rate < 0.95 && error(
        "policy-csv match rate $(round(100 * match_rate, digits=1))% < 95% on k∈{4,5} — key mismatch " *
        "(matched on CodConglomeradoPrudencial×mca_code×deposit_type×time_id; check the CSV keys).")
    n_adopted == 0 && error(
        "policy-csv: 0 adopted rows (every matched fitted value non-finite or |fit|>$(_POLFUNC_SANE_CAP_PP)pp) — units or upstream-fit bug.")

    med_gap    = median(abs.(fit_adopt .- obs_adopt))
    med_spread = median(abs.(obs_adopt))
    ρcorr      = cor(fit_adopt, obs_adopt)
    log_status("  [BBL] adopted-row sanity: median|fit−obs|=$(round(med_gap, digits=3))pp | " *
               "median|obs|=$(round(med_spread, digits=3))pp | corr(fit,obs)=$(round(ρcorr, digits=3))")
    if n_untrusted > 0
        @warn "  [BBL] $n_untrusted/$n_endog matched k∈{4,5} fitted spreads exceeded " *
              "$(_POLFUNC_SANE_CAP_PP)pp and fell back to observed. Cause: " *
              "bbl_polfunc.py::compute_fitted_values fills missing regressors with 0, " *
              "pinning early-panel B rows at the ≈190pp intercept. Restrict that prediction to complete " *
              "cases to adopt the fitted policy on those rows too."
    end
    ρcorr < 0.3 && error(
        "policy-csv sanity: corr(fitted,observed)=$(round(ρcorr, digits=3)) < 0.3 on adopted k∈{4,5} " *
        "rows — likely a units or key bug (the fitted policy should track observed spreads).")
    med_gap > med_spread && error(
        "policy-csv sanity: median|fit−obs|=$(round(med_gap, digits=3))pp > median|obs|=" *
        "$(round(med_spread, digits=3))pp — likely a units bug (check the ×$(_POLFUNC_QOQ_TO_ANN_PP) " *
        "qoq→annual-pp conversion).")
    return σ̂
end

"""
    deviation_shifts(S, scale, scheme, seed) -> Vector{Float64}

The additive spread shifts Δ_s (annualized ρ units) defining the S deviating
strategies. `grid` (default): a symmetric grid over [−scale, +scale] EXCLUDING 0, so
we probe RAISING and LOWERING the choice spread at graduated magnitudes — this is what
pins the FOC, since the eq:17 objective Σ min{g,0}² is only informative where a
deviation binds (g<0), and tiny i.i.d. normals mostly leave g>0. `normal`: legacy
N(0,scale) (kept for comparison). Deterministic in `grid` mode ⇒ shard-invariant by
global index with no RNG.
"""
function deviation_shifts(S::Int, scale::Float64, scheme::String, seed::Int)
    if scheme == "normal"
        rng = MersenneTwister(seed * 100003)
        return scale .* randn(rng, S)
    end
    scheme == "grid" || error("--dev-scheme must be grid|normal (got $scheme)")
    m = cld(S, 2)                              # magnitudes; each gets a ± pair
    mags = (collect(1:m) ./ m) .* scale        # graduated up to `scale`
    shifts = Float64[]
    for g in mags; push!(shifts, +g); push!(shifts, -g); end
    return shifts[1:S]
end

"""
    firm_endog_rows(ctx, st, firms) -> Vector{Vector{Int}}

Row indices of each firm's OWN choice spreads (k∈{4,5}), aligned to `firms`. Needed for the
unilateral deviation: firm j moves only its own rows, rivals stay at σ̂.
"""
function firm_endog_rows(ctx::CFDemandCtx, st::DepositSimState, firms::Vector{String})
    key = string.(ctx.df.CodConglomeradoPrudencial)
    idx = Dict(f => j for (j, f) in enumerate(firms))
    rows = [Int[] for _ in 1:length(firms)]
    @inbounds for i in eachindex(key)
        st.endog[i] || continue
        j = get(idx, key[i], 0)
        j > 0 && push!(rows[j], i)
    end
    return rows
end

"""
    apply_shift_firm(σ̂, rows_j, Δ) -> Vector{Float64}

UNILATERAL deviation: add Δ to firm j's own k∈{4,5} spreads only; every rival stays at σ̂.
This is the perturbation eq:17 requires (a Nash/MPE no-profitable-deviation condition).
Shifting every firm at once instead would be a coordinated (collusive) move and must NOT be
used to build the eq:17 moments — `rows_j` is what keeps the move firm-specific.
"""
function apply_shift_firm(σ̂::Vector{Float64}, rows_j::Vector{Int}, Δ::Float64)
    σ̃ = copy(σ̂)
    @inbounds for i in rows_j; σ̃[i] += Δ; end
    return σ̃
end

"""
    load_forward_rf(path, out_dir, T, ctx) -> Vector{Float64}

Quarterly forward r^f path (length T) from the market Selic curve written by
scrape_forward_rf.py (`forward_rf_qoq.csv`, column `rf_qoq`, read from `cf_in_dir`), padded/truncated to
T. A FLAT r^f makes ψ4 = r^f·Σβ^t Dep a rescaling of ψ2 = Σβ^t Dep (collinear) so ζ is
unidentified; the time-varying curve breaks that. Falls back to the flat panel median
(with a warning) if the curve file is absent.
"""
function load_forward_rf(path::Union{Nothing,String}, out_dir::String, T::Int, ctx::CFDemandCtx;
                         require::Bool=false)
    # forward r^f is fetched LOCALLY (compute nodes have no internet) and UPLOADED → data/input
    csv = path === nothing ? joinpath(cf_in_dir(out_dir, "COST_FWD"), "forward_rf_qoq.csv") : path
    if isfile(csv)
        lines = filter(l -> !isempty(strip(l)), readlines(csv))
        hdr = strip.(split(lines[1], ','))
        ci = findfirst(==("rf_qoq"), hdr)
        ci === nothing && error("rf_qoq column not found in $csv")
        rf = [parse(Float64, strip(split(l, ',')[ci])) for l in lines[2:end]]
        # Shorter than T: on the cluster (require) a refusal, as in cl_bbl_check_curve, unless
        # ALLOW_RF_PAD=1 accepts the flat tail deliberately; locally the tail repeats the last rate.
        if length(rf) < T && require && get(ENV, "ALLOW_RF_PAD", "0") != "1"
            error("$(basename(csv)) runs to h=$(length(rf)) but T=$T: the last $(T - length(rf)) quarters " *
                  "would repeat the last quoted rate. Rebuild it locally and upload it:\n" *
                  "    python scrape_forward_rf.py --horizon $T --start 2026Q1\n" *
                  "  (ALLOW_RF_PAD=1 accepts the padding deliberately.)")
        end
        length(rf) >= T || (rf = vcat(rf, fill(rf[end], T - length(rf))))
        log_status("  [BBL] forward r^f ← $(basename(csv)) (T=$T; " *
                   "$(round(rf[1],sigdigits=4))→$(round(rf[T],sigdigits=4)))")
        return rf[1:T]
    end
    # Missing curve. On the cluster this MUST be a hard failure: silently using a flat
    # r^f leaves ζ unidentified (ψ4∝ψ2) and wastes the expensive run. Locally, warn +
    # fall back so dev/smoke tests still run.
    msg = "Forward r^f curve not found at:\n    $csv\n" *
          "Generate it locally (needs internet) and upload it there:\n" *
          "    python scrape_forward_rf.py --horizon $T --start 2026Q1"
    require && error(msg)
    rf_q0, rfc = _first_present_rf_level(ctx.df, RF_LEVEL_CANDIDATES; default=NaN, what="fallback flat r^f")
    lvl = median(filter(isfinite, rf_q0))
    @warn "$msg\n  → FALLING BACK to FLAT r^f=median($rfc)=$(round(lvl,sigdigits=4)); ζ weakly identified."
    return fill(lvl, T)
end

# ==========================================================================
# ψ under a given (stationary) spread scenario
# ==========================================================================
"""
    psi_under(ctx, st, Z, markdown_q, spreads_ann; beta, T, asset_return_q, rf_path_q)
        -> (psi_firm, firms)

Simulate deposits forward under the stationary spread vector `spreads_ann`
(annualized, for shares) and accumulate the firm-level ψ basis. The per-period
markdown for the value flow is `markdown_q` (quarterly); for a deviation that
changes spreads, the markdown should move with it — by default we recompute the
quarterly markdown from the scenario spreads as `spreads_ann/4` shifted by the
observed wedge (see note) to stay consistent.

THE RISK-FREE PATH NOW REACHES THE DEPOSITS. `rf_path_q` used to be handed to
`accumulate_psi` alone, so the same r^f was moving inside ψ4 while the sleeper accrual in
the deposit recursion stayed pinned at its starting value — eq. 9's carry term is
(1 + r^f − ρ)·Dep_{t−1}, so one r^f was being used two different ways inside a single ψ
evaluation. It is now passed to `simulate_deposits` as well. With the near-flat single
Focus curve this was second order; across launch quarters whose Selic spans 2% to 14.25%
it is not.

EXPECTATION OVER RATE RISK (`rf_paths`, a P×T matrix). ψ is then E_p[ψ(σ, p)], averaged
over P simulated rate paths rather than evaluated along one deterministic curve. The share
kernel does not depend on r^f, so the single expensive `cf_shares_at` call is hoisted out
and reused across every path (`s_const_in`); the per-path cost is the deposit recursion and
the ψ accumulation, both O(N·T) vector work. That is what makes rate risk affordable here:
P paths cost roughly P× the cheap part and 1× the expensive part.

EVOLVING MARKET STATES (`ctx.state_ev`). The demographics a market is simulated under follow
their own estimated mean reversion instead of being pinned at the launch quarter, so the
demand block reads the same states the state block does. The share is then a function of the
horizon as well as the spread, and the hoisted object becomes an N×T path; the hoist itself
is unchanged, because the demographic path is identical across deviations and rate paths.
"""
function psi_under(ctx::CFDemandCtx, st::DepositSimState, Z::Matrix{Float64},
                   markdown_q0::Vector{Float64}, spreads_ann::Vector{Float64};
                   beta::Float64=bbl_beta(), T::Int=bbl_horizon(),
                   asset_return_q::Union{Nothing,Vector{Float64}}=nothing,
                   rf_path_q::Union{Nothing,Vector{Float64}}=nothing,
                   rf_paths::Union{Nothing,Matrix{Float64}}=nothing,
                   rf_curve_paths::Union{Nothing,Array{Float64,3}}=nothing,
                   row_curve::Union{Nothing,Vector{Int}}=nothing,
                   row_start::Union{Nothing,Vector{String}}=nothing)
    # Move the quarterly markdown with the scenario spread: Δρ^q ≈ Δρ_ann/4.
    # (Confirm annual→quarterly convention; only matters for k∈{4,5} deviations.)
    # markdown_q0 is the observed quarterly markdown (fraction); a scenario change in
    # the ANNUAL spread (pp = spread_ann/100) maps to a quarterly-fraction change via
    # /400 (×0.01 pp→fraction, ÷4 annual→quarter).
    markdown_q = markdown_q0 .+ (spreads_ann .- ctx.rho_hat) ./ 400.0
    assert_state_evolution(ctx)   # a missing transitions upload must not demote this to frozen states

    # Legacy single deterministic path (one curve for the whole panel).
    if rf_paths === nothing && rf_curve_paths === nothing
        sim = simulate_deposits(ctx, st; T=T, spreads_ann=spreads_ann, rf_path_q=rf_path_q,
                                state_ev=ctx.state_ev)
        res = accumulate_psi(ctx, st, sim.Dep, markdown_q, Z;
                             beta=beta, asset_return_q=asset_return_q, rf_path_q=rf_path_q)
        return res.psi_firm, res.firms
    end

    # One share evaluation for every path (see the docstring): r^f never enters the share. Under
    # evolving market states this is an N×T share PATH rather than one vector — the demographics
    # move with the horizon even at a fixed spread — but it is still hoisted over the P rate paths
    # for the same reason, and still costs one μ build (`cf_shares_path`).
    s_const = cf_shares_path(ctx, spreads_ann, ctx.state_ev; T=T)
    acc = nothing; firms = String[]; fkey = String[]; skey = String[]

    if rf_curve_paths !== nothing
        # MULTI-START x RATE RISK: (P paths) x (S launch quarters) x T.
        P = size(rf_curve_paths, 1)
        size(rf_curve_paths, 3) >= T || error("rf_curve_paths has < T periods")
        row_curve === nothing && error("rf_curve_paths needs row_curve")
        @inbounds for p in 1:P
            cur = Matrix{Float64}(rf_curve_paths[p, :, :])
            sim = simulate_deposits(ctx, st; T=T, spreads_ann=spreads_ann,
                                    s_const_in=s_const, rf_curves=cur, row_curve=row_curve)
            res = accumulate_psi(ctx, st, sim.Dep, markdown_q, Z;
                                 beta=beta, asset_return_q=asset_return_q,
                                 rf_curves=cur, row_curve=row_curve, row_start=row_start)
            acc = acc === nothing ? copy(res.psi_firm) : (acc .+ res.psi_firm)
            firms = res.firms; fkey = res.firm_of_key; skey = res.start_of_key
        end
        return acc ./ P, firms, fkey, skey
    end

    # Rate risk with ONE common curve (no start dimension).
    P = size(rf_paths, 1)
    size(rf_paths, 2) >= T || error("rf_paths has $(size(rf_paths,2)) periods < T=$T")
    @inbounds for p in 1:P
        rp = Vector{Float64}(vec(rf_paths[p, 1:T]))
        sim = simulate_deposits(ctx, st; T=T, spreads_ann=spreads_ann,
                                rf_path_q=rp, s_const_in=s_const)
        res = accumulate_psi(ctx, st, sim.Dep, markdown_q, Z;
                             beta=beta, asset_return_q=asset_return_q, rf_path_q=rp)
        acc = acc === nothing ? copy(res.psi_firm) : (acc .+ res.psi_firm)
        firms = res.firms
    end
    return acc ./ P, firms
end

# ==========================================================================
# Firm type (B/D) for the κ-specific cost blocks
# ==========================================================================
"""
    firm_is_B(ctx, firms) -> BitVector

Classify each firm (CodConglomeradoPrudencial) as B-type if the majority of its
observations are B (brick-and-mortar). The cost parameters (ω,ζ,γ) differ by κ∈{B,D}.
"""
function firm_is_B(ctx::CFDemandCtx, firms::Vector{String})::BitVector
    key = string.(ctx.df.CodConglomeradoPrudencial)
    isB = BitVector(Bool.(coalesce.(ctx.df.is_B, false)))
    nB = Dict{String,Int}(); nT = Dict{String,Int}()
    for i in eachindex(key)
        nT[key[i]] = get(nT, key[i], 0) + 1
        isB[i] && (nB[key[i]] = get(nB, key[i], 0) + 1)
    end
    return BitVector([get(nB, f, 0) >= get(nT, f, 1) / 2 for f in firms])
end

# ==========================================================================
# Multi-start launch quarters and forward-rate risk
# ==========================================================================
# h=0 of a vintage is the REALISED rate at its launch quarter, so it must reproduce the estimation
# panel's OWN contemporaneous r^f for that quarter. 0.002/quarter (≈0.8pp/yr) is the largest gap a
# same-quarter convention difference (compounded period-average Selic in the panel vs the
# quarter-end Focus print) can open; anything larger means the curve file and the panel disagree
# about which quarter a row sits in, which would hand every row the wrong point of the Selic cycle.
# MEASURED on demand_4_index_time_spec_12 × forward_rf_vintages.csv: max gap 0.00164 at 2023Q1,
# mean 0.00035 across the 36 quarters — inside the tolerance with room to spare.
const _MS_ANCHOR_TOL = 0.002

"""
    _quarter_labels(df) -> (labels::Vector{String}, source::String)

The "YYYYQn" launch-quarter label of every panel row, in the exact form
`forward_rf_vintages.csv` keys `start_q` on. `time_id` IS the panel's quarter key and the demand
prep already writes it in that form (verified on demand_4_index_time_spec_12: 36 distinct labels,
2016Q1…2024Q4), so it is taken as-is whenever every row matches; `year`+`quarter` is the fallback
for a parquet vintage that stores the two components separately. Nothing is reconstructed from a
date column: a quarter label that is silently off by one would put every row on the wrong forward
curve and the run would still look healthy.
"""
function _quarter_labels(df::DataFrame)
    if "time_id" in names(df)
        lab = string.(df.time_id)
        all(l -> occursin(r"^\d{4}Q[1-4]$", l), lab) && return lab, "time_id"
    end
    if "year" in names(df) && "quarter" in names(df)
        y = [Int(coalesce(v, 0)) for v in df.year]
        q = [Int(coalesce(v, 0)) for v in df.quarter]
        (all(1 .<= q .<= 4) && all(y .>= 1900)) || error(
            "year/quarter hold values outside {1900+} × {1..4} — cannot build a YYYYQn label.")
        return [string(y[i], "Q", q[i]) for i in eachindex(y)], "year×quarter"
    end
    error("No quarter label in the demand parquet: need `time_id` in YYYYQn form, or `year` and " *
          "`quarter`. --multi-start cannot assign a launch quarter to a row without one.")
end

"""
    _load_rf_vintages(path, T; strict=false) -> (starts, curves, anchor, source)

Parse `forward_rf_vintages.csv` (scrape_forward_rf.py) into, per launch quarter `start_q`:
`curves[q]` = the length-T forward path (h=1…T of `rf_qoq`), `anchor[q]` = h=0, the realised rate
at the launch quarter, and `source[q]` = the provenance string. `starts` is sorted.

h=1…h_max must be COMPLETE for a quarter: a hole would otherwise be filled by whatever the padding
rule reached for, putting one horizon of one vintage on a rate it was never quoted at. Horizons
beyond h_max repeat h_max (the same rule `load_forward_rf` uses for the single curve); with
`strict` (the cluster, --hpc) a vintage shorter than T is a refusal instead, as in
cl_bbl_check_curve, unless ALLOW_RF_PAD=1. Minimal comma split, as in `_load_policy_map` —
verified that no field in the file contains a comma (the `source` field separates its parts with `|`).
"""
function _load_rf_vintages(path::String, T::Int; strict::Bool=false)
    isfile(path) || error("--rf-vintages file not found:\n    $path\n" *
                          "Generate it locally (needs internet) and upload it:\n" *
                          "    python scrape_forward_rf.py --vintage-from 2016Q1 --vintage-to 2024Q4 --horizon $T")
    lines = filter(l -> !isempty(strip(l)), readlines(path))
    length(lines) < 2 && error("rf-vintages has no data rows: $path")
    hdr = [String(strip(String(h))) for h in split(lines[1], ',')]
    ci  = Dict(h => i for (i, h) in enumerate(hdr))
    for c in ("start_q", "h", "rf_qoq", "source")
        haskey(ci, c) || error("rf-vintages missing column '$c' in $(basename(path)).")
    end
    i_s = ci["start_q"]; i_h = ci["h"]; i_r = ci["rf_qoq"]; i_src = ci["source"]
    raw = Dict{String,Dict{Int,Float64}}(); source = Dict{String,String}()
    for line in Iterators.drop(lines, 1)               # drop(…, 1) skips the header row
        f = split(line, ',')
        length(f) < length(hdr) && continue
        s = String(strip(String(f[i_s])))
        h = tryparse(Int, strip(String(f[i_h])));      h === nothing && continue
        v = tryparse(Float64, strip(String(f[i_r])));  (v === nothing || !isfinite(v)) && continue
        get!(raw, s, Dict{Int,Float64}())[h] = v
        get!(source, s, String(strip(String(f[i_src]))))
    end
    isempty(raw) && error("rf-vintages parsed 0 usable rows: $path")

    starts = sort(collect(keys(raw)))
    curves = Dict{String,Vector{Float64}}(); anchor = Dict{String,Float64}()
    n_pad = 0; hmin = typemax(Int)
    for q in starts
        d = raw[q]
        hmax = maximum(h for h in keys(d) if h >= 1; init=0)
        hmax >= 1 || error("rf-vintages start $q has no h≥1 forward horizon.")
        for h in 1:hmax
            haskey(d, h) || error("rf-vintages start $q is missing horizon h=$h (h runs 1…$hmax).")
        end
        curves[q] = [d[min(h, hmax)] for h in 1:T]
        T > hmax && (n_pad += 1)
        hmin = min(hmin, hmax)
        haskey(d, 0) && (anchor[q] = d[0])
    end
    if n_pad > 0 && strict && get(ENV, "ALLOW_RF_PAD", "0") != "1"
        error("$(basename(path)): $n_pad/$(length(starts)) vintages are shorter than T=$T (the " *
              "shortest runs to h=$hmin), so their tails would repeat the last quoted rate. Rebuild " *
              "it locally and upload it:\n" *
              "    python scrape_forward_rf.py --vintage-from 2016Q1 --vintage-to 2024Q4 --horizon $T\n" *
              "  (ALLOW_RF_PAD=1 accepts the padding deliberately.)")
    end
    n_pad == 0 || @warn "  [BBL] $n_pad/$(length(starts)) vintages are shorter than T=$T; their " *
                        "tail horizons repeat the last quoted rate."
    return starts, curves, anchor, source
end

"""
    _load_rate_sd_by_h(path, T) -> (sd::Vector{Float64}, mode::String, hmax::Int)

The horizon term structure of Focus forecast error from `bbl_transitions.json` (bbl_transitions.py),
as `sd[h]` for h=1…T. Only `rate.process.mode == "focus_mean_plus_horizon_shock"` is implemented:
the deviation from the Focus mean is `d_h = sd[h]·z` with ONE standard normal z per simulated path.
The pooled AR(1) alternative in that file is flagged `ar1_rejected` (φ_d = 1.031, explosive) and
its `cir` / `ar1_level` blocks carry `usable_as_mean: false`, so any other mode is a hard stop
rather than a silent fallback. Horizons beyond the largest key repeat it (the empirical term
structure saturates, which is why no AR(1) matches it).
"""
function _load_rate_sd_by_h(path::String, T::Int)
    isfile(path) || error("--transitions file not found:\n    $path\n" *
                          "Produce it with: python bbl_transitions.py")
    j = JSON3.read(read(path, String))
    haskey(j, "rate") || error("bbl_transitions.json has no `rate` block: $path")
    haskey(j["rate"], "process") || error("bbl_transitions.json `rate` has no `process` block.")
    pr = j["rate"]["process"]
    mode = String(pr["mode"])
    mode == "focus_mean_plus_horizon_shock" || error(
        "rate.process.mode is '$mode'; this driver implements only " *
        "'focus_mean_plus_horizon_shock'. Re-run bbl_transitions.py, or run with --n-paths 1 " *
        "(the Focus mean curve, no rate risk).")
    get(pr, "provisional", false) && @warn "  [BBL] rate.process is flagged provisional in " *
                                           "$(basename(path)) — the sd(h) term structure is not final."
    kv = Dict{Int,Float64}()
    for (k, v) in pairs(pr["sd_by_h"])
        h = tryparse(Int, String(k)); h === nothing && continue
        (v isa Number && isfinite(Float64(v))) || continue
        kv[h] = Float64(v)
    end
    isempty(kv) && error("rate.process.sd_by_h is empty in $(basename(path)).")
    hmax = maximum(keys(kv))
    for h in 1:hmax
        haskey(kv, h) || error("rate.process.sd_by_h is missing h=$h (h runs 1…$hmax).")
    end
    return [kv[min(h, hmax)] for h in 1:T], mode, hmax
end

"""
    _horizon_shock_paths(rf_mean, sd, P, seed) -> (paths::Array{Float64,3}, z, n_clipped, max_bias)

Build the P×S×T rate paths: `paths[p,s,h] = max(0, rf_mean[s,h] + sd[h]·z_p)`, with ONE z per path
shared across every launch quarter and horizon (that is the `focus_mean_plus_horizon_shock`
process — a horizon-scaled LEVEL shock, not an independent shock per horizon).

ANTITHETIC draws, and an ODD P keeps path 1 at z=0 (the Focus mean curve itself) so the rest still
pair up. `sum(z) == 0` exactly for every P ⇒ the simulated mean stays on the Focus curve instead of
drifting off it by the sampling error of a small P, which is the whole point of taking the
expectation this way. P=1 therefore reduces to the pure Focus mean with no RNG consumed at all.

`seed` is the run's `--seed`, so every shard of a resharded run draws the SAME z vector and the
shards still merge into the unsharded result. The multiplier differs from `deviation_shifts`'s so
the two streams are not the same numbers in a different role.

The `max(0, ·)` floor keeps a path from quoting a negative nominal Selic (the deposit accrual
1+r^dep would otherwise run the wrong way). It is the ONE thing that breaks the antithetic
guarantee, and one-sidedly: clipping only ever raises a path, so wherever it binds the P-path mean
sits ABOVE the Focus curve. `n_clipped` counts the entries it touched and `max_bias` is the largest
|mean_p paths − rf_mean| it produced, both reported by the caller — the r̄^f spread across launch
quarters is the identification claim here, so a silent upward drift in the low-rate quarters is
exactly the failure that must not go unlogged. sd(h) SATURATES (measured 2026-09-09:
0.00108 at h=1 rising to 0.01363 from h=16 on) while the 2020 curves sit near 0.008/quarter, so
this binds for real at moderate P — check the two numbers before trusting a large-P run.
"""
function _horizon_shock_paths(rf_mean::Matrix{Float64}, sd::Vector{Float64}, P::Int, seed::Int)
    S, T = size(rf_mean)
    length(sd) >= T || error("sd_by_h shorter ($(length(sd))) than T=$T")
    z = zeros(P)
    if P > 1
        rng = MersenneTwister(seed * 100019)
        off = isodd(P) ? 1 : 0                    # odd P ⇒ z[1]=0, the Focus mean path
        for q in 1:((P - off) ÷ 2)
            d = randn(rng); z[off + 2q - 1] = d; z[off + 2q] = -d
        end
    end
    paths = Array{Float64,3}(undef, P, S, T)
    nclip = 0
    @inbounds for p in 1:P, s in 1:S, h in 1:T
        v = rf_mean[s, h] + sd[h] * z[p]
        v < 0.0 && (nclip += 1)
        paths[p, s, h] = max(0.0, v)
    end
    max_bias = 0.0
    @inbounds for s in 1:S, h in 1:T
        m = 0.0
        for p in 1:P; m += paths[p, s, h]; end
        b = abs(m / P - rf_mean[s, h])
        b > max_bias && (max_bias = b)
    end
    return paths, z, nclip, max_bias
end

# ==========================================================================
# Atomic artifact writes
# ==========================================================================
"""
    _atomic_write(write_fn, final) -> final

Write `final` through a temporary name in the SAME directory, then rename it into place.

WHY. A psi file is the only evidence that its shard finished, and the sweep job and the solve both
decide completeness from what sits under the final name. Written in place, a task killed at the
wall mid-write, or two tasks writing the same shard (overlapping arrays did this on 2026-09-21),
leave a truncated parquet that passes a file-exists check. rename(2) within one directory is
atomic: the final name holds either the previous complete file or the new complete file, never a
fragment. Concurrent writers each use their own temporary (job id + pid), so the last rename wins
and both candidates are complete — the sim is deterministic, so they are also identical.

The temporary starts with `.tmp_` and does not end in `.parquet`/`.json`, so no `psi_*` or
`*.parquet` glob (bbl_solve.py, make_bbl_cost_tables.py, bbl_shards.py) can ever pick it up. A
temporary left by a killed task is inert debris; cluster_archive.sh excludes it.
"""
function _atomic_write(write_fn::Function, final::AbstractString)
    dir, base = dirname(final), basename(final)
    tmp = joinpath(dir, ".tmp_" * base * "." * get(ENV, "SLURM_JOB_ID", "local") * "_" *
                        string(getpid()))
    try
        write_fn(tmp)
        _rename_replace(tmp, final)
    catch
        isfile(tmp) && rm(tmp; force=true)
        rethrow()
    end
    return final
end

# libuv rename: POSIX rename(2) on Linux, MoveFileEx(REPLACE_EXISTING) on Windows. Called
# directly rather than through `mv(...; force=true)`, which on some Julia versions removes the
# destination FIRST and so reopens exactly the window this exists to close.
function _rename_replace(src::AbstractString, dst::AbstractString)
    rc = ccall(:jl_fs_rename, Int32, (Cstring, Cstring), src, dst)
    rc == 0 || error("atomic rename failed ($rc): $src -> $dst")
    return nothing
end

# ==========================================================================
# Driver
# ==========================================================================
function _parse_cost2_args()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--estim";         arg_type = Int;     default = 6
        "--spec";          arg_type = Int;     default = 12
        "--stage";         arg_type = String;  default = "extended"
        "--R";             arg_type = Int;     default = 2000
        "--seed";          arg_type = Int;     default = 42
        "--hpc";           action   = :store_true
        "--local-dir";     arg_type = String;  default = nothing
        "--suffix";        arg_type = String;  default = ""
        # nothing = BBL_BETA / BBL_HORIZON from bbl_discount.env (resolve_discount!); bbl_run.sh
        # always passes both explicitly.
        "--beta";          arg_type = Float64; default = nothing
        "--horizon";       arg_type = Int;     default = nothing
        "--shocks";        arg_type = Int;     default = 50      # TOTAL number of σ̃ deviations
        # σ̃ grid half-width in annualized ρ units (pp). 2.0 = 200bp ≈ 54% of the median observed
        # choice spread (~3.7pp); the ± grid gives graduated magnitudes up to that. The old default
        # of 0.02 (=2bp) sat so deep in the linear regime that g carried no curvature at all
        # (even/|odd| of Δψ1 ≈ 0.002 vs 0.16 at 200bp), leaving the FOC as the only signal.
        "--perturb-scale"; arg_type = Float64; default = 2.0
        "--dev-scheme";    arg_type = String;  default = "grid"  # grid (directed) | normal (legacy)
        "--rf-curve";      arg_type = String;  default = nothing # forward-r^f CSV; default the uploaded forward_rf_qoq.csv (cf_in_dir)
        "--asset-return-col"; arg_type = String; default = nothing # r^j source col (e.g. gross_return_lag)
        "--asset-margin";  arg_type = Float64; default = 0.0     # constant quarterly (r^j−r^f) if no col
        "--dbar";          arg_type = Float64; default = -1.0   # <=0 => per-type auto-calibrate
        "--time-filter";   arg_type = String;  default = nothing # COMMA-SEPARATED list of quarters
        "--policy-csv";    arg_type = String;  default = nothing
        "--n-shards";      arg_type = Int;     default = 1       # split deviations across jobs
        "--shard-id";      arg_type = Int;     default = 0       # 0-based; = SLURM_ARRAY_TASK_ID
        # ── MULTI-START × RATE RISK ───────────────────────────────────────────────────────────
        # OFF by default: without --multi-start every row is simulated under the ONE curve
        # --rf-curve resolves, and every artifact keeps its untagged name. The CONTENTS are not
        # those of a frozen-accrual run: psi_under routes that curve into the deposit accrual and
        # the market states evolve (CF_EVOLVING_STATES), so a single-curve run reproduces only a
        # run made with the same code and inputs.
        "--multi-start";   action   = :store_true
        # Both default to the uploaded COST_FWD inputs, resolved in main_cost2 because the path
        # needs `out_dir`, which the arg table cannot see (same `nothing` sentinel as --rf-curve).
        "--rf-vintages";   arg_type = String;  default = nothing # forward_rf_vintages.csv (cf_in_dir)
        "--transitions";   arg_type = String;  default = nothing # bbl_transitions.json (cf_in_dir)
        # Rate paths per launch quarter: 1 = the Focus mean curve alone (no rate risk). Each extra
        # path costs one O(N·T) deposit recursion, NOT another share evaluation — psi_under hoists
        # the share out because the share kernel is a function of the spread only.
        "--n-paths";       arg_type = Int;     default = 1
        # Filename tag separating a multi-start artifact from a single-start one. Empty auto-fills
        # to "_ms{P}" (P = --n-paths) under --multi-start and stays "" otherwise; bbl_solve.py
        # and make_bbl_cost_tables.py select a vintage by this tag.
        "--psi-tag";       arg_type = String;  default = ""
    end
    return parse_args(s)
end

function main_cost2()
    a = _parse_cost2_args()
    resolve_discount!(a; horizon=true, who="BBL")
    # --time-filter takes a COMMA-SEPARATED list of quarters, so a --multi-start smoke can span a
    # few launch quarters ("2016Q1,2020Q2,2024Q4") and still keep whole (mca, quarter) markets
    # together — the property build_cf_context needs for its share aggregation to stay exact. One
    # quarter parses to the same single-element vector build_cf_context has always received.
    tf = nothing
    if a["time-filter"] !== nothing
        parts = [String(strip(p)) for p in split(a["time-filter"], ',')]
        filter!(!isempty, parts)
        isempty(parts) && error("--time-filter parsed to an empty list: '$(a["time-filter"])'")
        tf = parts
    end
    ctx = build_cf_context(a["estim"], a["spec"], a["stage"];
                           R=a["R"], seed=a["seed"], hpc=a["hpc"],
                           local_dir=a["local-dir"], suffix=a["suffix"], time_filter=tf)
    # Market size: load_sim_state now takes M_mt/M_nat straight from the demand parquet — the SAME
    # market size the BLP was estimated under (V_Main d̄_mt = bc_mt·r̂_max). The old per-type d-bar
    # auto-calibration that used to live here back-solved a scalar dbar·pop_total that ignored
    # banked_correction, so the CFs simulated under a market size the demand model never saw. It is
    # retired; --dbar remains only as a manual override/diagnostic. See counterfactuals_plan.md §9.8.
    dbar = a["dbar"] > 0.0 ? a["dbar"] : 1.0
    st  = load_sim_state(ctx; dbar=dbar)
    Z, znames = load_Z(ctx)
    markdown_q0, mc = _first_present(ctx.df, ["spread_qoq", "spread_q"]; default=NaN)
    if all(isnan, markdown_q0)
        markdown_q0 = ctx.rho_hat ./ 400.0; mc = "rho_hat/400"          # annual pp -> quarterly fraction
    else
        markdown_q0 = clamp.(markdown_q0 ./ 1e4, -0.1, 0.1); mc = "$(mc)/1e4"  # bps -> quarterly fraction
    end
    log_status("  [BBL] markdown ρ^q ← $mc | β=$(a["beta"]) | T=$(a["horizon"]) | shocks=$(a["shocks"])")

    _, _, out_dir = get_paths(a["hpc"]; local_dir=a["local-dir"])

    # Forward r^f path for ψ4 (BCB market Selic curve via scrape_forward_rf.py). A FLAT path
    # makes ψ4 = r^f·Σβ^t Dep a rescaling of ψ2 = Σβ^t Dep (collinear) ⇒ ζ unidentified;
    # the time-varying curve breaks that. Fallback: flat panel median (warns).
    # --multi-start replaces this single curve with one curve per launch quarter, so the
    # single-curve file is then neither read nor required (it would only be an unused vector, and
    # requiring an unread upload on the cluster is a failure mode for nothing).
    ms = a["multi-start"]
    rf_path = ms ? Float64[] :
              load_forward_rf(a["rf-curve"], out_dir, a["horizon"], ctx; require=a["hpc"])

    # Asset return r^j in ψ1 (revenue). A firm-constant r^j is COLLINEAR with the ω (ψ2)
    # regressor, so it mainly RELABELS ω̂ — default 0 (deposit-funding value). A column
    # (e.g. gross_return_lag, a quarterly GROSS factor) or a constant net margin may be
    # supplied; both are read as the quarterly net margin (r^j − r^f) entering ψ1.
    asset_ret = zeros(nrow(ctx.df))
    if a["asset-return-col"] !== nothing
        col = a["asset-return-col"]
        col in names(ctx.df) || error("--asset-return-col '$col' not in demand parquet")
        # LAGGED level, to match the vintage of asset_gross_return_lag (= 1 + LAGGED quarterly
        # asset yield; see the producer contract at panel_bank_chars.py:441-444). The
        # contemporaneous `risk_free_qoq` must NOT be used here — it would introduce a
        # one-quarter vintage error on top of the level fix.
        rfq, rfq_c = _first_present_rf_level(ctx.df, RF_LEVEL_LAG_CANDIDATES; default=0.0,
                                             what="asset-margin r^f (lagged)")
        gr = Float64.(coalesce.(ctx.df[!, col], NaN)) .- 1.0
        asset_ret = gr .- rfq
        # A row with no reported asset yield (bank absent from the IF-Data bank-chars panel) must NOT
        # silently become r^j = 0 — that hands it a margin of −r^f (≈ −10.5pp/yr), the exact pathology
        # this column exists to remove. Impute the cross-sectional MEDIAN margin instead.
        fin = isfinite.(asset_ret)
        any(fin) || error("--asset-return-col '$col' has no finite rows (check units/merge)")
        med = median(asset_ret[fin]); n_imp = count(!, fin)
        asset_ret[.!fin] .= med
        log_status("  [BBL] r^j ← ($col − 1) − $rfq_c | median net margin " *
                   "$(round(med, sigdigits=3))/q = $(round(med*400, sigdigits=3)) pp/yr | " *
                   "$n_imp/$(length(asset_ret)) rows imputed at the median")
        med > 0 || @warn "  [BBL] median asset margin is NOT positive ($med) — deposits earn less " *
                         "than r^f, which will force ω̂ < 0. Check the asset-return column."
    elseif a["asset-margin"] != 0.0
        asset_ret = fill(a["asset-margin"], nrow(ctx.df))
        log_status("  [BBL] r^j − r^f = $(a["asset-margin"]) (constant)")
    end

    # ── MULTI-START × RATE RISK ────────────────────────────────────────────────────────────────
    # One pass over the panel with a PER-ROW forward curve IS S independent per-quarter
    # simulations: build_precomp keys a market (mca_code, time_id), so rows from different
    # quarters never share a share denominator and never meet in the deposit recursion. That is
    # exact, not an approximation, and it is what lets the S launch quarters — and the P rate
    # paths inside psi_under — ride on ONE share evaluation instead of S·P of them.
    # WHY IT MATTERS: within one launch quarter every firm discounts the SAME forward curve, so
    # ψ4 stays a rescaling of ψ2 and ω/ζ sit on a ridge. Crossing 36 vintage curves whose Selic
    # spans 2% to 14.25% is what separates them.
    starts = String[]; row_curve = Int[]; row_start = String[]
    rf_curve_paths = nothing
    rf_src = Dict{String,String}(); rf_bar = Dict{String,Float64}()
    nfirm_by_start = Dict{String,Int}()
    vpath = ""; tpath = ""
    n_paths = a["n-paths"]
    psi_tag = a["psi-tag"]
    # Vintage curves and rate risk exist only inside the multi-start pass. A flag that would be
    # silently ignored is an error, not a run that quietly did something else than it was asked.
    (ms || (n_paths == 1 && a["rf-vintages"] === nothing && a["transitions"] === nothing)) || error(
        "--n-paths ($n_paths) / --rf-vintages / --transitions require --multi-start; without it " *
        "every row is simulated on the single --rf-curve path and none of them is read.")
    if ms
        n_paths >= 1 || error("--n-paths must be ≥ 1 (got $n_paths)")
        Tms = a["horizon"]; βms = a["beta"]
        vpath = a["rf-vintages"] === nothing ?
                joinpath(cf_in_dir(out_dir, "COST_FWD"), "forward_rf_vintages.csv") : a["rf-vintages"]
        _, curves, anchor, srcmap = _load_rf_vintages(vpath, Tms; strict=a["hpc"])

        # Every panel row must land on a curve. `starts` is the intersection by construction —
        # sorted panel quarters, each of which is checked to exist in the CSV — and an unmatched
        # label is listed and thrown rather than dropped: dropping would quietly shrink the panel
        # the ψ basis is built on, which is exactly the kind of change that leaves no trace.
        labels, lab_src = _quarter_labels(ctx.df)
        starts = sort(unique(labels))
        unmatched = [q for q in starts if !haskey(curves, q)]
        isempty(unmatched) || error(
            "--multi-start: $(length(unmatched)) panel quarter(s) have no forward curve in " *
            "$(basename(vpath)): $(join(unmatched, ", ")). Extend the vintage file " *
            "(scrape_forward_rf.py --vintage-from/--vintage-to) or restrict the run with " *
            "--time-filter.")
        sidx = Dict(q => i for (i, q) in enumerate(starts))
        row_curve = [sidx[l] for l in labels]
        row_start = labels
        S_ms = length(starts)

        rf_mean = Matrix{Float64}(undef, S_ms, Tms)
        for (i, q) in enumerate(starts); rf_mean[i, :] .= curves[q]; end
        # r̄^f_s = Σ_h β^h r_{s,h} / Σ_h β^h over the FORWARD path h=1…T (h=0 is the realised
        # anchor, not part of the simulated horizon). It is the single number that summarises where
        # a launch quarter sits in the rate cycle, and the object whose spread ACROSS starts is the
        # ζ identification the solver then reads off.
        wsum = sum(βms^h for h in 1:Tms)
        for (i, q) in enumerate(starts)
            rf_bar[q] = sum(βms^h * rf_mean[i, h] for h in 1:Tms) / wsum
            rf_src[q] = get(srcmap, q, "")
        end

        # ANCHOR SELF-CHECK (see _MS_ANCHOR_TOL). CONTEMPORANEOUS r^f, not the lagged variant:
        # h=0 is dated at the launch quarter itself. The column is constant within a quarter
        # (verified: 1 distinct value per quarter on demand_4_index_time_spec_12), so the per-start
        # mean is the quarter's level and one pass over the rows suffices.
        rf_lvl, rf_lvl_c = _first_present_rf_level(ctx.df, RF_LEVEL_CANDIDATES; default=NaN,
                                                   what="anchor r^f (contemporaneous)")
        sum_lvl = zeros(S_ms); cnt_lvl = zeros(Int, S_ms)
        @inbounds for i in eachindex(rf_lvl)
            isfinite(rf_lvl[i]) || continue
            c = row_curve[i]; sum_lvl[c] += rf_lvl[i]; cnt_lvl[c] += 1
        end
        max_gap = 0.0; worst = ""; n_cmp = 0; no_anchor = String[]
        for (i, q) in enumerate(starts)
            (haskey(anchor, q) && cnt_lvl[i] > 0) || (push!(no_anchor, q); continue)
            g = abs(anchor[q] - sum_lvl[i] / cnt_lvl[i])
            n_cmp += 1
            g > max_gap && (max_gap = g; worst = q)
        end
        log_status("  [BBL] multi-start: $S_ms launch quarters ($(starts[1])…$(starts[end])) " *
                   "from $(basename(vpath)); row→quarter via $lab_src")
        log_status("  [BBL] anchor check vs $rf_lvl_c: $n_cmp/$S_ms quarters | " *
                   "max |h=0 − panel r^f| = $(round(max_gap, sigdigits=3)) at $worst " *
                   "(tol $(_MS_ANCHOR_TOL))")
        isempty(no_anchor) || @warn "  [BBL] no h=0 anchor (or no finite panel r^f) for " *
                                    "$(length(no_anchor)) quarter(s): $(join(no_anchor, ", "))"
        if max_gap > _MS_ANCHOR_TOL
            gapmsg = "multi-start anchor mismatch: max |h=0 − panel r^f| = " *
                     "$(round(max_gap, sigdigits=4)) at $worst > $(_MS_ANCHOR_TOL). The vintage " *
                     "file and the estimation panel disagree about which quarter a row is in, so " *
                     "every row would be simulated on the wrong point of the Selic cycle. Check " *
                     "the start_q convention in $(basename(vpath)) against $rf_lvl_c."
            a["hpc"] ? error(gapmsg) : @warn gapmsg
        end

        # Rate risk. sd stays at 0 for P=1, so _horizon_shock_paths returns the pure Focus mean
        # (z=0) and bbl_transitions.json is not even opened — a no-rate-risk multi-start run has
        # one fewer required upload.
        sd = zeros(Tms)
        if n_paths > 1
            tpath = a["transitions"] === nothing ?
                    joinpath(cf_in_dir(out_dir, "COST_FWD"), "bbl_transitions.json") : a["transitions"]
            sd, sd_mode, sd_hmax = _load_rate_sd_by_h(tpath, Tms)
            log_status("  [BBL] rate risk ← $(basename(tpath)) mode=$sd_mode | sd(h) over h=1…$Tms: " *
                       "$(round(sd[1], sigdigits=3))→$(round(sd[Tms], sigdigits=3)) " *
                       "(file h_max=$sd_hmax" *
                       (Tms > sd_hmax ? "; h>$sd_hmax repeats h=$sd_hmax)" : ")"))
        end
        rf_curve_paths, zdraw, nclip, zbias = _horizon_shock_paths(rf_mean, sd, n_paths, a["seed"])
        log_status("  [BBL] rf_curve_paths $(size(rf_curve_paths)) (P×S×T) | seed=$(a["seed"]) | " *
                   "z∈[$(round(minimum(zdraw), sigdigits=3)), $(round(maximum(zdraw), sigdigits=3))] " *
                   "Σz=$(round(sum(zdraw), sigdigits=3)) | $nclip/$(n_paths * S_ms * Tms) entries " *
                   "clipped at 0 | max |path mean − Focus| = $(round(zbias, sigdigits=3))")
        # The clip is one-sided (it can only raise a path), so a large max_bias means the paths no
        # longer average back to the Focus curve for the low-rate launch quarters — the very ones
        # carrying the r̄^f spread ζ is identified from. 1e-4/quarter is 0.04pp/yr, comfortably
        # below the 0.0092…0.0296 spread the 36 curves span, and 10x smaller than the anchor
        # tolerance the same quarters already have to clear.
        zbias > 1e-4 && @warn "  [BBL] the r^f≥0 floor moved the P-path mean off the Focus curve " *
                              "by up to $(round(zbias, sigdigits=3))/quarter " *
                              "($(round(400 * zbias, sigdigits=3)) pp/yr). It only ever raises a " *
                              "path, so the low-rate starts are biased UP and the r̄^f spread " *
                              "across launch quarters is compressed. Lower --n-paths (fewer " *
                              "extreme z) or take the shock to a log/shifted rate."
        # "_ms{P}", P = rate paths per deviation: the number bbl_run.sh stamps, so a standalone
        # run and a bbl_run.sh run name the same design alike.
        isempty(psi_tag) && (psi_tag = "_ms$(n_paths)")
    end

    # Equilibrium ψ
    σ̂ = equilibrium_spreads(ctx; policy_csv=a["policy-csv"])
    # keys_eq labels the ψ rows: a firm without --multi-start, (firm, launch quarter) with it.
    # fkey/skey split that label back apart. Without --multi-start psi_under returns the same two
    # values it always has and skey is the "all" filler accumulate_psi supplies, which is never
    # written out — so the single-curve artifacts keep their exact columns.
    keys_eq, fkey, skey = String[], String[], String[]
    if ms
        psi_eq, keys_eq, fkey, skey = psi_under(ctx, st, Z, markdown_q0, σ̂; beta=a["beta"],
                                                T=a["horizon"], asset_return_q=asset_ret,
                                                rf_curve_paths=rf_curve_paths,
                                                row_curve=row_curve, row_start=row_start)
    else
        psi_eq, keys_eq = psi_under(ctx, st, Z, markdown_q0, σ̂; beta=a["beta"], T=a["horizon"],
                                    asset_return_q=asset_ret, rf_path_q=rf_path)
        fkey = keys_eq; skey = fill("all", length(keys_eq))
    end
    isB = firm_is_B(ctx, fkey)
    log_status("  [BBL] ψ_eq: $(size(psi_eq)) over $(length(keys_eq)) firms " *
               "($(sum(isB)) B / $(sum(.!isB)) D)")

    # Per-start summary, ECHOED TO STDOUT because nothing is downloaded from the cluster: the
    # r̄^f spread across launch quarters is the identification claim, so it has to be legible in
    # the SLURM log even when no artifact ever comes back.
    if ms
        for s in skey; nfirm_by_start[s] = get(nfirm_by_start, s, 0) + 1; end
        # The ψ_eq line above counts KEYS, which under --multi-start is (firm × launch quarter);
        # this line separates the two so the log cannot be misread as a firm count.
        log_status("  [BBL] multi-start summary (beta=$(a["beta"]) T=$(a["horizon"]) " *
                   "S=$(length(starts)) P=$n_paths | $(length(keys_eq)) ψ keys over " *
                   "$(length(unique(fkey))) firms):")
        log_status("    start_q   rbar_f_q     rbar_f_pp_yr   n_firms   rf_source")
        for q in starts
            log_status("    " * rpad(q, 10) *
                       rpad(string(round(rf_bar[q], sigdigits=6)), 13) *
                       rpad(string(round(400 * rf_bar[q], digits=3)), 15) *
                       rpad(string(get(nfirm_by_start, q, 0)), 10) * get(rf_src, q, ""))
        end
        rb = [rf_bar[q] for q in starts]
        log_status("    rbar_f across starts: min=$(round(minimum(rb), sigdigits=5)) " *
                   "max=$(round(maximum(rb), sigdigits=5)) " *
                   "cv=$(round(std(rb) / mean(rb), sigdigits=4)) " *
                   "(the spread ζ is identified from)")
    end

    # ── Deviation ψ's — UNILATERAL (Nash) deviations, SHARDED over the (firm × Δ) grid ──
    # eq:17 is an MPE no-profitable-deviation condition: firm j deviates ALONE, rivals hold σ̂:
    #     g_jt = [ψ_j(σ̂) − ψ_j(σ̃_j, σ̂_{−j})]′·θ_c ≥ 0.
    # Shifting every firm together instead would be the COLLUSIVE direction — a common spread hike
    # is profitable for all — and would ask the estimator to certify a false inequality. Its
    # signature, if this ever regresses: exactly 50% of deviations "violate" g≥0 (the Δ>0 half),
    # the deposit response collapses to the industry-wide elasticity (no business stealing:
    # −0.10/pp instead of α≈−0.19/pp), and ω̂ runs to ≈ −0.8/quarter. See counterfactuals_plan.md §9.
    # The @assert below is the cheap standing guard on exactly that.
    # COST: one forward sim per (firm, Δ) instead of per Δ — so shard over the PAIR grid.
    # With ~300 choice firms × 50 Δ that is ~15k sims: use N_SHARDS ≫ 10 on the cluster.
    # `uf` is the UNIQUE firms; `keys_of_firm[j]` the ψ rows that belong to firm uf[j]. Without
    # --multi-start there is exactly one ψ row per firm and this is the identity map (keys_eq is
    # already the sorted unique firm list), so the pair grid, the sharding and every ψ row this
    # loop writes are unchanged. With it, one deviation sim yields the deviator's ψ under EVERY
    # launch quarter at once — which is why the shard index stays (firm × Δ) and never gains a
    # start dimension.
    nb = size(psi_eq, 2)
    uf = ms ? sort(unique(fkey)) : keys_eq
    nf = length(uf)
    keys_of_firm = [Int[] for _ in 1:nf]
    let kidx = Dict(f => j for (j, f) in enumerate(uf))
        for (r, f) in enumerate(fkey); push!(keys_of_firm[kidx[f]], r); end
    end
    S = a["shocks"]; nsh = a["n-shards"]; sid = a["shard-id"]
    (0 <= sid < nsh) || error("shard-id ($sid) must be in 0:$(nsh-1)")
    # Global shift vector (by index) — deterministic in grid mode ⇒ σ̃ is identical
    # whether or not the run is sharded, so shard outputs merge into the unsharded result.
    shifts = deviation_shifts(S, a["perturb-scale"], a["dev-scheme"], a["seed"])
    log_status("  [BBL] σ̃ scheme=$(a["dev-scheme"]) scale=$(a["perturb-scale"]) → " *
               "Δ∈[$(round(minimum(shifts), sigdigits=3)), $(round(maximum(shifts), sigdigits=3))]")
    rows_by_firm = firm_endog_rows(ctx, st, uf)
    # No panel row may be claimed by two firms: a shared row would move more than the deviator
    # and the moment would stop being unilateral.
    let seen = falses(length(σ̂))
        for rows in rows_by_firm, i in rows
            seen[i] && error("row $i is claimed by two firms — deviations would not be unilateral")
            seen[i] = true
        end
    end
    dev_firms = [j for j in 1:nf if !isempty(rows_by_firm[j])]
    log_status("  [BBL] UNILATERAL deviations: $(length(dev_firms)) of $nf firms set a choice spread")
    # Global (firm, shock) grid — deterministic ⇒ shard-invariant by global index.
    pairs = [(j, s) for j in dev_firms for s in 1:S]
    loc = [p for (i, p) in enumerate(pairs) if (i - 1) % nsh == sid]
    log_status("  [BBL] shard $sid/$nsh → $(length(loc)) of $(length(pairs)) (firm × Δ) sims")
    # One output row per (deviating firm, Δ, launch quarter). accumulate_psi builds its key list
    # from ctx.df and row_start alone — never from the spreads — so a deviation's ψ rows are in the
    # SAME order as the equilibrium's and `keys_of_firm` indexes both.
    nout = sum(length(keys_of_firm[j]) for (j, _s) in loc; init=0)
    psi_dev = Array{Float64,2}(undef, nout, nb)
    d_shock = Vector{Int}(undef, nout); d_firm = Vector{String}(undef, nout)
    d_start = Vector{String}(undef, nout); d_isB = Vector{Bool}(undef, nout)
    li = 0
    for (si, (j, s)) in enumerate(loc)
        σ̃ = apply_shift_firm(σ̂, rows_by_firm[j], shifts[s])       # only firm j moves
        @assert count(i -> !isequal(σ̃[i], σ̂[i]), eachindex(σ̃)) ==
                (shifts[s] == 0 ? 0 : length(rows_by_firm[j])) "σ̃ moved rows outside firm $(uf[j])"
        pd = if ms
            psi_under(ctx, st, Z, markdown_q0, σ̃; beta=a["beta"], T=a["horizon"],
                      asset_return_q=asset_ret, rf_curve_paths=rf_curve_paths,
                      row_curve=row_curve, row_start=row_start)[1]
        else
            psi_under(ctx, st, Z, markdown_q0, σ̃; beta=a["beta"], T=a["horizon"],
                      asset_return_q=asset_ret, rf_path_q=rf_path)[1]
        end
        for r in keys_of_firm[j]
            li += 1
            psi_dev[li, :] .= @view pd[r, :]                        # only the DEVIATOR's ψ
            d_shock[li] = s; d_firm[li] = uf[j]; d_start[li] = skey[r]; d_isB[li] = isB[r]
        end
        si % 25 == 0 && log_status("    [BBL] shard $sid: $si/$(length(loc)) sims done")
    end

    cost_dir = cf_out_dir(out_dir, "COST_FWD"); mkpath(cost_dir)   # cluster: data/output/bbl
    # `psi_tag` is "" unless --multi-start, so a single-curve run writes the filenames it always
    # has and a multi-start run cannot overwrite them (bbl_solve.py --psi-tag selects a vintage).
    tag = "E$(a["estim"])_spec_$(a["spec"])_$(a["stage"])$(a["suffix"])$psi_tag"
    blocks = vcat(["psi1", "psi2_omega"], ["psi3_gamma_$z" for z in znames], ["psi4_zeta"])

    # Equilibrium ψ is shard-invariant → write once (shard 0).
    if sid == 0
        eq_df = DataFrame(firm=fkey, is_B=collect(isB))
        if ms
            # start_q makes (firm, start_q) the alignment key bbl_solve.py joins on; rf_source and
            # rf_bar_beta travel with it so the solve can report WHICH vintage a launch quarter came
            # from and where in the rate cycle it sat without re-reading the curve file.
            eq_df[!, "start_q"]     = skey
            eq_df[!, "rf_source"]   = [get(rf_src, q, "") for q in skey]
            eq_df[!, "rf_bar_beta"] = [get(rf_bar, q, NaN) for q in skey]
        end
        for (j, b) in enumerate(blocks); eq_df[!, b] = psi_eq[:, j]; end
        _atomic_write(joinpath(cost_dir, "psi_eq_$tag.parquet")) do tmp
            Parquet2.writefile(tmp, eq_df)
        end
        log_status("  [BBL] wrote psi_eq_$tag.parquet")

        # Provenance sidecar: what the ψ files were simulated under. Written from shard 0 only,
        # for the same reason psi_eq is — it is shard-invariant.
        if ms
            meta = Dict("starts"      => starts,
                        "sources"     => [get(rf_src, q, "") for q in starts],
                        "rf_bar_beta" => [get(rf_bar, q, NaN) for q in starts],
                        "n_firms"     => [get(nfirm_by_start, q, 0) for q in starts],
                        "beta"        => a["beta"],
                        "T"           => a["horizon"],
                        "S"           => length(starts),
                        "P"           => n_paths,
                        "seed"        => a["seed"],
                        "psi_tag"     => psi_tag,
                        "rf_vintages" => basename(vpath),
                        "transitions" => isempty(tpath) ? "" : basename(tpath))
            _atomic_write(joinpath(cost_dir, "psi_starts_$tag.json")) do tmp
                open(tmp, "w") do f; JSON3.write(f, meta); end
            end
            log_status("  [BBL] wrote psi_starts_$tag.json (S=$(length(starts)) P=$n_paths " *
                       "seed=$(a["seed"]))")
        end
    end

    # Deviation ψ for this shard: ONE row per (deviating firm, Δ) pair — `firm` is the DEVIATOR
    # and the ψ blocks are that firm's own value under its unilateral move. GLOBAL shock ids, so
    # shards concatenate into the unsharded result (the Python solver globs psi_dev_*).
    # Under --multi-start a pair contributes one row per launch quarter the deviator appears in.
    nloc = length(loc)
    dev_df = DataFrame(shock=d_shock, firm=d_firm, is_B=d_isB)
    ms && (dev_df[!, "start_q"] = d_start)
    for (c, b) in enumerate(blocks)
        dev_df[!, b] = psi_dev[:, c]
    end
    shard_tag = nsh == 1 ? "" : "_shard$(sid)of$(nsh)"
    # Through a temporary + rename (see _atomic_write): this file's existence is what tells the
    # sweep and the solve that shard `sid` is done, so it must never exist half-written.
    _atomic_write(joinpath(cost_dir, "psi_dev_$tag$shard_tag.parquet")) do tmp
        Parquet2.writefile(tmp, dev_df)
    end
    log_status("  [BBL] wrote psi_dev_$tag$shard_tag.parquet ($nloc firm×Δ deviations" *
               (ms ? " → $nout firm×Δ×start rows)" : ")"))
    # This process's peak resident memory (Sys.maxrss is in bytes). Measurement only: a packed job
    # runs k shards in one allocation, where sacct's MaxRSS of the .batch step sums them, so each
    # shard reports its own on ONE greppable line that bbl_job.sh, bbl_sizing.sh and
    # bbl_status.sh --rss read.
    log_status("  [BBL] shard $sid peak RSS $(round(Sys.maxrss() / 2^30; digits=1)) GiB " *
               "(T=$(a["horizon"]), this process)")
    log_status("[DONE] estimation_bbl_2_fwd_sim shard $sid — run bbl_solve.py after ALL shards")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_cost2()
end
