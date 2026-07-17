"""
estimation_bbl_2_fwd_sim.jl
=================
BBL Step 2, part 1: forward-simulate the value-function basis ψ under the
EQUILIBRIUM strategy σ̂ and under a battery of DEVIATING strategies σ̃, then export
the firm-level ψ's for the eq:17 minimization (estimation_bbl_3_solve.py).

Shares come from the real RC demand (foundation_demand_eval), deposits from the real
law of motion (foundation_deposit_sim), and ψ from the exact basis (foundation_psi_basis)
— not the crude exp(δ-shift) share proxy with hard-coded r_f of the earlier prototype.

Pipeline
--------
  σ̂  (equilibrium)  → simulate deposits → accumulate ψ_eq         (per firm)
  σ̃₁…σ̃_S (deviations)→ simulate deposits → accumulate ψ_dev[s]     (per firm)
  export {ψ_eq, ψ_dev, firms, firm_is_B, Z_names}  → COST_FWD/psi_bbl_*.jls
  estimation_bbl_3_solve.py reads these and minimizes Σ min{g,0}² (eq:17).

DEVIATING STRATEGY σ̃ (`--dev-scheme`, default `grid`):
  σ̃ shifts the CHOICE spreads (k∈{4,5}) by Δ and holds the perturbed policy for the
  whole horizon (a stationary deviation, as in BBL forward simulation). `grid`: Δ takes
  a symmetric grid over [−scale,+scale] excluding 0, so both raising AND lowering are
  probed at graduated magnitudes — the eq:17 objective Σ min{g,0}² is only informative
  where a deviation binds, so directed small deviations pin the FOC far better than the
  tiny symmetric normals (`normal`, legacy) they replace. `--perturb-scale` is the grid
  half-width in annualized-ρ units (ρ=spread_ann/100).

  NOTE (deferred): the deviation is applied industry-wide (all firms shift together),
  not as a strict UNILATERAL deviation of firm j alone. The exact BBL object perturbs
  one firm at a time (N_firms× the share evals); left as a first-pass approximation.

FORWARD r^f (`--rf-curve`, default `COST_FWD/forward_rf_qoq.csv` from cf_forward_rf.py):
  the market Selic curve enters ψ4. A FLAT r^f makes ψ4 collinear with ψ2, leaving ζ
  unidentified; the time-varying curve separates ζ from ω.

ASSET RETURN r^j (`--asset-return-col` / `--asset-margin`, default 0): enters ψ1 (V_Main
  eq 16, row 1). Both flags are read as the QUARTERLY NET margin (r^j − r^f); foundation_psi_basis
  adds r^f back so ψ1 carries the GROSS r^j the paper requires.
    ⚠ DO NOT pass `--asset-return-col gross_return_lag`. Despite the name, that column is
      `1 + deposit_rate_lag` (estimation_1_demand_1_prep.py:167) — the rate the bank PAYS
      DEPOSITORS (liability side), not what it earns on assets. It sits BELOW r^f by
      construction (that gap IS the markdown this paper estimates); using it as r^j would
      hand the model a negative asset margin. There is currently NO asset-return column in
      the demand parquet — one must be built (COSIF asset yield; see counterfactuals_plan.md §9).
    Default 0 ⇒ r^j = r^f: deposits earn exactly the risk-free rate, so a deposit is worth
      only (ρ − c). That is a SUBSTANTIVE assumption (it says the marginal deposit funds
      reserves/govvies, not credit), and with the observed spreads it forces ω̂ < 0.
  A firm-constant r^j is collinear with the ω regressor (ψ2), so it shifts ω̂ one-for-one:
  the data cannot pin r^j down internally — it must be measured/calibrated from outside.

EQUILIBRIUM σ̂ (knob):
  Default σ̂ = observed spreads ρ̂ (the data IS the equilibrium). Pass
  `--policy-csv` to instead use the smoothed fitted policy from
  estimation_bbl_1_polfunc.py (polfunc_fitted_spec_*.csv).

⚠ COMPUTE: each σ̃ costs one deposit simulation (≈ one share evaluation when spreads
are held flat). With S deviations on the full panel at R=2000 this is the heavy,
GPU/cluster step. Develop locally with --R small, --time-filter one quarter, and
--shocks small; run headline on Bouchet.

Usage (write-only here; run only after data is downloaded AND author authorizes):
  julia --project=. --threads=4 estimation_bbl_2_fwd_sim.jl --estim 6 --spec 12 \\
      --stage extended --R 300 --time-filter 2024Q4 --shocks 20 --beta 0.9 --horizon 50
"""

include(joinpath(@__DIR__, "foundation_psi_basis.jl"))

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
# estimation_bbl_1_polfunc.py::compute_fitted_values predicts with missing regressors filled to 0
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

Parse the BBL Step-1 fitted-policy CSV (`polfunc_fitted_spec_*.csv` from
estimation_bbl_1_polfunc.py) into `(firm,mca,k,time) → fitted spread (annual pp)`, keeping only
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
                               "Re-run estimation_bbl_1_polfunc.py --spec <spec>.")
    end
    i_firm = ci["CodConglomeradoPrudencial"]; i_mca = ci["mca_code"]; i_k = ci["deposit_type"]
    i_isB = ci["is_B"]; i_t = ci["time_id"]
    i_k4B = ci["fitted_k4_Time_CDB_B"]; i_k4D = ci["fitted_k4_Time_CDB_D_optB"]
    i_k5B = ci["fitted_k5_Prepaid_B"]; i_k5D = ci["fitted_k5_Prepaid_D_optB"]
    ncol = length(hdr)
    m = Dict{String,Float64}()
    @inbounds for li in 2:length(lines)
        line = lines[li]
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

If `policy_csv` is given (the BBL Step-1 fitted policy, `polfunc_fitted_spec_*.csv`), the CHOICE rows
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
              "estimation_bbl_1_polfunc.py::compute_fitted_values fills missing regressors with 0, " *
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
    apply_shift(σ̂, endog, Δ) -> Vector{Float64}

Stationary deviation: add the scalar shift `Δ` (annualized) to the choice spreads
k∈{4,5}; regulated types pass through unchanged.
"""
function apply_shift(σ̂::Vector{Float64}, endog::BitVector, Δ::Float64)
    σ̃ = copy(σ̂)
    @inbounds for i in eachindex(σ̃); endog[i] && (σ̃[i] += Δ); end
    return σ̃
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
Contrast `apply_shift` above, which moves EVERY firm at once — that is a coordinated
(collusive) move, not a unilateral one, and must NOT be used to build the eq:17 moments.
"""
function apply_shift_firm(σ̂::Vector{Float64}, rows_j::Vector{Int}, Δ::Float64)
    σ̃ = copy(σ̂)
    @inbounds for i in rows_j; σ̃[i] += Δ; end
    return σ̃
end

"""
    load_forward_rf(path, out_dir, T, ctx) -> Vector{Float64}

Quarterly forward r^f path (length T) from the market Selic curve written by
cf_forward_rf.py (`COST_FWD/forward_rf_qoq.csv`, column `rf_qoq`), padded/truncated to
T. A FLAT r^f makes ψ4 = r^f·Σβ^t Dep a rescaling of ψ2 = Σβ^t Dep (collinear) so ζ is
unidentified; the time-varying curve breaks that. Falls back to the flat panel median
(with a warning) if the curve file is absent.
"""
function load_forward_rf(path::Union{Nothing,String}, out_dir::String, T::Int, ctx::CFDemandCtx;
                         require::Bool=false)
    csv = path === nothing ? joinpath(dirname(out_dir), "COST_FWD", "forward_rf_qoq.csv") : path
    if isfile(csv)
        lines = filter(l -> !isempty(strip(l)), readlines(csv))
        hdr = strip.(split(lines[1], ','))
        ci = findfirst(==("rf_qoq"), hdr)
        ci === nothing && error("rf_qoq column not found in $csv")
        rf = [parse(Float64, strip(split(l, ',')[ci])) for l in lines[2:end]]
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
          "    python cf_forward_rf.py --horizon $T --start 2026Q1"
    require && error(msg)
    rf_q0, rfc = _first_present(ctx.df, ["risk_free_qoq", "risk_free_qoq_lag", "selic_qoq"]; default=NaN)
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
"""
function psi_under(ctx::CFDemandCtx, st::DepositSimState, Z::Matrix{Float64},
                   markdown_q0::Vector{Float64}, spreads_ann::Vector{Float64};
                   beta::Float64=0.9, T::Int=50,
                   asset_return_q::Union{Nothing,Vector{Float64}}=nothing,
                   rf_path_q::Union{Nothing,Vector{Float64}}=nothing)
    sim = simulate_deposits(ctx, st; T=T, spreads_ann=spreads_ann)
    # Move the quarterly markdown with the scenario spread: Δρ^q ≈ Δρ_ann/4.
    # (Confirm annual→quarterly convention; only matters for k∈{4,5} deviations.)
    # markdown_q0 is the observed quarterly markdown (fraction); a scenario change in
    # the ANNUAL spread (pp = spread_ann/100) maps to a quarterly-fraction change via
    # /400 (×0.01 pp→fraction, ÷4 annual→quarter).
    markdown_q = markdown_q0 .+ (spreads_ann .- ctx.rho_hat) ./ 400.0
    res = accumulate_psi(ctx, st, sim.Dep, markdown_q, Z;
                         beta=beta, asset_return_q=asset_return_q, rf_path_q=rf_path_q)
    return res.psi_firm, res.firms
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
        "--beta";          arg_type = Float64; default = 0.9
        "--horizon";       arg_type = Int;     default = 50
        "--shocks";        arg_type = Int;     default = 50      # TOTAL number of σ̃ deviations
        # σ̃ grid half-width in annualized ρ units (pp). 2.0 = 200bp ≈ 54% of the median observed
        # choice spread (~3.7pp); the ± grid gives graduated magnitudes up to that. The old default
        # of 0.02 (=2bp) sat so deep in the linear regime that g carried no curvature at all
        # (even/|odd| of Δψ1 ≈ 0.002 vs 0.16 at 200bp), leaving the FOC as the only signal.
        "--perturb-scale"; arg_type = Float64; default = 2.0
        "--dev-scheme";    arg_type = String;  default = "grid"  # grid (directed) | normal (legacy)
        "--rf-curve";      arg_type = String;  default = nothing # forward-r^f CSV; default COST_FWD/forward_rf_qoq.csv
        "--asset-return-col"; arg_type = String; default = nothing # r^j source col (e.g. gross_return_lag)
        "--asset-margin";  arg_type = Float64; default = 0.0     # constant quarterly (r^j−r^f) if no col
        "--dbar";          arg_type = Float64; default = -1.0   # <=0 => per-type auto-calibrate
        "--time-filter";   arg_type = String;  default = nothing
        "--policy-csv";    arg_type = String;  default = nothing
        "--n-shards";      arg_type = Int;     default = 1       # split deviations across jobs
        "--shard-id";      arg_type = Int;     default = 0       # 0-based; = SLURM_ARRAY_TASK_ID
    end
    return parse_args(s)
end

function main_cost2()
    a = _parse_cost2_args()
    tf = a["time-filter"] === nothing ? nothing : String[a["time-filter"]]
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

    # Forward r^f path for ψ4 (BCB market Selic curve via cf_forward_rf.py). A FLAT path
    # makes ψ4 = r^f·Σβ^t Dep a rescaling of ψ2 = Σβ^t Dep (collinear) ⇒ ζ unidentified;
    # the time-varying curve breaks that. Fallback: flat panel median (warns).
    rf_path = load_forward_rf(a["rf-curve"], out_dir, a["horizon"], ctx; require=a["hpc"])

    # Asset return r^j in ψ1 (revenue). A firm-constant r^j is COLLINEAR with the ω (ψ2)
    # regressor, so it mainly RELABELS ω̂ — default 0 (deposit-funding value). A column
    # (e.g. gross_return_lag, a quarterly GROSS factor) or a constant net margin may be
    # supplied; both are read as the quarterly net margin (r^j − r^f) entering ψ1.
    asset_ret = zeros(nrow(ctx.df))
    if a["asset-return-col"] !== nothing
        col = a["asset-return-col"]
        col in names(ctx.df) || error("--asset-return-col '$col' not in demand parquet")
        rfq, _ = _first_present(ctx.df, ["risk_free_qoq", "risk_free_qoq_lag", "selic_qoq"]; default=0.0)
        gr = Float64.(coalesce.(ctx.df[!, col], NaN)) .- 1.0
        asset_ret = gr .- rfq
        # A row with no reported asset yield (bank absent from the IF-Data bank-chars panel) must NOT
        # silently become r^j = 0 — that hands it a margin of −r^f (≈ −10.5pp/yr), the exact pathology
        # this column exists to remove. Impute the cross-sectional MEDIAN margin instead.
        fin = isfinite.(asset_ret)
        any(fin) || error("--asset-return-col '$col' has no finite rows (check units/merge)")
        med = median(asset_ret[fin]); n_imp = count(!, fin)
        asset_ret[.!fin] .= med
        log_status("  [BBL] r^j ← ($col − 1) − r^f_q | median net margin " *
                   "$(round(med, sigdigits=3))/q = $(round(med*400, sigdigits=3)) pp/yr | " *
                   "$n_imp/$(length(asset_ret)) rows imputed at the median")
        med > 0 || @warn "  [BBL] median asset margin is NOT positive ($med) — deposits earn less " *
                         "than r^f, which will force ω̂ < 0. Check the asset-return column."
    elseif a["asset-margin"] != 0.0
        asset_ret = fill(a["asset-margin"], nrow(ctx.df))
        log_status("  [BBL] r^j − r^f = $(a["asset-margin"]) (constant)")
    end

    # Equilibrium ψ
    σ̂ = equilibrium_spreads(ctx; policy_csv=a["policy-csv"])
    psi_eq, firms = psi_under(ctx, st, Z, markdown_q0, σ̂; beta=a["beta"], T=a["horizon"],
                              asset_return_q=asset_ret, rf_path_q=rf_path)
    isB = firm_is_B(ctx, firms)
    log_status("  [BBL] ψ_eq: $(size(psi_eq)) over $(length(firms)) firms " *
               "($(sum(isB)) B / $(sum(.!isB)) D)")

    # ── Deviation ψ's — UNILATERAL (Nash) deviations, SHARDED over the (firm × Δ) grid ──
    # eq:17 is an MPE no-profitable-deviation condition: firm j deviates ALONE, rivals hold σ̂:
    #     g_jt = [ψ_j(σ̂) − ψ_j(σ̃_j, σ̂_{−j})]′·θ_c ≥ 0.
    # Until 2026-07-13 this loop called apply_shift(σ̂, st.endog, Δ), moving EVERY firm at once.
    # A common spread hike is the COLLUSIVE direction — profitable for all — so the estimator was
    # being asked to certify a false inequality. Symptoms: exactly 50% of deviations "violated"
    # g≥0 (the Δ>0 half), the deposit response collapsed to the industry-wide elasticity (no
    # business stealing: −0.10/pp instead of α≈−0.19/pp), and ω̂ ran to ≈ −0.8/quarter. See
    # counterfactuals_plan.md §9.
    # COST: one forward sim per (firm, Δ) instead of per Δ — so shard over the PAIR grid.
    # With ~300 choice firms × 50 Δ that is ~15k sims: use N_SHARDS ≫ 10 on the cluster.
    nf, nb = size(psi_eq)
    S = a["shocks"]; nsh = a["n-shards"]; sid = a["shard-id"]
    (0 <= sid < nsh) || error("shard-id ($sid) must be in 0:$(nsh-1)")
    # Global shift vector (by index) — deterministic in grid mode ⇒ σ̃ is identical
    # whether or not the run is sharded, so shard outputs merge into the unsharded result.
    shifts = deviation_shifts(S, a["perturb-scale"], a["dev-scheme"], a["seed"])
    log_status("  [BBL] σ̃ scheme=$(a["dev-scheme"]) scale=$(a["perturb-scale"]) → " *
               "Δ∈[$(round(minimum(shifts), sigdigits=3)), $(round(maximum(shifts), sigdigits=3))]")
    rows_by_firm = firm_endog_rows(ctx, st, firms)
    dev_firms = [j for j in 1:nf if !isempty(rows_by_firm[j])]
    log_status("  [BBL] UNILATERAL deviations: $(length(dev_firms)) of $nf firms set a choice spread")
    # Global (firm, shock) grid — deterministic ⇒ shard-invariant by global index.
    pairs = [(j, s) for j in dev_firms for s in 1:S]
    loc = [p for (i, p) in enumerate(pairs) if (i - 1) % nsh == sid]
    log_status("  [BBL] shard $sid/$nsh → $(length(loc)) of $(length(pairs)) (firm × Δ) sims")
    psi_dev = Array{Float64,2}(undef, length(loc), nb)
    for (li, (j, s)) in enumerate(loc)
        σ̃ = apply_shift_firm(σ̂, rows_by_firm[j], shifts[s])       # only firm j moves
        pd, _ = psi_under(ctx, st, Z, markdown_q0, σ̃; beta=a["beta"], T=a["horizon"],
                          asset_return_q=asset_ret, rf_path_q=rf_path)
        psi_dev[li, :] .= @view pd[j, :]                            # only the DEVIATOR's ψ
        li % 25 == 0 && log_status("    [BBL] shard $sid: $li/$(length(loc)) sims done")
    end

    cost_dir = joinpath(dirname(out_dir), "COST_FWD"); mkpath(cost_dir)
    tag = "E$(a["estim"])_spec_$(a["spec"])_$(a["stage"])$(a["suffix"])"
    blocks = vcat(["psi1", "psi2_omega"], ["psi3_gamma_$z" for z in znames], ["psi4_zeta"])

    # Equilibrium ψ is shard-invariant → write once (shard 0).
    if sid == 0
        eq_df = DataFrame(firm=firms, is_B=collect(isB))
        for (j, b) in enumerate(blocks); eq_df[!, b] = psi_eq[:, j]; end
        Parquet2.writefile(joinpath(cost_dir, "psi_eq_$tag.parquet"), eq_df)
        log_status("  [BBL] wrote psi_eq_$tag.parquet")
    end

    # Deviation ψ for this shard: ONE row per (deviating firm, Δ) pair — `firm` is the DEVIATOR
    # and the ψ blocks are that firm's own value under its unilateral move. GLOBAL shock ids, so
    # shards concatenate into the unsharded result (the Python solver globs psi_dev_*).
    nloc = length(loc)
    dev_df = DataFrame(shock=[s for (_j, s) in loc],
                       firm=[firms[j] for (j, _s) in loc],
                       is_B=[isB[j] for (j, _s) in loc])
    for (c, b) in enumerate(blocks)
        dev_df[!, b] = psi_dev[:, c]
    end
    shard_tag = nsh == 1 ? "" : "_shard$(sid)of$(nsh)"
    Parquet2.writefile(joinpath(cost_dir, "psi_dev_$tag$shard_tag.parquet"), dev_df)
    log_status("  [BBL] wrote psi_dev_$tag$shard_tag.parquet ($nloc firm×Δ deviations)")
    log_status("[DONE] estimation_bbl_2_fwd_sim shard $sid — run estimation_bbl_3_solve.py after ALL shards")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_cost2()
end
