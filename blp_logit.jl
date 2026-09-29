"""
blp_logit.jl
==============
Non-random-coefficients logit demand estimation for BLP (θ₂ = 0).

The sleepiness function was re-estimated locally, changing the demand-prep outputs. The
estimation routines are **auto-discovered** from the demand-prep parquets (see
`discover_estim_strategies()` — id AND prefix, newest file per id wins), so a relabelled
or new routine needs no code change, just its `demand_<id>_*_spec_12.parquet`. The
lineup (base links + their +Time variants):

  E1 Local B-type   E2 Pooled Linear
  E3 Pooled Single-Index        E4 Pooled Single-Index + Time

Each parquet already carries every column the logit needs (spread_ann in bps, share_D /
share_B_cond, is_B, deposit_type, CodConglomeradoPrudencial, the X_COLS, and all
LOO/cost/capital instruments), so no separate "finalization" step is required. The
link-based routines (E3+) are produced by sleep_demand_prep_link.py, which mirrors
the linear demand prep exactly (same columns/scaling), so the logit treats every routine
identically.

Run all discovered routines with `julia blp_logit.jl`, or a single one with
`julia blp_logit.jl --est 3`.

Four sub-models per routine (the in-file LaTeX table generator below reads these keys):
  (a) priceonly:           δ = α · spread
  (b) core:                δ = α · spread + X_core · β
  (c) full:                δ = α · spread + X · β
  (d) full_dtype:          δ = α · spread + X · β + γ · dummy_D_type

Outputs cluster-robust standard errors (conglomerate clustering), GMM Q-values, and
IK2016 effective clusters G*.

THE ESTIMATOR (and why it is the θ₂ = 0 nest of the RC model)
  δ is the Berry (1994) inversion δ_jkmt = ln s_jkmt − ln s_0mt, with the outside share
  recovered in closed form from the SAME market structure the RC engine integrates over —
  see `berry_inversion`, which reproduces the engine's θ₂ = 0 contraction fixed point to
  ~1e-15. θ₁ is then OLS of δ on [spread_hat, X]: only the k = 4,5 spread is endogenous and
  it carries its own per-type first-stage projection as the instrument, while the product
  characteristics are exogenous. That is exactly `estimate_theta1` in blp_engine_cpu.jl, so
  running this file and the RC engine's `logit` stage on one parquet must return the same α.
  Both properties were absent before 2026-09: δ omitted −ln s_0, and the linear step projected
  the whole design (characteristics included) onto the 16 EXCLUDED instruments.

Usage
-----
  # All discovered routines + combined summary + LaTeX tables:
  julia --project=. --threads=auto blp_logit.jl

  # A single routine:
  julia --project=. blp_logit.jl --est 3

  # Rebuild all LaTeX tables from the existing combined summary (no estimation):
  julia --project=. blp_logit.jl --tables-only

  # On the cluster (HEAD/data tree): parquets from data/output/demand_prep, everything this
  # step produces into data/output/logit — δ warm-starts, .jls fits, summary and .tex:
  julia --project=. --threads=auto blp_logit.jl --est 3 --hpc

LaTeX outputs (→ ESTIMATION_OUTPUT/Rout + Drafts/Deposit Competition; data/output/logit on
the cluster, where the Drafts folder does not exist):
  est{id}_spec12_logit.tex                 per-routine, 4 sub-model columns
  est1-4_spec12_logit_comparison.tex       cross-routine, `Price + Chars` column each —
                                           the sub-model whose X matches the BLP X₁
                                           (tab:demand_logit_spec12_comparison)

References
----------
  Berry, Levinsohn & Pakes (1995, Econometrica)
  Conlon & Gortmaker (2020, RAND J. Econ.)
  Egan, Hortaçsu & Matvos (2017, AER)
  Imbens & Kolesár (2016, REStat)
"""

using Parquet2, DataFrames, LinearAlgebra, Statistics
using JSON3, Serialization, Printf, Dates
using Distributions   # t p-values for the LaTeX result tables
using Random          # MersenneTwister for the wild cluster bootstrap
using TOML            # stdlib (resolves from @stdlib): config/table_notes.toml, the shared table notes

include(joinpath(@__DIR__, "of_root.jl"))
# The lineup, the \ref map and the demand prefixes come from config/routines.toml, which
# utils/routines.py reads too. Guarded because several files in one session want the consts.
isdefined(Main, :ROUTINE_REGISTRY) || include(joinpath(@__DIR__, "routines.jl"))

# ==========================================================================
# 0. Constants
# ==========================================================================
const SPEC_ID = SPEC12_ID

const X_COLS = ["fgc_covered", "has_ip", "seg_S2", "seg_S3", "seg_S4", "seg_S5",
                "log_total_assets_lag", "is_state_owned"]
const CORE_COLS = ["fgc_covered", "has_ip", "log_total_assets_lag"]

# Full instrument set (15 = these 11 + IV_COST + IV_CAPITAL). A `mean_loo_`-trimmed set was TESTED
# 2026-07-09 and REJECTED: dropping the aggregates roughly doubled the logit α SE and made it
# insignificant (t −2.2→−0.75), even though the subsample eff-F rose (13→21, a mechanical dilution
# effect) — the `mean_loo_` contribute identifying variation despite their collinearity. Kept only
# as a robustness discussion (review §4). See blp_weak_iv.py PARSIMONIOUS_IV for the diagnostic.
const IV_BLP_LOO = ["loo_log_assets", "mean_loo_log_assets",
                    "loo_equity_ratio", "mean_loo_equity_ratio",
                    "loo_basileia", "mean_loo_basileia",
                    "loo_credit_assets", "mean_loo_credit_assets",
                    "loo_npl_provision", "mean_loo_npl_provision",
                    "n_rivals"]
const IV_COST    = ["personnel_cost_ratio_lag", "admin_cost_ratio_lag",
                    "tax_cost_ratio_lag"]
const IV_CAPITAL = ["indice_basileia_lag"]
# ESTBAN branch-competition IV (panel_10): log1p lagged RIVAL-branch count in the MCA. The only
# instrument with within-conglomerate variation (across municipalities) — independent + relevant for
# demand-deposit spreads; validated against reserve requirements (rejected: macro-endogenous).
const IV_ESTBAN  = ["estban_rival_branches_lag"]

# Estimation routines are AUTO-DISCOVERED from the demand-prep parquets — see
# discover_estim_strategies() / const ESTIM_STRATEGIES below (defined after get_paths()).

# Sub-model definitions (keys consumed by the in-file LaTeX table generator below)
const SUB_MODELS = [
    (name="priceonly",  xcols=String[],    add_dtype=false),
    (name="core",       xcols=CORE_COLS,   add_dtype=false),
    (name="core_dtype", xcols=CORE_COLS,   add_dtype=true),   # Price + Core + D-Type (no seg/"Chars")
    (name="full",       xcols=X_COLS,      add_dtype=false),
    (name="full_dtype", xcols=X_COLS,      add_dtype=true),
]

# ==========================================================================
# 0c. Standard-error method (shared convention with the RC engine + sleepiness)
# ==========================================================================
# `se_method`, `wcb_reps`, `wcb_scheme`, `wild_weights` and `wcb_se` come from the SHARED module
# blp_se_common.jl — the same definitions blp_engine_cpu.jl / blp_engine_gpu.jl use, so every
# demand SE in the paper is produced by one implementation. This file used to carry a private
# copy of them; the copy had drifted, referencing a NORMAL null distribution where the shared
# `wcb_se` takes a Student-t(G*) few-cluster reference (`dof`), which is the convention every
# other table in the paper reports. Guarded because a session that has already included an
# engine has these definitions.
isdefined(Main, :gmm_cluster_ses) || include(joinpath(@__DIR__, "blp_se_common.jl"))

# BLP_LOGIT_INTERCEPT=1 adds a constant to the logit design. It is OFF by default and should
# stay off for anything the paper reports: the published logit column is the literal θ₂ = 0
# nest of the RC model, and the RC engines' X₁ (`X_COLS` in blp_engine_cpu.jl) carries NO
# constant, so an intercept here would break that nesting — and the equality check against the
# engine's own logit stage with it. Deposit-type / quarter fixed effects (which nest a constant)
# belong to the identification battery, not to this estimator.
logit_intercept() = get(ENV, "BLP_LOGIT_INTERCEPT", "0") == "1"

# ==========================================================================
# 0b. Paths
# ==========================================================================
"""Whether this run targets the cluster tree. `--hpc` is the flag blp_engine_cpu.jl takes;
SLURM's own job environment is honoured as well, so a job step that omits the flag still
resolves data/{input,output} instead of throwing inside `resolve_of_root()` (the local root
validation cannot pass on a compute node). Neither holds on a local run."""
is_hpc_run()::Bool = ("--hpc" in ARGS) || haskey(ENV, "SLURM_JOB_ID")

"""
    get_paths([is_hpc]) -> (input_dir, output_dir)

`input_dir` is the primary demand-prep location, `output_dir` the results root — the same
two-tuple every call site here already destructures.

On the cluster these are HEAD/data/{input,output}, mirroring `get_paths` in
blp_engine_cpu.jl. The demand parquets are read through `demand_dirs()`, which resolves to
the one directory the prep step writes. Everything produced here lands in the output root's
`logit/` step folder — the δ warm-starts (`delta_dir`), the summary and per-key JLS
(`logit_dir`), the LaTeX fragments (`tex_out_dir`) — so the step is one directory to package,
download and read back.

Off the cluster the tree is ESTIMATION_OUTPUT/{DEMAND_PREP, BLP_RESULTS}, `demand_dirs()` is
the single DEMAND_PREP entry, and every path resolves as it does with no flag.
"""
function get_paths(is_hpc::Bool = is_hpc_run())
    if is_hpc
        return joinpath(@__DIR__, "..", "data", "input"),
               joinpath(@__DIR__, "..", "data", "output")
    end
    data_dir  = joinpath(resolve_of_root(), "BCB", "Egan_et_al_2025_Rep", "processed")
    input_dir = joinpath(data_dir, "ESTIMATION_OUTPUT", "DEMAND_PREP")
    output_dir= joinpath(data_dir, "ESTIMATION_OUTPUT", "BLP_RESULTS")
    return input_dir, output_dir
end

"""Demand-prep search path (`demand_search_dirs`, of_root.jl): one directory in either tree —
on the cluster the single one the prep step writes, `data/output/demand_prep`; off it the local
ESTIMATION_OUTPUT/DEMAND_PREP. Resolution is PER FILE (`load_spec_data`) and per id
(`discover_estim_strategies`), and with a single candidate neither can pick up a second vintage
of a routine that the RC stack would then disagree with."""
demand_dirs() = demand_search_dirs(get_paths()...)

# Logit outputs live in a dedicated `logit/` subfolder of the results root, kept separate from
# the RC outputs (see cluster_ingest_blp.py). `logit_dir(out_dir)` in of_root.jl is the single
# definition of that path — the RC engine resolves the warm-start deltas through the same call —
# and this method pins it to this run's output root and creates it on demand.
function logit_dir()
    d = logit_dir(get_paths()[2])
    isdir(d) || mkpath(d)
    return d
end

"""Directory for the δ warm-starts: `logit_dir()` in either tree. The RC engine resolves
`logit_delta_E{k}_spec_{s}.{bin,jls}` through the same `logit_dir(out_dir)` helper, so writer
and reader name one directory and a warm start cannot land somewhere the engine never looks."""
delta_dir() = logit_dir()

"""Directory for the LaTeX tables. On the cluster it is `logit_dir()`: the fragments are
products of this step, so they belong with the δ warm-starts, the .jls fits and the summary in
the one folder the archiver packages and the download brings back whole. Locally they join the
rest of the paper's tables in ESTIMATION_OUTPUT/Rout, a sibling of the results root, which is
where the exporters mirror from."""
function tex_out_dir()
    out_dir = get_paths()[2]
    d = is_cluster_out(out_dir) ? logit_dir(out_dir) :
                                  joinpath(dirname(out_dir), "Rout")
    mkpath(d)
    return d
end

# Canonical combined-summary path.
combined_summary_path() = joinpath(logit_dir(),
                                   "logit_summary_spec_$(SPEC_ID).json")

"""
    discover_estim_strategies() -> Vector of (id, label, prefix)

Auto-discover the estimation routines for spec SPEC_ID by scanning the
demand-prep search path (`demand_dirs()`) for `demand_<id>_*_spec_<SPEC_ID>.parquet`,
excluding the legacy `*_final_*` files. The prefix is the filename minus the
`_spec_<SPEC_ID>.parquet` tail (e.g. `demand_1`, `demand_3_index`). Sorted by id.

Newly-added/relabelled routines are picked up with NO code change. If MULTIPLE non-final
parquets exist for the same id (e.g. a leftover old-scheme file alongside a freshly
rebuilt one), the **most recently modified** wins — so a stale file can't shadow the new
one. (Still: deleting old `demand_*_spec_<SPEC_ID>.parquet` after a relabel is tidiest.)
That mtime contest runs WITHIN a directory: an id already resolved from an earlier entry of
the search path is not reconsidered, so an uploaded parquet outranks a cluster-produced one
whatever their timestamps say. Off the cluster the path has one entry and the two rules
coincide.
"""
function discover_estim_strategies()
    best = Dict{Int, Tuple{Float64, String}}()   # id => (mtime, prefix); newest wins
    pat  = Regex("^demand_(\\d+)(?:_.*)?_spec_$(SPEC_ID)\\.parquet\$")
    for input_dir in demand_dirs()
        isdir(input_dir) || continue
        claimed = Set(keys(best))                # ids resolved from an earlier directory
        for f in readdir(input_dir)
            occursin("_final_", f) && continue
            m = match(pat, f)
            m === nothing && continue
            id = parse(Int, m.captures[1])
            id in claimed && continue
            mt = mtime(joinpath(input_dir, f))
            if !haskey(best, id) || mt > best[id][1]
                best[id] = (mt, replace(f, "_spec_$(SPEC_ID).parquet" => ""))
            end
        end
    end
    # Auto-discovery is by FILE PRESENCE, so a parquet left over from a routine that is no
    # longer in the lineup silently re-enters the run: on 2026-08-05 such files were rebuilt
    # from the new panel while carrying a stale phi, i.e. mixed-vintage inputs that look
    # current by mtime. This filter is the guard — the logit cannot pick up an id outside the
    # active set even if its parquet exists. `active_from_env` (routines.jl) is the same
    # reader sleep_demand_prep.py / sleep_export_all.py use through utils/routines.py,
    # over the same config/routines.toml, so the three cannot drift apart.
    active = Set(active_from_env())
    ids = sort!(collect(keys(best)))
    skipped = [id for id in ids if !(id in active)]
    if !isempty(skipped)
        skipped_str = join(["E$id" for id in skipped], ", ")
        @info "discover_estim_strategies: skipping $skipped_str (not in SLEEP_ACTIVE_ESTS); " *
              "their parquets are present but deliberately excluded."
    end
    return [(id = id, label = "E$id", prefix = best[id][2]) for id in ids if id in active]
end

const ESTIM_STRATEGIES = discover_estim_strategies()

# ==========================================================================
# 1. Data Loading
# ==========================================================================
"""Load the demand-prep parquet for one routine (drops the old `_final`), from the first
entry of `demand_dirs()` that holds it."""
function load_spec_data(estim)
    fname = "$(estim.prefix)_spec_$(SPEC_ID).parquet"
    dirs  = demand_dirs()
    path  = resolve_in_then_out(fname, dirs...)
    path === nothing && error("Missing demand-prep parquet: $fname. Searched:\n" *
                              describe_search_dirs(dirs...))
    df = DataFrame(Parquet2.Dataset(path); copycols=true)
    return df
end

# ==========================================================================
# 1b. Mean utility: the Berry (1994) inversion, in the engine's market structure
# ==========================================================================
"""
    berry_inversion(df) -> (delta, diag)

Mean utility δ from the data shares, inverted in the SAME market structure the RC engine
integrates over (`compute_model_shares!`, blp_engine_cpu.jl): inside a market (m,t) the choice
set is the outside option, the B products of that market, and EVERY D product of that quarter,
while a D product's reported share is the population-weighted average of its local shares
across markets (eq. B-3-D).

Write `A_mt = Σ_{j∈B(m,t)} e^{δ_j}`, `C_t = Σ_{j∈D(t)} e^{δ_j}` and let `g_mt = 1/(1+A_mt+C_t)`
be the outside share. The data then satisfy

    S^B_mt ≡ Σ_{j∈B(m,t)} s_j = A_mt · g_mt              (local,    observed)
    S^D_t  ≡ Σ_{j∈D(t)}  s_j = C_t · Σ_m w_mt · g_mt     (national, observed)

with `w_mt` the engine's own D-aggregation weights (`pop_weights` in `build_precomp`: the
per-pair mean of `banked_correction · pop_total`, normalised within the quarter). Eliminating
A and C leaves a CLOSED FORM — no contraction is needed at θ₂ = 0:

    K_t  = Σ_m w_mt (1 − S^B_mt)
    G_t  ≡ Σ_m w_mt g_mt = K_t − S^D_t
    g_mt = (1 − S^B_mt) · G_t / K_t
    δ_j  = ln s_j − ln g_mt   (B rows)      δ_j = ln s_j − ln G_t   (D rows)

VERIFIED against the engine's own kernel: pushing this δ back through the θ₂ = 0 share map
returns the data shares to 2.8e-16 (B) and 6.6e-16 (D) on E3/spec 12 — it *is* the fixed point
the contraction converges to, so the logit is the exact θ₂ = 0 nest of the RC model.

Why this replaced `δ = ln s_j`: the missing `− ln s_0` is not a constant. On E3/spec 12 it has
mean 0.45, sd 0.72 and range [0, 3.11] — it varies with market structure (it is a function of
how much of the market the incumbents already hold), so dropping it is an omitted regressor
correlated with the spread, not an intercept the design absorbs.
"""
function berry_inversion(df::DataFrame)
    N    = nrow(df)
    is_B = BitVector(Bool.(coalesce.(df.is_B, false)))
    mca  = string.(df.mca_code)
    tid  = string.(df.time_id)
    sB   = Float64.(coalesce.(df.share_B_cond, NaN))
    sD   = Float64.(coalesce.(df.share_D,      NaN))
    pop  = Float64.(coalesce.(df.pop_total, 0.0))
    bc   = Float64.(coalesce.(df.banked_correction, 1.1))   # engine default (build_precomp)

    # Quarter index (every row) and market index (B rows only: a D row's mca_code is NATIONAL).
    time_idx = Dict{String,Int}(); row_time = zeros(Int, N); n_times = 0
    for i in 1:N
        j = get(time_idx, tid[i], 0)
        if j == 0; n_times += 1; time_idx[tid[i]] = n_times; j = n_times; end
        row_time[i] = j
    end
    pair_idx = Dict{Tuple{String,String},Int}(); row_pair = zeros(Int, N); n_pairs = 0
    for i in 1:N
        is_B[i] || continue
        k = (mca[i], tid[i]); j = get(pair_idx, k, 0)
        if j == 0; n_pairs += 1; pair_idx[k] = n_pairs; j = n_pairs; end
        row_pair[i] = j
    end
    n_pairs == 0 && error("berry_inversion: no B rows — cannot build the market structure.")

    pair_time = zeros(Int, n_pairs)
    SB        = zeros(n_pairs)      # Σ_j share_B_cond within the market
    bcpop     = zeros(n_pairs)      # Σ over B rows of bc·pop, divided below by the row count
    n_rows_p  = zeros(Int, n_pairs)
    @inbounds for i in 1:N
        is_B[i] || continue
        p = row_pair[i]
        pair_time[p] = row_time[i]
        isfinite(sB[i]) && (SB[p] += sB[i])
        bcpop[p]    += bc[i] * pop[i]
        n_rows_p[p] += 1
    end
    bcpop ./= max.(n_rows_p, 1)                       # per-pair mean, exactly as build_precomp
    tot_bcpop = zeros(n_times)
    @inbounds for p in 1:n_pairs; tot_bcpop[pair_time[p]] += bcpop[p]; end
    w = @inbounds [bcpop[p] / max(tot_bcpop[pair_time[p]], 1e-30) for p in 1:n_pairs]

    K = zeros(n_times)
    @inbounds for p in 1:n_pairs; K[pair_time[p]] += w[p] * (1.0 - SB[p]); end
    SD = zeros(n_times)
    @inbounds for i in 1:N
        (!is_B[i] && isfinite(sD[i])) && (SD[row_time[i]] += sD[i])
    end
    G = K .- SD                                        # G_t = Σ_m w_mt g_mt

    bad = findall(t -> !(G[t] > 0.0), 1:n_times)
    isempty(bad) || error("berry_inversion: Σ_m w·(1−S^B) ≤ Σ_j s^D in $(length(bad)) quarter(s) " *
                          "(e.g. t=$(first(bad)): K=$(K[first(bad)]), S^D=$(SD[first(bad)])). " *
                          "The implied outside share is non-positive, so δ is undefined — the " *
                          "market-size construction in the demand prep needs revisiting.")

    g_pair = @inbounds [clamp((1.0 - SB[p]) * G[pair_time[p]] / max(K[pair_time[p]], 1e-30),
                              1e-12, 1.0) for p in 1:n_pairs]
    delta  = zeros(N)
    @inbounds for i in 1:N
        delta[i] = is_B[i] ? log(clamp(sB[i], 1e-15, Inf)) - log(g_pair[row_pair[i]]) :
                             log(clamp(sD[i], 1e-15, Inf)) - log(G[row_time[i]])
    end

    diag = (n_pairs = n_pairs, n_times = n_times,
            s0_min = minimum(g_pair), s0_med = median(g_pair), s0_max = maximum(g_pair),
            G_min = minimum(G), G_max = maximum(G))
    return delta, diag
end

# ==========================================================================
# 2. Regressor and IV Construction
# ==========================================================================
"""Build spread vector, product-characteristics matrix, and IV matrix."""
function build_matrices(df::DataFrame, xcols::Vector{String}, add_dtype::Bool)
    N = nrow(df)

    # Spread in percentage points (÷100 to convert from bps stored in parquet)
    spread = Float64.(coalesce.(df.spread_ann, 0.0)) ./ 100.0

    # Product characteristics matrix X. A constant is prepended only under
    # BLP_LOGIT_INTERCEPT=1 (off by default — see `logit_intercept`), so the reported design
    # is exactly the RC engines' X₁.
    use_const = logit_intercept()
    n_x = length(xcols) + (add_dtype ? 1 : 0) + (use_const ? 1 : 0)
    X   = zeros(N, n_x)
    off = 0
    if use_const
        X[:, 1] .= 1.0
        off = 1
    end
    for (i, col) in enumerate(xcols)
        if col in names(df)
            X[:, off + i] .= Float64.(coalesce.(df[!, col], 0.0))
        end
    end
    if add_dtype
        # D-type dummy: is_B == false (i.e., national D-type institutions)
        is_B = Bool.(coalesce.(df.is_B, false))
        X[:, n_x] .= Float64.(.!is_B)
    end

    # Full regressor matrix: [spread, X]
    X_full = hcat(spread, X)

    # IV matrix
    all_iv_names = vcat(IV_BLP_LOO, IV_ESTBAN, IV_COST, IV_CAPITAL)

    # A missing instrument COLUMN used to be dropped in silence, which is how the entire IV_BLP_LOO
    # block vanished when a market_panel rebuild skipped panel_loo_instruments.py (it OVERWRITES
    # market_panel.csv with the LOO instruments + FGC dummy). The logit then ran on the cost shifters
    # alone. A missing column is a broken panel, not a modelling choice — say so.
    _absent = [c for c in all_iv_names if !(c in names(df))]
    if !isempty(_absent)
        error("Demand parquet is missing instrument column(s): " * join(_absent, ", ") *
              ".\nRebuild in order: panel_6 → panel_7 (LOO instruments + FGC; OVERWRITES " *
              "market_panel.csv) → panel_9 --patch-market, then re-run the demand prep.")
    end

    # Zero-variance IVs are still dropped (a constant column carries no identifying information and
    # would make Z'Z singular) — but say WHICH, so a silently-degenerate instrument is visible.
    _zerovar = [c for c in all_iv_names if
        std(replace(Float64.(coalesce.(df[!, c], 0.0)), Inf=>0.0, -Inf=>0.0)) <= 1e-10]
    if !isempty(_zerovar)
        @warn "Dropping zero-variance instrument(s): " * join(_zerovar, ", ")
    end
    iv_avail = [c for c in all_iv_names if !(c in _zerovar)]

    Z = zeros(N, length(iv_avail))
    for (i, col) in enumerate(iv_avail)
        v = Float64.(coalesce.(df[!, col], 0.0))
        replace!(v, Inf=>0.0, -Inf=>0.0)
        Z[:, i] .= v
    end

    return spread, X, X_full, Z, iv_avail
end

"""First-stage projection of endogenous spreads for k=4,5."""
function project_spreads(spread::Vector{Float64}, X::Matrix{Float64},
                         Z::Matrix{Float64}, dep_types::Vector{Int})
    spread_hat = copy(spread)
    H = hcat(X, Z)  # full set of exogenous regressors + instruments

    for k_val in [4, 5]
        k_mask = dep_types .== k_val
        any(k_mask) || continue
        H_k      = H[k_mask, :]
        spread_k = spread[k_mask]
        valid    = all.(isfinite, eachrow(H_k)) .& isfinite.(spread_k)
        sum(valid) > size(H_k, 2) || continue
        try
            beta_fs = H_k[valid, :] \ spread_k[valid]
            spread_hat[k_mask] .= H_k * beta_fs
        catch; end
    end
    return spread_hat
end

# ==========================================================================
# 3. 2SLS/IV Estimation with Cluster-Robust SEs
# ==========================================================================
"""
Linear step θ₁ given the generated instrument: OLS of δ on `X_hat = [spread_hat, X]`.

This is the RC engines' own θ₁ step (`estimate_theta1`, blp_engine_cpu.jl) and it encodes the
paper's identification design (Appendix B, step 4): **only the spread on the bank-set deposit
types k = 4,5 is endogenous**, and it is instrumented by its own per-type first-stage
projection `spread_hat` (built on `H = [X, Z]` by `project_spreads`); the regulated types 1–2
enter with the raw spread, and every product characteristic is exogenous — its own instrument.
Because the per-type projection makes `spread − spread_hat` orthogonal to `[spread_hat, X]`
block by block, `X̂′X_full = X̂′X̂` exactly, so this OLS *is* exactly-identified IV
(Frisch–Waugh–Lovell) and the sandwich below is the corresponding 2SLS sandwich.

It replaced a projection of the WHOLE design onto the 16 EXCLUDED instruments, i.e.
`θ̂ = (X̃′X)⁻¹X̃′δ` with `X̃ = Z(Z′Z)⁻¹Z′X`. That treated `fgc_covered`, the segment dummies,
`log_total_assets_lag` and `is_state_owned` as endogenous — instrumenting exogenous regressors
with rival accounting variables, which is neither the paper's design nor the engines', and
made this file a different estimator from the RC model it is supposed to nest.

`Q` is retained as a DIAGNOSTIC: the one-step GMM criterion `ḡ′Wḡ` on the 16 excluded moments
with `W = (Z′Z/N)⁻¹`, exactly as the engines form it. It is not a test statistic: `W` is not the
inverse of the clustered moment covariance and `ḡ` is not scaled by the variance of ξ, so `N·Q`
(= ξ′P_Z ξ) is neither a Hansen J nor a Sargan statistic, and the tables print `Q` with no χ²
reference. `q_df = L − 1` counts the overidentifying restrictions (the estimator sets the
X-moments to zero exactly and pins only the `spread_hat` direction of the Z-space); it is stored
in the summary but not printed.

Returns `(theta1, se, pval, xi, Q, q_df, n_clusters, G_star)`.
"""
function estimate_theta1_logit(delta::Vector{Float64}, X_full::Matrix{Float64},
                               X_hat::Matrix{Float64}, Z::Matrix{Float64},
                               clusters::Vector{String})
    N, K = size(X_full)
    L    = size(Z, 2)

    # Validity mask
    valid = all.(isfinite, eachrow(X_hat)) .& isfinite.(delta) .&
            all.(isfinite, eachrow(Z))

    X_v  = X_hat[valid, :]
    Xf_v = X_full[valid, :]
    d_v  = delta[valid]
    Z_v  = Z[valid, :]
    cl_v = clusters[valid]
    N_v  = sum(valid)

    # --- θ₁ = (X̂′X̂)⁻¹ X̂′δ  (= exactly-identified IV; see the docstring) ---
    XhX     = X_v' * X_v
    XhX_inv = try inv(XhX) catch; pinv(XhX) end
    theta1  = XhX_inv * (X_v' * d_v)

    # Residuals on the ORIGINAL design (raw spread), as the structural ξ requires
    xi_v = d_v .- Xf_v * theta1
    xi   = delta .- X_full * theta1

    # --- GMM Q diagnostic: ḡ′Wḡ, ḡ = mean(ξ ⊙ Z), W = (Z′Z/N)⁻¹ ---
    W    = try inv(Z_v' * Z_v ./ N_v) catch; Matrix(1.0I, L, L) end
    g    = vec(mean(xi_v .* Z_v; dims=1))
    Q    = dot(g, W * g)
    q_df = max(L - 1, 1)

    # --- Cluster-robust sandwich + per-cluster influence functions (for the WCB) ---
    # Bread: (X̂′X̂)⁻¹.  Meat: Σ_c (X̂_c′ξ_c)(X̂_c′ξ_c)′.  IF_g = (X̂′X̂)⁻¹ (X̂_c′ξ_c).
    unique_cl = unique(cl_v)
    G         = length(unique_cl)
    meat      = zeros(K, K)
    IF_cl     = zeros(G, K)
    for (gi, c) in enumerate(unique_cl)
        c_mask  = cl_v .== c
        score_c = X_v[c_mask, :]' * xi_v[c_mask]   # (K,)
        meat   .+= score_c * score_c'
        IF_cl[gi, :] = XhX_inv * score_c
    end
    # Small-sample correction: G/(G-1) × N/(N-K)
    correction = (G / (G - 1)) * (N_v / (N_v - K))
    se_sand    = sqrt.(max.(diag(XhX_inv * meat * XhX_inv' .* correction), 0.0))

    # IK2016 effective clusters G* = G/(1+CV²), from the shared helper (blp_se_common.jl)
    G_star = max(1.0, effective_clusters(cl_v))

    # --- SE method: analytical sandwich or wild cluster bootstrap (default) ---
    # Both take the t(G*) few-cluster reference — the convention every other table reports.
    if se_method() == "sandwich"
        se   = se_sand
        pval = Float64[2 * ccdf(TDist(G_star), abs(theta1[k]) / max(se[k], 1e-15)) for k in 1:K]
    else  # "wcb" — score wild cluster bootstrap, matching the sleepiness estimation
        se, pval = wcb_se(theta1, IF_cl; dof=G_star)
    end

    return theta1, se, pval, xi, Q, q_df, G, G_star
end

# ==========================================================================
# 4. Per-routine driver
# ==========================================================================
"""
Estimate all four sub-models for a single routine.

Loads the routine's demand-prep parquet, builds δ from data shares, runs 2SLS for each
sub-model, writes per-(routine, sub-model) JLS and the δ warm-start checkpoints, and
returns a Dict keyed `"{label}_{submodel}"`.
"""
function run_strategy(estim)
    _, output_dir = get_paths()
    mkpath(output_dir)

    println("\n  ── Estimation $(estim.label) ──")

    df = load_spec_data(estim)
    println("    Loaded: $(nrow(df)) observations ($(estim.prefix)_spec_$(SPEC_ID).parquet)")

    # Mean utility by the Berry inversion, δ = ln s_j − ln s_0, in the engine's market
    # structure (see `berry_inversion`). The outside share is NOT 1 − Σ_j s_j of one column:
    # a market's choice set holds the B products of that market AND every D product of the
    # quarter, and a D share is the population-weighted average of local shares.
    delta, inv_diag = berry_inversion(df)
    @printf("    δ = ln s − ln s₀ | outside share s₀: min %.4f  median %.4f  max %.4f  (%d markets, %d quarters)\n",
            inv_diag.s0_min, inv_diag.s0_med, inv_diag.s0_max, inv_diag.n_pairs, inv_diag.n_times)

    # Cluster identifiers: CodConglomeradoPrudencial (matches sleep estimation)
    clusters = string.(df.CodConglomeradoPrudencial)

    # Deposit types for first-stage projection
    dep_types = Int.(coalesce.(df.deposit_type, 0))

    # ── Save logit δ checkpoint for BLP σ-stage warm-start ──
    # delta_dir() is the logit step folder, the same `logit_dir(out_dir)` the RC engine
    # searches, so these two files reach the warm start with no upload step in between.
    delta_chk_path = joinpath(delta_dir(),
        "logit_delta_E$(estim.id)_spec_$(SPEC_ID).jls")
    try
        serialize(delta_chk_path, Dict{String,Any}(
            "delta"    => delta,
            "estim_id" => estim.id,
            "spec_id"  => SPEC_ID,
            "N"        => nrow(df),
            "build"    => "logit",
        ))
        println("    [δ checkpoint] $(basename(delta_chk_path)) ($(nrow(df)) obs)")
    catch _e
        println("    [δ checkpoint] WARN: could not save — $(_e)")
    end
    # Version-agnostic binary (no Julia serialization-version dependency)
    bin_path = replace(delta_chk_path, ".jls" => ".bin")
    try
        open(bin_path, "w") do io
            write(io, Int64(length(delta)))
            write(io, delta)
        end
        println("    [δ .bin] $(basename(bin_path))")
    catch _e
        println("    [δ .bin] WARN: $(_e)")
    end

    routine_results = Dict{String, Any}()

    for sm in SUB_MODELS
        println("    Sub-model: $(sm.name)")

        # Build regressor and IV matrices
        spread, X, X_full, Z, iv_names = build_matrices(df, sm.xcols, sm.add_dtype)

        # First-stage projection of spreads
        spread_hat = project_spreads(spread, X, Z, dep_types)
        X_hat = hcat(spread_hat, X)

        # Parameter names — must track build_matrices' column order:
        # [spread, (constant), xcols…, (dummy_D_type)]
        pnames = vcat(["alpha"],
                      logit_intercept() ? ["constant"] : String[],
                      sm.xcols,
                      sm.add_dtype ? ["dummy_D_type"] : String[])

        # Linear step given the generated instrument. SEs by BLP_SE_METHOD (WCB by default);
        # `pval` are the matching p-values (t(G*) reference) used for the significance stars.
        theta1, se, pval, _xi, Q, q_df, n_cl, G_star =
            estimate_theta1_logit(delta, X_full, X_hat, Z, clusters)

        # t-statistics
        tstat = theta1 ./ max.(se, 1e-15)

        # Display
        println("      SE method: $(se_method())  |  Q-value: $(round(Q, sigdigits=6))")
        println("      ┌─────────────────────────────┬───────────┬──────────┬──────────┐")
        println("      │ Parameter                   │   Coeff   │    SE    │  t-stat  │")
        println("      ├─────────────────────────────┼───────────┼──────────┼──────────┤")
        for (i, pn) in enumerate(pnames)
            @printf("      │ %-27s │ %9.5f │ %8.5f │ %8.3f │\n",
                    pn, theta1[i], se[i], tstat[i])
        end
        println("      └─────────────────────────────┴───────────┴──────────┴──────────┘")

        # Store results
        key = "$(estim.label)_$(sm.name)"
        res = Dict{String,Any}(
            "estim"       => estim.label,
            "estim_id"    => estim.id,
            "spec_id"     => SPEC_ID,
            "build"       => "logit",
            "sub_model"   => sm.name,
            "param_names" => pnames,
            "theta1"      => theta1,
            "se"          => se,
            "tstat"       => tstat,
            "pval"        => pval,
            "se_method"   => se_method(),
            "Q_value"     => Q,
            "q_df"        => q_df,            # overid df = L − 1 (see estimate_theta1_logit)
            "n_obs"       => nrow(df),
            "n_iv"        => length(iv_names),
            "iv_names"    => iv_names,
            "n_clusters"  => n_cl,
            "G_star"      => G_star,
            "converged"   => true,
        )
        routine_results[key] = res

        # Save individual JLS
        jls_path = joinpath(logit_dir(),
            "logit_$(key)_spec_$(SPEC_ID).jls")
        try
            serialize(jls_path, res)
        catch; end
    end

    return routine_results
end

# ==========================================================================
# 5. Combined-summary serialization
# ==========================================================================
"""Convert a results Dict into JSON-friendly form (round Float64 vectors)."""
function _to_json_data(all_results::Dict{String,Any})
    json_data = Dict{String,Any}()
    for (k, v) in all_results
        jv = Dict{String,Any}()
        for (k2, v2) in v
            jv[k2] = v2 isa Vector{Float64} ? round.(v2, sigdigits=8) : v2
        end
        json_data[k] = jv
    end
    return json_data
end

"""Write the combined summary JSON from a full results Dict."""
function write_combined_summary(all_results::Dict{String,Any})
    json_path = combined_summary_path()
    open(json_path, "w") do f
        JSON3.pretty(f, _to_json_data(all_results))
    end
    println("\n  Combined summary saved: $(basename(json_path))")
    return json_path
end

"""
Merge one routine's keys into the existing combined summary (creating it if
absent). Lets single-routine entrypoints compose into the same summary file the table
generator reads.
"""
function merge_into_combined_summary(routine_results::Dict{String,Any})
    json_path = combined_summary_path()
    existing = Dict{String,Any}()
    if isfile(json_path)
        try
            existing = copy(JSON3.read(read(json_path, String), Dict{String,Any}))
        catch _e
            println("  [merge] WARN: could not read existing summary — $(_e). Recreating.")
            existing = Dict{String,Any}()
        end
    end
    for (k, v) in _to_json_data(routine_results)
        existing[k] = v
    end
    open(json_path, "w") do f
        JSON3.pretty(f, existing)
    end
    println("  [merge] Updated $(basename(json_path)) with $(length(routine_results)) sub-model key(s).")
    return json_path
end

# ==========================================================================
# 5b. LaTeX result tables (ported from make_blp_logit_table.py)
# ==========================================================================
# Writes est{id}_spec12_logit.tex (xltabular: 4 sub-model cols, parameter rows
# with SE underneath + IK2016 t(G*) significance stars, footer with obs/Q/G*). Output
# matches the former Python generator so V_Main.tex \input{} stays unchanged.

const VAR_MAP = Dict(
    "alpha"                => raw"Price coefficient ($\alpha$)",
    "constant"             => "Constant",
    "fgc_covered"          => "FGC Covered",
    "has_ip"               => "Group Contains IP",
    "log_total_assets_lag" => raw"$\ln(\text{Total Assets}_{t-1})$",
    "seg_S2" => "Segment S2", "seg_S3" => "Segment S3",
    "seg_S4" => "Segment S4", "seg_S5" => "Segment S5",
    "is_state_owned"       => "State-Owned",
    "dummy_D_type"         => "D Type",
)
const ROW_ORDER = ["alpha", "constant", "fgc_covered", "has_ip", "log_total_assets_lag",
                   "seg_S2", "seg_S3", "seg_S4", "seg_S5", "is_state_owned", "dummy_D_type"]
const TABLE_SUBMODELS = [("priceonly", "Price Only"), ("core", "Price + Core"),
                         ("full", "Price + Chars"), ("full_dtype", "+ D-Type"),
                         ("core_dtype", "Price + Core + D-Type")]
# Cross-estimator comparison table (est1-4_spec12_logit_comparison.tex): one column per
# demand routine, each showing its `full` (Price + Chars) sub-model. That sub-model's X is
# exactly the BLP X₁ (X_COLS, no dummy_D_type), so the published logit column is the literal
# θ₂ = 0 nest of the RC model in blp_engine_cpu/gpu — the paper's nesting claim depends on
# this key staying aligned with the engines' X_COLS. The D-type variant (full_dtype) remains
# in the per-routine tables. Column headers \ref{} the sleepiness-strategy enumerate items
# in V_Main §(sec:empirical:sleep) — same convention as est1-4_spec12_stage2_comparison.tex.
# ESTIMATION_ENUM_REF (routines.jl) carries a \ref for every id listed here; an id absent
# from it falls back to a plain E<id> header.
const COMPARISON_SUBMODEL = "full"
const COMPARISON_IDS  = LINK_ROUTINES
# Routines that get a per-routine est{id}_spec12_logit.tex. The routines are AUTO-DISCOVERED from the
# demand parquets, so this list is what keeps a stray parquet (an exploratory routine, an id outside
# the reported lineup) from silently writing a table into the Drafts folder on every run. An explicit
# `--est N` bypasses the filter, so any discovered routine stays reachable on demand.
const REPORTED_IDS    = ACTIVE_ROUTINES
const COMPARISON_ROWS = ["alpha", "fgc_covered", "has_ip", "log_total_assets_lag",
                         "is_state_owned"]   # seg_S2-S5 included in the spec, not reported
const COMPARISON_ROWS_SEG = ["alpha", "fgc_covered", "has_ip", "log_total_assets_lag",
                             "seg_S2", "seg_S3", "seg_S4", "seg_S5", "is_state_owned"]  # segments shown
# The Effective-F rows/note are suppressed until Table \ref{tab:weakiv_spec12} (the weak-IV
# battery table) exists; flip to `true` to print them again once it does.
const PRINT_EFF_F = false
const DRAFTS_DIR = drafts_dir()
const TROW = " \\\\"   # LaTeX row terminator ` \\` (a raw " \\" would collapse to one backslash)

map_var(name::AbstractString) = get(VAR_MAP, name, replace(name, "_" => raw"\_"))

_stars(p) = p < 0.01 ? raw"^{***}" : p < 0.05 ? raw"^{**}" : p < 0.10 ? raw"^{*}" : ""

"""Thousands-separated integer (mirrors Python's f'{n:,}')."""
function _commas(n::Integer)
    s = string(abs(n)); parts = String[]
    while length(s) > 3; pushfirst!(parts, s[end-2:end]); s = s[1:end-3]; end
    pushfirst!(parts, s)
    return (n < 0 ? "-" : "") * join(parts, ",")
end

"""Round to exactly 3 decimals as a plain numeral (no \$ wrapping) -- the DISPLAY RULE every
demand table follows: coefficients, SEs, Q, elasticities, G*. A value that rounds to zero
prints "0.000", never "-0.000" (printf preserves the sign of a negative value that rounds to
zero; this neutralizes it). Never rescales: a nonzero value that rounds to 0.000 at 3dp still
prints "0.000"."""
function fmt3(v::Real)
    r = round(v; digits=3)
    r == 0 && (r = abs(r))
    return @sprintf("%.3f", r)
end

"""(coef_cell, se_cell) with significance stars from the stored p-value `pval` (WCB Wald p)
when supplied, else the IK2016 t(G*) p; ('-','') if missing."""
function format_cell(coef, se, gstar, pval=nothing)
    (coef === nothing || se === nothing || isnan(coef) || isnan(se)) && return ("-", "")
    t = coef / max(se, 1e-15)
    p = (pval !== nothing && !isnan(pval)) ? pval :
        (gstar !== nothing && gstar > 1)   ? 2 * ccdf(TDist(gstar), abs(t)) :
                                             2 * ccdf(Normal(), abs(t))
    return (@sprintf("\$%s%s\$", fmt3(coef), _stars(p)), @sprintf("\$(%s)\$", fmt3(se)))
end

"""Q-value cell (the GMM criterion), '---' if missing. No stars and no degrees of freedom: `Q`
is not scaled by the moment variance and `W` is not the efficient weight, so it has no χ²
reference distribution (see `estimate_theta1_logit`)."""
function format_q_value(qv)
    (qv === nothing || isnan(qv)) && return "---"
    return @sprintf("\$%s\$", fmt3(qv))
end

# Footer label for Q; make_blp_rc_table.py (Q_ROW_LABEL) prints the same label under the RC tables.
const Q_ROW_LABEL = raw"GMM Criterion ($Q$)"

"""The `[demand]` table of config/table_notes.toml: the note sentences the logit and RC demand
tables share (make_blp_rc_table.py reads the same file), or `nothing` with a warning when the file
is absent — a cluster step without it then skips its .tex fragments instead of failing after the
estimates are saved."""
function demand_notes()
    p = joinpath(@__DIR__, "config", "table_notes.toml")
    if !isfile(p)
        println("    [table] WARN: $p not found — LaTeX tables not written " *
                "(add config/table_notes.toml next to config/routines.toml)")
        return nothing
    end
    return TOML.parsefile(p)["demand"]
end

"""
    demand_note(notes, which; se_method="wcb", subs=Dict()) -> String

The note body for table `which` (a key of `[demand.order]`): its sentences in the listed order,
with `se` resolved by `se_method`, every key in `skip` left out (e.g. `eff_f` when no F is
printed) and `@NAME@` tokens filled from `subs`.
"""
function demand_note(notes::AbstractDict, which::AbstractString;
                     se_method::AbstractString="wcb", skip=String[],
                     subs::AbstractDict=Dict{String,String}())
    parts = String[]
    for k in notes["order"][which]
        k in skip && continue
        s = k == "se" ? notes[se_method == "sandwich" ? "se_sandwich" : "se_wcb"] : notes[k]
        push!(parts, s)
    end
    txt = join(parts, " ")
    for (k, v) in subs
        txt = replace(txt, "@$(k)@" => v)
    end
    return txt
end

"""`\\footnotesize \\textit{Notes:} <body>` in the size the notes file sets."""
note_cell(notes::AbstractDict, body::AbstractString) =
    notes["note_size"] * raw" \textit{Notes:} " * body

"""Path of the demand weak-IV battery written by blp_weak_iv.py: `cluster_processed/weak_iv.json`
under the BLP results directory (`data/output/BLP_RESULTS` on the cluster)."""
function weak_iv_path()
    out_dir = get_paths()[2]
    res_dir = is_cluster_out(out_dir) ? joinpath(out_dir, "BLP_RESULTS") : out_dir
    return joinpath(res_dir, "cluster_processed", "weak_iv.json")
end

"""
    first_stage_eff_F(data, ids) -> Dict{Int, NTuple{4,Float64}}

Effective first-stage F of the spread for each routine, read from the weak-IV battery
(blp_weak_iv.py): the Montiel Olea–Pflueger cluster-robust F on the excluded instruments within
deposit type 4 and within type 5 — the two blocks `project_spreads` instruments, each with its
own first stage — after partialling out a constant and X_COLS. Those controls are the `full`
sub-model's X, so the values belong to the comparison table only.

A routine is kept only when the battery's full-sample block has the logit fit's observation
count, i.e. both were computed on the same demand parquet; otherwise it is skipped with a
warning and its cells print '---'. Returns id => (F type 4, F type 5, critical value type 4,
critical value type 5), the critical values being the stored Montiel Olea–Pflueger 10%
worst-case-bias thresholds (`mop_cv.bias10`); NaN for a missing entry.
"""
function first_stage_eff_F(data::AbstractDict, ids::Vector{Int})
    out = Dict{Int,NTuple{4,Float64}}()
    p = weak_iv_path()
    if !isfile(p)
        println("    [table] no weak-IV battery at $p — first-stage F rows omitted")
        return out
    end
    wk = try
        copy(JSON3.read(read(p, String), Dict{String,Any}))
    catch _e
        println("    [table] WARN: could not read $(basename(p)) — $_e"); return out
    end
    for id in ids
        blk = get(wk, string(id), nothing)
        blk === nothing && continue
        n_fit = get(get(data, "E$(id)_$(COMPARISON_SUBMODEL)", Dict{String,Any}()), "n_obs", nothing)
        n_bat = get(get(blk, "all", Dict{String,Any}()), "n_obs", nothing)
        if n_fit === nothing || n_bat === nothing || Int(n_fit) != Int(n_bat)
            println("    [table] WARN: E$id weak-IV battery N=$(n_bat) ≠ logit N=$(n_fit) — " *
                    "different demand-prep vintage; first-stage F omitted for E$id")
            continue
        end
        effF(t) = (v = get(get(blk, t, Dict{String,Any}()), "effective_F", nothing);
                   v === nothing ? NaN : Float64(v))
        cv10(t) = (v = get(get(get(blk, t, Dict{String,Any}()), "mop_cv", Dict{String,Any}()),
                           "bias10", nothing);
                   v === nothing ? NaN : Float64(v))
        out[id] = (effF("type4"), effF("type5"), cv10("type4"), cv10("type5"))
    end
    return out
end

"""Critical value `k` (3 = type 4, 4 = type 5) of `eff_f` across the printed routines, rounded
to an integer for the note: one number when they agree, `lo--hi` when they do not, '---' if none."""
function _cv_text(eff_f::AbstractDict, ids::Vector{Int}, k::Int)
    v = sort(unique([round(Int, eff_f[id][k]) for id in ids
                     if haskey(eff_f, id) && isfinite(eff_f[id][k])]))
    isempty(v) && return "---"
    return length(v) == 1 ? string(v[1]) : "$(v[1])--$(v[end])"
end

# ── plain-text formatting (comparison table matches est1-4_spec12_stage2_comparison.tex,
#    which prints `-0.1494***` / `(0.0677)` without math mode) ──────────────────────────
_stars_plain(p) = p < 0.01 ? "***" : p < 0.05 ? "**" : p < 0.10 ? "*" : ""

"""(coef_cell, se_cell) in the stage2-comparison plain style; stars from `pval` (WCB) when
supplied, else t(G*); ('-','-') if missing."""
function format_cell_plain(coef, se, gstar, pval=nothing)
    (coef === nothing || se === nothing || isnan(coef) || isnan(se)) && return ("-", "-")
    t = coef / max(se, 1e-15)
    p = (pval !== nothing && !isnan(pval)) ? pval :
        (gstar !== nothing && gstar > 1)   ? 2 * ccdf(TDist(gstar), abs(t)) :
                                             2 * ccdf(Normal(), abs(t))
    return (@sprintf("%s%s", fmt3(coef), _stars_plain(p)), @sprintf("(%s)", fmt3(se)))
end

"""Sample mean of ρ(1−s) for one routine — multiplied by that routine's α̂ it gives the
mean own-price ELASTICITY row of the comparison table: α̂·ρ_jkmt·(1−s_jkmt) averaged over
the estimation sample. Because the spread ρ enters in levels, α̂·ρ·(1−s) = ∂ln s/∂ln ρ is
the unit-free own-price elasticity (a semi-elasticity would be α̂·(1−s), per pp). ρ = spread
in pp (spread_ann/100); s = the share used to build δ (share_D for D-type products,
conditional share_B_cond for B-type). NaN if unavailable."""
function _mean_rho_one_minus_s(estim)
    df   = load_spec_data(estim)
    ρ    = Float64.(coalesce.(df.spread_ann, NaN)) ./ 100.0
    is_B = Bool.(coalesce.(df.is_B, false))
    sD   = Float64.(coalesce.(df.share_D, NaN))
    sB   = Float64.(coalesce.(df.share_B_cond, NaN))
    s    = ifelse.(is_B, sB, sD)
    m    = isfinite.(ρ) .& isfinite.(s) .& (s .>= 0.0) .& (s .< 1.0)
    return any(m) ? mean(ρ[m] .* (1.0 .- s[m])) : NaN
end

"""Build est1-4_spec12_logit_comparison.tex: columns = routines (each its COMPARISON_SUBMODEL
sub-model — `full`, the BLP-X₁-matching spec), rows = COMPARISON_ROWS, stats block =
elasticity / N / Q / effective first-stage F (types 4 and 5, from `eff_f`; the two rows and
their note sentence are omitted when `eff_f` is empty) / G*. The note is assembled from
config/table_notes.toml (`notes`, see `demand_note`), the wording Table 6 shares.
Layout and label conventions mirror est1-4_spec12_stage2_comparison.tex."""
function build_logit_comparison_tex(data::AbstractDict, ids::Vector{Int},
                                    elas::AbstractDict, notes::AbstractDict;
                                    rows::Vector{String}=COMPARISON_ROWS,
                                    with_seg::Bool=false,
                                    eff_f::AbstractDict=Dict{Int,NTuple{4,Float64}}())::String
    n    = length(ids)
    hdr  = "Variable & " * join([get(ESTIMATION_ENUM_REF, id, "E$id") for id in ids], " & ") * TROW
    # The segment dummies are nuisance controls already printed IN FULL, per routine, by the appendix
    # tables est{id}_spec12_logit.tex — so we omit them here rather than carrying a duplicate `_seg`
    # twin of this table (four extra rows, no information).
    skip = String[]
    with_seg && push!(skip, "segments")
    isempty(eff_f) && push!(skip, "eff_f")
    sem  = String(get(get(data, "E$(ids[1])_$(COMPARISON_SUBMODEL)", Dict{String,Any}()),
                      "se_method", "wcb"))
    body = demand_note(notes, "logit_comparison"; se_method=sem, skip=skip,
                       subs=Dict("CV4" => _cv_text(eff_f, ids, 3),
                                 "CV5" => _cv_text(eff_f, ids, 4)))
    note = raw"\multicolumn{" * string(n + 1) *
        raw"}{p{\dimexpr\textwidth-2\tabcolsep\relax}}{" * note_cell(notes, body) *
        "}"   # no trailing TROW (matches stage2 template)
    lines = String[
        raw"\setstretch{1.0}",
        raw"\begin{xltabular}{\textwidth}{>{\raggedright\arraybackslash}p{0.26\textwidth} *{" *
            string(n) * raw"}{>{\centering\arraybackslash}X}}",
        "\\caption{Demand Logit Estimation}\\label{tab:demand_logit_spec12_comparison" *
            (with_seg ? "_seg" : "") * "}" * TROW,
        raw"\toprule", hdr, raw"\midrule", raw"\endfirsthead",
        raw"\multicolumn{" * string(n + 1) *
            raw"}{c}{{\bfseries \tablename\ \thetable{} (continued from previous page)}}" * TROW,
        raw"\toprule", hdr, raw"\midrule", raw"\endhead",
        raw"\midrule",
        raw"\multicolumn{" * string(n + 1) * raw"}{r}{{Continued on next page}}" * TROW,
        raw"\endfoot",
        raw"\bottomrule", note, raw"\endlastfoot",
    ]
    for (i, p) in enumerate(rows)
        row_c = String[]; row_s = String[]
        for id in ids
            entry  = get(data, "E$(id)_$(COMPARISON_SUBMODEL)", Dict{String,Any}())
            pnames = String.(get(entry, "param_names", String[]))
            j = findfirst(==(p), pnames)
            if j !== nothing
                gs = get(entry, "G_star", nothing)
                pv = get(entry, "pval", nothing)
                c, s = format_cell_plain(Float64(entry["theta1"][j]), Float64(entry["se"][j]),
                                         gs === nothing ? nothing : Float64(gs),
                                         (pv !== nothing && j <= length(pv)) ? Float64(pv[j]) : nothing)
                push!(row_c, c); push!(row_s, s)
            else
                push!(row_c, "-"); push!(row_s, "-")
            end
        end
        push!(lines, raw"\multirow[t]{2}{0.26\textwidth}{\raggedright " * map_var(p) *
                     "} & " * join(row_c, " & ") * " \\\\*")
        push!(lines, " & " * join(row_s, " & ") * TROW)
        i < length(rows) && push!(lines, raw"\addlinespace")
    end
    push!(lines, raw"\midrule")
    elas_l = String[]; obs_l = String[]; q_l = String[]; gstar_l = String[]
    f4_l = String[]; f5_l = String[]
    _fcell(v) = isfinite(v) ? fmt3(v) : "---"
    for id in ids
        entry = get(data, "E$(id)_$(COMPARISON_SUBMODEL)", Dict{String,Any}())
        ev = get(elas, id, NaN)
        push!(elas_l, isfinite(ev) ? fmt3(ev) : "---")
        obs = get(entry, "n_obs", nothing)
        push!(obs_l, (obs === nothing || obs == 0) ? "---" : _commas(Int(obs)))
        qv = get(entry, "Q_value", nothing)
        push!(q_l, qv === nothing ? "---" : fmt3(Float64(qv)))
        f4, f5 = get(eff_f, id, (NaN, NaN, NaN, NaN))[1:2]
        push!(f4_l, _fcell(f4)); push!(f5_l, _fcell(f5))
        gs = get(entry, "G_star", nothing)
        push!(gstar_l, gs === nothing ? "---" : fmt3(Float64(gs)))
    end
    append!(lines, [
        "Mean own-price elasticity & " * join(elas_l, " & ") * TROW,
        "Observations & " * join(obs_l, " & ") * TROW,
        Q_ROW_LABEL * " & " * join(q_l, " & ") * TROW,
    ])
    isempty(eff_f) || append!(lines, [
        "Effective \$F\$, Type 4 & " * join(f4_l, " & ") * TROW,
        "Effective \$F\$, Type 5 & " * join(f5_l, " & ") * TROW,
    ])
    append!(lines, [
        "Effective Clusters (\$G^*\$) & " * join(gstar_l, " & ") * TROW,
        raw"\end{xltabular}",
        raw"\doublespacing",
    ])
    return join(lines, "\n")
end

"""Write est1-4_spec12_logit_comparison.tex (COMPARISON_IDS × COMPARISON_SUBMODEL) to
Rout + Drafts. Computes the mean own-price elasticity per routine from its demand parquet
(skipped with a '---' cell if the parquet is unavailable) and reads the effective first-stage F
from the weak-IV battery (`first_stage_eff_F`)."""
function write_logit_comparison_table(data::AbstractDict)
    ids = [id for id in COMPARISON_IDS if haskey(data, "E$(id)_$(COMPARISON_SUBMODEL)")]
    if length(ids) < 2
        println("    [table] comparison skipped — need ≥2 of E$(COMPARISON_IDS) $(COMPARISON_SUBMODEL) entries")
        return
    end
    notes = demand_notes()
    notes === nothing && return
    elas = Dict{Int,Float64}()
    for id in ids
        entry  = data["E$(id)_$(COMPARISON_SUBMODEL)"]
        pnames = String.(get(entry, "param_names", String[]))
        j = findfirst(==("alpha"), pnames)
        estim = findfirst(s -> s.id == id, ESTIM_STRATEGIES)
        if j === nothing || estim === nothing
            elas[id] = NaN; continue
        end
        elas[id] = try
            Float64(entry["theta1"][j]) * _mean_rho_one_minus_s(ESTIM_STRATEGIES[estim])
        catch _e
            println("    [table] WARN: E$id elasticity failed — $_e"); NaN
        end
    end
    eff_f = PRINT_EFF_F ? first_stage_eff_F(data, ids) : Dict{Int,NTuple{4,Float64}}()
    rout_dir = tex_out_dir()
    dests = isdir(DRAFTS_DIR) ? [rout_dir, DRAFTS_DIR] : [rout_dir]
    # ONLY the segment-suppressed version is emitted. The `_seg` twin was retired: the segment dummies
    # it added are already printed per routine by est{3,4}_spec12_logit.tex (both live in the
    # appendix), so it duplicated four rows and no information while lengthening the appendix. The
    # footnote now cross-references those tables. Re-add the COMPARISON_ROWS_SEG/`true` tuple here if a
    # referee ever wants the with-segments layout back.
    for (rws, wseg, fn) in ((COMPARISON_ROWS, false, "est1-4_spec12_logit_comparison.tex"),)
        tex = build_logit_comparison_tex(data, ids, elas, notes; rows=rws, with_seg=wseg, eff_f=eff_f)
        for d in dests
            path = joinpath(d, fn)
            try
                open(path, "w") do f; write(f, tex); end
                println("    [table] $fn → $(basename(d))/")
            catch _e
                println("    [table] WARN: could not write $path — $_e")
            end
        end
    end
end

"""Build the xltabular LaTeX for one routine from the combined-summary `data`; the note comes
from config/table_notes.toml (`notes`, order `logit_routine`)."""
function build_logit_table_tex(est_id::Int, data::AbstractDict, notes::AbstractDict)::String
    ncols   = length(TABLE_SUBMODELS)
    sem     = String(get(get(data, "E$(est_id)_$(COMPARISON_SUBMODEL)", Dict{String,Any}()),
                         "se_method", "wcb"))
    col_fmt = raw">{\raggedright\arraybackslash}p{0.24\textwidth} *{" * string(ncols) *
              raw"}{>{\centering\arraybackslash}X}"
    hdr = "    Parameter & " * join([sm[2] for sm in TABLE_SUBMODELS], " & ") * TROW
    lines = String[
        raw"\begin{spacing}{1.0}",
        "\\begin{xltabular}{\\textwidth}{$col_fmt}",
        "    \\caption{Demand Logit Estimation --- Estimation " *
            "$(get(ESTIMATION_ENUM_REF, est_id, "E$est_id"))}",
        "    \\label{tab:demand_logit_est$(est_id)_spec12} \\\\",
        raw"    \toprule", hdr, raw"    \midrule", raw"    \endfirsthead", "",
        raw"    \multicolumn{" * string(ncols+1) *
            raw"}{c}{\bfseries Table \thetable\ continued from previous page}" * TROW,
        raw"    \toprule", hdr, raw"    \midrule", raw"    \endhead", "",
        raw"    \midrule",
        raw"    \multicolumn{" * string(ncols+1) * raw"}{r}{\textit{Continued on next page}}" * TROW,
        raw"    \endfoot", "",
        raw"    \bottomrule",
        raw"    \multicolumn{" * string(ncols+1) *
            raw"}{p{\dimexpr\textwidth-2\tabcolsep\relax}}{" *
            note_cell(notes, demand_note(notes, "logit_routine"; se_method=sem)) * "}" * TROW,
        raw"    \endlastfoot", "",
    ]
    # Only rows some sub-model actually estimated. `constant` is in ROW_ORDER for the
    # BLP_LOGIT_INTERCEPT=1 diagnostic run; with the intercept off (the reported design) it is
    # in no `param_names`, and this filter keeps it out of the table instead of printing a row
    # of dashes.
    _row_present(p) = any(sm -> p in String.(get(get(data, "E$(est_id)_$(sm[1])",
                                                     Dict{String,Any}()), "param_names", String[])),
                          TABLE_SUBMODELS)
    rows_used = [p for p in ROW_ORDER if _row_present(p)]
    for (i, p) in enumerate(rows_used)
        row_c = String[map_var(p)]; row_s = String[""]
        for (sm_key, _) in TABLE_SUBMODELS
            entry  = get(data, "E$(est_id)_$(sm_key)", Dict{String,Any}())
            pnames = String.(get(entry, "param_names", String[]))
            j = findfirst(==(p), pnames)
            if j !== nothing
                gs = get(entry, "G_star", nothing)
                pv = get(entry, "pval", nothing)
                c, s = format_cell(Float64(entry["theta1"][j]), Float64(entry["se"][j]),
                                   gs === nothing ? nothing : Float64(gs),
                                   (pv !== nothing && j <= length(pv)) ? Float64(pv[j]) : nothing)
                push!(row_c, c); push!(row_s, s)
            else
                push!(row_c, "-"); push!(row_s, "")
            end
        end
        push!(lines, "    " * join(row_c, " & ") * TROW)
        push!(lines, "    " * join(row_s, " & ") * TROW)
        i < length(rows_used) && push!(lines, raw"    \addlinespace")
    end
    push!(lines, raw"    \midrule")
    obs_l = String[]; q_l = String[]; gstar_l = String[]
    for (sm_key, _) in TABLE_SUBMODELS
        entry = get(data, "E$(est_id)_$(sm_key)", Dict{String,Any}())
        obs   = get(entry, "n_obs", nothing)
        push!(obs_l, (obs === nothing || obs == 0) ? "---" : _commas(Int(obs)))
        qv  = get(entry, "Q_value", nothing)
        push!(q_l, qv === nothing ? "---" : format_q_value(Float64(qv)))
        gs = get(entry, "G_star", nothing)
        push!(gstar_l, gs === nothing ? "---" : fmt3(Float64(gs)))
    end
    append!(lines, [
        "    Observations & " * join(obs_l, " & ") * TROW,
        "    " * Q_ROW_LABEL * " & " * join(q_l, " & ") * TROW,
        "    Effective Clusters (\$G^*\$) & " * join(gstar_l, " & ") * TROW,
        raw"\end{xltabular}", raw"\end{spacing}",
    ])
    return join(lines, "\n")
end

"""Read the combined summary JSON back into a Dict (for table generation)."""
function read_combined_summary()
    p = combined_summary_path()
    isfile(p) || return Dict{String,Any}()
    try
        return copy(JSON3.read(read(p, String), Dict{String,Any}))
    catch _e
        println("  [table] WARN: could not read summary — $_e"); return Dict{String,Any}()
    end
end

"""Write est{id}_spec12_logit.tex for each id to `tex_out_dir()` and Drafts."""
function write_logit_tables(ids::Vector{Int}, data::AbstractDict)
    notes = demand_notes()
    notes === nothing && return
    rout_dir = tex_out_dir()                             # ESTIMATION_OUTPUT/Rout, or logit/ on the cluster
    dests = isdir(DRAFTS_DIR) ? [rout_dir, DRAFTS_DIR] : [rout_dir]
    for id in ids
        tex = build_logit_table_tex(id, data, notes)
        for d in dests
            path = joinpath(d, "est$(id)_spec12_logit.tex")
            try
                open(path, "w") do f; write(f, tex); end
                println("    [table] est$(id)_spec12_logit.tex → $(basename(d))/")
            catch _e
                println("    [table] WARN: could not write $path — $_e")
            end
        end
    end
end

"""Discover routine ids present in the combined summary (keys `E<id>_<submodel>`)."""
function _summary_ids(data::AbstractDict)
    ids = Set{Int}()
    for k in keys(data)
        m = match(r"^E(\d+)_", String(k))
        m === nothing || push!(ids, parse(Int, m.captures[1]))
    end
    return sort!(collect(ids))
end

# ==========================================================================
# 6. Orchestrator entrypoint
# ==========================================================================
"""Parse an optional single-routine selector from ARGS: `--est N` or a bare integer.
Returns the id (Int) or nothing (run all discovered routines)."""
function _parse_est_arg()
    for (i, a) in enumerate(ARGS)
        if a == "--est" && i < length(ARGS)
            return tryparse(Int, ARGS[i + 1])
        elseif (v = tryparse(Int, a)) !== nothing
            return v
        end
    end
    return nothing
end

function main()
    _, output_dir = get_paths()
    mkpath(output_dir)

    sel = _parse_est_arg()

    # --tables-only: rebuild every LaTeX table from the existing combined summary,
    # skipping estimation entirely (the elasticity row still reads the parquets).
    if "--tables-only" in ARGS
        summary = read_combined_summary()
        isempty(summary) && error("--tables-only: no combined summary at $(combined_summary_path()).")
        table_ids = sel === nothing ? filter(in(REPORTED_IDS), _summary_ids(summary)) : [sel]
        println("  [tables-only] Regenerating LaTeX tables for $(join("E" .* string.(table_ids), ", "))…")
        write_logit_tables(table_ids, summary)
        write_logit_comparison_table(summary)
        println("  [DONE] Tables regenerated (no estimation run).")
        return
    end

    routines = sel === nothing ? ESTIM_STRATEGIES : filter(s -> s.id == sel, ESTIM_STRATEGIES)
    if isempty(routines)
        avail = isempty(ESTIM_STRATEGIES) ? "(none)" :
                join([s.label for s in ESTIM_STRATEGIES], ", ")
        error(sel === nothing ?
            "No demand-prep parquets found for spec $SPEC_ID. Searched:\n" *
                describe_search_dirs(demand_dirs()...) :
            "No demand-prep parquet for E$sel (spec $SPEC_ID). Available: $avail. Searched:\n" *
                describe_search_dirs(demand_dirs()...))
    end

    println("=" ^ 70)
    println("  BLP Logit (Non-RC) — Spec $SPEC_ID")
    println("  $(length(routines)) routine(s): $(join([s.label for s in routines], ", ")) " *
            "× $(length(SUB_MODELS)) sub-models")
    println("=" ^ 70)

    if sel === nothing
        # Full build: run every discovered routine, write the combined summary.
        all_results = Dict{String, Any}()
        for estim in ESTIM_STRATEGIES
            try
                merge!(all_results, run_strategy(estim))
            catch e
                println("  [!] Routine $(estim.label) failed: $e. Skipping.")
            end
        end
        write_combined_summary(all_results)
    else
        # Single routine: run it and merge into the existing combined summary.
        merge_into_combined_summary(run_strategy(routines[1]))
    end

    # LaTeX result tables (replaces the former make_blp_logit_table.py step).
    summary   = read_combined_summary()
    table_ids = sel === nothing ? filter(in(REPORTED_IDS), _summary_ids(summary)) : [sel]
    println("\n  Generating LaTeX tables for $(join("E" .* string.(table_ids), ", "))…")
    write_logit_tables(table_ids, summary)
    write_logit_comparison_table(summary)

    println("  [DONE] Logit estimation + tables complete.")
end

# Only auto-run when executed directly.
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
