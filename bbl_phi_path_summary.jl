"""
bbl_phi_path_summary.jl
=======================
The sleepy share φ along the forward paths of the multi-start BBL simulation (bbl_fwd_sim.jl
--multi-start --n-paths 1 --phi-path evolving), summarised for the paper: for each of the launch
quarters and each horizon h ∈ {0, 1, 4, 20, 40, 100, 250}, the national φ under three weightings,
and their average over the launches. It evaluates the objects the simulation evaluates
(cf_phi_path.jl: phi_path_inputs, phi_at!) on the inputs the simulation reads:

  sleep_link_E{k}_spec_{s}.json   the routine's exported sleepiness link (bbl_sleep_link.py)
  bbl_transitions.json            the market-state AR(1) ρ (market_states)
  demand_{k}_*_spec_{s}.parquet   the demand-prep rows: launch states, phi_mt, phi_t, pop_total,
                                  deposit_balance
  forward_rf_vintages.csv         each launch quarter's Focus curve (h ≥ 1) and its own rate (h = 0)
and, for `pop_table4` only, the sleep side's est{k}/national_phi_t.csv (not a simulation input).

The φ path depends on those states alone, never on a spread or on the solve, so the summary is
final once the inputs are. Their sha256s are recorded; the simulation records the same hashes of
the link and the transitions in every psi file (sleep_link_sha256, state_transitions_sha256).

THREE NATIONAL φ_t OBJECTS, and which one each output starts from at h = 0:
  (ii) the demand prep's stored φ_t (the parquet column `phi_t`), which the demand inversion reads
       for the D rows (eq. 14-D) and which the simulation's D rows start from EXACTLY;
  (i)  the sleep side's national φ̂_t (DEMAND_PREP/est{k}/national_phi_t.csv, column
       phi_t_<spec key>), whose average over its quarters is Table 4's "Mean φ̂";
  (iii) the population-weighted mean over the demand parquet's MCA cells of their φ at the launch
       states (cf_phi_path.jl `agg0`), the base the path's CHANGE is measured from.
  All three are population-weighted means over (mca_code, quarter) cells that include the D rows'
  'NATIONAL' cell (its pop_total is the national population, so it carries half the weight). They
  differ in that cell's φ: the sleep panel fills the D rows' demographic states with a mean
  weighted by population over BANK ROWS (pop × number of rows), the demand prep with a mean over
  MCAs, each once, weighted by population; the MCA cells themselves agree to ~1e-4. So (i) ≠ (ii)
  by ~0.16 pp on average (reported in `reconciliation`), while (iii) ≈ (ii) to ~2e-3 (the prep
  aggregated before its Dep_Act > 1e-6 row filter). Neither pipeline is changed here.

Weightings, per launch quarter q and horizon h:
  pop        HEADLINE. The national φ_t path a D row of quarter q follows: (ii) moved by
             Φ_q(h) − Φ_q(0), Φ the population-weighted mean over the quarter's MCA cells (each
             counted once, weights held at the launch quarter). At h = 0 it EQUALS (ii).
  pop_table4 the same path change on Table 4's base: (i) + [Φ_q(h) − Φ_q(0)]. At h = 0 it EQUALS
             (i); null for a launch quarter the sleep side does not cover (2016Q1), and its
             `pooled` averages the quarters it covers.
  row        the mean over the quarter's demand rows of the φ each row is simulated with (a B row
             its own link path, a D row the national path), clamped to [0, 0.999] as simulated.
  dep        the same, weighted by each row's launch deposit balance (deposit_balance, negatives 0).
`pooled` is the unweighted mean of the by-start values over the launch quarters.

Usage (from the repo folder, one line each):
  julia --project=. bbl_phi_path_summary.jl
  julia --project=. bbl_phi_path_summary.jl --estim 3 --spec 12 --eo "<ESTIMATION_OUTPUT folder>"
Without --estim it runs E3 and E4. --eo defaults to the local ESTIMATION_OUTPUT (of_root.jl). It
writes <eo>/COST_FWD/phi_path_summary_E{k}_spec_{s}.json and prints the pooled table.
"""

using DataFrames, Statistics, Printf
import Parquet2, JSON3

log_status(msg::AbstractString) = println(msg)
include(joinpath(@__DIR__, "of_root.jl"))
include(joinpath(@__DIR__, "cf_phi_path.jl"))

const SUMMARY_HORIZONS = [0, 1, 4, 20, 40, 100, 250]
const SUMMARY_SCHEMA = "bbl_phi_path_summary/1"

"""
    _vintage_curves(path, T) -> (starts, anchor::Dict, curves::Dict)

forward_rf_vintages.csv as the multi-start simulation reads it (bbl_fwd_sim.jl
_load_rf_vintages): per start_q, h = 0 (the realised launch-quarter rate) and h = 1…T. A start
without h = 0, or without every horizon 1…T, is an error here: the simulation on the cluster
refuses a curve shorter than T as well.
"""
function _vintage_curves(path::AbstractString, T::Int)
    lines = filter(l -> !isempty(strip(l)), readlines(path))
    hdr = [String(strip(h)) for h in split(lines[1], ',')]
    i_s, i_h, i_r = (findfirst(==(c), hdr) for c in ("start_q", "h", "rf_qoq"))
    (i_s === nothing || i_h === nothing || i_r === nothing) &&
        error("$(basename(path)): need columns start_q, h, rf_qoq")
    raw = Dict{String,Dict{Int,Float64}}()
    for l in lines[2:end]
        f = split(l, ',')
        h = tryparse(Int, strip(f[i_h])); v = tryparse(Float64, strip(f[i_r]))
        (h === nothing || v === nothing || !isfinite(v)) && continue
        get!(raw, String(strip(f[i_s])), Dict{Int,Float64}())[h] = v
    end
    starts = sort(collect(keys(raw)))
    anchor = Dict{String,Float64}(); curves = Dict{String,Vector{Float64}}()
    for q in starts
        d = raw[q]
        haskey(d, 0) || error("$(basename(path)): start $q has no h = 0 row")
        all(h -> haskey(d, h), 1:T) || error("$(basename(path)): start $q does not reach h = $T")
        anchor[q] = d[0]; curves[q] = [d[h] for h in 1:T]
    end
    return starts, anchor, curves
end

"""
    _sleep_national_phi(path, spec_key) -> Dict(quarter => φ_t)

The sleep side's national φ̂_t of one specification from est{k}/national_phi_t.csv: the column
phi_t_<spec key with spaces as '_' and '.' removed> (sleep_est_single._calculate_phis' safe key),
keyed by year_quarter.
"""
function _sleep_national_phi(path::AbstractString, spec_key::AbstractString)
    lines = filter(l -> !isempty(strip(l)), readlines(path))
    hdr = [String(strip(h)) for h in split(lines[1], ',')]
    col = "phi_t_" * replace(replace(spec_key, " " => "_"), "." => "")
    iq = findfirst(==("year_quarter"), hdr); ic = findfirst(==(col), hdr)
    (iq === nothing || ic === nothing) && error("$(basename(path)): need year_quarter and $col")
    d = Dict{String,Float64}()
    for l in lines[2:end]
        f = split(l, ','); v = tryparse(Float64, strip(f[ic]))
        v === nothing || (d[String(strip(f[iq]))] = v)
    end
    return d
end

function _demand_parquet(dprep::AbstractString, estim::Int, spec::Int)
    hits = filter(f -> occursin(Regex("^demand_$(estim)_.*spec_$(spec)\\.parquet\$"), f), readdir(dprep))
    length(hits) == 1 || error("expected one demand_$(estim)_*_spec_$(spec).parquet in $dprep, found $(hits)")
    return joinpath(dprep, only(hits))
end

wmean(x, w) = sum(w .* x) / sum(w)

"""
    phi_path_summary(estim, spec, eo) -> Dict (the JSON written)

Build the summary of one routine and write <eo>/COST_FWD/phi_path_summary_E{estim}_spec_{spec}.json.
"""
function phi_path_summary(estim::Int, spec::Int, eo::AbstractString)
    cost = joinpath(eo, "COST_FWD"); dprep = joinpath(eo, "DEMAND_PREP")
    lpath = joinpath(cost, "sleep_link_E$(estim)_spec_$(spec).json")
    tpath = joinpath(cost, "bbl_transitions.json")
    vpath = joinpath(cost, "forward_rf_vintages.csv")
    ppath = _demand_parquet(dprep, estim, spec)
    npath = joinpath(dprep, "est$(estim)", "national_phi_t.csv")
    T = maximum(SUMMARY_HORIZONS)
    L = load_sleep_link(lpath; estim=estim, spec=spec)
    spec_key = String(JSON3.read(read(lpath, String))[:spec_key])
    sleep_nat = _sleep_national_phi(npath, spec_key)
    df = DataFrame(Parquet2.Dataset(ppath); copycols=true)
    keep = unique(vcat(L.cols[L.cols .!= "constant"],
                       ["phi_mt", "phi_t", "pop_total", "is_B", "mca_code", "time_id",
                        "risk_free_qoq_lag_level", "deposit_balance"]))
    df = df[:, [c for c in keep if c in names(df)]]
    inp = phi_path_inputs(df, L; transitions_path=tpath)
    starts, anchor, curves = _vintage_curves(vpath, T)
    tid = string.(df.time_id)
    miss = setdiff(unique(tid), starts)
    isempty(miss) || error("launch quarters without a vintage curve: $(join(sort(miss), ", "))")
    sidx = Dict(q => i for (i, q) in enumerate(starts))
    rc = [sidx[q] for q in tid]
    h0 = [anchor[q] for q in tid]
    fwd = permutedims(reduce(hcat, [curves[q] for q in starts]))       # S × T
    dep = max.(Float64.(coalesce.(df.deposit_balance, 0.0)), 0.0)
    quarters = inp.nat.quarters                                          # national-path order
    qs = sort(quarters)
    rows_of = Dict(q => findall(==(q), tid) for q in qs)
    nq = length(quarters)
    by = Dict(q => Dict{String,Any}(w => Float64[] for w in ("pop", "row", "dep")) for q in qs)
    v = zeros(nrow(df)); nat = zeros(nq)
    for h in SUMMARY_HORIZONS
        phi_at!(v, inp, h; rf_h0=h0, fwd=fwd, row_curve=rc, national=nat)
        natd = Dict(quarters[k] => nat[k] for k in 1:nq)
        for q in qs
            r = rows_of[q]
            push!(by[q]["pop"], natd[q])
            push!(by[q]["row"], mean(v[r]))
            push!(by[q]["dep"], wmean(v[r], dep[r]))
        end
    end
    # t = 0 identities the simulation relies on (a refusal here means the inputs moved)
    stored = Dict(quarters[k] => inp.nat.phi_t[k] for k in 1:nq)            # (ii)
    all(by[q]["pop"][1] == clamp(stored[q], 0.0, PHI_SIM_MAX) for q in qs) ||
        error("the national path at h = 0 is not the stored phi_t")
    # pop_table4: the same change on the sleep side's level (i); the path change is pop − pop(h=0)
    t4q = [q for q in qs if haskey(sleep_nat, q)]
    for q in qs
        by[q]["pop_table4"] = haskey(sleep_nat, q) ?
            [sleep_nat[q] + (x - by[q]["pop"][1]) for x in by[q]["pop"]] : nothing
    end
    pooled = Dict{String,Any}(w => [mean(by[q][w][j] for q in qs) for j in eachindex(SUMMARY_HORIZONS)]
                              for w in ("pop", "row", "dep"))
    pooled["pop_table4"] = [mean(by[q]["pop_table4"][j] for q in t4q) for j in eachindex(SUMMARY_HORIZONS)]
    agg0 = Dict(quarters[k] => inp.nat.agg0[k] for k in 1:nq)               # (iii)
    d_i_ii = [stored[q] - sleep_nat[q] for q in t4q]
    d_iii_ii = [agg0[q] - stored[q] for q in qs]
    reconciliation = Dict{String,Any}(
        "note" => "(i) sleep national_phi_t.csv (Table 4 base); (ii) the demand prep's stored phi_t " *
                  "(the demand inversion's D-row phi; the simulation's D rows start from it exactly); " *
                  "(iii) the demand parquet's MCA-cell aggregate at the launch states. pop(h=0) == (ii) " *
                  "and pop_table4(h=0) == (i) exactly.",
        "table4_mean_phi_hat_i" => mean(values(sleep_nat)), "n_quarters_i" => length(sleep_nat),
        "launch_quarters_without_i" => [q for q in qs if !haskey(sleep_nat, q)],
        "mean_ii_minus_i" => mean(d_i_ii), "max_abs_ii_minus_i" => maximum(abs, d_i_ii),
        "mean_iii_minus_ii" => mean(d_iii_ii), "max_abs_iii_minus_ii" => maximum(abs, d_iii_ii),
        "by_quarter" => Dict(q => Dict("i_sleep_table4" => get(sleep_nat, q, nothing),
                                       "ii_demand_stored" => stored[q], "iii_cells_t0" => agg0[q]) for q in qs))
    out = Dict{String,Any}(
        "schema" => SUMMARY_SCHEMA, "estim" => estim, "spec" => spec,
        "definition" => "Sleepy share phi along the multi-start forward paths (bbl_fwd_sim.jl " *
            "--multi-start --n-paths 1 --phi-path evolving), per launch quarter and horizon h " *
            "(quarters after launch; h = 0 is the launch quarter). pop (headline): the national " *
            "phi_t path the simulation's D rows follow = the demand prep's STORED phi_t of the launch " *
            "quarter (the phi_t the demand inversion used for D rows, eq. 14-D) moved by the change in " *
            "the population-weighted mean over the demand parquet's MCA cells (each counted once, the " *
            "D rows' NATIONAL cell included as in the sleep side's aggregation, weights held at " *
            "launch); at h = 0 it EQUALS the stored phi_t. pop_table4: the same change added to the " *
            "sleep side's national phi_t (national_phi_t.csv, whose mean over quarters is Table 4's " *
            "Mean phi-hat); at h = 0 it EQUALS that series; null where the sleep side has no such " *
            "quarter (2016Q1). The two levels differ (see reconciliation) because the sleep panel " *
            "fills the D rows' demographic states with a pop x bank-row weighted national mean and " *
            "the demand prep with a pop-weighted mean over MCAs. row: mean over the launch quarter's " *
            "demand rows of the phi each row is simulated with. dep: the same weighted by launch " *
            "deposit balance. pooled: unweighted mean over launch quarters (pop_table4: over the " *
            "quarters it covers). States: Pix (and E4's time block) held at launch; the lagged Selic " *
            "follows the launch vintage (h = 0 its own rate, then its Focus curve); market " *
            "demographics mean-revert within their MCA at the bbl_transitions AR(1). Values are " *
            "fractions (0.98 = 98 pp).",
        "horizons" => SUMMARY_HORIZONS,
        "n_starts" => length(qs),
        "by_start" => Dict(q => Dict("pop" => by[q]["pop"], "pop_table4" => by[q]["pop_table4"],
                                     "row" => by[q]["row"], "dep" => by[q]["dep"],
                                     "n_rows" => length(rows_of[q]),
                                     "n_D_rows" => count(!, inp.isB[rows_of[q]])) for q in qs),
        "pooled" => pooled,
        "reconciliation" => reconciliation,
        "n_market_cells" => length(inp.nat.cell_n),
        "inputs" => Dict(
            "sleep_link" => basename(lpath), "sleep_link_sha256" => _file_sha256(lpath),
            "state_transitions" => basename(tpath), "state_transitions_sha256" => _file_sha256(tpath),
            "demand_parquet" => basename(ppath), "demand_parquet_sha256" => _file_sha256(ppath),
            "rf_vintages" => basename(vpath), "rf_vintages_sha256" => _file_sha256(vpath),
            "sleep_national_phi_t" => "est$(estim)/" * basename(npath),
            "sleep_national_phi_t_sha256" => _file_sha256(npath),
            "sleep_national_phi_t_column" => "phi_t_" * replace(replace(spec_key, " " => "_"), "." => ""),
            "cf_phi_path_jl_sha256" => _file_sha256(joinpath(@__DIR__, "cf_phi_path.jl"))),
        "rho" => Dict(L.cols[k] => inp.rho[k] for k in eachindex(L.cols) if L.role[k] == :market),
        "t0_max_abs_phi0_minus_phi_mt" => inp.t0_maxdiff)
    opath = joinpath(cost, "phi_path_summary_E$(estim)_spec_$(spec).json")
    tmp = opath * ".tmp"
    open(tmp, "w") do io; JSON3.pretty(io, out); end
    mv(tmp, opath; force=true)
    println("\nE$estim spec $spec: $(nrow(df)) rows, $(length(qs)) launch quarters " *
            "($(first(qs))…$(last(qs))), $(length(inp.nat.cell_n)) market cells -> $opath")
    @printf("  %-11s %s\n", "pooled", join([lpad("h=$h", 9) for h in SUMMARY_HORIZONS]))
    for w in ("pop", "pop_table4", "row", "dep")
        @printf("  %-11s %s\n", w, join([@sprintf("%9.5f", x) for x in pooled[w]]))
    end
    @printf("  Table 4 Mean phi-hat (i, %d quarters) %.5f | (ii) - (i): mean %+.5f, max |.| %.5f | (iii) - (ii): mean %+.2e, max |.| %.2e\n",
            length(sleep_nat), reconciliation["table4_mean_phi_hat_i"], reconciliation["mean_ii_minus_i"],
            reconciliation["max_abs_ii_minus_i"], reconciliation["mean_iii_minus_ii"], reconciliation["max_abs_iii_minus_ii"])
    return out
end

function main(args)
    estims = [3, 4]; spec = 12
    eo = joinpath(resolve_of_root(), "BCB", "Egan_et_al_2025_Rep", "processed", "ESTIMATION_OUTPUT")
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--estim"; estims = [parse(Int, args[i + 1])]; i += 2
        elseif a == "--spec"; spec = parse(Int, args[i + 1]); i += 2
        elseif a == "--eo"; eo = args[i + 1]; i += 2
        else error("unknown argument $a (see the usage in the file's docstring)")
        end
    end
    for k in estims
        phi_path_summary(k, spec, eo)
        GC.gc()
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
