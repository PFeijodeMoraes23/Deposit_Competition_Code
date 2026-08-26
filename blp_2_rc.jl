"""
blp_2_rc.jl
===========
Orchestrator for the random-coefficients BLP estimation on the Yale Bouchet HPC
cluster (GPU), specification 12.

Background
----------
The BLP RC estimation runs per routine, each consuming its own
`demand_<id>_*_spec_12.parquet`. The routine list — id AND prefix — is AUTO-DISCOVERED
from the parquets on disk (`demand_prefix` / `discover_routine_ids`; newest file
wins per id), so a relabelled/new routine needs NO edit here. The parquets are read from the
one directory the sleepiness phase's prep step writes — `data/output/demand_prep` on the
cluster, the local DEMAND_PREP off it — via `demand_search_dirs` in of_root.jl, shared with
the engine so the two cannot disagree about a routine's panel. The lineup:

    E1 Local B-type   E2 Pooled Linear
    E3 Pooled Single-Index        E4 Pooled Single-Index + Time

The DEFAULT cluster run targets **E3/E4** (the single-index links); the
rest stay available (ROUTINES env / --all-routines) for robustness. Each routine is
warm-started from the δ the logit step produced beforehand:

    logit_delta_E{k}_spec_12.bin    (the engine resolves it in data/input, where a
                                     hand-staged delta may be uploaded, then in the logit
                                     step folder `logit_dir(out_dir)` — data/output/logit —
                                     which is where blp_1_logit.jl writes it in either tree)

(the logit step owns those files; this orchestrator only
*consumes* them — see `warm_start_path`).  If a delta is missing the
underlying engine falls back to a log-share initialisation, but a warm start is
preferred, so its absence is surfaced as a warning.

Results/checkpoints/summaries are written un-suffixed for the IFT engine, e.g.
`blp_results_E1_spec_12_sigma.jls`; the numerical engine appends `_num` so the two
engines never clobber each other (via ENV["BLP_OUTPUT_SUFFIX"]).

This file is a THIN driver over the merged GPU estimation engine
(`blp_gpu_engine.jl`, IFT analytical gradient by default).  It does not re-implement the
estimation; it only fixes spec=12, maps routine -> `--estim`, and forwards engine
flags.  Switch to the numerical-gradient engine by exporting `BLP_ENGINE=numerical`.

Usage
-----
  # one routine (one GPU job — the normal cluster pattern):
  julia --project=. --threads=auto blp_2_rc.jl --estim 6 --hpc --R 2000

  # the default routine set (E3 + E4) sequentially on a single GPU:
  julia --project=. --threads=auto blp_2_rc.jl --all --hpc --R 2000

  # every discovered routine sequentially on a single GPU:
  julia --project=. --threads=auto blp_2_rc.jl --all-routines --hpc --R 2000

  # local dry-run timing (needs a CUDA GPU):
  julia --project=. blp_2_rc.jl --estim 6 --dry-run

Routines are selected via `--estim N` / `--all` / `--all-routines`; the cluster submit
scripts (`submit_blp_rc_{all,grouped,stage}.sh`) drive this for the chains.
"""

# The lineup, the routine descriptions and the headline spec id come from
# config/routines.toml, which utils/routines.py reads too. Guarded because several files in
# one session want the consts.
isdefined(Main, :ROUTINE_REGISTRY) || include(joinpath(@__DIR__, "routines.jl"))

const RC_SPEC = SPEC12_ID

# Warm-start delta suffix. The engine builds its warm-start filename from
# ENV["BLP_DELTA_SUFFIX"]; the logit step (blp_1_logit.jl) writes the un-suffixed
# logit_delta_E{k}_spec_12.{bin,jls} into `logit_dir(out_dir)`, so the default "" is correct.
const DELTA_SUFFIX = ""

# The routine list is AUTO-DISCOVERED at runtime from the demand-prep parquets
# (demand_<id>_*_spec_<spec>.parquet, excluding legacy *_final_*), so adding a routine
# (a new routine) needs no edit here — just its parquet on disk. ROUTINE_DESC/routine_desc
# (routines.jl) supply the cosmetic labels for the ids that discovery turns up.

"""Discover routine `estim`'s demand-prep prefix by scanning `input_dirs` for
demand_<estim>_*_spec_<spec>.parquet (excluding legacy *_final_*). The directories are
searched in order and the FIRST one holding a match settles the routine; `demand_search_dirs`
hands over one directory per tree, so that rule never has to arbitrate. Within that directory,
if several non-final parquets share the id the MOST RECENTLY MODIFIED wins (so a leftover
old-scheme file can't shadow a freshly rebuilt one). Returns the prefix or nothing if
absent."""
function demand_prefix(estim::Int, input_dirs::Vector{String}; spec::Int = RC_SPEC)
    pat = Regex("^demand_$(estim)(?:_.*)?_spec_$(spec)\\.parquet\$")
    for input_dir in input_dirs
        isdir(input_dir) || continue
        best_mt = -Inf; best_pfx = nothing
        for f in readdir(input_dir)
            occursin("_final_", f) && continue
            occursin(pat, f) || continue
            mt = mtime(joinpath(input_dir, f))
            if mt > best_mt
                best_mt = mt
                best_pfx = replace(f, "_spec_$(spec).parquet" => "")
            end
        end
        best_pfx === nothing || return best_pfx
    end
    return nothing
end

"""Discover all available routine ids across `input_dirs` (sorted, de-duplicated)."""
function discover_routine_ids(input_dirs::Vector{String}; spec::Int = RC_SPEC)
    ids = Set{Int}()
    pat = Regex("^demand_(\\d+)(?:_.*)?_spec_$(spec)\\.parquet\$")
    for input_dir in input_dirs
        isdir(input_dir) || continue
        for f in readdir(input_dir)
            occursin("_final_", f) && continue
            m = match(pat, f)
            m === nothing || push!(ids, parse(Int, m.captures[1]))
        end
    end
    return sort!(collect(ids))
end

# The default cluster routine set is DEFAULT_ROUTINES (routines.jl): the single-index links
# and their +Time variants, read from config/routines.toml's `link_ests`.
# Override with the ROUTINES env in the submit scripts, or --all-routines for all.

# Estimation engine (GPU). Both engines live in blp_gpu_engine.jl: the IFT analytical
# gradient (`main_gpu_ift`, default) and the numerical finite-difference engine
# (`main_gpu_numerical`, BLP_ENGINE=numerical). They share input_filename,
# get_paths, log_status, … and the CPU baseline blp_1_estimation.jl (include()d there).
const ENGINE = lowercase(get(ENV, "BLP_ENGINE", "ift"))
# Engine → output suffix. This Dict is the ONLY authority for suffixes, and membership in it is the
# engine allow-list. Guard placed BEFORE the engine include below so a typo fails in <1s with no CUDA
# initialisation: an unrecognised value previously fell through to IFT with suffix "" and silently
# OVERWROTE the production results/checkpoints/summaries.
const ENGINE_SUFFIX = Dict("ift" => "", "numerical" => "_num", "cue" => "_cue")
haskey(ENGINE_SUFFIX, ENGINE) || error(
    "BLP_ENGINE='$(ENGINE)' is not a recognised engine (" *
    join(sort(collect(keys(ENGINE_SUFFIX))), " | ") * "). Refusing to run: an unrecognised engine " *
    "would fall through to IFT with an empty output suffix and overwrite the production artifacts.")

if !isdefined(Main, :main_gpu_ift)
    include(joinpath(@__DIR__, "blp_gpu_engine.jl"))
end

"""Resolve the expected warm-start delta path for a routine, in the SAME order the engine
resolves it: in_dir (data/input, an operator-supplied delta) first, then `logit_dir(out_dir)`,
where the logit step writes on both trees. Returns the first existing candidate, or — when
neither is present — the `logit_dir` path, because that is the location the producing step
fills: a warning naming it points at the step that failed rather than at an upload slot the
pipeline does not fill on its own. Used only for the missing-delta warning; the engine does its
own resolution."""
function warm_start_path(estim_id::Int; is_hpc::Bool, local_dir=nothing)
    in_dir, _, out_dir = get_paths(is_hpc; local_dir = local_dir)
    _fname = "logit_delta_E$(estim_id)_spec_$(RC_SPEC)$(DELTA_SUFFIX).bin"
    for _cand in (joinpath(in_dir, _fname), joinpath(logit_dir(out_dir), _fname))
        isfile(_cand) && return _cand
    end
    return joinpath(logit_dir(out_dir), _fname)
end

"""Strip routine-controlled flags (`--estim`, `--spec`) from a passthrough vector
so the orchestrator can set them itself without ArgParse seeing duplicates."""
function _strip_controlled(a::Vector{String})
    out = String[]
    i = 1
    while i <= length(a)
        if a[i] in ("--estim", "--spec")
            i += 2                      # drop the flag and its value
        else
            push!(out, a[i]); i += 1
        end
    end
    return out
end

"""
    run_routine(estim_id; passthrough=String[])

Run one routine through the GPU engine for spec 12. The routine's demand-prep prefix is
auto-discovered from the parquets on disk. `passthrough` forwards engine flags
(e.g. `["--hpc", "--R", "2000", "--stage", "sigma", "--dry-run"]`). Defaults the stage to
the full `sequence` unless the caller supplies `--stage`.
"""
function run_routine(estim_id::Int; passthrough::Vector{String} = String[])
    pass = _strip_controlled(passthrough)
    is_hpc = "--hpc" in pass
    local_dir = nothing
    if (li = findfirst(==("--local-dir"), pass)) !== nothing && li < length(pass)
        local_dir = pass[li + 1]
    end

    label = "E$estim_id"
    in_dir, _, out_dir = get_paths(is_hpc; local_dir = local_dir)
    # Same search path the engine uses (of_root.jl `demand_search_dirs`): the single directory
    # the sleepiness phase's prep step writes, data/output/demand_prep on the cluster.
    in_dirs = demand_search_dirs(in_dir, out_dir)

    # Auto-discover the routine's input prefix. A missing parquet hard-fails HERE — the
    # engine would otherwise print "[!] Missing" and silently no-op every stage (empty
    # blp_summary_*.json, exit 0), propagating a fake "success" down the afterok chain.
    prefix = demand_prefix(estim_id, in_dirs)
    println("\n", "="^72)
    println("  RC-BLP $label: $(routine_desc(estim_id))  |  spec $RC_SPEC")
    if prefix === nothing
        avail = discover_routine_ids(in_dirs)
        error("RC-BLP $label: no demand-prep parquet " *
              "demand_$(estim_id)_*_spec_$(RC_SPEC).parquet found in\n" *
              describe_search_dirs(in_dirs...) * "\n" *
              "Available routines there: " *
              (isempty(avail) ? "(none)" : join("E" .* string.(avail), ", ")) *
              ". Upload $label's parquet (or run its demand prep) before submitting.")
    end
    in_path = resolve_in_then_out("$(prefix)_spec_$(RC_SPEC).parquet", in_dirs...)
    println("  input : $(basename(in_path))")
    ws = warm_start_path(estim_id; is_hpc = is_hpc, local_dir = local_dir)
    println("  warm  : $(basename(ws))")
    println("="^72)
    if !isfile(ws)
        @warn "RC-BLP $label: warm-start delta not found — engine will fall " *
              "back to log-share init. Run the local logit first." path = ws
    end

    # Fix estim + spec; default to the full RC sequence unless overridden.
    base = ["--estim", string(estim_id), "--spec", string(RC_SPEC)]
    if !("--stage" in pass)
        append!(base, ["--stage", "sequence"])
    end

    # Warm-start from the logit delta written by the logit step. The output suffix is
    # engine-aware so the numerical (blp_1) and IFT (blp_2) runs never clobber each
    # other's results:
    #   IFT (blp_2)       -> un-suffixed
    #   numerical (blp_1) -> *_num
    ENV["BLP_DELTA_SUFFIX"]  = DELTA_SUFFIX
    ENV["BLP_OUTPUT_SUFFIX"] = ENGINE_SUFFIX[ENGINE]
    # Route input_filename to the demand parquet: the auto-discovered prefix is passed
    # explicitly so the engine reads e.g. demand_3_index_spec_12.parquet.
    ENV["BLP_DEMAND_PREFIX"] = prefix
    # θ₂ box, applied UNIFORMLY to every routine so the cross-routine comparison shares one
    # box (default 5.0; an explicit export of either var still takes precedence). Only E3
    # actually binds at the old 2.0 — its demographic interaction π(FGC×Age65+) pinned there
    # (the earlier "σ₇" reading was a positional mislabel; it is a π, not a σ). E4 is
    # interior (max|θ₂| ≤ 1) so the wider box leaves them unchanged. Both engines read these,
    # so IFT and the numerical cross-check stay on the same box. Watch for any parameter that
    # pins at the NEW bound — that is a weak-identification signal (visible in blp_compare_*).
    ENV["BLP_SIGMA_UB"] = get(ENV, "BLP_SIGMA_UB", "5.0")
    ENV["BLP_PI_BOUND"] = get(ENV, "BLP_PI_BOUND", "5.0")

    empty!(ARGS); append!(ARGS, vcat(base, pass))
    # cue rides the NUMERICAL main (its FD gradient stays consistent when the objective changes);
    # the CUE objective is selected inside run_blp_estimation_gpu from BLP_ENGINE.
    ENGINE == "ift" ? main_gpu_ift() : main_gpu_numerical()
end

"""Run a set of routines sequentially on a single GPU. Defaults to the single-index
routines (E3/E4); pass `ids` (e.g. `1:4`) to run a different set."""
function run_all_routines(; passthrough::Vector{String} = String[],
                          ids::Vector{Int} = DEFAULT_ROUTINES)
    for id in ids
        run_routine(id; passthrough = passthrough)
    end
end

# ── Direct CLI entry: `--estim k` (one routine), `--all` (default E3/E4), or
#    `--all-six`/`--all-routines` (every routine discovered on disk). ─────────────
function _main()
    a = copy(ARGS)
    if ("--all-six" in a) || ("--all-routines" in a)
        rest      = filter(x -> x ∉ ("--all-six", "--all-routines"), a)
        is_hpc    = "--hpc" in rest
        local_dir = (li = findfirst(==("--local-dir"), rest)) !== nothing && li < length(rest) ?
                    rest[li + 1] : nothing
        in_dir, _, out_dir = get_paths(is_hpc; local_dir = local_dir)
        run_all_routines(; passthrough = rest,
                         ids = discover_routine_ids(demand_search_dirs(in_dir, out_dir)))
    elseif "--all" in a
        run_all_routines(; passthrough = filter(!=("--all"), a))   # default E3/E4
    else
        ei = findfirst(==("--estim"), a)
        (ei === nothing || ei == length(a)) &&
            error("Provide `--estim N`, `--all` (default E3/E4), or `--all-routines` " *
                  "(plus engine flags). Got: $a")
        estim_id = parse(Int, a[ei + 1])
        deleteat!(a, ei:ei + 1)          # run_routine re-adds --estim
        run_routine(estim_id; passthrough = a)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    _main()
end
