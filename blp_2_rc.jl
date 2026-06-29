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
wins per id), so a relabelled/new routine needs NO edit here. The 2026-06-24 scheme:

    E1 Local B-type   E2 Pooled Linear
    E3 Pooled Logistic            E4 Pooled Logistic + Time
    E5 Pooled Single-Index        E6 Pooled Single-Index + Time
    E7 Pooled Joint Single-Index  E8 Pooled Joint Single-Index + Time

The DEFAULT cluster run targets **E5-E8** (the single-index links + their +Time variants); the
rest stay available (ROUTINES env / --all-routines) for robustness. Each routine is
warm-started from the local logit delta produced beforehand:

    BLP_RESULTS/logit_delta_E{k}_spec_12.bin   (blp_1_logit.jl output)

(the logit step is owned by the local-logit workflow; this orchestrator only
*consumes* those deltas — see `warm_start_path`).  If a delta is missing the
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

  # the default routine set (E4 + E6 + E8) sequentially on a single GPU:
  julia --project=. --threads=auto blp_2_rc.jl --all --hpc --R 2000

  # every discovered routine sequentially on a single GPU:
  julia --project=. --threads=auto blp_2_rc.jl --all-routines --hpc --R 2000

  # local dry-run timing (needs a CUDA GPU):
  julia --project=. blp_2_rc.jl --estim 6 --dry-run

Routines are selected via `--estim N` / `--all` / `--all-routines`; the cluster submit
scripts (`submit_blp_rc_{all,grouped,stage}.sh`) drive this for the chains.
"""

const RC_SPEC = 12

# Warm-start delta suffix. The engine builds its warm-start filename from
# ENV["BLP_DELTA_SUFFIX"]; the logit (blp_1_logit.jl) writes the un-suffixed
# logit_delta_E{k}_spec_12.{bin,jls}, so the default "" is correct.
const DELTA_SUFFIX = ""

# The routine list is AUTO-DISCOVERED at runtime from the demand-prep parquets
# (demand_<id>_*_spec_<spec>.parquet, excluding legacy *_final_*), so adding a routine
# (E7, E8, …) needs no edit here — just its parquet on disk. These descriptions are
# cosmetic labels only.
# Routine scheme (2026-06-24 relabel): odd-numbered base links + their +Time variants.
const ROUTINE_DESC = Dict(
    1 => "Local B-type",            2 => "Pooled Linear",
    3 => "Pooled Logistic",         4 => "Pooled Logistic + Time",
    5 => "Pooled Single-Index",     6 => "Pooled Single-Index + Time",
    7 => "Pooled Joint Single-Idx", 8 => "Pooled Joint Single-Idx + Time",
)
routine_desc(id::Int) = get(ROUTINE_DESC, id, "E$id")

"""Discover routine `estim`'s demand-prep prefix by scanning `input_dir` for
demand_<estim>_*_spec_<spec>.parquet (excluding legacy *_final_*). If several non-final
parquets share the id, the MOST RECENTLY MODIFIED wins (so a leftover old-scheme file
can't shadow a freshly rebuilt one). Returns the prefix or nothing if absent."""
function demand_prefix(estim::Int, input_dir::String; spec::Int = RC_SPEC)
    isdir(input_dir) || return nothing
    pat = Regex("^demand_$(estim)(?:_.*)?_spec_$(spec)\\.parquet\$")
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
    return best_pfx
end

"""Discover all available routine ids in `input_dir` (sorted)."""
function discover_routine_ids(input_dir::String; spec::Int = RC_SPEC)
    isdir(input_dir) || return Int[]
    ids = Set{Int}()
    pat = Regex("^demand_(\\d+)(?:_.*)?_spec_$(spec)\\.parquet\$")
    for f in readdir(input_dir)
        occursin("_final_", f) && continue
        m = match(pat, f)
        m === nothing || push!(ids, parse(Int, m.captures[1]))
    end
    return sort!(collect(ids))
end

# Default cluster routine set: the single-index links and their +Time variants —
# E5 (Single-Index), E6 (Single-Index+Time), E7 (Joint Single-Index), E8 (Joint+Time).
# Override with the ROUTINES env in the submit scripts, or --all-routines for all.
const DEFAULT_ROUTINES = [5, 6, 7, 8]

# Estimation engine (GPU). Both engines live in blp_gpu_engine.jl: the IFT analytical
# gradient (`main_gpu_ift`, default) and the numerical finite-difference engine
# (`main_gpu_numerical`, BLP_ENGINE=numerical). They share input_filename,
# get_paths, log_status, … and the CPU baseline blp_1_estimation.jl (include()d there).
const ENGINE = lowercase(get(ENV, "BLP_ENGINE", "ift"))

if !isdefined(Main, :main_gpu_ift)
    include(joinpath(@__DIR__, "blp_gpu_engine.jl"))
end

"""Resolve the expected warm-start delta path for a routine, the same way the
engine does (`get_paths(is_hpc; local_dir)` -> out_dir)."""
function warm_start_path(estim_id::Int; is_hpc::Bool, local_dir=nothing)
    _, _, out_dir = get_paths(is_hpc; local_dir = local_dir)
    return joinpath(out_dir,
        "logit_delta_E$(estim_id)_spec_$(RC_SPEC)$(DELTA_SUFFIX).bin")
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
    in_dir, _, _ = get_paths(is_hpc; local_dir = local_dir)

    # Auto-discover the routine's input prefix. A missing parquet hard-fails HERE — the
    # engine would otherwise print "[!] Missing" and silently no-op every stage (empty
    # blp_summary_*.json, exit 0), propagating a fake "success" down the afterok chain.
    prefix = demand_prefix(estim_id, in_dir)
    println("\n", "="^72)
    println("  RC-BLP $label: $(routine_desc(estim_id))  |  spec $RC_SPEC")
    if prefix === nothing
        avail = discover_routine_ids(in_dir)
        error("RC-BLP $label: no demand-prep parquet " *
              "demand_$(estim_id)_*_spec_$(RC_SPEC).parquet found in\n    $in_dir\n" *
              "Available routines there: " *
              (isempty(avail) ? "(none)" : join("E" .* string.(avail), ", ")) *
              ". Upload $label's parquet (or run its demand prep) before submitting.")
    end
    in_path = joinpath(in_dir, "$(prefix)_spec_$(RC_SPEC).parquet")
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
    ENV["BLP_OUTPUT_SUFFIX"] = ENGINE == "numerical" ? "_num" : ""
    # Route input_filename to the demand parquet: the auto-discovered prefix is passed
    # explicitly so the engine reads e.g. demand_7_sijoint_spec_12.parquet.
    ENV["BLP_DEMAND_PREFIX"] = prefix
    # θ₂ box, applied UNIFORMLY to every routine so the cross-routine comparison shares one
    # box (default 5.0; an explicit export of either var still takes precedence). Only E5
    # actually binds at the old 2.0 — its demographic interaction π(FGC×Age65+) pinned there
    # (the earlier "σ₇" reading was a positional mislabel; it is a π, not a σ). E6/E7/E8 are
    # interior (max|θ₂| ≤ 1) so the wider box leaves them unchanged. Both engines read these,
    # so IFT and the numerical cross-check stay on the same box. Watch for any parameter that
    # pins at the NEW bound — that is a weak-identification signal (visible in blp_compare_*).
    ENV["BLP_SIGMA_UB"] = get(ENV, "BLP_SIGMA_UB", "5.0")
    ENV["BLP_PI_BOUND"] = get(ENV, "BLP_PI_BOUND", "5.0")

    empty!(ARGS); append!(ARGS, vcat(base, pass))
    ENGINE == "numerical" ? main_gpu_numerical() : main_gpu_ift()
end

"""Run a set of routines sequentially on a single GPU. Defaults to the single-index
routines (E5-E8); pass `ids` (e.g. `1:8`) to run a different set."""
function run_all_routines(; passthrough::Vector{String} = String[],
                          ids::Vector{Int} = DEFAULT_ROUTINES)
    for id in ids
        run_routine(id; passthrough = passthrough)
    end
end

# ── Direct CLI entry: `--estim k` (one routine), `--all` (default E5-E8), or
#    `--all-six`/`--all-routines` (every routine discovered on disk). ─────────────
function _main()
    a = copy(ARGS)
    if ("--all-six" in a) || ("--all-routines" in a)
        rest      = filter(x -> x ∉ ("--all-six", "--all-routines"), a)
        is_hpc    = "--hpc" in rest
        local_dir = (li = findfirst(==("--local-dir"), rest)) !== nothing && li < length(rest) ?
                    rest[li + 1] : nothing
        in_dir, _, _ = get_paths(is_hpc; local_dir = local_dir)
        run_all_routines(; passthrough = rest, ids = discover_routine_ids(in_dir))
    elseif "--all" in a
        run_all_routines(; passthrough = filter(!=("--all"), a))   # default E5-E8
    else
        ei = findfirst(==("--estim"), a)
        (ei === nothing || ei == length(a)) &&
            error("Provide `--estim N`, `--all` (default E5-E8), or `--all-routines` " *
                  "(plus engine flags). Got: $a")
        estim_id = parse(Int, a[ei + 1])
        deleteat!(a, ei:ei + 1)          # run_routine re-adds --estim
        run_routine(estim_id; passthrough = a)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    _main()
end
