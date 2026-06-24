"""
blp_2_rc_coherence.jl
=============================
Orchestrator for the post-fix ("coherence") random-coefficients BLP estimation on
the Yale Bouchet HPC cluster (GPU), specification 12.

Background
----------
After the demand-degeneracy fixes (full-rank `has_ip`/`fgc_covered`) and the expanded
sleepiness methodology, the BLP RC estimation runs per routine, each consuming its own
demand-prep parquet. The routine list is AUTO-DISCOVERED from the parquets on disk
(`coherence_prefix` / `discover_coherence_ids`), so a new routine needs no edit here:

    E1  Local B-type             ->  demand_1_spec_12.parquet
    E2  Pooled Linear            ->  demand_2_spec_12.parquet
    E3  Pooled Logistic          ->  demand_3_logistic_spec_12.parquet
    E4  Pooled Constrained Lin.  ->  demand_4_constrained_spec_12.parquet
    E5  Pooled Probit            ->  demand_5_probit_spec_12.parquet
    E6  Pooled Single-Index      ->  demand_6_index_spec_12.parquet
    E7  Pooled Joint Single-Idx  ->  demand_7_sijoint_spec_12.parquet
    (E8+ appear automatically once their demand-prep parquet is on disk.)

The DEFAULT cluster run targets E3, E6 and E7 (the headline sleepiness links); the
remaining routines stay available (ROUTINES env / --all-routines) for robustness. Each
routine is warm-started from the local logit delta produced beforehand:

    BLP_RESULTS/logit_delta_E{k}_spec_12_coherence.bin   (logit-coherence output)

(the logit step is owned by the local-logit workflow; this orchestrator only
*consumes* those deltas — see `warm_start_path`).  If a delta is missing the
underlying engine falls back to a log-share initialisation, but coherence runs are
meant to warm-start, so its absence is surfaced as a warning.

All results/checkpoints/summaries are written with a `_coherence` suffix
(via ENV["BLP_OUTPUT_SUFFIX"]) so they never clobber the legacy outputs, e.g.
`blp_results_E1_spec_12_sigma_coherence.jls`.

This file is a THIN driver over the validated GPU estimation engine
(`blp_2_estimation_gpu.jl`, IFT analytical gradient).  It does not re-implement the
estimation; it only fixes spec=12, maps routine -> `--estim`, and forwards engine
flags.  Switch to the numerical-gradient engine by exporting
`BLP_COHERENCE_ENGINE=numerical`.

Usage
-----
  # one routine (one GPU job — the normal cluster pattern):
  julia --project=. --threads=auto blp_2_rc_coherence.jl --estim 6 --hpc --R 2000

  # the default routine set (E3 + E6) sequentially on a single GPU:
  julia --project=. --threads=auto blp_2_rc_coherence.jl --all --hpc --R 2000

  # every routine (E1..E6) sequentially on a single GPU:
  julia --project=. --threads=auto blp_2_rc_coherence.jl --all-six --hpc --R 2000

  # local dry-run timing (needs a CUDA GPU):
  julia --project=. blp_2_rc_coherence.jl --estim 6 --dry-run

The per-routine entry scripts `blp_2_rc_e{1..6}_coherence.jl` are even thinner:
they just `include` this file and call `run_coherence_routine(k)`.
"""

const COHERENCE_SPEC = 12

# Warm-start delta suffix. Must match COH_SUFFIX in blp_1_logit_coherence.jl, which
# writes logit_delta_E{k}_spec_12_coherence.{bin,jls}. The engine builds its
# warm-start filename from ENV["BLP_DELTA_SUFFIX"] (default "" = legacy name).
const COHERENCE_DELTA_SUFFIX = "_coherence"

# The routine list is AUTO-DISCOVERED at runtime from the demand-prep parquets
# (demand_<id>_*_spec_<spec>.parquet, excluding legacy *_final_*), so adding a routine
# (E7, E8, …) needs no edit here — just its parquet on disk. These descriptions are
# cosmetic labels only.
const COHERENCE_DESC = Dict(
    1 => "Local B-type",        2 => "Pooled Linear",       3 => "Pooled Logistic",
    4 => "Pooled Constrained",  5 => "Pooled Probit",       6 => "Pooled Single-Index",
    7 => "Joint SI (sieve)",    8 => "Joint SI (kernel)",
)
coherence_desc(id::Int) = get(COHERENCE_DESC, id, "E$id")

"""Discover routine `estim`'s demand-prep prefix by scanning `input_dir` for
demand_<estim>_*_spec_<spec>.parquet (excluding legacy *_final_*). Returns the prefix
(e.g. "demand_7_sijoint") or nothing if absent."""
function coherence_prefix(estim::Int, input_dir::String; spec::Int = COHERENCE_SPEC)
    isdir(input_dir) || return nothing
    pat = Regex("^demand_$(estim)(?:_.*)?_spec_$(spec)\\.parquet\$")
    for f in sort(readdir(input_dir))
        occursin("_final_", f) && continue
        occursin(pat, f) && return replace(f, "_spec_$(spec).parquet" => "")
    end
    return nothing
end

"""Discover all available coherence routine ids in `input_dir` (sorted)."""
function discover_coherence_ids(input_dir::String; spec::Int = COHERENCE_SPEC)
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

# Default cluster routine set: the headline sleepiness links E3, E6, E7. Override with
# the ROUTINES env in the submit scripts, or --all-routines to run every discovered one.
const COHERENCE_DEFAULT_ROUTINES = [3, 6, 7]

# Estimation engine (GPU). IFT analytical gradient by default; set
# BLP_COHERENCE_ENGINE=numerical for the finite-difference engine. Both define
# `main_gpu()` and inherit `input_filename`, `get_paths`, `log_status`, …
const COHERENCE_ENGINE = lowercase(get(ENV, "BLP_COHERENCE_ENGINE", "ift"))
const _ENGINE_FILE = COHERENCE_ENGINE == "numerical" ?
                     "blp_1_estimation_gpu.jl" : "blp_2_estimation_gpu.jl"

if !isdefined(Main, :main_gpu)
    include(joinpath(@__DIR__, _ENGINE_FILE))
end

"""Resolve the expected warm-start delta path for a routine, the same way the
engine does (`get_paths(is_hpc; local_dir)` -> out_dir)."""
function warm_start_path(estim_id::Int; is_hpc::Bool, local_dir=nothing)
    _, _, out_dir = get_paths(is_hpc; local_dir = local_dir)
    return joinpath(out_dir,
        "logit_delta_E$(estim_id)_spec_$(COHERENCE_SPEC)$(COHERENCE_DELTA_SUFFIX).bin")
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
    run_coherence_routine(estim_id; passthrough=String[])

Run one coherence routine through the GPU engine for spec 12. The routine's demand-prep
prefix is auto-discovered from the parquets on disk. `passthrough` forwards engine flags
(e.g. `["--hpc", "--R", "2000", "--stage", "sigma", "--dry-run"]`). Defaults the stage to
the full `sequence` unless the caller supplies `--stage`.
"""
function run_coherence_routine(estim_id::Int; passthrough::Vector{String} = String[])
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
    prefix = coherence_prefix(estim_id, in_dir)
    println("\n", "="^72)
    println("  COHERENCE $label: $(coherence_desc(estim_id))  |  spec $COHERENCE_SPEC")
    if prefix === nothing
        avail = discover_coherence_ids(in_dir)
        error("Coherence $label: no demand-prep parquet " *
              "demand_$(estim_id)_*_spec_$(COHERENCE_SPEC).parquet found in\n    $in_dir\n" *
              "Available routines there: " *
              (isempty(avail) ? "(none)" : join("E" .* string.(avail), ", ")) *
              ". Upload $label's parquet (or run its demand prep) before submitting.")
    end
    in_path = joinpath(in_dir, "$(prefix)_spec_$(COHERENCE_SPEC).parquet")
    println("  input : $(basename(in_path))")
    ws = warm_start_path(estim_id; is_hpc = is_hpc, local_dir = local_dir)
    println("  warm  : $(basename(ws))")
    println("="^72)
    if !isfile(ws)
        @warn "Coherence $label: warm-start delta not found — engine will fall " *
              "back to log-share init. Run the local logit first." path = ws
    end

    # Fix estim + spec; default to the full RC sequence unless overridden.
    base = ["--estim", string(estim_id), "--spec", string(COHERENCE_SPEC)]
    if !("--stage" in pass)
        append!(base, ["--stage", "sequence"])
    end

    # Warm-start from the coherence-suffixed delta written by the logit-coherence step.
    # Output suffix is engine-aware so the numerical (blp_1) and IFT (blp_2) coherence
    # runs never clobber each other's results, and neither touches the legacy outputs:
    #   IFT (blp_2)      -> *_coherence
    #   numerical (blp_1) -> *_coherence_num
    ENV["BLP_DELTA_SUFFIX"]     = COHERENCE_DELTA_SUFFIX
    ENV["BLP_OUTPUT_SUFFIX"]    = COHERENCE_DELTA_SUFFIX *
                                  (COHERENCE_ENGINE == "numerical" ? "_num" : "")
    # Route input_filename to the fresh coherence parquet: the auto-discovered prefix
    # is passed explicitly so the engine reads e.g. demand_7_sijoint_spec_12.parquet.
    ENV["BLP_COHERENCE_INPUTS"] = "1"
    ENV["BLP_COHERENCE_PREFIX"] = prefix

    empty!(ARGS); append!(ARGS, vcat(base, pass))
    main_gpu()
end

"""Run a set of coherence routines sequentially on a single GPU. Defaults to the
headline routines (E3, E6, E7); pass `ids` (e.g. `1:7`) to run a different set."""
function run_all_coherence(; passthrough::Vector{String} = String[],
                            ids::Vector{Int} = COHERENCE_DEFAULT_ROUTINES)
    for id in ids
        run_coherence_routine(id; passthrough = passthrough)
    end
end

# ── Direct CLI entry: `--estim k` (one routine), `--all` (default E3+E6+E7), or
#    `--all-six`/`--all-routines` (every routine discovered on disk). ─────────────
function _coherence_main()
    a = copy(ARGS)
    if ("--all-six" in a) || ("--all-routines" in a)
        rest      = filter(x -> x ∉ ("--all-six", "--all-routines"), a)
        is_hpc    = "--hpc" in rest
        local_dir = (li = findfirst(==("--local-dir"), rest)) !== nothing && li < length(rest) ?
                    rest[li + 1] : nothing
        in_dir, _, _ = get_paths(is_hpc; local_dir = local_dir)
        run_all_coherence(; passthrough = rest, ids = discover_coherence_ids(in_dir))
    elseif "--all" in a
        run_all_coherence(; passthrough = filter(!=("--all"), a))   # default E3+E6+E7
    else
        ei = findfirst(==("--estim"), a)
        (ei === nothing || ei == length(a)) &&
            error("Provide `--estim N`, `--all` (default E3+E6+E7), or `--all-routines` " *
                  "(plus engine flags). Got: $a")
        estim_id = parse(Int, a[ei + 1])
        deleteat!(a, ei:ei + 1)          # run_coherence_routine re-adds --estim
        run_coherence_routine(estim_id; passthrough = a)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    _coherence_main()
end
