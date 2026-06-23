"""
blp_2_rc_coherence.jl
=============================
Orchestrator for the post-fix ("coherence") random-coefficients BLP estimation on
the Yale Bouchet HPC cluster (GPU), specification 12.

Background
----------
After the demand-degeneracy fixes (full-rank `has_ip`/`fgc_covered`) and the expanded
sleepiness methodology, the BLP RC estimation can be run for SIX routines, each
consuming its own demand-prep parquet:

    E1  Local B-type             ->  demand_1_spec_12.parquet
    E2  Pooled Linear            ->  demand_2_spec_12.parquet
    E3  Pooled Logistic          ->  demand_3_logistic_spec_12.parquet
    E4  Pooled Constrained Lin.  ->  demand_4_constrained_spec_12.parquet
    E5  Pooled Probit            ->  demand_5_probit_spec_12.parquet
    E6  Pooled Single-Index      ->  demand_6_index_spec_12.parquet

The DEFAULT cluster run targets E3 and E6 (the two headline sleepiness links); the
remaining routines stay available for robustness. Each routine is warm-started from the
local logit delta produced beforehand:

    BLP_RESULTS/logit_delta_E{1..6}_spec_12_coherence.bin   (logit-coherence output)

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

# Routine id -> (label, description, demand-prep prefix). Matches the 6 sleepiness
# estimators (estimation_{1..6}) and their demand-prep parquets.
const COHERENCE_ROUTINES = [
    (id = 1, label = "E1", desc = "Local B-type",          prefix = "demand_1"),
    (id = 2, label = "E2", desc = "Pooled Linear",         prefix = "demand_2"),
    (id = 3, label = "E3", desc = "Pooled Logistic",       prefix = "demand_3_logistic"),
    (id = 4, label = "E4", desc = "Pooled Constrained",    prefix = "demand_4_constrained"),
    (id = 5, label = "E5", desc = "Pooled Probit",         prefix = "demand_5_probit"),
    (id = 6, label = "E6", desc = "Pooled Single-Index",   prefix = "demand_6_index"),
    (id = 7, label = "E7", desc = "Joint SI (sieve)",      prefix = "demand_7_sijoint"),
    (id = 8, label = "E8", desc = "Joint SI (kernel)",     prefix = "demand_8_sikernel"),
]

# Default cluster routine set: the two headline sleepiness links (E3 logistic, E6 index).
const COHERENCE_DEFAULT_ROUTINES = [3, 6]

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

Run one coherence routine (1..6) through the GPU engine for spec 12.
`passthrough` forwards engine flags (e.g. `["--hpc", "--R", "2000", "--stage",
"sigma", "--dry-run"]`). Defaults the stage to the full `sequence` unless the
caller supplies `--stage`.
"""
function run_coherence_routine(estim_id::Int; passthrough::Vector{String} = String[])
    idx = findfirst(r -> r.id == estim_id, COHERENCE_ROUTINES)
    idx === nothing &&
        error("Coherence routine must be one of 1..8 (E1..E8); got $estim_id")
    r = COHERENCE_ROUTINES[idx]

    pass = _strip_controlled(passthrough)
    is_hpc = "--hpc" in pass
    local_dir = nothing
    if (li = findfirst(==("--local-dir"), pass)) !== nothing && li < length(pass)
        local_dir = pass[li + 1]
    end

    println("\n", "="^72)
    println("  COHERENCE $(r.label): $(r.desc)  |  spec $COHERENCE_SPEC")
    println("  input : $(r.prefix)_spec_$(COHERENCE_SPEC).parquet")
    ws = warm_start_path(estim_id; is_hpc = is_hpc, local_dir = local_dir)
    println("  warm  : $(basename(ws))")
    println("="^72)
    if !isfile(ws)
        @warn "Coherence $(r.label): warm-start delta not found — engine will fall " *
              "back to log-share init. Run the local logit first." path = ws
    end

    # Hard-fail on a missing input parquet. Otherwise the engine just prints
    # "[!] Missing: <path>" and silently no-ops EVERY stage (writing empty
    # blp_summary_*.json files and exiting 0), which under the afterok chain would
    # propagate a fake "success" to the ext2/extended jobs. Fail loudly at startup
    # so the job — and the chain — stop with a clear, actionable message.
    in_dir, _, _ = get_paths(is_hpc; local_dir = local_dir)
    in_path = joinpath(in_dir, "$(r.prefix)_spec_$(COHERENCE_SPEC).parquet")
    isfile(in_path) || error(
        "Coherence $(r.label): required input parquet not found —\n    $in_path\n" *
        "Upload $(basename(in_path)) to the cluster's data/input/ directory before submitting.")

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
    # Route input_filename to the fresh coherence parquets (no _final suffix):
    #   E1 → demand_1_spec_12.parquet            E4 → demand_4_constrained_spec_12.parquet
    #   E2 → demand_2_spec_12.parquet            E5 → demand_5_probit_spec_12.parquet
    #   E3 → demand_3_logistic_spec_12.parquet   E6 → demand_6_index_spec_12.parquet
    ENV["BLP_COHERENCE_INPUTS"] = "1"

    empty!(ARGS); append!(ARGS, vcat(base, pass))
    main_gpu()
end

"""Run a set of coherence routines sequentially on a single GPU. Defaults to the two
headline routines (E3, E6); pass `ids` (e.g. `1:6`) to run a different set."""
function run_all_coherence(; passthrough::Vector{String} = String[],
                            ids::Vector{Int} = COHERENCE_DEFAULT_ROUTINES)
    for id in ids
        run_coherence_routine(id; passthrough = passthrough)
    end
end

# ── Direct CLI entry: `--estim k` (one routine), `--all` (default E3+E6), or
#    `--all-six`/`--all-routines` (every routine, now E1..E8). ───────────────────
function _coherence_main()
    a = copy(ARGS)
    if ("--all-six" in a) || ("--all-routines" in a)
        run_all_coherence(; passthrough = filter(x -> x ∉ ("--all-six", "--all-routines"), a),
                            ids = [r.id for r in COHERENCE_ROUTINES])
    elseif "--all" in a
        run_all_coherence(; passthrough = filter(!=("--all"), a))   # default E3 + E6
    else
        ei = findfirst(==("--estim"), a)
        (ei === nothing || ei == length(a)) &&
            error("Provide `--estim {1..8}`, `--all` (default E3+E6), or `--all-routines` " *
                  "(plus engine flags). Got: $a")
        estim_id = parse(Int, a[ei + 1])
        deleteat!(a, ei:ei + 1)          # run_coherence_routine re-adds --estim
        run_coherence_routine(estim_id; passthrough = a)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    _coherence_main()
end
