"""
blp_2_rc_coherence.jl
=============================
Orchestrator for the post-fix ("coherence") random-coefficients BLP estimation on
the Yale Bouchet HPC cluster (GPU), specification 12.

Background
----------
After the demand-degeneracy fixes (full-rank `has_ip`/`fgc_covered`) and the new
3-routine sleepiness methodology, the BLP RC estimation is run for THREE routines
(down from five), each consuming its own demand-prep parquet:

    E1  Local B-type      ->  demand_1_final_spec_12.parquet
    E2  Pooled Linear     ->  demand_2_final_spec_12.parquet
    E3  Pooled Logistic   ->  demand_3_final_spec_12.parquet

Each routine is warm-started from the local logit delta produced beforehand:

    BLP_RESULTS/logit_delta_E{1,2,3}_spec_12_coherence.bin   (logit-coherence output)

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
  julia --project=. --threads=auto blp_2_rc_coherence.jl --estim 1 --hpc --R 2000

  # all three routines sequentially on a single GPU:
  julia --project=. --threads=auto blp_2_rc_coherence.jl --all --hpc --R 2000

  # local dry-run timing (needs a CUDA GPU):
  julia --project=. blp_2_rc_coherence.jl --estim 1 --dry-run

The per-routine entry scripts `blp_2_rc_e{1,2,3}_coherence.jl` are even thinner:
they just `include` this file and call `run_coherence_routine(k)`.
"""

const COHERENCE_SPEC = 12

# Warm-start delta suffix. Must match COH_SUFFIX in blp_1_logit_coherence.jl, which
# writes logit_delta_E{k}_spec_12_coherence.{bin,jls}. The engine builds its
# warm-start filename from ENV["BLP_DELTA_SUFFIX"] (default "" = legacy name).
const COHERENCE_DELTA_SUFFIX = "_coherence"

# Routine id -> (label, description, demand-prep prefix). Matches the 3 sleepiness
# estimators and the demand-prep orchestrator's EST_LIST = [1, 2, 3].
const COHERENCE_ROUTINES = [
    (id = 1, label = "E1", desc = "Local B-type",    prefix = "demand_1"),
    (id = 2, label = "E2", desc = "Pooled Linear",   prefix = "demand_2"),
    (id = 3, label = "E3", desc = "Pooled Logistic", prefix = "demand_3"),
]

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

Run one coherence routine (1, 2, or 3) through the GPU engine for spec 12.
`passthrough` forwards engine flags (e.g. `["--hpc", "--R", "2000", "--stage",
"sigma", "--dry-run"]`). Defaults the stage to the full `sequence` unless the
caller supplies `--stage`.
"""
function run_coherence_routine(estim_id::Int; passthrough::Vector{String} = String[])
    idx = findfirst(r -> r.id == estim_id, COHERENCE_ROUTINES)
    idx === nothing &&
        error("Coherence routine must be 1 (E1), 2 (E2), or 3 (E3); got $estim_id")
    r = COHERENCE_ROUTINES[idx]

    pass = _strip_controlled(passthrough)
    is_hpc = "--hpc" in pass
    local_dir = nothing
    if (li = findfirst(==("--local-dir"), pass)) !== nothing && li < length(pass)
        local_dir = pass[li + 1]
    end

    println("\n", "="^72)
    println("  COHERENCE $(r.label): $(r.desc)  |  spec $COHERENCE_SPEC")
    println("  input : $(r.prefix)_final_spec_$(COHERENCE_SPEC).parquet")
    ws = warm_start_path(estim_id; is_hpc = is_hpc, local_dir = local_dir)
    println("  warm  : $(basename(ws))")
    println("="^72)
    if !isfile(ws)
        @warn "Coherence $(r.label): warm-start delta not found — engine will fall " *
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
    ENV["BLP_DELTA_SUFFIX"]  = COHERENCE_DELTA_SUFFIX
    ENV["BLP_OUTPUT_SUFFIX"] = COHERENCE_DELTA_SUFFIX *
                               (COHERENCE_ENGINE == "numerical" ? "_num" : "")

    empty!(ARGS); append!(ARGS, vcat(base, pass))
    main_gpu()
end

"""Run all three coherence routines sequentially (single GPU)."""
function run_all_coherence(; passthrough::Vector{String} = String[])
    for r in COHERENCE_ROUTINES
        run_coherence_routine(r.id; passthrough = passthrough)
    end
end

# ── Direct CLI entry: `--estim k` (one routine) or `--all` (all three) ──────────
function _coherence_main()
    a = copy(ARGS)
    if "--all" in a
        run_all_coherence(; passthrough = filter(!=("--all"), a))
    else
        ei = findfirst(==("--estim"), a)
        (ei === nothing || ei == length(a)) &&
            error("Provide `--estim {1,2,3}` or `--all` (plus engine flags). Got: $a")
        estim_id = parse(Int, a[ei + 1])
        deleteat!(a, ei:ei + 1)          # run_coherence_routine re-adds --estim
        run_coherence_routine(estim_id; passthrough = a)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    _coherence_main()
end
