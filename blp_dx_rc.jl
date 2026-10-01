"""
blp_dx_rc.jl
============
Entry point of the random-coefficients ladder for the `dx` demand variant (blp_dx.jl): the D-type
dummy in the linear part, everything else as blp_rc.jl runs it. One routine per invocation, on the
GPU IFT engine.

It include()s blp_rc.jl (and through it the engine) and blp_dx.jl, then for routine `--estim k`:

  1. NESTING CHECK, before any GPU work: the engine's linear step on the logit δ̂ under both
     designs, each against the logit sub-model it nests (`full`, `full_dtype`). The record goes to
     blp_linear_E{k}_spec_12_dx.json; a variant that does not reproduce `full_dtype` stops here.
  2. THE LADDER, through blp_rc.jl's `run_routine` with the variant switched on and the output
     suffix `_dx`: results, checkpoints and summaries are the engine's files with `_dx` before the
     extension, and each stage warm-starts θ₂ from the variant's own previous checkpoint. The
     logit δ̂ warm start is the main specification's file, which does not depend on X₁.
  3. POST-CHECK of every stage asked for: the result exists, carries the ten θ₁ names and the
     variant's record; a sidecar blp_meta_E{k}_spec_12_{stage}_dx.json is written next to it.
     If the SE guard tripped at the stage whose SEs the variant reports (`DX_SE_STAGE`), the
     process ends with a nonzero status after the point estimates are on disk.

Only BLP_ENGINE=ift is accepted: the numerical and CUE engines carry their own suffixes and seeds
and are not part of the variant. Stage `logit` is refused (it would rewrite the shared δ̂).

Usage
-----
  julia --project=. --threads=auto blp_dx_rc.jl --estim 3 --hpc --R 2000 --stage sigma,rc2,rc3,rc4,ext1
  julia --project=. --threads=auto blp_dx_rc.jl --estim 3 --hpc --R 2000 --stage ext2
  julia --project=. blp_dx_rc.jl --estim 3 --dry-run            # timing only, writes nothing

The cluster runs it through blp_dx_stage_job.sh; dx_suite_20261001.sh builds the chain.
"""

let _eng = lowercase(get(ENV, "BLP_ENGINE", "ift"))
    _eng == "ift" || error("blp_dx_rc.jl runs the IFT engine only (BLP_ENGINE='$(_eng)').")
end

isdefined(Main, :run_routine) || include(joinpath(@__DIR__, "blp_rc.jl"))
isdefined(Main, :DX_SUFFIX) || include(joinpath(@__DIR__, "blp_dx.jl"))
DX_WRAPS_IFT || error("blp_dx_rc.jl: blp_dx.jl was loaded before the GPU engine, so the IFT entry " *
                      "point is not wrapped. Start a fresh session.")

"The stages a `--stage` value names, in the order the engine runs them."
function dx_stages(pass::Vector{String})::Vector{String}
    si = findfirst(==("--stage"), pass)
    val = (si === nothing || si == length(pass)) ? "sequence" : pass[si + 1]
    val == "sequence" && return ["sigma", "rc2", "rc3", "rc4", "full", "ext1", "ext2", "extended"]
    return occursin(',', val) ? String.(strip.(split(val, ','))) : [val]
end

"""
    dx_run_routine(estim_id; passthrough=String[])

Run routine `estim_id` of the variant: nesting check, the ladder stages `passthrough` names, the
post-check. `passthrough` carries the engine flags blp_rc.jl's `run_routine` takes.
"""
function dx_run_routine(estim_id::Int; passthrough::Vector{String} = String[])
    ENGINE == "ift" || error("dx: engine '$(ENGINE)' is not part of the variant (ift only).")
    pass   = _strip_controlled(passthrough)
    stages = dx_stages(pass)
    "logit" in stages && error("dx: stage 'logit' is not part of the variant; the ladder starts " *
                               "from the main specification's logit δ̂.")
    is_hpc    = "--hpc" in pass
    local_dir = (li = findfirst(==("--local-dir"), pass)) !== nothing && li < length(pass) ?
                pass[li + 1] : nothing
    dry       = "--dry-run" in pass
    in_dir, _, out_dir = get_paths(is_hpc; local_dir = local_dir)
    res_dir   = blp_dir(out_dir)

    println("\n", "="^72)
    println("  RC-BLP E$(estim_id), variant dx: X1 = [spread, X_COLS, $(DX_COL)] | suffix '$(DX_SUFFIX)'")
    println("  stages: $(join(stages, ", ")) | SEs reported at $(DX_SE_STAGE)")
    println("="^72)

    # ── 1. nesting check ──────────────────────────────────────────────────
    prefix = demand_prefix(estim_id, demand_search_dirs(in_dir, out_dir))
    prefix === nothing && error("dx: no demand-prep parquet for E$(estim_id) spec $(RC_SPEC).")
    ENV["BLP_DEMAND_PREFIX"] = prefix
    rec = dx_nesting_check(estim_id, RC_SPEC, Dict{String,Any}("hpc" => is_hpc, "local_dir" => local_dir))
    if !dry
        mkpath(res_dir)
        dx_write_json(joinpath(res_dir, dx_linear_name(estim_id, RC_SPEC)), rec)
    end

    # ── 2. the ladder ─────────────────────────────────────────────────────
    DX_ACTIVE[] = true
    ENGINE_SUFFIX["ift"] = DX_SUFFIX        # run_routine exports it as BLP_OUTPUT_SUFFIX
    try
        run_routine(estim_id; passthrough = pass)
    finally
        DX_ACTIVE[] = false
        ENGINE_SUFFIX["ift"] = ""
        ENV["BLP_OUTPUT_SUFFIX"] = ""
    end
    dry && return nothing

    # ── 3. post-check ─────────────────────────────────────────────────────
    se_failed = String[]
    for st in stages
        rpath = dx_result_path(out_dir, estim_id, RC_SPEC, st)
        isfile(rpath) || error("dx: stage '$(st)' of E$(estim_id) left no result ($(basename(rpath))); " *
                               "see the engine's messages above.")
        res = deserialize(rpath)
        get(res, "param_names_theta1", String[]) == dx_theta1_names() || error(
            "dx: $(basename(rpath)) does not carry the variant's ten θ₁ names.")
        haskey(res, "dx") || error("dx: $(basename(rpath)) has no variant record.")
        guard = res["dx"]["se_guard"]
        meta = Dict{String,Any}(
            "estim" => estim_id, "spec_id" => RC_SPEC, "stage" => st,
            "result" => basename(rpath),
            "param_names_theta1" => res["param_names_theta1"],
            "alpha" => res["theta1"][1], "dummy_D_type" => res["theta1"][end],
            "Q_value" => get(res, "Q_value", nothing), "converged" => get(res, "converged", nothing),
            "n_obs" => get(res, "n_obs", nothing),
            "se_reported" => st == DX_SE_STAGE,
            "dx" => res["dx"])
        dx_write_json(joinpath(res_dir, dx_meta_name(estim_id, RC_SPEC, st)), meta)
        gtxt = !guard["ran"] ? "not run" : guard["ok"] ? "ok (K=$(guard["K"]) on L=$(guard["L"]))" :
               "TRIPPED: $(guard["reason"])"
        log_status("[DX] E$(estim_id) $(st): alpha=$(round(res["theta1"][1], digits=4)) " *
                   "D=$(round(res["theta1"][end], digits=4)) Q=$(round(get(res, "Q_value", NaN), digits=5)) " *
                   "| SE guard $(gtxt)")
        (st == DX_SE_STAGE && guard["ran"] && !guard["ok"]) && push!(se_failed, st)
    end
    isempty(se_failed) || error(
        "dx: the SE guard tripped at $(join(se_failed, ", ")), the stage whose standard errors the " *
        "variant reports, for E$(estim_id). The point estimates are on disk; no SE was produced.")
    return nothing
end

function _dx_main()
    a  = copy(ARGS)
    ei = findfirst(==("--estim"), a)
    (ei === nothing || ei == length(a)) &&
        error("Provide `--estim N` (plus engine flags). Got: $a")
    estim_id = parse(Int, a[ei + 1])
    deleteat!(a, ei:ei + 1)
    dx_run_routine(estim_id; passthrough = a)
end

if abspath(PROGRAM_FILE) == @__FILE__
    _dx_main()
end
