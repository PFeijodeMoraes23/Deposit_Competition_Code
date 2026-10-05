"""
blp_dx_rc.jl
============
Entry point of the random-coefficients ladder for the `dx` demand variant (blp_dx.jl): the D-type
dummy in the linear part, everything else as blp_rc.jl runs it. One routine per invocation, on the
GPU IFT engine.

It include()s blp_rc.jl (and through it the engine) and blp_dx.jl, then for routine `--estim k`:

  0. SELF-TEST of the SE wrapper on synthetic moments (milliseconds): a wrapper that does not
     work in this Julia stops the job here.
  1. NESTING CHECK, before any GPU work: the engine's linear step on the logit δ̂ under both
     designs, each against the logit sub-model it nests (`full`, `full_dtype`). The record goes to
     blp_linear_E{k}_spec_12_dx.json; a variant that does not reproduce `full_dtype` stops here.
  2. THE LADDER, through blp_rc.jl's `run_routine` with the variant switched on and the output
     suffix `_dx`: results, checkpoints and summaries are the engine's files with `_dx` before the
     extension. It is a cold ladder: no stage is seeded from the main specification
     (BLP_THETA2_INIT_FILE is cleared). rc2, rc3, rc4, ext2 and extended start θ₂ from the
     variant's own previous checkpoint. ext1 starts from the seed: the engine looks for a `full`
     checkpoint, which this ladder (like the main run's grouped ladder) does not write. The logit
     δ̂ warm start is the main specification's file, which does not depend on X₁.
     A stage is not run when an earlier stage of the same job left no result (blp_dx.jl
     `dx_require_previous`): the engine itself would go on to the next rung from a cold start.
  3. POST-CHECK of every stage asked for: the result exists, carries the ten θ₁ names and the
     variant's record; a sidecar blp_meta_E{k}_spec_12_{stage}_dx.json is written next to it,
     with `se_reported` true only for stage `DX_SE_STAGE` and only when its joint standard errors
     exist. If that stage has none (the guard tripped, or the engine's SE block failed), the
     process ends with a nonzero status after the point estimates are on disk. The verdict is read
     from the result on disk, so a re-run of a ladder whose `DX_SE_STAGE` result is there ends
     the same way. `DX_ALLOW_NO_SE=1` in the environment accepts that stage without joint
     standard errors: the process goes on, the sidecar says `se_reported` false with the reason
     and `se_missing_accepted` true, and the tables print no standard error for the column.

Only BLP_ENGINE=ift is accepted: the numerical and CUE engines carry their own suffixes and seeds
and are not part of the variant. Stage `logit` is refused (it would rewrite the shared δ̂).

`--dry-run` (the smoke) runs steps 0 and 1 without writing, then the engine's own dry run of each
stage: the design with the D column, the rank check, the GPU buffers and share check, ten
contraction iterations and the IFT forward passes. No result, checkpoint or sidecar is written;
the engine does write one empty `blp_summary_E{k}_{stage}_gpu_ift_dx.json` per stage, which the
real stage overwrites. The engine catches a stage that throws and skips a stage whose result is
on disk, so the dry run ends with a nonzero status unless every step left positive evidence
(blp_dx.jl `dx_smoke_verdict`): `[DX SMOKE] PASS` is printed only then.

Usage
-----
  julia --project=. --threads=auto blp_dx_rc.jl --estim 3 --hpc --R 2000 --stage sigma,rc2,rc3,rc4,ext1
  julia --project=. --threads=auto blp_dx_rc.jl --estim 3 --hpc --R 2000 --stage ext2
  julia --project=. --threads=auto blp_dx_rc.jl --estim 3 --hpc --R 2000 --stage sigma --dry-run

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

"`DX_ALLOW_NO_SE=1`: stage `DX_SE_STAGE` without joint standard errors is accepted and flagged."
dx_allow_no_se()::Bool = get(ENV, "DX_ALLOW_NO_SE", "0") == "1"

"""
    dx_run_routine(estim_id; passthrough=String[])

Run routine `estim_id` of the variant: self-test, nesting check, the ladder stages `passthrough`
names, the post-check. `passthrough` carries the engine flags blp_rc.jl's `run_routine` takes.
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
    println("  stages: $(join(stages, ", ")) | SEs reported at $(DX_SE_STAGE)" * (dry ? " | DRY RUN" : ""))
    println("="^72)

    # The variant's ladder is cold: an explicit θ₂ seed (the cross-engine hook of the main
    # pipeline) would start every stage of this job from another fit.
    if !isempty(get(ENV, "BLP_THETA2_INIT_FILE", ""))
        log_status("[DX] BLP_THETA2_INIT_FILE='$(ENV["BLP_THETA2_INIT_FILE"])' is ignored: the variant's ladder takes no seed")
        delete!(ENV, "BLP_THETA2_INIT_FILE")
    end

    # ── 0. self-test, 1. nesting check ────────────────────────────────────
    dx_selftest_se()
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
    DX_RUN_STAGES[] = stages
    empty!(DX_DRY)
    ENGINE_SUFFIX["ift"] = DX_SUFFIX        # run_routine exports it as BLP_OUTPUT_SUFFIX
    try
        run_routine(estim_id; passthrough = pass)
    finally
        DX_ACTIVE[] = false
        DX_RUN_STAGES[] = String[]
        ENGINE_SUFFIX["ift"] = ""
        ENV["BLP_OUTPUT_SUFFIX"] = ""
    end
    if dry
        miss = dx_smoke_verdict(estim_id, stages, rec)
        isempty(miss) || error("dx smoke: FAIL for E$(estim_id), stage(s) $(join(stages, ", ")). " *
                               "Not shown by this dry run: " * join(miss, "; ") * ".")
        ev = DX_DRY[stages[1]]
        log_status("[DX SMOKE] PASS E$(estim_id): self-test; nesting against logit full_dtype " *
                   "(max rel. diff $(round(rec["dx"]["max_rel_diff"], sigdigits=3))); stage(s) " *
                   "$(join(stages, ", ")): design with the D column ($(ev["d_rows"]) digital rows of " *
                   "$(ev["n_rows"])), rank $(ev["rank"]) on $(ev["rank_cols"]) columns, the engine's dry " *
                   "run returned. No result was written.")
        return nothing
    end

    # ── 3. post-check ─────────────────────────────────────────────────────
    allow = dx_allow_no_se()
    no_se = String[]
    for st in stages
        rpath = dx_result_path(out_dir, estim_id, RC_SPEC, st)
        isfile(rpath) || error("dx: stage '$(st)' of E$(estim_id) left no result ($(basename(rpath))); " *
                               "no later stage of this job was run. See the engine's messages above.")
        res = deserialize(rpath)
        get(res, "param_names_theta1", String[]) == dx_theta1_names() || error(
            "dx: $(basename(rpath)) does not carry the variant's ten θ₁ names.")
        haskey(res, "dx") || error("dx: $(basename(rpath)) has no variant record.")
        dxr    = res["dx"]
        joint  = get(dxr, "se_joint", false)
        status = get(dxr, "se_status", "not recorded")
        wanted = get(dxr, "se_method_requested", "") in ("wcb", "sandwich")
        se_missing = st == DX_SE_STAGE && wanted && !joint
        meta = Dict{String,Any}(
            "estim" => estim_id, "spec_id" => RC_SPEC, "stage" => st,
            "result" => basename(rpath),
            "param_names_theta1" => res["param_names_theta1"],
            "alpha" => res["theta1"][1], "dummy_D_type" => res["theta1"][end],
            "Q_value" => get(res, "Q_value", nothing), "converged" => get(res, "converged", nothing),
            "n_obs" => get(res, "n_obs", nothing),
            "se_reported" => st == DX_SE_STAGE && joint,
            "se_joint" => joint, "se_status" => status,
            "se_missing_accepted" => se_missing && allow,
            "dx" => dxr)
        dx_write_json(joinpath(res_dir, dx_meta_name(estim_id, RC_SPEC, st)), meta)
        log_status("[DX] E$(estim_id) $(st): alpha=$(round(res["theta1"][1], digits=4)) " *
                   "D=$(round(res["theta1"][end], digits=4)) Q=$(round(get(res, "Q_value", NaN), digits=5)) " *
                   "| SE: $(joint ? "joint ($(get(res, "se_method", "?")))" : "NONE, " * status)")
        if se_missing && allow
            log_status("[DX] E$(estim_id) $(st): NO JOINT STANDARD ERRORS ($(status)), ACCEPTED under " *
                       "DX_ALLOW_NO_SE=1. The ladder goes on; the sidecar says se_reported = false and " *
                       "the tables print no standard error for this column.")
        elseif se_missing
            push!(no_se, "$(st) ($(status))")
        end
    end
    isempty(no_se) || error(
        "dx: no joint standard errors at $(join(no_se, ", ")), the stage whose standard errors the " *
        "variant reports, for E$(estim_id). The point estimates are on disk; the result carries no SE. " *
        "A re-run reads the same result and stops here again; DX_ALLOW_NO_SE=1 accepts the stage " *
        "without standard errors (flagged in the sidecar and in the tables).")
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
