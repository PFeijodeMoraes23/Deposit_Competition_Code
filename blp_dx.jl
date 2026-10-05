# blp_dx.jl
# =========
# The `dx` robustness variant of the demand model: a digital-vs-brick-and-mortar firm dummy
# (`dummy_D_type` = 1 on D rows, i.e. `is_B == false`) enters the LINEAR part only.
#
#   X₁ (θ₁ design)      = [spread, X_COLS..., dummy_D_type]     10 columns, D last
#   random coefficients = prod_vec = [spread, X_COLS...]        unchanged (coef_dim = 1 + L_PROD)
#   moments             = the excluded instruments              unchanged
#
# The column order is the logit's `full_dtype` sub-model (blp_logit.jl `build_matrices` with
# add_dtype=true), so that sub-model is the θ₂ = 0 nest of this variant exactly as `full` is the
# θ₂ = 0 nest of the main model. No constant is added. The demand parquets do not carry the dummy;
# it is built here from `is_B`, the way the logit builds it.
#
# HOW IT ATTACHES. This file changes no other file. It is include()d after the engine by the
# variant's own entry points (blp_dx_rc.jl on the cluster, blp_dx_linear.jl for the linear step)
# and wraps five engine functions. Each wrapper calls the engine's own method through
# `Base.invoke_in_world` at the world age recorded before the wrapper was defined, so the engine
# code that runs is the engine's, and with the switch `DX_ACTIVE[]` off every wrapper returns the
# engine's result untouched:
#
#   build_regressor_matrices     appends the D column to x_mat (the only change to the model)
#   check_design_rank            names the tenth column in its messages
#   gmm_cluster_ses              the SE guard (below)
#   run_blp_estimation_ift_gpu   refuses a stage whose predecessors in the same job left no result;
#                                records the ten θ₁ names and whether joint SEs exist in the result;
#                                in a dry run, records what the engine's dry run did (the smoke)
#   run_blp_estimation           refuses under the variant: the CPU path writes unsuffixed names
#
# NAMESPACE. Every artifact of the variant carries the suffix `_dx` through the engine's
# `output_suffix()` (ENV["BLP_OUTPUT_SUFFIX"]); `dx_require_namespace` refuses to estimate when the
# switch is on and the suffix is anything else, so a variant fit cannot be written under a main
# name. `dx_written_names` lists every file name the variant writes.
#
# SE GUARD. `gmm_cluster_ses` builds the Jacobian D = [−Z'X, Z'∂δ/∂θ₂] on the L excluded moments,
# so it needs K₁ + K₂(kept) ≤ L. The variant has K₁ = 10 against the main model's 9, which makes
# `extended` exactly identified in those terms with σ on its bound and over-parameterised (17 > 16)
# if σ leaves it; D'WD is then singular and the routine's `inv` falls back to `pinv` without a
# word. Under the variant the wrapper counts the kept parameters with the routine's own profiling
# rule and forms D'WD itself first. It returns NaN for every SE and p-value, logs the reason and
# records it in the result when (a) the count exceeds L, or (b) D'WD is numerically singular: the
# reciprocal condition number of D'WD scaled to unit diagonal is below K·eps (`DX_RCOND_FLOOR`),
# or `inv` throws. Nothing short of that is guarded: a badly but not singularly conditioned D'WD
# gives large standard errors, which are reported as they are.
#
# NO JOINT SE, NO SE. The engine wraps its SE block in a try/catch that, on any failure, keeps the
# linear-IV θ₁ standard errors and leaves `se_method` saying "wcb". Under the variant a result
# whose joint SEs do not exist (the guard tripped, or the engine's block failed before or after
# it) carries NaN standard errors, no p-values, `se_method = "none"` and the reason in
# `res["dx"]["se_status"]`; `res["dx"]["se_joint"]` is true only when the joint routine returned.
# The SEs the variant reports are those of stage `DX_SE_STAGE` (ext1: 14 or 15 parameters on 16
# moments); blp_dx_rc.jl ends with a nonzero status when that stage has no joint SE, unless
# DX_ALLOW_NO_SE=1 accepts it (flagged in the sidecar; the tables then print no SE).
#
# THE SMOKE. The engine's stage loop catches an exception and goes on, and it skips a stage whose
# result is on disk, so a dry run that exercised nothing ends like one that passed. Under the
# variant the wrappers record, per stage, what the engine's dry run did (`DX_DRY`): the entry point
# reached, the design built with the D column, the rank check's verdict, the engine's `dry_run`
# marker returned (which it returns only after its GPU share check, contraction iterations and IFT
# forward passes), any exception. `dx_smoke_verdict` lists what is missing; blp_dx_rc.jl fails a
# dry run unless the list is empty.

isdefined(Main, :DX_SUFFIX) && error("blp_dx.jl is already loaded in this session; include it once.")
isdefined(Main, :X_COLS) || Base.include(Main, joinpath(@__DIR__, "blp_engine_cpu.jl"))

const DX_SUFFIX   = "_dx"
const DX_COL      = "dummy_D_type"
const DX_SE_STAGE = "ext1"
const DX_ACTIVE   = Ref(false)
const DX_SE_GUARD = Ref{Any}(nothing)     # verdict of the last gmm_cluster_ses call under the variant
const DX_RUN_STAGES = Ref{Vector{String}}(String[])   # the stages of the running job, in order (blp_dx_rc.jl)
const DX_QUIET = Ref(false)               # the self-test silences the guard's log lines
const DX_DRY = Dict{String,Any}()         # stage => what the engine's dry run did under the variant
const DX_DRY_TRACE = Ref{Any}(nothing)    # the entry of DX_DRY being filled while a dry-run stage runs
const DX_SELFTEST_RTOL = 1e-8             # self-test: the wrapper's SEs against the engine's, relative

dx_active()::Bool = DX_ACTIVE[]

"θ₁ names of the variant: the engine's, then the dummy (the logit `full_dtype` order)."
dx_theta1_names()::Vector{String} = vcat(["alpha"], X_COLS, [DX_COL])

"The D-type dummy of a demand panel: 1.0 where `is_B` is false (blp_logit.jl `build_matrices`)."
function dx_column(df::DataFrame)::Vector{Float64}
    "is_B" in names(df) || error("dx: the demand parquet has no `is_B` column, so the D-type dummy " *
                                 "cannot be built.")
    return Float64.(.!Bool.(coalesce.(df.is_B, false)))
end

"""Refuse to estimate the variant under any output suffix but `DX_SUFFIX`, and refuse the stage
that writes the logit δ warm start both specifications share."""
function dx_require_namespace(stage::AbstractString)
    output_suffix() == DX_SUFFIX || error(
        "dx: the variant is active but BLP_OUTPUT_SUFFIX='$(output_suffix())' (expected " *
        "'$(DX_SUFFIX)'). Refusing: its results would be written under another specification's " *
        "names. Run it through blp_dx_rc.jl.")
    stage == "logit" && error(
        "dx: stage 'logit' writes logit_delta_E{k}_spec_{s}.bin, the warm start the main " *
        "specification reads. The variant starts from that file and never rewrites it.")
    return nothing
end

"""
    dx_written_names(estim, spec_id, stages) -> Vector{String}

Every file NAME the variant writes for routine `estim`: per stage the engine's result (.jls,
.json), checkpoint (.jls, .bin) and summary, and this layer's sidecar; per routine the linear-step
record and the main-vs-variant comparison. All in `blp_dir(out_dir)`. (The suite's hand-off adds
one more file there per BBL launch, blp_bblfit_E{k}_spec_{s}_extended_dx<psi tag>.json, the
sha256 of the fit that launch was simulated on; its name depends on the psi tag.)
"""
function dx_written_names(estim::Int, spec_id::Int, stages::Vector{String})::Vector{String}
    out = String[]
    for st in stages
        push!(out, "blp_results_E$(estim)_spec_$(spec_id)_$(st)$(DX_SUFFIX).jls",
                   "blp_results_E$(estim)_spec_$(spec_id)_$(st)$(DX_SUFFIX).json",
                   "blp_checkpoint_E$(estim)_spec_$(spec_id)_$(st)$(DX_SUFFIX).jls",
                   "blp_checkpoint_E$(estim)_spec_$(spec_id)_$(st)$(DX_SUFFIX).bin",
                   "blp_summary_E$(estim)_$(st)_gpu_ift$(DX_SUFFIX).json",
                   dx_meta_name(estim, spec_id, st))
    end
    push!(out, dx_linear_name(estim, spec_id), dx_compare_name(estim, spec_id))
    return out
end
dx_meta_name(estim, spec_id, stage) = "blp_meta_E$(estim)_spec_$(spec_id)_$(stage)$(DX_SUFFIX).json"
dx_linear_name(estim, spec_id)      = "blp_linear_E$(estim)_spec_$(spec_id)$(DX_SUFFIX).json"
dx_compare_name(estim, spec_id)     = "blp_compare_E$(estim)_spec_$(spec_id)$(DX_SUFFIX).json"

"""
    dx_result_path(out_dir, estim, spec_id, stage) -> String

The variant's RC result for a stage: always the flat suffixed file in `blp_dir(out_dir)`, in the
cluster tree and in a local one. Local readers of the variant resolve through this and not through
cf_demand_eval.jl's `_result_path`, whose local `extended` branch returns the consolidated
main-specification fit whatever the suffix.
"""
dx_result_path(out_dir, estim, spec_id, stage)::String =
    joinpath(blp_dir(out_dir), "blp_results_E$(estim)_spec_$(spec_id)_$(stage)$(DX_SUFFIX).jls")

# ==========================================================================
# 1. The linear design
# ==========================================================================
const _DX_WORLD_ENGINE = Base.get_world_counter()

function build_regressor_matrices(df::DataFrame)
    spread_cols, x_mat, Z_mat, iv_cols = Base.invoke_in_world(
        _DX_WORLD_ENGINE, build_regressor_matrices, df)::Tuple{Vector{Float64},Matrix{Float64},
                                                              Matrix{Float64},Vector{String}}
    dx_active() || return spread_cols, x_mat, Z_mat, iv_cols
    size(x_mat, 2) == L_PROD || error("dx: expected $(L_PROD) product characteristics from the " *
                                      "engine, got $(size(x_mat, 2)).")
    d = dx_column(df)
    t = DX_DRY_TRACE[]
    if t !== nothing
        t["design_cols"] = size(x_mat, 2) + 1
        t["n_rows"] = length(d); t["d_rows"] = Int(sum(d))
    end
    return spread_cols, hcat(x_mat, d), Z_mat, iv_cols
end

function check_design_rank(pc::Precomp;
                           colnames::Vector{String}=vcat(["spread_hat"], X_COLS,
                                                         dx_active() ? [DX_COL] : String[]),
                           abort::Bool=true)::Int
    r = Base.invoke_in_world(_DX_WORLD_ENGINE, check_design_rank, pc;
                             colnames=colnames, abort=abort)::Int
    t = DX_DRY_TRACE[]
    if t !== nothing && dx_active()
        t["rank"] = r; t["rank_cols"] = length(colnames)
    end
    return r
end

# ==========================================================================
# 2. The SE guard
# ==========================================================================
"Floor on the reciprocal condition number of the scaled D'WD: below it the matrix is singular in double precision."
DX_RCOND_FLOOR(K::Int) = K * eps(Float64)

"""
    dx_se_guard(theta, Z, X, Ddelta, W; n_sigma, bound_tol, jac_tol) -> NamedTuple

Whether `gmm_cluster_ses` can produce standard errors for these inputs without a pseudo-inverse.
Counts the parameters the routine keeps after its own profiling (an on-bound σ, or a θ₂ column
whose moment Jacobian is negligible, is dropped) and forms the same D'WD:

  ok = false, reason "K > L"      when the kept count exceeds the L moments
  ok = false, reason "singular"   when D'WD is numerically singular: scaled to unit diagonal its
                                  reciprocal condition number (λ_min / λ_max) is below
                                  `DX_RCOND_FLOOR(K) = K·eps`, a diagonal entry is zero, or `inv`
                                  throws. Above that floor the routine's inverse is taken as it is.
"""
function dx_se_guard(theta::Vector{Float64}, Z::Matrix{Float64}, X::Matrix{Float64},
                     Ddelta::Matrix{Float64}, W::Matrix{Float64};
                     n_sigma::Int=0, bound_tol::Float64=1e-3, jac_tol::Float64=1e-8)
    L  = size(Z, 2)
    K1 = size(X, 2); K2 = size(Ddelta, 2)
    ZtX  = Z' * X
    ZtDd = K2 > 0 ? Z' * Ddelta : Matrix{Float64}(undef, L, 0)
    cn(A, j) = sqrt(sum(abs2, view(A, :, j)))
    scale = 0.0
    for k in 1:K1; scale = max(scale, cn(ZtX, k)); end
    for j in 1:K2; scale = max(scale, cn(ZtDd, j)); end
    keep = [j for j in 1:K2 if !((j <= n_sigma && abs(theta[K1 + j]) < bound_tol) ||
                                 (scale > 0 && cn(ZtDd, j) / scale < jac_tol))]
    Kk = K1 + length(keep)
    base = (K1=K1, K2=K2, K2_kept=length(keep), K=Kk, L=L)
    Kk > L && return merge(base, (ok=false, rcond=NaN, reason="K > L: $(Kk) estimated parameters on $(L) moments"))
    D    = hcat(-ZtX, ZtDd[:, keep])
    DtWD = (D' * W) * D
    d    = sqrt.(abs.(diag(DtWD)))
    any(iszero, d) && return merge(base, (ok=false, rcond=0.0,
        reason="singular: D'WD ($(Kk)×$(Kk)) has a zero diagonal entry"))
    A  = DtWD ./ (d * d')
    ev = eigvals(Symmetric((A .+ A') ./ 2))
    rc = minimum(ev) / maximum(ev)
    (isfinite(rc) && rc >= DX_RCOND_FLOOR(Kk)) || return merge(base, (ok=false, rcond=rc,
        reason="singular: scaled D'WD ($(Kk)×$(Kk)) has reciprocal condition number " *
               "$(round(rc, sigdigits=3)) < K·eps = $(round(DX_RCOND_FLOOR(Kk), sigdigits=3))"))
    invertible = try inv(DtWD); true catch; false end
    invertible || return merge(base, (ok=false, rcond=rc, reason="singular: D'WD ($(Kk)×$(Kk)) has no inverse"))
    return merge(base, (ok=true, rcond=rc, reason=""))
end

function gmm_cluster_ses(method::AbstractString, theta::Vector{Float64},
                         Z::Matrix{Float64}, X::Matrix{Float64}, Ddelta::Matrix{Float64},
                         xi::Vector{Float64}, W::Matrix{Float64}, cl::Vector{String};
                         B::Int=wcb_reps(), scheme::AbstractString=wcb_scheme(), seed::Int=0,
                         n_sigma::Int=0, bound_tol::Float64=1e-3, jac_tol::Float64=1e-8)
    if dx_active()
        g = dx_se_guard(theta, Z, X, Ddelta, W; n_sigma=n_sigma, bound_tol=bound_tol, jac_tol=jac_tol)
        DX_SE_GUARD[] = g
        if !g.ok
            DX_QUIET[] || log_status("  [DX SE GUARD] NO STANDARD ERRORS at this stage: $(g.reason) " *
                       "(K1=$(g.K1), K2 kept=$(g.K2_kept) of $(g.K2), L=$(g.L)). Every SE and " *
                       "p-value is NaN; the point estimates are unaffected.")
            K = length(theta)
            return fill(NaN, K), fill(NaN, K)
        end
        DX_QUIET[] || log_status("  [DX SE GUARD] ok: $(g.K) estimated parameters on $(g.L) moments " *
                   "(K1=$(g.K1), K2 kept=$(g.K2_kept) of $(g.K2); scaled D'WD rcond $(round(g.rcond, sigdigits=3)))")
    end
    return Base.invoke_in_world(_DX_WORLD_ENGINE, gmm_cluster_ses, method, theta, Z, X, Ddelta,
                                xi, W, cl; B=B, scheme=scheme, seed=seed, n_sigma=n_sigma,
                                bound_tol=bound_tol, jac_tol=jac_tol)::Tuple{Vector{Float64},Vector{Float64}}
end

# ==========================================================================
# 3. The estimation entry points
# ==========================================================================
# Recorded AFTER the wrappers above, so the engine's estimation routines, run at this world age,
# call them.
const _DX_WORLD_DESIGN = Base.get_world_counter()

"""
    dx_stamp!(res, stage) -> res

Stamp a variant result: the ten θ₁ names, and whether its standard errors are the joint GMM ones.

`se_joint` is true only when a joint method was asked for (BLP_SE_METHOD wcb | sandwich), the
guard ran and passed, and the engine's SE block returned (it then fills `theta1_pval`, which its
failure branch leaves empty). Otherwise the result is made to say so: standard errors NaN, no
p-values, `se_method = "none"`, the reason in `se_status`. When the engine's block failed it had
kept the linear-IV θ₁ standard errors; those are moved to `res["dx"]["theta1_se_linear_iv"]`.
"""
function dx_stamp!(res::AbstractDict, stage::AbstractString)
    names = dx_theta1_names()
    K1 = length(res["theta1"])
    K1 == length(names) || error(
        "dx: θ₁ has $(K1) entries, the variant's design has $(length(names)).")
    res["param_names_theta1"] = names
    g         = DX_SE_GUARD[]
    requested = se_method()
    wanted    = requested in ("sandwich", "wcb")
    joint     = wanted && g !== nothing && g.ok && length(get(res, "theta1_pval", Float64[])) == K1
    status    = joint ? "joint" :
                !wanted ? "not requested (BLP_SE_METHOD=$(requested))" :
                (g !== nothing && !g.ok) ? "guard: $(g.reason)" :
                g === nothing ? "failed: the engine's SE block stopped before the joint routine ran" :
                                "failed: the joint routine did not return after the guard passed"
    dx = Dict{String,Any}(
        "suffix"   => DX_SUFFIX,
        "x1_extra" => [DX_COL],
        "stage"    => String(stage),
        "se_stage" => DX_SE_STAGE,
        "se_joint" => joint,
        "se_status" => status,
        "se_method_requested" => requested,
        "se_guard" => g === nothing ? Dict{String,Any}("ran" => false) :
                      Dict{String,Any}("ran" => true, "ok" => g.ok, "reason" => g.reason,
                                       "K1" => g.K1, "K2" => g.K2, "K2_kept" => g.K2_kept,
                                       "K" => g.K, "L" => g.L, "rcond" => g.rcond))
    if wanted && !joint
        (g === nothing || g.ok) &&
            (dx["theta1_se_linear_iv"] = collect(Float64, get(res, "theta1_se", Float64[])))
        res["theta1_se"]   = fill(NaN, K1)
        res["theta2_se"]   = fill(NaN, length(get(res, "theta2", Float64[])))
        res["theta1_pval"] = Float64[]
        res["theta2_pval"] = Float64[]
        res["se_method"]   = "none"
        log_status("  [DX SE] NO JOINT STANDARD ERRORS at stage $(stage): $(status). The result carries " *
                   "no standard error (se_method = none); the point estimates are unaffected.")
    end
    res["dx"] = dx
    return res
end

"""
    dx_require_previous(estim, spec_id, stage, args)

Refuse stage `stage` when an earlier stage of the same job (`DX_RUN_STAGES`) left no result. The
engine catches a failed stage and goes on to the next one; without this a later rung would start
from the seed instead of its predecessor's checkpoint and the ladder would no longer be the main
run's. Every remaining stage is refused the same way, and blp_dx_rc.jl's post-check then fails
the job naming the first missing result.
"""
function dx_require_previous(estim::Int, spec_id::Int, stage::AbstractString, args)
    stages = DX_RUN_STAGES[]
    i = findfirst(==(String(stage)), stages)
    (i === nothing || i == 1 || get(args, "dry_run", false)) && return nothing
    _, _, out_dir = get_paths(args["hpc"]; local_dir=get(args, "local_dir", nothing))
    for st in stages[1:i-1]
        isfile(dx_result_path(out_dir, estim, spec_id, st)) || error(
            "dx: stage '$(st)' of E$(estim) left no result in this job, so stage '$(stage)' is " *
            "not run after it.")
    end
    return nothing
end

function run_blp_estimation(estim::Int, spec_id::Int, args,
                            nu_draws::Matrix{Float64},
                            draws_3d::Array{Float64,3},
                            key_index::Dict{Tuple{String,String},Int})
    dx_active() && error(
        "dx: the CPU estimation path (run_blp_estimation) writes its checkpoints without the " *
        "output suffix, so a variant run would overwrite the main specification's. The variant " *
        "runs on the GPU IFT engine (blp_dx_rc.jl); its linear step runs through blp_dx_linear.jl.")
    return Base.invoke_in_world(_DX_WORLD_DESIGN, run_blp_estimation,
                                estim, spec_id, args, nu_draws, draws_3d, key_index)
end

# The GPU IFT entry point exists only in a session that loaded blp_engine_gpu.jl before this file
# (blp_dx_rc.jl does); the CPU-only linear step never reaches it.
const DX_WRAPS_IFT = isdefined(Main, :run_blp_estimation_ift_gpu)
if DX_WRAPS_IFT
    function run_blp_estimation_ift_gpu(estim::Int, spec_id::Int, args,
                                        nu_draws::Matrix{Float64},
                                        draws_3d::Array{Float64,3},
                                        key_index::Dict{Tuple{String,String},Int})
        if !dx_active()
            return Base.invoke_in_world(_DX_WORLD_DESIGN, run_blp_estimation_ift_gpu,
                                        estim, spec_id, args, nu_draws, draws_3d, key_index)
        end
        stage = String(args["stage"])
        dx_require_namespace(stage)
        dx_require_previous(estim, spec_id, stage, args)
        DX_SE_GUARD[] = nothing
        ev = nothing
        if get(args, "dry_run", false) === true
            ev = Dict{String,Any}("design_cols" => 0, "n_rows" => 0, "d_rows" => 0, "rank" => 0,
                                  "rank_cols" => 0, "returned" => false, "error" => "")
            DX_DRY[stage] = ev
            DX_DRY_TRACE[] = ev
        end
        res = try
            Base.invoke_in_world(_DX_WORLD_DESIGN, run_blp_estimation_ift_gpu,
                                 estim, spec_id, args, nu_draws, draws_3d, key_index)
        catch e
            ev === nothing || (ev["error"] = sprint(showerror, e))
            rethrow()
        finally
            DX_DRY_TRACE[] = nothing
        end
        ev === nothing || (ev["returned"] = res isa AbstractDict && get(res, "dry_run", false) === true)
        (res isa AbstractDict && !get(res, "dry_run", false)) && dx_stamp!(res, stage)
        return res
    end
end

"""
    dx_smoke_verdict(estim, stages, nest) -> Vector{String}

What a dry run of the variant (the smoke) did NOT show, as text; empty means it passed. Only
positive evidence counts, because the engine's stage loop catches a failed stage and goes on, and
skips a stage whose result is on disk:

  - `nest` (the record of `dx_nesting_check`): the variant's linear step was compared with the
    logit `full_dtype` sub-model, is nested in it, and its design has full rank;
  - for every stage in `stages`, from `DX_DRY`: the engine reached the wrapped entry point; the
    design it built there carried the D column (L_PROD + 1 characteristics, some rows digital and
    some not); its rank check returned full rank on the ten columns; the engine's dry run came
    back with its `dry_run` marker (returned after the GPU share check, the ten contraction
    iterations and the IFT forward passes) and threw nothing.
"""
function dx_smoke_verdict(estim::Int, stages::Vector{String}, nest)::Vector{String}
    miss = String[]
    K = length(dx_theta1_names())
    d = nest isa AbstractDict ? get(nest, "dx", nothing) : nothing
    if d === nothing
        push!(miss, "the nesting check left no record")
    else
        if !get(d, "logit_found", false)
            push!(miss, "nesting: the logit summary has no E$(estim)_full_dtype, so the variant's " *
                        "linear step was compared with nothing")
        elseif !get(d, "nested", false)
            push!(miss, "nesting: the variant's linear step does not reproduce logit full_dtype")
        end
        get(d, "rank", 0) == K || push!(miss, "nesting: the design has rank $(get(d, "rank", 0)), not $(K)")
    end
    for st in stages
        ev = get(DX_DRY, st, nothing)
        if ev === nothing
            push!(miss, "stage $(st): the engine never reached the variant's entry point (it skips a " *
                        "stage whose result is on disk; smoke a stage that has none)")
            continue
        end
        isempty(ev["error"]) || push!(miss, "stage $(st): the engine's dry run threw: $(ev["error"])")
        ev["design_cols"] == L_PROD + 1 ||
            push!(miss, "stage $(st): the design carried $(ev["design_cols"]) characteristics, not $(L_PROD + 1)")
        0 < ev["d_rows"] < ev["n_rows"] ||
            push!(miss, "stage $(st): the D column has $(ev["d_rows"]) ones in $(ev["n_rows"]) rows")
        (ev["rank"] == K && ev["rank_cols"] == K) ||
            push!(miss, "stage $(st): rank check gave rank $(ev["rank"]) on $(ev["rank_cols"]) columns, not $(K) on $(K)")
        (ev["returned"] || !isempty(ev["error"])) ||
            push!(miss, "stage $(st): the engine's dry run did not return its dry_run marker")
    end
    return miss
end

# ==========================================================================
# 4. The linear step at a given δ (θ₂ = 0 nest when δ is the logit δ̂)
# ==========================================================================
"""
    dx_linear_step(df, delta; with_d) -> Dict

The engine's linear step — `build_regressor_matrices`, the first-stage projection of the type-4/5
spread on H = [X₁, Z], `estimate_theta1` and the GMM criterion at W = (Z'Z/N)⁻¹ — assembled as
`run_blp_estimation_ift_gpu` assembles it, on mean utilities `delta`. `with_d` selects the variant
design (true) or the engine's own (false); the switch is restored on exit.
"""
function dx_linear_step(df::DataFrame, delta::Vector{Float64}; with_d::Bool)
    was = DX_ACTIVE[]
    DX_ACTIVE[] = with_d
    try
        N_obs = nrow(df)
        length(delta) == N_obs || error("dx: δ has $(length(delta)) entries, the panel $(N_obs) rows.")
        dep_types = Int.(coalesce.(df.deposit_type, 0))
        spread_cols, x_mat, Z_mat, iv_cols = build_regressor_matrices(df)
        iv_avail = [c for c in iv_cols
                    if std(replace(coalesce.(df[!, c], 0.0), Inf=>0.0, -Inf=>0.0)) > 1e-10]
        Z = zeros(N_obs, length(iv_avail))
        for (i, c) in enumerate(iv_avail)
            v = coalesce.(df[!, c], 0.0); replace!(v, Inf=>0.0, -Inf=>0.0)
            Z[:, i] .= v
        end
        W = try inv(Z' * Z ./ N_obs) catch; Matrix(I(length(iv_avail)) * 1.0) end
        H          = hcat(x_mat, Z_mat)
        spread_hat = project_endogenous_spreads(spread_cols, H, dep_types)
        X_full     = hcat(spread_cols, x_mat)
        X_hat      = hcat(spread_hat,  x_mat)
        valid      = BitVector(all.(isfinite, eachrow(X_hat)))
        clusters   = string.(df.CodConglomeradoPrudencial)
        pc         = build_precomp(df, Z, X_full, X_hat, valid, clusters)
        rnk        = check_design_rank(pc)
        theta1, xi = estimate_theta1(delta, pc)
        g          = compute_gmm_moments(xi, pc)
        return Dict{String,Any}(
            "with_d"      => with_d,
            "param_names" => with_d ? dx_theta1_names() : vcat(["alpha"], X_COLS),
            "theta1"      => theta1,
            "Q_value"     => dot(g, W * g),
            "rank"        => rnk,
            "n_obs"       => N_obs,
            "n_d_rows"    => Int(sum(pc.d_mask)),
            "n_iv"        => length(iv_avail))
    finally
        DX_ACTIVE[] = was
    end
end

"""The logit δ̂ of routine `estim`: the warm start the engine loads, resolved in its order
(the upload folder first, then the logit step folder)."""
function dx_logit_delta(estim::Int, spec_id::Int, in_dir, out_dir)
    fname = "logit_delta_E$(estim)_spec_$(spec_id).bin"
    for cand in (joinpath(in_dir, fname), joinpath(logit_dir(out_dir), fname))
        isfile(cand) || continue
        d = load_delta_bin(cand)
        d === nothing || return d, cand
    end
    return nothing, joinpath(logit_dir(out_dir), fname)
end

"""θ₁ of logit sub-model `sub` for routine `estim` from the logit step's combined summary, or
`nothing` when the summary or the entry is absent."""
function dx_logit_theta1(estim::Int, spec_id::Int, out_dir, sub::AbstractString)
    path = joinpath(logit_dir(out_dir), "logit_summary_spec_$(spec_id).json")
    isfile(path) || return nothing
    data  = JSON3.read(read(path, String))
    entry = get(data, Symbol("E$(estim)_$(sub)"), nothing)
    entry === nothing && return nothing
    return (theta1 = Float64.(collect(entry[:theta1])),
            names  = String.(collect(entry[:param_names])),
            Q      = Float64(entry[:Q_value]))
end

"""
    dx_nesting_check(estim, spec_id, args; tol=5e-7) -> Dict

The θ₂ = 0 nesting check of both specifications on the logit δ̂: the engine's linear step
against the logit sub-model it nests (`full` for the main design, `full_dtype` for the variant).
The summary stores eight significant digits, so `tol` is a relative 5e-7.

Hard errors: no demand parquet or no logit δ̂ for the routine; a sub-model entry whose parameter
names are not the engine design's, for EITHER design (the summary is then not the one these
parquets produced); the variant not reproducing `full_dtype` within `tol`. Reported, not errors:
a summary that is absent or lacks an entry (the linear step is then recorded with nothing to
compare with), and a main design that misses `tol` (flagged in the log line and in the record).
"""
function dx_nesting_check(estim::Int, spec_id::Int, args; tol::Float64=5e-7)
    in_dir, _, out_dir = get_paths(args["hpc"]; local_dir=get(args, "local_dir", nothing))
    path = demand_parquet_path(estim, spec_id, args)
    path === nothing && error("dx: no demand parquet for E$(estim) spec $(spec_id).")
    df = DataFrame(Parquet2.Dataset(path); copycols=true)
    delta, dpath = dx_logit_delta(estim, spec_id, in_dir, out_dir)
    delta === nothing && error("dx: the logit δ̂ $(dpath) is missing; run the logit step first.")
    out = Dict{String,Any}("estim" => estim, "spec_id" => spec_id,
                           "parquet" => basename(path), "delta" => basename(dpath))
    for (key, with_d, sub) in (("main", false, "full"), ("dx", true, "full_dtype"))
        r   = dx_linear_step(df, delta; with_d=with_d)
        ref = dx_logit_theta1(estim, spec_id, out_dir, sub)
        if ref === nothing
            r["logit_sub_model"] = sub; r["logit_found"] = false
            log_status("  [DX NEST] E$(estim) $(key): no E$(estim)_$(sub) in the logit summary — not compared")
        else
            ref.names == r["param_names"] || error(
                "dx: E$(estim)_$(sub) has parameters $(ref.names), the engine design $(r["param_names"]).")
            rel = maximum(abs.(r["theta1"] .- ref.theta1) ./ max.(abs.(ref.theta1), 1e-12))
            r["logit_sub_model"] = sub; r["logit_found"] = true
            r["logit_theta1"] = ref.theta1; r["logit_Q"] = ref.Q
            r["max_rel_diff"] = rel; r["nested"] = rel <= tol
            log_status("  [DX NEST] E$(estim) $(key) vs logit $(sub): alpha $(round(r["theta1"][1], digits=6)) " *
                       "| max rel. diff $(round(rel, sigdigits=3)) | Q $(round(r["Q_value"], digits=6)) " *
                       "(logit $(round(ref.Q, digits=6)))" * (rel <= tol ? "" : "  ** NOT NESTED **"))
        end
        out[key] = r
    end
    d = out["dx"]
    if get(d, "logit_found", false) && !d["nested"]
        error("dx: at θ₂ = 0 the variant's linear step does not reproduce the logit full_dtype " *
              "coefficients for E$(estim) (max relative difference $(d["max_rel_diff"])).")
    end
    return out
end

"""
    dx_same_ses(se0, pv0, se1, pv1; rtol=DX_SELFTEST_RTOL) -> (same, maxrel)

Whether two evaluations of the SE routine agree: the same entries finite, every finite standard
error within a relative `rtol`, every finite p-value within an absolute `rtol`. `maxrel` is the
largest relative difference between the standard errors (Inf when the finite entries differ).
"""
function dx_same_ses(se0::Vector{Float64}, pv0::Vector{Float64}, se1::Vector{Float64},
                     pv1::Vector{Float64}; rtol::Float64=DX_SELFTEST_RTOL)
    (length(se0) == length(se1) && length(pv0) == length(pv1)) || return false, Inf
    fs = isfinite.(se0); fp = isfinite.(pv0)
    (fs == isfinite.(se1) && fp == isfinite.(pv1)) || return false, Inf
    maxrel = any(fs) ? maximum(abs.(se1[fs] .- se0[fs]) ./ max.(abs.(se0[fs]), floatmin())) : 0.0
    maxp   = any(fp) ? maximum(abs.(pv1[fp] .- pv0[fp])) : 0.0
    return (maxrel <= rtol && maxp <= rtol), maxrel
end

"""
    dx_selftest_se()

A check of a few milliseconds that the `gmm_cluster_ses` wrapper works in THIS Julia: on
synthetic moments it must return the engine's own standard errors when the guard passes (14
parameters on 16 moments) and NaN when it trips (17 on 16). Errors otherwise. blp_dx_rc.jl runs it
before the ladder, so a wrapper that does not work on the cluster's Julia stops the job in its
first seconds instead of at the end of the first stage.

"The engine's own" is tested to a relative `DX_SELFTEST_RTOL` = 1e-8 on every standard error (an
absolute 1e-8 on the p-values), with the same entries finite, not bit for bit: the two calls are
two evaluations of the same linear algebra, and a threaded BLAS need not return identical bits
for identical inputs. A wrapper that changed an argument would miss by many orders more.

The engine's messages during the two calls (its "profiling out" line for the synthetic on-bound
σ) are suppressed: they describe these synthetic moments, not the fit.
"""
function dx_selftest_se()
    rng = MersenneTwister(20261001)
    N = 600; L = 16
    Z = randn(rng, N, L); W = inv(Z' * Z ./ N); cl = string.(rand(rng, 1:40, N))
    X = randn(rng, N, 10) .+ Z[:, 1:10]
    Dd = randn(rng, N, 5) .+ Z[:, 11:15]
    xi = randn(rng, N)
    th = vcat(randn(rng, 10), [0.0, 1.0, -0.3, -0.2, -0.3])
    was = DX_ACTIVE[]; was_g = DX_SE_GUARD[]; was_q = DX_QUIET[]
    maxrel = 0.0
    try
        DX_QUIET[] = true
        Base.CoreLogging.with_logger(Base.CoreLogging.NullLogger()) do
            DX_ACTIVE[] = false
            se0, pv0 = gmm_cluster_ses("sandwich", th, Z, X, Dd, xi, W, cl; n_sigma=1)
            DX_ACTIVE[] = true
            se1, pv1 = gmm_cluster_ses("sandwich", th, Z, X, Dd, xi, W, cl; n_sigma=1)
            same, maxrel = dx_same_ses(se0, pv0, se1, pv1)
            (same && count(isfinite, se1) == 14) ||
                error("dx self-test: the wrapped SE routine does not return the engine's standard " *
                      "errors (largest relative difference $(maxrel), tolerance $(DX_SELFTEST_RTOL); " *
                      "$(count(isfinite, se1)) finite entries, 14 expected).")
            Dd7 = hcat(Dd, randn(rng, N, 2))
            th7 = vcat(th[1:10], [0.4, 1.0, -0.3, -0.2, -0.3, 0.2, 0.1])
            se2, _ = gmm_cluster_ses("sandwich", th7, Z, X, Dd7, xi, W, cl; n_sigma=1)
            all(isnan, se2) || error("dx self-test: 17 parameters on 16 moments were not refused.")
        end
    finally
        DX_ACTIVE[] = was; DX_SE_GUARD[] = was_g; DX_QUIET[] = was_q
    end
    log_status("  [DX SELFTEST] SE wrapper ok on Julia $(VERSION), on synthetic moments: engine SEs " *
               "reproduced through the wrapper to a relative $(DX_SELFTEST_RTOL) (largest difference " *
               "$(round(maxrel, sigdigits=2)); 14 parameters on 16 moments), 17 on 16 refused")
    return true
end

"A JSON-ready copy: a non-finite number (a NaN standard error, an rcond that was not computed) becomes null."
dx_jsonable(x::AbstractFloat) = isfinite(x) ? x : nothing
dx_jsonable(x::AbstractDict)  = Dict{String,Any}(String(k) => dx_jsonable(v) for (k, v) in x)
dx_jsonable(x::Union{AbstractVector,Tuple}) = Any[dx_jsonable(v) for v in x]
dx_jsonable(x) = x

"Write `obj` as JSON through a temporary file and a rename."
function dx_write_json(path::AbstractString, obj)
    tmp = path * ".tmp"
    open(tmp, "w") do f; JSON3.write(f, dx_jsonable(obj)); end
    mv(tmp, path; force=true)
    return path
end
