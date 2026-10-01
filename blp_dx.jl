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
# and wraps four engine functions. Each wrapper calls the engine's own method through
# `Base.invoke_in_world` at the world age recorded before the wrapper was defined, so the engine
# code that runs is the engine's, and with the switch `DX_ACTIVE[]` off every wrapper returns the
# engine's result untouched:
#
#   build_regressor_matrices     appends the D column to x_mat (the only change to the model)
#   check_design_rank            names the tenth column in its messages
#   gmm_cluster_ses              the SE guard (below)
#   run_blp_estimation_ift_gpu   records the ten θ₁ names and the guard's verdict in the result
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
# rule and inverts D'WD itself first: when the count exceeds L, or the inverse does not exist, it
# returns NaN for every SE and p-value, logs the reason, and records it in the result. The SEs the
# variant reports are those of stage `DX_SE_STAGE` (ext1: 14 or 15 parameters on 16 moments);
# blp_dx_rc.jl ends with a nonzero status when the guard trips at that stage.

isdefined(Main, :DX_SUFFIX) && error("blp_dx.jl is already loaded in this session; include it once.")
isdefined(Main, :X_COLS) || Base.include(Main, joinpath(@__DIR__, "blp_engine_cpu.jl"))

const DX_SUFFIX   = "_dx"
const DX_COL      = "dummy_D_type"
const DX_SE_STAGE = "ext1"
const DX_ACTIVE   = Ref(false)
const DX_SE_GUARD = Ref{Any}(nothing)     # verdict of the last gmm_cluster_ses call under the variant

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
record and the main-vs-variant comparison. All in `blp_dir(out_dir)`.
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
    return spread_cols, hcat(x_mat, dx_column(df)), Z_mat, iv_cols
end

function check_design_rank(pc::Precomp;
                           colnames::Vector{String}=vcat(["spread_hat"], X_COLS,
                                                         dx_active() ? [DX_COL] : String[]),
                           abort::Bool=true)::Int
    return Base.invoke_in_world(_DX_WORLD_ENGINE, check_design_rank, pc;
                                colnames=colnames, abort=abort)::Int
end

# ==========================================================================
# 2. The SE guard
# ==========================================================================
"""
    dx_se_guard(theta, Z, X, Ddelta, W; n_sigma, bound_tol, jac_tol) -> NamedTuple

Whether `gmm_cluster_ses` can produce standard errors for these inputs without a pseudo-inverse.
Counts the parameters the routine keeps after its own profiling (an on-bound σ, or a θ₂ column
whose moment Jacobian is negligible, is dropped) and forms the same D'WD:

  ok = false, reason "K > L"      when the kept count exceeds the L moments
  ok = false, reason "singular"   when D'WD has no inverse
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
    Kk > L && return merge(base, (ok=false, reason="K > L: $(Kk) estimated parameters on $(L) moments"))
    D    = hcat(-ZtX, ZtDd[:, keep])
    DtWD = (D' * W) * D
    invertible = try inv(DtWD); true catch; false end
    invertible || return merge(base, (ok=false, reason="singular: D'WD ($(Kk)×$(Kk)) has no inverse"))
    return merge(base, (ok=true, reason=""))
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
            log_status("  [DX SE GUARD] NO STANDARD ERRORS at this stage: $(g.reason) " *
                       "(K1=$(g.K1), K2 kept=$(g.K2_kept) of $(g.K2), L=$(g.L)). Every SE and " *
                       "p-value is NaN; the point estimates are unaffected.")
            K = length(theta)
            return fill(NaN, K), fill(NaN, K)
        end
        log_status("  [DX SE GUARD] ok: $(g.K) estimated parameters on $(g.L) moments " *
                   "(K1=$(g.K1), K2 kept=$(g.K2_kept) of $(g.K2))")
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

"Stamp a variant result: the ten θ₁ names and the guard's verdict."
function dx_stamp!(res::AbstractDict, stage::AbstractString)
    names = dx_theta1_names()
    length(res["theta1"]) == length(names) || error(
        "dx: θ₁ has $(length(res["theta1"])) entries, the variant's design has $(length(names)).")
    res["param_names_theta1"] = names
    g = DX_SE_GUARD[]
    res["dx"] = Dict{String,Any}(
        "suffix"   => DX_SUFFIX,
        "x1_extra" => [DX_COL],
        "stage"    => String(stage),
        "se_stage" => DX_SE_STAGE,
        "se_guard" => g === nothing ? Dict{String,Any}("ran" => false) :
                      Dict{String,Any}("ran" => true, "ok" => g.ok, "reason" => g.reason,
                                       "K1" => g.K1, "K2" => g.K2, "K2_kept" => g.K2_kept,
                                       "K" => g.K, "L" => g.L))
    return res
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
        DX_SE_GUARD[] = nothing
        res = Base.invoke_in_world(_DX_WORLD_DESIGN, run_blp_estimation_ift_gpu,
                                   estim, spec_id, args, nu_draws, draws_3d, key_index)
        (res isa AbstractDict && !get(res, "dry_run", false)) && dx_stamp!(res, stage)
        return res
    end
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
The summary stores eight significant digits, so `tol` is a relative 5e-7. Errors when the variant
does not reproduce `full_dtype`; a summary without the entry is reported, not an error.
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

"Write `obj` as JSON through a temporary file and a rename."
function dx_write_json(path::AbstractString, obj)
    tmp = path * ".tmp"
    open(tmp, "w") do f; JSON3.write(f, obj); end
    mv(tmp, path; force=true)
    return path
end
