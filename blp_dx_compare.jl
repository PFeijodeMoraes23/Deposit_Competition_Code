"""
blp_dx_compare.jl
=================
Main specification against the `dx` demand variant (blp_dx.jl), from the RC results on disk.

For each routine and stage it reads `blp_results_E{k}_spec_12_{stage}.jls` (main) and
`blp_results_E{k}_spec_12_{stage}_dx.jls` (variant) in one folder and prints, side by side:
α and its SE, the D-type coefficient, Q, every θ₂ entry, and a summary of the difference between
the two mean-utility vectors δ̂ (mean, mean absolute, RMS, quantiles of the absolute difference,
maximum, correlation). The BBL cost estimation reads a demand fit only through δ̂, α and θ₂, so
this is the whole of what the variant changes for it. The same numbers are written to
`blp_compare_E{k}_spec_12_dx.json` in that folder (`--no-write` prints only).

Standard library only (Serialization, Statistics, Printf): it starts in seconds on any node and
loads no package, so the hand-off job of dx_suite_20261001.sh can run it on a CPU node without a
sysimage. Exit status 1 when a file it needs is missing or the two fits do not share the θ₂
structure or the number of rows.

Usage
-----
  julia blp_dx_compare.jl --dir ../data/output/blp --routines "3 4" --stages "ext1 extended"
"""

using Serialization, Statistics, Printf

const CMP_SUFFIX = "_dx"
const CMP_SPEC   = 12
const CMP_X = ["spread", "fgc_covered", "has_ip", "seg_S2", "seg_S3", "seg_S4", "seg_S5",
               "log_total_assets_lag", "is_state_owned"]
const CMP_D = ["gdp_per_capita", "fraction_65plus", "fraction_young", "pix_users_pf_per1000",
               "connections_per100", "frac_4g5g", "branches_per1000", "cadunico_families_per1000"]

_nm(v, i) = 1 <= i <= length(v) ? v[i] : "#$i"
theta2_labels(sig, pis) = vcat(["sigma($(_nm(CMP_X, s)))" for s in sig],
                               ["pi($(_nm(CMP_X, c)) x $(_nm(CMP_D, d)))" for (c, d) in pis])

_jnum(x::Real) = isfinite(x) ? @sprintf("%.10g", x) : "null"
_jnum(::Nothing) = "null"
_jstr(s) = "\"" * replace(String(s), "\\" => "\\\\", "\"" => "\\\"") * "\""
_jvec(v) = "[" * join(_jnum.(v), ", ") * "]"

function _arg(a::Vector{String}, flag::String, default::String)
    i = findfirst(==(flag), a)
    return (i === nothing || i == length(a)) ? default : a[i + 1]
end

function compare_stage(dir::String, k::Int, stage::String)
    pm = joinpath(dir, "blp_results_E$(k)_spec_$(CMP_SPEC)_$(stage).jls")
    pd = joinpath(dir, "blp_results_E$(k)_spec_$(CMP_SPEC)_$(stage)$(CMP_SUFFIX).jls")
    for p in (pm, pd)
        isfile(p) || error("missing $(p)")
    end
    m = deserialize(pm); d = deserialize(pd)
    sig = collect(m["sigma_indices"]); pis = [Tuple(p) for p in m["pi_interactions"]]
    (collect(d["sigma_indices"]) == sig && [Tuple(p) for p in d["pi_interactions"]] == pis) ||
        error("E$(k) $(stage): the two fits do not share the theta2 structure")
    dm = Float64.(m["delta"]); dd = Float64.(d["delta"])
    length(dm) == length(dd) || error("E$(k) $(stage): delta has $(length(dm)) rows in the main fit, $(length(dd)) in the variant")
    t1m = Float64.(m["theta1"]); t1d = Float64.(d["theta1"])
    length(t1d) == length(t1m) + 1 || error("E$(k) $(stage): the variant has $(length(t1d)) theta1, the main fit $(length(t1m)) (expected one more)")
    sem = Float64.(get(m, "theta1_se", fill(NaN, length(t1m)))); sed = Float64.(get(d, "theta1_se", fill(NaN, length(t1d))))
    t2m = Float64.(m["theta2"]); t2d = Float64.(d["theta2"])
    diff = dd .- dm; ad = abs.(diff)
    q(p) = quantile(ad, p)
    return (k=k, stage=stage, labels=theta2_labels(sig, pis),
            alpha_m=t1m[1], alpha_d=t1d[1], alpha_se_m=sem[1], alpha_se_d=sed[1],
            d_coef=t1d[end], d_se=sed[end], Q_m=Float64(m["Q_value"]), Q_d=Float64(d["Q_value"]),
            conv_m=Bool(get(m, "converged", false)), conv_d=Bool(get(d, "converged", false)),
            t1m=t1m, t1d=t1d, t2m=t2m, t2d=t2d,
            n=length(diff), dmean=mean(diff), dmae=mean(ad), drms=sqrt(mean(abs2, diff)),
            dq50=q(0.5), dq90=q(0.9), dq99=q(0.99), dmax=maximum(ad), dcorr=cor(dm, dd),
            guard=get(get(d, "dx", Dict()), "se_guard", nothing),
            se_status=get(get(d, "dx", Dict()), "se_status", nothing), files=(basename(pm), basename(pd)))
end

function print_stage(r)
    println("-- E$(r.k), stage $(r.stage):  $(r.files[1])  vs  $(r.files[2])")
    @printf("   %-46s %12s %12s %12s\n", "", "main", "dx", "dx - main")
    @printf("   %-46s %12.4f %12.4f %12.4f\n", "alpha", r.alpha_m, r.alpha_d, r.alpha_d - r.alpha_m)
    @printf("   %-46s %12.4f %12.4f\n", "  SE", r.alpha_se_m, r.alpha_se_d)
    @printf("   %-46s %12s %12.4f   (SE %.4f)\n", "dummy_D_type", "", r.d_coef, r.d_se)
    @printf("   %-46s %12.5f %12.5f %12.5f\n", "Q", r.Q_m, r.Q_d, r.Q_d - r.Q_m)
    @printf("   %-46s %12s %12s\n", "converged", string(r.conv_m), string(r.conv_d))
    for (j, lab) in enumerate(r.labels)
        @printf("   %-46s %12.4f %12.4f %12.4f\n", lab, r.t2m[j], r.t2d[j], r.t2d[j] - r.t2m[j])
    end
    @printf("   delta_dx - delta_main over %d rows: mean %.4f | mean abs %.4f | RMS %.4f | corr %.6f\n",
            r.n, r.dmean, r.dmae, r.drms, r.dcorr)
    @printf("   |delta_dx - delta_main|: median %.4f | p90 %.4f | p99 %.4f | max %.4f\n",
            r.dq50, r.dq90, r.dq99, r.dmax)
    if r.se_status !== nothing
        println("   variant standard errors at this stage: ", r.se_status == "joint" ? "joint GMM" : "NONE ($(r.se_status))")
    end
    if r.guard !== nothing && get(r.guard, "ran", false)
        println("   variant SE guard at this stage: ", r.guard["ok"] ? "ok (K=$(r.guard["K"]) on L=$(r.guard["L"]))" :
                "TRIPPED ($(r.guard["reason"])) - no SE at this stage")
    end
end

function stage_json(r)
    return "{" * join([
        "\"stage\": $(_jstr(r.stage))",
        "\"main_file\": $(_jstr(r.files[1]))", "\"dx_file\": $(_jstr(r.files[2]))",
        "\"alpha_main\": $(_jnum(r.alpha_m))", "\"alpha_dx\": $(_jnum(r.alpha_d))",
        "\"alpha_se_main\": $(_jnum(r.alpha_se_m))", "\"alpha_se_dx\": $(_jnum(r.alpha_se_d))",
        "\"dummy_D_type\": $(_jnum(r.d_coef))", "\"dummy_D_type_se\": $(_jnum(r.d_se))",
        "\"Q_main\": $(_jnum(r.Q_m))", "\"Q_dx\": $(_jnum(r.Q_d))",
        "\"converged_main\": $(r.conv_m)", "\"converged_dx\": $(r.conv_d)",
        "\"theta2_labels\": [" * join(_jstr.(r.labels), ", ") * "]",
        "\"theta2_main\": $(_jvec(r.t2m))", "\"theta2_dx\": $(_jvec(r.t2d))",
        "\"theta2_max_abs_diff\": $(_jnum(isempty(r.t2m) ? 0.0 : maximum(abs.(r.t2d .- r.t2m))))",
        "\"delta_n\": $(r.n)", "\"delta_mean_diff\": $(_jnum(r.dmean))",
        "\"delta_mean_abs_diff\": $(_jnum(r.dmae))", "\"delta_rms_diff\": $(_jnum(r.drms))",
        "\"delta_abs_diff_p50\": $(_jnum(r.dq50))", "\"delta_abs_diff_p90\": $(_jnum(r.dq90))",
        "\"delta_abs_diff_p99\": $(_jnum(r.dq99))", "\"delta_max_abs_diff\": $(_jnum(r.dmax))",
        "\"delta_corr\": $(_jnum(r.dcorr))"], ", ") * "}"
end

function compare_main(argv::Vector{String}=ARGS)
    a = copy(argv)
    dir      = _arg(a, "--dir", "")
    isempty(dir) && error("Provide --dir <folder holding the blp_results_*.jls files>.")
    routines = parse.(Int, split(_arg(a, "--routines", "3 4")))
    stages   = String.(split(_arg(a, "--stages", "ext1 extended")))
    write_it = !("--no-write" in a)
    println("MAIN vs DX (demand variant with the D-type dummy in the linear part): $(dir)")
    bad = 0
    for k in routines
        rows = String[]
        for st in stages
            r = try
                compare_stage(dir, k, st)
            catch e
                println("-- E$(k), stage $(st): NOT COMPARED: ", sprint(showerror, e)); bad += 1; nothing
            end
            r === nothing && continue
            print_stage(r)
            push!(rows, stage_json(r))
        end
        if write_it && !isempty(rows)
            dest = joinpath(dir, "blp_compare_E$(k)_spec_$(CMP_SPEC)$(CMP_SUFFIX).json")
            tmp = dest * ".tmp"
            open(tmp, "w") do f
                write(f, "{\"estim\": $(k), \"spec_id\": $(CMP_SPEC), \"suffix\": $(_jstr(CMP_SUFFIX)), \"stages\": [" *
                         join(rows, ", ") * "]}\n")
            end
            mv(tmp, dest; force=true)
            println("   wrote $(basename(dest))")
        end
    end
    return bad
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(compare_main() == 0 ? 0 : 1)
end
