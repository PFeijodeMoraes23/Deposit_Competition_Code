"""
blp_dx_linear.jl
================
The linear step of the `dx` demand variant (blp_dx.jl) at θ₂ = 0, on the CPU and without draws.

For each routine it loads the demand parquet and the logit δ̂ the RC ladder starts from, runs the
engine's linear step under the main design (X₁ = [spread, X_COLS]) and under the variant's
(X₁ = [spread, X_COLS, dummy_D_type]), and compares each with the logit sub-model it nests:
`full` for the main design, `full_dtype` for the variant. The record is written to
`blp_dir(out_dir)/blp_linear_E{k}_spec_12_dx.json`; `--no-write` prints only.

A few seconds and under 1 GB per routine. It reads the parquet, the logit δ̂ and the logit
summary; it estimates nothing by simulation and writes no result the RC ladder or the
counterfactuals read.

Usage
-----
  julia --project=. blp_dx_linear.jl --estim 3              # local tree
  julia --project=. blp_dx_linear.jl --estim 3 --no-write
  julia --project=. blp_dx_linear.jl --estim 4 --hpc        # cluster tree
"""

isdefined(Main, :DX_SUFFIX) || include(joinpath(@__DIR__, "blp_dx.jl"))

function dx_linear_main(argv::Vector{String}=ARGS)
    a = copy(argv)
    ei = findfirst(==("--estim"), a)
    (ei === nothing || ei == length(a)) && error("Provide --estim N (plus --hpc, --local-dir DIR, --no-write).")
    estim = parse(Int, a[ei + 1])
    li = findfirst(==("--local-dir"), a)
    args = Dict{String,Any}("hpc" => "--hpc" in a,
                            "local_dir" => (li === nothing || li == length(a)) ? nothing : a[li + 1])
    spec_id = SPEC12_ID
    log_status("dx linear step — E$(estim) spec $(spec_id) | hpc=$(args["hpc"])")
    rec = dx_nesting_check(estim, spec_id, args)
    for key in ("main", "dx")
        r = rec[key]
        println("  ", rpad(key, 5), " Q = ", round(r["Q_value"], digits=6), "   rank ", r["rank"],
                "   N = ", r["n_obs"], " (D rows ", r["n_d_rows"], ")")
        for (nm, v) in zip(r["param_names"], r["theta1"])
            println("      ", rpad(nm, 24), lpad(string(round(v, digits=6)), 14))
        end
    end
    if !("--no-write" in a)
        _, _, out_dir = get_paths(args["hpc"]; local_dir=args["local_dir"])
        dest = joinpath(blp_dir(out_dir), dx_linear_name(estim, spec_id))
        mkpath(dirname(dest))
        dx_write_json(dest, rec)
        log_status("  wrote $(dest)")
    end
    return rec
end

if abspath(PROGRAM_FILE) == @__FILE__
    dx_linear_main()
end
