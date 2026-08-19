#!/usr/bin/env julia
# export_rc_delta.jl — dump the RC-BLP mean utility δ(θ̂₂) from a stage result into a plain binary,
# so the Python weak-IV battery can invert AR/LM against the STRUCTURAL δ instead of the log-share
# (θ₂ = 0) δ it builds itself.
#
# Why this exists: weak_iv_analysis.py constructs δ = ln(s_data) — the logit moment. In the RC model δ
# is θ₂-dependent, so the AR/LM/Hansen-J sets it reports characterise the logit moment, not the rung
# actually reported in the paper. (The FIRST-STAGE statistics — eff-F, KP-F, Cragg-Donald, partial R² —
# are unaffected: they are computed from the spread and Z alone and never touch δ.)
#
# Row alignment: the engine reads the demand parquet in file order and never reorders, so δ[i]
# corresponds to parquet row i. Verified 2026-08-12: length(delta) == parquet nrows on every
# routine exported.
#
# Format (matches the engine's save_delta_bin): Int64 length, then that many Float64, little-endian.
#
# Usage:
#   julia --project=. export_rc_delta.jl [--stage ext1] [--routines 3,4] [--outdir DIR]

using Serialization

include(joinpath(@__DIR__, "of_root.jl"))

function get_paths()
    data = joinpath(resolve_of_root(), "BCB", "Egan_et_al_2025_Rep", "processed",
                    "ESTIMATION_OUTPUT", "BLP_RESULTS")
    return joinpath(data, "cluster_raw"), joinpath(data, "cluster_processed")
end

function argval(flag, default)
    i = findfirst(==(flag), ARGS)
    return i === nothing || i == length(ARGS) ? default : ARGS[i + 1]
end

function main()
    stage    = argval("--stage", "ext1")
    routines = parse.(Int, split(argval("--routines", "3,4"), ","))
    raw, processed = get_paths()
    outdir = argval("--outdir", processed)
    mkpath(outdir)

    println("RC δ export | stage=$stage | routines=$(join(routines, ","))")
    for k in routines
        src = joinpath(raw, "blp_results_E$(k)_spec_12_$(stage).jls")
        if !isfile(src)
            println("  E$k: MISSING $(basename(src)) — skipped"); continue
        end
        d = deserialize(src)
        if !haskey(d, "delta")
            println("  E$k: no `delta` key in $(basename(src)) — skipped"); continue
        end
        delta = Vector{Float64}(d["delta"])
        dst = joinpath(outdir, "rc_delta_E$(k)_spec_12_$(stage).bin")
        open(dst, "w") do io
            write(io, Int64(length(delta)))
            write(io, delta)
        end
        # θ₁[1] is α at this rung — printed so the Python side can be sanity-checked against it.
        a = haskey(d, "theta1") && !isempty(d["theta1"]) ? d["theta1"][1] : NaN
        println("  E$k: wrote $(basename(dst))  n=$(length(delta))  α(rung)=$(round(a, digits=4))  " *
                "Q=$(round(get(d, "Q_value", NaN), digits=5))")
    end
end

main()
