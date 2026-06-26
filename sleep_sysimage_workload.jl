# sleep_sysimage_workload.jl — exercise the sieve engine's hot path on a tiny
# synthetic so PackageCompiler bakes the type-specialized methods into the sysimage.
include(joinpath(@__DIR__, "sleep_joint_sieve.jl"))
let
    N, d, nE, nT = 600, 5, 50, 12
    base = collect(1:N) ./ N
    Sn = hcat([sin.(base .* (j + 1.0)) for j in 1:d]...)          # deterministic, no RNG
    Z  = 0.5 .+ abs.(cos.(base .* 3.0))
    y  = sin.(base .* 4.0); cf = cos.(base .* 2.0)
    ec = Int32.(mod.(0:N-1, nE)); tc = Int32.(mod.(0:N-1, nT))
    ecn = counts_of(ec, nE); tcn = counts_of(tc, nT)
    th0 = fill(1.0 / sqrt(d), d)
    for (tw, ntw, tcc) in ((nothing, 0, zeros(1)), (tc, nT, tcn))      # one-way and two-way
        for ls in (:robust, :ls)
            for nb in (0, 400)                                          # exact and binned ramp
                prob = Problem(Sn, Z, y, cf, ec, nE, ecn, tw, ntw, tcc, 5, 3, ls, 5 + 3 + 1, d, nb)
                run_estimator(prob, [th0, fill(1.0, d) ./ sqrt(d)], 15)
            end
        end
    end
    @info "sleep sysimage workload done"
end
