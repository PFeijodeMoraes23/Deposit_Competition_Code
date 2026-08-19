# build_sleep_sysimage.jl
# ========================
# Build `sleep_sysimage.so` with Optim baked in, so each `sleep_joint_sieve.jl`
# launch starts in seconds instead of recompiling Optim (~30-60 s per launch, the
# dominant per-fit startup cost across the sieve fits).
#
# sleep_joint_julia.py picks it up automatically (--sysimage sleep_sysimage.so) when present.
#
# Usage:  julia --project=. build_sleep_sysimage.jl     (~10-20 min, one-time)
using Pkg
Pkg.activate(@__DIR__)
if !haskey(Pkg.project().dependencies, "PackageCompiler")
    Pkg.add("PackageCompiler")
end
using PackageCompiler

direct = keys(Pkg.project().dependencies)
pkgs = Symbol.(filter(in(direct), ["Optim"]))     # LinearAlgebra/Statistics/Printf are stdlib (baked by default)
@info "Baking packages into sleep sysimage" pkgs

sysimg   = joinpath(@__DIR__, "sleep_sysimage.so")
workload = joinpath(@__DIR__, "sleep_sysimage_workload.jl")
kw = Dict{Symbol,Any}(:sysimage_path => sysimg)
if isfile(workload)
    @info "Including precompile workload (bakes the sieve hot path)" workload
    kw[:precompile_execution_file] = workload
end

@info "Building sleep sysimage — this takes ~10-20 min..."
create_sysimage(pkgs; kw...)
@info "Done." sysimage = sysimg
