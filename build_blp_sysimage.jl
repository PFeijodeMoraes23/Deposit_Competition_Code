# build_blp_sysimage.jl
# =====================
# Build a Julia sysimage (`blp_sysimage.so`) with the heavy BLP packages — above all
# CUDA — baked in, so each cluster job starts in seconds instead of re-running
# `Pkg.precompile()` + `using CUDA` (the dominant per-job startup cost across 27-48 jobs).
#
# Build it ON a GPU node (gpu_devel) so CUDA's GPU-specific code is included — see
# `submit_build_sysimage.sh`. Output goes next to this file; the estimation submit scripts
# pick it up automatically (`--sysimage blp_sysimage.so`) when present.
#
#   BLP_SYSIMAGE_WORKLOAD=1  also runs `sysimage_precompile_workload.jl` during the build
#   (a tiny --dry-run that bakes the estimation hot path too — needs data + a GPU present).
#
# Usage (on a GPU node):
#   julia --project=. build_blp_sysimage.jl

using Pkg
Pkg.activate(@__DIR__)
# PackageCompiler is a build-only tool; add it if absent (harmless project dep).
if !haskey(Pkg.project().dependencies, "PackageCompiler")
    Pkg.add("PackageCompiler")
end
using PackageCompiler

# Bake the heavy direct dependencies actually present in this project's Manifest.
direct = keys(Pkg.project().dependencies)
wanted = ["CUDA", "Parquet2", "DataFrames", "Optim", "LBFGSB", "QuasiMonteCarlo",
          "Distributions", "JSON3", "ArgParse", "SparseArrays"]
pkgs   = Symbol.(filter(in(direct), wanted))
@info "Baking packages into sysimage" pkgs

sysimg   = joinpath(@__DIR__, "blp_sysimage.so")
workload = joinpath(@__DIR__, "sysimage_precompile_workload.jl")
kw = Dict{Symbol,Any}(:sysimage_path => sysimg)
if get(ENV, "BLP_SYSIMAGE_WORKLOAD", "0") == "1" && isfile(workload)
    @info "Including precompile workload (bakes the estimation hot path)" workload
    kw[:precompile_execution_file] = workload
end

@info "Building sysimage — this takes ~20-40 min..."
create_sysimage(pkgs; kw...)
@info "Done." sysimage = sysimg
