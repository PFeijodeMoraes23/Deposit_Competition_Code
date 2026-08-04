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

# BLP_SYSIMAGE_CPU=1 builds the CPU-partition image: no CUDA (with CF_GPU=0 the CF/BBL stack never
# includes blp_gpu_engine.jl, so CUDA is dead weight) and a DIFFERENT output name. This exists
# because a sysimage bakes the BUILD node's CPU target: the gpu_h200 image (sapphirerapids) is
# REJECTED on `day` nodes ("Unable to find compatible target in cached code image"), which sends
# every task back to precompiling and stampedes the shared NFS depot. Build this one ON a day node.
const CPU_ONLY = get(ENV, "BLP_SYSIMAGE_CPU", "0") == "1"

# Bake the heavy direct dependencies actually present in this project's Manifest.
direct = keys(Pkg.project().dependencies)
wanted = ["CUDA", "Parquet2", "DataFrames", "Optim", "LBFGSB", "QuasiMonteCarlo",
          "Distributions", "JSON3", "ArgParse", "SparseArrays"]
CPU_ONLY && (wanted = filter(!=("CUDA"), wanted))
pkgs   = Symbol.(filter(in(direct), wanted))
@info "Baking packages into sysimage" pkgs cpu_only=CPU_ONLY

sysimg   = joinpath(@__DIR__, CPU_ONLY ? "blp_sysimage_cpu.so" : "blp_sysimage.so")
workload = joinpath(@__DIR__, "sysimage_precompile_workload.jl")
kw = Dict{Symbol,Any}(:sysimage_path => sysimg)
if get(ENV, "BLP_SYSIMAGE_WORKLOAD", "0") == "1" && isfile(workload)
    @info "Including precompile workload (bakes the estimation hot path)" workload
    kw[:precompile_execution_file] = workload
end

@info "Building sysimage — this takes ~20-40 min..."
create_sysimage(pkgs; kw...)
@info "Done." sysimage = sysimg
