# blp_build_sysimage_workload.jl
# ===============================
# OPTIONAL precompile workload for blp_build_sysimage.jl (enabled with
# BLP_SYSIMAGE_WORKLOAD=1). Runs a tiny IFT --dry-run so PackageCompiler also bakes the
# estimation HOT PATH (share kernels, GPU contraction, GMM) into the sysimage, not just
# the package loads. Best-effort: if data or a GPU is unavailable during the build, it
# catches the error and the build still proceeds (packages baked, hot path not).
#
# Uses E3 (demand_3_logistic) at R=50, stage sigma, --dry-run (times ~10 inner iters).

try
    ENV["BLP_DEMAND_PREFIX"] = "demand_3_logistic"
    empty!(ARGS)
    append!(ARGS, ["--estim", "3", "--spec", "12", "--stage", "sigma",
                   "--hpc", "--R", "50", "--seed", "42", "--dry-run"])
    include(joinpath(@__DIR__, "blp_rc.jl"))
    @info "[sysimage workload] dry-run completed — hot path traced."
catch e
    @info "[sysimage workload] dry-run skipped (data/GPU not available) — packages " *
          "still baked; this is fine." exception = (e,)
end
