# blp_smoke.jl — the stage C GPU smoke in ONE Julia process.
#
# A thin driver over blp_rc.jl, NOT an engine: for each routine it makes exactly the call
# `julia blp_rc.jl --estim <k> <passthrough>` makes (`run_routine(k; passthrough)`, blp_rc.jl), so
# the engine sees the same arguments the four separate smoke processes gave it. What changes is
# only the process: Julia, the sysimage and the engine include are paid once, not four times.
#
#   julia --project=. [--sysimage <img>] --threads=<n> blp_smoke.jl --routines "1 2 3 4" \
#         --stage sigma,ext1,extended --hpc --R 2000 --seed 42 --dry-run
#   julia blp_smoke.jl --routines "1 2 3 4" --print-plan <passthrough>   # prints the calls, runs nothing
#
# Per routine the log carries
#   === SMOKE E<k> BEGIN ===  ...the engine's own output...  === SMOKE E<k> END (ok|ERROR) ===
# so stagec_suite_20261012.sh --gpu cuts one log into the per-routine segments its verdict reads.
# A routine that throws is reported (`=== SMOKE E<k> ERROR: ...`) and the loop goes on to the next,
# as the four separate processes did; the exit code is 1 if any routine threw.
# Between routines the previous routine's arrays are released (`GC.gc(true)`, then
# `CUDA.reclaim()` when CUDA is loaded), so the host and device peak is one routine's, not the sum.

args = copy(ARGS)
ri = findfirst(==("--routines"), args)
(ri === nothing || ri == length(args)) && error("blp_smoke.jl: give --routines \"1 2 3 4\"")
routines = parse.(Int, split(args[ri + 1]))
deleteat!(args, ri:ri + 1)
plan_only = "--print-plan" in args
filter!(!=("--print-plan"), args)
any(a -> a in ("--estim", "--all", "--all-six", "--all-routines"), args) &&
    error("blp_smoke.jl chooses the routines itself; drop --estim/--all* from the arguments")
passthrough = String.(args)

if plan_only
    for k in routines
        println("E", k, ": run_routine(", k, "; passthrough=", repr(passthrough), ")")
    end
    exit(0)
end

include(joinpath(@__DIR__, "blp_rc.jl"))   # defines run_routine; its CLI entry runs only as PROGRAM_FILE

failed = Int[]
for k in routines
    println("\n=== SMOKE E", k, " BEGIN ===")
    status = "ok"
    try
        run_routine(k; passthrough = copy(passthrough))
    catch e
        status = "ERROR"
        push!(failed, k)
        println("=== SMOKE E", k, " ERROR: ", sprint(showerror, e), " ===")
    end
    GC.gc(true)
    if isdefined(Main, :CUDA)
        try
            Main.CUDA.reclaim()
        catch
        end
    end
    println("=== SMOKE E", k, " END (", status, ") ===")
    flush(stdout)
end
isempty(failed) || println("blp_smoke.jl: routines that threw: ", join(failed, ", "))
exit(isempty(failed) ? 0 : 1)
