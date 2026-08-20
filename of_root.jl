# of_root.jl — validated Open-Finance root resolution for the Julia entry points.
#
# Mirrors the anchor logic of utils/paths.py (OPEN_FINANCE_ROOT override, else
# three levels up from this file).  The validation exists because a June 2026 run
# resolved the root one level short — at .../Open-Finance/Code — and mkpath'd a
# full BCB/Egan_et_al_2025_Rep/processed/... tree there, quietly accumulating
# 8.9 GB of draws nobody read.  A wrong root must fail loudly, not create dirs.

"""
    resolve_of_root() -> String

The Open-Finance data root, validated by basename and by sentinel directories.
Throws rather than returning a root that would be silently wrong.
"""
function resolve_of_root()::String
    root = get(ENV, "OPEN_FINANCE_ROOT",
               dirname(dirname(dirname(abspath(@__FILE__)))))

    basename(root) == "Open-Finance" || error(
        "Refusing to run: resolved root '$root' has basename " *
        "'$(basename(root))', expected 'Open-Finance'. " *
        "Set OPEN_FINANCE_ROOT to override.")

    (isdir(joinpath(root, "shared")) &&
     isdir(joinpath(root, "BCB", "Egan_et_al_2025_Rep", "processed"))) || error(
        "Refusing to run: '$root' lacks the sentinel dirs shared/ and " *
        "BCB/Egan_et_al_2025_Rep/processed/. Not creating them — a missing " *
        "sentinel means the root is wrong, not that the tree needs building.")

    return root
end

"""
    drafts_dir() -> String

The paper directory the exporters mirror their .tex fragments and figures into, mirroring
`drafts_dir()` in utils/paths.py so the Julia and Python exporters write to one place.

Returns `""` when the root cannot be resolved. The cluster has no Drafts tree, and every
caller already guards the copy with `isdir`, so an unresolvable root means "no mirror to
write" — the local results directory still gets the fragment — rather than an error while
the constants are being defined.
"""
function drafts_dir()::String
    try
        return joinpath(resolve_of_root(), "Drafts", "Deposit Competition")
    catch
        return ""
    end
end

# ==========================================================================
# Cluster input search order (data/input before data/output)
# ==========================================================================
# On the cluster HEAD/data/input holds everything UPLOADED and HEAD/data/output everything
# the CLUSTER produced.  An input can now arrive by either route: the sleepiness phase writes
# its demand parquets under data/output/DEMAND_PREP (SLEEP_OUT_ROOT) and the Υ_pix export
# writes under data/output, while the same files may still be staged by hand into data/input.
# Every consumer therefore searches data/input FIRST and falls back to the data/output
# location, so a manually uploaded file always wins — the precedence blp_gpu_engine.jl
# already applies to the logit δ warm-starts.  These helpers are the single definition of
# that order; they live here because of_root.jl is the one path module every Julia entry
# point already reaches (blp_1_logit.jl directly, the BLP/CF stack through
# blp_1_estimation.jl), so sharing them adds no include edge.

"""
    search_dirs(dirs...) -> Vector{String}

The given directories as `String`s, in order, dropping `nothing`/empty entries and repeats.
De-duplication matters because the cluster and local trees collapse different candidates
onto the same directory.
"""
function search_dirs(dirs...)::Vector{String}
    out = String[]
    for d in dirs
        d === nothing && continue
        s = String(d)
        (isempty(s) || s in out) && continue
        push!(out, s)
    end
    return out
end

"""
    is_cluster_out(out_dir) -> Bool

Whether `out_dir` is the cluster's `data/output` rather than a local results directory.
Detection is by tree shape — the directory is named `output` — mirroring `_hpc_tree` in
foundation_demand_eval.jl, so one code path serves both trees and a local run never picks up
a cluster-only fallback.
"""
is_cluster_out(out_dir)::Bool =
    basename(rstrip(String(out_dir), ['/', '\\'])) == "output"

"""
    resolve_in_then_out(fname, dirs...) -> Union{String,Nothing}

The first `dir/fname` that exists, scanning `dirs` in the given order; `nothing` when no
candidate exists. Per-file resolution, so a routine uploaded to data/input and a routine
produced under data/output are both found in the same run.
"""
function resolve_in_then_out(fname::AbstractString, dirs...)
    for d in search_dirs(dirs...)
        p = joinpath(d, fname)
        isfile(p) && return p
    end
    return nothing
end

"""
    describe_search_dirs(dirs...) -> String

The searched locations, one indented line each, for hard-fail messages that name every place
a missing input was looked for.
"""
describe_search_dirs(dirs...)::String =
    join(["    " * d for d in search_dirs(dirs...)], "\n")

"""
    demand_search_dirs(in_dir, out_dir) -> Vector{String}

Search path for the demand-prep parquets: `in_dir` (data/input on the cluster, where uploads
land) before `out_dir/DEMAND_PREP` (where the on-cluster sleepiness phase writes). Off the
cluster the list is just `in_dir`, the local ESTIMATION_OUTPUT/DEMAND_PREP. Shared by
blp_1_logit.jl, blp_2_rc.jl and blp_gpu_engine.jl so the three cannot disagree about where a
routine's parquet lives.
"""
demand_search_dirs(in_dir, out_dir)::Vector{String} =
    is_cluster_out(out_dir) ?
        search_dirs(in_dir, joinpath(String(out_dir), "DEMAND_PREP")) :
        search_dirs(in_dir)
