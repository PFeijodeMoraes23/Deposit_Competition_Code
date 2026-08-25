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
# Cluster tree: uploads, step directories, search order
# ==========================================================================
# On the cluster HEAD/data/input holds everything UPLOADED and HEAD/data/output everything
# the CLUSTER produced, split into one folder per pipeline step (sleep/, demand_prep/,
# logit/, blp/, blp/draws/, bbl/, counterfactuals/, download/).  Every produced family thus
# has exactly ONE producer writing to exactly ONE directory, and the helpers below are the
# single definition of where that is: a consumer resolving a file through them cannot pick
# up a second copy from elsewhere, which is what keeps the exporter and the CF/BBL stack on
# the same vintage.  Locally out_dir is a results directory under ESTIMATION_OUTPUT and the
# sibling-directory layout applies instead; the split is decided by tree shape
# (`is_cluster_out`), so one code path serves both trees.  These live here because of_root.jl
# is the one path module every Julia entry point already reaches (blp_1_logit.jl directly,
# the BLP/CF stack through blp_1_estimation.jl), so sharing them adds no include edge.

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
Detection is by tree shape — the directory is named `output` — so one code path serves both
trees and a local run never picks up a cluster-only fallback. This is the only cluster/local
test in the Julia stack: every step-directory helper below branches on it.
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

# --------------------------------------------------------------------------
# Step directories
# --------------------------------------------------------------------------
# get_paths(is_hpc=true) returns out_dir = <repo>/../data/output, so dirname(out_dir) == data/.
# The dict maps a family to its step folder; two families share `bbl` because the folder is the
# unit the archiver packages and downloads, not the unit a single script writes. Locally out_dir
# is …/ESTIMATION_OUTPUT/BLP_RESULTS and each family keeps a sibling directory named for the key.
const _HPC_SUBDIR = Dict("CF_FOUNDATION" => "counterfactuals",
                         "COST_FWD"      => "bbl",
                         "COST_POLFUNC"  => "bbl")

"""Directory the CF/BBL stack WRITES to (cluster: data/output/{counterfactuals,bbl}; local: sibling of out_dir)."""
cf_out_dir(out_dir, which::AbstractString="CF_FOUNDATION") =
    is_cluster_out(out_dir) ? joinpath(String(out_dir), get(_HPC_SUBDIR, which, lowercase(which))) :
                              joinpath(dirname(out_dir), which)

"""Directory holding UPLOADED inputs (cluster: data/input; local: sibling of out_dir)."""
cf_in_dir(out_dir, which::AbstractString="CF_FOUNDATION") =
    is_cluster_out(out_dir) ? joinpath(dirname(String(out_dir)), "input") :
                              joinpath(dirname(out_dir), which)

"""
    blp_dir(out_dir) -> String

Directory holding the RC-BLP results, checkpoints and summaries. On the cluster the family gets
its own step folder, `data/output/blp`, so the archiver can package it without sweeping the rest
of data/output. Locally `out_dir` IS the BLP results directory
(…/ESTIMATION_OUTPUT/BLP_RESULTS) and the files sit directly in it.
"""
blp_dir(out_dir)::String =
    is_cluster_out(out_dir) ? joinpath(String(out_dir), "blp") : String(out_dir)

"""
    logit_dir(out_dir) -> String

Directory holding the logit δ vectors, .jls fits, summary and .tex fragments: `out_dir/logit`.
There is no cluster/local branch because the two trees already agree — the local run keeps the
logit outputs in a `logit` subdirectory of the BLP results directory, which is exactly where the
cluster's logit step folder sits under data/output — so a branch would have two equal arms.
"""
logit_dir(out_dir)::String = joinpath(String(out_dir), "logit")

"""
    draws_dir(out_dir) -> String

Directory holding the Halton ν draws, the demographic draws and the key index. On the cluster
they belong to the BLP step folder (`data/output/blp/draws`): the draws job produces them and
they travel in the BLP zip. Locally they live in ESTIMATION_OUTPUT/BLP_DRAWS, a sibling of the
BLP results directory, which is the location `get_paths` returns for a local run.
"""
draws_dir(out_dir)::String =
    is_cluster_out(out_dir) ? joinpath(String(out_dir), "blp", "draws") :
                              joinpath(dirname(String(out_dir)), "BLP_DRAWS")

"""
    demand_search_dirs(in_dir, out_dir) -> Vector{String}

Search path for the demand-prep parquets. On the cluster it is the single directory
`out_dir/demand_prep`: the parquets have one producer (the sleepiness phase's prep step) writing
to one place, and offering no second candidate is what stops the exporter and the CF/BBL stack
from resolving two different vintages of the same routine. Off the cluster the list is just
`in_dir`, the local ESTIMATION_OUTPUT/DEMAND_PREP. Shared by blp_1_logit.jl, blp_2_rc.jl and
blp_gpu_engine.jl so the three cannot disagree about where a routine's parquet lives.
"""
demand_search_dirs(in_dir, out_dir)::Vector{String} =
    is_cluster_out(out_dir) ?
        search_dirs(joinpath(String(out_dir), "demand_prep")) :
        search_dirs(in_dir)
