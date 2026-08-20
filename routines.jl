# routines.jl — the Julia view of config/routines.toml.
#
# The estimator lineup, the \ref{estimation:*} map, the demand-parquet prefixes and the
# headline spec key are defined once in config/routines.toml; this file parses that with the
# TOML stdlib and exposes the names the Julia entry points already use. utils/routines.py is
# the Python view of the SAME file, so a routine added, renamed or retired in the toml
# reaches both languages at once.
#
# Include it the way the entry points include of_root.jl, behind a guard so the consts are
# defined once even though several files in one session want them:
#
#     isdefined(Main, :ROUTINE_REGISTRY) || include(joinpath(@__DIR__, "routines.jl"))
#
# TOML is a stdlib and resolves from @stdlib under `--project=.`, so this adds no dependency
# and needs no Project.toml entry.

using TOML

"""
    _read_registry() -> Dict

Parse config/routines.toml, naming the fix when the file is not where it should be. On the
cluster this file lands at `scripts/routines.jl` and looks for `scripts/config/routines.toml`
— the same copy utils/routines.py reads. It gets there only if the upload manifest carries
it, so a missing registry means the manifest needs the entry, not that the lineup should be
pasted back in here.
"""
function _read_registry()
    p = joinpath(@__DIR__, "config", "routines.toml")
    isfile(p) || error(
        "The routine registry is missing: $p. Every consumer of the lineup reads it, so " *
        "nothing runs without it. If this is a cluster run, add config/routines.toml to " *
        "the upload manifest with destination scripts/config (utils/routines.py reads the " *
        "same copy).")
    return TOML.parsefile(p)
end

"""The parsed config/routines.toml. Also the include guard's sentinel."""
const ROUTINE_REGISTRY = _read_registry()

const _ROUTINE_ROWS = ROUTINE_REGISTRY["routines"]

_routine_map(field) =
    Dict{Int,String}(Int(r["id"]) => String(r[field]) for r in _ROUTINE_ROWS)

"""The reported lineup, in the order the toml lists it."""
const ACTIVE_ROUTINES = Int[Int(e) for e in ROUTINE_REGISTRY["active"]]

"""The routines estimated through the shared single-index link machinery."""
const LINK_ROUTINES = Int[Int(e) for e in ROUTINE_REGISTRY["link_ests"]]

"""Default cluster routine set: the single-index links and their +Time variants.
Override with the ROUTINES env in the submit scripts, or --all-routines for all."""
const DEFAULT_ROUTINES = LINK_ROUTINES

"""id => the routine's description, for logs, JSON metadata and cluster output."""
const ROUTINE_DESC = _routine_map("name")

"""id => the routine's column header in .tex tables."""
const ROUTINE_TEX_NAME = _routine_map("tex_name")

"""id => the \\ref into V_Main's `\\item\\label{estimation:*}` enumerate in
sec:empirical:sleep. An id absent here falls back to a plain E<id> header."""
const ESTIMATION_ENUM_REF = _routine_map("tex_ref")

"""id => demand-prep parquet stem, completed as `{prefix}_spec_{S}.parquet`. Used only when
the driver has not auto-discovered and passed BLP_DEMAND_PREFIX."""
const DEMAND_PREFIXES = _routine_map("demand_prefix")

"""id => the V_Main enumerate label, without the \\ref wrapper."""
const ROUTINE_SLUG = _routine_map("slug")

"""id => whether the routine carries the Time block."""
const ROUTINE_TIME_BLOCK =
    Dict{Int,Bool}(Int(r["id"]) => Bool(r["time_block"]) for r in _ROUTINE_ROWS)

"""The routines carrying the Time block."""
const TIME_ROUTINES = Int[e for e in ACTIVE_ROUTINES if ROUTINE_TIME_BLOCK[e]]

const SPEC12      = String(ROUTINE_REGISTRY["spec12"])
const SPEC12_TAG  = String(ROUTINE_REGISTRY["spec12_tag"])
const SPEC12_ID   = Int(ROUTINE_REGISTRY["spec12_id"])
const FE_TIME_COL = String(ROUTINE_REGISTRY["fe_time_col"])

"""The SLEEP_ACTIVE_ESTS default, as the env var itself spells it."""
const ACTIVE_ENV_DEFAULT = join(string.(ACTIVE_ROUTINES), " ")

"""
    active_from_env() -> Vector{Int}

The active lineup, narrowed by ENV["SLEEP_ACTIVE_ESTS"]. The variable is a space-separated
id list ("3 4"); unset means the whole lineup, and the empty string means an empty lineup —
the same parse the Python orchestrators apply, so the two sides select the same routines.
"""
active_from_env()::Vector{Int} =
    Int[parse(Int, x) for x in split(get(ENV, "SLEEP_ACTIVE_ESTS", ACTIVE_ENV_DEFAULT))]

"""Routine id -> its description (fallback E<id> for an id outside the lineup)."""
routine_desc(id::Int) = get(ROUTINE_DESC, id, "E$id")

"""Routine id -> its V_Main enumerate \\ref (fallback E<id> for an id with no live label)."""
est_ref(id::Int) = get(ESTIMATION_ENUM_REF, id, "E$id")

"""Routine id -> its static demand-parquet prefix (fallback the bare demand_<id>)."""
routine_demand_prefix(id::Int) = get(DEMAND_PREFIXES, id, "demand_$(id)")

"""Routine id -> its .tex column header (fallback E<id>)."""
routine_tex_name(id::Int) = get(ROUTINE_TEX_NAME, id, "E$id")
