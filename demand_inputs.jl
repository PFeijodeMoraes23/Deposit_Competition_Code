# demand_inputs.jl
# ================
# How the demand estimators read their regressors and instruments from a demand parquet.
# Shared by blp_engine_cpu.jl (and through it blp_engine_gpu.jl and cf_demand_eval.jl) and by
# blp_logit.jl, so the logit and the random-coefficient engines apply one rule.
#
#   * THE LINEAR DESIGN carries the D-type dummy (`dummy_D_type` = 1 on a digital firm's rows,
#     i.e. where `is_B` is false), built here from `is_B`; the demand parquets do not store it.
#   * A MISSING OR NON-FINITE VALUE in the price, a product characteristic or an instrument is an
#     error (`DemandInputError`), raised once for the whole frame with the count of every column
#     and of the rows concerned. The demand prep repairs or drops those rows (DEMAND_REPAIR=1,
#     utils/demand_repair.py), so a repaired parquet passes untouched.
#   * A parquet built before that repair names the borrowings instruments `loo_credit_assets` /
#     `mean_loo_credit_assets`; read under those names, it is the same error, reported together
#     with the gap counts.
#   * BLP_READ_PRE_REPAIR=1 reads a parquet built before that repair as every fit made from it was
#     made: the old names are accepted and the missing or non-finite values are set to 0.0 (the
#     counts are printed).
#
# Include once per session (guarded by the includers).

"""A demand parquet holds a missing or non-finite value the estimator does not fill."""
struct DemandInputError <: Exception
    msg::String
end
Base.showerror(io::IO, e::DemandInputError) = print(io, "DemandInputError: ", e.msg)

"Environment switch that reads a demand parquet built before the 2026-10 repair (see the file header)."
const PRE_REPAIR_ENV = "BLP_READ_PRE_REPAIR"
read_pre_repair()::Bool = strip(get(ENV, PRE_REPAIR_ENV, "0")) == "1"

"Name of the D-type dummy in the linear design and in every θ₁ name list."
const DTYPE_COL = "dummy_D_type"

"The borrowings instruments under their current names => the names a pre-repair parquet carries."
const PRE_REPAIR_IV_NAMES = Dict("loo_borrow_assets"      => "loo_credit_assets",
                                 "mean_loo_borrow_assets" => "mean_loo_credit_assets")

"""
    iv_source_column(df, name) -> String

The parquet column that holds instrument `name`: `name` when present, else its pre-repair name
when only that is present (`enforce_inputs!` then accepts it only under BLP_READ_PRE_REPAIR=1, and
otherwise reports it with the gap counts). Otherwise `name`, so the caller's absent-column check
names the current column.
"""
function iv_source_column(df, name::AbstractString)::String
    name in names(df) && return String(name)
    old = get(PRE_REPAIR_IV_NAMES, String(name), nothing)
    (old !== nothing && old in names(df)) && return old
    return String(name)
end

"""What the input rule found in a block of columns: per column the missing and the non-finite
counts, and which rows have at least one of them."""
struct InputGaps
    what       ::String
    cols       ::Vector{String}
    sources    ::Vector{String}
    n_missing  ::Vector{Int}
    n_nonfinite::Vector{Int}
    rows       ::BitVector
end

"""
    read_input_block(df, cols, what) -> (M::Matrix{Float64}, gaps::InputGaps)

Columns `cols` of `df` as a Float64 matrix, a missing value as NaN, nothing filled. `what` names
the block in messages ("price", "X", "instrument"); an instrument is looked up through
`iv_source_column`. An absent column is an error here, whatever the switch: it is a broken
parquet, not a value to fill.
"""
function read_input_block(df, cols::Vector{String}, what::AbstractString)
    N    = size(df, 1)
    M    = Matrix{Float64}(undef, N, length(cols))
    srcs = String[what == "instrument" ? iv_source_column(df, c) : c for c in cols]
    absent = [c for (c, s) in zip(cols, srcs) if !(s in names(df))]
    isempty(absent) || error("Demand parquet is missing $(what) column(s): " * join(absent, ", ") * ".")
    nm = zeros(Int, length(cols)); nn = zeros(Int, length(cols)); rows = falses(N)
    for (j, s) in enumerate(srcs)
        raw = df[!, s]
        @inbounds for i in 1:N
            x = raw[i]
            if x === missing || x === nothing
                M[i, j] = NaN; nm[j] += 1; rows[i] = true
            else
                f = Float64(x)
                M[i, j] = f
                isfinite(f) || (nn[j] += 1; rows[i] = true)
            end
        end
    end
    return M, InputGaps(String(what), copy(cols), srcs, nm, nn, rows)
end

"""
    enforce_inputs!(blocks; context="") -> Int

Apply the input rule to blocks returned by `read_input_block` (a vector of `(M, gaps)` pairs).
No gap and every column under its current name: returns 0 and changes nothing. Otherwise, by
default, throws one `DemandInputError` that names the instruments read under a pre-repair name
and every column with a gap (its missing and non-finite counts), with the rows concerned per
block. Under BLP_READ_PRE_REPAIR=1 it accepts the pre-repair names, sets every non-finite entry of
the blocks to 0.0, prints the same counts, and returns the number of rows filled.
"""
function enforce_inputs!(blocks::AbstractVector; context::AbstractString="")
    lines = String[]; per_block = String[]; any_rows = nothing; renamed = String[]
    for (_, g) in blocks
        for j in eachindex(g.cols)
            g.sources[j] == g.cols[j] || push!(renamed, "$(g.cols[j]) (column $(g.sources[j]))")
            (g.n_missing[j] + g.n_nonfinite[j]) == 0 && continue
            push!(lines, "    $(g.what) $(g.cols[j])" * (g.sources[j] == g.cols[j] ? "" : " (column $(g.sources[j]))") *
                         ": $(g.n_missing[j]) missing, $(g.n_nonfinite[j]) non-finite")
        end
        if any(g.rows)
            push!(per_block, "$(g.what): $(count(g.rows)) of $(length(g.rows)) rows")
            any_rows = any_rows === nothing ? copy(g.rows) : (any_rows .| g.rows)
        end
    end
    isempty(lines) && isempty(renamed) && return 0
    where_ = isempty(context) ? "" : " ($(context))"
    n_rows = any_rows === nothing ? 0 : count(any_rows)
    head = n_rows == 0 ? "no missing or non-finite value$(where_)" :
           "$(n_rows) row(s)$(where_) hold a missing or non-finite value — " * join(per_block, "; ")
    name_note = isempty(renamed) ? "" :
        "  Read under the pre-repair name: " * join(renamed, ", ") * ".\n"
    if read_pre_repair()
        for (M, _) in blocks
            M[.!isfinite.(M)] .= 0.0
        end
        println("  [INPUT FILL] $(PRE_REPAIR_ENV)=1: $(head)" *
                (n_rows == 0 ? ".\n" : ". Set to 0.0, as every fit made before the 2026-10 repair did:\n" *
                                       join(lines, "\n") * "\n") * name_note)
        flush(stdout)
        return n_rows
    end
    throw(DemandInputError("$(head)" * (isempty(lines) ? ".\n" : ":\n" * join(lines, "\n") * "\n") *
        name_note *
        "  This is a demand parquet built before the 2026-10 repair; the estimators no longer turn " *
        "these values into 0.0 or read the borrowings instruments under their old names. Build the " *
        "demand parquets with the repair (DEMAND_REPAIR=1, utils/demand_repair.py), which fills or " *
        "drops these rows; or set $(PRE_REPAIR_ENV)=1 to read this parquet as every fit before the " *
        "repair did (zero fill, old names)."))
end

"""
    dtype_column(df) -> Vector{Float64}

The D-type dummy: 1.0 where `is_B` is false (a digital firm), 0.0 where it is true. A missing
`is_B` follows the input rule: an error, or under BLP_READ_PRE_REPAIR=1 a D row (the estimators'
earlier `coalesce(is_B, false)`), with the count printed.
"""
function dtype_column(df)::Vector{Float64}
    "is_B" in names(df) || error("Demand parquet has no `is_B` column: the D-type dummy cannot be built.")
    raw = df[!, "is_B"]
    nmiss = count(x -> x === missing || x === nothing, raw)
    if nmiss > 0
        msg = "is_B: $(nmiss) missing of $(length(raw)) rows (the D-type dummy is built from it)"
        read_pre_repair() || throw(DemandInputError(msg * ". Rebuild the demand parquets, or set " *
                                                    "$(PRE_REPAIR_ENV)=1 to read them as D rows."))
        println("  [INPUT FILL] $(PRE_REPAIR_ENV)=1: $(msg); read as D rows."); flush(stdout)
    end
    return Float64[(x === missing || x === nothing) ? 1.0 : (Bool(x) ? 0.0 : 1.0) for x in raw]
end
