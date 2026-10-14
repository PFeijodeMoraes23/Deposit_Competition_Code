# demand_inputs.jl
# ================
# How the demand estimators read their regressors and instruments from a demand parquet.
# Shared by blp_engine_cpu.jl (and through it blp_engine_gpu.jl and cf_demand_eval.jl) and by
# blp_logit.jl, so the logit and the random-coefficient engines apply one rule.
#
#   * THE LINEAR DESIGN carries the D-type dummy (`dummy_D_type` = 1 on a digital firm's rows,
#     i.e. where `is_B` is false), built here from `is_B`; the demand parquets do not store it.
#   * A MISSING OR NON-FINITE VALUE in the price, a product characteristic, an instrument, a
#     market share or the deposit type is an error (`DemandInputError`), raised once for the whole
#     frame with the count of every column and of the rows concerned. The demand preps fill or
#     drop those rows (utils/demand_chars.py), so their parquets pass untouched.
#   * The borrowings instruments are read as `loo_borrow_assets` / `mean_loo_borrow_assets`. A
#     parquet that carries them only as `loo_credit_assets` / `mean_loo_credit_assets` (the column
#     names in the stored parquets) is an error too, reported together with the gap counts.
#   * BLP_ZERO_FILL_MISSING=1 reproduces the zero fill of the stored results from the stored
#     parquets: the missing or non-finite values are set to 0.0 (the counts are printed) and the
#     borrowings instruments are read under the column names of the stored parquets.
#
# Include once per session (guarded by the includers).

"""A demand parquet holds a missing or non-finite value the estimator does not fill."""
struct DemandInputError <: Exception
    msg::String
end
Base.showerror(io::IO, e::DemandInputError) = print(io, "DemandInputError: ", e.msg)

"Environment switch that reproduces the zero fill of the stored results (see the file header)."
const ZERO_FILL_ENV = "BLP_ZERO_FILL_MISSING"
zero_fill_missing()::Bool = strip(get(ENV, ZERO_FILL_ENV, "0")) == "1"

"Name of the D-type dummy in the linear design and in every θ₁ name list."
const DTYPE_COL = "dummy_D_type"

"The borrowings instruments => their column names in the stored parquets."
const STORED_IV_NAMES = Dict("loo_borrow_assets"      => "loo_credit_assets",
                             "mean_loo_borrow_assets" => "mean_loo_credit_assets")

"""
    stored_iv_column(df, name) -> Union{String,Nothing}

The column name of instrument `name` in the stored parquets, when `df` lacks `name` but carries
that column; otherwise nothing.
"""
function stored_iv_column(df, name::AbstractString)
    name in names(df) && return nothing
    old = get(STORED_IV_NAMES, String(name), nothing)
    return (old !== nothing && old in names(df)) ? old : nothing
end

"""What the input rule found in a block of columns: per column the column read (`sources`), the
missing and the non-finite counts, the stored-parquet column found in place of a current one and
not read (`unread`, "" when none), and which rows have a gap."""
struct InputGaps
    what       ::String
    cols       ::Vector{String}
    sources    ::Vector{String}
    unread     ::Vector{String}
    n_missing  ::Vector{Int}
    n_nonfinite::Vector{Int}
    rows       ::BitVector
end

"""
    read_input_block(df, cols, what) -> (M::Matrix{Float64}, gaps::InputGaps)

Columns `cols` of `df` as a Float64 matrix, a missing value as NaN, nothing filled. `what` names
the block in messages ("price", "X", "instrument"). An instrument the parquet carries only under
its stored-parquet column name is read from that column under BLP_ZERO_FILL_MISSING=1; otherwise
it is left unread (NaN, not counted) and `enforce_inputs!` reports it. Any other absent column is
an error here, whatever the switch: it is a broken parquet, not a value to fill.
"""
function read_input_block(df, cols::Vector{String}, what::AbstractString)
    N      = size(df, 1)
    M      = Matrix{Float64}(undef, N, length(cols))
    stored = [what == "instrument" ? stored_iv_column(df, c) : nothing for c in cols]
    srcs   = String[(s !== nothing && zero_fill_missing()) ? s : c for (c, s) in zip(cols, stored)]
    unread = String[(s !== nothing && !zero_fill_missing()) ? s : "" for s in stored]
    absent = [c for (c, s, u) in zip(cols, srcs, unread) if isempty(u) && !(s in names(df))]
    isempty(absent) || error("Demand parquet is missing $(what) column(s): " * join(absent, ", ") * ".")
    nm = zeros(Int, length(cols)); nn = zeros(Int, length(cols)); rows = falses(N)
    for (j, s) in enumerate(srcs)
        if !isempty(unread[j])
            M[:, j] .= NaN
            continue
        end
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
    return M, InputGaps(String(what), copy(cols), srcs, unread, nm, nn, rows)
end

"""
    enforce_inputs!(blocks; context="") -> Int

Apply the input rule to blocks returned by `read_input_block` (a vector of `(M, gaps)` pairs).
No gap and every column read under its current name: returns 0 and changes nothing. Otherwise, by
default, throws one `DemandInputError` that names the instruments found only under their
stored-parquet column names and every column with a gap (its missing and non-finite counts), with
the rows concerned per block. Under BLP_ZERO_FILL_MISSING=1 it sets every non-finite entry of the
blocks to 0.0, prints the same counts and the stored-parquet columns read, and returns the number
of rows filled.
"""
function enforce_inputs!(blocks::AbstractVector; context::AbstractString="")
    lines = String[]; per_block = String[]; any_rows = nothing; read_stored = String[]; unread = String[]
    for (_, g) in blocks
        for j in eachindex(g.cols)
            g.sources[j] == g.cols[j] || push!(read_stored, "$(g.cols[j]) (column $(g.sources[j]))")
            isempty(g.unread[j]) || push!(unread, "$(g.cols[j]) (the parquet has $(g.unread[j]))")
            (g.n_missing[j] + g.n_nonfinite[j]) == 0 && continue
            push!(lines, "    $(g.what) $(g.cols[j])" * (g.sources[j] == g.cols[j] ? "" : " (column $(g.sources[j]))") *
                         ": $(g.n_missing[j]) missing, $(g.n_nonfinite[j]) non-finite")
        end
        if any(g.rows)
            push!(per_block, "$(g.what): $(count(g.rows)) of $(length(g.rows)) rows")
            any_rows = any_rows === nothing ? copy(g.rows) : (any_rows .| g.rows)
        end
    end
    isempty(lines) && isempty(read_stored) && isempty(unread) && return 0
    where_ = isempty(context) ? "" : " ($(context))"
    n_rows = any_rows === nothing ? 0 : count(any_rows)
    head = n_rows == 0 ? "no missing or non-finite value$(where_)" :
           "$(n_rows) row(s)$(where_) hold a missing or non-finite value — " * join(per_block, "; ")
    if zero_fill_missing()
        for (M, _) in blocks
            M[.!isfinite.(M)] .= 0.0
        end
        println("  [INPUT FILL] $(ZERO_FILL_ENV)=1: $(head)" *
                (n_rows == 0 ? ".\n" : ". Set to 0.0, as in the stored results:\n" * join(lines, "\n") * "\n") *
                (isempty(read_stored) ? "" :
                 "  Read under the column name of the stored parquets: " * join(read_stored, ", ") * ".\n"))
        flush(stdout)
        return n_rows
    end
    throw(DemandInputError("$(head)" * (isempty(lines) ? ".\n" : ":\n" * join(lines, "\n") * "\n") *
        (isempty(unread) ? "" :
         "  Instrument(s) present only under the column name of the stored parquets, not read: " *
         join(unread, ", ") * ".\n") *
        "  The estimators do not fill these values. Build the demand parquets with the current preps " *
        "(utils/demand_chars.py fills or drops these rows and names the instruments), or set " *
        "$(ZERO_FILL_ENV)=1 to reproduce the zero fill of the stored results."))
end

"""
    check_shares_and_types(df) -> Int

The input rule for the market shares and the deposit type: a D row (`is_B` false) needs a finite
`share_D`, a B row a finite `share_B_cond`, every row a `deposit_type`. The estimators read them
with `coalesce(…, 0)`, so a gap would enter as a zero share (an infinitely negative mean utility)
or as deposit type 0 (outside every per-type first stage). A gap is a `DemandInputError` with the
counts; under BLP_ZERO_FILL_MISSING=1 the counts are printed and the zero reading stands.
Returns the number of rows with a gap.
"""
function check_shares_and_types(df)::Int
    for c in ("share_D", "share_B_cond", "deposit_type", "is_B")
        c in names(df) || error("Demand parquet has no `$(c)` column.")
    end
    bad(x) = x === missing || x === nothing || !isfinite(Float64(x))
    isb  = [x === missing || x === nothing ? false : Bool(x) for x in df[!, "is_B"]]
    cD, cB = df[!, "share_D"], df[!, "share_B_cond"]
    sD   = [!isb[i] && bad(cD[i]) for i in eachindex(isb)]
    sB   = [ isb[i] && bad(cB[i]) for i in eachindex(isb)]
    dt   = [x === missing || x === nothing for x in df[!, "deposit_type"]]
    rows = sD .| sB .| dt
    n = count(rows)
    println("  [INPUT CHECK] shares and deposit type: $(n) row(s) with a gap (share_D on D rows " *
            "$(count(sD)), share_B_cond on B rows $(count(sB)), deposit_type $(count(dt)))"); flush(stdout)
    n == 0 && return 0
    msg = "$(n) row(s) of the demand parquet lack a share or a deposit type — share_D on D rows: " *
          "$(count(sD)); share_B_cond on B rows: $(count(sB)); deposit_type: $(count(dt))"
    if zero_fill_missing()
        println("  [INPUT FILL] $(ZERO_FILL_ENV)=1: $(msg); read as 0, as in the stored results."); flush(stdout)
        return n
    end
    throw(DemandInputError(msg * ". Rebuild the demand parquets, or set $(ZERO_FILL_ENV)=1 to " *
                           "reproduce the zero fill of the stored results."))
end

"""
    dtype_column(df) -> Vector{Float64}

The D-type dummy: 1.0 where `is_B` is false (a digital firm), 0.0 where it is true. A missing
`is_B` follows the input rule: an error, or under BLP_ZERO_FILL_MISSING=1 a D row (the
`coalesce(is_B, false)` of the stored results), with the count printed.
"""
function dtype_column(df)::Vector{Float64}
    "is_B" in names(df) || error("Demand parquet has no `is_B` column: the D-type dummy cannot be built.")
    raw = df[!, "is_B"]
    nmiss = count(x -> x === missing || x === nothing, raw)
    if nmiss > 0
        msg = "is_B: $(nmiss) missing of $(length(raw)) rows (the D-type dummy is built from it)"
        zero_fill_missing() || throw(DemandInputError(msg * ". Rebuild the demand parquets, or set " *
                                                      "$(ZERO_FILL_ENV)=1 to read them as D rows."))
        println("  [INPUT FILL] $(ZERO_FILL_ENV)=1: $(msg); read as D rows."); flush(stdout)
    end
    return Float64[(x === missing || x === nothing) ? 1.0 : (Bool(x) ? 0.0 : 1.0) for x in raw]
end
