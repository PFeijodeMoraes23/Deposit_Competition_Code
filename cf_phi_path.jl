"""
cf_phi_path.jl
================
The sleepy share φ along each simulated path, for the BBL forward simulation (bbl_fwd_sim.jl):

    φ_{i,t} = clamp( G( Σ_k θ_k · S_{k,i,t} ), 0, PHI_SIM_MAX ),        t = 1…T

G and θ are the routine's OWN fitted link and index direction, exported by bbl_sleep_link.py as
`sleep_link_E{k}_spec_{s}.json` (the monotone I-spline stored as a grid over the native index,
evaluated with numpy.interp semantics). How each state moves along a path (user decisions
2026-09-28):

  constant  1.
  held      pix_exists (firms forecast with launch-time information: no foresight of the 2020Q4
            launch) and E4's +Time block (gdp_growth_yoy): the row's launch-quarter value.
  selic     risk_free_qoq_lag, the lagged quarterly Selic, follows the row's launch vintage
            lagged one quarter, in estimation units:
                S_rf,t = r^f(h = t−1) / scale − mean,
            where h = 0 is the launch quarter's own (realised) rate and h ≥ 1 the vintage's Focus
            mean. These are the same rates the lagged deposit accrual uses (rf_h0 and the curve
            columns in simulate_deposits), so φ and the carry read one rate path — with ONE rate
            path per launch quarter (--n-paths 1, the production design). With rate risk
            (--n-paths P > 1) φ stays on the Focus MEAN curve, built once, while the carry
            follows each shocked path; at P = 1 the two coincide.
  market    a market demographic (65+, mobile lines, CadÚnico) mean-reverts within its MCA at the
            market_states AR(1) of bbl_transitions.json — the same ρ, the same ρ field
            (CF_STATE_RHO_FIELD) and the same 0 < ρ < 1 guard as the demand block's
            StateEvolution, and the same long-run level (the MCA's mean over the quarters the
            panel observes, one value per (mca_code, time_id) market key):
                S_t = S_0 + (ρ^t − 1)·(S_0 − S̄_m).

At t = 0 every shift is exactly zero, so φ_0 is the index the demand prep built, summed in the
same order: it reproduces the parquet's phi_mt (bit for bit on the shipped vintage), and
`phi_path_inputs` refuses a link that does not (PHI_T0_TOL). With every ρ = 1 and the Selic
held, S_t ≡ S_0 exactly and so φ_t ≡ φ_0.

WHICH φ A ROW FOLLOWS. The demand block inverts a B row with its own φ_mt and a D row (is_B false,
a national firm) with the NATIONAL φ_t (eq. 14-D; sleep_demand_prep_link.process_specification
builds Dep_Act with `phi_mt` for B rows and `phi_t` for D rows). The path keeps that split:

  B rows  φ_{i,t} = clamp(G(Σ_k θ_k S_{k,i,t}), 0, PHI_SIM_MAX), the row's own link path above.
  D rows  φ_{i,t} = clamp(φ_t^stored(q) + [Φ_q(t) − Φ_q(0)], 0, PHI_SIM_MAX), q = the row's launch
          quarter, with the national aggregate built as the sleep side builds its national φ_t
          (sleep_est_single._calculate_phis, sleep_est_e1.calculate_phis):
              Φ_q(t) = Σ_m M_m φ̄_m(t) / Σ_m M_m,
          m running over the quarter's MARKET CELLS, the distinct (mca_code, time_id) pairs of the
          parquet, each counted ONCE whatever its number of rows; φ̄_m(t) the mean over the cell's
          rows of their own link value in [0, 1] along their state path; M_m the cell's mean
          pop_total (a missing pop_total counts 0, as the sleep side's fillna(0)), taken AT THE
          LAUNCH QUARTER and held there along the path. The cell set is the one the sleep side
          sums over, the D rows' 'NATIONAL' cell included, restricted to the cells the demand
          parquet holds.
  The level is the parquet's stored φ_t and the path moves it by the change in the aggregate.
  That is exact at t = 0 (the bracket is formed first and is 0.0), constant when no state moves,
  and needed because the stored φ_t is the demand prep's own aggregate, formed BEFORE its
  Dep_Act > 1e-6 row filter over its (mca_code, time_id, phi_mt, pop_total) tuples: the parquet's
  cells reproduce it only to ~3e-3 (0 of 36 quarters exact, demand_{3,4}_*_spec_12), so the
  aggregate's LEVEL cannot be rebuilt from the parquet, its change can.

The path depends on the states alone — never on a spread — so it is built ONCE per shard as an
N×T matrix and shared by the equilibrium and every deviation, exactly like the share path.
"""

using DataFrames, Statistics
import JSON3

# The sleepy-share clamp of the forward simulation. load_sim_state clamps the parquet's phi_mt to
# [0, 0.999] for the frozen path; the evolving path applies the same bounds every period, so the
# two differ only in the states the link is evaluated at.
const PHI_SIM_MAX = 0.999
# The t = 0 gate: φ_0 from the exported link must reproduce the parquet's phi_mt row by row.
const PHI_T0_TOL = 1e-10
const SLEEP_LINK_SCHEMA = "bbl_sleep_link/1"

# SHA-256 of an input file, for the provenance the psi files carry. SHA is a stdlib, loaded by
# its package id so it does not need to be a direct dependency of the project; "" if unavailable.
const _PHI_SHA = try
    Base.require(Base.PkgId(Base.UUID("ea8e919c-243c-51af-8825-aaa63cd721ce"), "SHA"))
catch
    nothing
end
function _file_sha256(path::AbstractString)::String
    (_PHI_SHA === nothing || !isfile(path)) && return ""
    return open(io -> bytes2hex(Base.invokelatest(_PHI_SHA.sha256, io)), path)
end

# ==========================================================================
# The exported sleepiness link
# ==========================================================================
struct SleepLink
    estim    ::Int
    spec     ::Int
    cols     ::Vector{String}   # the index states, in the estimator's summation order
    coef     ::Vector{Float64}  # θ (params_native), same order
    role     ::Vector{Symbol}   # :constant | :held | :selic | :market
    vgrid    ::Vector{Float64}  # native-index grid (numpy.interp xp), strictly increasing
    ggrid    ::Vector{Float64}  # the link on it (numpy.interp fp)
    rf_scale ::Float64          # SCALE divisor of the Selic state (state_transform.py)
    rf_mean  ::Float64          # its grand mean in estimation units (state_transform.py)
    path     ::String
    sha256   ::String
end

const _PHI_ROLES = (:constant, :held, :selic, :market)

"""
    load_sleep_link(path; estim=nothing, spec=nothing) -> SleepLink

Read `sleep_link_E{k}_spec_{s}.json` (bbl_sleep_link.py). Refuses a file of another schema, a
link kind other than `index_sieve`, a grid that is not strictly increasing, an unknown role, or
anything but exactly one Selic state; `estim`/`spec`, when given, must match the file.
"""
function load_sleep_link(path::AbstractString; estim::Union{Nothing,Int}=nothing,
                         spec::Union{Nothing,Int}=nothing)
    isfile(path) || error("sleep link not found: $path\n" *
                          "  Build it locally (python bbl_sleep_link.py) and upload it to data/input.")
    j = JSON3.read(read(path, String))
    String(get(j, :schema, "")) == SLEEP_LINK_SCHEMA ||
        error("$(basename(path)): schema '$(get(j, :schema, ""))' is not $SLEEP_LINK_SCHEMA")
    e = Int(j[:estim]); s = Int(j[:spec])
    (estim === nothing || estim == e) || error("$(basename(path)) is E$e, the run is E$estim")
    (spec === nothing || spec == s) || error("$(basename(path)) is spec $s, the run is spec $spec")
    String(j[:link][:kind]) == "index_sieve" ||
        error("$(basename(path)): link kind '$(j[:link][:kind])' is not index_sieve")
    terms = j[:index][:terms]
    cols = String[String(t[:state]) for t in terms]
    coef = Float64[Float64(t[:coef]) for t in terms]
    role = Symbol[Symbol(String(t[:role])) for t in terms]
    all(r -> r in _PHI_ROLES, role) || error("$(basename(path)): unknown role in $(unique(role))")
    count(==(:selic), role) == 1 || error("$(basename(path)): need exactly one :selic state")
    vgrid = Float64.(collect(j[:link][:vgrid])); ggrid = Float64.(collect(j[:link][:ggrid]))
    (length(vgrid) == length(ggrid) && length(vgrid) >= 2 && all(diff(vgrid) .> 0)) ||
        error("$(basename(path)): vgrid must be strictly increasing and match ggrid")
    krf = findfirst(==(:selic), role)
    st  = j[:states][Symbol(cols[krf])]
    return SleepLink(e, s, cols, coef, role, vgrid, ggrid, Float64(st[:scale]),
                     Float64(st[:mean_scaled]), String(path), _file_sha256(path))
end

"""
    _np_interp(x, xp, fp) -> Float64

numpy.interp for one point, with numpy's exact arithmetic: the slope of the bracketing segment
times (x − xp[j]) plus fp[j]; fp[1] left of the grid, fp[end] at and right of its last point,
fp[j] when x hits a grid point exactly, NaN in → NaN out. This is the link evaluation of
utils.sleep_links.phi_from_native and of the demand prep, so a φ built here reproduces theirs.
"""
@inline function _np_interp(x::Float64, xp::AbstractVector{Float64}, fp::AbstractVector{Float64})
    isnan(x) && return x
    n = length(xp)
    x < xp[1] && return fp[1]
    x >= xp[n] && return fp[n]
    j = searchsortedlast(xp, x)                     # xp[j] <= x < xp[j+1]
    xp[j] == x && return fp[j]
    slope = (fp[j+1] - fp[j]) / (xp[j+1] - xp[j])
    r = slope * (x - xp[j]) + fp[j]
    if isnan(r)
        r = slope * (x - xp[j+1]) + fp[j+1]
        (isnan(r) && fp[j] == fp[j+1]) && (r = fp[j])
    end
    return r
end

"""
    sleep_link_phi(L, v) -> φ ∈ [0, 1]

The link at native index `v`: clip(numpy.interp(v, vgrid, ggrid), 0, 1), as phi_from_native.
"""
@inline sleep_link_phi(L::SleepLink, v::Float64) = clamp(_np_interp(v, L.vgrid, L.ggrid), 0.0, 1.0)

# The Selic state in estimation units from a quarterly rate LEVEL.
@inline _selic_state(L::SleepLink, r::Float64) = r / L.rf_scale - L.rf_mean

# ==========================================================================
# Launch states, deviations and persistence
# ==========================================================================
"""
    NationalPhi

The national φ_t the D rows follow (see the module docstring): the market cells of every launch
quarter, their launch-quarter population weights, the stored φ_t it anchors to, and the aggregate
at t = 0 that the path's change is measured from.
"""
struct NationalPhi
    quarters ::Vector{String}   # launch quarters (time_id), in first-occurrence order
    row_cell ::Vector{Int}      # per row: its (mca_code, time_id) cell
    cell_n   ::Vector{Int}      # per cell: its number of rows (the φ̄_m denominator)
    cell_w   ::Vector{Float64}  # per cell: M_m, its mean pop_total at the launch quarter (held)
    cell_q   ::Vector{Int}      # per cell: its quarter, an index into `quarters`
    row_q    ::Vector{Int}      # per row: its quarter index for a D row, 0 for a B row
    phi_t    ::Vector{Float64}  # per quarter: the parquet's stored φ_t (the level)
    agg0     ::Vector{Float64}  # per quarter: Φ_q(0) over the parquet's cells
end

struct PhiPathInputs
    link    ::SleepLink
    S0      ::Matrix{Float64}   # N × K launch states, estimation units (the parquet columns)
    dev     ::Matrix{Float64}   # N × K, S0 − the MCA's long-run level (0 outside :market)
    rho     ::Vector{Float64}   # K, within-MCA AR(1) persistence; 1.0 = does not move
    k_rf    ::Int               # the Selic column
    phi0    ::Vector{Float64}   # the link at the launch states, before the simulation clamp
    t0_maxdiff::Float64         # max |phi0 − phi_mt|
    transitions::String         # the file the ρ were read from
    rho_field::String
    isB     ::BitVector         # the row's type: B rows follow their own link, D rows the national φ_t
    nat     ::NationalPhi
end

"""
    _mca_level_dev(vals, mca, tid) -> dev

`vals` minus its MCA's long-run level, where the level is the mean over the MCA's (mca, quarter)
market keys of the key's value (the first row of each key), i.e. the MCA's mean over the quarters
the panel observes — the same object `build_state_evolution` demeans by.
"""
function _mca_level_dev(vals::AbstractVector{Float64}, mca::Vector{String}, tid::Vector{String})
    N = length(vals)
    key_of = Dict{Tuple{String,String},Int}()
    first_row = Int[]; key_mca = Int[]
    mca_of = Dict{String,Int}()
    row_mca = Vector{Int}(undef, N)
    @inbounds for i in 1:N
        g = get!(mca_of, mca[i], length(mca_of) + 1)
        row_mca[i] = g
        k = get(key_of, (mca[i], tid[i]), 0)
        if k == 0
            key_of[(mca[i], tid[i])] = length(first_row) + 1
            push!(first_row, i); push!(key_mca, g)
        end
    end
    G = length(mca_of)
    s = zeros(G); c = zeros(Int, G)
    @inbounds for (k, i) in enumerate(first_row)
        s[key_mca[k]] += vals[i]; c[key_mca[k]] += 1
    end
    dev = Vector{Float64}(undef, N)
    @inbounds for i in 1:N
        dev[i] = vals[i] - s[row_mca[i]] / c[row_mca[i]]
    end
    return dev
end

"""
    _phi_market_rho(transitions_path, cols, role; rho_field) -> (rho::Vector, notes::Vector{String})

ρ per index column: from `market_states[state][rho_field]` for a :market state, kept only when
0 < ρ < 1 (the StateEvolution guard: a ρ ≥ 1 would be an explosive demographic and is frozen);
1.0 for every other role and for a market state whose transition is missing or rejected.
"""
function _phi_market_rho(transitions_path::AbstractString, cols::Vector{String},
                         role::Vector{Symbol}; rho_field::String="rho")
    isfile(transitions_path) || error("bbl_transitions.json not found: $transitions_path")
    ms = get(JSON3.read(read(transitions_path, String)), :market_states, nothing)
    rho = ones(length(cols)); notes = String[]
    for (k, c) in enumerate(cols)
        role[k] == :market || continue
        e = ms === nothing ? nothing : get(ms, Symbol(c), nothing)
        r = e === nothing ? nothing : get(e, Symbol(rho_field), nothing)
        if r isa Real && isfinite(r) && 0.0 < Float64(r) < 1.0
            rho[k] = Float64(r)
        else
            push!(notes, "$c: no usable market_states.$rho_field ($(r === nothing ? "missing" : r)) -> frozen")
        end
    end
    return rho, notes
end

"""
    phi_path_inputs(df, L; transitions_path, rho_field=CF_STATE_RHO_FIELD, t0_tol=PHI_T0_TOL)
        -> PhiPathInputs

Read the launch states from `df` (the demand parquet rows the simulation runs on), build the
MCA-level deviations of the :market states and their ρ, evaluate φ_0, and build the national φ_t
the D rows follow (`NationalPhi`, see the module docstring). Every check here is a refusal:

  * `phi_mt` must be present and φ_0 must reproduce it to `t0_tol` on every row — otherwise the
    simulation would walk a sleepiness model the demand block was not estimated with;
  * the Selic state must be the parquet's lagged-rate LEVEL in estimation units,
    risk_free_qoq_lag == risk_free_qoq_lag_level / scale − mean (mean 0.0216121875744514 on the
    shipped transform), to 1e-12 on every row — the transform the link applies along the path;
  * `phi_t` and `pop_total` must be present, `phi_t` one finite value per quarter, and every D
    row's quarter must carry a positive population weight.
"""
function phi_path_inputs(df::DataFrame, L::SleepLink; transitions_path::AbstractString,
                         rho_field::String=get(ENV, "CF_STATE_RHO_FIELD", "rho"),
                         t0_tol::Float64=PHI_T0_TOL)
    N = nrow(df); K = length(L.cols)
    need = ["phi_mt", "phi_t", "pop_total", "mca_code", "time_id", "is_B", "risk_free_qoq_lag_level"]
    miss = [c for c in need if !(c in names(df))]
    isempty(miss) || error("evolving φ needs the demand parquet columns $(miss) (phi_mt: the t = 0 " *
                           "check; phi_t, pop_total: the national path of the D rows; " *
                           "risk_free_qoq_lag_level: the Selic transform check). Rebuild the parquet.")
    S0 = Matrix{Float64}(undef, N, K)
    for (k, c) in enumerate(L.cols)
        if c in names(df)
            S0[:, k] .= Float64.(coalesce.(df[!, c], NaN))
        elseif L.role[k] == :constant
            S0[:, k] .= 1.0
        else
            error("sleep link state '$c' is not a column of the demand parquet")
        end
    end
    krf = findfirst(==(:selic), L.role)
    lvl = Float64.(coalesce.(df.risk_free_qoq_lag_level, NaN))
    drf = maximum(abs.(view(S0, :, krf) .- (lvl ./ L.rf_scale .- L.rf_mean)))
    (isfinite(drf) && drf <= 1e-12) || error(
        "the parquet's $(L.cols[krf]) is not risk_free_qoq_lag_level / $(L.rf_scale) − $(L.rf_mean) " *
        "(max |Δ| = $drf): the Selic transform in $(basename(L.path)) differs from the one the " *
        "parquet was centred with (utils/state_transform.py). Re-export the link or rebuild the parquet.")
    rho, notes = _phi_market_rho(transitions_path, L.cols, L.role; rho_field=rho_field)
    mca = string.(df.mca_code); tid = string.(df.time_id)
    dev = zeros(N, K)
    for k in 1:K
        L.role[k] == :market || continue
        dev[:, k] .= _mca_level_dev(view(S0, :, k), mca, tid)
    end
    phi0 = Vector{Float64}(undef, N)
    @inbounds for i in 1:N
        v = 0.0
        for k in 1:K; v += L.coef[k] * S0[i, k]; end
        phi0[i] = sleep_link_phi(L, v)
    end
    pm = Float64.(coalesce.(df.phi_mt, NaN))
    d0 = maximum(abs.(phi0 .- pm))
    (isfinite(d0) && d0 <= t0_tol) || error(
        "sleep link $(basename(L.path)) does NOT reproduce the parquet's phi_mt at t=0: " *
        "max |phi_0 − phi_mt| = $d0 > $t0_tol. The link and the demand parquet are different " *
        "vintages; re-export it (python bbl_sleep_link.py) from the fit the parquet was built with.")
    isB = BitVector(Bool.(coalesce.(df.is_B, false)))
    nat = _national_phi(df, tid, mca, isB, phi0)
    for n in notes; log_status("  [PHI] $n"); end
    return PhiPathInputs(L, S0, dev, rho, krf, phi0, d0, String(transitions_path), rho_field, isB, nat)
end

"""
    _national_phi(df, tid, mca, isB, phi0) -> NationalPhi

The sleep side's national aggregation (sleep_est_single._calculate_phis) over the parquet's rows:
market cells = the distinct (mca_code, time_id) pairs in first-occurrence order, each counted once;
a cell's φ̄ is the mean of its rows' link values and its weight M the mean of their pop_total, a
missing pop_total counting 0 (the sleep side's `pop_total.fillna(0)`). Φ_q(0) is taken over the
cells from φ_0 by `_national_agg`, the same arithmetic phi_at! does at every t, so the path's
change is exactly 0.0 at t = 0.
"""
function _national_phi(df::DataFrame, tid::Vector{String}, mca::Vector{String},
                       isB::BitVector, phi0::Vector{Float64})
    N = length(phi0)
    pop = Float64.(coalesce.(df.pop_total, NaN))
    pht = Float64.(coalesce.(df.phi_t, NaN))
    qidx = Dict{String,Int}(); quarters = String[]; phi_t = Float64[]
    cidx = Dict{Tuple{String,String},Int}()
    row_cell = Vector{Int}(undef, N); cell_n = Int[]; wsum = Float64[]; cell_q = Int[]
    @inbounds for i in 1:N
        q = get(qidx, tid[i], 0)
        if q == 0
            push!(quarters, tid[i]); push!(phi_t, pht[i]); q = length(quarters); qidx[tid[i]] = q
        elseif !isequal(pht[i], phi_t[q])
            error("the parquet's phi_t is not one value per quarter ($(tid[i]): $(pht[i]) vs $(phi_t[q]))")
        end
        c = get(cidx, (mca[i], tid[i]), 0)
        if c == 0
            push!(cell_n, 0); push!(wsum, 0.0); push!(cell_q, q)
            c = length(cell_n); cidx[(mca[i], tid[i])] = c
        end
        row_cell[i] = c
        cell_n[c] += 1
        wsum[c] += isfinite(pop[i]) ? pop[i] : 0.0            # pop_total.fillna(0)
    end
    cell_w = wsum ./ cell_n                                    # the cell's mean pop_total
    nq = length(quarters)
    den = zeros(nq); for c in eachindex(cell_w); den[cell_q[c]] += cell_w[c]; end
    row_q = zeros(Int, N)
    @inbounds for i in 1:N
        isB[i] && continue
        q = qidx[tid[i]]
        den[q] > 0.0 || error("D row $i (quarter $(tid[i])): the quarter's market cells carry no population weight")
        isfinite(phi_t[q]) || error("D row $i (quarter $(tid[i])): the parquet's phi_t is not finite")
        row_q[i] = q
    end
    tmp = NationalPhi(quarters, row_cell, cell_n, cell_w, cell_q, row_q, phi_t, zeros(nq))
    csum = zeros(length(cell_n))
    @inbounds for i in 1:N; csum[row_cell[i]] += phi0[i]; end
    agg0 = _national_agg(tmp, csum)
    return NationalPhi(quarters, row_cell, cell_n, cell_w, cell_q, row_q, phi_t, agg0)
end

"""
    _national_agg(nat, csum) -> Φ per quarter

Σ_m M_m φ̄_m / Σ_m M_m over each quarter's cells in cell order, φ̄_m = csum[m] / (its row count):
`csum` holds each cell's sum of its rows' link values, accumulated in row order. NaN for a quarter
whose weights sum to 0 (no D row reads one: `_national_phi` refuses it).
"""
function _national_agg(nat::NationalPhi, csum::AbstractVector{Float64})
    nq = length(nat.quarters)
    num = zeros(nq); den = zeros(nq)
    @inbounds for m in eachindex(nat.cell_n)
        q = nat.cell_q[m]
        num[q] += nat.cell_w[m] * (csum[m] / nat.cell_n[m]); den[q] += nat.cell_w[m]
    end
    return [den[q] > 0.0 ? num[q] / den[q] : NaN for q in 1:nq]
end

# ==========================================================================
# The path
# ==========================================================================
# The index of row `i` at horizon `t`, summed in the estimator's order, each state shifted in
# place: a zero shift (t = 0, ρ = 1, the Selic held) leaves it bit-identical to φ_0's.
@inline function _index_at(inp::PhiPathInputs, i::Int, t::Int, f::Vector{Float64}, move_rf::Bool,
                           rf_h0, fwd, row_curve)
    L = inp.link; kr = inp.k_rf
    v = 0.0
    @inbounds for k in 1:size(inp.S0, 2)
        s = inp.S0[i, k]
        if k == kr
            move_rf && (s = _selic_state(L, t == 1 ? rf_h0[i] : fwd[row_curve[i], t - 1]))
        elseif f[k] != 0.0
            s = s + f[k] * inp.dev[i, k]
        end
        v += L.coef[k] * s
    end
    return v
end

"""
    phi_at!(out, inp, t; rf_h0, fwd, row_curve, selic=:curve, hi=PHI_SIM_MAX) -> out

φ of every row at horizon `t` (t = 0 is the launch quarter), clamped to [0, `hi`]: a B row's own
link path, a D row's national path (module docstring). The Selic state at t ≥ 1 is `rf_h0[i]`
for t = 1 (h = 0) and `fwd[row_curve[i], t−1]` after that; `selic = :held` keeps it at its launch
value instead (the identity check). `national`, when given (length = the launch quarters, in
`inp.nat.quarters` order), receives the national φ_t path at `t` of every quarter, clamped as the
D rows are: the value a D row of that quarter takes (bbl_phi_path_summary.jl reports it).
"""
function phi_at!(out::AbstractVector{Float64}, inp::PhiPathInputs, t::Int;
                 rf_h0::Union{Nothing,AbstractVector{Float64}}=nothing,
                 fwd::Union{Nothing,AbstractMatrix{Float64}}=nothing,
                 row_curve::Union{Nothing,AbstractVector{Int}}=nothing,
                 selic::Symbol=:curve, hi::Float64=PHI_SIM_MAX,
                 national::Union{Nothing,AbstractVector{Float64}}=nothing)
    L = inp.link; N, K = size(inp.S0)
    selic in (:curve, :held) || error("selic must be :curve or :held (got $selic)")
    f = [inp.rho[k] == 1.0 ? 0.0 : inp.rho[k]^t - 1.0 for k in 1:K]   # 0 at t = 0 and ρ = 1
    move_rf = t >= 1 && selic == :curve
    if move_rf
        (rf_h0 !== nothing && length(rf_h0) == N) || error("phi_at!: rf_h0 must be a length-N vector")
        if t >= 2
            (fwd !== nothing && row_curve !== nothing && length(row_curve) == N &&
             size(fwd, 2) >= t - 1) || error("phi_at!: fwd/row_curve do not reach h = $(t - 1)")
        end
    end
    nat = inp.nat
    national === nothing || length(national) == length(nat.quarters) ||
        error("phi_at!: `national` must have one entry per launch quarter ($(length(nat.quarters)))")
    has_D = !all(inp.isB)
    agg_on = has_D || national !== nothing
    csum = agg_on ? zeros(length(nat.cell_n)) : Float64[]
    @inbounds for i in 1:N
        g = sleep_link_phi(L, _index_at(inp, i, t, f, move_rf, rf_h0, fwd, row_curve))   # in [0, 1]
        agg_on && (csum[nat.row_cell[i]] += g)                    # every row enters its market cell
        inp.isB[i] && (out[i] = clamp(g, 0.0, hi))                # D rows: the national path below
    end
    if agg_on
        agg = _national_agg(nat, csum)
        # the anchored national path of quarter q: the stored level moved by the aggregate's change
        natq(q) = clamp(nat.phi_t[q] + (agg[q] - nat.agg0[q]), 0.0, hi)
        @inbounds for i in 1:N
            inp.isB[i] && continue
            out[i] = natq(nat.row_q[i])
        end
        national === nothing || (national .= natq.(eachindex(nat.quarters)))
    end
    return out
end

"""
    phi_launch(inp) -> Vector{Float64}

Each row's φ at t = 0 as the path starts from it, clamped to [0, PHI_SIM_MAX]: φ_0 (= phi_mt) for
a B row, the parquet's stored national φ_t for a D row.
"""
phi_launch(inp::PhiPathInputs) =
    [inp.isB[i] ? clamp(inp.phi0[i], 0.0, PHI_SIM_MAX) :
                  clamp(inp.nat.phi_t[inp.nat.row_q[i]], 0.0, PHI_SIM_MAX) for i in eachindex(inp.isB)]

"""
    build_phi_path(inp, T; rf_h0, fwd, row_curve, selic=:curve) -> Matrix (N × T)

Column t is φ at horizon t, t = 1…T: what `simulate_deposits(...; phi_path)` uses in period t.
"""
function build_phi_path(inp::PhiPathInputs, T::Int;
                        rf_h0::Union{Nothing,AbstractVector{Float64}}=nothing,
                        fwd::Union{Nothing,AbstractMatrix{Float64}}=nothing,
                        row_curve::Union{Nothing,AbstractVector{Int}}=nothing,
                        selic::Symbol=:curve)
    N = size(inp.S0, 1)
    Φ = Matrix{Float64}(undef, N, T)
    Threads.@threads for t in 1:T
        phi_at!(view(Φ, :, t), inp, t; rf_h0=rf_h0, fwd=fwd, row_curve=row_curve, selic=selic)
    end
    return Φ
end

"""
    selic_h0_rows(row_curve, starts, anchor) -> Vector{Float64}

Each row's h = 0 rate: the realised rate of its launch quarter, the h = 0 row of its vintage
(`anchor`, as `_load_rf_vintages` returns it). A start without an h = 0 row is an error — the
lagged accrual and the lagged Selic state both need it at t = 1.
"""
function selic_h0_rows(row_curve::Vector{Int}, starts::Vector{String}, anchor::AbstractDict)
    miss = [q for q in starts if !haskey(anchor, q)]
    isempty(miss) || error("forward_rf_vintages: no h=0 row for $(join(miss, ", ")). The lagged " *
                           "carry and the lagged Selic state need each launch quarter's own rate.")
    a = [Float64(anchor[q]) for q in starts]
    return [a[c] for c in row_curve]
end

"""
    phi_path_report(Φ, inp; w=nothing, ts=(1, 4, 20, 40, 100, T)) -> nothing

Log the launch φ and, at a few horizons and separately for B and D rows, the mean (row-weighted,
and `w`-weighted when given), p10 / p90, min / max, and the share of rows at the PHI_SIM_MAX
clamp. One block per job.
"""
function phi_path_report(Φ::AbstractMatrix{Float64}, inp::PhiPathInputs;
                         w::Union{Nothing,AbstractVector{Float64}}=nothing, ts=nothing)
    N, T = size(Φ)
    tsel = ts === nothing ? unique(filter(t -> 1 <= t <= T, [1, 4, 20, 40, 100, T])) : collect(ts)
    q(x, p) = quantile(x, p)
    wm(x, m) = w === nothing ? NaN : sum(w[m] .* x) / sum(w[m])
    p0 = phi_launch(inp)
    nat = inp.nat
    gap = filter(isfinite, [abs(nat.agg0[k] - nat.phi_t[k]) for k in eachindex(nat.quarters)])
    isempty(gap) && (gap = [0.0])
    log_status("  [PHI] evolving φ_t: link $(basename(inp.link.path)) (E$(inp.link.estim), " *
               "sha256 $(first(inp.link.sha256, 12))) | ρ ← $(basename(inp.transitions)) " *
               "($(inp.rho_field)): " * join(["$(inp.link.cols[k])=$(round(inp.rho[k], digits=4))"
                                               for k in eachindex(inp.rho) if inp.link.role[k] == :market], ", "))
    log_status("  [PHI] t=0 reproduces phi_mt: max|Δ| = $(inp.t0_maxdiff) | B rows $(count(inp.isB)) " *
               "on their own link path, D rows $(count(!, inp.isB)) on the national φ_t path " *
               "($(length(nat.cell_n)) market cells over $(length(nat.quarters)) quarters, population " *
               "weights held at launch; the cells' own aggregate is off the stored φ_t by max " *
               "$(round(maximum(gap), sigdigits=3)), which is why the path anchors to the stored level)")
    for (lab, m) in (("B", inp.isB), ("D", .!inp.isB))
        any(m) || continue
        log_status("  [PHI] $lab   t     mean    dep-wtd     p10      p90      min      max   at $(PHI_SIM_MAX)")
        for t in vcat(0, tsel)
            c = t == 0 ? p0[m] : Φ[m, t]
            log_status("  [PHI] $lab " * lpad(t, 3) * "  " * join(lpad.(string.(round.(
                [mean(c), wm(c, m), q(c, 0.10), q(c, 0.90), minimum(c), maximum(c)], digits=5)), 8), " ") *
                "  " * lpad(string(round(100 * count(>=(PHI_SIM_MAX), c) / length(c), digits=2)), 6) * "%")
        end
    end
    mv = 0.0                                   # column by column: no N×T temporary
    for t in 1:T
        c = view(Φ, :, t)
        @inbounds for i in 1:N; d = abs(c[i] - p0[i]); d > mv && (mv = d); end
    end
    log_status("  [PHI] max |φ_t − φ_0| over the path: $(round(mv, sigdigits=4)) " *
               "(0 would mean φ never moves)")
    return nothing
end
