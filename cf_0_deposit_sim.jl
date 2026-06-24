"""
cf_0_deposit_sim.jl
===================
Foundation 0b: forward simulator for the deposit law of motion (eqs 8-B / 8-D),
built on the demand-evaluation context from cf_0_demand_eval.jl.

Law of motion (per jkmt, with sleeper persistence φ):

    Dep_t = (1 − φ_mt)·M_mt·s^Act_t(ρ_t)  +  φ_mt·(1 + r^dep_{t−1})·Dep_{t−1}
            └─────── active demand ───────┘    └──── sleeper retention ────┘

The active term uses the BLP active share s^Act recomputed at the period's spreads
(via `cf_shares_at`); the sleeper term accrues last period's stock at the depositor
deposit rate r^dep.

⚠ TWO DISTINCT RATE CONCEPTS — keep them separate:
  * DEMAND spread ρ (annualized, = spread_ann/100): the variable that enters utility
    and therefore the shares. `cf_shares_at` expects spreads in THESE units.
  * ACCRUAL deposit rate r^dep (QUARTERLY): what sleepers actually earn each quarter,
    r^dep_q = r^f_q − ρ_q. The retention factor is (1 + r^dep_q). Use quarterly
    columns (deposit_rate_qoq / risk_free_qoq / spread_qoq), NOT the annualized ρ.

OPEN MODELING KNOBS (confirm with author before headline runs — see
counterfactuals_plan.md "Open methodological choices"):
  * M_mt market size: proxied by d̄·Pop_mt (draft footnote). `dbar` defaults to 1.0;
    only the LEVEL of franchise value scales with d̄, not the φ̂-vs-φ=0 ratio.
  * r^f forward path: defaults to holding the last observed quarterly risk-free rate
    flat over the horizon; pass `rf_path` to use the BCB forward curve.
  * Dep₀ source: starting deposit stock column (see `load_sim_state` candidates).

This file defines reusable functions; running it as a script does a small smoke
simulation (only once data is present and you've authorized running).
"""

include(joinpath(@__DIR__, "cf_0_demand_eval.jl"))

using DataFrames, Statistics

# ==========================================================================
# Simulation state: per-observation vectors aligned with ctx.df rows
# ==========================================================================
struct DepositSimState
    phi      ::Vector{Float64}   # φ̂_mt sleeper share (∈[0,1))
    Dep0     ::Vector{Float64}   # starting deposit stock at t₀
    rdep_q   ::Vector{Float64}   # quarterly depositor rate r^dep_q (accrual)
    M        ::Vector{Float64}   # market size proxy d̄·Pop_mt
    dep_type ::Vector{Int}
    is_B     ::BitVector
    endog    ::BitVector         # true for k∈{4,5} (spreads are choice objects)
end

"""
    _first_present(df, candidates; default) -> (Vector, name)

Return the first column in `candidates` present in `df` (coalesced to Float64),
else a constant `default` vector. Lets the simulator tolerate naming differences
across demand-prep vintages without running anything to "discover" the schema.
"""
function _first_present(df::DataFrame, candidates::Vector{String}; default::Float64=NaN)
    for c in candidates
        c in names(df) && return (Float64.(coalesce.(df[!, c], default)), c)
    end
    return (fill(default, nrow(df)), "<none:default=$default>")
end

"""
    load_sim_state(ctx; dbar=1.0, sidecar=nothing, ...) -> DepositSimState

Assemble the simulation state from `ctx.df` (and optionally a `sidecar` DataFrame
merged 1:1 on the same row order, e.g. the demand-prep parquet that carries φ̂ and
Dep^Act if the BLP input parquet does not). Column names are resolved against
candidate lists; missing essentials raise a clear error.
"""
function load_sim_state(ctx::CFDemandCtx; dbar::Union{Float64,AbstractVector{<:Real}}=1.0,
                        sidecar::Union{Nothing,DataFrame}=nothing)
    df = sidecar === nothing ? ctx.df : hcat(ctx.df, sidecar; makeunique=true)

    phi, phi_c   = _first_present(df, ["phi_mt", "phi_local_mt", "phi_local", "phi"]; default=NaN)
    Dep0, dep_c  = _first_present(df, ["deposit_balance", "Dep", "Dep_total", "Dep_Act", "active_deposits"]; default=NaN)
    pop, pop_c   = _first_present(df, ["pop_total", "M_mt", "pop"]; default=NaN)
    # Quarterly accrual rate: prefer an explicit deposit rate; else r^f_q − ρ_q.
    rdep, rdep_c = _first_present(df, ["deposit_rate_qoq", "rdep_qoq", "r_dep_qoq"]; default=NaN)
    if all(isnan, rdep)
        rf_q, rf_c   = _first_present(df, ["risk_free_qoq", "risk_free_qoq_lag", "selic_qoq", "rf_qoq"]; default=NaN)
        sp_q, sp_c   = _first_present(df, ["spread_qoq", "spread_q"]; default=NaN)
        sp_q = sp_q ./ 1e4   # spread_qoq is BASIS POINTS -> per-quarter fraction
        rdep = rf_q .- sp_q
        rdep_c = "($rf_c − $sp_c)"
    end

    any(isnan, phi)  && error("Sleeper share φ not found (tried phi_mt/…). Pass a sidecar with φ̂.")
    any(isnan, Dep0) && error("Starting deposit stock not found. Pass a sidecar with the Dep column.")
    any(isnan, pop)  && error("Population/market-size proxy not found (tried pop_total/…).")
    any(isnan, rdep) && error("Quarterly deposit rate not resolvable (need deposit_rate_qoq or rf_qoq & spread_qoq).")

    log_status("  [SIM] φ←$phi_c  Dep₀←$dep_c  M←$(pop_c)·dbar($(dbar isa Number ? round(dbar,sigdigits=4) : "per-row"))  r^dep_q←$rdep_c")
    dep_type = Int.(coalesce.(df.deposit_type, 0))
    is_B     = BitVector(Bool.(coalesce.(df.is_B, false)))
    endog    = BitVector((dep_type .== 4) .| (dep_type .== 5))
    # Bound the quarterly deposit rate to an economically sensible, STABILITY-
    # guaranteeing range: r^dep in [0, 0.10] keeps the sleeper accrual β·φ̂·(1+r^dep)<1
    # (β=0.9, φ̂≤0.999 ⇒ need r^dep<0.11), so the franchise-value integral converges.
    # Values outside this are residual implicit-rate artifacts (≈0.3% of cells).
    return DepositSimState(clamp.(phi, 0.0, 0.999), max.(Dep0, 0.0),
                           clamp.(rdep, 0.0, 0.10),
                           dbar .* pop, dep_type, is_B, endog)
end

# ==========================================================================
# Forward simulation of the deposit path
# ==========================================================================
"""
    simulate_deposits(ctx, st; T=50, spreads_ann=nothing, rf_path_q=nothing,
                      phi_override=nothing) -> NamedTuple

Roll deposits forward `T` periods from `st.Dep0`. Returns:
  * `Dep`      :: Matrix (N × (T+1)), Dep[:,1] = Dep0, columns = t=0…T
  * `s_act`    :: Matrix (N × T), the active share used each period
  * `spreads`  :: the annualized spread path actually applied (N × T)

Arguments:
  * `spreads_ann` : annualized demand spreads (units of ρ=spread_ann/100). Either
    `nothing` (hold observed ρ̂ flat), a length-N vector (constant over t), or an
    N×T matrix (time-varying — e.g. a counterfactual policy). Only k∈{4,5} should
    differ from ρ̂ in equilibrium counterfactuals; regulated types pass through.
  * `rf_path_q`   : quarterly risk-free path, length-T vector (default: last
    observed r^f implied by st.rdep_q held flat — see note). Enters accrual via
    r^dep_q = r^f_q − ρ_q only if you choose to recompute it; by default the
    accrual uses `st.rdep_q` held constant (banks-believe-state-constant).
  * `phi_override`: replace φ (e.g. zeros(N) for the φ=0 counterfactual in CF1).

PERFORMANCE: if `spreads_ann` is constant over t, the active share is computed ONCE
(one forward pass) and reused — so a constant-spread sim is ~1 share evaluation,
not T. Time-varying spreads cost one `cf_shares_at` per period.
"""
function simulate_deposits(ctx::CFDemandCtx, st::DepositSimState;
                           T::Int=50,
                           spreads_ann::Union{Nothing,Vector{Float64},Matrix{Float64}}=nothing,
                           rf_path_q::Union{Nothing,Vector{Float64}}=nothing,
                           phi_override::Union{Nothing,Vector{Float64}}=nothing)
    N = nrow(ctx.df)
    φ = phi_override === nothing ? st.phi : phi_override
    length(φ) == N || error("phi length ≠ N")

    # Resolve the spread path into an N×T view and detect the constant-spread fast path.
    time_varying = spreads_ann isa Matrix
    base_spread  = spreads_ann === nothing ? ctx.rho_hat :
                   (spreads_ann isa Vector ? spreads_ann : spreads_ann[:, 1])

    # Accrual factor (1 + r^dep_q). Default: hold st.rdep_q constant. If a forward
    # r^f path is supplied, recompute r^dep_q = r^f_q − ρ_q with the period spread.
    Dep   = zeros(N, T + 1); Dep[:, 1] .= st.Dep0
    s_out = zeros(N, T)
    sp_out = zeros(N, T)

    s_const = time_varying ? nothing : cf_shares_at(ctx, base_spread)

    @inbounds for t in 1:T
        ρ_t = time_varying ? spreads_ann[:, t] : base_spread
        s_t = time_varying ? cf_shares_at(ctx, ρ_t) : s_const
        # Quarterly accrual: prefer recompute from forward r^f if provided.
        rdep_t = if rf_path_q === nothing
            st.rdep_q
        else
            rf_path_q[t] .- (ρ_t ./ 4.0)   # NOTE: ρ is annualized; quarterly ≈ ρ/4 (confirm convention)
        end
        accr = 1.0 .+ rdep_t
        Dep[:, t+1] .= (1.0 .- φ) .* st.M .* s_t .+ φ .* accr .* Dep[:, t]
        s_out[:, t] .= s_t
        sp_out[:, t] .= ρ_t
    end
    return (Dep=Dep, s_act=s_out, spreads=sp_out)
end

# ==========================================================================
# Validation helper: one-step-ahead reproduction of observed deposits
# ==========================================================================
"""
    validate_onestep(ctx, st) -> NamedTuple

Sanity check (NOT to be run until data is downloaded and the author authorizes):
one-step deposit update from Dep₀ at observed spreads should track the observed
next-period stock when φ̂ and Dep^Act are internally consistent (eq 8 is an identity
given the estimated active deposits). Reports the relative error distribution.
"""
function validate_onestep(ctx::CFDemandCtx, st::DepositSimState)
    sim = simulate_deposits(ctx, st; T=1)
    pred = sim.Dep[:, 2]
    obs, obs_c = _first_present(ctx.df, ["deposit_balance_next", "Dep_next"]; default=NaN)
    if all(isnan, obs)
        log_status("  [SIM] No next-period column to validate against ($obs_c); returning prediction only.")
        return (pred=pred, rel_err=fill(NaN, length(pred)))
    end
    rel = abs.(pred .- obs) ./ max.(abs.(obs), 1e-6)
    log_status("  [SIM] one-step rel-err: median=$(round(median(rel), sigdigits=3)) " *
               "p90=$(round(quantile(rel, 0.9), sigdigits=3))")
    return (pred=pred, rel_err=rel)
end

# ==========================================================================
# CLI smoke test (does NOT run unless invoked directly AND data is present)
# ==========================================================================
function main_cf_depositsim()
    a = _parse_cf_args()
    ctx = build_cf_context(a["estim"], a["spec"], a["stage"];
                           R=a["R"], seed=a["seed"], hpc=a["hpc"],
                           local_dir=a["local-dir"], suffix=a["suffix"])
    st = load_sim_state(ctx)
    sim = simulate_deposits(ctx, st; T=50)
    log_status("  [SIM] Dep₀ total=$(round(sum(st.Dep0), sigdigits=4)) → " *
               "Dep_T total=$(round(sum(sim.Dep[:, end]), sigdigits=4))")
    log_status("[DONE] cf_0_deposit_sim smoke")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_cf_depositsim()
end
