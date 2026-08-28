"""
cf5_passthrough.jl
===================
CF5 — monetary pass-through (SCAFFOLD, reuses the CF3 equilibrium engine).

Shock the forward risk-free (Selic) path by Δ, re-solve the deposit-spread equilibrium
(cf_3_equilibrium_spreads), and measure the pass-through to equilibrium spreads and deposit
volume:

    ∂ρ*/∂Selic  ≈  mean(σ*_shock − σ*_base) / Δ          (on choice types k∈{4,5})
    ∂Dep/∂Selic ≈  (ΣDep_shock − ΣDep_base) / Δ

Δ enters through the accrual r^dep_q = r^f_q − ρ^q (deposit dynamics) and the ψ4 funding
base; banks then re-optimize their k∈{4,5} spreads. This is a SCAFFOLD: it runs the machinery
at logit for validation; the credible headline needs the RC costs (cluster).

Local dev:
  julia --project=. cf5_passthrough.jl --estim 6 --spec 12 --stage logit --R 50 \\
      --time-filter 2025Q4 --n-markets 15 --selic-shock 0.01
"""

include(joinpath(@__DIR__, "cf3_equilibrium.jl"))

using Printf, Statistics

# Total deposits at a given equilibrium spread vector (last-period stock), reusing the sim.
function _total_deposits(P, σ; T, rf=P.rf)
    sim = simulate_deposits(P.ctx, P.st; T=T, spreads_ann=σ, rf_path_q=rf)
    return sum(@view sim.Dep[:, end])
end

"""CF5 compare: read the base + Selic-shocked equilibria (σ solved by the cluster Jacobi) and
report the pass-through, without re-solving. `P` is the base context; the shocked deposits use
the shocked r^f."""
function cf5_compare(a)
    shock = a["selic-shock"] == 0.0 ? 0.01 : a["selic-shock"]
    a0 = copy(a); a0["selic-shock"] = 0.0
    P = cf3_setup(a0)
    σ0 = _read_sigma(a["sigma-base"]); σ1 = _read_sigma(a["sigma-scn"])
    (length(σ0) == nrow(P.ctx.df) && length(σ1) == nrow(P.ctx.df)) || error("σ length ≠ N")
    Δq = (1.0 + shock)^0.25 - 1.0
    endog = P.st.endog
    dσ = (σ1[endog] .- σ0[endog]) ./ shock
    dep0 = _total_deposits(P, σ0; T=a["horizon"])
    dep1 = _total_deposits(P, σ1; T=a["horizon"], rf=P.rf .+ Δq)
    @printf("\n  === CF5 monetary pass-through (Selic +%.3g) ===\n", shock)
    @printf("  ∂ρ*/∂Selic on k∈{4,5}:  mean=%.4g  median=%.4g\n", mean(dσ), median(dσ))
    @printf("  ΣDep: base=%.4g  shock=%.4g  ∂Dep/∂Selic=%.4g\n", dep0, dep1, (dep1 - dep0) / shock)
    df = DataFrame(CodConglomeradoPrudencial=string.(P.ctx.df.CodConglomeradoPrudencial),
                   deposit_type=P.st.dep_type, endog=endog, sigma_base=σ0, sigma_shock=σ1)
    cf_dir = cf_out_dir(P.out_dir); mkpath(cf_dir)    # cluster: data/output/counterfactuals
    Parquet2.writefile(joinpath(cf_dir, "cf5_passthrough_$(P.tag).parquet"), df)
    log_status("  [CF5] wrote cf5_passthrough_$(P.tag).parquet")
end

function main_cf5()
    a = _parse_cf3_args()   # shared CF3 CLI (includes --selic-shock)
    a["compare"] && return cf5_compare(a)   # cluster: combine two solved equilibria
    shock = a["selic-shock"] == 0.0 ? 0.01 : a["selic-shock"]   # descriptive default
    scheme = Symbol(replace(a["fixed-point"], "-" => "_") == "gauss_seidel" ? :gauss_seidel : :jacobi)
    a0 = copy(a); a0["selic-shock"] = 0.0        # build the BASE r^f here; we apply the shock below
    P = cf3_setup(a0)
    Δq = (1.0 + shock)^0.25 - 1.0    # annual Selic shock → quarterly r^f increment
    log_status("  [CF5] Selic shock $(shock) (annual) → Δr^f_q=$(round(Δq, sigdigits=3)) | box [$(round(P.lo,sigdigits=3)),$(round(P.hi,sigdigits=3))]")

    solve(rf) = solve_equilibrium(P.ctx, P.st, P.Z, P.mq0, P.θc_B, P.θc_D, P.isB, P.firms;
                                  beta=a["beta"], T=a["horizon"], aret=P.aret, rf=rf,
                                  scheme=scheme, damping=a["damping"], tol=a["tol"],
                                  max_iter=a["max-iter"], lo=P.lo, hi=P.hi,
                                  ngrid=a["br-grid"], window=a["br-window"])
    log_status("  [CF5] solving BASE equilibrium…");   eq0 = solve(P.rf)
    log_status("  [CF5] solving SHOCKED equilibrium…"); eq1 = solve(P.rf .+ Δq)

    endog = P.st.endog
    dσ = (eq1.sigma[endog] .- eq0.sigma[endog]) ./ shock          # ∂ρ*/∂Selic
    dep0 = _total_deposits(P, eq0.sigma; T=a["horizon"])
    # Shocked accrual on the shocked leg, so ∂Dep/∂Selic includes the DIRECT accrual channel
    # (r^dep_q = r^f_q − ρ_q) and not only the re-priced-spread channel. Matches cf5_compare (:45);
    # dep0 stays on the base r^f. Without the `rf` override these were inconsistent between the two
    # CF5 entry points (in-process here vs the cluster `--compare` path).
    dep1 = _total_deposits(P, eq1.sigma; T=a["horizon"], rf=P.rf .+ Δq)
    @printf("\n  === CF5 monetary pass-through (Selic +%.3g) ===\n", shock)
    @printf("  ∂ρ*/∂Selic on k∈{4,5}:  mean=%.4g  median=%.4g  (spread units per unit Selic)\n",
            mean(dσ), median(dσ))
    @printf("  ΣDep: base=%.4g  shock=%.4g  ∂Dep/∂Selic=%.4g\n", dep0, dep1, (dep1 - dep0) / shock)

    df = DataFrame(CodConglomeradoPrudencial=string.(P.ctx.df.CodConglomeradoPrudencial),
                   deposit_type=P.st.dep_type, endog=endog,
                   sigma_base=eq0.sigma, sigma_shock=eq1.sigma)
    cf_dir = cf_out_dir(P.out_dir); mkpath(cf_dir)    # cluster: data/output/counterfactuals
    out_path = joinpath(cf_dir, "cf5_passthrough_$(P.tag).parquet")
    Parquet2.writefile(out_path, df)
    log_status("  [CF5] wrote $(basename(out_path))")
    log_status("[DONE] cf_5_passthrough (scaffold)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_cf5()
end
