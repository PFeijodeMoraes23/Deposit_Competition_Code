"""
cf6_merger.jl
==============
CF6 — merger simulation (SCAFFOLD, reuses the CF3 equilibrium engine).

Merge two conglomerates into a single decision-maker (relabel firm B's rows to firm A), so
the merged entity chooses its k∈{4,5} spreads to maximize JOINT franchise value — internalizing
the cross-elasticities between the partners' deposit products within shared markets. Re-solve
the CF3 equilibrium and compare the merged franchise value to the pre-merger sum V_A + V_B.

  V_j = ψ_j(σ*)′·[1,−ω,−γ′,−(1+ζ)]        (V_Main eq 16), at the re-solved equilibrium σ*.

SCAFFOLD: products stay distinct (demand unchanged); only the pricing decision is merged; no
post-merger cost synergies. Runs at logit for validation; the headline needs RC costs (cluster).

Local dev (auto-picks the two biggest choosers if --merge is omitted):
  julia --project=. cf6_merger.jl --estim 6 --spec 12 --stage logit --R 50 \\
      --time-filter 2025Q4 --n-markets 15 [--merge "CONGL_A,CONGL_B"]
"""

include(joinpath(@__DIR__, "cf3_equilibrium.jl"))

using Printf, Statistics

# Per-firm franchise value at a given equilibrium spread vector.
function _firm_values_at(P, σ, isB; beta, T)
    pf, _ = psi_under(P.ctx, P.st, P.Z, P.mq0, σ; beta=beta, T=T, asset_return_q=P.aret, rf_path_q=P.rf)
    return firm_values(pf, isB, P.θc_B, P.θc_D)
end

"""CF6 compare: read the base + merged equilibria (σ solved by the cluster Jacobi) and report the
merger's franchise-value effect. Requires --merge "A,B" (the pair the merged σ was solved for)."""
function cf6_compare(a)
    isempty(a["merge"]) && error("CF6 compare needs --merge \"firmA,firmB\" (the solved pair)")
    parts = strip.(split(a["merge"], ",")); length(parts) == 2 || error("--merge needs \"firmA,firmB\"")
    fA, fB = String(parts[1]), String(parts[2])
    a0 = copy(a); a0["merge"] = ""; P0 = cf3_setup(a0)          # base (unmerged)
    σ0 = _read_sigma(a["sigma-base"]); V0 = _firm_values_at(P0, σ0, P0.isB; beta=a["beta"], T=a["horizon"])
    P1 = cf3_setup(a)                                            # merged (a has --merge)
    σ1 = _read_sigma(a["sigma-scn"]); V1 = _firm_values_at(P1, σ1, P1.isB; beta=a["beta"], T=a["horizon"])
    iA = findfirst(==(fA), P0.firms); iB = findfirst(==(fB), P0.firms); iM = findfirst(==(fA), P1.firms)
    Vpre = V0[iA] + V0[iB]; Vpost = V1[iM]
    @printf("\n  === CF6 merger (%s + %s) ===\n", fA, fB)
    @printf("  pre-merger V_A+V_B = %.4g | post-merger V_(AB) = %.4g  (Δ = %.4g, %.2f%%)\n",
            Vpre, Vpost, Vpost - Vpre, 100*(Vpost - Vpre)/max(abs(Vpre), 1e-12))
    cf_dir = cf_out_dir(P1.out_dir); mkpath(cf_dir)   # cluster: data/output/counterfactuals
    Parquet2.writefile(joinpath(cf_dir, "cf6_merger_$(P1.tag).parquet"),
                       DataFrame(firm=P1.firms, is_B=collect(P1.isB), V_merged_eq=V1))
    log_status("  [CF6] wrote cf6_merger_$(P1.tag).parquet")
end

function main_cf6()
    a = _parse_cf3_args()
    a["compare"] && return cf6_compare(a)   # cluster: combine base + merged equilibria
    scheme = Symbol(replace(a["fixed-point"], "-" => "_") == "gauss_seidel" ? :gauss_seidel : :jacobi)
    a0 = copy(a); a0["merge"] = ""        # BASE context (unmerged); we relabel + re-solve below
    P = cf3_setup(a0)
    solve(firms, isB) = solve_equilibrium(P.ctx, P.st, P.Z, P.mq0, P.θc_B, P.θc_D, isB, firms;
                                          beta=a["beta"], T=a["horizon"], aret=P.aret, rf=P.rf,
                                          scheme=scheme, damping=a["damping"], tol=a["tol"],
                                          max_iter=a["max-iter"], lo=P.lo, hi=P.hi,
                                          ngrid=a["br-grid"], window=a["br-window"])

    key = string.(P.ctx.df.CodConglomeradoPrudencial)
    # Choose the merger pair: --merge "A,B", else auto-pick the two firms with the most choice rows.
    if !isempty(a["merge"])
        parts = strip.(split(a["merge"], ",")); length(parts) == 2 || error("--merge needs \"firmA,firmB\"")
        fA, fB = String(parts[1]), String(parts[2])
    else
        endog = P.st.endog
        cnt = Dict{String,Int}()
        for i in eachindex(key); endog[i] && (cnt[key[i]] = get(cnt, key[i], 0) + 1); end
        top = first.(sort(collect(cnt), by=x -> -x[2])[1:min(2, length(cnt))])
        length(top) == 2 || error("need ≥2 choice firms to auto-merge")
        fA, fB = top[1], top[2]
        log_status("  [CF6] no --merge given → auto-merging top-2 choosers: $fA + $fB")
    end
    (fA in key && fB in key) || error("merge firms not found: $fA / $fB")

    # Baseline equilibrium + per-firm values.
    log_status("  [CF6] solving BASE equilibrium…"); eq0 = solve(P.firms, P.isB)
    V0 = _firm_values_at(P, eq0.sigma, P.isB; beta=a["beta"], T=a["horizon"])

    # Merge B → A (relabel), recompute firm list, re-solve.
    P.ctx.df[!, :CodConglomeradoPrudencial] = [k == fB ? fA : k for k in key]
    _, firms1 = psi_under(P.ctx, P.st, P.Z, P.mq0, P.ctx.rho_hat; beta=a["beta"], T=a["horizon"],
                          asset_return_q=P.aret, rf_path_q=P.rf)
    isB1 = firm_is_B(P.ctx, firms1)
    log_status("  [CF6] solving MERGED equilibrium ($fB → $fA)…"); eq1 = solve(firms1, isB1)
    V1 = _firm_values_at(P, eq1.sigma, isB1; beta=a["beta"], T=a["horizon"])

    iA = findfirst(==(fA), P.firms); iB = findfirst(==(fB), P.firms); iM = findfirst(==(fA), firms1)
    Vpre = V0[iA] + V0[iB]; Vpost = V1[iM]
    @printf("\n  === CF6 merger (%s + %s) ===\n", fA, fB)
    @printf("  pre-merger  V_A+V_B = %.4g\n", Vpre)
    @printf("  post-merger V_(AB)  = %.4g   (Δ = %.4g, %.2f%%)\n",
            Vpost, Vpost - Vpre, 100*(Vpost - Vpre)/max(abs(Vpre), 1e-12))
    dσ = eq1.sigma[P.st.endog] .- eq0.sigma[P.st.endog]
    @printf("  merged-firm spread change vs base: mean Δσ on k∈{4,5} = %.4g\n", mean(dσ))

    df = DataFrame(firm=firms1, is_B=collect(isB1), V_merged_eq=V1)
    cf_dir = cf_out_dir(P.out_dir); mkpath(cf_dir)    # cluster: data/output/counterfactuals
    out_path = joinpath(cf_dir, "cf6_merger_$(P.tag).parquet")
    Parquet2.writefile(out_path, df)
    log_status("  [CF6] wrote $(basename(out_path))")
    log_status("[DONE] cf_6_merger (scaffold)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_cf6()
end
