# V_Main edit plan: prepaid spreads are a CDI fallback, plus related stale claims (2026-10-16)

**Basis.**
- `V_Main.tex`, saved 10-16 16:24.
- `paper_numbers_macros.tex`, 10-16 00:25.
  - It was not refreshed today: `make_paper_numbers.py` stops on the spec-12 column rename (`HausmanFull` → `RivalSpread`) until the sleep rerun.
  - No key used below changes its displayed value.
- `paper_equations_macros.tex`, 10-05 18:30.
- Inputs regenerated today:
  - `Compressed_*`, `Summary_*` and `polfunc_k5*` at 16:20: the prepaid spread is dropped, with one note;
  - `fig_market_structure_by_year.png` at 16:24: panels (a), (c) and (d) only, relettered (a)–(c).

**Test.**
- All 16 SEARCH strings occur exactly once in the 16:24 save.
- A scratch copy with every block applied compiles (pdflatex twice, biber, pdflatex twice) to 91 pages, like the original.
- It shows 0 errors, 0 overfull boxes, 0 undefined references or citations, and no `??`.

**Revision 16:30.**
- **V1 is applied.** Your L281 says it in your own words, and the old sentence survives only in the commented copy on L282, so the plan no longer touches it.
- **Replaced by new blocks:**
  - V2 (the facts paragraph) and V3–V4 (the figure note), for the three-panel figure;
  - V11, because Table 1 no longer shows the prepaid spread.
- **New:**
  - V16, the `tab:deposit_types` note, since Table A.4 no longer prints type 5;
  - V17, the figure caption.

**This round.**
- After the 10-15 panel rebuild, no institution-level prepaid (type 5) rate exists before 2025. `panel_ip_rates` takes payment-institution (IP) rates only from 2025.
- Inside 2016 to 2024, every institution therefore carries the same fallback rate in every quarter: the CDI. The type-5 spread is the Selic minus CDI gap, below 0.013 bp in absolute value, and not a bank's price.
- The generated tables now say this in their notes. The blocks below fix the prose that still reads the zero as an economic result (pass-through, Bertrand competition).
- The blocks also fix:
  - the start of type 5: the data starts in 2020Q2, and every exhibit now uses that;
  - two policy-function claims the current tables contradict;
  - the lag description of the policy regressors.
- No new `\pn` keys are needed. Every key used below already exists.

---

## Part I. `V_Main.tex`

### V2a. Section 3, four facts, what the figure shows (L305). Required.
The figure now shows market structure only.

```latex
% SEARCH
Figure \ref{fig:market_structure_by_year} tracks concentration, firm counts, and spreads.
% REPLACE
Figure \ref{fig:market_structure_by_year} tracks concentration and firm counts.
```

### V2b. Same paragraph, prepaid sentence (L305). Required.
Prepaid spreads are no longer in the figure, and they are zero by construction (already said at L281).

```latex
% SEARCH
as they enter, a decline partly mechanical since entry enlarges the pooled firm set. Prepaid-account spreads stay near zero for both firm types.
% REPLACE
as they enter, a decline partly mechanical since entry enlarges the pooled firm set.
```

### V17. Figure `fig:market_structure_by_year` caption (L311). Required.
The figure no longer shows pricing.

```latex
% SEARCH
\caption{Market Structure and Deposit Pricing by Year}
% REPLACE
\caption{Market Structure by Year}
```

### V3. Same figure, note (L315–319). Required.
- Panels (b), (e) and (f) of the old figure are gone, and the remaining panels are relettered (a)–(c).
- The time-deposit spreads the old panel (e) showed are in `tab:deposit_types` and Table A.4.
- The semicolon after panel (a) is replaced by a full stop.

```latex
% SEARCH
		Panel (a): HHI within MCA for B firms and nationally for B and D pooled; the
		national decline is partly mechanical, since D entry enlarges the pooled firm
		set. Panel (b): Selic rate. Panels (c)--(d): D firms are a national count, B firms a per-market mean,
		so the two are not on a common scale. Panel (e): spreads, annualized by compounding and in percentage points, ranging from \pn[1]{desc.a4.spread4.D.2016} ($\mathrm{D}$, 2016) to \pn[1]{desc.a4.spread4.B.2022} ($\mathrm{B}$, 2022). Panel (f): type-5 is undefined before 2020; B and D coincide
		at zero (full pass-through), so D is dashed to keep B visible. The local $\mathrm{B}$ index in panel (a) and panels (d) to (f) are deposit-weighted. Panels (b), (c), and the national index in panel (a) are not.
% REPLACE
		Panel (a): HHI within MCA for B firms and nationally for B and D pooled. The
		national decline is partly mechanical, since D entry enlarges the pooled firm
		set. Panels (b) and (c): D firms are a national count, B firms a per-market mean,
		so the two are not on a common scale. The local $\mathrm{B}$ index in panel (a) and panel (c) are deposit-weighted. Panel (b) and the national index in panel (a) are not.
```

### V16. `tab:deposit_types` note (L290). Required.
- Table A.4 now prints only type-4 spreads.
- The prepaid row of the generated table carries a dagger, and the table explains it in its own last line.
- Type 5 starts in 2020Q2.

```latex
% SEARCH
Types 4 and 5: means of the annual balance-weighted values of Table \ref{tab:Compressed_MarketStructure_by_Year_AB}, type 5 from 2020.
% REPLACE
Type 4: means of the annual balance-weighted values of Table \ref{tab:Compressed_MarketStructure_by_Year_AB}. Type 5, from 2020Q2: the CDI fallback, marked $\dagger$.
```

### V5. Section 5.3, cost estimation, time deposits (L876). Required.
In `polfunc_k4.tex` the risk-free rate is significant for B firms (1.783***). For D firms it is not significant in either column. Only its square carries one star, in the national-demographics column.

```latex
% SEARCH
responds to bank characteristics and macroeconomic variables for both firm types, with several significant coefficients.
% REPLACE
responds to bank characteristics for both firm types, with several significant coefficients. The risk-free rate is strongly significant for $\mathrm{B}$ firms, while for $\mathrm{D}$ firms only its square is, at the 10 percent level in one specification.
```

### V6. Section 5.3, prepaid accounts (L876). Required.
- `\pn[2]{polfunc.k5.D_optA.mean_spread}` now prints 0.00.
- In the D column without demographics, six regressors besides the risk-free rate carry stars.
- The regression describes the Selic minus CDI gap, so no pricing reading is supported.

```latex
% SEARCH
Prepaid-account spreads $(k=5)$ stay close to zero for both firm types, essentially zero for $\mathrm{B}$ firms and \pn[2]{polfunc.k5.D_optA.mean_spread} basis points for $\mathrm{D}$ firms. For $\mathrm{B}$ firms the significant coefficients are small. For $\mathrm{D}$ firms only the risk-free rate and its square are significant.
% REPLACE
Prepaid-account spreads $(k=5)$ are zero by construction. No institution-level prepaid rate is observed over the sample, so the regressand is the gap between the Selic and CDI rates, a function of the quarter alone (Appendix \ref{app:data:panel}). Its coefficients describe that gap rather than pricing, and we do not interpret them.
```

### V7. Section 5.3, fit by firm type (L878). Required.
The k=5 $R^2$ is now \pn[2]{polfunc.k5.B.r2} for B firms and \pn{polfunc.k5.D_r2.lo} to \pn{polfunc.k5.D_r2.hi} for D firms. The contrast between firm types is mostly gone, and the prepaid fit says nothing about pricing.

```latex
% SEARCH
The fit differs by firm type rather than by deposit type. The $R^2$ is \pn[2]{polfunc.k4.B.r2} for $\mathrm{B}$ firms on time deposits and \pn[2]{polfunc.k5.B.r2} on prepaid accounts, but only \pn{polfunc.k4.D_r2.lo} to \pn{polfunc.k4.D_r2.hi} and \pn{polfunc.k5.D_r2.lo} to \pn{polfunc.k5.D_r2.hi} for $\mathrm{D}$ firms.
% REPLACE
On time deposits, the $R^2$ is \pn[2]{polfunc.k4.B.r2} for $\mathrm{B}$ firms but only \pn{polfunc.k4.D_r2.lo} to \pn{polfunc.k4.D_r2.hi} for $\mathrm{D}$ firms.
```

### V8. Section 5.3, Bertrand reading (L878). Required.
The near-zero spread is the fallback, so it is not evidence of competition.

```latex
% SEARCH
Prepaid spreads within a few basis points of zero are consistent with Bertrand competition in prepaid accounts, which drives spreads toward zero, so that the deposit rate is close to the risk-free rate. The competition pattern remains to be verified.
% REPLACE
The prepaid policy only tracks the gap between the Selic and CDI rates, so its fit carries no information on competition in prepaid accounts.
```

### V9. Appendix, panel formation, prepaid rate (L1203). Required.
- Prepaid payment accounts exist since 2013, so "a new phenomenon" is not accurate.
- The balance is reported from 2020Q2.

```latex
% SEARCH
\item Prepaid accounts are a new phenomenon, the data does not exist for them for before 2020. Afterwards, the endogenous rate is computed using a cascading fallback logic:
% REPLACE
\item Prepaid balances are reported from 2020Q2 onward. Their endogenous rate follows a cascading fallback logic:
```

### V10. Same list, second fallback (L1207). Required.
States which branch the sample actually uses.

```latex
% SEARCH
\item \textit{Second Fallback}: baseline CDI rate.
% REPLACE
\item \textit{Second Fallback}: baseline CDI rate. IP implicit rates are available only from 2025, after the sample ends. Within 2016 to 2024 every institution therefore receives this fallback, and the prepaid spread is the gap between the Selic and CDI rates.
```

### V11. Appendix, bank-type cross-section (L1228). Required.
- Type 5 starts in 2020Q2.
- Table 1 no longer shows the prepaid spread.
- The block also fixes "significant smaller".

```latex
% SEARCH
Notice that type 5 deposits and spreads have a significant smaller number of observations since this category was not available before 2020.
% REPLACE
Type 5 deposits have fewer observations because prepaid balances are reported only from 2020Q2. Type 5 spreads are not shown, since they equal the gap between the Selic and CDI rates by construction.
```

### V12. Appendix, D firms before and after Pix (L1242). Required.
- Type 5 is reported in two quarters before Pix.
- Log Deposits in the table sums types 1, 2, 4 and 5, so "only type 4" misdescribes it.

```latex
% SEARCH
Since Type 5 deposits were not available before that, we report only measures for deposit volume and spreads for Type 4.
% REPLACE
Prepaid balances are reported only from 2020Q2, two quarters before Pix, and prepaid spreads are zero by construction. We therefore report the volume and spread of time deposits (type 4) separately, next to total deposits.
```

### V13. Appendix, policy functions, type-5 regressand (L1535). Recommended.

```latex
% SEARCH
in percentage points for type 4 and in basis points for type 5.
% REPLACE
in percentage points for type 4 and in basis points for type 5. Because no institution-level prepaid rate is observed, the type-5 regressand is the gap between the Selic and CDI rates.
```

### V14. Same paragraph, lags (L1535). Recommended.
The table note says borrowings and onlendings are measured at $t$ (`panel_loo_instruments.py`: `emprestimos_repasses / total_assets_lag`).

```latex
% SEARCH
Accounting variables are lagged, winsorized
% REPLACE
Accounting variables are lagged one quarter, except borrowings and onlendings, which are measured at $t$ and divided by assets at $t-1$. They are winsorized
```

### V15. Same paragraph, the constant (L1535). Optional.
Dummies are not centered, so the constant fixes them at their reference category (`bbl_polfunc.py`, centering block).

```latex
% SEARCH
so the constant is the predicted spread at the average state.
% REPLACE
so the constant is the predicted spread at the average state with every indicator at its reference category.
```

---

## Items for other agents

1. **Model (main thread / BBL).** The fitted k=5 policy feeds the BBL forward simulation (`polfunc_results.pkl` → `bbl_fwd_sim.jl`). Within the sample, that policy is the Selic minus CDI gap. Whether k=5 stays in the cost stage is a modeling decision for you; nothing has been changed.
2. **Tables (done today).** The CDI-fallback sentence and the 2020Q2 start are now in the notes of:
   - `Compressed_*`, `Summary_*`, `polfunc_k5*`;
   - `tab_slide_deposit_types`, `tab_motiv_tightening`.

   The wording comes from one helper, `utils/prepaid.py`, and is printed only while the panel shows zero cross-institution dispersion in the type-5 rate.
