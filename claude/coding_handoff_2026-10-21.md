# Handoff: open coding tasks (merged, 2026-10-21, updated after your replies of 18:53 and 19:20)

**To:** coding agent. **From:** main thread, for Pedro.

**Replies to:** your replies of 10-21, 18:53 and 19:20 (both deleted after this update; their results are kept under "Done"). They closed K1, K2, K3, K5, K6, K7, K10 and K11, scoped S6 and built the Tesouro Direto panel.

**Rule:**
- V_Full.tex, V_Friction.tex and the decks are edited by the author only. Deliver numbers, keys, tables and figures.
- Exhibits serve both papers and are written once into `Drafts/Deposit Competition` (SHARED_MEMORY.md).
- Raw files stay untouched.

## Pedro's decisions (10-21)

```
decisions[6]{item,decision}:
  config/paper.toml,"stays V_Friction, matching the deck's \PEQpaperpdf"
  K9,on hold
  S6 posted rates,"scoped, not built"
  Tesouro Direto (S8a),aggregated to MCAs (done)
  K5,old-rule comparison dropped (done)
  K6,diagnostic aligned to the panel's conglomerates (done)
```

## Basis

```
files[5]{file,role,saved}:
  Drafts/Deposit Competition/V_Friction.tex,"working paper; equation labels renamed to their printed numbers (eq:10 = V_Full eq:13, eq:11-B = V_Full eq:14-B)",10-21 18:47
  Drafts/Deposit Competition/paper_edits_V_Friction_restructure_2026-10-21.tex,"Pedro's pending edits (Parts C and D); B10 is applied, so eq:13 is the Pix instrument",10-21 19:17
  Drafts/Deposit Competition/tab_state_vector_map.tex,"exhibit for K10 (generator make_state_vector_table.py)",10-21
  Drafts/Deposit Competition/tab_xi_diagnostics.tex,exhibit for K10,—
  Drafts/Deposit Competition/tab_negative_active.tex,exhibit for K10,—
```

## Done (from your reply)

```
done[11]{id,result}:
  K1,"tab_state_vector_map.tex notes: no app:estimation:* reference, no semicolons, laws of motion stated without the forward simulation"
  K2,"branch ESTBAN reconciles with municipal ESTBAN (exact in 90-95% of cells; total gap 0.004-0.017%); CODMUN_IBGE in the files; 99.97% map to an MCA"
  K3,"desc.branch.* keys; share_existing_dtime 0.895; corr_dcheck_dtime -0.009 (+0.016 before 2022Q2, -0.022 after); HSBC->Bradesco 2016Q4 counted as a transfer"
  K5,"report_entry_paths_estban regenerates (18:33), old-rule comparison removed"
  K6,"0.8% gap = wholesale and investment banks excluded by design; panel/raw on the same conglomerates 0.9996-0.9998 from 2016"
  K7,"double-read fix in code since 10-16; takes effect at the next panel rebuild (steps 14, stage 4 and 39)"
  S6 scoping,"posted rules cover about half of type-5 balances and are almost all 100% of CDI, so they mostly confirm the fallback; the other half is uncovered"
  S8a,"Rout/tesouro_investors_mca_quarter.parquet: new investors and stock per 1,000, MCA x quarter 2016-2024; 99.69% of investors matched"
  K10,"tab_state_vector_map, tab_xi_diagnostics and tab_negative_active notes carry no \eqref or \ref; no number changed"
  K11,"34 institutions mandated at launch (51 = ever mandatory); make_pix_exposure.py now keeps the 34, exposure unchanged; non-mandated = 46 mid-size private and foreign banks, 3.5% of 2019 B deposits, no state or cooperative bank; 1 - exposure is zero in 98.7% of MCAs (74.3% population-weighted); credit cooperatives are not in ESTBAN"
  C4/C8 logit/C9,"as before"
```

## Open tasks

```
tasks[4]{id,agent,output,status}:
  K4,code,"C8 RC keys demand.t6.no5.* and demand.t6.d5only.*",after Pedro's cluster run
  K8,code,"entry paths T5: D entrant-quarters on the national median g",optional (Pedro to confirm)
  K9,code,"C8(c) sleepiness without type 5",on hold (Pedro)
  B1,bibliography,"open the Santander and Bradesco PDFs for Pedro, then two References.bib entries",ready
```

### B1. Two `.bib` entries for the sweep programs (bibliography agent)

**Why.** V_Friction 7.2 (block C3 of Pedro's edit file) cites the sweep programs that move balances within a bank without a decision:
- Santander made its checking-to-CDB sweep a default feature of the account in 2022.
- Bradesco's sweep requires the client to opt in.

There is no Google Scholar entry for documents of this kind, so build `@misc` entries in the style of the existing institutional ones (for example `bcb_resolucao_145_2021`).

**Inputs.** Everything is in `DATA_RAW/breakfast_followups_2026-10-21/S7_sweep_programs/`:
- `sweep_products.csv` (rows SANT-PF-4 and the Bradesco Invest Fácil rows);
- the source log for the URLs and capture dates;
- the evidence files in `raw/<bank>/`.

```
entries[2]{key,document,must show}:
  santander_condicoes_pf_2022,"Santander PF account general conditions, form 1614-506 (05/2022), clauses 80-82","the automatic investment (ContaMax / Investimento Automatico) is a default feature of the account, and it invests in CDBs"
  bradesco_investfacil_2015,"Bradesco Invest Facil Termo de Adesao, stamped Versao 09/2015","the client opts in by signing the adhesion term"
```

**First, open both PDFs for Pedro to read**, before writing anything:
- the Santander form 1614-506 (05/2022), at clauses 80 to 82;
- the Bradesco Invest Fácil Termo de Adesão (Versão 09/2015).

Give him each file's full path, and the page numbers of the clauses. Wait for his go-ahead before adding the entries.

**Fields:** institution as author, the document title as printed, year, form or version number in `note`, URL (the archived capture if it came from the Wayback Machine) and `urldate`.

**Flags.**
- The Santander form is a scan read by OCR. Check clauses 80 to 82 against the page image, and report the exact wording in Portuguese.
- If the form does not say the sweep is a default, or does not say it invests in CDBs, stop and tell Pedro. The sentence in C3 rests on both.
- The Itaú and BB rate tables come only from press articles. Do not add entries for them.

**Acceptance.**
- Both keys resolve in `../References.bib`.
- Tell Pedro, who then applies C3.

### K4. C8, random-coefficients demand without type 5

Unchanged. **Pedro runs first**, on the cluster. Upload `ESTIMATION_OUTPUT/CLUSTER_OUT/upload_type5_20261021.zip` and its `.sha256` in binary mode to `~/project_pi_mf2263/pf382/dep_comp`, then run:

```bash
cd ~/project_pi_mf2263/pf382/dep_comp && unzip -o upload_type5_20261021.zip && sha256sum -c --quiet upload_type5_20261021.sha256 && echo UPLOAD-OK
```
```bash
cd ~/project_pi_mf2263/pf382/dep_comp/scripts && sbatch --parsable -p day -t 03:00:00 -c 1 --mem=2G -J t5v_build -o logs/t5v_build_%j.out --wrap "T5_DIAG_ROOT=$HOME/scratch_pi_mf2263/pf382 bash blp_type5_variants.sh --build-only && T5_DIAG_ROOT=$HOME/scratch_pi_mf2263/pf382 bash blp_type5_variants.sh --submit-built"
```
```bash
cd ~/project_pi_mf2263/pf382/dep_comp/scripts && T5_DIAG_ROOT=$HOME/scratch_pi_mf2263/pf382 bash blp_type5_variants.sh --report
```

**Then:**
- Build `demand.t6.no5.{E3,E4}.*` and `demand.t6.d5only.{E3,E4}.*`.
- Report the price coefficient, the elasticities and the change in the sample against `tab:blp_demand_comparison_noseg_spec12`.

**Flag:** this changes the sample, not the model.

### K8 and K9

- **K8 (optional):** 31% of D entrant-quarters use the national median accrual g. Report whether a closer fallback exists, only if Pedro confirms it is still wanted.
- **K9 (on hold):** `sleep_est_e2.py:236` drops only type 3, hard-coded, so a subset needs a code change. Do nothing until Pedro decides.

## Answer on the column headers (K10 reply)

The headers of `tab_xi_diagnostics` and `tab_negative_active` may keep `\ref{estimation:single_idx}` and `\ref{estimation:single_idx_time}`. Both labels exist in V_Full and in V_Friction (Section 5.1), so they do not print ??.

## Flags to keep visible

- **Santander's sweep spans every K3 result.** It became a default account feature in 2022-05, and nothing is adjusted for it. The within-branch checking and time-deposit correlation turns from +0.016 to −0.022 at 2022Q2 for all banks, and is +0.056 after for Santander alone.
- **Pre-2016 levels stay doubled** in the production panel `cac0a0c7` until the rebuild. No estimation path uses them under the window rule.
- **The mandated exposure varies only across a few large markets** (K11). Keys for the K11 shares exist only if Pedro asks for them.
- **Tesouro Direto values are registrations, not holdings.** 80,903 MCA-quarters with no B-firm row have no panel population, so their per-1,000 values are missing.

## Acceptance

- Every new key resolves in `paper_numbers_macros.tex` with no red `??`.
- Notes and feasibility reports state what was checked, and where.
- Exhibits are written once into the paper folder.

## Open questions for Pedro

1. **C8 headline sample:** decide after K4.
2. **K8:** still wanted?
3. **Type 5:** the uncovered half of type-5 balances (Stone, Original, BMP, Cora, Dock, Bradesco, plus the gaps for Neon, C6 and Inter) may pay zero, which would put the spread near the CDI rather than near zero. Treat this as a flag in the paper, or ask scraping for those issuers' balance rules?
