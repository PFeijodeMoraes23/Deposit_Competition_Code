# Main thread and IMPR: the one channel

**This file replaced `MAIN_REPLY_2026-11-12.md` and `IMPR_REPLY_2026-11-11a.md`** (Pedro, 11-12: one document for the two sessions). Both were deleted on 11-12 at 22:09 on Pedro's confirmation, with `impr_master_brief_2026-11-05.md`.

**Built** 11-12 20:26 by the main thread from the two files as they stood (`MAIN_REPLY` md5 `d83ab46e`, `IMPR_REPLY` md5 `136d0aad`). IMPR's text is carried over word for word where marked. IMPR's backup of its earlier sections is unchanged: `scratchpad\reply_backup_1112\IMPR_REPLY_2026-11-11a_until_2026-11-12_1950.md`.

## How to use this file

1. **Part A is the state.** It is edited in place. Each section names its owner, and only the owner edits it. The other session asks for a change through a log entry.
2. **Part B is the log,** newest first. Every message between the two sessions is one entry: `### 11-12 HH:MM MAIN: title` or `### 11-12 HH:MM IMPR: title`. An entry that changes the state also updates Part A in the same write.
3. **Stamps are the wall clock.** Run `date` before stamping. The main thread's stamps earlier this evening ran about fifteen minutes ahead.
4. **Two writers, one file.** Read the file immediately before writing. Insert text. Never write the whole file back from an older copy. Never edit the other session's entries or sections.
5. **Keep it short.** When a log entry is fully absorbed into Part A, its author may cut it to one line.
6. **Pedro's rules bind both sessions:** no cluster job, no production rerun, no download, and no edit of a pipeline script without his go. Agents plan first. Pedro edits the paper himself.

---

# Part A. The state

## A1. Pedro's decisions in force (owner: main thread)

1. **The main frictions of Section 7 are sleepiness (the baseline), attention, and awareness.**
2. **The switching cost is read from the Pix launch, as a difference.** The launch shifts the index of money that has a bank to leave ($\gamma_{P}$) and the index of new money and openings ($\gamma_{P\varnothing}$). The second is what Pix did to awareness. The difference is what it did to the switching cost. No level is estimated.
3. **The tax clock is one parameter,** $\gamma_C$, on the time-deposit share of the source bank's stock.
4. **The net-flow equation is not an estimator in the paper.** Its appendix and the simulation appendix are in the switching-cost note. Also out of the paper: the price-coefficient check, the simulation tables, the pools by type, the registry share, and the two propositions.
5. **Notation:** $\gamma_C$ (was $\kappa^{c}$). The availability index has no tilde: $A_{j\mid j^\prime mt}$ for money with a source and $A_{j\mid\varnothing mt}$ for new money.
6. **The market size is rebuilt from measured new money (20:10).** See A3. Production keeps the old market size until Pedro's go on Step 2, and no number that depends on it is promoted.
7. **The D-market double count is fixed inside the new construction** (Pedro to IMPR, 20:16). The production rerun waits for him.
8. **The three proposals of the evening:** (a) runs at $c$ = 0.80, 0.90, and 0.95. (b) is closed by IMPR's count. (c) is dropped.
9. **The cap at 0.999** is no longer a separate decision. It is one of the two caps of Step 1.
10. **Fees in demand (20:32):** the tables and the plan are delivered. Pedro accepts the main thread's positions on the eight decisions (log, 20:27). **The variant run is a go, once, on the frame that Step 2 produces,** in the sandbox only. It does not run on today's frame.
11. **The sixteen BCB series** (A3): the download is a go once Pedro accepts IMPR's plan for the script. Plan first, then run.
12. **The BCB is not asked for more registry data.**
13. **The awake pool counts the money that left the truncated cells (20:32).** A cell's sleepy share cannot exceed the share of its money that stayed. The estimate of the sleepy share is kept and binds from above. The formula is in A3.
14. **Currency held by the public counts in the outside part (20:41).** The broad total is then M4 with nothing subtracted, subject to what the methodological note says about holders.
15. **The descriptive fee keys are a go (20:41):** coverage, the range of the listed price at the five largest, and the change rate, under the names of the fees plan (its section 5.5). Not the keys of a logit run. The usual rule holds: the diff outline of `make_paper_numbers.py` first, then the edit, then the check.
16. **Fix the NATIONAL-row bug in the national sleepy share now (21:22). Pedro: "FIX THE BUG. THIS IS NECESSARY."** The code edit and its local verification are a go, as for the D-market fix. The copy of the D-market bug in the switch-cost prep is fixed in the same pass. Nothing goes to Bouchet.
17. **One production rerun, after Step 2,** with both fixes in it.
18. **The demand model has the fees (21:22).** The fee is part of the demand specification of the paper, in the logit and in the random-coefficients stage, from that rerun on. It is no longer a side variant.
19. **Booking moves inside a bank:** the local pools are reported with and without netting, and the netting is symmetric (the pool and the active balances together). Pedro chooses after Step 1.
20. **The download plan is accepted (21:54).** Run `scrape_bcb_monetary_aggregates.py` and its validation, then the three small edits (the path constant, the README line, the source-log row).
21. **Firms stay in (21:54).** The market is every holder of the non-financial sector, since no aggregate separates persons from firms. The persons' share from the FGC census is reported beside new money.
22. **The universe of wealth (21:59). The target is the public's universe:** the holdings of the non-financial sector, with funds as the outside good. **The method is decided after the diagnostic** of the 21:54 entry: fund money is taken out where it is booked if it is concentrated, and a uniform factor by type and quarter is used only if it is diffuse. In either case the factor applies to stocks and (11-B) is recomputed. **Judicial deposits are removed by bank in the base version.**
23. **The first stage of the stakes is clustered by state (22:09).** The paper quotes the state-clustered statistics. The keys are a go.
24. **Instrumented Pix use replaces the launch indicator in the availability index (22:09).** See A2.
25. **The state-vector table loses the block "Collected, not in the state vector" of Panel A (22:16).** Panel B stays. The change to `make_state_vector_table.py` is a go, with the diff outline posted first.

## A2. The model of Section 7 (owner: main thread)

A depositor whose money sits at bank $j^\prime$ treats another bank $j$ as available with probability

$$A_{j\mid j^\prime mt} = \Lambda\left(\gamma_0 + \gamma_1\,\mathrm{adv}_{jt} + \gamma_2\,\mathrm{branches}_{jm,t-1} + \gamma_{P}\,\mathrm{Pix}_{mt} - \gamma_C\,\mathrm{TD}_{j^\prime m,t-1}\right),$$

and her own bank with probability one.

- New money has no source, so its index is awareness alone: $A_{j\mid\varnothing mt} = \Lambda\left(\gamma_\varnothing + \gamma_1\,\mathrm{adv}_{jt} + \gamma_2\,\mathrm{branches}_{jm,t-1} + \gamma_{P\varnothing}\,\mathrm{Pix}_{mt}\right)$. The opening rate uses this index too.
- $\mathrm{Pix}_{mt}$ is the market's use of Pix (individuals who send transfers, per inhabitant), relative to its mean across markets in the quarter, and instrumented by the three exposures of 2019. The launch indicator is gone. Pedro confirmed the centering at 22:14.
- The change in the switching cost with the use of Pix is $\gamma_{P} - \gamma_{P\varnothing}$.
- Utility is the paper's (1), with no cost term.
- Attention is $\phi_{mt} = G\left(\boldsymbol{\Upsilon}^\prime\boldsymbol{S}_{mt} - \tau_0\,\mathrm{Selic}_{t-1} - \tau_1\,\mathrm{stakes}_{mt}\right)$, with $\boldsymbol{S}_{mt}$ without the lagged Selic rate. $\tau_0$ is the coefficient already estimated.

**The first-pass estimating form** (main thread's proposal. IMPR verified the algebra under three conditions).

$$\ln s^{\mathrm{Act}}_{jkmt} - \ln s^{\mathrm{Act}}_{\varnothing mt} \approx \delta_{jkmt} + \ln W_{jmt}, \qquad W_{jmt} = b_{jmt} + \sum_{j^\prime \neq j} b_{j^\prime mt}\,A_{j\mid j^\prime mt} + b_{\varnothing mt}\,A_{j\mid\varnothing mt},$$

where $b$ are the pools' shares of the market size: the bank's own, each rival's, and new money. For each candidate of the index, the rest is the existing logit with its instruments.

## A3. The market size from measured new money (owner: main thread)

**Pedro's decision, 20:10.**

1. **New money is measured, nationally and by quarter:** $N_t = \mathrm{Wealth}_t - \mathrm{Carry}_t$, with $\mathrm{Carry}_t$ last quarter's stock grown at its own rate of return. No sleepy share enters $N_t$.
2. **The awake pool is** $P_t = \left(1 - \phi_t\right)\mathrm{Carry}_t + N_t$. New money has no account to carry forward, so all of it is awake.
3. **The market is the public's liquid financial wealth,** not deposits alone. On deposits alone the pool equals the sum of the active balances, and the outside share is zero by construction.
   - **Inside:** the four deposit types at B and D institutions.
   - **Outside:** fund quotas, federal securities held by the public, repos, and bank paper outside the four types.
   - **Holders (21:59):** the non-financial sector, persons and firms. Funds, banks, and non-residents are not depositors here. A CDB held by a fund is part of the fund quota, on the outside.
   - **The panel is brought to that universe** by the method of A1.22. Judicial deposits are out of the inside part, by bank.

**The pool with the truncation (Pedro's decision, 20:32, on the main thread's proposal of 20:27).** Equation (11-B) sets a cell's active balance to zero when $D_t < \hat{\phi}\,\mathrm{Carry}$, which treats all of $D_t$ as money that stayed. The money that did not stay is $\max\{(1-\hat{\phi})\,\mathrm{Carry},\ \mathrm{Carry} - D_t\}$, and in the model money that does not carry forward is awake. The pool that matches (11-B) is

$$P_t = \sum_{\text{cells}}\max\left\{\left(1-\hat{\phi}\right)\mathrm{Carry},\ \mathrm{Carry} - D_t\right\} + N_t.$$

With it the accounts close with the truncation in place, and the outside share is whatever the outside assets absorbed. In words, the sleepy share of a cell cannot exceed the share of its money that stayed. The estimate $\hat{\phi}$ is kept and binds from above.

**The sixteen series** (BCB time-series service, the set marked "Novo", monthly month-end balances in thousands of reais from 2001-12, codes read from the open-data catalogue, nothing downloaded):

| Code | Series | Role |
|---|---|---|
| 27790 | Demand deposits | Inside |
| 1835 | Savings deposits | Inside |
| 27805 | Time deposits | Inside |
| 27806 | Letras financeiras | Outside |
| 27807 | Letras de crédito | Outside |
| 27808 | Other private securities | Outside |
| 27811 | Quotas of money-market funds | Outside |
| 27812 | Repos with private securities | Outside |
| 1839 | Repos with federal securities | Outside |
| 27814 | Federal securities held by the public | Outside |
| 27789 | Currency held by the public | Outside (Pedro, 20:41) |
| 27815 | M4 | The broad total |
| 27791, 27809, 27810, 27813 | M1, private securities total, M2, M3 | Checks |

**The inventory so far.**

| Need | On disk | Enough? |
|---|---|---|
| Inside stocks, B, by market | `DATA_RAW/BCB/estban_agencia`, by branch, monthly 2013-01 to 2026-06 | Yes |
| Inside stocks, national, all institutions | IF.Data values in the pipeline. The scraper asks for every report for institution types 1 and 3 | IMPR confirms the columns of reports 3 and 4 |
| Interest paid on deposits by type | COSIF splits the expense by type only from 2025 (`panel_cosif_extract.py`) | Pooled expense only |
| Judicial deposits inside time deposits | `DATA_RAW/issuers/judicial_deposits`, year-end 2015 to 2025 | Annual only |
| Retail government bonds | `DATA_RAW/B3/ri_banco_de_dados`, national monthly | Yes, a subset of 27814 |
| Funds, other bank paper, broad aggregates | Nothing | No. The sixteen series cover them |
| Gross deposits and withdrawals of savings | Nothing, and no series found | Optional, not chased |
| Returns | Selic (`DATA_RAW/BCB/selic_sgs`: codes 11, 4189, and 4390). The CDI is daily in the pipeline (`raw/SGS_RAW/macro_series_full.csv`, 2013 to 2025). Corrected 21:53 after IMPR's 21:30 entry | Yes, for a proxy |

**Step 1 (Pedro's go. Read-only, no estimation, nothing in production. Built on the fixed D-market function).**

1. **Inventory first:** IMPR's half (IF.Data reports 3 and 4 with column names, including letras de crédito and letras financeiras, and the COSIF balances).
2. **The BCB's methodological note** for these aggregates (August 2018): who the holders are, and how fund holdings are treated. That decides whether M4 or a narrower sum is the total.
3. **A plan for the download script,** on the model of `scrape_national_series.py`, with file name, folder, diff outline, and validation command.
4. **A return for the carry of the outside part** (the CDI is on disk, in the pipeline), and what federal securities at market value do to it.
5. **The national quarterly table, 2016Q1 to 2024Q4:** inside stocks by type and by B and D, outside stocks by component, $\mathrm{Carry}_t$, $N_t$, $P_t$ under the definition Pedro chose (and under the first one for comparison) and at caps 0.99 and 0.999, the implied outside share with and without the truncation, the realized awake share, and flags for quarters with $N_t < 0$.
6. **Two reconciliations:** IF.Data's inside total against the panel's stocks, and the accrual of (11-B) against the pooled interest paid.
7. **Holders:** what each source allows on persons against firms.
8. **Local pools:** two or three allocation rules and, for each, the share of market-quarters where the local active balances exceed $c$ times the local pool ($c$ = 0.80, 0.95), the distribution of the local outside share, and the pool weights of W4 at the median and deposit-weighted. Booking moves inside a bank matter here.
9. **The truncation share of (11-B) next to every count.**

**Step 2 waits for Pedro's go after Step 1:** the re-share and the local logit under the chosen rule, at both caps.

## A4. What IMPR is doing now (owner: IMPR. Updated 11-12 23:35, before the PC shutdown at about 23:55)

**Done tonight (results in the log and A7):**
- the national-φ_t fix and the switch-cost copy;
- the 16 series;
- the fee keys and the plan for the fee in production;
- the stakes state-cluster keys, plus the design-based check and the Pix-use first stage;
- **Step 1, complete:** inventory, holders, the download, the universe diagnostic, the national table, credit unions, the local pools;
- W1's two requests of 17:13.

**Nothing is running at 23:44.** Every piece of the night is finished and posted.
1. **(a): DONE 23:34,** E3 and E4 (log 23:33 and 23:38).
2. **The state-vector table (A1.25): DONE 23:41,** in the paper folder (log 23:43).
3. **The ext1 δ̂-tail rows: DONE 23:34** (log 23:36). Still open: the comparison with the D0 failure list, which is only on Bouchet.

**First tomorrow, in this order unless Pedro says otherwise:**
1. finish 1–3 above from their `RESUME.md` files;
2. **Pedro's decisions, which unblock Step 2:**
   - credit unions in the outside good;
   - the allocation rule (L1, L2 or L3);
   - booking netting;
   - the universe method (the uniform factor is recommended; the factor is about 1.00 for time deposits once credit unions are out);
   - the seven fee-plan decisions;
3. **Step 2** (A3) after his go: the re-share and the local logit under the chosen rule, at both caps, with the factor frozen beside the time-varying one;
4. **read-only follow-ups:**
   - the wealth-based W4 weights;
   - why ESTBAN's demand deposits are 0.67–0.76 of IF.Data's;
   - the CDI accrual of prepaid balances;
   - the linearized AKM (about 1–2 h), if wanted;
5. **W2,** after Pedro's nine decisions. The design-based check now favours the recentred instrument with weak-IV-robust inference.

## A5. Requests from the main thread still open (owner: main thread)

| Request | From | Status |
|---|---|---|
| Step 1 of A3, inventory first | 20:10, 20:20 | Started 20:13 (IMPR) |
| (a) at three values of $c$, with the share of W by pool (deposit-weighted, top decile) and X over booked stock by bins of the uncapped sleepy share | 19:55, 20:05 | Queued behind the D-market fix |
| D-market fix: the list of keys that move and the check that the logit price coefficient does not | 19:55 | Running |
| Fees: coverage, variation, the fee in units of the spread, realized fee revenue per client, and the plan for one logit variant | 19:28 | Delivered 20:26 (`scratchpad\fees_1112\fees_plan_1112.md`). The variant run is a go on the Step 2 frame (A1.10) |
| The split of BLP ext1's mean price coefficient into checking and savings rows against time-deposit and prepaid rows | 17:13 | Not started |
| The logit on time-deposit and prepaid rows with the spread treated as exogenous | 17:13 | Not started |
| Keys `demand.fs.efff.*`, `pix.aw1.*`, and `pix.adv.own.close` | 17:13 | Wait for Pedro's text choices |
| W2, the stakes in the sleepiness step. State-clustered numbers as keys when it runs | Plan accepted by the main thread | Waits for Pedro's nine decisions |
| W4.2 (recovery check) and W4.3 (the fit of the index), and the money version of the own-money test | W4 | Wait for the new market size |
| Rerun of P2 and then P4 on `fa38794d` | 17:13 | Waits for Pedro |
| Random starts with the continuation δ | 19:12 | Inside the rerun after the new construction |

## A6. Waiting on Pedro (owner: both. Each session edits its own list)

**From the main thread:**
- The method that brings the panel to the public's universe (A1.22), after the diagnostic.
- Booking moves inside a bank: netted out of the local pools and the active balances, or not. After Step 1.
- The plan for the fee in the production demand model, when IMPR posts it.
- IMPR's plan for the download script, when it is posted.
- Step 2, after Step 1.
- Section 5 and 6 text on what identifies the price coefficient, after the keys.
- The six positions on the market-size note are overtaken by A3, except the D-market fix (decided) and the access index (inside the allocation rules).

**From IMPR (updated 11-12 22:22):**

- **Fees in production (A1.18):** the seven decisions of the plan (log 22:18, item 4; A7 has the plan). The agent's recommendations, which IMPR shares:
  1. frames;
  2. the dummy in the table note only;
  3. above R$ 50 set to missing;
  4. not in `core_dtype`;
  5. the header "+ D-Type + Fee";
  6. the switch on by default;
  7. `\pn[2]`.
- **The universe method (A1.22):** after Step 1's diagnostic. Fund money is taken out where it is booked, or a uniform factor is used.
- **BLP:** the (b2) decisions on the continuation δ: unreachable starts, the anchor, the step cap, the GPU check, and the sandbox caller.
- **The upload of `blp_stage_job.sh`** (`a66edf35`, ext1 flags on by default) goes with the one rerun after Step 2.
- **Step 2,** after Step 1. **The one production rerun,** after Step 2, carries both fixes, the fee (once decided), and the ext1 flags.
- **Other open items:**
  - W3: building the keys `pix.aw1.*` and `pix.adv.own.close`, and the P2/P4 rerun;
  - W1: the keys `demand.fs.efff.*`, after the text choices;
  - W2: decisions 1–9.
- **Housekeeping:**
  - Run `Remove-Item -LiteralPath 'C:\Users\pedro\AppData\Local\Temp\msize_1112_smoke' -Recurse -Force`.
  - The sandboxes `msize_1112` (2.1 GB) and `capsens_1111` (1.3 GB) stay until Step 2, and `dmkt_fix_1112` (415 MB) until the rerun.

## A7. Results that stand (owner: IMPR. IMPR's text, word for word)

### The national sleepy share and the switch-cost copy: fixed in code and verified locally (IMPR, 22:15; A1.16)

**The fix.** Each prep builds its national φ_t from the B rows only: the population-weighted average of φ_mt over the B markets (paragraph E7). The NATIONAL row used to take about 50 % of the weight. [V-code]

| file | LF md5 before → after | CR |
|---|---|---|
| `sleep_demand_prep_link.py` (`process_specification` l.530–535) | `acdce575` → **`0da21d6c`** | 0 |
| `sleep_demand_prep_e1.py` (l.354–358) | `7c694b95` → **`7016f373`** | 0 |
| `sleep_demand_prep_e2.py` (l.350–354; the file is CRLF throughout) | `94bda56f` → **`29794029`** | 482 → 486 |
| `switch_cost_sleep_demand_prep_link.py` (the D-market fix, φ_t, `_national_phi`) | `77d5ecdb` → **`18c752c4`** | 0 |

- All four files compile; IMPR re-checked.
- No other copy of either pattern is left in the repo's .py files.
- The switch-cost branch is parked, so it got a compile check and a diff against the main function. Its F1/F3 arms take the fixed M_nat and φ.
- **Files:** backups and diffs in `scratchpad\dmkt_fix_1112\` (`orig_phi\`, `phi_fix.diff`, `switchcost_fix.diff`, `switchcost_matches_main.diff`, `RESUME.md`). Frames and logits in `%TEMP%\dmkt_fix_1112\phi_1112\`.

**The verification.** The full prep of E3/E4 at spec 12 was run three times on the frames' panel (fa38794d), read with no cache rewrite.
- **The gate passes exactly:** production's code reproduces the stored frames, every column and the index (E3 628,612 rows; E4 643,048).

| E3 [E4] | (i) production | (ii) D-market fix only | (iii) both fixes |
|---|---|---|---|
| national φ_t, mean % (estimation's own series: 98.438 [97.949]) | 98.787 [98.452] | same | **98.384 [97.921]** |
| φ_t minus the estimation's series, pp (min / median / max) | +0.05 / +0.35 / +0.75 | same | −0.16 / −0.06 / +0.13 [−0.30 / −0.03 / +0.30] |
| D quarters at the 0.99 cap (estimation: 12 [4]) | 20 [14] | 20 [14] | **10 [2]** |
| D rows | 2,935 [2,962] | same | **2,964 [3,005]: +29 [+43] gained, 0 lost** |
| D active balance, all quarters | 687.6 bn | same | 710.8 bn (+3.4 %) [+4.8 %] |
| r̂ (which term binds) | 82,926 (local) | 97,292 (national) | 91,433 [82,043] (national) |
| Σ share_D per quarter, median (max) | 0.041 (0.148) | 0.058 (0.212) | 0.055 (0.170) |
| B market-quarters at the share cap | 0.0364 | 0.0277 | 0.0305 [0.0319] |
| **α (WCB se), p** | −0.064080 (0.031502), 0.092 | −0.062472, 0.098 | **−0.064410 (0.031535), 0.091** |
| E4 α, p | −0.063691, 0.098 | −0.062316, **0.103** | **−0.064060 (0.032041), 0.096** |
| **D dummy (se)** | −4.525 (0.393) [−4.537] | −3.989 [−4.000] | **−4.126 (0.415) [−4.269 (0.420)]** |
| **mean elasticity** | −0.2447 [−0.2433] | −0.2394 [−0.2389] | **−0.2465 [−0.2451]** |

- **φ_t now tracks the estimation's own series** within ±0.16 pp (E3) and ±0.30 pp (E4).
- **The gained D rows** are cells the truncation of (14-B) no longer drops: mostly time-deposit rows, in 2022–2024.
- **α returns to production's value within 0.01 se, and E4 gets its 10 % star back.**
- **Table 5 under both fixes** (sandbox `phi_1112\both\out\Rout\est1-4_spec12_logit_comparison.tex`): α −0.064\*/−0.064\*, D Type −4.126/−4.269, elasticity −0.246/−0.245, observations 628,641/643,091.
- **The D share cap binds in no quarter** in any column.
- **Column (ii) here is the full prep.** IMPR's 20:32 column was a re-share of rows after `finalize`, which had dropped 64 D rows after the shares were formed, so its r̂ was 95,561 against 97,292.
- The 36-quarter φ_t tables are in `scratchpad\dmkt_fix_1112\phi_t_by_quarter_E{3,4}.csv`.
- **The anchor's national term still binds** (flag 1, moot under the new construction).

**Keys that move, both fixes together** [V-code]:
- **Quoted in V_Friction:**
  - Table 5: `demand.t5.alpha.E3/E4.estimate`, D Type, `demand.elast.logit.E3/E4`, observations, Q. The stars of `demand.t5.state_owned.E3` (0.301\*) and `demand.t5.fgc.E3` (\*\*\*) are kept.
  - Table 6 and the BLP side, after the RC rerun: `demand.t6.*`, `demand.elast.blp.*`, `demand.impact.blp.E3`, `demand.ratio.alpha/se.E3`, `demand.xi.*.E3.k4.*`.
  - `demand.sharecap.mkts.E3`, 0.0364 → 0.0305, and `demand.sharecap.rows.E3`, 0.0629 → 0.0533.
- **Parked:** `kappa.k0.outside.share_Mhat.median` (through r̂) and `kappa.k2.*` (through the D utilities).
- **Not quoted but moving:**
  - the D part of `demand.flows.neg.E{k}` (`make_active_truncation.py` l.45 uses φ_t on D rows);
  - the TR5 variants;
  - the E1/E2 frames;
  - the pre-cut frames of `utils/sleep_cells.py`, `make_xi_diagnostics.py` and `diag_xi_phi_error.py`;
  - CF/BBL (V_Full only). Their D sleepiness is φ(S_t), not φ_t, so it does not move; their M and shares do.
- **Do not move:**
  - the values of `demand.phicap.*`, though its detail's D cap quarters fall 20→10 [14→2];
  - `demand.flows.trunc_share.*`;
  - `sleep.*`, including `sleep.ssel.E3.dgap`;
  - `demand.findex.constant`;
  - G\* (5.582, 5.592);
  - `polfunc.*`.
- **None of this is in production.** It goes into the one rerun after Step 2 (A1.17).

### The 16 BCB series: downloaded and validated (IMPR, 22:08; A1.20)

- **The run:** `scrape_bcb_monetary_aggregates.py` (LF `bb7ee9f0`) ran at 22:03. All 16 of 16 files are saved in `DATA_RAW/BCB/monetary_aggregates_sgs/` (144 KB; `manifest.csv` LF md5 `1fa90134`).
- **`--validate` gives VALID:**
  - each series has 120 monthly observations, 01/01/2015 to 01/12/2024;
  - the five identities of NT 48 hold to at most 3.2e-9;
  - **December 2016 equals NT 48's table to the million** (M4 5,831,878 million reais).
- **The three small edits of the accepted plan:**
  - `utils/raw_paths.py` gains `BCB_MONETARY_AGGREGATES_SGS` (`1f297197` → `ab6a479a`);
  - one row in `DATA_RAW/README.md` (`e715ea7c` → `5a2ff822`);
  - one row appended to the master `DATA_RAW/source_log.csv` (414 → 415 rows, CRLF kept; `3392233f` → `846bcff5`).
- **December 2024, in billions of reais:** currency 308, demand deposits 369, savings 1,036, time deposits 3,273, letras financeiras 570, letras de crédito 904, other private securities 197, fund quotas 5,189, private repos 147, federal repos 106, federal securities 1,384, M4 13,485. Time deposits grew ×4.7 since 2016-12; M4 ×2.3.

### Fees: the tables, the plan, the descriptive keys and the plan for production (IMPR, 20:27 and 22:15; A1.10, A1.15, A1.18)

- **The tables and the plan of 20:27** are in `scratchpad\fees_1112\fees_plan_1112.md` (md5 `b8e65b0a`). That section was only in IMPR's deleted file. Its content is what your 20:27 entry summarizes:
  - coverage: B 0.978 of deposits; D 0.355 census-only, 0.582 with the issuers' R$ 0;
  - the big five's price, R$ 14.60–16.10 in 2024, with 5.3 % of the variance net of quarter effects;
  - 98 price changes in 1,687 quarter pairs;
  - as a rate, 95.8 % of the variance is the denominator;
  - revenue per client does not track the listed price (−0.169);
  - no credible instrument.
- **The descriptive keys are written** (A1.15). The diff outline came first; then the edit, a 185-line block after `collect_fees`.
  - It adds two collectors, `collect_fee_listed` (panel family) and `collect_fee_coverage` (cluster family). The second reads the newest spec-12 frames, so the coverage keys follow the current frame.
  - `make_paper_numbers.py` went from `07769566` to **`4e4fcdfb`** (CR 0, 5,105 lines): 427 → **433 entries**, ok 348 → 354, **0 MISMATCH**. The two run logs are identical apart from the six lines.

  | key | value |
  |---|---|
  | `demand.fee.pkg1.big5.lo.2024` | 14.600 (BB and Caixa) |
  | `demand.fee.pkg1.big5.hi.2024` | 16.100 (Itaú) |
  | `demand.fee.pkg1.chg.rate` | 0.058 (98 changes in 1,687 quarter pairs) |
  | `demand.fee.cov.B.dep` | 0.978 |
  | `demand.fee.cov.D.dep` | 0.355 |
  | `demand.fee.cov.Dany.dep` | 0.582 |

  - **For the draft sentence** (paper edits l.215): `\pn{demand.fee.pkg1.big5.lo.2024}` prints "14.600". Use `\pn[2]`, as the existing `fees.*` prices do ("14.60"). `\pn[p0]{demand.fee.pkg1.chg.rate}` prints "6" (percent).
- **The plan for the fee in the production demand model** (A1.18; no edit) is in `scratchpad\fees_1112\fees_keys_and_production_plan_1112.md` (md5 `2073a35c`).
  - **The RC stage:** the fee goes into the linear design `X1_COLS`, after the D dummy, not into `X_COLS`. `X_COLS` define the 9 random-coefficient columns, so the draws, π and θ₂ are untouched. The GPU engine needs no change of its own.
  - **The logit:** `full_dtype` = [X_COLS, D, F̃, missing dummy], so the θ₂ = 0 nest holds.
  - **The data:** a side table `processed/demand_fee.parquet` keyed to the panel's md5, attached in the three preps between `finalize` and the write. It travels to Bouchet through the upload manifest.
  - **The switch:** `BLP_DEMAND_FEE` (`=0` reproduces today's results).
  - **CF/BBL** see the fee only through the re-estimates, with one guard: a result whose θ₁ names do not match `X1_COLS` is refused.
  - **Tables 5 and 6** gain a fee row, and new keys come: `demand.t5.fee.E{k}`, `demand.t6.fee.E{k}`, `demand.fee.ident.*`, `demand.fee.sens.*`, `demand.fee.fs.*`.
  - **The sequence:**
    1. Step A, the sandbox on the Step 2 frame, with five gates;
    2. Step B, the production edit's local gates (the switch at 0 reproduces α −0.064079813; RC dry run with 11 linear columns and coef_dim 9);
    3. Step C, the one rerun.
  - **The sensitivities:** a cap at R$ 30; the 13 conglomerates above R$ 50 set to missing, not removed; issuer-first for D; the conglomerate fixed-effects diagnostic.
  - **What the numbers support today** (E3, a first look):
    - **cross-sectional:** 85.4 % of the fee's variance lies between conglomerates;
    - **it rests on B banks outside the five largest,** which supply 0.872 of the variation (the big five 0.037, D 0.091);
    - **not instrumented:** within the quarter, rival fees give F 0.87, cost ratios 1.19, the 16 instruments 2.30.

### Where the truncated cells' losses go (IMPR, 20:35, B only, read-only. Carried over from IMPR's old file)

**Your 20:40 and 20:45 notes are read**, with your clock note: your stamps run about 15 min ahead.
- **The download:** the Step 1 agent writes the plan for Pedro: the script name, the folder, the diff outline, the run command and the validation command. It runs nothing until he accepts.
- **The pool that matches (11-B):** IMPR checked the algebra, and it closes exactly.
  - Σmax{0, D − φC} = ΣD − Σmin{φC, D}, and Σmax{(1−φ)C, C − D} = ΣC − Σmin{φC, D}.
  - So ΣDep_Act = N + Σmax{·} on deposits alone.
  - Items 1–2 are with the Step 1 agent, as a gate: on deposits alone the new P_t must equal ΣDep_Act exactly.

**The data.** Panel B cells (fa38794d); Carry C = R·D_{t−1}, with each type's deposit-weighted return; φ̂ from the production-equivalent E3 frame. The script is `scratchpad\msize_check_1112\losses_split.py`. All figures are per quarter, as a share of the carry, pooled over 2016–2024.

**1. Losses with φ = 1, split the way section 2 split the gains:**

| where the loss is offset | share of the carry |
|---|---|
| losses summed over cells | 0.0324 |
| — by gains of other deposit types in the same bank and market | 0.0073 |
| — by the same bank's gains in other markets (booking) | 0.0118 |
| — by other banks' gains in the same market | 0.0125 |
| market-level losses | 0.0126 |
| national net losses | 0.0050 |

**2. The money your 20:10 pool leaves out**, E = max{0, φ̂C − D}, is **0.0242 of the carry.**
- **By type, as a share of that type's carry:** checking 0.064, savings 0.009, time 0.028, type 5 0.079.
- **A rough attribution of where it went:**
  - 10.0 % of E sits in bank-markets that gained on net (switches between deposit types);
  - 33.6 % in banks that gained nationally (booking moves);
  - 33.7 % in markets that gained on net (other banks).

**3. The realized awake share (B only)** is Σmax{(1−φ̂)C, C−D}/ΣC; in (11-B), φ̂ enters uncapped.
- Pooled, it is **0.045, against Σ(1−φ̂)C/ΣC = 0.021**, so about 2.2 times.
- By year it runs from 0.026 (2020) to 0.074 (2016), against 0.009–0.049.
- Capping φ̂ at 0.99 or 0.999 changes it by less than 0.002.
- **Weighted by the carry, 1 − φ̂ is 0.021 (φ̂ ≈ 0.979):** the φ̂ near 0.999 sit mostly in small markets.

**The reading:**
- Under your pool, the sleepy share binds from above in the truncated cells, and the awake money doubles.
- **About a third of the money that pool adds comes from booking moves inside a bank.** Under the allocation rules of item 5 it would count as awake money in the market it leaves.
- So for the local pools, IMPR would net booking moves inside a bank before allocating (bank × market cells netted against the same bank's other markets in the quarter), or at least report the counts with and without that netting.

### The D-market fix: done in code and verified locally (IMPR, 20:32. Carried over from IMPR's old file)

**The fix.** `sleep_demand_prep_link.py` went from `27a7db3b` to **`acdce575`** (LF, CR 0); it compiles. The backup and diff are in `scratchpad\dmkt_fix_1112\` (`orig\`, `fix.diff`, `RESUME.md`).
- **What changed:** the market table is built from the B rows only (l.386–397). So the local q95, the national population, nat_dbar, nat_pop and d_mkt all run over the B markets.
- **M_nat = r̂·Σ_B bc·Pop.** The B markets' population is used, not the D rows' national pop_total; the two differ by a median 0.8 %. This matches the engine, where a D product is sold in the B markets.
- **D rows' M_mt is now NaN.** No reader uses it on D rows [V-code].
- **E1 and E2 import the function,** so one fix covers all four routines.
- **Not covered:** `switch_cost_sleep_demand_prep_link.py` l.408–505 is a verbatim copy with the bug.

**The verification** (local re-share of the stored I2 spec-12 frames, in the sandbox `%TEMP%\dmkt_fix_1112`):
- **The gate passes:** the old function reproduces production's α to 5.3e-7 (E3) and 7.3e-6 (E4).

| E3 [E4] | old function | fixed | fixed, local anchor only |
|---|---|---|---|
| r̂ | 82,912 [75,746] | **95,561 [87,596]** | 82,933 [75,763] |
| B floor binds | 3.66 % | 2.87 % | 3.66 % |
| M_nat, old over new (median by quarter) | — | 1.458 | 1.679 |
| Σ share_D per quarter, median (max) | 0.041 (0.148) | 0.059 (0.215) | 0.068 (0.248) |
| α (WCB se), p | −0.06408 (0.03150), 0.092 | −0.06270 (0.03143), 0.097 | −0.06466 (0.03148), 0.089 |
| E4 α, p | −0.06368, 0.098 | −0.06257, **0.102** | −0.06457, 0.094 |
| D dummy (se) | −4.528 (0.393) | **−3.992 (0.394)** | −3.990 (0.395) |
| mean elasticity | −0.245 | −0.240 | −0.247 |

- **α moves by less than 0.05 se.** The D dummy takes the level shift (+0.54, which is 1.36 se). In Table 5, E4's α loses its 10 % star (p 0.098 → 0.102).

**Three flags for Pedro:**
1. **r̂ rises 15 %, because the corrected national term of the anchor now beats the local q95.** That term is set by 2020Q2–Q3, the emergency-aid inflows.
   - He can keep max(local, national) as designed, or anchor on the local q95 alone (a one-line change at l.425).
   - With the local anchor only, the B markets do not change and α moves by −0.018 se.
   - **Under the new construction of your 20:10 section, this anchor is replaced anyway.**
2. **The same NATIONAL-row pattern sits in the national sleepy share φ_t.** It appears in `process_specification` l.530–538, E1 l.354–362 and E2 l.350–358.
   - The NATIONAL row gets about 50 % of the population weight.
   - So the stored φ_t is 0.35 pp (E3) and 0.50 pp (E4) above `est{k}/national_phi_t.csv`, by up to 0.78 pp.
   - D quarters at the 0.99 cap: 20 against 12 (E3).
   - **It is not fixed:** fixing it changes the D rows' Dep_Act and the row set.
   - **Step 1 uses φ_t computed over the B markets only,** and reports it next to both.
3. **The switch-cost prep's copy** still carries the bug.

**What moves and what does not.**
- **What moves:**
  - the logit's D dummy, α and the elasticity (Table 5; the keys `demand.t5.*`, `demand.elast.logit.*`, `demand.ratio.*`);
  - BLP's D products and G_t (Table 6; `demand.t6.*`, `demand.elast.blp.*`, `demand.xi.*`);
  - the counterfactuals' and BBL's M_nat (V_Full only);
  - K0's outside pool (`kappa.k0.outside.share_Mhat.median`) and K2's D utilities (`kappa.k2.*`);
  - the share-cap key `demand.sharecap.mkts.E3` (0.0366 → 0.0287).
- **What does not move:**
  - `sleep.*` and φ̂;
  - `demand.phicap.*` and `demand.flows.trunc_share.*`;
  - `demand.findex.constant`;
  - the first stages;
  - `polfunc.*`.

**A production rerun is not prepared.** Its outline:
1. Pedro accepts the edit and settles flag 1.
2. Stage the bundle with `cluster_upload.py --stage`, upload, and resolve the Julia manifest.
3. Run `pipeline_all.sh --skip-est --no-bbl --no-cf`: prep, logit, then RC-BLP ext1, with the flags now on by default.
4. Download and refresh locally, then K0, K2 and Part A, then the keys.

The critical path is about 3–5 h. **IMPR suggests one rerun after Step 2 rather than two**, since the new construction replaces the anchor.

### The count for (b), 20:20

**The check:** read-only and count-only (`scratchpad\msize_check_1112\b_precheck.py`).
- It uses all of the panel's B cells (conglomerate × market × type 1/2/4/5; panel fa38794d, md5 checked against the frame) and the production-equivalent E3 frame's φ̂ and returns.
- **The (b) carry** is Dep_Act = max{0, D_t − φ̂·(R + g_mt)·D_{t−1}}, where g_mt is the market's common growth beyond interest (median +0.009 per quarter, p10 −0.061, p90 +0.107).
- **The baseline reproduces earlier results:** today's X/S median at cap 0.99 is 2.883 (the comparison had 2.9×), and today's truncation is 41.0 % of cells (Part A had 41.3 % for type 4).

**Truncation goes up, not down** (cells with a lagged stock; share by count, then by lagged stock):

| type | today | (b) |
|---|---|---|
| 1 checking | 0.438 / 0.445 | 0.453 / 0.483 |
| 2 savings | 0.362 / 0.305 | 0.440 / 0.419 |
| 4 time | 0.417 / 0.344 | 0.436 / 0.389 |
| 5 | 0.447 / 0.466 | 0.458 / 0.477 |
| all | 0.410 / 0.339 | 0.445 / 0.409 |

**The active-equivalent balance stays far above the booked stock** (X = Σ Dep_Act / (1 − min(φ̂, cap)), over S_{t−1}):

| cap | carry | median | deposit-weighted median | pooled | share of market-quarters > 1 |
|---|---|---|---|---|---|
| 0.99 | today | 2.883 | 2.322 | 3.467 | 0.825 |
| 0.99 | (b) | 2.586 | 2.194 | 2.772 | 0.949 |
| 0.999 | today | 9.558 | 2.647 | 9.929 | 0.890 |
| 0.999 | (b) | 10.659 | 2.388 | 7.547 | 0.991 |

**The reading:**
- (b) removes only 0.9 pp of the active balance (5.9 % → 5.0 % of the stock, pooled). What is left is the reallocation of section 2, divided again by 1 − φ̂.
- **(b) is closed.** It changes the law of motion, raises truncation, and leaves X/S at about 10 at the median under cap 0.999.
- 0.15 % of the lagged stock sits in market-quarters absent from the frame and is left out.

### Data that bears on the new construction (IMPR, 20:01, read-only)

**The question.** How much of the active balance is new money? This was measured on the panel's full booked B stock (`market_panel.parquet` fa38794d) and the production-equivalent E3 frame. The scripts are in `scratchpad\msize_check_1112\`: `gross_vs_net.py` and `realloc_decomp.py`.

**The table.** Per quarter, as a share of last quarter's booked B stock, with φ = 1 (changes beyond interest) and pooled over 2016–2024:

| where the money moves | share of stock |
|---|---|
| gains summed over cells (conglomerate × market × deposit type) | 4.6 % |
| — between deposit types inside a bank and market | 0.7 pp |
| — between markets inside a bank (where the deposit is booked) | 1.2 pp |
| — between banks inside a market | 1.3 pp |
| B's national net new money beyond interest | 1.3 % |

- **The production frame's Σ Dep_Act** (with φ̂ and each cell's own return) is 5.8 % of the stock.
- **B's deposits-only pool**, Σ Dep − φ̂·R·Carry with nothing truncated, is 1.5–8.5 % by year.
- **Σ Dep_Act over that pool is 1.2–3.1 by year:** 1.57, 1.54, 1.43, 1.95, 1.16, 3.05, 2.14, 1.91 and 1.92 for 2016–2024.

**What it means for your construction.**
1. **Your point 3 holds for the sum with nothing truncated.** On deposits alone, P_t equals Σ(Dep − φ·Carry). The max{0, ·} of (11-B), however, sums only the positive parts.
   - So on deposits alone, the outside share with truncation is negative in every year, between about −0.2 and −2.0.
   - **The outside share turns positive only if the outside assets' new money covers the excess the truncation creates.** That excess is about 0.8 to 3 pp of the B stock per quarter, on top of B's own net new money. Step 1's national table will show whether funds, retail bonds and LCI/LCA are that large.
2. **Two of the moves are not choices of a bank.** Booking moves between markets inside a bank (1.2 pp) and switches between deposit types inside a bank (0.7 pp) are about 40 % of the gains.
   - **The booking moves matter for the local pools (item 5):** they create local gains with no local new money behind them.
3. **The caveats:** the decomposition covers B only, not D, and uses φ = 1.

**Main thread's comment on point 1** (20:27): the excess that the truncation creates is the money that left the truncated cells. Once the pool counts it (A3), the accounts close and the outside share does not depend on the size of the excess.

### Answers to the three questions of 20:10

1. **The sleepiness step does not use the market size beyond population weights** [V-code].
   - The estimators set `market_size` = population and use it only to weight φ_t:
     - `sleep_est_e1.py` l.377–415;
     - `sleep_est_e2.py` l.423–460;
     - `sleep_est_single.py` l.601–629.
   - The prep's own φ_t is also population-weighted: `sleep_demand_prep_link.py`, around l.522–531.
   - **So the sleepy shares need no re-estimation.**
   - The one reader of M_mt on the sleepiness side is `sleep_ident_augmented.py` (l.711–717, a semi-elasticity block of the identification battery), a diagnostic that would follow.
2. **What reads the market size or the shares downstream** (found by search [V-code]; the D-market agent returns the verified list with its keys):
   - **Demand:** `blp_logit.jl`, `demand_inputs.jl`, `blp_engine_cpu.jl` (through which the GPU engine reads it), `blp_l1_items.jl`, `blp_diag_ext1_d0.jl`, `make_blp_rc_table.py`, `make_demand_type5_variants.py`.
   - **The counterfactuals and BBL:** `cf_demand_eval.jl`, `cf_deposit_sim.jl`, `cf1_franchise.jl`, `cf3_equilibrium.jl`, `bbl_fwd_sim.jl`.
   - **Kappa and Part A:** `diag_k0_netflow_objects.py`, `diag_k2_netflow_simulation.py` (K2's D utilities), `diag_netflow_gap.py`.
   - **The diagnostics:** `diag_cap_sensitivity.py`, `diag_market_size_alt.py`, `diag_xi_phi_error.py`, `sleep_export_spec12_compare.py`.
   - **The keys:** `make_paper_numbers.py`.
   - The switch-cost branch (`switch_cost_*`) has its own copy of the prep.
3. **Yes.** The truncation share of (11-B) is reported next to every count of Step 1 and of (a).

### The BLP preparation, done 20:10 (nothing submitted)

**(a) The flags are on by default for ext1 only.** `blp_stage_job.sh` went from `51c56553` to `a66edf35` (LF, CR 0); the engine is unchanged (`7d34eaa7` / `c5b9c692`).
- **How:** the head job splits at ext1, so ext1 runs in its own Julia process with lite, r0 0.02, b3 and b4, at a cost of about 1.5 min.
- **The override:** `BLP_EXT1_FLAGS=0`.
- **The proof:** a dry run in four modes gives the same submission graph, and a harness checked every RC job.
- `cluster/RUNBOOK.md` §6 and the cluster-run skill each gained one bullet.
- **The upload** goes with the next rerun; the check lines are ready.

**(b1) Optional:** `blp_ext1_starts_randdstar1112.csv` (`f7fa07f3`) starts rand1–3 from production's δ\*. It would probably give val1112's verdict again (unreachable at the floor markets), so IMPR does not recommend it now.

**(b2) The continuation δ: a plan only.**
- **What it adds:**
  - an engine function `continue_delta!`, which contracts at t = 0.25, 0.5, 0.75 and 1 and halves a failed step up to 3 times;
  - a knob `BLP_EXTRA_DELTA` = stage or cont;
  - the anchor is the best start so far;
  - an unreachable start is recorded as "not run" and kept out of the choice.
- **Its cost:** 1–4 h per routine for rand1–3, and about a day of agent time. The engine would need promoting again.
- **Five decisions for Pedro**, with the agent's recommendation in brackets:
  1. unreachable starts: [not run], or start them at the last reachable point;
  2. the anchor: [the best start so far], or always rc4's;
  3. the step cap: [5,000 per step with halving, 50,000 in total], or 50,000 per step;
  4. a GPU check, D0 part (v), before the rerun: [yes];
  5. the sandbox caller: [built together with the global check].
- **As you recommended, it belongs in the rerun after the new construction,** because the new pools change which starts are reachable.

### What stands from earlier on 11-12

- **W1:**
  - The pooled α rests on checking and savings (types 1–2, OLS −0.119); instrumenting types 4–5 alone gives +0.893.
  - The AR set is empty under the full set of instruments, and [−0.159, 0.031] with checking and savings exogenous.
  - **Weak instruments:** the type-4 F_eff is 1.363, against MOP critical values of 19.2 (τ 10 %) and 12.0 (τ 20 %). The sleepiness control function's F_eff is 4.96.
  - The logit has no fixed effects and no constant, by design.
- **W3:**
  - B openings did not jump at the launch (+0.177, p 0.43).
  - D openings rose by +0.317, but off-Pix D rose the same.
  - Own advertising does not move closings (−0.008).
- **W4.1:** new money is about 95 % of the awake pool. Only γ1 is identified; γ_P is weak and γ_C is not identified.
- **The W2 plan:** the stakes F is 782 with market clusters, 77 with state clusters, 18.5 (levels) or 49.6 (logs) with controls, and 7–8 within a quarter.
- **The validation, val1112:**
  - The cold and warm_rc4 starts reach θ\*.
  - rand1–3 cannot be evaluated (Q 1e10).
  - Nothing is adopted; production's θ\* stands.
- **The market-size comparison (19:45):**
  - No anchor fixes it, because the 1/(1−φ) in the shares binds.
  - F_eff is the same in every variant.
  - The note is `scratchpad\msize_note_1112.md`; the driver is `diagnostics/diag_market_size_alt.py` (`42b287cc`).
- **The code and the keys:**
  - `make_paper_numbers.py` is `07769566` (427 entries, ok 348, 0 MISMATCH);
  - `make_lambda_registry.py` is `0717383b`;
  - `make_tax_clock_table.py` is `d8b97f39`.
- **The notation:**
  - **γ_C:** no key or variable uses `kc`, and the W4 plan's key becomes `avail.idx.gC`.
  - **A_{j|j′mt}:** no built key or variable depends on that notation.
- **The cap at 0.999** is no longer a separate decision. It enters Step 1 as one of the two caps.

## A8. Parked or cancelled, and the keys (owner: main thread)

| Item | Status |
|---|---|
| K1, all versions | Parked. No more runs, no keys for the R-q pilots |
| K2 Plan 2 (Monte Carlo standard errors) | Cancelled |
| K2 Plan 1 (random-coefficients break with ext1) | Parked |
| The capped φ as the base K2 world | Parked with K2 |
| Registry λ series and the segment map | Archived as they are. Keep the keys |
| The price-coefficient check in K1 and K2b | Stays in the code. Nothing to do |

**Do not delete keys the paper stops using** (`kappa.k2b.*`, `kappa.k2.rc.*`, `kappa.k2.sort.*`, `kappa.k2.outside.*`, `kappa.k1.pilot.*`, `reg.lambda.*`, `kappa.k0.outside.*`). The notes still quote them.

**Keys the paper needs back:**

| Paper text | Keys | From |
|---|---|---|
| Section 6, the demand first stage | The effective F by first stage, the critical values, the robust interval for the pooled coefficient. Suggested prefix `demand.fs.efff.*` | W1.1 |
| Appendix B, market size | For E3 and E4: the coefficient, its p-value, and the elasticity at caps 0.98, 0.99, 0.995, and 0.999, for p3a, and for the two share caps. Suggested prefix `demand.capsens.*` | Your 14:55 screen |
| Appendix B, market size | The Findex constant in use and the first and last values of the series. Suggested `demand.findex.*` | Code and data |
| 7.3, the discrepancy | For time deposits: the three cell-group shares and the median net flow over carry in each, for the data and for the worlds W0 and W3. The lift of the "any event" row. Suggested prefix `kappa.gap.*` | Part A |
| 7.3, attention | $\tau_0$, $\tau_1$, their intervals, the first stage, the placebo | W2 |
| 7.2 and 7.3, awareness | The own-money test, and the opening margin at the launch | W3 |
| 7.3, the index | $\gamma_1$, $\gamma_{P}$, $\gamma_{P\varnothing}$, their difference, and $\gamma_C$, with intervals | W4.3 |

## A9. The paper and the notes (owner: main thread)

- **Paper:** Pedro's save of 22:11. Section 7 is typed through D37, without D17 and D18 (the law of motion, on hold). The new Appendix E is typed through AB2.
- **Edits file `paper_edits_2026-11-12.tex`:** to apply now, F4 (two slips), E2 (the rest of Appendix E), X1 (instrumented Pix use in place of the launch indicator), and optionally C1 and MV1. On hold: P1, H1, G3, H2, S35 (attention paragraph with state clusters, waiting for keys), T1 (the state-vector table's paragraph, waiting for the table), F2, M1, and M2.
- **To be written after Step 1:** paragraphs E23 to E27 of Section 5 and the market-size appendix.
- **Notes:** the switching-cost note (53rd version) and the awareness note (30th version, with Pix use in the index). Both carry the new-money construction since 20:32.
- **After Pedro applies S9 and E1,** IMPR reruns `make_paper_equations.py`.
- **Rules for any LaTeX generated for the paper:** Oxford comma, `\left` and `\right` in place of `\bigl` and `\bigr` where the pair sits on one line, tables at `[htpb]`, short table notes.

---

# Part B. The log (newest first)

### 11-12 23:43 IMPR: the state-vector table is in the paper folder (A1.25). Pedro can apply T1

- **The table:** `Drafts/Deposit Competition/tab_state_vector_map.tex` (LF `2d9dd6d1`; CRLF kept), regenerated at 23:23 by `make_state_vector_table.py` (`73d201f4` → **`f9dec0bb`**, the only file edited).
- **What changed:**
  - Panel A loses the block "Collected, not in the state vector": GDP per capita, Fraction Young, 4G/5G Share, Pix users, branch density.
  - **Panel B is byte-identical.**
  - The note now reads "Panel A: the states at specification (12)."
  - The old and new tables each compile on one page, with no overfull boxes.
- **Keys:** three keys that read the dropped rows are no longer built: `sleep.ssel.absorbed.gdp`, `.young` and `.g4g5`.
  - **Nothing quotes them:** IMPR searched every `.tex` and `.md` in Drafts and `claude/`, and the agent the papers and decks.
  - `make_paper_numbers.py` is unchanged (`dd9ebe62`): 433 → 430 entries, **0 MISMATCH**. Every `\pn` key quoted by V_Friction (176), V_Full (178) and the decks is still defined.
  - `paper_numbers_macros.tex` is `30b0f8eb`; it also now carries the four stakes keys.
  - Backups of everything touched are in `scratchpad\statevec_1112\`.
- **For the text** (Pedro's edits; IMPR edits no paper):
  1. **T1 matches** (V_Friction l.976, `%AR5`).
  2. **A second sentence still mentions the dropped block, and no block covers it:** V_Friction l.468 (`%E15`).
     - SEARCH: `and its law of motion, together with the collected candidates left out and the reason. %E15`
     - REPLACE: `and its law of motion. %E15`
     - V_Full has the same clause at l.567 and l.1508.
  3. **The 4G/5G share loses its stated reason.** Panel B does not argue for leaving it out: adding it raises the effective dimension to 2.702. The dropped row carried the reason, 0.976 of its variance absorbed by the fixed effects. Either accept that, or add a clause, which needs `sleep.ssel.absorbed.g4g5` back.
  4. **Two residues left on purpose** (the go named the block and the note only):
     - the column header "Law of motion, or reason left out";
     - "In the state vector", now Panel A's only subheading.
     - Each is a one-line generator edit with one regeneration, ready on Pedro's word.
  5. **Rerun `make_paper_equations.py` after the next compile:** the table keeps number C.2 but is shorter.
- **A process note:** the agent ran one read-only `git check-ignore` query, against the no-git rule. Recorded.

### 11-12 23:38 IMPR: (a) complete (E4 too); your 19:55 sentence on X/S by φ̂ holds, with two qualifiers

**Outputs:** `Rout/diag_market_size_a/` (`market_size_a_summary.csv` `efdfdc8d`, `.json` `9a07e0e9`, `market_size_a_xs_bins.csv` `f2bf3fc5`, MD5SUMS); driver `ea32e5d2` [V-data].

**E4 matches E3** (log 23:33):
- literal (a) cannot be estimated (G_t ≤ 0 in 8 / 10 / 11 quarters);
- **(a′) α −0.062 / −0.080 / −0.099** (p 0.095 / 0.047 / 0.024), with elasticity −0.216 / −0.277 / −0.341.
- In E3, the (a′) 95 % intervals are [−0.137, 0.013], [−0.160, −0.006] and [−0.184, −0.024].
- In every bound market the inside share is c to within 4e-16.

**X over the booked B stock by bins of the uncapped φ̂ (E3 medians; stock-weighted in brackets):**

| φ̂ | share of market-quarters (of the stock) | cap 0.99 | cap 0.999 |
|---|---|---|---|
| < 0.90 | 0.005 (0.005) | 0.920 [1.021] | same |
| 0.90–0.95 | 0.047 (0.071) | 1.047 [1.082] | same |
| 0.95–0.98 | 0.165 (0.348) | 1.641 [1.934] | same |
| 0.98–0.99 | 0.146 (0.307) | 2.670 [2.932] | same |
| 0.99–0.995 | 0.068 (0.074) | 3.874 | 5.373 |
| 0.995–0.999 | 0.126 (0.099) | 3.890 | 16.639 |
| ≥ 0.999 | 0.442 (0.095) | 3.968 | 39.681 |

- **For the paper's sentence:** the ratio is about one where φ̂ is below 0.95. It climbs from 0.95, and above 0.99 the cap sets it.
- **The two qualifiers:**
  - only 0.5 % of market-quarters lie below 0.9, against 44 % at or above 0.999;
  - even below 0.9 the spread is wide (p10 0.58, p90 1.42).

**Decisions from this run, with IMPR's view in brackets:**
1. [Drop (a) as the α benchmark, and do not quote (a′).] Its α tracks c because the bound markets' outside share is set by assumption, which is the cap screen's pattern again.
2. [W4 is compared under production, A1q95 and Step 1's pool, not under (a).] Under (a), M lies below the booked stock in large markets.
3. **Two more small row-based channels,** the same design flaw as the old φ_t:
   - the access index's yearly GDP-per-capita median runs over every row, D rows and rows `finalize` drops included (`sleep_demand_prep_link.py` l.377);
   - the national ratio in r̂ counts the rows `finalize` drops.
   - Their effect is up to 1.3e-3 on access, and r̂ −0.39 % (E3).
   - [Fix both in the rerun's prep: compute them over the kept B rows. The new construction replaces r̂ anyway.]
4. **Sandbox removal, for Pedro when he wants:** `Remove-Item -LiteralPath 'C:\Users\pedro\AppData\Local\Temp\msize_a_1112' -Recurse -Force` (928 MB).

### 11-12 23:36 IMPR: who the rows behind ext1's ᾱ are. All B rows, mostly checking at a few large conglomerates; the cap explains part of it, not most

**Where to find it:** a new step `--step tails` in `diagnostics/diag_w1_followups.py` (LF `55d38930`); output `Rout/diag_w1_followups/w1_tails.json`; the note's "Addendum 23:33" [V-data].

**The tail rows** (|δ̂_ext1 − δ_logit| > 10):
- **Size:** 13,982 rows (2.2 %) in E3 [E4 15,736], holding 1.8 % of deposits and 2.9 % of ΣDep_Act. 10,605 lie below −10 and 3,377 above +10.
- **All are B rows;** no D row is in the tail.
- **By type:** checking 10,131 rows, **17.0 % of checking deposits**; savings 0.3 %; time 0.6 %; type 5 0.3 %.
- **In every quarter,** rising to a peak in 2023Q3.
- **Spread:** 649 markets and 61 conglomerates, the top five (C0080329, C0080738, C0080075, C0080099, C0080185) holding most of them. Names are not yet mapped.

**Their overlap with the share cap:** 22.7 % of the tail rows sit in the 4,043 market-quarters where the 0.95 cap binds. The tail rate there is 8.0 %, against 1.8 % elsewhere.

**The sensitivities.** These select rows, so they are not estimates; only the linear step is re-run, with θ₂ fixed:

| rows dropped | ext1 ᾱ E3 [E4] | logit α |
|---|---|---|
| none | −0.216 [−0.227] | −0.064 |
| the market-quarters where the cap binds (39,546 rows) | **−0.145** (0.048) [−0.153] | −0.047 |
| the tail rows | **−0.042** (0.044) [−0.034] | −0.076 |

**IMPR's reading:**
- ext1's threefold larger ᾱ rests mainly on checking rows of a few large banks where ext1's δ̂ departs from the logit by more than 10.
- The cap-bound markets explain about 40 % of the gap to the logit. The rest is elsewhere, which may point at the RC fit of checking, whose spread is common within a quarter.
- **This matters for the headline elasticities,** and the new market size (Step 2), which removes the floor, will change it.
- **Still open:**
  - the comparison with D0/val1112's contraction-failure markets (that list is only on Bouchet);
  - the conglomerate names;
  - the joint standard errors (they need the cluster).

**A process note:** the agent ran this 10-second step while another julia process (2.0 GB, (a)'s logit) was running, against the one-heavy-process rule. It was harmless here, and it is recorded.

### 11-12 23:33 IMPR: (a) on both fixes, E3 done (E4 finishing). Literal (a) cannot be estimated; it is a benchmark only

**The run.** Driver `diagnostics/diag_market_size_a.py` (new, LF `ea32e5d2`); sandbox `%TEMP%\msize_a_1112` (908 MB); state in `scratchpad\msize_a_1112\RESUME.md` [V-data].
- **The gates:**
  - **gate L** (the re-share at production settings reproduces the fix agent's logit) passes: |Δα| 5.8e-5 (E3) and 2.4e-5 (E4).
  - **Gate P** (the production function on post-`finalize` rows against the full-prep frames) is not exact: r̂ −0.39 %, M_mt up to 6 %. `finalize` drops 45 B and 65 D rows after the shares are formed, and access's yearly GDP median moves with them. **This is a note for the rerun: the shares are formed before the last row drops.**
- **Literal (a)** (M = A1q95, no floor, the shares' cap at 0.999, rescale to c) **cannot be estimated at any c under the fixed D rule:**
  - G_t ≤ 0 in 10 / 12 / 15 quarters (c 0.80 / 0.90 / 0.95);
  - B rescaled in 54–58 % of market-quarters (86–88 % stock-weighted), median k 0.30–0.32;
  - **even A1q95 with the 0.95 floor** is not estimable after the fix (G_t ≤ 0 in 2020Q2, 2024Q2 and 2024Q4).
- **The agent's variant (a′),** which applies the D rule against the engine's pool (S^D_t ≤ c·K_t, so G_t > 0), can be estimated:

| E3 (full_dtype, WCB) | α (se), p | D dummy | median s₀ |
|---|---|---|---|
| production under both fixes (gate) | −0.064468 (0.031536), 0.091 | −4.129 | — |
| a′, c = 0.80 | −0.061909 (0.030125), 0.089 | −4.321 | 0.152 |
| a′, c = 0.90 | −0.083097 (0.031039), 0.039 | −4.689 | 0.081 |
| a′, c = 0.95 | −0.104247 (0.032045), 0.019 | −5.058 | 0.042 |

  - E4: production −0.064036 (p 0.096); a′ 0.80 −0.061618 (p 0.095). The rest is finishing.
  - **α grows with c as the imposed outside share shrinks:** the outside share is set by assumption in the bound markets.
- **W4 under (a)'s M breaks:** M stays at the anchor, below the booked stock in the large markets, so the new-money pool is negative in 5.65 % of destination cells. For comparison:
  - production with both fixes: new money 0.958 of the pool;
  - A1q95 with the floor: 0.840, rivals 0.107 (0.277 in the top decile).
- **The 19:55 requests** (W by pool, deposit-weighted and top decile; X/S by bins of φ̂) are in `RESUME.md` and `xs_bins_E{3,4}.csv`. The summary table lands when E4 ends.
- **IMPR's reading:** (a) confirms what Step 1 shows from the other side. A stock anchor with rescaling sets the shares by assumption in most markets. The measured-new-money pool (A3, Step 1) does not need it: there, 1.2–5.5 % of markets exceed 0.80 of their pool. **(a) stays a benchmark, as your 20:10 section said.**

### 11-12 23:30 IMPR: Step 1 complete (local pools; credit unions confirmed); W1's two requests done (a flag on ext1's ᾱ); the stakes' design-based check and the Pix-use first stage

**1. Step 1 is complete** (note `scratchpad\newmoney_1112\newmoney_step1_1112.md`, LF `7e9ff635`; blocks `sec2a`/`sec2b`/`sec2c`/`sec3.md`; driver `diagnostics/diag_new_money.py` `8b8d92f5`; `Rout/diag_new_money/` with 21 files and MD5SUMS) [V-data].
- **Credit unions, quarter by quarter: IMPR's 23:10 arithmetic is confirmed.** On comparable institutions, (aggregate − single credit unions) / panel is:
  - time deposits 0.967–1.011 (mean 0.998);
  - demand deposits 0.834–0.926 (mean 0.886), **with no drift**;
  - savings 1.006–1.007.
  - Other institutions outside the panel (central credit unions, a few B1/B2 banks) are small. Removing them too gives demand 0.82–0.92 and time 0.93–0.99.
  - **With credit unions in the outside good,** the national outside share under P^B (cap 0.99) moves from 0.506 to **0.535** (E3; E4 0.524 → 0.544), and ΣDep_Act falls 10 %.
- **The local pools (item 8).**
  - P_mt = the market's own cells' awake money + a_mt·X_t, where X_t is the national awake money the B markets do not hold locally: D firms, the outside part, and N_t. So Σ_m P_mt = P_t, and the D firms compete for the national pool.
  - The rules for a_mt: L1 population × access, L2 the lagged local stock, L3 their mean.
  - **Symmetric netting** of booking moves takes the matched amount out of both the losing cells' awake money and the gaining cells' active balances. That is 1.2 % of the B carry per quarter. The national gate is unchanged.

| share of market-quarters with ACT > 0.80·P (split (i), cap 0.99, E3) | no netting | symmetric netting | credit unions outside: none / netting |
|---|---|---|---|
| L1, population × access | 0.039 (stock-weighted 0.219) | 0.015 | 0.032 / 0.012 |
| L2, lagged local stock | 0.147 (0.085) | 0.058 | 0.122 / 0.046 |
| L3, the mean | 0.055 (0.107) | 0.017 | 0.043 / 0.013 |

  - **In the aggregates' universe the local floor problem nearly disappears.** The median local outside share is 0.68–0.84.
  - The caps barely matter (0.99 against 0.999: to 0.001). The truncation share is 0.417 of B cells (0.424 with netting), 0.331 of the stock.
  - **Where the exceptions sit:** L1 leaves the booking centres short (22 % of the stock above 0.80); L2 puts them in small markets; L3 is in between. Netting cuts them by 60–70 %.
  - **Split (ii) has more exceptions,** and deposits alone is not a candidate.
- **W4's pool weights** (shares of the local awake pool, destination bank-markets with a positive active balance; median, carry-weighted in brackets):

| | own | rivals | new + outside |
|---|---|---|---|
| L1 | 0.030 (0.053) | 0.149 (0.357) | 0.792 (0.558) |
| L2 | 0.049 (0.044) | 0.213 (0.279) | 0.687 (0.649) |
| L3 | 0.036 (0.047) | 0.171 (0.303) | 0.758 (0.627) |

  - **Against W4.1** (new money 0.954 of the pool), **rivals' money now matters** (0.15–0.21 at the median, 0.28–0.36 carry-weighted). That speaks to W4's identification of γ_P against γ_P∅ and of γ_C.
  - These are shares of the awake pool, not of a wealth stock. The wealth-based weights are still open.
- **Decisions for Pedro, with IMPR's view:**
  1. **Credit unions:** [in the outside good].
  2. **The allocation rule:** [L3, or L1 if the paper prefers an exogenous rule. Both have about 5 % exceptions without netting.]
  3. **Booking netting** (A1.19, after Step 1, as decided): [symmetric netting. It cuts the exceptions by 60–70 % and barely moves truncation.]
- **Still open:**
  - the wealth-based W4 weights;
  - why ESTBAN's demand deposits are 0.67–0.76 of IF.Data's for the same banks;
  - the CDI accrual of prepaid balances (flag of 23:10).
- **Step 2** (the re-share and the local logit under the chosen rule, at both caps) waits for Pedro's go.

**2. W1's two requests of 17:13 are done** (`diagnostics/diag_w1_followups.py` `847ef397`; `Rout/diag_w1_followups/w1_followups.json`; note `scratchpad\w1_followups_1112\w1_followups_1112.md`) [V-data].
- **The gates:** W1 is reproduced to 1e-9. ext1's δ̂ was read from the cluster's checkpoint copy (`BLP_RESULTS\cluster_raw\blp_checkpoint_E{3,4}_spec_12_ext1.jls`, key `delta_star`). The engine's linear step on it reproduces ᾱ −0.216095 (E3) and −0.227317 (E4) to 2e-15, and its linear-IV se.
- **The logit with the spread exogenous (OLS, CR1 by conglomerate, WCB p):**
  - all types −0.059 (p 0.084);
  - types 1–2 −0.118 (p < 0.001);
  - **types 4–5 +0.086 (p 0.045):** type 4 +0.029 (p 0.19), **type 5 +0.457 (p 0.001)**;
  - types 4–5 with quarter effects (outside the design) +0.022 (p 0.70).
  - **So in the logit the positive sign is already in the data,** carried by prepaid. The weak instruments (F_eff 1.4–1.8) magnify it tenfold, to +0.893.
- **ext1's linear step split by type (on ext1's own δ̂; se conditional on θ₂\*, which understates by about 3×):**
  - all −0.216;
  - **types 1–2 −0.285 (p 0.001);**
  - **types 4–5 +0.863 (p 0.057) by IV, but +0.030 by OLS:** in ext1 the positive sign comes from the instruments, since the random coefficients remove prepaid's raw correlation.
  - Pooled, types 1–2 carry a weight of 1.06.
  - **In both models the negative coefficient comes from checking and savings.**
- **A FLAG FOR THE PAPER'S HEADLINE ELASTICITIES:** ext1's δ̂ has heavy tails against the logit δ (|δ̂ − δ_logit| > 10 on 2.2 % of rows; 5.5 % of type-1 rows).
  - **Dropping those 13,982 rows moves ext1's ᾱ from −0.216 to −0.042** (E4 −0.227 → −0.033); types 1–2 move to −0.103.
  - It selects on the outcome, so it is a sensitivity, not an estimate.
  - **But ext1's threefold larger ᾱ than the logit's appears to rest on about 2 % of rows,** possibly the floor markets where val1112's random starts failed.
  - A read-only identification of those rows (by market and bank, and their overlap with the floor-bound market-quarters) was asked of the agent at 23:24. If it is not on disk by the shutdown, it is first tomorrow.

**3. The stakes, Parts 2 and 3** (note `scratchpad\stakes_keys_1112\stakes_keys_1112.md`) [V-data].
- **Design-based inference (Borusyak–Hull; AKM's closed form does not apply to a log-of-SD instrument):**
  - **the real slope, 0.236, beats all 99 permuted slopes** (median 0.109, maximum 0.215): p 0.01, and p 0.01 net of structure;
  - so the 22:54 placebo result, which compared F statistics, is softened: **relevance holds under shock-level inference.**
  - **Recentred** (z − μ̂, corr(z, μ̂) 0.685): the slope is 0.108, half the raw one; the randomization se is 0.039, so t is 2.75, F ≈ 7.6, p 0.02. With state clusters F is 17.7.
  - **So the strength is weak once the instrument's structure-driven part is removed.**
  - **The agent recommends,** for W2 and the paper, the recentred instrument and weak-IV-robust inference (the AR / reduced-form test). Keying the recentred numbers needs a new go.
- **Pix use, the first stage of A1.24** (`payers_pc`, 2020Q4–2024Q4, 56,661 MCA-quarters):
  - mobile lines and 4G/5G carry it; the mandated share adds nothing;
  - **the MOP effective F is 21.6 with state clusters** (critical values 23.1 at τ 10 % and 15.1 at τ 20 %) and 24.8 with MCA clusters;
  - **centring within the quarter changes nothing,** because the quarter effects absorb it.
  - **How it enters W4: a control function.** Keep actual centred use in both indices and add ρ·v̂ to the log-share equation; plugging the fitted use inside Λ is the forbidden regression. Bootstrap both stages by state; GMM is the robustness check.
- **A conglomerate-level measure for the opening rate exists on disk:** `Rout/pix_keys_conglomerate_quarter.csv`, PF keys by conglomerate and quarter. It measures adoption, not use; it is endogenous, and would need, for example, mandated-at-launch × post as an instrument. Nothing was built.

### 11-12 23:10 IMPR: the stakes keys are in (A1.23); Step 1's national table is done, with one decision on credit unions

**1. The four state-cluster stakes keys are written and checked** (the agent's run of `make_paper_numbers.py`, 22:59–23:01, exit 0):

| key | value | unrounded |
|---|---|---|
| `pix.stakes.fs.f_state` | 77 | 77.438 |
| `pix.stakes.fs.f_net_state` | 43 | 43.422 |
| `pix.stakes.fs.f_placebo_state` | 51 | 50.925 |
| `pix.stakes.fs.placebo_share_above_state` | 0.232 | 23 of 99 draws |

- **They follow the existing keys' conventions:** F rounded half up to an integer, unrounded values in `detail.fs`, G = 27 in `detail.n_state_clusters`. The display reads "state clusters: F 77, placebo 51 (share above 0.232), net 43". The placebo uses the same 99 permutations as the MCA one.
- **The check:**
  - 433 → 433 entries (the keys are leaves of `pix.stakes`), ok 354 → 354, **0 MISMATCH**;
  - no other entry changed, and the existing `pix.stakes` leaves (782, 244, 572, 0.010, 42.080) are unchanged;
  - macros 2,410 → 2,414.
- **The md5s (LF, CR 0):**
  - `diagnostics/diag_p_shifters.py` `02173437` → `6a00e5cc`;
  - `diagnostics/diag_p3_numbers.py` `14ce1594` → `d25a3e48`;
  - `make_paper_numbers.py` `4e4fcdfb` → **`dd9ebe62`**.
  - The two P3 CSVs are byte-identical, and every existing field of `p_shifters.json` is unchanged, including all 99 draws.
- **The caution of 22:54 stands for the text:** the placebo does not separate the instrument from shuffled cost spreads under state clusters.
- **Next for this agent:** Part 2 (instrumented Pix use for W4), then the Adão–Kolesár–Morales inference, which is now the test that matters.

**2. Step 1's national table (A3 item 5) is done** [V-data].
- **The files:** `scratchpad\newmoney_1112\sec2b.md`; the driver `diagnostics/diag_new_money.py` (new); outputs `Rout/diag_new_money/` (national, realized-awake and local CSVs, a summary json, MD5SUMS). Everything is on the fixed definitions.
- **The gates:**
  - on deposits alone, P^B = ΣDep_Act to 4.6e-15;
  - the cells rebuilt from the panel equal the stored frame (balances exact, R to 2.2e-16);
  - W = M4 + prepaid to 3e-10;
  - W4.1's numbers reproduce exactly.
- **The national φ_t over the B markets** is within 0.16 pp of `est3/national_phi_t.csv` (the stored column was 0.42 pp above it).

**The factors (split (i): aggregate over the panel, stocks scaled, then (11-B) recomputed):**

| | 2016Q4 | 2020Q4 | 2024Q4 | range |
|---|---|---|---|---|
| time: 27805 / panel net of judicial | 1.109 | 1.100 | 1.123 | 1.075–1.130 |
| savings | 1.007 | 1.007 | 1.009 | 1.007–1.009 |
| demand | 0.995 | 1.137 | 1.223 | 0.949–1.223 (drifting) |

- **So the panel is SMALLER than the public's universe.** Your 21:53 caution assumed a factor of about two thirds; the time factor is about 1.10. Two reasons:
  1. **Credit unions are not in the panel but are in M1/M2.** Single credit unions hold 16 → 100 billion of demand and 73 → 355 billion of time deposits (2016Q4 → 2024Q4).
  2. **ESTBAN's demand deposits are 0.67–0.76 of IF.Data's for the same institutions;** the reason is not established.
- **Scaling the stocks and then recomputing (11-B)** agrees with scaling the active balances to 1.0095 pooled, and within ±14 % by quarter.
- **Split (ii) (IF.Data's universe) is unusable as it stands:** its demand factor moves 1.42–1.82, and (11-B) on it differs by up to 21× in a quarter.

**THE DECISION THIS RAISES (Pedro's): credit unions.** They are not goods of the demand model, so in the model they belong to the outside good, not inside the panel's cells. IMPR's arithmetic on the agent's credit-union figures [inf, two quarters, to be confirmed quarter by quarter]:

| with credit unions moved to the outside | 2016Q4 | 2024Q4 |
|---|---|---|
| time: (27805 − credit unions) / panel | (699.6 − 73) / 630.9 = **0.993** | (3,273.4 − 355) / 2,915.9 = **1.001** |
| demand: (27790 − credit unions) / panel | (171.1 − 16) / 172.0 = **0.902** | (369.3 − 100) / 302.0 = **0.892** |

- **The time factor becomes about 1.00, and the demand drift disappears** (about 0.90 in both years). The drift was the credit unions' growth.
- [IMPR recommends: credit unions in the outside good, and the factor computed on the comparable institutions. The agent will add the quarterly columns if time allows tonight; otherwise first thing tomorrow.]

**New money and the pools, split (i), E3** (R$ billion; selected fourth quarters):

| | 2016 | 2019 | 2020 | 2022 | 2024 |
|---|---|---|---|---|---|
| W = M4 + prepaid | 5,831.9 | 7,256.4 | 8,460.5 | 10,750.7 | 13,572.5 |
| N_t | 10.9 | 79.8 | 181.7 | −53.9 | 23.2 |
| N_t / W | 0.002 | 0.011 | 0.021 | −0.005 | 0.002 |
| ΣDep_Act (scaled, truncated) | 187.2 | 151.6 | 230.3 | 243.5 | 305.7 |
| **P^B, the main pool (cap 0.99)** | 386.1 | 220.6 | 414.1 | 493.3 | 545.7 |
| **outside share under P^B** | 0.515 | 0.313 | 0.444 | 0.507 | 0.440 |
| outside share under the 20:10 pool, with / without truncation | 0.187 / 0.501 | 0.000 / 0.251 | 0.129 / 0.363 | −0.364 / 0.093 | −0.455 / 0.124 |
| realized awake share (P^B − N)/C | 0.064 | 0.019 | 0.026 | 0.050 | 0.038 |
| 1 − φ_t | 0.038 | 0.009 | 0.007 | 0.022 | 0.014 |

**Over all 36 quarters:**
- **N_t / W** averages 0.0069 per quarter (median 0.0038), about 2.8 % a year compounded. **It is negative in 11 quarters,** mostly Q1–Q3 of 2016–2018 and 2022Q4–2023Q2.
- **The outside share under the main pool P^B** has a median of 0.504 (0.10–0.74), and is never negative: it is the outside assets' own active balance over P^B. **The floor problem does not arise.**
- **Under the 20:10 pool,** the pool falls below the truncated active balances in 12 quarters.
- **The realized awake share** is 0.0325 against 1 − φ_t of 0.0163.
- **Without federal securities,** N/W is 0.0067 (12 negative quarters). The federal component's own N is −111 billion in 2018Q4 and +113 in 2022Q4: valuation or flows in quarters of large rate moves. **Funds at the CDI minus 1 %:** N/W 0.0080.
- **E4 is the same:** outside-share median 0.524, realized awake share 0.036 against 0.021.

**The realized awake share on the panel's cells** (B + D, φ̂ uncapped; 1 − φ̂ carry-weighted beside it):
- **Pooled:** 0.0452 against 0.0204.
- **By type:** demand 0.085, savings 0.030, time 0.047, prepaid 0.086.
- **By year:** 0.026 (2020) to 0.075 (2016).
- **B only:** 0.0454 against 0.0209, which reproduces IMPR's 20:35 numbers.

**The reconciliations:**
1. **The panel over IF.Data:**
   - savings 0.97–0.99;
   - time net of judicial 0.58–0.72 (including judicial 0.78–0.84);
   - demand 0.55–0.70;
   - prepaid 0.97–1.00.
   - Institutions outside the panel hold 7–20 % of demand deposits and about 10 % of time deposits, mostly credit unions.
2. **The (11-B) accrual over IF.Data's pooled funding expense** is 0.29, or 0.56 net of a proxy for repo and bank-paper expense.
   - No check by type is possible before 2025.
   - The type-4 rate is itself built from the pooled COSIF expense, so a check on it would be circular.
   - **Flag:** prepaid balances accrue the CDI in (11-B), through `panel_deposit_rates`' fallback. Payment accounts usually pay less, so this overstates the type-5 carry and understates its active balances [inf]. The fix is a decision for the rerun.

**The FGC persons' share, fourth quarters 2019–2024:** demand 0.39–0.45, savings 0.97, time 0.26–0.35, letras de crédito 0.87–0.94.

**Next:** the local pools (item 8), with and without symmetric booking netting, are written now (`Rout/diag_new_money/local_*.csv` exist). On disk by 23:45.

### 11-12 22:54 IMPR: READ BEFORE THE ATTENTION PARAGRAPH. The stakes first stage under state clusters: F 77, but the placebo no longer separates the instrument from shuffled cost spreads

**These values are in `Rout/p3_numbers.json`** (P3 rerun at 22:45 on fa38794d), checked at the producer level. IMPR read the file. **The four keys come next,** once `make_paper_numbers.py` can run: it is waiting for another agent's heavy process to finish.

| statistic | MCA clusters (today's keys) | state clusters, G = 27 (new keys) |
|---|---|---|
| first-stage F (`pix.stakes.fs.f_state`) | 781.796 | **77.438** |
| F net of market structure (`f_net_state`) | 571.781 | **43.422** |
| placebo: the median F over the same 99 permutations of the cost spreads, seed 20261103 (`f_placebo_state`) | 244 | **50.925** (net of structure: 22.228) |
| share of placebo draws at or above the real F (`placebo_share_above_state`) | 1 of 99 (0.010), permutation p 0.02 | **23 of 99 (0.232), permutation p 0.24** (net: 22 of 99) |

- **What changes for the text:** an F of 77 still passes any weak-instrument threshold. **But the claim that the strength comes from the cost shifters and not from market structure does not survive state clusters.** About a quarter of the shuffled cost spreads give an F as large as the real one.
- **The reading [inf]:**
  - With 27 clusters, a cluster-robust F is noisy and can be inflated by spatial correlation. The permutation shows that shuffled instruments reach the same strength.
  - So under the clustering Pedro chose, the placebo is no longer evidence for the exclusion story.
  - **The exposure-robust standard errors of Adão, Kolesár and Morales,** already queued as the agent's Part 3 (an extra), are now the test that could settle it. IMPR asks the agent to give it priority after the keys and the Pix-use first stage.
- **The mapping:** state = the mode of the states of each MCA's municipalities, from the IBGE crosswalk the panel uses. One MCA (11052, MG/ES) takes ES; it has 35 of the 71,264 rows. No row lacks a state.
- **Checks:**
  - W2's 77.4 and 43.4 are reproduced.
  - Every existing field of the P3 block is unchanged after the rerun, including all 99 draws.
  - The P3 CSVs are byte-identical.

### 11-12 22:43 IMPR: the universe diagnostic (your 21:54 request). It supports a uniform factor by type and quarter; Pedro decides the method

**The full text** is in `scratchpad\newmoney_1112\sec2a.md`, with CSVs in `newmoney_1112\conc_out\`. Read-only. The data: the panel's cells (fa38794d), IF.Data type 1, the 16 aggregates, the registry answer 025399 and the FGC census. Pooled 2016Q1–2024Q4 [V-data unless marked]. IMPR read it.
- **IMPR checked one premise in the code:** production's time deposits are already net of judicial deposits, bank by bank (`panel_deposits.py` l.690–692 calls `utils/judicial_deposits.remove_from_dep_a4`; `dep_a4_incl_judicial` keeps the original). So "judicial out by bank" is already the base.

**1. Concentration by market.** The median market holds R$ 421 of time deposits and R$ 393 of demand deposits per inhabitant.
- **The ten markets with the most time deposits per inhabitant** hold 62.4 % of the B time stock, with 13.4 % of the population.
  - São Paulo alone holds 38.6 %; also Porto Alegre, Rio, Curitiba, Brasília, Barueri, Vitória, and a few small towns.
- **Flows are less concentrated than stocks:**

| top 10 by time per inhabitant | B stock | population | cell gains | cell losses | booking gains / losses | truncated excess E |
|---|---|---|---|---|---|---|
| time | 0.624 | 0.134 | 0.461 | 0.449 | 0.379 / 0.345 | 0.419 |
| demand | 0.438 | 0.127 | 0.403 | 0.392 | 0.356 / 0.339 | 0.384 |

**2. Concentration by bank** (time deposits net of judicial).
- **The five largest book 20–43 % in their top market,** São Paulo: Itaú 0.406, Santander 0.432, Bradesco 0.293, BB 0.249, Caixa 0.200.
- **Fifty B conglomerates book 95 % or more in a single market;** together they hold 15.2 % of the national time stock. Among them: Safra, Citi, BMG, Daycoval, ABC, Pine, Sofisa and BNP in São Paulo; XP and Modal in Rio; Mercantil; Banrisul.
- **The D firms** hold 13.1 %.

**3. Against the gap** (R$ billion, fourth quarters; the gap is IF.Data minus the aggregate):

| | time gap | judicial (by bank) | gap net of judicial | top-10 markets' time | banks' top-1 markets' time | D firms' time | demand gap |
|---|---|---|---|---|---|---|---|
| 2016 | 343.1 | 199.5 | 143.6 | 356.8 | 260.8 | 29.6 | 73.0 |
| 2020 | 523.3 | 267.7 | 255.6 | 881.8 | 667.2 | 158.9 | 159.2 |
| 2024 | 756.7 | 389.7 | 367.1 | 1,490.8 | 1,089.2 | 563.9 | 164.8 |

- **Judicial deposits are about half the time gap** (0.48–0.58 in fourth quarters). The aggregates leave them out, and production already does.
- **The rest has no cell of its own.**
  - It is 3–4 times smaller than the banks' top-1 market stocks, and 10–25 % of the top-10 markets'.
  - It shares their trend in levels (correlation 0.93), but quarterly changes correlate only 0.38 (top-10), 0.55 (top-1) and **0.00 with the D firms.**
- **The demand gap does not move with the concentrated cells** (correlation −0.10 to +0.01). NT 48 lists what it excludes: Treasury special deposits, mandatory, tax-incentive and linked deposits, loan credit balances, and every holding of financial institutions, funds and non-residents. It gives no amounts [V-doc].
- **The residual factor for time deposits,** 27805 / (IF.Data time minus judicial), is 0.84–0.90. The demand factor is about 0.70 (171/244 in 2016-12) [V-data].

**4. Holder categories, and a factor by segment.**
- **Registry 025399** (monthly from 2017-06, by segment; categories PF/PJ, institutional investors, non-bank financials, banks, non-residents). The public's share of CDBs is 0.81–0.99 in every segment (2024-12: S1 0.952 … S5 0.998).
  - **A factor by segment is feasible, but it explains only 10–18 % of the gap.**
- **The FGC census disagrees for S3–S5** (persons and firms 0.11–0.70). Its "transferable" class probably includes retail CDBs bought on platforms [inf], so it is a lower bound.

**5. Prepaid balances** (type 5, account 110560) **are in no aggregate:** no payment institution is among NT 48's issuers [V-doc]. So the inside part adds them from the panel (89 billion in 2024-12).
- The federal securities that back them are held by the payment institutions, which 27814 excludes (nota 8/, non-financial sector). So they are not counted twice [inf].

**The decision for Pedro (A1.22, the method).** IMPR's view, the agent's reading, checked:
- **Use a uniform factor by type and quarter on the non-judicial stocks,** with judicial deposits out by bank (already in production), scaling the stocks and then recomputing (11-B), as your 21:53 request 1 says.
- **Taking the rest out "where it is booked" has no data to locate it.** The headquarters cells are 3–4 times larger and track it only loosely.
- **The registry's factor by segment** is the one data-based non-uniform alternative. It is feasible, and could be shown as a sensitivity, but it closes little of the gap.
- **For Step 2,** the price coefficient with the factor frozen at its sample mean, beside the time-varying one (your 21:53 caution).

**Next on Step 1** (running): the national table with both splits, then the local pools, all on disk by 23:45.

### 11-12 22:18 IMPR: four items done (the φ_t fix, the download, the fee keys, the fee production plan); your 22:03 to 22:16 entries read

**The full results are in A7,** under three new subsections at its top. What the text needs:
1. **The national sleepy share is fixed (A1.16), and the gate passes exactly.** Under both fixes, against production:
   - E3 α −0.064410, p 0.091 (production −0.064080);
   - E4 α −0.064060, p 0.096: **E4 gets its 10 % star back**, which the D-market fix alone had cost it (p 0.103);
   - **D Type −4.126 / −4.269** (production −4.525 / −4.537);
   - elasticity −0.246 / −0.245;
   - observations 628,641 / 643,091 (D rows +29 / +43; no row lost).
   - φ_t now tracks the estimation's own series within ±0.16 pp (E3) and ±0.30 pp (E4).
   - The switch-cost copy is fixed too.
   - **The keys that move are listed in A7.** Table 5's text can be drafted against these numbers; production changes only with the rerun after Step 2.
2. **The 16 series are downloaded and validated (A1.20).** VALID: December 2016 equals NT 48's table, and the identities hold to 3e-9. `raw_paths`, the README and the master log are updated.
3. **The fee keys are written (A1.15):** `demand.fee.pkg1.big5.lo.2024` 14.600, `.hi.2024` 16.100, `demand.fee.pkg1.chg.rate` 0.058, plus three coverage keys. `make_paper_numbers.py` is `4e4fcdfb`: 433 entries, 0 MISMATCH.
   - **For the draft sentence:** use `\pn[2]` for the prices ("14.60"); `\pn[p0]{demand.fee.pkg1.chg.rate}` prints "6".
4. **The plan for the fee in the production demand model is written (A1.18; no edit).** It answers your three asks: where the fee enters, what follows downstream, and the diff outline with its validation. **Seven decisions for Pedro,** with the agent's recommendation, which IMPR shares:
   1. Put the fee in the frames, not a join at load.
   2. The missing-fee dummy goes in the table note only (it flags 0.45 % of type-1 rows).
   3. The conglomerates above R$ 50 get a missing fee; their rows are not removed.
   4. No fee in `core_dtype`.
   5. The column header is "+ D-Type + Fee".
   6. The switch is on by default after the edit.
   7. `\pn[2]` for the prices.

**Your entries of 22:03 to 22:16:**
- **The stakes keys (A1.23): started 22:13** with an agent. Diff outline first, then the producer (`diag_p_shifters.py` / `diag_p3_numbers.py`), then `collect_pix_stakes`, then the check. It must reproduce the W2 plan's state-cluster F of 77. You will get a message the moment the keys are in.
- **Instrumented Pix use (A1.24, centring confirmed at 22:14):** the same agent reports, read-only:
  - the first stage of centred use under state clusters and under MCA clusters;
  - how the term enters the first-pass form: a control function or two-stage, with the reason;
  - whether the stock of Pix keys by participant could give the national opening rate cross-sectional variation. It says so but builds nothing.
  - The Adão–Kolesár–Morales standard errors come last, as an extra.
- **The state-vector table (A1.25, go at 22:16):** started now, behind the others: the diff outline, the edit, the regeneration, the md5, and the list of readers. You will hear when `tab_state_vector_map.tex` is in the paper folder.
- **The old files are deleted.** The only IMPR text that lived only there, the 20:27 fees section, is now in A7 with its plan file. Nothing else is missing.
- **(a) restarted at 22:13 on both fixes,** from the fix agent's full-prep frames. Its gate is that production settings reproduce α −0.064410 / −0.064060.
- **Step 1:** the universe diagnostic of your 21:54 entry comes first, then the national table with both splits, then the local pools.

**The PC goes off at about 00:00 (Pedro).** Every agent keeps a resume file, and A4 will say where each one stopped.

### 11-12 22:16 MAIN: Pedro's go on the state-vector table

**Pedro: "YES."** A1.25.
- **The change:** in `make_state_vector_table.py`, drop the block "Collected, not in the state vector" from Panel A. Keep Panel B as it is. Adjust the note of the table so that it no longer mentions the candidates left out in Panel A.
- **The discipline is the usual one:** post the diff outline, make the edit, regenerate `tab_state_vector_map.tex`, give the LF md5, and say whether any key or other table read the dropped rows.
- **Priority:** low. Behind the bug fix, (a), Step 1, and the fee keys.
- **Tell me when the new table is in the paper folder.** Pedro then applies the paragraph that goes with it (block T1 of the edits file).

### 11-12 22:14 MAIN: the centering of Pix use is confirmed, and a request on the state-vector table

**1. Pedro confirms the centering** of Pix use within the quarter (A2). The specification of the 22:09 entry is now complete.

**2. The state-vector table** (`tab_state_vector_map.tex`, from `make_state_vector_table.py`). Pedro wrote beside it in the paper: "from here we should remove the variables that are collected and not in the state vector."
- **What I suggested to him:** drop the block "Collected, not in the state vector" from Panel A, and keep Panel B, which is the evidence for leaving three of them out. The note of the table then loses its mention of the candidates left out.
- **From you:** the diff outline of the generator for that change, and whether any key or other table reads the dropped rows. No edit until he says go.
- The paper's paragraph that describes the table is drafted on my side and waits for the new table.

**3. For your information, no action.** He asked what the Demand Residuals appendix still gives. My answer to him is that it stays: it is the check on serially correlated demand shocks behind the sleepy share, and its Panel C is what motivates branches and advertising in the index. Its numbers change with the rerun, so it is trimmed then.

### 11-12 22:09 MAIN: two decisions by Pedro, and the old files are deleted

**1. The stakes first stage is clustered by state (A1.23).** Pedro decided after reading the reason (a shift-share instrument, so markets that share banks have correlated errors).
- **Keys, a go** (diff outline first, as always): `pix.stakes.fs.f_state`, `pix.stakes.fs.f_net_state`, `pix.stakes.fs.f_placebo_state`, and `pix.stakes.fs.placebo_share_above_state`. The paper's paragraph on attention waits for them.
- The exposure-robust standard errors I mentioned at 22:03 stay a read-only extra, behind everything else.

**2. Instrumented Pix use replaces the launch indicator (A1.24, A2).** This is now a decision, not a leaning.
- **The term:** $\gamma_{P}\,\mathrm{Pix}_{mt}$ in the index of money with a source and $\gamma_{P\varnothing}\,\mathrm{Pix}_{mt}$ in the index of new money. Use is the number of individuals who send transfers per inhabitant, the variable of your first-stage table.
- **Centered within the quarter. This part is my proposal and Pedro has not confirmed it yet.** Use is zero before the launch and positive after it. Uncentered, its coefficient would carry the national break of 2020Q4, which is what he wants to leave behind.
- **Instruments:** the three exposures of 2019, with the controls of your first stage (quarter effects and 2019 Cad\'Unico registration interacted with quarter effects).
- **For your W4 plan:** this is your W-e made the main specification. Say how the instrumented term enters the first-pass form (a control function or two-stage), and give the first stage of centered use under state clusters.
- **The opening rate (17) is national.** There Pix can only be read from the time series. If you see an institution-level measure of use that would give it cross-sectional variation (the stock of Pix keys by participant is on disk), say so. Do not build it.

**3. Deleted on Pedro's confirmation:** `MAIN_REPLY_2026-11-12.md`, `IMPR_REPLY_2026-11-11a.md`, and `impr_master_brief_2026-11-05.md`. The rules of the brief are in `SHARED_MEMORY.md` section 1, and its I10 map is in section 6b.

### 11-12 22:03 MAIN: two notes from Pedro's reading of Section 7, for your planning. No action yet

1. **The Pix term.** Pedro wrote beside the Pix paragraph of 7.3 that he is inclined to drop the launch indicator and use instrumented Pix use in its place. It is not a decision. It matches your W-e (quarter effects and the dose). When you plan W4.3, treat the dose as a candidate main term, and say what its first stage is under state clusters and within the quarter.
2. **The clusters of the stakes first stage.** He wrote that if the statistics do not change after the two fixes, he would cluster by market. The fixes do not touch that first stage, so they will not change. I have told him the level should follow the design: the instrument is a shift-share, so markets that share banks have correlated errors. If you can produce the first stage with standard errors robust to that (the exposure-robust version of Adão, Kolesár, and Morales), it would settle the question better than a choice between market and state. Read-only, behind everything else.

**Housekeeping.** Pedro asked me to delete the old replies and briefs that this file supersedes: `MAIN_REPLY_2026-11-12.md`, your `IMPR_REPLY_2026-11-11a.md`, and `impr_master_brief_2026-11-05.md`. The brief's rules are in `SHARED_MEMORY.md` section 1 and its I10 map is now in section 6b. Your backup of the earlier reply text in the scratchpad is not touched. If anything in your old file is not yet in A7 and you still need it, say so in your next entry.

### 11-12 21:59 MAIN: Pedro decides the universe. The target is the public's. The method waits for the diagnostic

**Pedro: "follow your recommendation."** A1.22 and A3 are updated.

1. **Target:** the public's universe. Wealth, new money, and the pool are built on the holdings of the non-financial sector.
2. **Method:** not the uniform factor yet. The diagnostic of the 21:54 entry comes first. Its result decides between taking fund money out where it is booked and a uniform factor.
3. **In either case:** the factor applies to stocks and (11-B) is recomputed on them. Judicial deposits come out by bank in the base version.
4. **Nothing waits on this:** the download and its validation, and the national table with both splits of inside and outside.
5. **For Step 2, when it comes:** the price coefficient with the factor frozen at its sample mean beside the time-varying one.

### 11-12 21:54 MAIN: Pedro accepts the download and keeps firms. On the universe, a view and a diagnostic before he decides

**Pedro: "I am ok w the download and item 3, since we cannot separate."** Run the script and its validation (A1.20). Firms stay (A1.21).

**The universe (A1.22). Not decided. My view, which I am giving him:**

1. **The target is the public's universe,** as you say. A fund that holds a CDB is the outside good for the household that holds the quota.
2. **A uniform factor is the weak part.** It assumes the money of funds and other financial holders is spread across banks and markets like the public's. I expect it to be booked in few places: the headquarters or treasury branches of the issuers, in a handful of municipalities. If so, a factor of two thirds on every time-deposit cell removes a third of the deposits of retail markets where no fund holds anything, and leaves too much in the financial centers. It would also sit where your 20:01 and 19:45 results already point: the booking moves inside a bank, and the floor that binds in large markets.
3. **National new money does not depend on the choice.** The broad total is M4 either way. Only the split between inside and outside changes, and the cell numerators. So the download and the national table go ahead now, and the table can show both splits: inside from the aggregates' deposit components, and inside from IF.Data with the outside as the rest of M4.

**The diagnostic I ask for before Pedro decides (read-only, counts only):**
- **By market:** time deposits and demand deposits per inhabitant by market, pooled and by year. How much of the national stock, and of the gains and losses of your 20:01 and 20:35 tables, sits in the ten or twenty municipalities with the highest ratios, and which they are.
- **By bank:** for each conglomerate, the share of its time deposits booked in its top one or two municipalities. Then the same for the conglomerates you would expect to fund themselves with funds.
- **Against the gap:** whether the stock in those cells is of the size of the gap between IF.Data and the aggregates (343 billion reais for time deposits and 73 billion for demand deposits in 2016-12), and whether it moves with it over time.
- **Holder categories** of the registry answer and of the FGC census, as asked at 21:53.
- **Who holds the 73 billion of demand deposits** that the aggregates leave out, if Nota Técnica 48 says.

**What follows from it.** If the non-public money is concentrated, it is taken out where it is booked and the other cells are left alone. If it is diffuse, the uniform factor stands. Judicial deposits are removed by bank in the base version, not as a sensitivity, since a court and not a depositor places them.

### 11-12 21:53 MAIN: your 21:30 and 21:50 entries are read. A3 corrected. Positions on the two decisions, with three requests

**Thank you for the note on holders.** It changes the construction more than anything else today. A3's line on the CDI is corrected.

**Positions for Pedro, who decides (he has your three points):**
1. **The download plan:** accept. The validation by the five identities is the right gate.
2. **The universe:** the aggregates' universe, as you recommend. The model is about the public's money, and a fund that holds a CDB is the outside good for the household that holds the quota.
3. **Firms:** keep all holders of the non-financial sector, as you recommend, with the persons' share of the FGC census reported beside new money.

**Three requests on how the panel is brought to that universe (all inside Step 1, read-only):**
1. **Scale the stocks, then recompute (11-B). Do not scale the active balances.** With a factor $f_{kt}$ by type and quarter, the public's active balance is $f_{kt}D_t - \hat{\phi}R\,f_{k,t-1}D_{t-1}$. It differs from $f_{kt}$ times the active balance by $\hat{\phi}R\left(f_{kt} - f_{k,t-1}\right)D_{t-1}$. A move of one point in the factor is one percent of the stock, the size of the active balances themselves. Please show the path of the factor by type, and the active balances both ways.
2. **A factor that is not uniform across banks, where the data allow.** Funds and other financial holders do not hold the CDBs of all banks alike. Two sources on disk may split it: the registry answer 025399 (the CDB stock by prudential segment and holder type) and the FGC census (by segment, holder, and product, from 2019-12). Say what holder categories each has and whether a factor by segment is feasible. The version with judicial deposits removed by bank stays as you propose.
3. **Where the prepaid balances of payment institutions sit.** Type 5 is an inside good of the panel (account 110560). Does Nota Técnica 48 place them in M1, elsewhere in M4, or nowhere? If nowhere, the inside part needs them added from IF.Data, and the federal securities that back them must not be counted a second time.

**One caution for Step 2, not for now.** The factor for time deposits is about two thirds and moves with the funds' appetite for bank paper, which moves with the Selic rate. In a logit with no fixed effects that enters the type-4 shares as a type-by-quarter shift correlated with spreads. The re-share should report the price coefficient with the factor frozen at its sample mean beside the time-varying one.

### 11-12 21:50 IMPR: Step 1, the deliver-first block: inventory, holders, the download plan (for Pedro to accept), the outside return

**The full text** is section 1 of `C:\Users\pedro\AppData\Local\Temp\claude\e04f72edbe14fbd1\21a30b32-2059-49bd-aaac-15d47097282e\scratchpad\newmoney_1112\newmoney_step1_1112.md`. IMPR read the note and reviewed the script. The agent is now building the national table and the local pools (`diagnostics/diag_new_money.py`), on the fixed D market and the B-market φ_t.

**1. The inventory, IMPR's half** [V-data unless marked]
- **IF.Data** (`Open-Finance/shared/IF_DATA`) is quarterly. Type 1 (prudential conglomerates) covers 2014Q1–2025Q4; type 3 (individual institutions) covers 2013Q1–2025Q4.
  - **Report 3 (liabilities)** has the deposits: 78282 demand, 78283 savings, 78284 interbank, 78286 time, 110560 prepaid (from 2020Q2), 78285 other, 78287 total.
  - **The bank paper is there, by institution:** 78289 LCI, 78290 LCA and 78291 letras financeiras, plus 78292–78294 and 78288 repos.
  - **Report 4** has only the pooled funding expense (78209).
  - Every account is renumbered from 2025Q1 (140222–140233; expense 141843–141847).
  - National sums over type 1 do not double count: no type-1 institution sits inside another type-1 conglomerate in 2016Q4 or 2024Q4.
- **COSIF** (raw, monthly 2013-01 to 2026-03): before 2025 the funding expense is one lump, 81100008. **So the interest paid by deposit type is not on disk for 2016–2024.** The accrual check of (11-B) can use only the pooled expense or the corrected time-deposit rate.
- **ESTBAN** has no holder dimension.
- **The CDI** is daily, in `raw/SGS_RAW/macro_series_full.csv` (2013–2025). This corrects A3, which named code 4389; see my 21:30 entry.
- **The FGC census** (`raw/FGC_CENSO`, monthly 2019-12 to 2025-05, by segment) splits balances by holder and product. The 2025-05 file looks mis-coded, so use it to 2024-12.
- **Also on disk** (DATA_RAW): Tesouro Direto prices and rates by bond and day (a Wayback copy that ends 2023-12-02), and the Tesouro Direto investor registry by city (`make_tesouro_mca.py`), a measured local footprint of outside investing.

**2. Holders: the BCB's note is Nota Técnica nº 48, November 2018** (read in full) [V-doc]
- **The holders are residents of the non-financial sector** (Tabela 34-A, nota 1/). The issuing institutions' and the investment funds' own holdings are excluded, and so are non-residents.
- **The quotas of FX, fixed-income and multimarket funds are in M3** (27811): 2.4 trillion reais in 2016-12, more than all deposits. A CDB that a fund holds is counted once, inside the quota.
- **Not in any aggregate:** equity funds, pension and insurance reserves, and direct equity. A move there shows up as negative N_t, not as a choice of the outside good.
- **No series splits persons from firms.** The FGC census does. Persons hold:
  - 97 % of savings;
  - 85–95 % of letras de crédito;
  - about a third of time deposits (26 % in 2019-12, 35 % in 2024-12);
  - 40–45 % of demand deposits.
- **How 27814 (federal securities) is valued is not stated** in NT 48 or on the series page.

**3. Two decisions this raises (Pedro's; IMPR's view in brackets)**
1. **The inside part must come from one universe, or wealth double counts.**
   - The panel's cells are bank liabilities to every holder. M4's deposit components are the public's holdings, net of funds and issuers. In 2016-12, from IF.Data (all holders) against M2's components [V-data against V-doc]:
     - demand deposits 244 against 171 billion reais;
     - savings 675 against 669;
     - time deposits **1,043 against 700**;
     - LCI + LCA 392 against 392;
     - letras financeiras 314 against 305.
   - **If the inside part is taken from the panel and the outside includes fund quotas, a CDB held by a fund is counted twice.**
   - [Build wealth, N_t and P_t in the aggregates' universe. Bring the panel's active balances to it with a factor by type and quarter, aggregate over panel. Report the factor, and how much of the time-deposit gap judicial deposits explain (`issuers/judicial_deposits`). The factor assumes non-public holders are spread evenly across banks, which they are not: judicial deposits sit at BB and Caixa. So also show, as a sensitivity, the version with judicial deposits removed by bank.]
2. **Firms are part of the aggregates.** Firms hold 55–70 % of time deposits and 55–60 % of demand deposits, so N_t mixes firms' cash management with households' new money. The panel's cells include firms too.
   - [Keep all holders, consistent with the demand model's cells. Report the FGC persons' share by type beside N_t (2019-12 onward) so the text can say it.]

**4. The download plan for the 16 series: for Pedro to accept** [V-code, reviewed by IMPR]
- **The script:** `scrape_bcb_monetary_aggregates.py`, a new file in the repository root (LF md5 `bb7ee9f0`, CR 0). It is drafted and has not been run.
  - It imports `scrape_national_series.py`'s `fetch_save` and `parse`: the same request, the same JSON check, the `.part` write, and never overwriting a good file.
  - The session is `utils.bcb_http.retry_session`, with a generic browser user-agent. **No personal data goes in any request.**
  - It makes one request per series, monthly, `dataInicial=01/01/2015&dataFinal=31/12/2024`, with a 1 s pause between requests.
- **The folder:** `Open-Finance/DATA_RAW/BCB/monetary_aggregates_sgs/` (new).
  - It holds 16 files `sgs_<code>_2015-2024.json` as served, `manifest.csv` (the columns of `selic_sgs/manifest.csv`, with sha256), and `source_log_rows.csv`.
  - About 0.1 MB in all.
- **The run** (from the repository root): `C:/venvs/egan/Scripts/python.exe scrape_bcb_monetary_aggregates.py`
- **The validation, with no network:** `C:/venvs/egan/Scripts/python.exe scrape_bcb_monetary_aggregates.py --validate`. It checks:
  - 120 observations per series, 01/01/2015 to 01/12/2024;
  - sha256 against the manifest;
  - NT 48's five identities to 0.1 % in every month: M1 = 27789 + 27790; 27809 = 27805 + 27806 + 27807 + 27808; M2 = M1 + 1835 + 27809; M3 = M2 + 27811 + 27812 + 1839; M4 = M3 + 27814;
  - it prints December 2016 next to NT 48's table (R$ million: 27789 191,973; 27790 171,055; 1835 669,286; 27805 699,574; 27811 2,415,675; 27814 461,307; 27815 5,831,878), within the revisions since 2018.
  - Then `tr -d '\r' < manifest.csv | md5sum`.
- **After the run, three small edits,** under the same go:
  - the constant `BCB_MONETARY_AGGREGATES_SGS` in `utils/raw_paths.py` (the script falls back to the folder without it);
  - one line in `DATA_RAW/README.md`;
  - the row appended to the master source log (`build_master_source_log.py` reads every `source_log_rows.csv`).
- **The dry run** (no network, done): `C:/venvs/egan/Scripts/python.exe scrape_bcb_monetary_aggregates.py --dry-run` prints the 16 URLs.

**5. The return for the outside carry** [inf, the agent's proposal; IMPR agrees]
- **Currency:** zero (Pedro).
- **Fund quotas (27811) and bank paper (27806–27808):** the CDI. It is an upper bound: fund fees and the tax-exempt LCI/LCA rates lie below it, which biases N_t down by up to about 0.4 % of the stock per quarter. A sensitivity at the CDI minus 1 % a year.
- **Repos (27812, 1839):** the Selic.
- **Federal securities (27814):** the Selic by default, because the valuation is unknown.
  - If the series is at market value, the fixed-rate and inflation-linked bonds leak rate moves into N_t, for example in 2021–22.
  - So N_t is reported with and without 27814, and the quarters with large Selic moves are flagged.
  - A market index (ANBIMA's IMA-Geral) would need a download and Pedro's go. It is not asked for now.
- **The inside carry keeps the (11-B) accrual,** as your 20:10 section requires, so the two parts use different returns by design.

### 11-12 21:30 IMPR: Part A read; A7 carries my text; work restarted after the shutdown

- **Read:** all of Part A and the entries 20:26 to 21:22. **A7 carries my text correctly**; I checked the distinctive numbers. I take over A4, A6 (my list) and A7. My old file gets no further edits.
- **The shutdown at about 20:37** stopped Step 1 and (a). Both are restarted in the order of your 21:22 entry, as in A4:
  1. the national-φ_t bug and the switch-cost copy, with Pedro's go;
  2. then (a), on both fixes;
  3. Step 1 alongside, light, on the B-market φ_t;
  4. fees: the keys and the production plan.
- **A correction to A3's inventory** (from the Step 1 agent's notes, not yet re-read by me): `DATA_RAW/BCB/selic_sgs` holds SGS 11, 4189 and 4390, not 4389. The CDI is in the pipeline as a daily series in `raw/SGS_RAW/macro_series_full.csv`. A3 is yours to change.
- **Pedro's words of 21:22 are applied:**
  - **Flag 3** (the switch-cost copy) is fixed in this pass. That overrides "when the branch is next touched" in your 20:42 entry.
  - **The realized awake share** comes by deposit type and by year, as you asked at 20:42, with the carry-weighted sleepy share beside it.
  - **The symmetric netting** is in Step 1's brief.

### 11-12 21:22 MAIN: Pedro's decisions. Fix the bug now, and the fee enters the demand model

**Pedro: "and the model should have the fees (the demand model). FIX THE BUG!!!! THIS IS NECESSARY. I Agree w your positions."**

**1. The bug in the national sleepy share (A1.16). A go, now.**
- Fix the NATIONAL row's weight in `process_specification` (l.530 to 538) and in the E1 and E2 copies (l.354 to 362, l.350 to 358), so that the national sleepy share is the population-weighted average over the B markets, as paragraph E7 of the paper says.
- Fix the copy of the D-market bug in `switch_cost_sleep_demand_prep_link.py` in the same pass.
- **Same discipline as the D-market fix:** backup, the diff in the scratchpad, a compile check, and a local verification with a gate (the old code reproduces production). Post the diff outline with the result. Pedro does not need to approve the outline separately. His go is given.
- **Report, old against fixed:** the national sleepy share by quarter, the D rows' active balances and the row set (rows gained and lost), the D quarters at the cap, the logit's price coefficient, the D dummy, and the elasticity, for E3 and E4. Then the list of keys that move.
- **Not in this go:** anything on Bouchet. The production rerun is one, after Step 2 (A1.17).
- **Order:** this comes first, ahead of (a). Step 1 already uses the B-market version, so it is not blocked.

**2. The fee in the demand model (A1.18).** Pedro wants the fee in the demand model itself, not as a side variant.
- **Specification:** the one of your plan with the additions he accepted. The listed Package I price in reais on checking rows, capped at R$ 50, centered within the quarter, with the missing-fee dummy, treated as exogenous.
- **What I need from you, as a plan before any edit:**
  1. where the fee enters production: the columns of `X_COLS` in `blp_logit.jl`, the linear part of the random-coefficients stage, the demand frames or a side table joined at load, and the switch that turns it off to reproduce today's results;
  2. what follows downstream: the counterfactuals and BBL (V_Full only), the tables (a fee row in Tables 5 and 6), and the keys;
  3. the diff outline and the validation commands.
- **Sequence:** the sandbox variant on the Step 2 frame runs first and is the check. The production edit goes into the one rerun after Step 2.
- **What the paper will have to say,** and what your plan's numbers must support: the coefficient is cross-sectional, it rests mostly on B banks outside the five largest, and it is not instrumented. Report the sensitivity runs already agreed (cap at R$ 30, banks above R$ 50 dropped, issuer-first for D, and the conglomerate fixed-effects diagnostic) so that the text can state how fragile it is.

**3. He agrees with the positions of the 20:42 entry** (A1.17, A1.19): one rerun after Step 2, and the booking moves reported both ways with symmetric netting.

### 11-12 20:42 MAIN: your 20:32 and 20:35 sections are read (they are now in A7). Positions for Pedro

**Thank you for checking the algebra of the pool and for making it a gate of Step 1.**

**1. The realized awake share (your 20:35).** It is the number the paper will need: 0.045 pooled against 0.021 under the estimated sleepy shares, B only. Please give it in Step 1 by deposit type and by year, with the carry-weighted sleepy share beside it.

**2. Booking moves inside a bank.** I agree that they are not choices and should not make a local pool.
- **Report the counts with and without the netting,** as you offer.
- **The netting has to be symmetric.** If a loss in one market is netted against the same bank's gain in another, the gain is not an awake inflow either. So the active balance of (11-B) in the receiving market falls by the same amount, and the shares change. Please show that version too: the pool and the active balances netted together, and the truncation share under it.
- Switches between deposit types inside a bank and market stay as they are. In the model they are choices of a product.
- Which version the paper adopts is Pedro's decision after Step 1.

**3. The D-market fix (your 20:32). Positions, not decisions:**
- **One production rerun, after Step 2,** as you suggest. Not two.
- **Flag 1, the anchor:** moot under the new construction. No change at l.425 now.
- **Flag 2, the NATIONAL row in the national sleepy share:** fix it in the same rerun, with a diff outline and the before and after counts first. The paper's paragraph E7 describes a population-weighted average over markets, so the fix brings the code to the text. Step 1 on the B-market version, as you do.
- **Flag 3, the copy in the switch-cost prep:** apply the same fix when that branch is next touched. It is parked.
- **For the paper:** noted that E4's price coefficient loses its star at 10 percent and that the D dummy moves by 0.54. Table 5 changes only with the rerun.

**4. Pedro's decisions of this evening are in A1** (items 13 to 15, and the entry below this one). Please write in this file from now on.

### 11-12 20:41 MAIN: currency counts, the fee keys have their go, and this file is the channel

**Pedro: "count it in. I agree w the rest of the decisions. Send the new IMPR to the agent."**

1. **Currency (A1.14).** Series 27789 is part of the outside assets. A withdrawal into cash is a choice of the outside good, and its carry has a zero return. Step 1's table shows its size and its path around 2020.
2. **Fee keys (A1.15).** Write the descriptive ones. The paper's draft sentence uses three: `demand.fee.pkg1.big5.lo.2024`, `demand.fee.pkg1.big5.hi.2024`, and `demand.fee.pkg1.chg.rate`. Post the diff outline first, as always.
3. **He confirms the two decisions of the 20:32 entry:** the pool that counts the money that left the truncated cells, and the fees variant once on the Step 2 frame.
4. **This file is the channel.** Your 20:27 section went into your old file. Please write here from now on, starting with one entry that confirms you have read Part A and that A4, A6, and A7 carry your text correctly.

### 11-12 20:32 MAIN: Pedro agrees. The truncation pool is decided, and the fees variant has its go

**Pedro: "agree. Send a reply, and update notes. Give it a go."**

1. **The pool (A1.13, A3).** The awake pool counts the money that left the truncated cells. This is now his decision, not a proposal.
   - Step 1's table uses it as the main definition and keeps the first one for comparison.
   - Step 2 will re-share under it. Nothing in production changes before his go on Step 2.
2. **Fees (A1.10).** He accepts the positions of the 20:27 entry.
   - **Go:** the variant runs once, on the frame that Step 2 produces, in the sandbox, with the driver and the gates of your plan.
   - **As specified there, with three additions:** the coefficient also at a cap of R$ 30 and with the conglomerates above R$ 50 dropped, the issuer-first version for D reported beside the census-first one, and the conglomerate fixed-effects diagnostic.
   - **Not a go:** a run on today's frame, and writing keys.
3. **Paper.** The drafts for Section 4 (the deposit dynamics with new money) and for equation (16) are in the edits file, on hold. They go in when the numbers do. If your code comments or tables name the pool, the paper's symbol is $P_{mt}$, with $C_{mt}$ for the carried wealth, $C^{\varnothing}_{mt}$ for its part outside deposits, and $N_{mt}$ for new money.
4. **Order unchanged:** the D-market fix, (a), Step 1 with the inventory first.

### 11-12 20:27 MAIN: the fees plan is read (`scratchpad\fees_1112\fees_plan_1112.md`)

**Read in full.** The tables answer Pedro's question, and the answer is mostly no: fees do not give the cross-bank price variation that checking and savings spreads lack.
- Among the five largest the listed Package I price is nearly common (R$ 14.60 to 16.10 in 2024, and 5.3 percent of its variance left net of quarter effects).
- The variation is in the tail: after the cap at R$ 50, B banks outside the five largest supply 87.2 percent of it, and some of those listings look like notional ceilings.
- As a rate the fee is 95.8 percent denominator. Realized fee revenue per client does not track the listed price (−0.169 pooled). No instrument is both strong and credible.

**Main thread's positions on the eight decisions, for Pedro:**
1. **Run the variant:** yes, once, on the frame that Step 2 produces. Not now. The queue is full and the shares are about to change.
2. **The fee object:** the listed Package I in reais, as you recommend.
3. **Rows:** checking only.
4. **Exogenous, with the conglomerate fixed-effects diagnostic,** and reported as a conditional association.
5. **The tail:** the cap at R$ 50, with the coefficient also at a cap of R$ 30 and with the conglomerates above R$ 50 dropped, since the tail carries the estimate.
6. **D source order:** census first, and the issuer-first version reported beside it, not as an option. Inter holds 45 percent of D checking deposits and lists R$ 20 while advertising a free account.
7. **Centered within the quarter.** The text must then say that the coefficient is cross-sectional.
8. **The missing dummy,** so that the sample equals the baseline's.

**One request, no run:** hold the keys of the descriptive tables (coverage, the range at the five largest, the change rate) ready. Pedro may want one sentence in the paper's footnote on fees. Writing them needs his go.

### 11-12 20:26 MAIN: this file

Pedro asked for one document. Part A holds everything that was live in the two files. From now on both sessions write here.
- **IMPR:** please confirm with one log entry, check that A4, A6, and A7 carry your text correctly, and take over their upkeep. Your own file needs no further edits.
- **Nothing in the work order changes.** The download go, the truncation proposal, and Step 1 are in A1 and A3.

### 11-12, earlier (both sessions, one line each. Full text in the two archived files)

- 20:27 IMPR (in its old file). Fees: the tables and the plan are done. Read. It matches the plan file the 20:27 MAIN entry answers.
- 20:27 MAIN. Rewritten IMPR reply read. Proposal on the truncation (A3). Stamped 20:45 in the old file.
- 20:22 MAIN. Pedro's go on the download of the sixteen series, plan first. Stamped 20:40.
- 20:21 IMPR. Reply rewritten at Pedro's request. The count for (b) closes it.
- 20:15 MAIN. The sixteen series and what IF.Data gives. Stamped 20:35.
- 20:13 IMPR. Step 1 started, inventory first.
- 20:10 MAIN. Pedro's decision on the market size, and Step 1. Inventory of `DATA_RAW` (stamped 20:20).
- 20:05 MAIN. Pedro's go on (a) and the count for (b). (c) dropped.
- 19:55 MAIN. Market-size note read. Six positions.
- 19:50 IMPR. The D-market double count is fixed first. Code edited at 19:55.
- 19:45 IMPR. Market-size alternatives compared. No anchor fixes it.
- 19:35 IMPR. Fees started. $\gamma_C$ rename acknowledged.
- 19:28, 19:31 MAIN. The rename to $\gamma_C$. Fees request and Pedro's go on its tables and plan.
- 19:20 IMPR. BLP validation: cold and warm starts reach the production optimum.
- 19:12 MAIN. Positions on the market size, the BLP sentence, and the cap.
- 17:13 MAIN. Positions on W1.1, W2, W3, W4.1, and two read-only requests.
- 14:55 to 16:40 IMPR. Cap screen, K1 pilots, D0, W1.1, W3, W4.1, the W2 plan, and the keys.
