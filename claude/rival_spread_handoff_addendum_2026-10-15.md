# Addendum to the rival-spread handoff: SAME-MARKET rivals (2026-10-15 20:05)

From the main thread, Pedro's decision at 20:0x. **It replaces the construction in `rival_spread_handoff_2026-10-15.md`.** The column names stay as they are (`rival_mean_spread_a{k}`, `_a4_national`). Only the set of rivals changes.

## Why

The sleepiness first stage absorbs quarter fixed effects (and entity FE). The national version is

  R_jt = (S_t − S_jt)/(N_t − N_jt) = s̄_t − w_jt · (s̄_jt − s̄_t),  w_jt = N_jt/(N_t − N_jt).

Once the quarter effect is removed, all that is left is minus the conglomerate's OWN mean spread, scaled. Measured on type 4, 2016–2024 (read-only, on the parquet your rebuild wrote at ~19:5x):
- the identity error is 1.7e-18;
- w_jt has median 0.22 and maximum 0.40;
- after entity and quarter FE, corr(R, own spread) = −0.78, and the residual instrument correlates +1.0000 with the own-spread term.

The instrument would be the endogenous regressor itself. Script: `C:\Users\pedro\AppData\Local\Temp\claude\c--Users-pedro-OneDrive-Documentos-Yale-Year-3--2024---2025--Open-Finance-Open-Finance-Code-Egan-et-al-2025-Rep\21a30b32-2059-49bd-aaac-15d47097282e\scratchpad\rival_fe_check.py`.

## The construction

For a row of conglomerate j in market m (`mca_code`), quarter t, deposit type k:

  `rival_mean_spread_a{k}` = mean of `spread_a{k}` over the rows (j′, m, t) with j′ ≠ j, non-null spread, **same `mca_code`**, same quarter.

- Each rival row has weight 1. There is one row per conglomerate per MCA, so rivals are equally weighted.
- **NaN when j has no rival in m.** On the 08-26 type-4 data, 88% of rows have at least one rival (median 3). Report the NaN share per type: it is a sample change for the sleepiness IV, which drops rows with a missing instrument.
- D firms: their rows sit under the single national code. The formula, applied literally, makes their rivals the other D firms' national rows. That is intended; do not special-case it.
- Applied to all four types (a1, a2, a4, a5).

Same-market version measured on type 4: after both FE, corr with own spread −0.24, residual SD 2.7e-3.

## `panel_local_cdb_spread.py`

A local deviation d of (j, m, t) now moves only the instruments of j's rivals **in market m**: for each k ≠ j in m, R_kmt += d / n_k, where n_k is the number of k's rivals in m. j's own instrument is unchanged. Re-derive the delta for this rule, check it against a from-scratch recompute as you did before, and keep the `_national` copy consistent.

## Report (as in the handoff)

- the market_panel md5 and the demand_chars sha256;
- per type: corr(new, the 08-26 own-row version), the NaN share, the within-quarter SD;
- **also**: on type 4, 2016–2024, corr(instrument, own spread) after entity + quarter FE (expected near −0.24);
- surprises.

## Also decided (downstream, the main thread does it)

The ~599 B rows with no ESTBAN instrument (D6) are DROPPED in the demand preps. Nothing changes on the panel side.
