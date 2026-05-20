"""
d_rate_scrape_6_diagnose.py
==============================
Diagnostic for merging scraped advertised rates into panel_3_rates.py.

Scope
-----
ONLY deposit types 4 (time / CDB) and 5 (prepaid).  Type 1 (demand) is
hardcoded to 0.0 in panel_3 because traditional banks don't pay interest on
demand accounts and digital banks don't offer demand accounts (they offer
prepaid, type 5).  Type 3 (interbank) is a true CDI relationship, not a
fallback.

Decision rule
-------------
For each (CodConglPrud, AnoMes, deposit_type) cell currently in panel_3's
output, we mark it as a "CDI fallback" iff every preferred source is NaN:

  Type 4:     cosif_type4_rate IS NaN  AND  cosif_implicit_rate IS NaN
  Type 5 (has_ip==1):
              ip_prepaid_rate IS NaN  AND  median_ip_rate IS NaN
  Type 5 (has_ip==0):
              median_ip_rate IS NaN

If a cell is a CDI fallback AND we have a Tier-A or Tier-B scraped rate for
that conglomerate at that AnoMes, we recommend overriding with
  advertised_rate_qoq = (advertised_pct / 100) * cdi_qoq
COSIF / IP rates are always preferred when present (regulatory > marketing).

Tiering of scraped series (per CodConglPrud x rate_index)
---------------------------------------------------------
  Tier A:  n_obs_quarters >= 4  AND  std_pct <= 5  AND  n_pct_changes <= 1
           -> trust forward-filled values across all quarters in span
  Tier B:  n_obs_quarters >= 2 and not A
           -> trust only OBSERVED quarters (is_filled == 0)
  Tier C:  n_obs_quarters < 2  OR  std_pct > 15
           -> do not merge, log only

Outputs (in processed/IP_SCRAPE/diagnose/)
-----------------------------------------
  diagnose_per_cod.csv          stability table per (cod, rate_index) + tier
  diagnose_per_quarter.csv      coverage per (AnoMes, deposit_type)
  diagnose_gap_vs_fallback.csv  row-level: panel rate vs advertised rate
  recommended_merge.csv         (cod, AnoMes, deposit_type, use_scraped,
                                 advertised_rate_qoq, reason)

The recommended_merge.csv is what panel_3_rates.py should consume.
"""
from utils.venv_guard import ensure_project_venv
ensure_project_venv(__file__)

import argparse
import logging
import numpy as np
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

root = Path(__file__).resolve().parents[2]
processed = root / 'BCB' / 'Egan_et_al_2025_Rep' / 'processed'
ip_scrape_dir = processed / 'IP_SCRAPE'
panel_dir     = processed / 'PANEL_INTERMED'
out_dir       = ip_scrape_dir / 'diagnose'

# Tier thresholds (tweak via CLI)
DEFAULT_TIER_A_NOBS   = 4
DEFAULT_TIER_A_STD    = 5.0      # pp of advertised_pct
DEFAULT_TIER_A_CHANGES = 1
DEFAULT_TIER_B_NOBS   = 2
DEFAULT_TIER_C_STD    = 15.0

# Safety bound on advertised_rate_qoq relative to cdi_qoq (drop weird ratios)
DEFAULT_REL_LOW  = 0.5
DEFAULT_REL_HIGH = 2.0


# ----------------------------- loading ---------------------------------------

def load_advertised_long(in_path):
    df = pd.read_csv(in_path)
    needed = {'CodConglPrud', 'AnoMes', 'rate_index',
              'advertised_pct', 'n_obs', 'is_filled'}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"{in_path} missing columns: {missing}")
    df['AnoMes'] = df['AnoMes'].astype(int)
    return df


def load_panel(panel_path):
    df = pd.read_csv(panel_path, low_memory=False)
    needed = {'CodConglPrud', 'AnoMes', 'deposit_type', 'has_ip',
              'cdi_qoq', 'deposit_rate_qoq',
              'cosif_type4_rate', 'cosif_implicit_rate',
              'ip_prepaid_rate', 'median_ip_rate'}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"{panel_path} missing columns: {missing}")
    df['AnoMes'] = df['AnoMes'].astype(int)
    return df


# ------------------------- per-conglomerate tiering --------------------------

def build_per_cod_tiers(adv_long,
                        a_nobs, a_std, a_changes,
                        b_nobs, c_std):
    """One row per (CodConglPrud, rate_index) with tier classification."""
    obs = adv_long[adv_long['is_filled'] == 0].copy()
    if obs.empty:
        return pd.DataFrame()

    def _summarize(g):
        g = g.sort_values('AnoMes')
        diffs = g['advertised_pct'].diff().abs()
        return pd.Series({
            'n_obs_q'      : len(g),
            'first_q'      : int(g['AnoMes'].min()),
            'last_q'       : int(g['AnoMes'].max()),
            'median_pct'   : float(g['advertised_pct'].median()),
            'mean_pct'     : float(g['advertised_pct'].mean()),
            'std_pct'      : float(g['advertised_pct'].std(ddof=0))
                              if len(g) > 1 else 0.0,
            'iqr_pct'      : float(g['advertised_pct'].quantile(0.75)
                                   - g['advertised_pct'].quantile(0.25))
                              if len(g) > 1 else 0.0,
            'n_pct_changes': int((diffs > 5).sum()),  # >5pp shifts
        })

    summ = (obs.groupby(['CodConglPrud', 'rate_index'])
               .apply(_summarize, include_groups=False)
               .reset_index())

    def _tier(row):
        if (row['n_obs_q'] >= a_nobs
                and row['std_pct'] <= a_std
                and row['n_pct_changes'] <= a_changes):
            return 'A'
        if row['n_obs_q'] < b_nobs or row['std_pct'] > c_std:
            return 'C'
        return 'B'
    summ['tier'] = summ.apply(_tier, axis=1)
    return summ


# ---------------------- per-cell merge candidate logic -----------------------

def mark_cdi_fallback(panel):
    """
    Tag each panel row with is_cdi_fallback for deposit types 4 & 5.
    Type 4:  fallback iff cosif_type4_rate.isna() AND cosif_implicit_rate.isna()
    Type 5 has_ip==1:  iff ip_prepaid_rate.isna() AND median_ip_rate.isna()
    Type 5 has_ip==0:  iff median_ip_rate.isna()
    """
    panel = panel.copy()
    is_t4 = panel['deposit_type'] == 4
    is_t5 = panel['deposit_type'] == 5
    fb_t4 = is_t4 & panel['cosif_type4_rate'].isna() & panel['cosif_implicit_rate'].isna()
    fb_t5_ip  = is_t5 & (panel['has_ip'] == 1) & panel['ip_prepaid_rate'].isna() & panel['median_ip_rate'].isna()
    fb_t5_nip = is_t5 & (panel['has_ip'] == 0) & panel['median_ip_rate'].isna()
    panel['is_cdi_fallback'] = fb_t4 | fb_t5_ip | fb_t5_nip
    panel['fallback_reason'] = np.select(
        [fb_t4, fb_t5_ip, fb_t5_nip],
        ['T4_no_cosif', 'T5_ip_no_data', 'T5_no_ip_no_median'],
        default='',
    )
    return panel


def pick_advertised_per_cell(adv_long, per_cod, panel_cells):
    """
    For each (cod, AnoMes) in panel_cells, pick the best scraped row:
      - Prefer CDI over SELIC (advertising convention in Brazil).
      - Tier A: use any row (observed or filled).
      - Tier B: use only is_filled == 0 rows.
      - Tier C: skip.
    Returns DataFrame with columns:
      CodConglPrud, AnoMes, rate_index, advertised_pct, n_obs, is_filled, tier
    """
    if adv_long.empty or per_cod.empty:
        return pd.DataFrame()

    adv = adv_long.merge(per_cod[['CodConglPrud', 'rate_index', 'tier']],
                         on=['CodConglPrud', 'rate_index'], how='left')
    # Tier B: drop filled rows
    adv = adv[~((adv['tier'] == 'B') & (adv['is_filled'] == 1))]
    # Tier C: drop entirely
    adv = adv[adv['tier'].isin(['A', 'B'])]

    # Restrict to panel cells of interest
    keys = panel_cells[['CodConglPrud', 'AnoMes']].drop_duplicates()
    adv = adv.merge(keys, on=['CodConglPrud', 'AnoMes'], how='inner')

    # Prefer CDI when both are present for same (cod, AnoMes)
    adv['rate_priority'] = adv['rate_index'].map({'CDI': 0, 'SELIC': 1}).fillna(9)
    adv = (adv.sort_values(['CodConglPrud', 'AnoMes', 'rate_priority'])
              .drop_duplicates(['CodConglPrud', 'AnoMes'], keep='first')
              .drop(columns=['rate_priority']))
    return adv


# -------------------------------- main ---------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--advertised-long',
                    default=str(ip_scrape_dir / 'advertised_rates_quarterly.csv'))
    ap.add_argument('--panel',
                    default=str(panel_dir / 'egan_panel_deposits.csv'))
    ap.add_argument('--out-dir', default=str(out_dir))
    ap.add_argument('--tier-a-nobs',    type=int,   default=DEFAULT_TIER_A_NOBS)
    ap.add_argument('--tier-a-std',     type=float, default=DEFAULT_TIER_A_STD)
    ap.add_argument('--tier-a-changes', type=int,   default=DEFAULT_TIER_A_CHANGES)
    ap.add_argument('--tier-b-nobs',    type=int,   default=DEFAULT_TIER_B_NOBS)
    ap.add_argument('--tier-c-std',     type=float, default=DEFAULT_TIER_C_STD)
    ap.add_argument('--rel-low',  type=float, default=DEFAULT_REL_LOW,
                    help='Reject advertised_rate_qoq < rel-low * cdi_qoq')
    ap.add_argument('--rel-high', type=float, default=DEFAULT_REL_HIGH,
                    help='Reject advertised_rate_qoq > rel-high * cdi_qoq')
    ap.add_argument('--ignore-preferred', action='store_true',
                    help='Override panel cells with scraped rates regardless of '
                         'whether COSIF/IP data is present. Use when you trust '
                         'advertised yields over regulatory implied rates.')
    ap.add_argument('--gap-threshold', type=float, default=None,
                    help='If set, only override (under --ignore-preferred) when '
                         '|gap_qoq| > threshold (pp). Lets you keep panel data '
                         'when the two sources broadly agree.')
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    # ---- Load ----
    logging.info(f"Loading advertised: {args.advertised_long}")
    adv_long = load_advertised_long(Path(args.advertised_long))
    logging.info(f"  rows: {len(adv_long):,}, conglomerates: "
                 f"{adv_long['CodConglPrud'].nunique()}")

    logging.info(f"Loading panel:      {args.panel}")
    panel = load_panel(Path(args.panel))
    logging.info(f"  rows: {len(panel):,}, conglomerates: "
                 f"{panel['CodConglPrud'].nunique()}")

    # ---- 1) Tier conglomerates ----
    per_cod = build_per_cod_tiers(
        adv_long,
        args.tier_a_nobs, args.tier_a_std, args.tier_a_changes,
        args.tier_b_nobs, args.tier_c_std,
    )
    per_cod_path = out / 'diagnose_per_cod.csv'
    per_cod.to_csv(per_cod_path, index=False)
    logging.info(f"Wrote {per_cod_path} ({len(per_cod)} cod x rate_index rows)")
    tier_counts = per_cod['tier'].value_counts().to_dict() if not per_cod.empty else {}
    logging.info(f"  tier distribution: {tier_counts}")

    # ---- 2) Mark CDI-fallback cells in panel (types 4 & 5 only) ----
    panel = mark_cdi_fallback(panel)
    cells = panel[panel['deposit_type'].isin([4, 5])].copy()
    n_fb_t4 = ((cells['deposit_type'] == 4) & cells['is_cdi_fallback']).sum()
    n_fb_t5 = ((cells['deposit_type'] == 5) & cells['is_cdi_fallback']).sum()
    n_tot_t4 = (cells['deposit_type'] == 4).sum()
    n_tot_t5 = (cells['deposit_type'] == 5).sum()
    logging.info(
        f"Panel CDI-fallback cells: "
        f"T4 = {n_fb_t4:,} / {n_tot_t4:,}  ({100*n_fb_t4/max(n_tot_t4,1):.1f}%),  "
        f"T5 = {n_fb_t5:,} / {n_tot_t5:,}  ({100*n_fb_t5/max(n_tot_t5,1):.1f}%)"
    )

    # ---- 3) Pick best advertised row per (cod, AnoMes) ----
    picked = pick_advertised_per_cell(adv_long, per_cod, cells)
    logging.info(f"Candidate scraped rows after tier filter: {len(picked):,}")

    # ---- 4) Build gap-vs-fallback table (every T4/T5 cell with scrape) ----
    _gap_cols = ['CodConglPrud', 'AnoMes', 'rate_index',
                 'advertised_pct', 'n_obs', 'is_filled', 'tier']
    if picked.empty:
        # No scraped data at all (upstream chain produced 0 rows). Emit empty
        # outputs rather than crashing on the merge.
        logging.warning(
            "No advertised rows survived tier filtering; emitting empty "
            "gap / coverage / recommendation tables."
        )
        picked = pd.DataFrame(columns=_gap_cols)
    gap = cells.merge(picked[_gap_cols],
                      on=['CodConglPrud', 'AnoMes'], how='inner')
    gap['advertised_rate_qoq'] = (gap['advertised_pct'] / 100.0) * gap['cdi_qoq']
    gap['gap_qoq'] = gap['advertised_rate_qoq'] - gap['deposit_rate_qoq']
    gap['rel_to_cdi'] = np.where(
        gap['cdi_qoq'].abs() > 1e-9,
        gap['advertised_rate_qoq'] / gap['cdi_qoq'],
        np.nan,
    )
    gap_cols = ['CodConglPrud', 'AnoMes', 'quarter', 'deposit_type', 'has_ip',
                'is_cdi_fallback', 'fallback_reason',
                'deposit_rate_qoq', 'cdi_qoq',
                'cosif_type4_rate', 'cosif_implicit_rate',
                'ip_prepaid_rate', 'median_ip_rate',
                'rate_index', 'advertised_pct', 'n_obs', 'is_filled', 'tier',
                'advertised_rate_qoq', 'gap_qoq', 'rel_to_cdi']
    gap_cols = [c for c in gap_cols if c in gap.columns]
    gap_path = out / 'diagnose_gap_vs_fallback.csv'
    gap[gap_cols].to_csv(gap_path, index=False)
    logging.info(f"Wrote {gap_path} ({len(gap):,} cells)")

    # ---- 5) Per-quarter coverage (T4 & T5 separately) ----
    cov_rows = []
    for dtype in (4, 5):
        sub = cells[cells['deposit_type'] == dtype]
        sub_fb = sub[sub['is_cdi_fallback']]
        cov_q = (sub.groupby('AnoMes')
                    .agg(n_panel=('CodConglPrud', 'nunique'))
                    .reset_index())
        cov_qfb = (sub_fb.groupby('AnoMes')
                       .agg(n_fallback=('CodConglPrud', 'nunique'))
                       .reset_index())
        picked_sub = gap[gap['deposit_type'] == dtype]
        cov_qsc = (picked_sub.groupby('AnoMes')
                       .agg(n_with_scrape=('CodConglPrud', 'nunique'))
                       .reset_index())
        picked_fb = picked_sub[picked_sub['is_cdi_fallback']]
        cov_qov = (picked_fb.groupby('AnoMes')
                       .agg(n_overrides=('CodConglPrud', 'nunique'),
                            mean_gap_qoq=('gap_qoq', 'mean'),
                            median_gap_qoq=('gap_qoq', 'median'))
                       .reset_index())
        merged = (cov_q.merge(cov_qfb, on='AnoMes', how='left')
                       .merge(cov_qsc, on='AnoMes', how='left')
                       .merge(cov_qov, on='AnoMes', how='left'))
        merged['deposit_type'] = dtype
        cov_rows.append(merged)
    coverage = (pd.concat(cov_rows, ignore_index=True)
                if cov_rows else pd.DataFrame())
    if not coverage.empty:
        for c in ('n_fallback', 'n_with_scrape', 'n_overrides'):
            coverage[c] = coverage[c].fillna(0).astype(int)
        coverage = coverage[['AnoMes', 'deposit_type', 'n_panel',
                             'n_fallback', 'n_with_scrape', 'n_overrides',
                             'mean_gap_qoq', 'median_gap_qoq']]
    cov_path = out / 'diagnose_per_quarter.csv'
    coverage.to_csv(cov_path, index=False)
    logging.info(f"Wrote {cov_path} ({len(coverage)} (AnoMes, deposit_type) rows)")

    # ---- 6) Recommended merge table ----
    rec = gap.copy()
    rec['in_rel_band'] = ((rec['rel_to_cdi'] >= args.rel_low)
                          & (rec['rel_to_cdi'] <= args.rel_high))
    if args.ignore_preferred:
        gap_ok = (rec['gap_qoq'].abs() > args.gap_threshold
                  if args.gap_threshold is not None
                  else pd.Series(True, index=rec.index))
        rec['use_scraped'] = (rec['in_rel_band']
                              & rec['tier'].isin(['A', 'B'])
                              & gap_ok)
    else:
        rec['use_scraped'] = (rec['is_cdi_fallback']
                              & rec['in_rel_band']
                              & rec['tier'].isin(['A', 'B']))
    def _reason(r):
        if r['tier'] == 'C':
            return 'skip_tier_C'
        if not r['in_rel_band']:
            return 'skip_out_of_rel_band'
        if args.ignore_preferred:
            if args.gap_threshold is not None and abs(r['gap_qoq']) <= args.gap_threshold:
                return 'skip_gap_below_threshold'
            tag = r['fallback_reason'] if r['is_cdi_fallback'] else 'force_override'
            return f"override_tier_{r['tier']}_{tag}"
        if not r['is_cdi_fallback']:
            return 'skip_panel_has_preferred_source'
        return f"override_tier_{r['tier']}_{r['fallback_reason']}"
    rec['reason'] = rec.apply(_reason, axis=1)
    rec_out = rec[['CodConglPrud', 'AnoMes', 'deposit_type', 'has_ip',
                   'use_scraped', 'advertised_rate_qoq', 'advertised_pct',
                   'rate_index', 'tier', 'is_filled',
                   'is_cdi_fallback', 'fallback_reason', 'in_rel_band',
                   'gap_qoq', 'reason']].sort_values(
        ['deposit_type', 'CodConglPrud', 'AnoMes']
    )
    rec_path = out / 'recommended_merge.csv'
    rec_out.to_csv(rec_path, index=False)
    logging.info(f"Wrote {rec_path} ({len(rec_out):,} candidate cells)")

    # ---- 7) Summary print ----
    print()
    print("=" * 72)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 72)
    print(f"Tier distribution (cod x rate_index): {tier_counts}")
    tier_a_cods = per_cod.loc[per_cod['tier'] == 'A', 'CodConglPrud'].nunique() if not per_cod.empty else 0
    tier_b_cods = per_cod.loc[per_cod['tier'] == 'B', 'CodConglPrud'].nunique() if not per_cod.empty else 0
    tier_c_cods = per_cod.loc[per_cod['tier'] == 'C', 'CodConglPrud'].nunique() if not per_cod.empty else 0
    print(f"  Tier A conglomerates: {tier_a_cods}   "
          f"Tier B: {tier_b_cods}   Tier C: {tier_c_cods}")
    print()
    for dtype in (4, 5):
        sub = rec_out[rec_out['deposit_type'] == dtype]
        n_override = sub['use_scraped'].sum()
        n_fallback_t = ((cells['deposit_type'] == dtype) & cells['is_cdi_fallback']).sum()
        share = 100 * n_override / max(n_fallback_t, 1)
        print(f"Deposit type {dtype}:")
        print(f"  Currently CDI-fallback cells: {n_fallback_t:,}")
        print(f"  Recommended overrides:         {n_override:,} ({share:.1f}%)")
        if n_override:
            sub_use = sub[sub['use_scraped']]
            print(f"    median advertised_rate_qoq: {sub_use['advertised_rate_qoq'].median():.5f}")
            print(f"    median gap vs current CDI:  {sub_use['gap_qoq'].median():+.5f}")
            print(f"    distinct conglomerates:     {sub_use['CodConglPrud'].nunique()}")
            print(f"    quarter range:              "
                  f"{int(sub_use['AnoMes'].min())} -> {int(sub_use['AnoMes'].max())}")
        skipped = sub[~sub['use_scraped']]['reason'].value_counts().to_dict()
        if skipped:
            print(f"  Skipped reasons:                {skipped}")
        print()


if __name__ == '__main__':
    main()
