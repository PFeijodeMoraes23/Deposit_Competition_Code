"""
scrape_deposit_rate_format.py
==============================
Formats the raw rate mentions extracted by scrape_deposit_rate_parse.py into a
clean conglomerate x quarter panel that merges directly with the main
deposit panel (panel_deposit_rates.py).

Inputs
------
  <IP_SCRAPE>/extracted_historical_rates.csv
      Columns: CodConglomerado, Snapshot_Date, Rate_Type, Rate_Value,
               Context, Source

Outputs
-------
  <IP_SCRAPE>/advertised_rates_quarterly.csv          (long form)
  <IP_SCRAPE>/advertised_rates_quarterly_wide.csv     (one row per cod x AnoMes)
  <IP_SCRAPE>/advertised_rates_dropped.csv            (filtered-out rows, audit)

Periodization
-------------
The deposit panel (panel_deposit_rates.py) keys observations by AnoMes (yyyyMM)
with month in {3, 6, 9, 12} -- i.e. quarter-end.  Each archival snapshot is
assigned to the AnoMes of the quarter it falls in (e.g. 2020-04-15 ->
202006 = 2020Q2).  Multiple snapshots within the same (cod, quarter, type)
are reduced to the median.  Quarters with no observation between the first
and last seen are forward-filled (banks change advertised rates infrequently).

Output schema (long form, matches panel_3 keys)
-----------------------------------------------
  CodConglPrud        - C-prefixed prudential conglomerate id (== CodInst for C-codes)
  AnoMes              - int yyyymm, month in {3,6,9,12}
  quarter             - "yyyyQn" string
  rate_index          - "CDI" or "SELIC"
  advertised_pct      - advertised rate as percent of the benchmark
                        (e.g. 100.0 = "100% do CDI"; 102.0 = "102% do CDI")
  n_obs               - number of raw mentions aggregated
  is_filled           - 1 if forward-filled, 0 if observed in that quarter

To convert into a quarter-over-quarter compounded yield aligned with
panel_3's deposit_rate_qoq, merge on (CodConglPrud, AnoMes) then compute
  advertised_rate_qoq = (advertised_pct / 100) * benchmark_qoq
where benchmark_qoq is cdi_qoq or selic_qoq from make_quarterly_macro().
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
ip_scrape_dir = root / 'BCB' / 'Egan_et_al_2025_Rep' / 'processed' / 'IP_SCRAPE'
_panel_intermed = root / 'BCB' / 'Egan_et_al_2025_Rep' / 'processed' / 'PANEL_INTERMED'

# Filtering thresholds. Typical digital-bank advertised rates are 80-130% of
# CDI/Selic. Anything outside [MIN, MAX] is almost surely either a loan rate,
# a fund-management performance fee ("20% sobre o que exceder 100% do CDI"),
# or a regex false positive (date / address number / etc.).
MIN_PCT = 50.0
MAX_PCT = 200.0

# Absolute-rate rescue: some PDFs quote CDI as a raw annual rate (e.g.
# "CDI 11.57% a.a.") instead of as a percentage of CDI. Values in
# [ABS_RATE_MIN, ABS_RATE_MAX] are treated as absolute annual rates and
# converted to % of CDI by dividing by the CDI annual rate for that quarter.
ABS_RATE_MIN = 3.0
ABS_RATE_MAX = 25.0

# Context-substring blocklist: keywords that signal the matched number is not
# a deposit-yield phrase. Case-insensitive substring match against `Context`.
BAD_CONTEXT_KEYWORDS = (
    'emprestimo', 'empréstimo',
    'financiamento',
    'rotativo',
    'cheque especial',
    'juros do cartao', 'juros do cartão',
    'taxa de performance', 'performance fee',
    'sobre o que exceder',
    'consignado',
    'antecipacao', 'antecipação',
    'parcelamento',
    'cdc ',  # crédito direto ao consumidor
)


def _normalize(s):
    return (
        s.lower()
         .replace('á', 'a').replace('à', 'a').replace('â', 'a').replace('ã', 'a')
         .replace('é', 'e').replace('ê', 'e')
         .replace('í', 'i')
         .replace('ó', 'o').replace('ô', 'o').replace('õ', 'o')
         .replace('ú', 'u').replace('ç', 'c')
    )


def _is_bad_context(ctx):
    if not isinstance(ctx, str):
        return False
    low = _normalize(ctx)
    return any(kw in low for kw in BAD_CONTEXT_KEYWORDS)


def _date_to_anomes(date_str):
    """Map a YYYY-MM-DD snapshot date to its containing quarter-end AnoMes."""
    try:
        dt = pd.to_datetime(date_str, errors='coerce')
    except Exception:
        return np.nan
    if pd.isna(dt):
        return np.nan
    q_month = ((dt.month - 1) // 3 + 1) * 3   # 3, 6, 9, 12
    return int(dt.year * 100 + q_month)


def _anomes_to_quarter(anomes):
    if pd.isna(anomes):
        return None
    a = int(anomes)
    return f"{a // 100}Q{(a % 100) // 3}"


def _load_macro_rates(path):
    """Load quarterly_macro_rates.csv and return a DataFrame indexed by AnoMes
    with columns cdi_annual and selic_annual (annualised from quarterly rates).
    Returns None if the file is missing."""
    p = Path(path)
    if not p.exists():
        logging.warning(f"Macro rates not found: {p} -- abs-rate conversion disabled.")
        return None
    df = pd.read_csv(p)
    df['cdi_annual']   = ((1 + df['cdi_qoq']   / 100) ** 4 - 1) * 100
    df['selic_annual'] = ((1 + df['selic_qoq'] / 100) ** 4 - 1) * 100
    df['AnoMes'] = df['AnoMes'].astype(int)
    return df[['AnoMes', 'cdi_annual', 'selic_annual']].set_index('AnoMes')


def rescue_abs_rates(dropped, macro_rates):
    """Rescue rows dropped as out_of_range that are absolute annual rate quotes.

    Rows where advertised_pct is in [ABS_RATE_MIN, ABS_RATE_MAX] are converted
    to % of CDI using the period's annualised CDI rate:
        pct_of_cdi = abs_rate_annual / cdi_annual * 100
    If the result is in [MIN_PCT, MAX_PCT] the row is promoted to kept.

    Returns (rescued_df, updated_dropped_df).
    """
    if macro_rates is None or dropped.empty:
        return pd.DataFrame(), dropped

    mask = (
        (dropped['drop_reason'] == 'out_of_range') &
        (dropped['advertised_pct'] >= ABS_RATE_MIN) &
        (dropped['advertised_pct'] <= ABS_RATE_MAX)
    )
    candidates = dropped[mask].copy()
    still_dropped = dropped[~mask].copy()

    if candidates.empty:
        return pd.DataFrame(), dropped

    # Join CDI annual rate by quarter
    candidates['_anomes_int'] = candidates['AnoMes'].astype(int)
    candidates = candidates.join(
        macro_rates[['cdi_annual']], on='_anomes_int', how='left'
    ).drop(columns=['_anomes_int'])

    has_cdi = candidates['cdi_annual'].notna()

    # Convert: abs annual % -> % of CDI
    candidates.loc[has_cdi, 'advertised_pct'] = (
        candidates.loc[has_cdi, 'advertised_pct']
        / candidates.loc[has_cdi, 'cdi_annual'] * 100
    )

    in_range = (
        has_cdi &
        (candidates['advertised_pct'] >= MIN_PCT) &
        (candidates['advertised_pct'] <= MAX_PCT)
    )
    rescued = candidates[in_range].drop(columns=['drop_reason', 'cdi_annual'])

    not_rescued = candidates[~in_range].drop(columns=['cdi_annual'])
    not_rescued['drop_reason'] = 'out_of_range'

    updated_dropped = pd.concat([still_dropped, not_rescued], ignore_index=True)

    logging.info(
        f"Abs-rate rescue: {len(candidates):,} candidates in [{ABS_RATE_MIN},{ABS_RATE_MAX}]% range, "
        f"{len(rescued):,} rescued (converted to % of CDI), "
        f"{len(not_rescued):,} still dropped"
    )
    return rescued, updated_dropped


def load_raw(in_path):
    if not in_path.exists():
        raise FileNotFoundError(f"{in_path} not found. Run scrape_deposit_rate_parse.py first.")
    df = pd.read_csv(in_path)
    needed = {'CodConglomerado', 'Snapshot_Date', 'Rate_Type', 'Rate_Value', 'Context', 'Source'}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {in_path}: {missing}")
    return df


def clean_and_filter(df):
    df = df.copy()

    # Parse rate value (e.g. "102%" -> 102.0)
    df['advertised_pct'] = (
        df['Rate_Value'].astype(str).str.rstrip('%').str.strip()
        .replace({'': np.nan}).astype(float)
    )

    # Parse snapshot date -> AnoMes
    df['AnoMes'] = df['Snapshot_Date'].apply(_date_to_anomes)
    df['quarter'] = df['AnoMes'].apply(_anomes_to_quarter)

    # Normalize id column name to match panel_3 schema
    df = df.rename(columns={'CodConglomerado': 'CodConglPrud', 'Rate_Type': 'rate_index'})

    # Tag drop reasons (in priority order)
    df['drop_reason'] = ''
    df.loc[df['AnoMes'].isna(), 'drop_reason'] = 'bad_date'
    df.loc[df['advertised_pct'].isna() & (df['drop_reason'] == ''),
           'drop_reason'] = 'bad_value'
    out_of_range = (df['advertised_pct'] < MIN_PCT) | (df['advertised_pct'] > MAX_PCT)
    df.loc[out_of_range & (df['drop_reason'] == ''),
           'drop_reason'] = 'out_of_range'
    bad_ctx = df['Context'].apply(_is_bad_context)
    df.loc[bad_ctx & (df['drop_reason'] == ''),
           'drop_reason'] = 'bad_context'

    kept = df[df['drop_reason'] == ''].drop(columns=['drop_reason'])
    dropped = df[df['drop_reason'] != '']
    return kept, dropped


def aggregate_quarterly(df_clean):
    """Median advertised_pct per (CodConglPrud, AnoMes, rate_index)."""
    if df_clean.empty:
        return pd.DataFrame(columns=['CodConglPrud', 'AnoMes', 'quarter',
                                     'rate_index', 'advertised_pct', 'n_obs',
                                     'is_filled'])
    df_clean['AnoMes'] = df_clean['AnoMes'].astype(int)
    g = (df_clean
         .groupby(['CodConglPrud', 'AnoMes', 'rate_index'], as_index=False)
         .agg(advertised_pct=('advertised_pct', 'median'),
              n_obs=('advertised_pct', 'size')))
    g['quarter'] = g['AnoMes'].apply(_anomes_to_quarter)
    g['is_filled'] = 0
    return g[['CodConglPrud', 'AnoMes', 'quarter', 'rate_index',
              'advertised_pct', 'n_obs', 'is_filled']]


def forward_fill_quarters(df_long):
    """
    Within each (CodConglPrud, rate_index) series, generate every quarter
    between first_seen and last_seen and forward-fill advertised_pct.
    Banks rarely change advertised rates intra-quarter, so an observation at
    2020Q2 is presumed in effect for 2020Q3, 2020Q4, ... until the next obs.
    """
    if df_long.empty:
        return df_long

    out_parts = []
    for (cod, idx), g in df_long.groupby(['CodConglPrud', 'rate_index'], sort=False):
        g = g.sort_values('AnoMes')
        # Build the full quarter grid between first and last observation
        first, last = int(g['AnoMes'].min()), int(g['AnoMes'].max())
        all_qs = _quarter_range(first, last)
        full = pd.DataFrame({'AnoMes': all_qs})
        full['CodConglPrud'] = cod
        full['rate_index']   = idx
        merged = full.merge(g[['AnoMes', 'advertised_pct', 'n_obs', 'is_filled']],
                            on='AnoMes', how='left')
        merged['is_filled'] = merged['advertised_pct'].isna().astype(int)
        merged['advertised_pct'] = merged['advertised_pct'].ffill()
        merged['n_obs'] = merged['n_obs'].fillna(0).astype(int)
        merged['quarter'] = merged['AnoMes'].apply(_anomes_to_quarter)
        out_parts.append(merged)

    df = pd.concat(out_parts, ignore_index=True)
    return df[['CodConglPrud', 'AnoMes', 'quarter', 'rate_index',
               'advertised_pct', 'n_obs', 'is_filled']]


def _quarter_range(anomes_start, anomes_end):
    """Yield AnoMes values for every quarter in [start, end]."""
    y, m = anomes_start // 100, anomes_start % 100
    out = []
    while True:
        am = y * 100 + m
        out.append(am)
        if am >= anomes_end:
            break
        m += 3
        if m > 12:
            m -= 12
            y += 1
    return out


def pivot_wide(df_long):
    """One row per (CodConglPrud, AnoMes); columns for CDI and SELIC."""
    if df_long.empty:
        return pd.DataFrame(columns=['CodConglPrud', 'AnoMes', 'quarter',
                                     'advertised_cdi_pct', 'advertised_selic_pct',
                                     'n_obs_cdi', 'n_obs_selic'])
    pct = df_long.pivot_table(index=['CodConglPrud', 'AnoMes', 'quarter'],
                              columns='rate_index', values='advertised_pct',
                              aggfunc='first').reset_index()
    nobs = df_long.pivot_table(index=['CodConglPrud', 'AnoMes', 'quarter'],
                               columns='rate_index', values='n_obs',
                               aggfunc='first').reset_index()
    pct = pct.rename(columns={'CDI': 'advertised_cdi_pct',
                              'SELIC': 'advertised_selic_pct'})
    nobs = nobs.rename(columns={'CDI': 'n_obs_cdi', 'SELIC': 'n_obs_selic'})
    out = pct.merge(nobs, on=['CodConglPrud', 'AnoMes', 'quarter'], how='outer')
    for c in ('advertised_cdi_pct', 'advertised_selic_pct'):
        if c not in out.columns:
            out[c] = np.nan
    for c in ('n_obs_cdi', 'n_obs_selic'):
        if c not in out.columns:
            out[c] = 0
        out[c] = out[c].fillna(0).astype(int)
    return out.sort_values(['CodConglPrud', 'AnoMes']).reset_index(drop=True)


def _load_manual_overrides(path):
    """Load operator-provided seed rates and normalize to long-form schema.

    Expected columns (case-insensitive):
        CodConglPrud, AnoMes, rate_index, advertised_pct, [note]
    AnoMes must be a quarter-end yyyymm (month in {3,6,9,12}).
    Rows are tagged is_manual=1 and override any scraped/ffilled value for
    the same (CodConglPrud, AnoMes, rate_index) key.
    """
    if not path or not Path(path).exists():
        return pd.DataFrame()
    df = pd.read_csv(path, comment='#')
    if df.empty:
        return df
    # Normalize column names
    df.columns = [c.strip() for c in df.columns]
    rename = {}
    for c in df.columns:
        cl = c.lower()
        if cl == 'codconglprud': rename[c] = 'CodConglPrud'
        elif cl == 'anomes': rename[c] = 'AnoMes'
        elif cl == 'rate_index': rename[c] = 'rate_index'
        elif cl in ('advertised_pct', 'pct'): rename[c] = 'advertised_pct'
        elif cl == 'note': rename[c] = 'note'
    df = df.rename(columns=rename)
    needed = {'CodConglPrud', 'AnoMes', 'rate_index', 'advertised_pct'}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"manual-overrides {path} missing columns: {missing}")
    df['AnoMes'] = pd.to_numeric(df['AnoMes'], errors='coerce').astype('Int64')
    df = df.dropna(subset=['AnoMes', 'advertised_pct'])
    df['AnoMes'] = df['AnoMes'].astype(int)
    df['rate_index'] = df['rate_index'].astype(str).str.upper().str.strip()
    df['CodConglPrud'] = df['CodConglPrud'].astype(str).str.strip()
    df['advertised_pct'] = df['advertised_pct'].astype(float)
    df['quarter'] = df['AnoMes'].apply(_anomes_to_quarter)
    df['n_obs'] = 1
    df['is_filled'] = 0
    df['is_manual'] = 1
    return df[['CodConglPrud', 'AnoMes', 'quarter', 'rate_index',
               'advertised_pct', 'n_obs', 'is_filled', 'is_manual']]


def _apply_manual_overrides(long_df, manual_df):
    """Overwrite (cod, anomes, index) rows in long_df with manual values;
    insert net-new rows; ensure is_manual column exists."""
    if 'is_manual' not in long_df.columns:
        long_df = long_df.copy()
        long_df['is_manual'] = 0
    if manual_df.empty:
        return long_df
    key = ['CodConglPrud', 'AnoMes', 'rate_index']
    base = long_df.merge(manual_df[key].assign(_drop=1), on=key, how='left')
    base = base[base['_drop'].isna()].drop(columns=['_drop'])
    out = pd.concat([base, manual_df], ignore_index=True)
    return out.sort_values(['CodConglPrud', 'rate_index', 'AnoMes']).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--in', dest='in_path',
                    default=str(ip_scrape_dir / 'extracted_historical_rates.csv'),
                    help='Input CSV from scrape_deposit_rate_parse.py.')
    ap.add_argument('--out-long',
                    default=str(ip_scrape_dir / 'advertised_rates_quarterly.csv'),
                    help='Long-form output CSV.')
    ap.add_argument('--out-wide',
                    default=str(ip_scrape_dir / 'advertised_rates_quarterly_wide.csv'),
                    help='Wide-form output CSV (one row per cod x quarter).')
    ap.add_argument('--out-dropped',
                    default=str(ip_scrape_dir / 'advertised_rates_dropped.csv'),
                    help='Audit CSV of rows filtered out.')
    ap.add_argument('--no-ffill', action='store_true',
                    help='Do not forward-fill quarters between observations.')
    ap.add_argument('--macro-rates',
                    default=str(_panel_intermed / 'quarterly_macro_rates.csv'),
                    help='quarterly_macro_rates.csv used for abs-rate conversion.')
    ap.add_argument('--no-abs-convert', action='store_true',
                    help='Skip absolute-rate conversion; keep as out_of_range.')
    ap.add_argument('--manual-overrides',
                    default=str(ip_scrape_dir / 'manual_advertised_rates.csv'),
                    help='CSV of operator-provided seed rates that override '
                         'scraped/ffilled values. Skipped silently if file '
                         'does not exist. Set to empty string to disable.')
    args = ap.parse_args()

    in_path = Path(args.in_path)
    logging.info(f"Loading {in_path}")
    raw = load_raw(in_path)
    logging.info(f"Raw rate mentions: {len(raw):,}")

    kept, dropped = clean_and_filter(raw)
    logging.info(f"Kept: {len(kept):,}   Dropped: {len(dropped):,}")
    if not dropped.empty:
        by_reason = dropped['drop_reason'].value_counts().to_dict()
        logging.info(f"Drop reasons: {by_reason}")

    if not args.no_abs_convert:
        macro_rates = _load_macro_rates(args.macro_rates)
        rescued, dropped = rescue_abs_rates(dropped, macro_rates)
        if not rescued.empty:
            kept = pd.concat([kept, rescued], ignore_index=True)
            logging.info(
                f"After abs-rate rescue: {len(kept):,} kept, {len(dropped):,} dropped"
            )

    quarterly = aggregate_quarterly(kept)
    logging.info(
        f"Quarterly aggregates: {len(quarterly):,} rows, "
        f"{quarterly['CodConglPrud'].nunique()} conglomerates, "
        f"{quarterly['AnoMes'].nunique()} quarters"
    )

    if args.no_ffill:
        long_df = quarterly
    else:
        long_df = forward_fill_quarters(quarterly)
        logging.info(
            f"After forward-fill: {len(long_df):,} rows "
            f"({(long_df['is_filled'] == 1).sum():,} filled)"
        )

    # Apply operator-provided manual overrides (Lever 4)
    manual_df = _load_manual_overrides(args.manual_overrides) if args.manual_overrides else pd.DataFrame()
    if not manual_df.empty:
        n_before = len(long_df)
        long_df = _apply_manual_overrides(long_df, manual_df)
        logging.info(
            f"Manual overrides: {len(manual_df):,} seed rows merged "
            f"({len(long_df) - n_before + len(manual_df):,} insertions/replacements). "
            f"Source: {args.manual_overrides}"
        )
    else:
        if 'is_manual' not in long_df.columns:
            long_df['is_manual'] = 0

    wide_df = pivot_wide(long_df)

    out_long = Path(args.out_long); out_long.parent.mkdir(parents=True, exist_ok=True)
    out_wide = Path(args.out_wide)
    out_drop = Path(args.out_dropped)
    long_df.to_csv(out_long, index=False)
    wide_df.to_csv(out_wide, index=False)
    dropped.to_csv(out_drop, index=False)

    logging.info(f"Wrote long-form:  {out_long}  ({len(long_df):,} rows)")
    logging.info(f"Wrote wide-form:  {out_wide}  ({len(wide_df):,} rows)")
    logging.info(f"Wrote dropped:    {out_drop}  ({len(dropped):,} rows)")

    # Top-line preview
    if not wide_df.empty:
        n_cdi = wide_df['advertised_cdi_pct'].notna().sum()
        n_sel = wide_df['advertised_selic_pct'].notna().sum()
        print(f"\nCoverage: {n_cdi:,} (cod, quarter) cells with CDI rate, "
              f"{n_sel:,} with SELIC rate.")
        print(f"Conglomerates with any CDI obs: "
              f"{wide_df.loc[wide_df['advertised_cdi_pct'].notna(), 'CodConglPrud'].nunique()}")
        print(f"Quarter range: {int(wide_df['AnoMes'].min())} -> {int(wide_df['AnoMes'].max())}")


if __name__ == '__main__':
    main()
