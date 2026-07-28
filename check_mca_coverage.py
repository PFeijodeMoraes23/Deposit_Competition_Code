"""check_mca_coverage.py -- READ-ONLY audit of the MCA-level state/demographic variables.

Author: Pedro Feijó de Moraes

Written after the 2026-07-28 finding that `connections_per100`, `frac_4g5g` and
`branches_per1000` were present for only ~317 of 3,737 market MCAs (8.5%), because
anatel_mca_panel.csv and bcb_inclusion_mca_panel.csv had been built on 2026-06-02 against
the PRE-refresh municipality->MCA crosswalk (468 / 462 MCAs) while the deposit panel and
the IBGE demographics had picked up the 2026-07-22 crosswalk (5,476 MCAs).

Nothing raised an error at the time, and nothing would have: estimation_2_sleep's
`df[col] = df[col].fillna(df[col].median())` turns a failed merge into a median. That is
what this script exists to catch -- run it after any panel rebuild, BEFORE re-estimating.

Two things are checked, because either alone is insufficient:
  COVERAGE  -- what share of market MCAs carry a non-NaN value at all;
  VARIATION -- whether the values that ARE there actually differ across MCAs. A column
               that is 100% non-NaN but constant (a national series broadcast to every
               market) is exactly as broken as a column that is 90% NaN, and only the
               second check can see it.
"""
from utils.venv_guard import ensure_project_venv
ensure_project_venv(__file__)

import argparse
import sys

import numpy as np
import pandas as pd

from utils import paths as _paths_mod

# Variables carried at (MCA, time). The first block is the sleepiness state block used by
# estimation_2_sleep.define_specifications; the second is the BLP demographics (D_COLS).
STATE_VARS = ['gdp_per_capita', 'cadunico_families_per1000', 'fraction_65plus',
              'fraction_young', 'connections_per100']
DEMO_VARS = ['pop_total', 'pix_users_pf_per1000', 'frac_4g5g', 'branches_per1000']

# A variable below this coverage is almost certainly a broken merge rather than a data
# limit: the sources are national administrative panels.
MIN_COVERAGE = 0.90
# Below this cross-MCA coefficient of variation the column carries no local signal.
MIN_CV = 0.01


def main(year: int, strict: bool) -> int:
    pc = _paths_mod.PROCESSED / "market_panel_with_fees.csv"
    if not pc.exists():
        pc = _paths_mod.PROCESSED / "market_panel.csv"
    head = pd.read_csv(pc, nrows=0)
    sv = [c for c in STATE_VARS + DEMO_VARS if c in head.columns]
    print(f"panel: {pc.name}")

    df = pd.read_csv(pc, usecols=['mca_code', 'CODMUN_IBGE', 'year'] + sv,
                     dtype={'mca_code': str, 'CODMUN_IBGE': str}, low_memory=False)
    # D-type rows carry national pop-weighted fills by construction, so coverage is only
    # meaningful on the LOCAL (B-type) rows.
    b = df[df['CODMUN_IBGE'].astype(str) != '0']
    n_mca = b['mca_code'].nunique()
    # one row per (MCA, year): these are market attributes repeated across banks, and
    # leaving them repeated would weight each MCA by how many banks operate in it.
    mv = b.groupby(['mca_code', 'year'])[sv].first().reset_index()
    print(f"rows {len(df):,} | local MCAs {n_mca:,} | MCA-year cells {len(mv):,}\n")

    yr = mv[mv['year'] == year]
    if yr.empty:
        yr, year = mv[mv['year'] == mv['year'].max()], int(mv['year'].max())
        print(f"(requested year absent; using {year})\n")

    hdr = f"{'variable':<28s}{'block':>7s}{'cover':>8s}{'MCAs':>8s}{'distinct':>10s}{'CV':>8s}  status"
    print(hdr)
    print('-' * len(hdr))
    fails = []
    for c in sv:
        block = 'state' if c in STATE_VARS else 'demo'
        cov = mv.loc[mv[c].notna(), 'mca_code'].nunique() / n_mca
        v = yr[c].dropna()
        nd = int(v.nunique())
        cv = float(v.std() / abs(v.mean())) if len(v) and v.mean() != 0 else np.nan
        bad = []
        if cov < MIN_COVERAGE:
            bad.append(f"COVERAGE {cov:.1%}")
        if nd <= 2:
            bad.append("BROADCAST (<=2 distinct values)")
        elif np.isfinite(cv) and cv < MIN_CV:
            bad.append(f"NO LOCAL VARIATION (CV {cv:.4f})")
        status = 'ok' if not bad else 'FAIL: ' + '; '.join(bad)
        if bad:
            fails.append((c, block, status))
        print(f"{c:<28s}{block:>7s}{cov:>7.1%}{mv.loc[mv[c].notna(),'mca_code'].nunique():>8,}"
              f"{nd:>10,}{cv:>8.3f}  {status}")

    print()
    if fails:
        print(f"{len(fails)} variable(s) FAILED in year {year}:")
        for c, block, s in fails:
            print(f"  - {c} [{block}] {s}")
        print("\nA state-block failure changes phi and therefore every downstream object.")
        print("Likely cause: a source *_mca_panel.csv built against an older municipality->MCA")
        print("crosswalk than IBGE/muni_mca_regions_*.csv. Compare their mtimes and MCA counts,")
        print("regenerate the stale source panel, then rebuild the market panel.")
    else:
        print(f"All {len(sv)} variables pass coverage and variation checks in {year}.")
    return 1 if (fails and strict) else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--year", type=int, default=2023, help="year for the variation check")
    ap.add_argument("--strict", action="store_true", help="exit non-zero on any failure")
    a = ap.parse_args()
    sys.exit(main(a.year, a.strict))
