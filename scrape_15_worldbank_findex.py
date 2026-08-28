## scrape_worldbank_findex.py
# Author: Pedro Feijó de Moraes
#
# Last edited: 2026-05-31
#
# Purpose: Pull World Bank Global Findex account-ownership / usage indicators for
#          Brazil, DISAGGREGATED BY DEMOGRAPHIC GROUP, to inform the demographic
#          loadings (Pi, Pi^q) on the extensive/intensive margins and to gauge
#          multi-homing correlates. Complements scrape_16 (which uses only the
#          headline FX.OWN.TOTL.ZS national series).
#
#   WHAT FINDEX CAN AND CANNOT DO FOR THIS PROJECT (verified 2026-05-31):
#     - SUBNATIONAL (municipality / state): NOT AVAILABLE. Findex publishes only
#       national figures for Brazil; the public microdata carries demographics but
#       NO municipality/state identifier. So Findex cannot feed the MCA-level
#       market panel directly.
#     - DEMOGRAPHIC CROSSTABS: AVAILABLE at the national level, by income
#       (poorest-40 / richest-60), gender, age (15-24 / 25+), and education
#       (primary-or-less / secondary-or-more), for survey years 2011, 2014, 2017,
#       2021, 2024. These are exactly the Nevo-style demographic moments that
#       discipline Pi / Pi^q: they pin down how account ownership and usage covary
#       with d_imt at the national level, which the within-MCA demographic sigma
#       (panel_9) then distributes across markets.
#     - MULTI-HOMING: not a native Findex indicator, but ownership of a financial-
#       institution account AND a mobile-money account is captured and is the
#       closest national proxy (see ACC_FI vs ACC_MOB below).
#
#   Output: WorldBank/Findex/findex_brazil_demographics.csv  (long format)
#     indicator, indicator_label, dimension, group, year, value_pct
#
#   CLI:
#     python scrape_worldbank_findex.py
#     python scrape_worldbank_findex.py --microdata-check   # report microdata availability only
###────────────────────────────────────────────────────────────────────────────

import os
import sys
import argparse
import logging

try:
    from utils.venv_guard import ensure_project_venv
except Exception:
    ensure_project_venv = None
if ensure_project_venv is not None:
    ensure_project_venv(__file__)

from utils.disclosure_common import get_json, http_get, data_root, now_iso

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    stream=sys.stdout)
log = logging.getLogger("scrape_21")

BASE = data_root(__file__)
OUT_DIR = os.path.join(BASE, "WorldBank", "Findex")
OUT_CSV = os.path.join(OUT_DIR, "findex_brazil_demographics.csv")

WB = "https://api.worldbank.org/v2/country/BR/indicator/{code}?format=json&per_page=200"

# (indicator code, human label, dimension, group). Account ownership at any
# institution + the FI-account and mobile-money variants for the multi-homing proxy.
INDICATORS = [
    ("FX.OWN.TOTL.ZS",    "Account ownership (% age 15+)",        "all",       "all"),
    ("FX.OWN.TOTL.40.ZS", "Account, income poorest 40%",          "income",    "poorest_40"),
    ("FX.OWN.TOTL.60.ZS", "Account, income richest 60%",          "income",    "richest_60"),
    ("FX.OWN.TOTL.FE.ZS", "Account, female",                      "gender",    "female"),
    ("FX.OWN.TOTL.MA.ZS", "Account, male",                        "gender",    "male"),
    ("FX.OWN.TOTL.YG.ZS", "Account, age 15-24",                   "age",       "young_15_24"),
    ("FX.OWN.TOTL.OL.ZS", "Account, age 25+",                     "age",       "older_25p"),
    ("FX.OWN.TOTL.PL.ZS", "Account, primary education or less",   "education", "primary_or_less"),
    ("FX.OWN.TOTL.SO.ZS", "Account, secondary education or more", "education", "secondary_or_more"),
    ("FX.OWN.TOTL.RU.ZS", "Account, rural",                       "location",  "rural"),
    # multi-homing proxy: FI account vs mobile-money account
    ("FX.OWN.TOTL.FI.ZS",  "Financial-institution account",       "channel",   "fi_account"),
    ("FX.OWN.TOTL.MO.ZS",  "Mobile-money account",                "channel",   "mobile_money"),
]

# Public Findex microdata releases (individual level; national, demographics, NO geo).
MICRODATA_LANDING = "https://microdata.worldbank.org/index.php/catalog/Findex"


def fetch_indicator(code: str):
    c, d = get_json(WB.format(code=code))
    out = []
    if isinstance(d, list) and len(d) > 1 and d[1]:
        for r in d[1]:
            if r.get("value") is not None:
                out.append((int(r["date"]), float(r["value"])))
    return sorted(out)


def microdata_check():
    """Report which Findex microdata files exist for Brazil and confirm no geo id."""
    log.info("Checking World Bank microdata catalog for Findex/Brazil ...")
    c, d = get_json("https://microdata.worldbank.org/index.php/api/catalog/search"
                    "?sk=findex&from=2011&to=2024&ps=50")
    found = []
    if isinstance(d, dict):
        rows = d.get("result", {}).get("rows", []) or []
        for r in rows:
            idno = str(r.get("idno", ""))
            title = str(r.get("title", ""))
            if "FINDEX" in idno.upper() or "findex" in title.lower():
                found.append((idno, title))
    log.info(f"Findex microdata datasets found: {len(found)} (showing Brazil-relevant)")
    for idno, title in found:
        if idno.upper().startswith("BRA") or "global" in title.lower() or "Findex" in title:
            log.info(f"   {idno}  {title}")
    log.info("NOTE: Findex microdata carries demographics (age/income/gender/education) "
             "but NO municipality/state identifier; it cannot localise to MCA. Manual "
             f"download (terms acceptance) at: {MICRODATA_LANDING}")
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--microdata-check", action="store_true")
    args = ap.parse_args()

    if args.microdata_check:
        microdata_check()
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    for code, label, dim, group in INDICATORS:
        series = fetch_indicator(code)
        if not series:
            log.warning(f"{code}: no data for Brazil (skipped)")
            continue
        for yr, val in series:
            rows.append(dict(indicator=code, indicator_label=label, dimension=dim,
                             group=group, year=yr, value_pct=val, retrieved_at=now_iso()))
        log.info(f"{code:20s} {dim:9s}/{group:18s} {len(series)} survey yrs")

    import pandas as pd
    df = pd.DataFrame(rows, columns=["indicator", "indicator_label", "dimension",
                                     "group", "year", "value_pct", "retrieved_at"])
    df = df.sort_values(["dimension", "group", "year"]).reset_index(drop=True)
    try:
        import pyarrow as pa, pyarrow.csv as pa_csv
        pa_csv.write_csv(pa.Table.from_pandas(df, preserve_index=False), OUT_CSV)
    except Exception:
        df.to_csv(OUT_CSV, index=False)
    log.info(f"Wrote {len(df)} rows -> {OUT_CSV}")

    # gradient summary that matters for Pi: ownership gap by demographic, latest year
    if not df.empty:
        latest = df["year"].max()
        print(f"\n--- demographic ownership gaps, {latest} (percentage points) ---")
        piv = df[df.year == latest].set_index("group")["value_pct"]
        for hi, lo, lab in [("richest_60", "poorest_40", "income (rich-poor)"),
                            ("male", "female", "gender (M-F)"),
                            ("older_25p", "young_15_24", "age (old-young)"),
                            ("secondary_or_more", "primary_or_less", "education (hi-lo)")]:
            if hi in piv.index and lo in piv.index:
                print(f"   {lab:22s}: {piv[hi]-piv[lo]:+5.1f} pp  ({piv[lo]:.1f} -> {piv[hi]:.1f})")
    log.info("Subnational note: Findex is NATIONAL only — see header. Use for Pi/Pi^q "
             "national demographic moments, not for MCA-level variation.")


if __name__ == "__main__":
    main()
