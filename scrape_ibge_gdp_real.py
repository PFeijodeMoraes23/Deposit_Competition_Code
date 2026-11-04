"""
scrape_ibge_gdp_real.py
=======================
The real growth of GDP per capita by market (MCA) and year: the time block of routine E4
(`gdp_growth_yoy`, read by sleep_est_e2.add_time_variables).

Construction
------------
GDP      nominal municipal GDP (IBGE SIDRA table 5938, variable 37, R$ thousand), summed to the
         MCA: IBGE/mca_demographics_panel.csv (scrape_ibge_demographics.py), observed years only
         (gdp_imputed False; 2013-2023).
Deflator the national GDP deflator, IBGE national accounts (SIDRA table 6784, variable 9811, annual
         change in percent), chained into an index. SIDRA publishes no state deflator.
Persons  a population built for this column only: the Census 2010 and Census 2022 totals per
         municipality, summed to the MCA, interpolated geometrically between the two censuses and
         extended past 2022 at the MCA's own 2010-2022 rate. The panel's pop_total (estimates to
         2021, the Census 2022 level in 2022, projections after) is not used, so its 2022 break does
         not reach the growth. Census 2010: paths.IBGE_RAW/census_age_1552_2010_raw.csv, its
         `Total` rows. Census 2022: SIDRA table 4709, variable 93, every municipality (the local
         2022 cache lacks the five municipalities created in 2013).

Growth of year y = (GDP_y / P_y / pop_y) / (GDP_{y-1} / P_{y-1} / pop_{y-1}) - 1, for y = 2014..2023,
per MCA and nationally (summed GDP over summed population, row mca_code NATIONAL). Every quarter of
year t uses the growth of year t-1 (add_time_variables), so 2016 uses 2015 over 2014 and 2024 uses
2023 over 2022: all observed, nothing imputed.

Check: the national row against IBGE's own real per-capita growth (table 6784, variable 9814).

Writes  processed/PANEL_INTERMED/mca_gdp_real_growth.csv (mca_code, year, gdp_pc_real_growth) and
        processed/PANEL_INTERMED/mca_gdp_real_growth_report.json.
Fetches paths.IBGE_RAW/sidra_6784_gdp_annual.json, paths.IBGE_RAW/sidra_4709_census2022.json
        (cached; --refresh fetches again).

    python scrape_ibge_gdp_real.py [--refresh]
"""
import argparse
import json
import sys
import time

import numpy as np
import pandas as pd
import requests

from utils import paths

SIDRA = "https://apisidra.ibge.gov.br/values"
DEFLATOR_URL = f"{SIDRA}/t/6784/n1/all/v/9811,9814/p/all"
CENSUS22_URL = f"{SIDRA}/t/4709/n6/all/v/93/p/2022"
DEFLATOR_JSON = paths.IBGE_RAW / "sidra_6784_gdp_annual.json"
CENSUS22_JSON = paths.IBGE_RAW / "sidra_4709_census2022.json"
CENSUS10_CSV = paths.IBGE_RAW / "census_age_1552_2010_raw.csv"
GDP_CSV = paths.IBGE_DIR / "mca_demographics_panel.csv"
XWALK_CSV = paths.IBGE_DIR / "muni_mca_regions_2010_2024_panel.csv"
OUT_CSV = paths.PROCESSED / "PANEL_INTERMED" / "mca_gdp_real_growth.csv"
OUT_REPORT = paths.PROCESSED / "PANEL_INTERMED" / "mca_gdp_real_growth_report.json"
YEARS = range(2014, 2024)
# The five municipalities created in 2013, absent from the 2010-based crosswalk: in 2010 their
# territory is counted inside their parents, so their 2022 population goes to the parents' MCAs.
# Paraiso das Aguas came out of three municipalities in three MCAs: its population is split in
# proportion to the parents' 2010 census populations.
NEW_2013 = {"1504752": ["1506807"],                         # Mojui dos Campos (PA) <- Santarem
            "4212650": ["4209409"],                         # Pescaria Brava (SC) <- Laguna
            "4220000": ["4207007"],                         # Balneario Rincao (SC) <- Icara
            "4314548": ["4302105"],                         # Pinto Bandeira (RS) <- Bento Goncalves
            "5006275": ["5000203", "5003256", "5002951"]}   # Paraiso das Aguas (MS) <- Agua Clara,
                                                            #   Costa Rica, Chapadao do Sul


def fetch(url: str, dest, refresh: bool) -> list:
    """A SIDRA values response (list of dicts, the first one the header), cached in `dest`."""
    if dest.is_file() and not refresh:
        return json.loads(dest.read_text(encoding="utf-8"))
    for attempt in range(4):
        try:
            r = requests.get(url, timeout=120)
            r.raise_for_status()
            data = r.json()
            dest.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            return data
        except (requests.RequestException, ValueError) as e:
            print(f"  fetch attempt {attempt + 1} failed: {e}")
            time.sleep(5 * (attempt + 1))
    sys.exit(f"could not fetch {url}")


def sidra_frame(data: list) -> pd.DataFrame:
    """The rows of a SIDRA response, with the dimensions named by the header row."""
    head, rows = data[0], data[1:]
    df = pd.DataFrame(rows)
    names = {k: head[k] for k in head if k.startswith("D") and k.endswith("N")}
    out = pd.DataFrame({"value": pd.to_numeric(df["V"], errors="coerce")})
    for kn, nm in names.items():
        out[nm] = df[kn[:-1] + "C"]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--refresh", action="store_true", help="fetch the SIDRA tables again")
    a = ap.parse_args()

    # deflator index and IBGE's real per-capita growth
    nat = sidra_frame(fetch(DEFLATOR_URL, DEFLATOR_JSON, a.refresh))
    nat["year"] = nat["Ano"].astype(int)
    defl = nat[nat["Variável"] == "9811"].set_index("year")["value"].sort_index()
    ibge_pc = nat[nat["Variável"] == "9814"].set_index("year")["value"].sort_index() / 100.0
    if not set(range(2013, 2024)) <= set(defl.dropna().index):
        sys.exit(f"deflator (6784/9811) lacks years: {sorted(set(range(2013, 2024)) - set(defl.dropna().index))}")
    P = (1 + defl / 100.0).cumprod()

    # census populations by MCA
    xw = pd.read_csv(XWALK_CSV, dtype=str)
    xw = xw.sort_values("year").drop_duplicates("municipality_code", keep="last")
    m2mca = xw.set_index("municipality_code")["mca_code"]
    c10 = pd.read_csv(CENSUS10_CSV, dtype=str)
    c10 = c10[c10["age_label"] == "Total"].assign(pop=lambda d: pd.to_numeric(d["value"], errors="coerce"))
    c22 = sidra_frame(fetch(CENSUS22_URL, CENSUS22_JSON, a.refresh))
    c22 = c22.rename(columns={"Município": "municipio_code", "value": "pop"})[["municipio_code", "pop"]]
    report = {"census2010_municipalities": int(c10["municipio_code"].nunique()),
              "census2022_municipalities": int(c22["municipio_code"].nunique())}
    # the 2013 municipalities, re-assigned to their parents (shares by the parents' 2010 population)
    pop10 = c10.set_index("municipio_code")["pop"]
    moved, keep = [], c22[~c22["municipio_code"].isin(NEW_2013)]
    for new, parents in NEW_2013.items():
        p_new = c22.loc[c22["municipio_code"] == new, "pop"]
        if p_new.empty:
            continue
        w = pop10.reindex(parents)
        w = w / w.sum()
        for par, share in w.items():
            moved.append({"municipio_code": par, "pop": float(p_new.iloc[0]) * float(share)})
    c22 = pd.concat([keep, pd.DataFrame(moved)], ignore_index=True)
    report["census2022_reassigned_2013_municipalities"] = {k: v for k, v in NEW_2013.items()}
    unmapped = {"2010": sorted(set(c10["municipio_code"]) - set(m2mca.index)),
                "2022": sorted(set(c22["municipio_code"]) - set(m2mca.index))}
    report["unmapped_municipalities"] = {k: len(v) for k, v in unmapped.items()}
    p10 = c10.assign(mca=c10["municipio_code"].map(m2mca)).groupby("mca")["pop"].sum()
    p22 = c22.assign(mca=c22["municipio_code"].map(m2mca)).groupby("mca")["pop"].sum()

    # GDP by MCA, observed years
    gdp = pd.read_csv(GDP_CSV, dtype={"mca_code": str})
    gdp = gdp[~gdp["gdp_imputed"].astype(bool)][["mca_code", "year", "gdp_total_r1000"]]
    gdp = gdp.pivot(index="mca_code", columns="year", values="gdp_total_r1000")
    mcas = gdp.index.intersection(p10.index).intersection(p22.index)
    report["mca_gdp"] = int(len(gdp))
    report["mca_with_both_censuses"] = int(len(mcas))
    report["mca_without_census"] = sorted(set(gdp.index) - set(mcas))[:50]
    rate = (p22.reindex(mcas) / p10.reindex(mcas)) ** (1 / 12)

    def pop(mca_idx, y):
        return p10.reindex(mca_idx) * rate.reindex(mca_idx) ** (y - 2010)

    rows = []
    for y in YEARS:
        g_now, g_prev = gdp.loc[mcas, y], gdp.loc[mcas, y - 1]
        rpc_now = g_now / P[y] / pop(mcas, y)
        rpc_prev = g_prev / P[y - 1] / pop(mcas, y - 1)
        gr = (rpc_now / rpc_prev - 1).replace([np.inf, -np.inf], np.nan)
        rows.append(pd.DataFrame({"mca_code": mcas, "year": y, "gdp_pc_real_growth": gr.to_numpy()}))
        nat_now = g_now.sum() / P[y] / pop(mcas, y).sum()
        nat_prev = g_prev.sum() / P[y - 1] / pop(mcas, y - 1).sum()
        rows.append(pd.DataFrame({"mca_code": ["NATIONAL"], "year": [y], "gdp_pc_real_growth": [nat_now / nat_prev - 1]}))
    out = pd.concat(rows, ignore_index=True)
    # A 2013 municipality outside the crosswalk is its own market in the panel (UNKNOWN_<code>):
    # it takes its parents' MCA growth (several parents: weighted by their 2010 population).
    m_par = pd.DataFrame([{"new": n, "mca": m2mca[p], "w": float(pop10.get(p, np.nan))}
                          for n, ps in NEW_2013.items() for p in ps])
    gm = out.merge(m_par, left_on="mca_code", right_on="mca")
    orphan = (gm.assign(wg=gm["gdp_pc_real_growth"] * gm["w"])
              .groupby(["new", "year"]).agg(wg=("wg", "sum"), w=("w", "sum")).reset_index())
    orphan = pd.DataFrame({"mca_code": "UNKNOWN_" + orphan["new"], "year": orphan["year"],
                           "gdp_pc_real_growth": orphan["wg"] / orphan["w"]})
    out = pd.concat([out, orphan], ignore_index=True)
    report["orphan_markets_from_parents"] = sorted(orphan["mca_code"].unique())
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)

    natl = out[out["mca_code"] == "NATIONAL"].set_index("year")["gdp_pc_real_growth"]
    check = pd.DataFrame({"here": natl, "ibge_6784_9814": ibge_pc.reindex(natl.index)})
    check["diff_pp"] = 100 * (check["here"] - check["ibge_6784_9814"])
    report["national_check"] = {int(k): {c: round(float(v), 6) for c, v in r.items()} for k, r in check.iterrows()}
    loc = out[out["mca_code"] != "NATIONAL"]
    report["rows"] = int(len(out))
    report["missing_growth"] = int(loc["gdp_pc_real_growth"].isna().sum())
    report["by_year_mca_quantiles"] = {int(y): {q: round(float(v), 4) for q, v in
                                                g["gdp_pc_real_growth"].quantile([0.05, 0.5, 0.95]).items()}
                                       for y, g in loc.groupby("year")}
    OUT_REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {OUT_CSV} ({len(out):,} rows; {len(mcas):,} MCAs + NATIONAL, years {YEARS.start}-{YEARS.stop - 1})")
    print(f"MCAs with GDP {report['mca_gdp']:,}; with both censuses {report['mca_with_both_censuses']:,}; "
          f"unmapped municipalities {report['unmapped_municipalities']}; missing growth {report['missing_growth']}")
    print("national real per-capita growth, here vs IBGE (6784/9814):")
    print(check.round(4).to_string())


if __name__ == "__main__":
    main()
