"""
diagnostics/diag_pix_mandate_2019.py
====================================
Task R1b (coding handoff 2026-11-02, Update 11-02k): the 2019 share of B deposits, by type, held
by conglomerates with an institution mandated into Pix at launch. Same mapping as
make_pix_exposure.py: the 34 institutions of shared/PIX/pix_mandatory_institutions.csv whose first
mandatory roster is dated by 2020-11-16, CNPJ roots mapped to conglomerates with the 2019 IF.data
lists. Balances are summed over the four quarters of 2019 and over markets.

Prints the shares by type (1 demand, 2 savings, 4 time, 5 prepaid, and all four) and the
non-mandated B conglomerates with any 2019 savings. Reads only; writes nothing.

Usage:  python diagnostics/diag_pix_mandate_2019.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.venv_guard import ensure_project_venv
ensure_project_venv(__file__)

import pandas as pd

from make_pix_exposure import PANEL, MANDATORY, PIX_LAUNCH
from utils import conglomerates as cc

TYPES = (1, 2, 4, 5)


def main():
    p = pd.read_parquet(PANEL, columns=["CodConglomeradoPrudencial", "NomeInstituicao", "is_B", "year"]
                        + [f"dep_a{t}" for t in TYPES])
    p = p[(p["is_B"] == 1) & (p["year"] == 2019)]
    mand = pd.read_csv(MANDATORY, dtype=str)
    mand = mand[pd.to_datetime(mand["mandatory_since"]) <= PIX_LAUNCH]
    roots = set(mand["cnpj"].str.replace(r"\D", "", regex=True).str.zfill(14).str[:8])
    m = cc.build_cnpj_to_congl()
    m = m[m["AnoMes_list"].astype(float).between(201901, 201912)]
    congl = set(m.loc[m["CNPJ"].astype(str).str.zfill(8).isin(roots), "CodConglomeradoPrudencial"])
    p["mandated"] = p["CodConglomeradoPrudencial"].isin(congl)

    cols = [f"dep_a{t}" for t in TYPES]
    g = p.groupby("mandated")[cols].sum()
    g["all"] = g.sum(axis=1)
    share = g.loc[True] / g.sum()
    print(f"institutions mandated at launch: {len(mand)}; conglomerates (2019 lists): {len(congl)}; "
          f"B conglomerates in 2019: {p['CodConglomeradoPrudencial'].nunique()}, of which mandated "
          f"{p.loc[p['mandated'], 'CodConglomeradoPrudencial'].nunique()}")
    print("2019 B balances, R$ billion (summed over quarters), and the mandated share:")
    for c in cols + ["all"]:
        print(f"  {c:7s} total {g[c].sum() / 1e9:10.1f}   non-mandated {g.loc[False, c] / 1e9:8.3f}   "
              f"mandated share {share[c]:.6f}")
    nm = (p[~p["mandated"]].groupby("CodConglomeradoPrudencial")
          .agg(name=("NomeInstituicao", "last"), **{c: (c, "sum") for c in cols}))
    sav = nm[nm["dep_a2"] > 0].sort_values("dep_a2", ascending=False)
    print(f"non-mandated B conglomerates with 2019 savings: {len(sav)}")
    for k, r in sav.iterrows():
        print(f"  {k} {r['name'][:45]:45s} savings R$ {r['dep_a2'] / 1e6:10.1f} m (sum of 2019 quarters)")


if __name__ == "__main__":
    main()
