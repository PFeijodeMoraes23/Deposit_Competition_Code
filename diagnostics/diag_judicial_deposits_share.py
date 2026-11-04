"""
diag_judicial_deposits_share.py
===============================
Quarterly share of judicial deposits inside Cosif 4.1.5 (the base of dep_a4), step 4 of task JD1
(claude/bb_handoff_2026-11-03.md).

Before 2025 the public balancetes publish 4.1.5 only as a total, so the judicial stock comes from the banks'
DFP notes (year-ends, transcribed with the page in DATA_RAW/issuers/judicial_deposits/compiled/
dfp_judicial_extracts.csv). The judicial LEVEL is interpolated linearly between year-ends and divided by the
actual quarter-end 4.1.5 total, so movements of the total (for example BB's 2022 jump) are not smoothed away.
From 2025-01 the share is observed in Cosif (4155000009 over the 4.1.5 total).

Booking: Banrisul kept its judicial deposits outside 4.1.5 until the IN BCB 459/2024 reclassification
(data-base 2024-05; its 4.1.5 total rises R$9.0 bn that month), so its share inside 4.1.5 is zero before
2024Q2. The other banks' notes and Cosif totals show the judicial deposits inside 4.1.5 throughout.
Caixa does not file DFPs with the CVM; its year-ends come from its BRGAAP statements (investor-relations site),
whose time-deposit total equals its Cosif 4.1.5 total at every December 2016-2025.

Inputs (read only): the compiled DFP extracts above; Rout/cosif_415_subaccounts_monthly.csv
(diag_judicial_deposits_cosif.py).
Output: Rout/judicial_deposits_share.csv (conglomerate x quarter: judicial stock inside 4.1.5, the
institution and conglomerate 4.1.5 totals, both shares, source and flag), and a summary on screen.
Usage:  python diagnostics/diag_judicial_deposits_share.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from utils import paths

OUT = paths.rout_dir()
COSIF = OUT / "cosif_415_subaccounts_monthly.csv"
DFP_BANKS = {"00000000": "BB", "00000208": "BRB", "28127603": "Banestes", "13009717": "Banese", "92702067": "Banrisul",
             "00360305": "Caixa"}
RECLASS = 202405          # IN BCB 459/2024 data-base: Banrisul's judicial deposits enter 4.1.5


def extracts_path():
    here = Path(__file__).resolve()
    for up in here.parents:
        p = up / "DATA_RAW" / "issuers" / "judicial_deposits" / "compiled" / "dfp_judicial_extracts.csv"
        if p.exists():
            return p
    raise FileNotFoundError("dfp_judicial_extracts.csv not found above " + str(here))


def qidx(ym):
    return (ym // 100) * 4 + (ym % 100) // 3


def quarter_ends(m0=201512, m1=202603):          # 2015Q4 is the lag of 2016Q1
    out, y, m = [], m0 // 100, m0 % 100
    while y * 100 + m <= m1:
        out.append(y * 100 + m)
        m += 3
        if m > 12:
            y, m = y + 1, m - 12
    return out


def main():
    ex = pd.read_csv(extracts_path(), dtype={"cnpj": str})
    d = pd.to_datetime(ex["year_end"])
    ex["AnoMes"] = d.dt.year * 100 + d.dt.month            # year-ends, and Caixa's quarter-ends in 2023-2024
    ex["j"] = ex["judicial_thousand_brl"] * 1e3
    cos = pd.read_csv(COSIF, dtype={"CNPJ": str, "CodConglomeradoPrudencial": str})
    inst = cos[cos["level"] == "institution"]
    cg = cos[cos["level"] == "conglomerate"].set_index(["CodConglomeradoPrudencial", "AnoMes"])["stk_415_total"]
    qe = quarter_ends()

    # cross-check: DFP year-end 2025 against Cosif 4155 (BB)
    bb25 = inst[(inst["CNPJ"] == "00000000") & (inst["AnoMes"] == 202512)]["stk_4155_judicial"]
    if len(bb25):
        dfp25 = ex[(ex["cnpj"] == "00000000") & (ex["AnoMes"] == 202512)]["j"].iloc[0]
        print(f"check BB 2025-12: DFP judicial R$ {dfp25/1e9:.1f} bn, Cosif 4155 R$ {bb25.iloc[0]/1e9:.1f} bn")

    jd_banks = inst[(inst["AnoMes"] >= 202501) & (inst["stk_4155_judicial"] > 0)]["CNPJ"].unique()
    rows = []
    for c in sorted(set(jd_banks) | set(DFP_BANKS)):
        g = inst[inst["CNPJ"] == c].set_index("AnoMes")
        if g.empty:
            continue
        code = g["CodConglomeradoPrudencial"].dropna().iloc[-1]
        name = DFP_BANKS.get(c, g["NOME_INSTITUICAO"].dropna().iloc[-1])
        e = ex[ex["cnpj"] == c].set_index("AnoMes")
        pts = {}
        for ym, r in e.iterrows():
            pts[ym] = r["j"] if (r["inside_cosif_415"] == "yes" or ym >= RECLASS) else 0.0
        for ym in qe:
            tot = g["stk_415_total"].get(ym, np.nan)
            ctot = cg.get((code, ym), np.nan)
            if ym >= 202503:
                j = g["stk_4155_judicial"].get(ym, np.nan)
                j = 0.0 if pd.isna(j) and not pd.isna(tot) else j
                src, flag = "Cosif 4155000009", "observed"
            elif c not in DFP_BANKS:
                j, src, flag = np.nan, "", "not computed before 2025 (2025 share below 10%)"
            elif name == "Banrisul" and ym < RECLASS:
                j, src, flag = 0.0, "DFP 2024 (restated) and Cosif step 2024-05", "zero: booked outside 4.1.5 before IN 459/2024"
            elif ym in pts:
                j = pts[ym]
                r = e.loc[ym]
                src, flag = f"{r['document']} p{r['page']}", "DFP year-end" if ym % 100 == 12 else "statement quarter-end"
            else:
                before = [k for k in pts if k < ym]
                after = [k for k in pts if k > ym]
                if name == "Banrisul":
                    before = [k for k in before if k >= RECLASS - 100]   # level path from 2023-12 on
                if before and after:
                    a, b = max(before), min(after)
                    ia, ib = qidx(a), qidx(b)
                    w = (qidx(ym) - ia) / (ib - ia)
                    j = pts[a] + w * (pts[b] - pts[a])
                    if name == "Banrisul":
                        j = e.loc[a, "j"] + w * (e.loc[b, "j"] - e.loc[a, "j"])   # level, now inside
                    src = f"linear between {a // 100}-12 and {b // 100}-12"
                    flag = "interpolated" if (ib - ia) == 4 else f"interpolated across missing year-ends ({a // 100 + 1}-{b // 100 - 1})"
                elif after:
                    j = pts[min(after)]
                    src, flag = f"held from {min(after) // 100}-12", "held back from the first year-end"
                else:
                    j, src, flag = np.nan, "", "no DFP point after this quarter"
            rows.append(dict(CodConglomeradoPrudencial=code, bank=name, cnpj=c, quarter=f"{ym // 100}Q{(ym % 100) // 3}",
                             AnoMes=ym, judicial_inside_415_bn=j / 1e9 if pd.notna(j) else np.nan,
                             total_415_institution_bn=tot / 1e9, total_415_conglomerate_bn=ctot / 1e9,
                             share_institution=j / tot if pd.notna(j) and tot > 0 else np.nan,
                             share_conglomerate=j / ctot if pd.notna(j) and ctot > 0 else np.nan,
                             window="2015Q4 (lag of 2016Q1)" if ym < 201601 else "2016-2024" if ym <= 202412 else "2025+ (outside the window)", source=src, flag=flag))
    out = pd.DataFrame(rows).sort_values(["bank", "AnoMes"])
    out.to_csv(OUT / "judicial_deposits_share.csv", index=False, float_format="%.4f")
    print(f"wrote {OUT / 'judicial_deposits_share.csv'}: {len(out)} rows, {out['bank'].nunique()} banks")
    bad = out[(out["share_institution"] > 1.0)]
    print("rows with share above 1 (judicial above the 4.1.5 total):", len(bad))
    s = out.groupby(["bank", "window"])["share_conglomerate"].agg(["min", "max"]).unstack("window").round(3)
    print("\nShare of the conglomerate's 4.1.5 total, min and max by window:")
    print(s.to_string())
    yr = out[out["quarter"].str.endswith("Q4") & (out["AnoMes"] <= 202412) & out["bank"].isin(DFP_BANKS.values())]
    print("\nDecember share of the conglomerate 4.1.5 total, 2016-2024:")
    print(yr.pivot_table(index="quarter", columns="bank", values="share_conglomerate").round(3).to_string())


if __name__ == "__main__":
    main()
