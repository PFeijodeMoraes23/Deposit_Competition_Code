"""diag_ccs_main_entity.py -- how much double counting moves the D share of PF gross openings in the
CCS relationship flows (LAI 18810.025400/2026-04), asked in the handoff of 2026-11-02 (CCS1).
Descriptive: no paper numbers.

A relationship is one client (CPF) at one institution (CNPJ root). A conglomerate with several
institutions counts a client once per institution, and a client who adds a second institution of the
same conglomerate months later appears as a new opening. Four versions of the D share, by opening year:
  (a) as the key ccs.pf.dshare_open is built: every institution, D gross / all gross;
  (b) main entity only: one fixed institution per conglomerate, the one with the most opened-and-active
      PF relationships over 2016-2024, all months (a lower bound on the new clients of multi-entity
      conglomerates); an unmapped institution is its own conglomerate;
  (c) the cross bounds: D at its main entities against B and the rest at every institution (low),
      and D at every institution against B and the rest at their main entities (high);
  (d) every conglomerate with two or more institutions among the 20 largest by PF gross openings
      2016-2024, with each institution's openings and its CCS segment.

Reads (utils.paths.rout_dir()): ccs_flows_institution_month_mapped.csv (make_ccs_flows_conglomerate.py);
the segments from the compiled flows file. Writes rout_dir() / ccs_main_entity_ccs1.md.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root: utils/
from utils import paths, raw_paths  # noqa: E402

ROUT = paths.rout_dir()
MONTHLY = ROUT / "ccs_flows_institution_month_mapped.csv"
FLOWS = raw_paths.BCB_CCS_LAI.parent / "compiled" / "ccs_flows_institution_month.csv"
OUT = ROUT / "ccs_main_entity_ccs1.md"
GROUPS = ["B", "D", "not in panel"]


def md(df: pd.DataFrame, fmt="{:.3f}") -> str:
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join([str(df.index.name or "")] + cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for i, r in df.iterrows():
        out.append("| " + " | ".join([str(i)] + [fmt.format(v) if isinstance(v, (float, np.floating))
                                                  and np.isfinite(v) else str(v) for v in r]) + " |")
    return "\n".join(out)


def main():
    d = pd.read_csv(MONTHLY, dtype={"cnpj8": str})
    d = d[d["type"] == "PF"].copy()
    d["group"] = np.select([(d["in_panel"] == 1) & (d["is_B"] == 1), (d["in_panel"] == 1) & (d["is_B"] == 0)],
                           ["B", "D"], "not in panel")
    d["year"] = d["ym"].str[:4].astype(int)
    d = d[(d["year"] >= 2016) & (d["year"] <= 2024)]

    # the main entity of each conglomerate: most opened-and-active PF relationships over 2016-2024
    inst = d.groupby(["conglomerate", "cnpj8"]).agg(active=("opened_active", "sum"), gross=("opened_gross", "sum"),
                                                     name=("name", "first"), group=("group", "first")).reset_index()
    main_ent = (inst.sort_values(["active", "gross", "cnpj8"], ascending=[False, False, True])
                .drop_duplicates("conglomerate").set_index("conglomerate")["cnpj8"])
    d["is_main"] = d["cnpj8"].eq(d["conglomerate"].map(main_ent))

    all_g = d.groupby(["year", "group"])["opened_gross"].sum().unstack("group").reindex(columns=GROUPS).fillna(0)
    main_g = (d[d["is_main"]].groupby(["year", "group"])["opened_gross"].sum().unstack("group")
              .reindex(columns=GROUPS).fillna(0))
    rest = [g for g in GROUPS if g != "D"]
    t = pd.DataFrame({
        "(a) all institutions": all_g["D"] / all_g.sum(axis=1),
        "(b) main entities": main_g["D"] / main_g.sum(axis=1),
        "(c) low: D main, rest all": main_g["D"] / (main_g["D"] + all_g[rest].sum(axis=1)),
        "(c) high: D all, rest main": all_g["D"] / (all_g["D"] + main_g[rest].sum(axis=1)),
    })
    t.index.name = "opening year"
    gross = pd.DataFrame({f"{g} all (m)": all_g[g] / 1e6 for g in GROUPS} |
                         {f"{g} main (m)": main_g[g] / 1e6 for g in GROUPS})
    gross.index.name = "opening year"

    # (d) multi-institution conglomerates among the 20 largest by PF gross openings
    seg = (pd.read_csv(FLOWS, dtype={"cnpj8": str}, usecols=["cnpj8", "segment"]).dropna()
           .drop_duplicates("cnpj8").set_index("cnpj8")["segment"])
    cg = inst.groupby("conglomerate").agg(gross=("gross", "sum"), n_inst=("cnpj8", "nunique"),
                                          group=("group", "first"))
    top = cg.sort_values("gross", ascending=False).head(20)
    multi = top[top["n_inst"] >= 2]
    rows = []
    for c, r in multi.iterrows():
        for _, i in inst[inst["conglomerate"] == c].sort_values("gross", ascending=False).iterrows():
            rows.append(dict(conglomerate=c, group=r["group"], rank=int(list(top.index).index(c)) + 1,
                             institution=i["name"], cnpj8=i["cnpj8"], segment=seg.get(i["cnpj8"], ""),
                             main="yes" if main_ent.get(c) == i["cnpj8"] else "",
                             gross_m=i["gross"] / 1e6, active_m=i["active"] / 1e6,
                             share_of_conglomerate=i["gross"] / r["gross"]))
    tl = pd.DataFrame(rows).set_index("conglomerate")

    out = ["# CCS1: double counting and the D share of PF gross openings, 2016-2024", "",
           "PF relationships, LAI 18810.025400/2026-04. Main entity: the conglomerate's institution with the "
           "most opened-and-active PF relationships over 2016-2024, fixed for all months. Version (b) keeps only "
           "main entities, a lower bound on the new clients of multi-entity conglomerates; (c) mixes the two "
           "counts to bound the share.", "",
           "## D share of PF gross openings", "", md(t), "",
           "## Gross openings behind it (millions)", "", md(gross, "{:.2f}"), "",
           f"Conglomerates in the window: {inst['conglomerate'].nunique()}; with two or more institutions: "
           f"{int((inst.groupby('conglomerate')['cnpj8'].nunique() >= 2).sum())}.", "",
           "## Multi-institution conglomerates among the 20 largest by PF gross openings", "",
           md(tl, "{:.3f}"), ""]
    OUT.write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
