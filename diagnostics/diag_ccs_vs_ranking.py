"""diag_ccs_vs_ranking.py -- do the CCS relationship flows (LAI 18810.025400/2026-04) agree with the
client counts of the BCB complaints ranking?

What the ranking measures. The BCB's documentation of the ranking dataset ("Ranking de Instituicoes
por Indice de Reclamacoes - v1.0", p. 3, https://www.bcb.gov.br/conteudo/dadosabertos/BCBDeati/
Documentacao%20do%20Conjunto%20de%20dados%20-%20Ranking.pdf) defines "Quantidade de clientes - CCS"
as "Total de clientes com relacionamentos ativos na instituicao ou conglomerado financeiro", and the
joint CCS+SCR count as "nº de CPFs e CNPJs distintos". It states no reference date.

Why they should agree. The CCS keeps one relationship per CPF or CNPJ per institution, so at the
institution level (the ranking's "Banco/financeira" rows, keyed by CNPJ root) a client count is a
relationship count, and its change over a quarter is that quarter's openings minus closures:
    d_clients(q) = n_clients_ccs(q) - n_clients_ccs(q-1)   vs   net(q) = sum of opened_gross - closed
over the three months, PF and PJ together (the ranking does not split them). The identity holds if
the ranking's stock is taken at quarter ends and covers the same relationships; the gaps say where it
does not. At the conglomerate level a client is counted once in the ranking but once per institution
in the flows, so only institution rows are compared.

Pairs: institution x quarter, 2017Q1 to 2024Q4 (the ranking is quarterly from 2017), consecutive
quarters only.

Reference date. The fit is closest when the flows are summed over the three months ending at the
quarter's SECOND month (the reference-date test, lag k = 1), so the ranking's stock for quarter q is the
stock at the end of q's second month; net_to_month2 and gap_to_month2 are that comparison.

Outputs (utils.paths.rout_dir()):
  ccs_vs_ranking.csv            one row per pair
  ccs_vs_ranking_summary.json   correlation, median ratio, exact-match shares, the 10 largest gaps
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root: utils/
from utils import paths, raw_paths  # noqa: E402

FLOWS = raw_paths.BCB_CCS_LAI.parent / "compiled" / "ccs_flows_institution_month.csv"
RANKING = paths.OPEN_FINANCE / "BCB" / "Quality" / "processed" / "complaints_institution_quarterly.csv"
OUT = paths.rout_dir() / "ccs_vs_ranking.csv"
SUMMARY = paths.rout_dir() / "ccs_vs_ranking_summary.json"


def main():
    r = pd.read_csv(RANKING, dtype={"cnpj": str})
    r = r[r["year"].between(2017, 2024) & r["n_clients_ccs"].notna()].copy()
    r["qi"] = r["year"] * 4 + r["quarter"] - 1
    r = r.sort_values(["cnpj", "qi"]).drop_duplicates(["cnpj", "qi"], keep="last")
    r["prev"] = r.groupby("cnpj")["n_clients_ccs"].shift()
    r["prev_qi"] = r.groupby("cnpj")["qi"].shift()
    r = r[r["prev_qi"] == r["qi"] - 1].copy()
    r["d_clients"] = r["n_clients_ccs"] - r["prev"]

    f = pd.read_csv(FLOWS, dtype={"cnpj8": str})
    f = f[f["cnpj_missing"] == 0]
    f["qi"] = f["ym"].str[:4].astype(int) * 4 + (f["ym"].str[5:7].astype(int) - 1) // 3
    q = f.groupby(["cnpj8", "qi"])[["opened_gross", "closed"]].sum().reset_index()
    q["net"] = q["opened_gross"] - q["closed"]

    m = r.merge(q, left_on=["cnpj", "qi"], right_on=["cnpj8", "qi"], how="inner")
    m["quarter_label"] = (m["qi"] // 4).astype(str) + "Q" + (m["qi"] % 4 + 1).astype(str)
    m["gap"] = m["d_clients"] - m["net"]
    m["gap_share_of_stock"] = m["gap"] / m["prev"].where(m["prev"] > 0)
    m["ratio"] = m["d_clients"] / m["net"].where(m["net"].abs() >= 1000)
    out = m[["cnpj", "instituicao", "quarter_label", "prev", "n_clients_ccs", "d_clients", "opened_gross",
             "closed", "net", "gap", "gap_share_of_stock", "ratio"]].rename(
        columns={"cnpj": "cnpj8", "instituicao": "name", "quarter_label": "quarter", "prev": "n_clients_ccs_prev"})

    # Reference-date test: the net flow over the three months ending k months before the quarter's
    # last month, k = 0..3. The k with the closest fit is the lag of the ranking's stock.
    f["mi"] = f["ym"].str[:4].astype(int) * 12 + f["ym"].str[5:7].astype(int) - 1
    mon = f.groupby(["cnpj8", "mi"])[["opened_gross", "closed"]].sum()
    mon = (mon["opened_gross"] - mon["closed"]).rename("net_m").reset_index()
    align = {}
    for k in range(4):
        rows = []
        for cn, g in mon.groupby("cnpj8"):
            s = g.set_index("mi")["net_m"]
            rows.append(pd.DataFrame({"cnpj8": cn, "end_mi": s.index + k,
                                      "net_k": s.rolling(3, min_periods=3).sum().values}))
        w = pd.concat(rows).dropna()
        w["qi"] = (w["end_mi"] - 2) // 3            # the quarter whose last month is end_mi
        w = w[(w["end_mi"] % 3) == 2]
        a = m.merge(w, left_on=["cnpj", "qi"], right_on=["cnpj8", "qi"], how="inner", suffixes=("", "_w"))
        gap = a["d_clients"] - a["net_k"]
        align[f"lag_{k}_months"] = {
            "pairs": int(len(a)),
            "pearson": round(float(a["d_clients"].corr(a["net_k"])), 4),
            "share_|gap|<=1pct_of_stock": round(float((gap.abs() / a["prev"].where(a["prev"] > 0) <= 0.01).mean()), 4),
            "median_|gap|": float(gap.abs().median()),
        }
        if k == 1:
            a["quarter"] = (a["qi"] // 4).astype(str) + "Q" + (a["qi"] % 4 + 1).astype(str)
            out = out.merge(a[["cnpj", "quarter", "net_k"]].rename(
                columns={"cnpj": "cnpj8", "net_k": "net_to_month2"}), on=["cnpj8", "quarter"], how="left")
            out["gap_to_month2"] = out["d_clients"] - out["net_to_month2"]
    unmatched = r[~r["cnpj"].isin(set(f["cnpj8"]))]["cnpj"].nunique()
    big = out[out["n_clients_ccs_prev"] >= 100_000]
    summ = {
        "ranking_definition": "Quantidade de clientes - CCS: Total de clientes com relacionamentos ativos na "
                              "instituicao ou conglomerado financeiro (BCB, Documentacao do Conjunto de dados - "
                              "Ranking, v1.0, p. 3); no reference date stated",
        "pairs": int(len(out)), "institutions": int(out["cnpj8"].nunique()),
        "ranking_institutions_without_ccs_flows": int(unmatched),
        "pearson_d_vs_net": round(float(out["d_clients"].corr(out["net"])), 4),
        "spearman_d_vs_net": round(float(out["d_clients"].corr(out["net"], method="spearman")), 4),
        "median_ratio_d_over_net_(|net|>=1000)": round(float(out["ratio"].median()), 4),
        "share_exact_(gap==0)": round(float((out["gap"] == 0).mean()), 4),
        "share_|gap|<=1pct_of_stock": round(float((out["gap_share_of_stock"].abs() <= 0.01).mean()), 4),
        "institutions_ge_100k_clients": {
            "pairs": int(len(big)),
            "pearson_d_vs_net": round(float(big["d_clients"].corr(big["net"])), 4) if len(big) > 2 else None,
            "median_ratio": round(float(big["ratio"].median()), 4) if len(big) else None,
            "share_|gap|<=1pct_of_stock": round(float((big["gap_share_of_stock"].abs() <= 0.01).mean()), 4)
            if len(big) else None,
        },
        "reference_date_test": align,
        "largest_gaps": out.reindex(out["gap"].abs().sort_values(ascending=False).index).head(10)[
            ["cnpj8", "name", "quarter", "n_clients_ccs_prev", "n_clients_ccs", "d_clients", "net", "gap"]
        ].to_dict("records"),
    }
    out.to_csv(OUT, index=False)
    SUMMARY.write_text(json.dumps(summ, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print(json.dumps({k: v for k, v in summ.items() if k != "largest_gaps"}, indent=1, ensure_ascii=False))
    print(pd.DataFrame(summ["largest_gaps"]).to_string(index=False))


if __name__ == "__main__":
    main()
