"""
make_dep_anova.py
=================
Order-free (Shapley) split of the balance-weighted variance of log deposits of B firms, cells
conglomerate x deposit type x MCA x quarter (market_panel.parquet, types 1, 2, 4, 5, positive
balances, the paper's year window), among bank, market, product (deposit type) and time (quarter)
effects and the rest; and a second split that adds bank x product and market x product effects.
Weights: the cell's deposit balance, as in tab_motiv_anova_levels.tex. The fixed-effect algebra
and the Shapley value are make_xi_diagnostics.py's; a set of effects is reduced to the space it
spans (bank x product contains bank and product), so equal spaces are fitted once.

Outputs: Drafts/Deposit Competition/tab_dep_anova.tex (label tab:dep_anova); Rout/dep_anova.json
(read by make_paper_numbers.py, desc.dep_anova.B.*, and make_slide_figures.py --only depanova).
"""
from utils.venv_guard import ensure_project_venv
ensure_project_venv(__file__)

import datetime
import itertools
import json
import math
import sys

import numpy as np
import pandas as pd

from utils import paths as _paths
from utils.window import MIN_YEAR, MAX_YEAR
from make_xi_diagnostics import _codes, _wr2, f3, compile_check

PANEL = _paths.PROCESSED / "market_panel.parquet"
SIDECAR = _paths.rout_dir() / "dep_anova.json"
TEX = _paths.drafts_dir() / "tab_dep_anova.tex"
TYPES = {"dep_a1": 1, "dep_a2": 2, "dep_a4": 4, "dep_a5": 5}
# A set of effects spans the space of its largest members: an interaction absorbs its parts.
NESTS = {"bank_product": ("bank", "product"), "market_product": ("market", "product")}


def frame() -> pd.DataFrame:
    d = pd.read_parquet(PANEL, columns=["CodConglomeradoPrudencial", "mca_code", "is_B", "year",
                                        "quarter", *TYPES])
    d = d[(d["is_B"] == 1) & d["year"].between(MIN_YEAR, MAX_YEAR)]
    d = d.melt(id_vars=["CodConglomeradoPrudencial", "mca_code", "year", "quarter"],
               value_vars=list(TYPES), var_name="col", value_name="dep")
    d = d[d["dep"] > 0].copy()
    d["k"] = d["col"].map(TYPES)
    d["t"] = d["year"] * 4 + d["quarter"] - 1
    return d.reset_index(drop=True)


def shapley(y, w, factors: dict) -> tuple[dict, float]:
    names = list(factors)
    n = len(names)
    cache = {}

    def v(S):
        S = set(S)
        for big, parts in NESTS.items():
            if big in S:
                S -= set(parts)
        key = tuple(sorted(S))
        if key not in cache:
            cache[key] = _wr2(y, w, [factors[f] for f in key]) if key else 0.0
        return cache[key]

    phi = {}
    for i in names:
        others = [f for f in names if f != i]
        phi[i] = sum(math.factorial(r) * math.factorial(n - r - 1) / math.factorial(n)
                     * (v(S + (i,)) - v(S))
                     for r in range(n) for S in itertools.combinations(others, r))
    return phi, v(names)


def main():
    d = frame()
    y, w = np.log(d["dep"].to_numpy()), d["dep"].to_numpy()
    f = dict(bank=_codes(d["CodConglomeradoPrudencial"]), market=_codes(d["mca_code"]),
             product=_codes(d["k"]), time=_codes(d["t"]))
    base, r2 = shapley(y, w, f)
    base["rest"] = 1.0 - r2
    fi = dict(f, bank_product=_codes(d["CodConglomeradoPrudencial"], d["k"]),
              market_product=_codes(d["mca_code"], d["k"]))
    inter, r2i = shapley(y, w, fi)
    inter["rest"] = 1.0 - r2i
    for s in (base, inter):
        assert abs(sum(s.values()) - 1.0) < 1e-9, s
    meta = dict(written=datetime.datetime.now().isoformat(timespec="seconds"),
                panel=str(PANEL), panel_mtime=datetime.datetime.fromtimestamp(
                    PANEL.stat().st_mtime).isoformat(timespec="seconds"),
                n_cells=int(len(d)), n_banks=int(d["CodConglomeradoPrudencial"].nunique()),
                n_markets=int(d["mca_code"].nunique()), n_quarters=int(d["t"].nunique()),
                years=[MIN_YEAR, MAX_YEAR], weights="deposit balance of the cell")
    SIDECAR.write_text(json.dumps(dict(B=dict(base=base, int=inter), _meta=meta), indent=2))
    print("base  " + ", ".join(f"{k} {v:.3f}" for k, v in base.items()))
    print("inter " + ", ".join(f"{k} {v:.3f}" for k, v in inter.items()))
    print(f"  {meta['n_cells']:,} cells, {meta['n_banks']} conglomerates, {meta['n_markets']:,} MCAs")

    labels = (("bank", "Bank"), ("market", "Market"), ("product", "Product (deposit type)"),
              ("time", "Time (quarter)"), ("bank_product", r"Bank $\times$ product"),
              ("market_product", r"Market $\times$ product"), ("rest", "Rest"))
    body = [f"\\quad {lab} & {f3(base[k]) if k in base else '---'} & {f3(inter[k])} \\\\"
            for k, lab in labels]
    body += [r"\midrule", f"\\quad Cells & {meta['n_cells']:,} & {meta['n_cells']:,} \\\\",
             f"\\quad Conglomerates & {meta['n_banks']:,} & {meta['n_banks']:,} \\\\"]
    note = (r"\textit{Notes:} Shares of the balance-weighted variance of log deposits of "
            r"brick-and-mortar (B) firms, conglomerate $\times$ deposit type $\times$ minimum "
            r"comparable area (MCA) $\times$ quarter cells with a positive balance, deposit types "
            r"1, 2, 4 and 5, "
            f"{MIN_YEAR}Q1 to {MAX_YEAR}Q4. "
            r"Weights: the cell's deposit balance. Order-free (Shapley) split among conglomerate "
            r"(bank), MCA (market), deposit type (product) and quarter (time) effects. The second "
            r"column adds bank $\times$ product and market $\times$ product effects, which "
            r"contain their parts, so a set of effects counts the space it spans.")
    tex = "\n".join([
        "% tab_dep_anova.tex -- GENERATED by make_dep_anova.py. Do not edit.",
        r"\begin{spacing}{1.0}", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
        r"\par\addvspace{\LTpre}\noindent\begin{minipage}{\textwidth}\captionsetup{position=top}",
        r"\captionof{table}{Variance of Log Deposits of B Firms: Bank, Market, Product and Time}"
        r"\label{tab:dep_anova}",
        r"\begin{tabularx}{\textwidth}{@{}X>{\raggedleft\arraybackslash}p{2.6cm}"
        r">{\raggedleft\arraybackslash}p{2.6cm}@{}}",
        r"\toprule", r" & Main effects & With interactions \\", r"\midrule", *body, r"\bottomrule",
        rf"\multicolumn{{3}}{{@{{}}p{{\dimexpr\textwidth-2\tabcolsep\relax}}@{{}}}}"
        rf"{{\footnotesize {note}}} \\",
        r"\end{tabularx}", r"\end{minipage}\par\prevdepth=0pt\addvspace{\LTpost}",
        r"\end{spacing}", ""])
    TEX.write_text(tex, encoding="utf-8")
    print(f"  wrote {TEX}")
    import make_xi_diagnostics as X
    X._STANDALONE = X._STANDALONE.replace("tab_xi_diagnostics.tex", "tab_dep_anova.tex")
    if not compile_check(TEX.parent):
        sys.exit("tab_dep_anova.tex did not compile cleanly on its own")


if __name__ == "__main__":
    main()
