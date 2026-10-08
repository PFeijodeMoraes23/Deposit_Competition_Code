"""
diag_cosif_timeagg.py
=====================
Diagnostic for the COSIF time-aggregation see-saw that pollutes inferred
spreads and the sleepiness estimation.  Confirms two defects before we fix
them:

  (1) EXPENSE NOT DISACCUMULATED.  `Despesa_Captacao_Marginal` in the
      pipeline output `custos_implicitos_*.csv` still holds COSIF
      cumulative-to-date balances (group 8 / acct 811).  We detect, per
      institution, where the cumulative series RESETS (a month-over-month
      drop) and report whether the reset period is annual (Jan only) or
      semiannual (Jan + Jul), and how heterogeneous it is across banks.

  (2) STOCK DOUBLED AT SEMESTER CLOSE.  At June and December BCB ships a
      second document `4016` (semester balanco patrimonial) alongside the
      monthly balancete `4010`.  The (missing) producer summed both, so every
      group-4 balance is double-counted.  We quantify the doubling factor in
      the pipeline output AND confirm in the RAW files that (a) 4016 appears
      only at Jun/Dec, (b) it duplicates group-4 rows, and (c) whether it also
      carries group-8 (expense) rows.

Read-only.  Writes a short text report next to this script.

Usage:
  python diagnostics/diag_cosif_timeagg.py
"""
from __future__ import annotations

import glob
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root: utils/ and pipeline modules
try:
    from utils.venv_guard import ensure_project_venv
except Exception:
    ensure_project_venv = None
if ensure_project_venv is not None:
    ensure_project_venv(__file__)

from utils import paths

RAW = paths.COSIF_RAW
PROC = Path(str(paths.COSIF_RAW)).parent.parent / "BCB" / "Egan_et_al_2025_Rep" / "processed" / "COSIF_PROCESSED"
REPORT = Path(__file__).resolve().parent / "cosif_timeagg_diag_report.txt"

_lines: list[str] = []


def out(s: str = "") -> None:
    print(s)
    _lines.append(s)


def _sal(s: pd.Series) -> pd.Series:
    return pd.to_numeric(
        s.astype(str).str.replace(".", "", regex=False).str.replace(",", ".", regex=False),
        errors="coerce",
    ).fillna(0.0)


def _load_raw(ym: str, cat: str = "BANCOS") -> pd.DataFrame | None:
    for fp in glob.glob(str(RAW / f"{ym}{cat}.*")):
        try:
            if fp.lower().endswith(".csv"):
                df = pd.read_csv(fp, sep=";", encoding="latin1", dtype=str, skiprows=3)
            else:
                with zipfile.ZipFile(fp) as zf:
                    with zf.open(zf.namelist()[0]) as f:
                        df = pd.read_csv(f, sep=";", encoding="latin1", dtype=str, skiprows=3)
            df.columns = [c.replace("#", "").strip() for c in df.columns]
            return df
        except Exception:
            continue
    return None


# ---------------------------------------------------------------------------
# (1) Reset-period detection from the pipeline's monthly cumulative expense
# ---------------------------------------------------------------------------
def diagnose_reset_period() -> None:
    out("=" * 72)
    out("(1) EXPENSE DISACCUMULATION — reset-period detection")
    out("=" * 72)

    f = PROC / "custos_implicitos_BANCOS_MULTIPLOS.csv"
    df = pd.read_csv(f, dtype={"CNPJ": str})
    df["DATA_BASE"] = pd.to_datetime(df["DATA_BASE"])
    df["year"] = df["DATA_BASE"].dt.year
    df["month"] = df["DATA_BASE"].dt.month
    col = "Despesa_Captacao_Marginal"

    # Keep institutions with a meaningful, mostly-complete monthly series
    sizes = df.groupby("CNPJ")[col].apply(lambda s: (s.abs() > 0).sum())
    big_exp = df.groupby("CNPJ")[col].max().sort_values(ascending=False)
    cands = [c for c in big_exp.index if sizes.get(c, 0) >= 24][:40]

    out(f"\nAnalyzing {len(cands)} institutions with >=24 nonzero monthly expense obs.\n")

    reset_month_counter = {m: 0 for m in range(1, 13)}
    classifications = {"annual (Jan only)": 0, "semiannual (Jan+Jul)": 0, "other/irregular": 0}
    per_inst = []

    for cnpj in cands:
        sub = df[df["CNPJ"] == cnpj].sort_values("DATA_BASE")
        v = sub[col].abs().values
        months = sub["month"].values
        if len(v) < 12:
            continue
        # A reset = a month whose cumulative value drops materially below the
        # previous month (cumulative should otherwise rise within a period).
        prev = v[:-1]
        cur = v[1:]
        drop = cur < 0.5 * prev  # >50% drop signals period reset
        reset_months = months[1:][drop]
        rm_set = set(int(m) for m in reset_months)
        for m in rm_set:
            reset_month_counter[m] += 1
        # Classify by the dominant reset months
        jan = sum(1 for m in reset_months if m == 1)
        jul = sum(1 for m in reset_months if m == 7)
        n_years = max(1, (len(v)) // 12)
        if jul >= 0.5 * n_years and jan >= 0.5 * n_years:
            cls = "semiannual (Jan+Jul)"
        elif jan >= 0.5 * n_years and jul == 0:
            cls = "annual (Jan only)"
        else:
            cls = "other/irregular"
        classifications[cls] += 1
        per_inst.append((cnpj, sorted(rm_set), cls))

    out("Reset-month frequency across institutions (month -> # institutions that")
    out("show a >50% cumulative drop entering that month):")
    for m in range(1, 13):
        bar = "#" * reset_month_counter[m]
        out(f"  month {m:2d}: {reset_month_counter[m]:3d}  {bar}")

    out("\nPer-institution reset-period classification:")
    for cls, n in classifications.items():
        out(f"  {cls:24s}: {n}")

    out("\nSample (first 12 institutions):")
    for cnpj, rms, cls in per_inst[:12]:
        out(f"  CNPJ {cnpj}: resets at months {rms}  -> {cls}")

    out("\n=> Disaccumulation must be PER-INSTITUTION and reset-aware: a fixed")
    out("   'reset annually' rule would be wrong for semiannual filers (and")
    out("   vice-versa).  Robust rule: flow_m = cum_m - cum_{m-1} within a run;")
    out("   when cum drops (new period), flow_m = cum_m.")


# ---------------------------------------------------------------------------
# (2) Stock doubling — quantify in pipeline output AND confirm raw mechanism
# ---------------------------------------------------------------------------
def diagnose_stock_doubling() -> None:
    out("\n" + "=" * 72)
    out("(2) STOCK DOUBLING at semester close (4010 + 4016 double-count)")
    out("=" * 72)

    # --- 2a. Pipeline output: ratio of Jun/Dec stock to neighbour months ---
    f = PROC / "custos_implicitos_BANCOS_MULTIPLOS.csv"
    df = pd.read_csv(f, dtype={"CNPJ": str})
    df["DATA_BASE"] = pd.to_datetime(df["DATA_BASE"])
    df["year"] = df["DATA_BASE"].dt.year
    df["month"] = df["DATA_BASE"].dt.month
    df = df[df["Estoque_Total"] > 0]

    ratios = {6: [], 12: []}
    for (cnpj, yr), sub in df.groupby(["CNPJ", "year"]):
        s = sub.set_index("month")["Estoque_Total"]
        for close_m, nb in [(6, [5, 7]), (12, [11])]:
            nbvals = [s[m] for m in nb if m in s.index and s[m] > 0]
            if close_m in s.index and nbvals:
                ratios[close_m].append(s[close_m] / np.mean(nbvals))

    out("\nPipeline output: Estoque_Total at close month / mean(neighbour months):")
    for m, label in [(6, "June  (vs May/Jul)"), (12, "December (vs Nov)")]:
        arr = np.array(ratios[m])
        arr = arr[np.isfinite(arr)]
        if len(arr):
            out(f"  {label}: median={np.median(arr):.2f}  mean={arr.mean():.2f}  "
                f"n={len(arr)}  (=2.0 means perfectly doubled)")

    # --- 2b. Raw files: confirm 4016 presence, group-4 dup, group-8 presence ---
    out("\nRaw COSIF (BANCOS) document structure by month, Itau CNPJ 60701190:")
    out(f"  {'month':8s}{'DOCUMENTOs':18s}{'grp4 rows 4010/4016':22s}{'grp8 rows 4010/4016':22s}{'sum(41) ratio':14s}")
    base41 = {}
    for ym in ["202305", "202306", "202307", "202311", "202312"]:
        d = _load_raw(ym)
        if d is None:
            out(f"  {ym}: (missing)")
            continue
        itau = d[d["CNPJ"].str.startswith("60701190", na=False)].copy()
        itau["g"] = itau["CONTA"].astype(str).str[0]
        docs = sorted(itau["DOCUMENTO"].dropna().unique())
        g4 = itau[itau["CONTA"].astype(str).str.startswith("4")]
        g8 = itau[itau["CONTA"].astype(str).str.startswith("8")]
        g4_by = g4.groupby("DOCUMENTO").size().to_dict()
        g8_by = g8.groupby("DOCUMENTO").size().to_dict()
        a41 = _sal(itau[itau["CONTA"].astype(str).str.startswith("41")]["SALDO"]).sum()
        a41_4010 = _sal(itau[(itau["CONTA"].astype(str).str.startswith("41")) & (itau["DOCUMENTO"] == "4010")]["SALDO"]).sum()
        mm = int(ym[4:6])
        base41[mm] = a41_4010
        g4s = f"{g4_by.get('4010',0)}/{g4_by.get('4016',0)}"
        g8s = f"{g8_by.get('4010',0)}/{g8_by.get('4016',0)}"
        out(f"  {ym:8s}{str(docs):18s}{g4s:22s}{g8s:22s}{a41/ max(a41_4010,1):.2f}x (all/4010)")

    out("\n=> If 4016 appears only at Jun/Dec, duplicates group-4 rows, and the")
    out("   all-docs/4010-only ratio ~2.0, the doubling is the 4016 balanco")
    out("   double-count.  If 4016 has 0 group-8 rows, the EXPENSE is NOT")
    out("   doubled (only the cumulative-flow defect applies to it).")


def main() -> None:
    out("COSIF time-aggregation diagnostic")
    out(f"RAW : {RAW}")
    out(f"PROC: {PROC}")
    diagnose_reset_period()
    diagnose_stock_doubling()
    REPORT.write_text("\n".join(_lines), encoding="utf-8")
    out(f"\nReport written -> {REPORT}")


if __name__ == "__main__":
    main()
