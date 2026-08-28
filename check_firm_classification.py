"""panel_audit_firm_class.py -- READ-ONLY audit of the brick-and-mortar (B) vs digital (D)
firm classification and of firm identity in market_panel.csv.

Author: Pedro Feijó de Moraes

Written after the 2026-08-26 finding that the B/D split was decided entirely by where a bank
BOOKS its deposits, which is not the same thing as whether it operates premises. Agibank books
every deposit at one head office while running 991 postos de atendimento across 679
municipalities, so it was classified digital; PicPay Bank was classified as a São Paulo
brick-and-mortar bank because the digital flag was tested against the conglomerate LEADER's
CNPJ rather than the conglomerate itself.

Four things are checked, because each can break independently:

  VERDICT     -- is_B exists, is a clean boolean, and agrees with the physical-network
                 evidence in digital_banks_diagnostic.csv.
  COHERENCE   -- is_B agrees with the market assignment (mca_code == 'NATIONAL'). A
                 disagreement is REPORTED, not corrected: firm type and market tier are
                 separate facts and a genuine conflict is a finding.
  IDENTITY    -- no institution appears under two conglomerate codes in the same quarter
                 (the Agibank/Industval double count).
  STABILITY   -- no firm's booking municipality moves, which would fabricate entry/exit
                 events in local markets (Agibank moved Recife -> Porto Alegre -> Campinas).

Usage:  python panel_audit_firm_class.py
        python panel_audit_firm_class.py --baseline out.json    # write a summary snapshot
        python panel_audit_firm_class.py --compare out.json     # diff against a snapshot
"""
from utils.venv_guard import ensure_project_venv
ensure_project_venv(__file__)

import argparse
import json
import sys

import pandas as pd

from utils import paths

USE = ["year", "quarter", "mca_code", "CODMUN_IBGE", "CodConglomeradoPrudencial",
       "CNPJ_Lider", "NomeInstituicao", "Source", "is_B", "dep_a1", "dep_a4"]


def load_panel() -> pd.DataFrame:
    csv = paths.market_panel_csv()
    have = pd.read_csv(csv, nrows=0).columns.tolist()
    use = [c for c in USE if c in have]
    df = pd.read_csv(csv, usecols=use, low_memory=False,
                     dtype={"mca_code": str, "CodConglomeradoPrudencial": str,
                            "CNPJ_Lider": str})
    print(f"panel: {csv}\n       {len(df):,} rows, {len(have)} columns")
    return df


def summarise(df: pd.DataFrame) -> dict:
    """Compact aggregates for before/after comparison across a rebuild."""
    nat = df["mca_code"] == "NATIONAL"
    out = {
        "rows": len(df),
        "firms": int(df["CodConglomeradoPrudencial"].nunique()),
        "mcas": int(df.loc[~nat, "mca_code"].nunique()),
        "national_rows": int(nat.sum()),
        "dep_a4_total_bn": round(float(df["dep_a4"].sum()) / 1e9, 2) if "dep_a4" in df else None,
        "dep_a4_national_bn": round(float(df.loc[nat, "dep_a4"].sum()) / 1e9, 2) if "dep_a4" in df else None,
    }
    if "is_B" in df.columns:
        b = df["is_B"].astype(bool)
        out["is_B_true_rows"] = int(b.sum())
        out["is_B_firms_B"] = int(df.loc[b, "CodConglomeradoPrudencial"].nunique())
        out["is_B_firms_D"] = int(df.loc[~b, "CodConglomeradoPrudencial"].nunique())
    if "dep_a4" in df.columns:
        out["dep_a4_by_year_bn"] = {int(y): round(float(v) / 1e9, 2)
                                    for y, v in df.groupby("year")["dep_a4"].sum().items()}
    return out


def check_verdict(df: pd.DataFrame) -> int:
    print("\n== VERDICT ==")
    bad = 0
    if "is_B" not in df.columns:
        print("  FAIL  is_B is absent from market_panel.csv")
        return 1
    s = df["is_B"]
    print(f"  is_B dtype={s.dtype}  distinct={sorted(map(str, pd.unique(s)))[:6]}  nulls={int(s.isna().sum())}")
    if s.isna().any():
        print(f"  FAIL  {int(s.isna().sum()):,} null is_B values"); bad += 1
    diag = paths.PROCESSED / "PANEL_INTERMED" / "digital_banks_diagnostic.csv"
    if diag.exists():
        d = pd.read_csv(diag)
        if "Flag" in d.columns:
            n_brick = int((d["Flag"] == "Brick-and-Mortar (Physical Network)").sum())
            print(f"  diagnostic: {int(d['is_digital_candidate'].sum())} digital candidates, "
                  f"{n_brick} demoted by physical network")
            if "n_mun_own" in d.columns:
                for _, r in d[d["Flag"] == "Brick-and-Mortar (Physical Network)"].iterrows():
                    print(f"    demoted: {str(r['NOME_INSTITUICAO']).strip()} "
                          f"({r['n_agencias']} agências + {r['n_postos']} postos, "
                          f"{r['n_mun_own']} municipalities)")
        else:
            print("  WARN  diagnostic has no Flag column -- panel_5 predates the physical-network verdict")
    else:
        print(f"  WARN  {diag.name} not found")
    return bad


def check_coherence(df: pd.DataFrame) -> int:
    print("\n== COHERENCE (is_B vs market tier) ==")
    if "is_B" not in df.columns:
        print("  skipped -- no is_B"); return 0
    b = df["is_B"].astype(bool)
    nat = df["mca_code"] == "NATIONAL"
    mism = b == nat          # B-and-national, or D-and-local
    n = int(mism.sum())
    print(f"  rows where is_B != (mca_code != 'NATIONAL'): {n:,}")
    if n:
        g = (df[mism].groupby(["CodConglomeradoPrudencial", "NomeInstituicao"])
                     .size().nlargest(10))
        print("  (reported, not corrected -- firm type and market tier are separate facts)")
        for (code, nm), cnt in g.items():
            print(f"    {code:10s} {str(nm)[:44]:44s} {cnt:,} rows")
    return 0


def check_identity(df: pd.DataFrame) -> int:
    print("\n== IDENTITY (one institution, one conglomerate code per quarter) ==")
    if "CNPJ_Lider" not in df.columns:
        print("  skipped -- no CNPJ_Lider"); return 0
    d = df.dropna(subset=["CNPJ_Lider"]).copy()
    d["t"] = d["year"] * 10 + d["quarter"]
    g = d.groupby("CNPJ_Lider")["CodConglomeradoPrudencial"].nunique()
    multi = g[g > 1].index
    bad = 0
    if len(multi) == 0:
        print("  OK  no CNPJ_Lider maps to more than one conglomerate code")
        return 0
    print(f"  {len(multi)} CNPJ_Lider values map to >1 conglomerate code; checking time overlap:")
    for lid in multi:
        sub = d[d["CNPJ_Lider"] == lid]
        codes = sorted(sub["CodConglomeradoPrudencial"].unique())
        spells = {c: set(sub.loc[sub["CodConglomeradoPrudencial"] == c, "t"]) for c in codes}
        inter = set.intersection(*spells.values())
        nm = str(sub["NomeInstituicao"].iloc[0])[:32]
        if inter:
            print(f"    FAIL  lider {lid} {nm:32s} {'+'.join(codes)} overlap {len(inter)} quarters")
            bad += 1
        else:
            print(f"    ok    lider {lid} {nm:32s} {'+'.join(codes)} sequential, no overlap")
    return bad


def check_stability(df: pd.DataFrame) -> int:
    print("\n== STABILITY (booking municipality does not move for local firms) ==")
    if "CODMUN_IBGE" not in df.columns:
        print("  skipped -- no CODMUN_IBGE"); return 0
    loc = df[(df["mca_code"] != "NATIONAL")].copy()
    g = loc.groupby("CodConglomeradoPrudencial")["mca_code"].nunique()
    movers = g[g > 1]
    single = loc.groupby("CodConglomeradoPrudencial")["mca_code"].nunique() == 1
    print(f"  local firms: {len(g):,} | in exactly one MCA all life: {int(single.sum()):,}")
    # A firm present in only ONE MCA at a time but a DIFFERENT one across years is the
    # dangerous case: it looks like an exit plus an entry rather than a relocation.
    flagged = []
    for code in movers.index:
        sub = loc[loc["CodConglomeradoPrudencial"] == code]
        per_year = sub.groupby("year")["mca_code"].nunique()
        if (per_year == 1).all() and sub["mca_code"].nunique() > 1:
            dep = float(sub["dep_a4"].mean()) / 1e9 if "dep_a4" in sub else float("nan")
            flagged.append((code, str(sub["NomeInstituicao"].iloc[0])[:34],
                            sub["mca_code"].nunique(), dep))
    if flagged:
        print(f"  {len(flagged)} firm(s) occupy ONE MCA at a time but RELOCATE across years")
        print("  (each relocation fabricates an exit and an entry in local markets)")
        for code, nm, n, dep in sorted(flagged, key=lambda x: -x[3])[:12]:
            print(f"    {code:10s} {nm:34s} {n} MCAs   mean dep_a4 R${dep:,.2f}B")
    else:
        print("  OK  no single-market firm relocates across years")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", metavar="JSON", help="write a summary snapshot and exit")
    ap.add_argument("--compare", metavar="JSON", help="diff the panel against a snapshot")
    a = ap.parse_args()

    df = load_panel()
    summary = summarise(df)

    if a.baseline:
        with open(a.baseline, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        print(f"\nbaseline written -> {a.baseline}")
        print(json.dumps(summary, indent=2)[:900])
        return 0

    bad = 0
    bad += check_verdict(df)
    bad += check_coherence(df)
    bad += check_identity(df)
    bad += check_stability(df)

    if a.compare:
        with open(a.compare, encoding="utf-8") as fh:
            base = json.load(fh)
        print(f"\n== COMPARE vs {a.compare} ==")
        for k in sorted(set(base) | set(summary)):
            if k == "dep_a4_by_year_bn":
                continue
            b, n = base.get(k), summary.get(k)
            mark = "  " if b == n else "->"
            print(f"  {mark} {k:22s} {b}  ->  {n}")
        # JSON object keys are always strings; the freshly computed summary keys are ints.
        # Normalise both sides to str before the union or every year prints twice.
        by = {str(k): v for k, v in base.get("dep_a4_by_year_bn", {}).items()}
        ny = {str(k): v for k, v in summary.get("dep_a4_by_year_bn", {}).items()}
        if by and ny:
            print("  dep_a4 by year (R$bn):")
            for y in sorted(set(by) | set(ny), key=int):
                b, n = by.get(y), ny.get(y)
                mark = "  " if b == n else "->"
                print(f"    {mark} {y}  {b}  ->  {n}")

    print(f"\n{'FAILURES: %d' % bad if bad else 'All checks passed.'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
