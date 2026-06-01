## analysis_1_disclosure_join.py
# Author: Pedro Feijó de Moraes
#
# Last edited: 2026-05-31
#
# Purpose: Join the firm-level public-disclosure data (customer / account counts
#          and matching deposit balances scraped by scrape_9..scrape_15) onto the
#          project's market panel, producing a CONGLOMERATE x QUARTER analysis
#          table for an "account-share vs volume-share" diagnostic — i.e. the
#          extensive- vs intensive-margin decomposition of the deposit market.
#          See V_Main.tex appendix `app:extensions:intensive`.
#
#   Idea: incumbents hold a large *volume* share with a moderate customer base
#         (high deposits-per-customer / intensive margin); digital banks acquire
#         a large *account* (extensive) share but very thin balances per customer.
#         This table puts the panel's national volume_share next to the firms'
#         disclosed customer counts so that contrast can be drawn.
#
#   NOTE on geography: the firm disclosures have NO municipality dimension — they
#         are headline group figures. This join is therefore deliberately a
#         FIRM x QUARTER (national) join, NOT an MCA-level join. Panel deposits are
#         aggregated over mca_code (summed) to the (conglomerate, year, quarter)
#         national level before merging.
#
#   NOTE on annual deposit obs: EDGAR deposit rows are annual (period_quarter
#         null). We BROADCAST each annual deposit figure to all four quarters of
#         its year (flagged deposits_annual_broadcast=True) so it lines up with the
#         quarterly panel; CVM deposits are already booked at Q4.
#
#   INPUTS  (BASE = .../Open-Finance, two levels above this repo):
#     BASE/BCB/Panel/market_panel.csv                      (cong x mca x year x quarter)
#     BASE/FirmDisclosures/SEC/edgar_disclosures.csv           (long; disclosure_common schema)
#     BASE/FirmDisclosures/CVM/cvm_deposits.csv                (long)
#     BASE/FirmDisclosures/Parent/parent_disclosures.csv       (long)
#     BASE/FirmDisclosures/Incumbents/incumbent_client_counts.csv  (long; scrape_11 Tier-2/4)
#     + utils.firm_registry.load_registry(BASE) for the firm_key -> conglomerate map
#
#   OUTPUT: BASE/BCB/Egan_et_al_2025_Rep/processed/DESCRIPTIVES/account_vs_volume_panel.csv
#
#   CLI:  PYTHONIOENCODING=utf-8 python analysis_1_disclosure_join.py
###────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import os
import re
import sys
import logging

try:
    from utils.venv_guard import ensure_project_venv
except Exception:  # noqa: BLE001
    ensure_project_venv = None
if ensure_project_venv is not None:
    ensure_project_venv(__file__)

import numpy as np
import pandas as pd

from utils.firm_registry import load_registry

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    stream=sys.stdout)
log = logging.getLogger("analysis_1")

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
BASE = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

PANEL_CSV = os.path.join(BASE, "BCB", "Panel", "market_panel.csv")
EDGAR_CSV = os.path.join(BASE, "FirmDisclosures", "SEC", "edgar_disclosures.csv")
CVM_CSV = os.path.join(BASE, "FirmDisclosures", "CVM", "cvm_deposits.csv")
PARENT_CSV = os.path.join(BASE, "FirmDisclosures", "Parent", "parent_disclosures.csv")
INCUMBENT_CSV = os.path.join(BASE, "FirmDisclosures", "Incumbents", "incumbent_client_counts.csv")

OUT_DIR = os.path.join(BASE, "BCB", "Egan_et_al_2025_Rep", "processed", "DESCRIPTIVES")
OUT_CSV = os.path.join(OUT_DIR, "account_vs_volume_panel.csv")

# Canonical project definition (matches desc_1/desc_2/panel_3): a1+a2+a4+a5.
_TOTAL_DEP_PARTS = ["dep_a1", "dep_a2", "dep_a4", "dep_a5"]

OUT_COLUMNS = [
    "firm_key", "cong_prud", "cnpj_root", "NomeInstituicao", "segment",
    "year", "quarter", "period_label",
    "panel_total_deposits", "panel_dep_a5", "volume_share", "dep_a5_share",
    "disclosed_customers_brazil", "disclosed_customers_broad",
    "disclosed_deposits_brl", "disclosed_deposits_usd",
    "deposits_per_customer_brl", "customer_scope_flag",
    "deposits_annual_broadcast",
]


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def _cnpj_root(x) -> str:
    """First-8-digit, zero-padded CNPJ root from any CNPJ-ish value."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    if isinstance(x, float):
        x = f"{int(x)}"
    digits = re.sub(r"\D", "", str(x))
    if not digits:
        return ""
    return digits[:8].zfill(8)


def write_csv(df: pd.DataFrame, out_csv: str) -> None:
    """CSV via pyarrow, with a pandas fallback (project I/O convention)."""
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    try:
        import pyarrow as pa
        import pyarrow.csv as pa_csv
        pa_csv.write_csv(pa.Table.from_pandas(df, preserve_index=False), out_csv)
    except Exception as e:  # noqa: BLE001 - fall back to pandas writer
        log.info(f"pyarrow CSV write unavailable ({type(e).__name__}); using pandas writer.")
        df.to_csv(out_csv, index=False)


# ----------------------------------------------------------------------------
# A. Panel aggregation -> conglomerate x year x quarter (national)
# ----------------------------------------------------------------------------
def build_panel_aggregate() -> pd.DataFrame:
    log.info(f"Reading market panel: {PANEL_CSV}")
    head = pd.read_csv(PANEL_CSV, nrows=0)
    cols = set(head.columns)

    keep = ["CodConglomeradoPrudencial", "CNPJ_Lider", "NomeInstituicao",
            "year", "quarter"]
    keep = [c for c in keep if c in cols]
    dep_cols = [c for c in (_TOTAL_DEP_PARTS + ["dep_a5"]) if c in cols]
    if "total_deposits" in cols:
        dep_cols.append("total_deposits")
    usecols = keep + sorted(set(dep_cols))
    df = pd.read_csv(PANEL_CSV, usecols=usecols)

    # Derive total_deposits if the panel file does not carry it (this copy of the
    # panel does not). Match the canonical a1+a2+a4+a5 definition with min_count=1.
    if "total_deposits" not in df.columns:
        parts = [c for c in _TOTAL_DEP_PARTS if c in df.columns]
        log.info(f"total_deposits absent; deriving from {parts} (min_count=1).")
        df["total_deposits"] = df[parts].sum(axis=1, min_count=1)
    if "dep_a5" not in df.columns:
        df["dep_a5"] = np.nan

    df["CodConglomeradoPrudencial"] = df["CodConglomeradoPrudencial"].astype(str)
    df["cnpj_root_lider"] = df["CNPJ_Lider"].apply(_cnpj_root)

    # Aggregate over mca_code (and any municipality dimension): sum deposits,
    # carry the most frequent firm name, the lead CNPJ root.
    grp = ["CodConglomeradoPrudencial", "year", "quarter"]
    agg = (df.groupby(grp, dropna=False)
             .agg(panel_total_deposits=("total_deposits", "sum"),
                  panel_dep_a5=("dep_a5", "sum"),
                  NomeInstituicao=("NomeInstituicao",
                                   lambda s: s.dropna().iloc[0] if s.notna().any() else np.nan),
                  cnpj_root_lider=("cnpj_root_lider",
                                   lambda s: s.replace("", np.nan).dropna().iloc[0]
                                   if s.replace("", np.nan).notna().any() else ""))
             .reset_index())

    # National shares within each (year, quarter).
    tot_by_q = agg.groupby(["year", "quarter"])["panel_total_deposits"].transform("sum")
    a5_by_q = agg.groupby(["year", "quarter"])["panel_dep_a5"].transform("sum")
    agg["volume_share"] = np.where(tot_by_q > 0, agg["panel_total_deposits"] / tot_by_q, np.nan)
    agg["dep_a5_share"] = np.where(a5_by_q > 0, agg["panel_dep_a5"] / a5_by_q, np.nan)

    log.info(f"Panel aggregate: {len(agg):,} (cong x year x quarter) rows, "
             f"{agg['CodConglomeradoPrudencial'].nunique():,} conglomerates, "
             f"years {int(agg.year.min())}-{int(agg.year.max())}.")
    return agg


# ----------------------------------------------------------------------------
# Firm registry -> firm_key -> conglomerate mapping
# ----------------------------------------------------------------------------
def build_firm_map(panel_agg: pd.DataFrame) -> pd.DataFrame:
    """One row per registry firm with the best-resolved CodConglomeradoPrudencial.

    Resolution order:
      1. registry cong_prud (set by IF-Data name match), if present in the panel;
      2. else match registry cnpj_root to the panel lead-CNPJ root (first 8 digits).
    Firms that resolve to neither keep cong_prud='' (disclosures still emitted
    with panel fields NaN).
    """
    reg = load_registry(BASE)
    panel_congs = set(panel_agg["CodConglomeradoPrudencial"].unique())
    # lead-CNPJ-root -> conglomerate (pick the cong with the largest deposits to
    # avoid tiny shell-entity collisions when a root maps to several congs).
    root2cong: dict[str, str] = {}
    tmp = panel_agg.dropna(subset=["cnpj_root_lider"]).copy()
    tmp = tmp[tmp["cnpj_root_lider"] != ""]
    if not tmp.empty:
        sized = (tmp.groupby(["cnpj_root_lider", "CodConglomeradoPrudencial"])
                    ["panel_total_deposits"].sum().reset_index()
                    .sort_values("panel_total_deposits", ascending=False))
        for _, r in sized.iterrows():
            root2cong.setdefault(r["cnpj_root_lider"], r["CodConglomeradoPrudencial"])

    rows = []
    for f in reg:
        cong = str(f.get("cong_prud", "") or "")
        source = ""
        if cong and cong in panel_congs:
            source = "registry_cong"
        else:
            cong = ""
        if not cong:
            root = str(f.get("cnpj_root", "")).zfill(8)
            if root in root2cong:
                cong = root2cong[root]
                source = "cnpj_root_match"
        rows.append(dict(firm_key=f["firm_key"],
                         segment=f.get("segment", ""),
                         cnpj_root=str(f.get("cnpj_root", "")).zfill(8),
                         country_scope=f.get("country_scope", ""),
                         cong_prud=cong,
                         cong_source=source))
    fm = pd.DataFrame(rows)
    n_res = (fm["cong_prud"] != "").sum()
    log.info(f"Firm map: {n_res}/{len(fm)} registry firms resolved to a conglomerate.")
    for _, r in fm.iterrows():
        log.info(f"  {r['firm_key']:13s} {r['segment']:9s} "
                 f"cong={r['cong_prud'] or '(none)':10s} via {r['cong_source'] or '-'}")
    return fm


# ----------------------------------------------------------------------------
# B. Disclosure aggregation -> firm_key x year x quarter
# ----------------------------------------------------------------------------
def _read_disclosure(path: str, label: str) -> pd.DataFrame:
    if not os.path.exists(path):
        log.warning(f"{label} disclosures missing: {path}")
        return pd.DataFrame()
    d = pd.read_csv(path)
    d["__src"] = label
    log.info(f"Read {label} disclosures: {len(d)} rows ({path}).")
    return d


def build_disclosure_aggregate() -> pd.DataFrame:
    parts = [
        _read_disclosure(EDGAR_CSV, "sec"),
        _read_disclosure(CVM_CSV, "cvm"),
        _read_disclosure(PARENT_CSV, "parent"),
        _read_disclosure(INCUMBENT_CSV, "ipe"),   # CVM-IPE Tier-2/4 client counts
    ]
    parts = [p for p in parts if not p.empty]
    if not parts:
        raise SystemExit("No disclosure files found — cannot build the join.")
    d = pd.concat(parts, ignore_index=True)

    for c in ("metric", "geo_scope", "currency", "unit"):
        if c in d.columns:
            d[c] = d[c].astype(str).str.strip().str.lower().replace({"nan": ""})
    d["period_year"] = pd.to_numeric(d["period_year"], errors="coerce").astype("Int64")
    d["period_quarter"] = pd.to_numeric(d["period_quarter"], errors="coerce").astype("Int64")
    d["value"] = pd.to_numeric(d["value"], errors="coerce")

    # --- expand annual deposit obs (no quarter) to all four quarters ---------
    is_dep = d["metric"].eq("deposits")
    annual_dep = is_dep & d["period_quarter"].isna()
    if annual_dep.any():
        log.info(f"Broadcasting {int(annual_dep.sum())} annual deposit obs to Q1-Q4.")
    expanded = []
    for _, r in d.iterrows():
        if r["metric"] == "deposits" and pd.isna(r["period_quarter"]):
            for q in (1, 2, 3, 4):
                rr = r.copy()
                rr["period_quarter"] = q
                rr["__annual_broadcast"] = True
                expanded.append(rr)
        else:
            rr = r.copy()
            rr["__annual_broadcast"] = False
            expanded.append(rr)
    d = pd.DataFrame(expanded).reset_index(drop=True)
    d = d.dropna(subset=["period_year", "period_quarter"])
    d["period_year"] = d["period_year"].astype(int)
    d["period_quarter"] = d["period_quarter"].astype(int)

    # --- per firm x (year, quarter) reductions -------------------------------
    out_rows = []
    grp = ["firm_key", "period_year", "period_quarter"]
    for (fk, yy, qq), g in d.groupby(grp, dropna=False):
        rec = dict(firm_key=fk, year=int(yy), quarter=int(qq),
                   period_label=f"{int(qq)}Q{int(yy)}")

        cust = g[g["metric"].isin(["customers_total", "customers_active"])]

        # Brazil customers: prefer customers_total over active; take max value.
        cust_br = cust[cust["geo_scope"].eq("brazil")]
        rec["disclosed_customers_brazil"] = _pick_customer(cust_br)

        # Broad customers (consolidated / latam): for NU / MELI headline base.
        cust_broad = cust[cust["geo_scope"].isin(["consolidated", "latam"])]
        rec["disclosed_customers_broad"] = _pick_customer(cust_broad)

        # Deposits: BRL (CVM, or EDGAR currency=BRL) vs USD (EDGAR currency=USD).
        dep = g[g["metric"].eq("deposits")]
        dep_brl = dep[dep["currency"].eq("brl")]
        dep_usd = dep[dep["currency"].eq("usd")]
        rec["disclosed_deposits_brl"] = dep_brl["value"].max() if not dep_brl.empty else np.nan
        rec["disclosed_deposits_usd"] = dep_usd["value"].max() if not dep_usd.empty else np.nan
        rec["deposits_annual_broadcast"] = bool(dep["__annual_broadcast"].any())

        # customer_scope_flag: which geography the customer count we kept covers.
        if pd.notna(rec["disclosed_customers_brazil"]):
            rec["customer_scope_flag"] = "brazil"
        elif pd.notna(rec["disclosed_customers_broad"]):
            rec["customer_scope_flag"] = "consolidated"
        else:
            rec["customer_scope_flag"] = "none"

        out_rows.append(rec)

    da = pd.DataFrame(out_rows)
    log.info(f"Disclosure aggregate: {len(da)} firm x year x quarter rows "
             f"({da['firm_key'].nunique()} firms).")
    return da


def _pick_customer(sub: pd.DataFrame):
    """Prefer customers_total over customers_active; return the max value."""
    if sub.empty:
        return np.nan
    tot = sub[sub["metric"].eq("customers_total")]
    use = tot if not tot.empty else sub
    v = use["value"].max()
    return v if pd.notna(v) else np.nan


# ----------------------------------------------------------------------------
# D. Merge A + B
# ----------------------------------------------------------------------------
def merge_all(panel_agg: pd.DataFrame, firm_map: pd.DataFrame,
              disc_agg: pd.DataFrame) -> pd.DataFrame:
    # disclosure firm -> (cong, segment, cnpj_root)
    disc = disc_agg.merge(
        firm_map[["firm_key", "cong_prud", "segment", "cnpj_root"]],
        on="firm_key", how="left")

    # Panel side keyed by cong x year x quarter.
    panel_keyed = panel_agg.rename(
        columns={"CodConglomeradoPrudencial": "cong_prud"}).copy()
    panel_keyed["__has_panel"] = True

    # Merge disclosures onto panel by (cong_prud, year, quarter). Firms with no
    # cong_prud ('') will not match and keep panel fields NaN (kept by outer-left).
    merged = disc.merge(
        panel_keyed[["cong_prud", "year", "quarter", "panel_total_deposits",
                     "panel_dep_a5", "volume_share", "dep_a5_share",
                     "NomeInstituicao", "__has_panel"]],
        on=["cong_prud", "year", "quarter"], how="left")

    # deposits_per_customer_brl: disclosed BRL deposits, else disclosed USD->BRL
    # (latam firms like Nubank/MercadoPago report deposits in USD), else panel
    # (R$ thousands -> R$). Customer count: Brazil preferred, else broad scope so
    # the deposit and customer scopes match.
    FX_USD_BRL = 5.0
    dep_brl_for_ratio = merged["disclosed_deposits_brl"]
    dep_brl_for_ratio = dep_brl_for_ratio.where(
        dep_brl_for_ratio.notna(), merged["disclosed_deposits_usd"] * FX_USD_BRL)
    dep_brl_for_ratio = dep_brl_for_ratio.where(
        dep_brl_for_ratio.notna(), merged["panel_total_deposits"] * 1000.0)
    cust = merged["disclosed_customers_brazil"].where(
        merged["disclosed_customers_brazil"].notna(),
        merged.get("disclosed_customers_broad"))
    dpc = np.where(cust.notna() & (cust > 0), dep_brl_for_ratio / cust, np.nan)
    # Clamp to a plausible per-customer balance (R$50–200k). Values outside flag a
    # unit mismatch or a mis-parsed customer count (text extraction is noisy) and
    # are dropped rather than shown.
    merged["deposits_per_customer_brl"] = np.where((dpc >= 50) & (dpc <= 2e5), dpc, np.nan)

    # Finalise columns.
    for c in OUT_COLUMNS:
        if c not in merged.columns:
            merged[c] = np.nan
    out = merged[OUT_COLUMNS].copy()
    out = out.sort_values(["year", "quarter", "volume_share"],
                          ascending=[True, True, False],
                          na_position="last").reset_index(drop=True)
    return out


# ----------------------------------------------------------------------------
# Summary: the core extensive-vs-intensive contrast
# ----------------------------------------------------------------------------
def print_contrast(out: pd.DataFrame) -> None:
    has_cust = out["disclosed_customers_brazil"].notna() | out["disclosed_customers_broad"].notna()
    has_vol = out["volume_share"].notna()
    elig = out[has_cust & has_vol].copy()

    chosen = None
    if not elig.empty:
        # Most recent quarter with >=2 digital and >=2 incumbent firms.
        elig["yq"] = elig["year"] * 10 + elig["quarter"]
        for yq in sorted(elig["yq"].unique(), reverse=True):
            sub = elig[elig["yq"] == yq]
            n_dig = sub[sub["segment"] == "digital"]["firm_key"].nunique()
            n_inc = sub[sub["segment"] == "incumbent"]["firm_key"].nunique()
            if n_dig >= 2 and n_inc >= 2:
                chosen = sub
                break

    print("\n" + "=" * 78)
    print("CORE CONTRAST — extensive (accounts) vs intensive (volume) deposit margin")
    print("=" * 78)
    if chosen is not None:
        yy, qq = int(chosen.iloc[0]["year"]), int(chosen.iloc[0]["quarter"])
        print(f"Quarter with >=2 digital & >=2 incumbent firms having BOTH a customer "
              f"count and a volume_share: {qq}Q{yy}\n")
        _print_table(chosen)
    else:
        print("No single quarter has BOTH a customer count and a volume_share for "
              ">=2 digital AND >=2 incumbent firms.")
        print("Reason: disclosed customer counts are 2025-2026 (EDGAR/parent), while the")
        print("market panel ends 2024 — so customer obs rarely share a quarter with a")
        print("panel volume_share. Showing the best-available contrast instead:\n")
        if not elig.empty:
            elig["yq"] = elig["year"] * 10 + elig["quarter"]
            best_yq = elig["yq"].max()
            _print_table(elig[elig["yq"] == best_yq])
        else:
            # Fall back: most recent quarter that has a volume_share for the firms,
            # alongside whatever customer figure exists for them (any quarter).
            print("(no firm-quarter has both simultaneously; see CSV for the full join)")
            _print_fallback(out)


def _print_table(sub: pd.DataFrame) -> None:
    cols = ["firm_key", "segment", "volume_share", "disclosed_customers_brazil",
            "disclosed_customers_broad", "deposits_per_customer_brl"]
    view = sub[cols].copy().sort_values(
        ["segment", "volume_share"], ascending=[True, False])
    with pd.option_context("display.width", 160,
                           "display.float_format", lambda v: f"{v:,.4g}"):
        print(view.to_string(index=False))


def _print_fallback(out: pd.DataFrame) -> None:
    """Per resolved firm: its latest panel-quarter volume_share, its most recent
    disclosed customer count (Brazil if any, else broad), its most recent
    disclosed deposit (BRL preferred, else USD), and an ILLUSTRATIVE
    deposits-per-customer using those nearest figures.

    NOTE: this nearest-quarter ratio is for the printed contrast only; the CSV
    column deposits_per_customer_brl keeps strict same-firm-quarter semantics.
    """
    vol = out[out["volume_share"].notna()].copy()
    if vol.empty:
        print("(no resolved firm has a panel volume_share)")
        return
    vol["yq"] = vol["year"] * 10 + vol["quarter"]
    latest = (vol.sort_values("yq").groupby("firm_key").tail(1)
                 [["firm_key", "segment", "year", "quarter", "volume_share"]]
                 .rename(columns={"year": "vol_year", "quarter": "vol_quarter"}))

    # most recent disclosed customer count (prefer Brazil scope, else broad)
    def _latest_metric(df_in: pd.DataFrame, col: str) -> pd.DataFrame:
        d = out[out[col].notna()].copy()
        if d.empty:
            return pd.DataFrame(columns=["firm_key", col])
        d["yq"] = d["year"] * 10 + d["quarter"]
        return d.sort_values("yq").groupby("firm_key").tail(1)[["firm_key", col]]

    cust_br = _latest_metric(out, "disclosed_customers_brazil")
    cust_bd = _latest_metric(out, "disclosed_customers_broad")

    # most recent disclosed deposit: BRL if available else USD
    dep = out[out[["disclosed_deposits_brl", "disclosed_deposits_usd"]].notna().any(axis=1)].copy()
    dep["yq"] = dep["year"] * 10 + dep["quarter"]
    dep = dep.sort_values("yq").groupby("firm_key").tail(1)[
        ["firm_key", "disclosed_deposits_brl", "disclosed_deposits_usd"]]

    tab = (latest.merge(cust_br, on="firm_key", how="left")
                 .merge(cust_bd, on="firm_key", how="left")
                 .merge(dep, on="firm_key", how="left"))
    tab["cust_used"] = tab["disclosed_customers_brazil"].where(
        tab["disclosed_customers_brazil"].notna(), tab["disclosed_customers_broad"])
    # illustrative R$ deposit (BRL disclosed, else USD@~5.0, else panel*1000 unavailable here)
    dep_brl = tab["disclosed_deposits_brl"].where(
        tab["disclosed_deposits_brl"].notna(), tab["disclosed_deposits_usd"] * 5.0)
    tab["dep_per_cust_brl_illustr"] = np.where(
        tab["cust_used"].notna() & (tab["cust_used"] > 0), dep_brl / tab["cust_used"], np.nan)

    view = tab[["firm_key", "segment", "volume_share", "cust_used",
                "dep_per_cust_brl_illustr"]].sort_values(
        ["segment", "volume_share"], ascending=[True, False])
    with pd.option_context("display.width", 160,
                           "display.float_format", lambda v: f"{v:,.4g}"):
        print("(volume_share at firm's latest panel quarter; cust_used = latest disclosed")
        print(" customer count [Brazil pref., else broad]; dep_per_cust_brl_illustr uses")
        print(" the firm's latest disclosed deposit, USD->BRL at ~5.0 — ILLUSTRATIVE only)")
        print(view.to_string(index=False))


# ----------------------------------------------------------------------------
def main() -> None:
    panel_agg = build_panel_aggregate()
    firm_map = build_firm_map(panel_agg)
    disc_agg = build_disclosure_aggregate()
    out = merge_all(panel_agg, firm_map, disc_agg)

    write_csv(out, OUT_CSV)
    log.info(f"Wrote {len(out)} rows x {len(out.columns)} cols -> {OUT_CSV}")

    n_panel = out["volume_share"].notna().sum()
    n_cust = (out["disclosed_customers_brazil"].notna()
              | out["disclosed_customers_broad"].notna()).sum()
    n_depc = out["deposits_per_customer_brl"].notna().sum()
    log.info(f"Coverage: {n_panel} rows w/ volume_share, {n_cust} w/ a customer "
             f"count, {n_depc} w/ deposits_per_customer_brl.")

    print_contrast(out)


if __name__ == "__main__":
    main()
