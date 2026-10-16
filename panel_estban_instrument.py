"""
panel_estban_instrument.py
==============================
Append the ESTBAN branch-competition instrument to market_panel.csv.

MOTIVATION.  The BLP leave-one-out instruments (loo_*/mean_loo_*, n_rivals, cost/capital ratios) are
all functions of CONGLOMERATE-level accounting characteristics, which carry no within-conglomerate
variation and are severely collinear (max VIF ~1,350) → the deposit-spread first stage is weak
(cluster-robust effective-F ~4) under conglomerate clustering. ESTBAN (Estatística Bancária Mensal
por município) reports the number of branches (AGEN_PROCESSADAS) per (institution CNPJ root ×
municipality × month), i.e. genuine WITHIN-conglomerate variation across the municipalities a
conglomerate operates in. The leave-one-out RIVAL branch count in the local market (MCA) is a
local-competition shifter of the deposit markdown that is plausibly excluded from own deposit demand
(BLP rival-characteristic logic) and, unlike the accounting instruments, is independent (own VIF ~5)
and relevant (own cluster-robust first-stage F ~10 for demand deposits). See scratchpad diagnostics.

CONSTRUCTION.  For each firm i in local market cell (mca, quarter t):
    estban_rival_branches_lag = log1p( Σ_{j≠i} branches_j at t-1 )   (rivals in the same MCA)
Branches are summed to (conglomerate, mca, quarter); CNPJ→conglomerate via the IF-Data List
crosswalk (cosif_process_2_calibrate helpers); CODMUN_IBGE→mca via the IBGE MCA crosswalk. The
instrument is LAGGED one CALENDAR quarter (predetermined branch network → strengthens the exclusion
argument): the value on (i, mca, t) is the rival count ESTBAN reports for (i, mca, t-1) exactly, so
a gap in the source never pulls in an earlier quarter. log1p because branch counts are right-skewed.

B-ONLY INSTRUMENT.  This is a branch-network instrument, defined for brick-and-mortar firms
(is_B == 1). Digital firms (is_B == 0) have no branch network, so their rows carry 0 by
construction, not because data are missing.

MISSING B ROWS.  A B row whose (conglomerate, mca, t-1) has no ESTBAN record has no measured rival
count: it is left NaN and flagged estban_rival_branches_missing = 1 (0 on every other row, D rows
included). A NaN here is missing data, never "no rival branches".

PIPELINE PLACEMENT.  Run AFTER panel_loo_instruments.py (which overwrites market_panel.csv with the LOO
instruments) and BEFORE fees/panel_fee_merge.py --patch-market (which passes all columns through to
market_panel_with_fees.csv). Adds TWO columns (the instrument and its missing flag), keyed on
(CodConglomeradoPrudencial, mca_code, year, quarter); the merge is 1:1 with the wide market panel.
Idempotent: re-running overwrites both columns.

Usage:  python panel_estban_instrument.py
"""
from __future__ import annotations
import logging
import numpy as np
import pandas as pd

from utils import paths
from utils import conglomerates as cc
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

PANEL_CSV = paths.PROCESSED / "market_panel.csv"
ESTBAN_CSV = paths.ESTBAN_CSV
MCA_XWALK = paths.IBGE_DIR / "muni_mca_regions_2010_2024_panel.csv"
INSTRUMENT = "estban_rival_branches_lag"
MISSING_FLAG = "estban_rival_branches_missing"   # 1 = B row with no ESTBAN record at t-1
KEYS = ["CodConglomeradoPrudencial", "mca_code", "year", "quarter"]
# Own branch counts per (conglomerate, MCA, quarter). Computed here anyway as the input to the
# leave-one-out rival count; persisted as a sidecar because it is the only branch-network
# series at market granularity in the project. sleep_ident_entry_dynamics.py uses it to date entry
# by BRANCHES rather than by deposits, which screens out markets where deposits are booked
# before any branch exists (the analogue of Egan et al.'s Summary-of-Deposits caveat).
# Sidecar only -- own_br is not added to market_panel.csv.
OWN_BRANCH_SIDECAR = paths.PROCESSED / "PANEL_INTERMED" / "estban_own_branches.csv"


def build_estban_rival_branches() -> pd.DataFrame:
    """Return a DataFrame keyed on KEYS with the single column ``estban_rival_branches_lag``."""
    logging.info(f"Reading ESTBAN {ESTBAN_CSV}")
    est = pd.read_csv(ESTBAN_CSV, usecols=["CNPJ", "CODMUN_IBGE", "YEAR", "MONTH", "AGEN_PROCESSADAS"],
                      dtype={"CNPJ": str}, low_memory=False, encoding="latin-1")
    est = est[est["MONTH"].isin([3, 6, 9, 12])].copy()                       # quarter-end months
    est["CNPJ"] = est["CNPJ"].str.strip().str.zfill(8)
    est["CODMUN_IBGE"] = pd.to_numeric(est["CODMUN_IBGE"], errors="coerce")
    est = est.dropna(subset=["CODMUN_IBGE"])
    est["CODMUN_IBGE"] = est["CODMUN_IBGE"].astype(int)
    est["AGEN_PROCESSADAS"] = pd.to_numeric(est["AGEN_PROCESSADAS"], errors="coerce").fillna(0.0)
    est["AnoMes"] = (est["YEAR"].astype(int) * 100 + est["MONTH"].astype(int)).astype("int64")
    est["quarter"] = (est["MONTH"] // 3).astype(int)

    # CNPJ root -> prudential conglomerate (time-varying as-of merge); map unique pairs, then merge back.
    logging.info("Mapping CNPJ -> conglomerate (IF-Data List as-of merge)")
    uniq = est[["CNPJ", "AnoMes"]].drop_duplicates()
    cmap = cc.map_foundation_to_congl(uniq, cc.build_cnpj_to_congl())[
        ["CNPJ", "AnoMes", "CodConglomeradoPrudencial"]]
    est = est.merge(cmap, on=["CNPJ", "AnoMes"], how="inner")

    # CODMUN_IBGE -> mca_code (crosswalk deduped to latest year per municipality; MCA is stable).
    logging.info("Mapping CODMUN_IBGE -> mca_code (IBGE crosswalk)")
    xw = pd.read_csv(MCA_XWALK, usecols=["municipality_code", "year", "mca_code"])
    xw = xw.sort_values("year").drop_duplicates("municipality_code", keep="last")
    xw["mca_code"] = xw["mca_code"].astype(str)
    est = est.merge(xw[["municipality_code", "mca_code"]],
                    left_on="CODMUN_IBGE", right_on="municipality_code", how="inner")

    # Branches per (conglomerate, mca, year, quarter); leave-one-out rival count in the MCA.
    C, M, Y, Q = "CodConglomeradoPrudencial", "mca_code", "YEAR", "quarter"
    own = est.groupby([C, M, Y, Q], as_index=False)["AGEN_PROCESSADAS"].sum().rename(
        columns={"AGEN_PROCESSADAS": "own_br", "YEAR": "year"})
    mkt = own.groupby([M, "year", Q])["own_br"].transform("sum")
    own["rival_br"] = (mkt - own["own_br"]).clip(lower=0.0)

    # Persist own_br (see the OWN_BRANCH_SIDECAR note at the top). Never fatal: a locked or
    # unwritable sidecar must not take down the instrument build, which is what the panel
    # actually depends on.
    try:
        OWN_BRANCH_SIDECAR.parent.mkdir(parents=True, exist_ok=True)
        sc = own[[C, M, "year", Q, "own_br"]].copy()
        sc.columns = KEYS + ["own_br"]
        sc.to_csv(OWN_BRANCH_SIDECAR, index=False)
        logging.info(f"Wrote branch sidecar {OWN_BRANCH_SIDECAR.name}: {len(sc):,} rows")
    except Exception as e:                                    # noqa: BLE001
        logging.warning(f"branch sidecar not written ({e}); step_entry_dynamics will "
                        "fall back to --no-branch-screen")

    # Calendar lag: every (conglomerate, mca, quarter s) record is re-keyed to the next calendar
    # quarter s+1, so the value on key t is the rival count observed at t-1 exactly. A key t
    # exists only if ESTBAN reports (conglomerate, mca) at t-1; the panel merge leaves the rest
    # unmatched. log1p.
    q_next = own["year"].astype(int) * 4 + (own[Q].astype(int) - 1) + 1
    out = pd.DataFrame({C: own[C].astype(str).to_numpy(), M: own[M].to_numpy(),
                        "year": (q_next // 4).to_numpy(), Q: (q_next % 4 + 1).to_numpy(),
                        INSTRUMENT: np.log1p(own["rival_br"].to_numpy(dtype=float))})
    out.columns = KEYS + [INSTRUMENT]
    logging.info(f"Built {INSTRUMENT}: {len(out):,} conglomerate-mca-quarters, "
                 f"years {int(out['year'].min())}-{int(out['year'].max())} (t-1 source "
                 f"{int(own['year'].min())}-{int(own['year'].max())})")
    return out


def attach_instrument(panel: pd.DataFrame, inst: pd.DataFrame) -> pd.DataFrame:
    """Merge the instrument onto the panel; set D rows to 0, leave unmatched B rows NaN + flagged."""
    panel = panel.drop(columns=[c for c in (INSTRUMENT, MISSING_FLAG) if c in panel.columns])
    merged = panel.merge(inst, on=KEYS, how="left", validate="many_to_one")
    is_b = pd.to_numeric(merged["is_B"], errors="coerce")
    b_rows, d_rows = (is_b == 1).to_numpy(), (is_b == 0).to_numpy()
    n_match_b = int((b_rows & merged[INSTRUMENT].notna().to_numpy()).sum())
    # B-only instrument: D firms have no branch network, so 0 on D rows is by construction
    # (whether or not ESTBAN lists a branch for the D conglomerate in that MCA).
    merged.loc[d_rows, INSTRUMENT] = 0.0
    # A B row with no ESTBAN record at t-1 has no measured rival count: NaN, flagged 1.
    miss = b_rows & merged[INSTRUMENT].isna().to_numpy()
    merged[MISSING_FLAG] = miss.astype("int8")
    n_other_nan = int((~b_rows & ~d_rows & merged[INSTRUMENT].isna().to_numpy()).sum())
    logging.info(f"Merged {INSTRUMENT} onto {len(merged):,} panel rows: B {int(b_rows.sum()):,} "
                 f"({n_match_b:,} matched ESTBAN at t-1, {int(miss.sum()):,} NaN + "
                 f"{MISSING_FLAG}=1); D {int(d_rows.sum()):,} set to 0 by construction")
    if n_other_nan:
        logging.warning(f"{n_other_nan:,} rows with is_B neither 0 nor 1 hold a NaN {INSTRUMENT}")
    return merged


def main() -> None:
    if not PANEL_CSV.exists():
        raise FileNotFoundError(f"market_panel.csv not found at {PANEL_CSV} — run panel_7 first.")
    inst = build_estban_rival_branches()

    logging.info(f"Loading {PANEL_CSV}")
    panel = pd.read_csv(PANEL_CSV, low_memory=False, dtype={"mca_code": str})
    panel["CodConglomeradoPrudencial"] = panel["CodConglomeradoPrudencial"].astype(str)
    panel["mca_code"] = panel["mca_code"].astype(str)
    inst["CodConglomeradoPrudencial"] = inst["CodConglomeradoPrudencial"].astype(str)
    inst["mca_code"] = inst["mca_code"].astype(str)

    merged = attach_instrument(panel, inst)                # drops both columns first: idempotent

    merged.to_csv(PANEL_CSV, index=False)
    # Keep the Parquet sidecar in step with the CSV. estimation_1_sleep reads the panel
    # through utils.load_panel_cached, which serves that sidecar, so leaving it stale
    # here would hand the sleepiness estimation the previous build.
    from utils import refresh_panel_cache
    refresh_panel_cache(PANEL_CSV, merged)
    logging.info(f"Wrote {PANEL_CSV} with {INSTRUMENT}")


if __name__ == "__main__":
    main()
