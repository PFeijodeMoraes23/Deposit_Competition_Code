## panel_cosif_calibrate.py
# Author: Pedro Feijo de Moraes
# Objective: Build the CORRECTED k=4 (CDB / time-deposit) implicit rate per
#            (CodConglomeradoPrudencial x quarter), fixing the artifact in the
#            old blended/residual approach.
#
# Method (per-bank effective rates + precision-weighted shrinkage to segment):
#   1. From each conglomerate's 2025+ 10-digit COSIF leaves, estimate effective
#      repo / bills rates (as x CDI), pool them to a PRUDENTIAL SEGMENT (S1-S5)
#      profile (deposit-weighted over banks in the segment with 2025 data) and
#      shrink each bank toward it with w_i = n_i / (n_i + N0), N0 = 2, where n_i is
#      the number of the bank's 2025+ quarters with a CDB leaf, a lagged time stock
#      and a CDI.  A bank with no own ratio takes the segment's.  Only banks with both
#      own ratios get an own k_i (step 2).
#   2. The strip: the lump 8.1.1 minus the imputed repo, bills, savings, interbank and
#      judicial-deposit costs (strip_components).  Judicial deposits (4.1.5.50) sit inside
#      the 4.1.5 stock and their cost (8114000008) inside the lump, but neither is a CDB:
#      their cost is the 2025+ leaf, or rho_i x savings yield x lagged judicial stock before.
#      Where the bank's statements give a component's annual or semester total
#      (DATA_RAW/issuers/funding_expense), the component is rescaled to it.
#   3. The per-bank de-bias factor
#        k_i = (true CDB rate from leaf 8113000009) / (strip-implied CDB rate),
#      measured on that strip over the bank's 2025+ quarters.  k_i is pooled and shrunk
#      like the ratios.  Banks with no 2025 data -> the segment profile; segment also
#      missing -> the global one.
#   4. Apply, over the ORDINARY time-deposit stock (4.1.5 minus judicial deposits):
#        pre-2025: rate_a4 = k_shrunk * strip / lag(ordinary stock)
#        2025+   : rate_a4 = |8113000009| / lag(ordinary stock)        (direct leaf)
#      A failed rate (strip or leaf <= 0, above RATE_Q_CAP) stays missing with rate_flag.
#
# Inputs  : custos_implicitos_v2_foundation.csv      (panel_cosif_extract.py, step 12; BANCOS
#             members, and the SCFI members used only where a conglomerate has no bank)
#           SGS_RAW/macro_series_full.csv and SGS_RAW/savings_yields/  (quarterly macro and
#             savings rates computed here through utils.macro_rates, as fractions; step 16
#             writes the same rates in percent)
#           IF-Data List  (CNPJ<->conglomerate merge_asof crosswalk, utils.conglomerates,
#             keyed as deposits_panel.csv files each CNPJ; and the segment Sr per conglomerate)
#           Rout/judicial_deposits_share.csv  (utils.judicial_deposits)
#           DATA_RAW/issuers/funding_expense/compiled/dfp_funding_expense*.csv  (statements)
# Output  : COSIF_PROCESSED/cosif_cdb_rate_corrected.csv
#             columns: CodConglomeradoPrudencial, AnoMes, rate_a4_corrected_qoq,
#                      rate_a4_corrected_ann, tier, rate_flag (0 measured, 2 strip or leaf
#                      <= 0, 3 above RATE_Q_CAP), rate_a4_vintage_src (statement or none)
#
# NOTE: rate_a4_corrected_qoq is a fraction (e.g. 0.025 = 2.5% per quarter), the scale
# panel_deposit_rates.py uses internally.
## ---------------------------------------------------------------------------

import os
import sys
import glob
import logging

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
from utils import paths  # noqa: E402
from utils.conglomerates import (build_cnpj_to_congl, map_foundation_to_congl,  # noqa: E402
                                 canonicalize_codes, deposit_panel_codes)
from utils.macro_rates import (load_macro_series, make_quarterly_macro,  # noqa: E402
                               savings_cost_qoq, OLD_RULE_SHARE)
from utils import judicial_deposits, raw_paths  # noqa: E402

PROCESSED = str(paths.PROCESSED)
COSIF_PROC = os.path.join(PROCESSED, "COSIF_PROCESSED")

FOUNDATION = os.path.join(COSIF_PROC, "custos_implicitos_v2_foundation.csv")
IF_LIST_DIR = str(paths.IF_DATA_LIST)
SEGMENTS = ["S1", "S2", "S3", "S4", "S5"]
OUT_CORRECTED = os.path.join(COSIF_PROC, "cosif_cdb_rate_corrected.csv")
DFP_FUNDING_DIR = str(raw_paths.ISSUERS_FUNDING_EXPENSE / "compiled")

N0 = 2                  # precision-shrinkage prior count
RATE_Q_CAP = 0.5        # economically implausible quarterly-rate guard

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("panel_cosif_calibrate")

STOCK_COLS = ["stk_total", "stk_demand", "stk_savings", "stk_interb",
              "stk_time", "stk_judicial", "stk_repos", "stk_bills", "stk_sub", "stk_prepaid"]
EXPENSE_COLS = ["exp_lump", "exp_savings", "exp_interb", "exp_cdb", "exp_judicial",
                "exp_repos", "exp_bills", "exp_sub", "exp_prepaid"]


def annualize(q):
    return (1.0 + q) ** 4 - 1.0


def wmean(s, w):
    s = np.asarray(s, dtype=float)
    w = np.asarray(w, dtype=float)
    m = np.isfinite(s) & np.isfinite(w) & (w > 0)
    return float(np.average(s[m], weights=w[m])) if m.any() else np.nan


def load_macro_fraction():
    """Quarterly macro rates as fractions, computed from the staged SGS series. Step 16 writes
    the same rates, in percent, to PANEL_INTERMED/quarterly_macro_rates.csv; computing them
    here keeps this step from reading the output of a later step."""
    return make_quarterly_macro(load_macro_series())


def load_segment_map():
    """conglomerate -> prudential segment (S1-S5), from the IF-Data List files: for each
    canonical code, the modal Sr over the rows of its latest snapshot that carry a valid Sr
    (ties -> the lower segment number).  The List files are a raw input, so this step reads
    no later pipeline output.  On the 2026-10-15 panel this agrees with the market panel's
    latest segment for 385 of the 386 conglomerates both carry.  Raises if no List file has
    an Sr column: every bank would otherwise fall back to the global profile."""
    files = sorted(glob.glob(os.path.join(IF_LIST_DIR, "IF_DATA_List_*.csv")))
    frames = []
    for f in files:
        if os.path.getsize(f) <= 500:          # headers-only stub
            continue
        if "Sr" not in pd.read_csv(f, nrows=0, encoding="latin1").columns:
            continue
        frames.append(pd.read_csv(f, encoding="latin1", dtype=str,
                                  usecols=["Data", "CodConglomeradoPrudencial", "Sr"]))
    if not frames:
        raise RuntimeError(
            f"No IF-Data List file in {IF_LIST_DIR} carries an Sr column; every bank "
            "would fall back to the global profile.")
    lf = pd.concat(frames, ignore_index=True).dropna(subset=["CodConglomeradoPrudencial"])
    lf["Sr"] = lf["Sr"].astype(str).str.strip().str.upper()
    lf = lf[lf["Sr"].isin(SEGMENTS)].copy()
    lf["Data"] = pd.to_numeric(lf["Data"], errors="coerce")
    lf["CodConglomeradoPrudencial"] = canonicalize_codes(lf["CodConglomeradoPrudencial"])
    tail = lf[lf["Data"] == lf.groupby("CodConglomeradoPrudencial")["Data"].transform("max")]
    seg = (tail.groupby(["CodConglomeradoPrudencial", "Sr"]).size().rename("n").reset_index()
               .sort_values(["CodConglomeradoPrudencial", "n", "Sr"], ascending=[True, False, True])
               .drop_duplicates("CodConglomeradoPrudencial")
               .rename(columns={"Sr": "segment"})[["CodConglomeradoPrudencial", "segment"]])
    log.info("Segment map: %d conglomerates from %d IF-Data List files", len(seg), len(frames))
    return seg


def build_congl_panel(found, macro):
    df_map = build_cnpj_to_congl()
    # key each CNPJ by the code deposits_panel.csv files its deposits under
    df_map["CodConglomeradoPrudencial"] = deposit_panel_codes(df_map["CNPJ"], df_map["CodConglomeradoPrudencial"])
    merged = map_foundation_to_congl(found, df_map)
    # A conglomerate-quarter is measured on its BANCOS members; its credit-finance companies
    # (src SOCIEDADES) count only when it has no BANCOS member that quarter, i.e. for the
    # conglomerates whose deposit-taking entity is an SCFI (Nubank, Mercado Pago, Neon, ...).
    merged["scfi"] = False
    if "src" in merged.columns:
        cq = ["CodConglomeradoPrudencial", "AnoMes"]
        has_bank = (merged["src"] == "BANCOS").groupby([merged[c] for c in cq]).transform("any")
        merged["scfi"] = ~has_bank
        n_scfi = int(((merged["src"] != "BANCOS") & ~has_bank).sum())
        merged = merged[(merged["src"] == "BANCOS") | ~has_bank]
        log.info("SCFI rows used for conglomerate-quarters without a BANCOS member: %d", n_scfi)
    agg_cols = [c for c in STOCK_COLS + EXPENSE_COLS if c in merged.columns]
    # scfi: the conglomerate-quarter is measured on credit-finance companies, not banks
    congl = (merged.groupby(["CodConglomeradoPrudencial", "AnoMes"])
             .agg(**{c: (c, lambda s: s.sum(min_count=1)) for c in agg_cols}, scfi=("scfi", "any"))
             .reset_index())
    congl["year"] = congl["AnoMes"] // 100
    congl = congl.sort_values(["CodConglomeradoPrudencial", "AnoMes"])

    # Judicial deposits J (4.1.5.50, inside stk_time): Cosif 4155000009 from 2025, the statements'
    # series before (utils.judicial_deposits); a bank with neither carries none.  The ordinary
    # time-deposit stock stk_time_ord is the base of every rate below.
    jf = judicial_deposits.load().set_index(["CodConglomeradoPrudencial", "AnoMes"])["J"]
    key = pd.MultiIndex.from_arrays([congl["CodConglomeradoPrudencial"].astype(str), congl["AnoMes"]])
    j_file = pd.Series(jf.reindex(key).to_numpy(), index=congl.index)
    j_cosif = congl["stk_judicial"] if "stk_judicial" in congl.columns else pd.Series(np.nan, index=congl.index)
    congl["J"] = j_cosif.where(j_cosif.notna(), j_file).fillna(0.0)
    congl["stk_time_ord"] = (congl["stk_time"] - congl["J"]).clip(lower=0)
    # The letters stock: 4.3 net of the subordinated debt 4.3.9 that the 2025 plan books inside
    # it (before 2025, 4.3 holds none).  It is the scope of exp_bills and of the bills ratio.
    sub = congl["stk_sub"].fillna(0.0) if "stk_sub" in congl.columns else 0.0
    congl["stk_letters"] = (congl["stk_bills"] - sub).clip(lower=0)

    for c in STOCK_COLS + ["J", "stk_time_ord", "stk_letters"]:
        if c in congl.columns:
            congl[c + "_lag"] = congl.groupby("CodConglomeradoPrudencial")[c].shift(1)
    return congl.merge(macro[["AnoMes", "cdi_qoq", "savings_rate_qoq", "savings_old_qoq"]],
                       on="AnoMes", how="left")


def cdb_quarters(g):
    """The 2025+ quarters that measure the bank's CDB cost: a leaf expense, a positive lagged
    ordinary time-deposit stock and a CDI (the macro file can end before the COSIF data)."""
    q = g.dropna(subset=["exp_cdb", "stk_time_ord_lag", "cdi_qoq"])
    return q[(q["stk_time_ord_lag"] > 0) & (q["cdi_qoq"] > 0)].copy()


def estimate_perbank(post):
    """Per-conglomerate 2025+ own repo/bills ratios (as x CDI), n_i (the bank's CDB quarters,
    cdb_quarters) and the weight dep_w.  k_i comes later, from strip_k, once the strip it is
    measured on is built."""
    def ratio_rows(d, exp_col, stk_lag_col):
        x = d.dropna(subset=[exp_col, stk_lag_col, "cdi_qoq"]).copy()
        x = x[(x[stk_lag_col] > 0) & (x["cdi_qoq"] > 0)]
        x["r"] = (x[exp_col].abs() / x[stk_lag_col]) / x["cdi_qoq"]
        return x[(x["r"] >= 0) & (x["r"] < 5)]

    rows = []
    for code, g in post.groupby("CodConglomeradoPrudencial"):
        seg = g["segment"].dropna().iloc[0] if g["segment"].notna().any() else None
        rp = ratio_rows(g, "exp_repos", "stk_repos_lag")
        repo_i = wmean(rp["r"], rp["stk_repos_lag"]) if len(rp) else np.nan
        # bills: exp_bills over the letters stock; a quarter where the bank holds 4.3.9 but books
        # no 8119100006 is left out, because its subordinated cost then sits in the LF leaf
        sub_unbooked = (g["stk_sub_lag"].fillna(0) > 0) & (g["exp_sub"].fillna(0) == 0)
        bl = ratio_rows(g[~sub_unbooked], "exp_bills", "stk_letters_lag")
        bills_i = wmean(bl["r"], bl["stk_letters_lag"]) if len(bl) else np.nan
        cdbq = cdb_quarters(g)
        n_i = len(cdbq)
        rows.append({"CodConglomeradoPrudencial": code, "segment": seg,
                     "repo_ratio_i": repo_i, "bills_ratio_i": bills_i, "n_i": n_i,
                     "dep_w": cdbq["stk_time_ord_lag"].sum() if n_i else 0.0,
                     "bills_w": bl["stk_letters_lag"].sum() if len(bl) else 0.0,
                     # an SCFI conglomerate is measured on SCFIs in every 2025+ quarter; a bank group
                     # whose bank member stops reporting late in the sample stays a bank
                     "scfi": bool(g["scfi"].all()) if "scfi" in g else False})
    return pd.DataFrame(rows)


def judicial_expense(congl, seg_map):
    """exp_judicial_used: the cost of the judicial deposits, which the lump (8110000002) carries
    and exp_cdb does not.  2025+: the leaf 8114000008.  Before: rho_i x savings_rate_qoq x J_lag,
    where rho_i = sum |8114| / sum(savings_rate_qoq x J_lag) over the bank's 2025+ quarters (at
    least two); the J-weighted segment value where the bank has none, the global one where the
    segment has none.  Judicial deposits pay a savings-like rate: on BB's statements 2016-2024 the
    ratio to the new-rule savings yield is the most stable of the candidates (CV 0.10, against
    0.16 for the CDI)."""
    is_post = congl["year"] >= 2025
    x = congl[is_post].dropna(subset=["exp_judicial", "J_lag", "savings_rate_qoq"])
    x = x[x["J_lag"] > 0].assign(e=lambda d: d["exp_judicial"].abs(),
                                  w=lambda d: d["savings_rate_qoq"] * d["J_lag"])
    g = x.groupby("CodConglomeradoPrudencial")
    own = pd.DataFrame({"rho": g["e"].sum() / g["w"].sum(), "n": g.size(), "Jw": g["J_lag"].mean()})
    own = own[own["n"] >= 2].join(seg_map.set_index("CodConglomeradoPrudencial"))
    seg_rho = {s: wmean(o["rho"], o["Jw"]) for s, o in own.dropna(subset=["segment"]).groupby("segment")}
    glob_rho = wmean(own["rho"], own["Jw"])
    log.info("Judicial rho (x new-rule savings yield): %d own; segment %s; global %.3f",
             len(own), {k: round(v, 3) for k, v in seg_rho.items()}, glob_rho)
    seg = congl["CodConglomeradoPrudencial"].map(seg_map.set_index("CodConglomeradoPrudencial")["segment"])
    rho = congl["CodConglomeradoPrudencial"].map(own["rho"])
    rho = rho.fillna(seg.map(seg_rho)).fillna(glob_rho)
    imputed = rho * congl["savings_rate_qoq"].fillna(0) * congl["J_lag"].fillna(0)
    observed = congl["exp_judicial"].abs() if "exp_judicial" in congl.columns else np.nan
    congl["exp_judicial_used"] = np.where(is_post & pd.notna(observed), observed, imputed)
    return congl


# ---- statement benchmarks (DATA_RAW/issuers/funding_expense) ----------------------------
# The transcriptions read: the BB agent's compiled file and BB's own.  The folder also holds
# dfp_funding_expense_alternatives.csv, other presentations of years the compiled file already
# covers, which are never added to it.
DFP_FUNDING_FILES = ["dfp_funding_expense.csv", "dfp_funding_expense_bb.csv"]
# Lines of the statements' funding-expense note that match each imputed strip component; bills
# take bills_total, else the sum of its parts.
DFP_COMPONENT_LINES = {
    "savings": ["savings"],
    "repo": ["repo"],
    "bills": ["bills_total"],
    "judicial": ["judicial"],
}
DFP_BILLS_PARTS = ["bills_lca", "bills_lci", "bills_lf", "bills_other"]
# The subordinated and hybrid debt cost.  From 2025 its leaf 8119100006 is in the strip; before
# 2025 the lump carries it from a start year per bank (subordinated_in_lump.csv, from the bank's
# statements; SUB_IN_LUMP_DEFAULT where they do not say: BB's lump matches its statements' lines
# without this cost in 2016-2021 and with it from 2022).
DFP_SUB_LINES = {"subordinated": ["subordinated", "hybrid"]}
SUB_IN_LUMP_DEFAULT = 2022


def load_sub_in_lump():
    """The first year whose lump carries the bank's subordinated cost, per conglomerate:
    compiled/subordinated_in_lump.csv (cnpj, from_year, ...) where it exists; the banks it does
    not list take SUB_IN_LUMP_DEFAULT (logged as the default)."""
    f = os.path.join(DFP_FUNDING_DIR, "subordinated_in_lump.csv")
    if not os.path.exists(f):
        return pd.Series(dtype=float)
    d = pd.read_csv(f, dtype={"cnpj": str})
    return d.assign(CodConglomeradoPrudencial=d["cnpj"].astype(int).map(_cnpj_to_code())) \
            .dropna(subset=["CodConglomeradoPrudencial"]).set_index("CodConglomeradoPrudencial")["from_year"]


def _cnpj_to_code():
    """Issuer CNPJ (as an integer) -> conglomerate code, from the judicial-deposits file, which
    lists the banks whose statements are transcribed."""
    jmap = pd.read_csv(judicial_deposits.JUDICIAL_CSV, dtype={"CodConglomeradoPrudencial": str})
    return (jmap.assign(cnpj=jmap["cnpj"].astype(int))
                .drop_duplicates("cnpj").set_index("cnpj")["CodConglomeradoPrudencial"])


def load_dfp_benchmarks(component_lines=None):
    """Annual (FY) and second-semester (H2) funding expense per conglomerate and strip
    component, R$, from the transcribed statements (one file per transcription round, columns
    bank, cnpj, year, line, value_brl, ..., and an optional period FY/H2).  A component-year with
    an H2 value is split into H1 = FY - H2 and H2; otherwise it is one FY period.  The issuer's
    CNPJ maps to its conglomerate through the judicial-deposits file, which lists the same
    banks.  A line twice for the same bank, year and period stops the step.  Columns:
    CodConglomeradoPrudencial, year, period (FY, H1, H2), component, value."""
    component_lines = DFP_COMPONENT_LINES if component_lines is None else component_lines
    cols = ["CodConglomeradoPrudencial", "year", "period", "component", "value"]
    files = [f for f in (os.path.join(DFP_FUNDING_DIR, n) for n in DFP_FUNDING_FILES) if os.path.exists(f)]
    if not files:
        log.warning("No statement benchmarks in %s: every strip component stays imputed.", DFP_FUNDING_DIR)
        return pd.DataFrame(columns=cols)
    d = pd.concat([pd.read_csv(f, dtype={"cnpj": str}) for f in files], ignore_index=True)
    if "period" not in d.columns:
        d["period"] = "FY"
    d["period"] = d["period"].fillna("FY")
    d["CodConglomeradoPrudencial"] = d["cnpj"].astype(int).map(_cnpj_to_code())
    unmapped = sorted(d.loc[d["CodConglomeradoPrudencial"].isna(), "bank"].unique())
    if unmapped:
        log.warning("Statement benchmarks: no conglomerate for %s (dropped)", unmapped)
    d = d.dropna(subset=["CodConglomeradoPrudencial"])
    # bills: bills_total where the note gives it, else the sum of its parts
    parts = (d[d["line"].isin(DFP_BILLS_PARTS)]
             .groupby(["CodConglomeradoPrudencial", "year", "period"], as_index=False)["value_brl"].sum()
             .assign(line="bills_total"))
    have = d[d["line"] == "bills_total"].set_index(["CodConglomeradoPrudencial", "year", "period"]).index
    parts = parts[~parts.set_index(["CodConglomeradoPrudencial", "year", "period"]).index.isin(have)]
    d = pd.concat([d, parts], ignore_index=True)
    line2comp = {ln: comp for comp, lines in component_lines.items() for ln in lines}
    d = d[d["line"].isin(line2comp)].assign(component=lambda x: x["line"].map(line2comp))
    dup = d.duplicated(["CodConglomeradoPrudencial", "year", "period", "line"], keep=False)
    if dup.any():
        raise ValueError("Statement benchmarks: a line appears twice for the same bank, year and period: "
                         f"{d.loc[dup, ['bank', 'year', 'period', 'line']].drop_duplicates().head(10).to_dict('records')}")
    v = (d.groupby(["CodConglomeradoPrudencial", "year", "component", "period"])["value_brl"]
           .sum().unstack("period"))
    out = []
    for (code, year, comp), r in v.iterrows():
        fy, h2 = r.get("FY", np.nan), r.get("H2", np.nan)
        if pd.notna(fy) and pd.notna(h2):
            out += [(code, year, "H1", comp, fy - h2), (code, year, "H2", comp, h2)]
        elif pd.notna(fy):
            out.append((code, year, "FY", comp, fy))
    b = pd.DataFrame(out, columns=cols)
    log.info("Statement benchmarks %s: %d conglomerates, %d component-periods, from %s", sorted(component_lines),
             b["CodConglomeradoPrudencial"].nunique(), len(b), [os.path.basename(f) for f in files])
    return b


def strip_components(congl):
    """The imputed costs the strip removes from the lump, per conglomerate-quarter (R$), at the
    bank's shrunk ratios: repo at ratio x CDI x lagged stock, bills at ratio x CDI x lagged
    letters stock, the subordinated-debt cost (its 2025+ leaf; before, the statements' figure
    from the year the lump carries it), savings at the savings
    rate (the old-rule premium on the share OLD_RULE_SHARE), interbank at the CDI, and the
    judicial cost exp_judicial_used.  Where the bank's statements give a component's total for a
    year or semester, the component is rescaled to that total over the period's quarters (the
    statements fix the level, the imputation the quarterly shape); an observed 2025+ judicial
    leaf is kept as it is.  rate_a4_vintage_src records how the savings cost was obtained."""
    cdi = congl["cdi_qoq"]
    sub_leaf = congl["exp_sub"].abs() if "exp_sub" in congl.columns else pd.Series(np.nan, index=congl.index)
    comp = pd.DataFrame({
        "repo": congl["stk_repos_lag"].fillna(0) * congl["repo_shrunk"] * cdi,
        "bills": congl["stk_letters_lag"].fillna(0) * congl["bills_shrunk"] * cdi,
        # subordinated debt (4.3.9) from 2025: its leaf; before 2025 it sat outside 4.3
        "subordinated": sub_leaf.where(congl["year"] >= 2025).fillna(0.0),
        "savings": savings_cost_qoq(congl, OLD_RULE_SHARE).fillna(0) * congl["stk_savings_lag"].fillna(0),
        "interb": cdi.fillna(0) * congl["stk_interb_lag"].fillna(0),
        "judicial": congl["exp_judicial_used"],
    }, index=congl.index)

    bench = load_dfp_benchmarks()
    month = congl["AnoMes"] % 100
    keyframe = pd.DataFrame({"CodConglomeradoPrudencial": congl["CodConglomeradoPrudencial"].astype(str),
                             "year": congl["year"], "H": np.where(month <= 6, "H1", "H2")}, index=congl.index)
    vintage = pd.Series("none", index=congl.index)
    observed_jud = (congl["year"] >= 2025) & congl.get("exp_judicial", pd.Series(np.nan, index=congl.index)).notna()
    for c in DFP_COMPONENT_LINES:
        bc = bench[bench["component"] == c]
        if bc.empty:
            continue
        for period, n_q in (("H1", 2), ("H2", 2), ("FY", 4)):
            bp = bc[bc["period"] == period].set_index(["CodConglomeradoPrudencial", "year"])["value"]
            if bp.empty:
                continue
            in_p = keyframe["H"].eq(period) if period != "FY" else pd.Series(True, index=congl.index)
            k = keyframe[in_p]
            target = pd.Series(bp.reindex(pd.MultiIndex.from_frame(k[["CodConglomeradoPrudencial", "year"]])).to_numpy(),
                               index=k.index)
            ok = target.notna()
            if c == "judicial":
                ok &= ~observed_jud[k.index]
            if not ok.any():
                continue
            idx = target.index[ok]
            grp = [keyframe.loc[idx, "CodConglomeradoPrudencial"], keyframe.loc[idx, "year"]]
            tot = comp.loc[idx, c].groupby(grp).transform("sum")
            nq = comp.loc[idx, c].groupby(grp).transform("size")
            use = (tot > 0) & (nq == n_q)
            comp.loc[idx[use], c] = comp.loc[idx[use], c] * target[idx[use]] / tot[use]
            if c == "savings":
                vintage.loc[idx[use]] = "statement"
    congl["rate_a4_vintage_src"] = vintage

    # Before 2025, from the bank's start year, the lump carries the subordinated cost that no
    # Cosif leaf separates: where the statements give it, it is spread over the period's quarters
    # in proportion to the CDI (the debt is floating-rate) and leaves the strip.
    sb = load_dfp_benchmarks(DFP_SUB_LINES)
    if len(sb):
        start = load_sub_in_lump()
        codes = sb["CodConglomeradoPrudencial"].unique()
        default = sorted(set(codes) - set(start.index))
        start = pd.concat([start, pd.Series(SUB_IN_LUMP_DEFAULT, index=default)])
        log.info("Subordinated cost in the pre-2025 lump: start year from the statements for %s; "
                 "default %d for %s", start.drop(default).to_dict(), SUB_IN_LUMP_DEFAULT, default)
        from_year = keyframe["CodConglomeradoPrudencial"].map(start)
        for period, n_q in (("H1", 2), ("H2", 2), ("FY", 4)):
            bp = sb[sb["period"] == period].set_index(["CodConglomeradoPrudencial", "year"])["value"]
            if bp.empty:
                continue
            in_p = keyframe["H"].eq(period) if period != "FY" else pd.Series(True, index=congl.index)
            k = keyframe[in_p]
            target = pd.Series(bp.reindex(pd.MultiIndex.from_frame(k[["CodConglomeradoPrudencial", "year"]])).to_numpy(),
                               index=k.index)
            ok = target.notna() & (k["year"] < 2025) & (k["year"] >= from_year[k.index])
            if not ok.any():
                continue
            idx = target.index[ok]
            grp = [keyframe.loc[idx, "CodConglomeradoPrudencial"], keyframe.loc[idx, "year"]]
            w = cdi.loc[idx].fillna(0)
            wsum = w.groupby(grp).transform("sum")
            nq = w.groupby(grp).transform("size")
            use = (wsum > 0) & (nq == n_q)
            comp.loc[idx[use], "subordinated"] = target[idx[use]] * w[use] / wsum[use]
    return comp


def strip_cdb_expense(congl):
    """|lump| - repo - bills - subordinated - savings - interbank - judicial (strip_components),
    clipped at 0:
    the CDB expense the pre-2025 rate and k_i are built from."""
    comp = strip_components(congl)
    return (congl["exp_lump"].abs() - comp.sum(axis=1)).clip(lower=0)


def strip_k(post, eligible):
    """k_i = true CDB rate (leaf) / strip-implied CDB rate, both x CDI and ordinary-stock-
    weighted over cdb_quarters, i.e. sum(exp_cdb/CDI) / sum(strip/CDI): the stock weights cancel.
    The strip is the one the pre-2025 rate uses (post carries it), so k corrects the formula as
    it is applied.  Only a bank with its own repo AND its own bills ratio gets an own k (the
    `eligible` codes); the others take the segment's.  Extending own k to banks that hold no
    repos or bills was measured (2026-10-16) and rejected: their strips are small next to the
    CDB leaf, and k ran from 0.19 to 13.3."""
    k = {}
    for code, g in post[post["CodConglomeradoPrudencial"].isin(eligible)].groupby("CodConglomeradoPrudencial"):
        cdbq = cdb_quarters(g)
        if not len(cdbq):
            continue
        true_x = (cdbq["exp_cdb"].abs() / cdbq["cdi_qoq"]).sum()
        strip_x = (cdbq["strip"] / cdbq["cdi_qoq"]).sum()
        k[code] = true_x / strip_x if (np.isfinite(strip_x) and strip_x > 0) else np.nan
    return pd.Series(k, name="k_i", dtype=float)


def build_corrected():
    """The corrected type-4 rate (the frame main() writes) and the per-bank calibration table."""
    macro = load_macro_fraction()
    found = pd.read_csv(FOUNDATION, dtype={"CNPJ": str})
    found["CNPJ"] = found["CNPJ"].str.zfill(8)

    congl = build_congl_panel(found, macro)
    seg_map = load_segment_map()
    congl = congl.merge(seg_map, on="CodConglomeradoPrudencial", how="left")

    post = congl[congl["year"] >= 2025].copy()
    pb = estimate_perbank(post)
    has_q = pb["n_i"] > 0

    # ---- segment profiles (deposit-weighted over BANK conglomerates with 2025 data; the
    #      SCFI-measured ones have a different funding structure and do not enter them) ----
    bank = has_q & ~pb["scfi"]

    def profile(col, wcol="dep_w"):
        prof = {s: wmean(g.loc[bank.loc[g.index], col], g.loc[bank.loc[g.index], wcol])
                for s, g in pb.dropna(subset=["segment"]).groupby("segment")}
        return prof, wmean(pb.loc[bank, col], pb.loc[bank, wcol])

    def seg_val(prof, seg, fallback):
        v = prof.get(seg, np.nan)
        return v if pd.notna(v) else fallback

    # ---- shrinkage (w_i = n_i/(n_i+N0)); per-component fallback to segment ----
    pb["w_i"] = pb["n_i"] / (pb["n_i"] + N0)

    def shrink(own, prof, fb):
        segv = pb["segment"].apply(lambda s: seg_val(prof, s, fb))
        w = pb["w_i"].where(pb[own].notna(), 0.0)
        return w * pb[own].fillna(0.0) + (1 - w) * segv

    def assign_all(col, prof, fb):
        """col for EVERY conglomerate (incl. those absent in 2025): own shrunk value, else the
        segment's, else the global one."""
        v = all_codes["CodConglomeradoPrudencial"].map(pb.set_index("CodConglomeradoPrudencial")[col])
        segv = all_codes["segment"].apply(lambda s: seg_val(prof, s, fb))
        return v.where(v.notna(), segv)

    all_codes = congl[["CodConglomeradoPrudencial"]].drop_duplicates().merge(
        seg_map, on="CodConglomeradoPrudencial", how="left")
    repo_prof, repo_fb = profile("repo_ratio_i")
    # the bills profile is weighted by the letters stock: a bank whose 4.3 is mostly
    # subordinated debt measures its letters ratio on a small stock and must not set it
    bills_prof, bills_fb = profile("bills_ratio_i", "bills_w")
    pb["repo_shrunk"] = shrink("repo_ratio_i", repo_prof, repo_fb)
    pb["bills_shrunk"] = shrink("bills_ratio_i", bills_prof, bills_fb)
    all_codes["repo_shrunk"] = assign_all("repo_shrunk", repo_prof, repo_fb)
    all_codes["bills_shrunk"] = assign_all("bills_shrunk", bills_prof, bills_fb)
    congl = congl.merge(all_codes[["CodConglomeradoPrudencial", "repo_shrunk", "bills_shrunk"]],
                        on="CodConglomeradoPrudencial", how="left")

    # ---- the strip, with the judicial cost and the statement benchmarks ----
    congl = judicial_expense(congl, seg_map)
    congl["strip"] = strip_cdb_expense(congl)
    post = congl[congl["year"] >= 2025]

    # k is measured on the strip it is applied to (strip_k), then shrunk like the ratios
    eligible = pb.loc[pb["repo_ratio_i"].notna() & pb["bills_ratio_i"].notna(), "CodConglomeradoPrudencial"]
    pb["k_i"] = pb["CodConglomeradoPrudencial"].map(strip_k(post, set(eligible)))
    k_prof, k_fb = profile("k_i")
    pb["k_shrunk"] = shrink("k_i", k_prof, k_fb)
    seg_prof = pd.DataFrame({"repo_ratio_seg": repo_prof, "bills_ratio_seg": bills_prof,
                             "k_seg": k_prof})
    glob_fb = {"repo": repo_fb, "bills": bills_fb, "k": k_fb}
    log.info("Segment profiles:\n%s", seg_prof.to_string())
    log.info("Global fallback: repo=%.3f bills=%.3f k=%.3f",
             glob_fb["repo"], glob_fb["bills"], glob_fb["k"])
    log.info("Own k_i for %d of %d conglomerates with 2025 CDB quarters",
             int(pb.loc[has_q, "k_i"].notna().sum()), int(has_q.sum()))

    all_codes["k_shrunk"] = assign_all("k_shrunk", k_prof, k_fb)
    all_codes["n_i"] = all_codes["CodConglomeradoPrudencial"].map(
        pb.set_index("CodConglomeradoPrudencial")["n_i"]).fillna(0)

    def tier(r):
        if r["n_i"] >= 2:
            return "perbank"
        if r["n_i"] >= 1:
            return "perbank_weak"
        s = r["segment"]
        if s in seg_prof.index and pd.notna(seg_prof.loc[s, "k_seg"]):
            return "segment"
        return "global"
    all_codes["tier"] = all_codes.apply(tier, axis=1)

    congl = congl.merge(all_codes[["CodConglomeradoPrudencial", "k_shrunk", "tier"]],
                        on="CodConglomeradoPrudencial", how="left")

    # ---- apply: pre-2025 strip*k ; 2025+ direct leaf; both over the ORDINARY stock ----
    # A row is attempted where the lagged ordinary stock is positive and the expense exists; the
    # rows that are not attempted are absent from the output, so panel_deposit_rates falls back
    # to its residual for them.  An attempted rate that fails is kept as missing with rate_flag
    # (2: strip or leaf <= 0; 3: above RATE_Q_CAP), and panel_deposit_rates does not replace it.
    is_post = congl["year"] >= 2025
    den = congl["stk_time_ord_lag"]
    num = np.where(is_post, congl["exp_cdb"].abs(), congl["strip"] * congl["k_shrunk"])
    num = pd.Series(num, index=congl.index)
    attempted = (den > 0) & num.notna() & np.where(is_post, congl["exp_cdb"].notna(), congl["exp_lump"].notna())
    rate = num / den
    congl["rate_flag"] = 0
    congl.loc[attempted & (num <= 0), "rate_flag"] = 2
    congl.loc[attempted & (num > 0) & (~np.isfinite(rate) | (rate > RATE_Q_CAP)), "rate_flag"] = 3
    congl["rate_a4_corrected_qoq"] = rate.where(congl["rate_flag"] == 0)
    congl["rate_a4_corrected_ann"] = annualize(congl["rate_a4_corrected_qoq"])
    out = congl.loc[attempted, ["CodConglomeradoPrudencial", "AnoMes", "rate_a4_corrected_qoq",
                                "rate_a4_corrected_ann", "tier", "rate_flag", "rate_a4_vintage_src"]].copy()
    log.info("Corrected rate: %d attempted conglomerate-quarters; strip or leaf <= 0: %d; above cap: %d",
             len(out), int((out["rate_flag"] == 2).sum()), int((out["rate_flag"] == 3).sum()))
    return out, pb


def main():
    if not os.path.exists(FOUNDATION):
        log.error("Foundation not found: %s -- run panel_cosif_extract.py first.",
                  FOUNDATION)
        sys.exit(1)
    out, _ = build_corrected()
    out.to_csv(OUT_CORRECTED, index=False)
    log.info("Wrote corrected CDB rate: %s  (%d rows, %d conglomerates, %d quarters)",
             OUT_CORRECTED, len(out), out["CodConglomeradoPrudencial"].nunique(),
             out["AnoMes"].nunique())


if __name__ == "__main__":
    main()
