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
#      A failed rate (strip or leaf <= 0, above RATE_Q_CAP) stays missing with rate_flag.  A bank
#      whose strip has a component rescaled to a statement (benchmarked) keeps each semester's
#      total and spreads it over the two quarters in proportion to CDI x lagged ordinary stock.
#   5. Statement banks (statement_cdb_expense): before 2025, a year (or semester) whose statement
#      prints a time-deposit line takes it as its CDB expense.  A bank that also prints a 2025 line
#      takes lambda_i x the line net of its judicial cost, lambda_i = Cosif 8113 over the 2025 line
#      so netted (td_lambda); a bank without one takes the line net of the scope items it
#      includes, where the scope is resolved.  Items a note prints inside the line are netted
#      first (DFP_NET, DFP_INLINE).  The expense is spread over the period's quarters in
#      proportion to CDI x lagged ordinary stock, so rate/CDI is constant within the period, and
#      k does not enter.  The other bank-years keep step 4.
#
# Inputs  : custos_implicitos_v2_foundation.csv      (panel_cosif_extract.py, step 12; BANCOS
#             members, and the SCFI members used only where a conglomerate has no bank)
#           SGS_RAW/macro_series_full.csv and SGS_RAW/savings_yields/  (quarterly macro and
#             savings rates computed here through utils.macro_rates, as fractions; step 16
#             writes the same rates in percent)
#           IF-Data List  (CNPJ<->conglomerate merge_asof crosswalk, utils.conglomerates,
#             keyed as deposits_panel.csv files each CNPJ; and the segment Sr per conglomerate)
#           Rout/judicial_deposits_share.csv  (utils.judicial_deposits)
#           DATA_RAW/issuers/funding_expense/compiled/  (statements: dfp_funding_expense.csv and
#             dfp_funding_expense_bb.csv; time_deposit_scope*.csv, what each time-deposit line
#             includes; subordinated_in_lump.csv)
# Output  : COSIF_PROCESSED/cosif_cdb_rate_corrected.csv
#             columns: CodConglomeradoPrudencial, AnoMes, rate_a4_corrected_qoq,
#                      rate_a4_corrected_ann, tier, rate_flag (0 measured, 2 strip or leaf
#                      <= 0, 3 above RATE_Q_CAP or marked failed in RATE_FAIL, 4 a
#                      benchmarked rate outside BENCH_RANGE x CDI, 5 an imputed rate more than
#                      IMPUTED_GAP_PP above the CDI or below IMPUTED_FLOOR x CDI),
#                      rate_a4_vintage_src (statement or none),
#                      rate_a4_src (statement: the time-deposit line; benchmarked: strip x k with
#                      a component rescaled to a statement; imputed: strip x k; leaf: 2025+)
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
# Pedro's rules on the statement lines (2026-11-03, 21:50; 2026-11-04, N2).  Left out: BB's
# savings line in 2018-19 (it exceeds any lawful remuneration of the stock, so it is not pure
# savings cost; BB's savings cost stays at the official yield in those years); Banrisul's 2023
# judicial line (inferred, not printed); Santander's 2024 time-deposit, repo and bills lines,
# which carry R$42.2 bn of exchange-rate effects the note does not split by line.  Also left out,
# from the BB agent's B8 notes, rows that are part of another line or outside the lump, so that
# no strip component counts them: Caixa's subordinated rows (inside bills_lf, 2020-24) and its
# IHCD (hybrid; outside Cosif 8.1.1 in every year), and Santander's subordinated rows (interest on
# capital instruments, inside its time-deposit line, which DFP_INLINE nets them from).
DFP_EXCLUDE = [  # (cnpj, lines, years; None = every year)
    ("00000000", ["savings"], [2018, 2019]),
    ("92702067", ["judicial"], [2023]),
    ("90400888", ["time_deposits", "repo", "bills_total"] + DFP_BILLS_PARTS, [2024]),
    ("00360305", ["subordinated", "hybrid"], None),
    ("90400888", ["subordinated"], None),
]
# Netted: the exchange-rate variation Santander's notes print inside a line.  Its 2020 bills line
# (footnote 3), and its time-deposit line every year (footnote 2; N5, Pedro 2026-11-04): an expense
# except 2016; 2019 as DFP 2020 labels it, where DFP 2019 calls the same amount an income.  The
# 2025 footnote (an income of R$478,899 thousand) is shared with repo and bills and not split, so
# the 2025 line keeps it.
DFP_NET = [  # (cnpj, line, year, R$ taken out of the printed value; negative: an income added back)
    ("90400888", "bills_total", 2020, 22_189_857_000),
    ("90400888", "time_deposits", 2016, -2_416_234_000),  # DFP 2017 p104, the 2016 comparative
    ("90400888", "time_deposits", 2017, 494_178_000),     # DFP 2017 p104
    ("90400888", "time_deposits", 2018, 2_514_803_000),   # DFP 2018 p101
    ("90400888", "time_deposits", 2019, 1_207_360_000),   # DFP 2020 p105 (DFP 2019 p94: income)
    ("90400888", "time_deposits", 2020, 9_586_000),       # DFP 2020 p105
    ("90400888", "time_deposits", 2021, 1_528_068_000),   # DFP 2021 p89
    ("90400888", "time_deposits", 2022, 5_414_167_000),   # DFP 2022 p82
    ("90400888", "time_deposits", 2023, 6_460_243_000),   # DFP 2023 p77
]
# Rows that sit inside another line of the same bank, year and period, netted from it: footnote
# (1) to Santander's 'Depósitos a Prazo', the interest on its capital instruments (the external
# rows 2016-23, the subordinated rows 2024-25; N5).
DFP_INLINE = [("90400888", "time_deposits", ["external", "subordinated"])]
# What each statement's time-deposit line includes beyond ordinary domestic CDBs (D10): the BB
# agent's file for the eight banks, the coding thread's for BB.
TD_SCOPE_FILES = ["time_deposit_scope.csv", "time_deposit_scope_bb.csv"]
# The scope items netted out of a time-deposit line, with the cost each is netted at.  For a bank
# without a lambda_i, a line that includes any other item (special or foreign-currency deposits,
# interest on capital instruments, exchange-rate effects), or leaves one undetermined, is not used.
TD_NETTED = {"judicial": "exp_judicial_used"}
# The year whose Cosif CDB leaf measures lambda_i, the CDB share of a bank's time-deposit line (N1,
# Pedro 2026-11-04).
TD_LAMBDA_YEAR = 2025
# Scope items read as not included: inside the line, negligible, and with no expense line of their
# own.  BRB's special deposits ('Obrigações para depósitos específicos e de fundos e programas',
# then consignment deposits) are R$3-8 m against a time-deposit class of R$7-34 bn
# (brb_deposit_balances.csv).
TD_NEGLIGIBLE = [("00000208", "special")]
# Bank-years whose pre-2025 rate is marked failed (rate_flag 3) and left to panel_deposit_rates'
# carry (N3, Pedro 2026-11-04): Santander 2024, whose lump carries exchange-rate effects the note
# does not split by line.
RATE_FAIL = [("90400888", [2024])]
# The rate/CDI range a benchmarked pre-2025 rate must lie in to count as measured (I14, Pedro
# 2026-11-04): outside it the rate is marked failed (rate_flag 4) and panel_deposit_rates carries
# it.  Itau's semesters at 2-3.6 x CDI (2016, 2018, 2020) are the case.
BENCH_RANGE = (0.5, 1.5)
# The band an imputed pre-2025 rate (strip x k, no statement behind it) must lie in to count as
# measured (I17; Pedro, D1 2026-11-05): at most IMPUTED_GAP_PP percentage points a year above the
# CDI, both compounded annual, and at least IMPUTED_FLOOR x CDI.  Outside it the rate is marked
# failed (rate_flag 5) and panel_deposit_rates carries it.  The excess at the foreign, wholesale and
# fixed-rate issuers is a level above the CDI (4.5-11 pp a year), so the upper bound is in points,
# not a multiple: a multiple would bite only when the CDI is low.  panel_deposit_rates applies the
# same band to its own uncorrected rate.
IMPUTED_GAP_PP = 4.0
IMPUTED_FLOOR = 0.5


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


def _apply_statement_rules(d):
    """DFP_NET, DFP_INLINE and DFP_EXCLUDE on the transcribed lines (columns as read, with
    period), before any line is used.  The netting reads every row, the excluded ones included."""
    cnpj = d["cnpj"].astype(int)
    drop = pd.Series(False, index=d.index)
    for c, lines, years in DFP_EXCLUDE:
        m = (cnpj == int(c)) & d["line"].isin(lines)
        drop |= m & d["year"].isin(years) if years is not None else m
    d = d.copy()
    for c, line, year, amount in DFP_NET:
        m = (cnpj == int(c)) & (d["line"] == line) & (d["year"] == year)
        d.loc[m, "value_brl"] = d.loc[m, "value_brl"] - amount
    for c, line, inner in DFP_INLINE:
        part = d[(cnpj == int(c)) & d["line"].isin(inner)].groupby(["year", "period"])["value_brl"].sum()
        m = (cnpj == int(c)) & (d["line"] == line)
        take = part.reindex(pd.MultiIndex.from_frame(d.loc[m, ["year", "period"]])).fillna(0).to_numpy()
        d.loc[m, "value_brl"] = d.loc[m, "value_brl"] - take
        log.info("Netted from %s's %s: %s (DFP_INLINE), R$ bn by year %s", c, line, inner,
                 {int(y): round(v / 1e9, 3) for (y, _), v in part.items()})
    if drop.any():
        log.info("Statement lines left out (DFP_EXCLUDE): %s",
                 d[drop].groupby(["bank", "line"])["year"].apply(lambda y: sorted(set(y))).to_dict())
    return d[~drop]


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
    d = _apply_statement_rules(d.dropna(subset=["CodConglomeradoPrudencial"]))
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


def _period_keys(congl):
    """Conglomerate, year and semester (H1, H2) of each row: the keys a statement period matches."""
    month = congl["AnoMes"] % 100
    return pd.DataFrame({"CodConglomeradoPrudencial": congl["CodConglomeradoPrudencial"].astype(str),
                         "year": congl["year"], "H": np.where(month <= 6, "H1", "H2")}, index=congl.index)


def strip_components(congl):
    """The imputed costs the strip removes from the lump, per conglomerate-quarter (R$), at the
    bank's shrunk ratios: repo at ratio x CDI x lagged stock, bills at ratio x CDI x lagged
    letters stock, the subordinated-debt cost (its 2025+ leaf; before, the statements' figure
    from the year the lump carries it), savings at the savings
    rate (the old-rule premium on the share OLD_RULE_SHARE), interbank at the CDI, and the
    judicial cost exp_judicial_used.  Where the bank's statements give a component's total for a
    year or semester, the component is rescaled to that total over the period's quarters (the
    statements fix the level, the imputation the quarterly shape); an observed 2025+ judicial
    leaf is kept as it is.  rate_a4_vintage_src records how the savings cost was obtained, and
    strip_benchmarked marks the rows where any component comes from a statement."""
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
    keyframe = _period_keys(congl)
    vintage = pd.Series("none", index=congl.index)
    benchmarked = pd.Series(False, index=congl.index)
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
            benchmarked.loc[idx[use]] = True
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
            benchmarked.loc[idx[use]] = True
    congl["strip_benchmarked"] = benchmarked
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


def load_td_scope():
    """What each statement's time-deposit line includes beyond ordinary domestic CDBs, per
    conglomerate and year, from compiled/time_deposit_scope*.csv (cnpj, year_from, year_to,
    includes_judicial, includes_special, includes_foreign, page, note; any further includes_<item>
    column is read the same way; an empty year_to runs to the current year).  true/false, yes/no
    or 1/0; anything else is undetermined (NaN).  A frame indexed by (CodConglomeradoPrudencial,
    year), one column per item, or None when no scope file exists.  A year that two rows of a bank
    cover stops the step.  The TD_NEGLIGIBLE items are read as not included."""
    files = [f for f in (os.path.join(DFP_FUNDING_DIR, n) for n in TD_SCOPE_FILES) if os.path.exists(f)]
    if not files:
        return None
    s = pd.concat([pd.read_csv(f, dtype={"cnpj": str}) for f in files], ignore_index=True)
    items = [c for c in s.columns if c.startswith("includes_")]
    truth = {"true": True, "yes": True, "y": True, "1": True, "1.0": True,
             "false": False, "no": False, "n": False, "0": False, "0.0": False}
    s["CodConglomeradoPrudencial"] = s["cnpj"].astype(int).map(_cnpj_to_code())
    rows = []
    for r in s.dropna(subset=["CodConglomeradoPrudencial"]).to_dict("records"):
        last = int(r["year_to"]) if pd.notna(r["year_to"]) else pd.Timestamp.today().year
        for y in range(int(r["year_from"]), last + 1):
            rows.append({"CodConglomeradoPrudencial": r["CodConglomeradoPrudencial"], "year": y,
                         **{c[len("includes_"):]: truth.get(str(r[c]).strip().lower(), np.nan) for c in items}})
    sc = pd.DataFrame(rows)
    dup = sc.duplicated(["CodConglomeradoPrudencial", "year"], keep=False)
    if dup.any():
        raise ValueError("time_deposit_scope: a year is covered twice for "
                         f"{sc.loc[dup, ['CodConglomeradoPrudencial', 'year']].drop_duplicates().head(10).to_dict('records')}")
    for c, item in TD_NEGLIGIBLE:
        m = sc["CodConglomeradoPrudencial"].eq(_cnpj_to_code().get(int(c))) & sc[item].eq(True)
        if m.any():
            sc.loc[m, item] = False
            log.info("Time-deposit scope: %s's %s read as not included (negligible, no expense line) in %s",
                     c, item, sorted(sc.loc[m, "year"]))
    log.info("Time-deposit scope: %d conglomerates, %d bank-years, items %s, from %s",
             sc["CodConglomeradoPrudencial"].nunique(), len(sc), [c[len("includes_"):] for c in items],
             [os.path.basename(f) for f in files])
    return sc.set_index(["CodConglomeradoPrudencial", "year"])


def _netted_cost(congl, scope, rows):
    """The cost of the TD_NETTED items the scope includes, per row of `rows` (the quarters' keys:
    CodConglomeradoPrudencial, year); 0 where the scope excludes an item, leaves it undetermined
    or has no row."""
    sc = scope.reindex(pd.MultiIndex.from_frame(rows[["CodConglomeradoPrudencial", "year"]]))
    sc.index = rows.index
    net = pd.Series(0.0, index=rows.index)
    for item, col in TD_NETTED.items():
        if item in sc.columns:
            net += congl.loc[rows.index, col].fillna(0) * sc[item].eq(True)
    return net, sc


def td_lambda(congl, td, scope):
    """N1 (Pedro, 2026-11-04): lambda_i, the share of a bank's statement time-deposit line that is
    domestic CDB expense, for each bank whose statements print a TD_LAMBDA_YEAR line: sum |8113|
    over that year's four quarters over the year's line net of the judicial cost where the scope
    includes judicial deposits.  Items a note prints inside the line are netted before
    (DFP_NET, DFP_INLINE); those it does not print (special and foreign deposits) leave through
    lambda_i at their TD_LAMBDA_YEAR share.  A share that moves is the method's limit, measured on
    BB, whose special and foreign deposits can be imputed item by item: 2018-19, when their share
    was larger, sit +0.14 and +0.17 x CDI above that imputation.  A Series indexed by
    conglomerate."""
    y = TD_LAMBDA_YEAR
    line = td[td["year"] == y].groupby("CodConglomeradoPrudencial")["value"].sum()
    g = congl[(congl["year"] == y) & congl["CodConglomeradoPrudencial"].astype(str).isin(line.index)]
    keys = _period_keys(g)
    net, _ = _netted_cost(g, scope, keys)
    by = keys["CodConglomeradoPrudencial"]
    leaf = g["exp_cdb"].abs().groupby(by).sum(min_count=1)
    n_leaf = g["exp_cdb"].notna().groupby(by).sum()
    base = line - net.groupby(by).sum().reindex(line.index).fillna(0)
    lam = (leaf.reindex(line.index) / base).where(n_leaf.reindex(line.index).eq(4) & (base > 0))
    if lam.isna().any():
        log.warning("lambda_i: no value for %s (fewer than four %d quarters with the CDB leaf, or a "
                    "line net of judicial <= 0); their lines follow the scope rule", sorted(lam.index[lam.isna()]), y)
    lam = lam.dropna()
    log.info("lambda_i (Cosif 8113 over the %d time-deposit line net of judicial): %s.  It holds the line's "
             "non-CDB share at its %d value; measured limit: BB 2018-19 at +0.14 and +0.17 x CDI against "
             "the item-by-item imputation of its special and foreign deposits", y,
             {c: round(v, 3) for c, v in lam.items()}, y)
    return lam


def statement_cdb_expense(congl):
    """D10 and N1 (Pedro, 2026-11-03 and 11-04): before 2025, a statement bank's CDB expense for a
    period (FY, or H1 and H2 where the note prints the second semester) comes from its
    time-deposit line.  A bank with a lambda_i (td_lambda) takes lambda_i x (the line - the cost
    of the TD_NETTED items its scope includes).  A bank without one takes the line net of that
    cost where its scope row (load_td_scope) has every item determined, with none included beyond
    those in TD_NETTED.  Used where every quarter of the period is present and the expense is
    positive.  It is spread over the quarters in proportion to CDI x lagged ordinary stock (N4,
    Pedro 2026-11-04), so the rate is a constant multiple of the CDI within the period: the
    statement fixes the period's level, and the strip's quarterly profile carries the lump's noise
    (exchange-rate effects, accrual timing).  Returns the quarterly expense (R$), NaN where no line
    is used, and logs the bank-years on this shape and why each statement period with a line is
    not used."""
    out = pd.Series(np.nan, index=congl.index)
    td = load_dfp_benchmarks({"time_deposits": ["time_deposits"]})
    if td.empty:
        return out
    scope = load_td_scope()
    if scope is None:
        log.warning("No time-deposit scope file in %s: no statement time-deposit line is used.", DFP_FUNDING_DIR)
        return out
    lam = td_lambda(congl, td, scope)
    keyframe = _period_keys(congl)
    shape = (congl["cdi_qoq"] * congl["stk_time_ord_lag"]).fillna(0)
    status = []
    for period, n_q in (("H1", 2), ("H2", 2), ("FY", 4)):
        bp = td[td["period"] == period].set_index(["CodConglomeradoPrudencial", "year"])["value"]
        if bp.empty:
            continue
        in_p = keyframe["H"].eq(period) if period != "FY" else pd.Series(True, index=congl.index)
        k = keyframe[in_p & (keyframe["year"] < 2025)]
        key = pd.MultiIndex.from_frame(k[["CodConglomeradoPrudencial", "year"]])
        target = pd.Series(bp.reindex(key).to_numpy(), index=k.index).dropna()
        if target.empty:
            continue
        k = k.loc[target.index]
        grp = [k["CodConglomeradoPrudencial"], k["year"]]
        net, sc = _netted_cost(congl, scope, k)
        lam_k = k["CodConglomeradoPrudencial"].map(lam)
        scoped = lam_k.isna()
        has_row = sc.notna().any(axis=1)
        determined = sc.notna().all(axis=1)
        others = [c for c in sc.columns if c not in TD_NETTED]
        inc = sc[others].eq(True)
        other_in = inc.any(axis=1)
        inc_label = (inc.apply(lambda r: "includes " + "+".join(c for c in others if r[c]), axis=1)
                     if others else pd.Series("", index=k.index))
        expense = (target - net.groupby(grp).transform("sum")) * lam_k.fillna(1.0)
        w = shape.loc[k.index]
        wsum = w.groupby(grp).transform("sum")
        nq = target.groupby(grp).transform("size")
        why = np.select([scoped & ~has_row, scoped & ~determined, scoped & other_in, nq != n_q,
                         expense <= 0, wsum <= 0],
                        ["no scope row", "scope undetermined", inc_label.to_numpy(dtype=object),
                         "incomplete period", "net expense <= 0", "no shape"], "used")
        use = why == "used"
        out.loc[k.index[use]] = (expense * w / wsum)[use]
        status.append(pd.DataFrame({"code": k["CodConglomeradoPrudencial"], "year": k["year"],
                                    "period": period, "why": why,
                                    "rule": np.where(scoped, "scope", "lambda")}).drop_duplicates())
    if status:
        st = pd.concat(status, ignore_index=True)
        used = st[st["why"] == "used"]
        log.info("Statement time-deposit lines: %d bank-periods used of %d with a line (lambda %d, scope %d)",
                 len(used), len(st), int((used["rule"] == "lambda").sum()), int((used["rule"] == "scope").sum()))
        log.info("Shape: CDI x lagged ordinary stock for the statement bank-years %s; every other "
                 "pre-2025 bank-year is strip x k", used.groupby("code")["year"].apply(lambda y: sorted(set(y))).to_dict())
        for (why, code), g in st[st["why"] != "used"].groupby(["why", "code"]):
            log.info("  not used (%s): %s %s", why, code, sorted(set(g["year"])))
    return out


def benchmarked_semester_shape(congl, pre):
    """I12 (Pedro, 2026-11-04): before 2025, a benchmarked bank's CDB expense (strip x k with a
    component rescaled to a statement) keeps each semester's total and is spread over its two
    quarters in proportion to CDI x lagged ordinary stock, as a statement period is (N4): at these
    banks the strip's quarterly profile follows the semester accounts' uneven booking, not
    pricing (Itau 2016-2020).  Applied where both quarters of the semester are benchmarked, with
    an expense, a positive total and a positive shape; elsewhere `pre` is returned unchanged."""
    keys = _period_keys(congl)
    m = (congl["rate_a4_src"] == "benchmarked") & (congl["year"] < 2025) & pre.notna()
    k = keys[m]
    grp = [k["CodConglomeradoPrudencial"], k["year"], k["H"]]
    w = congl.loc[m, "cdi_qoq"] * congl.loc[m, "stk_time_ord_lag"]
    tot = pre[m].groupby(grp).transform("sum")
    wsum = w.groupby(grp).transform("sum")
    whole = (pre[m].groupby(grp).transform("size") == 2) & w.notna().groupby(grp).transform("all")
    use = whole & (tot > 0) & (wsum > 0)
    out = pre.copy()
    out.loc[use.index[use]] = (tot * w / wsum)[use]
    log.info("Benchmarked semesters on the CDI x stock shape: %s",
             (k[use].drop_duplicates(["CodConglomeradoPrudencial", "year", "H"])
              .groupby("CodConglomeradoPrudencial").size().to_dict()))
    return out


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
    congl["stmt_cdb"] = statement_cdb_expense(congl)
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

    # ---- apply: pre-2025 the statement line (D10) or strip*k; 2025+ direct leaf; all over the
    #      ORDINARY stock ----
    # A row is attempted where the lagged ordinary stock is positive and the expense exists; the
    # rows that are not attempted are absent from the output, so panel_deposit_rates falls back
    # to its residual for them.  An attempted rate that fails is kept as missing with rate_flag
    # (2: strip or leaf <= 0; 3: above RATE_Q_CAP, or a bank-year in RATE_FAIL; 4: a benchmarked
    # rate outside BENCH_RANGE x CDI; 5: an imputed rate outside the IMPUTED_* band), and
    # panel_deposit_rates does not replace it.
    is_post = congl["year"] >= 2025
    den = congl["stk_time_ord_lag"]
    congl["rate_a4_src"] = np.select([is_post, congl["stmt_cdb"].notna(), congl["strip_benchmarked"]],
                                     ["leaf", "statement", "benchmarked"], "imputed")
    pre = congl["stmt_cdb"].where(congl["stmt_cdb"].notna(), congl["strip"] * congl["k_shrunk"])
    pre = benchmarked_semester_shape(congl, pre)
    num = pd.Series(np.where(is_post, congl["exp_cdb"].abs(), pre), index=congl.index)
    attempted = (den > 0) & num.notna() & np.where(is_post, congl["exp_cdb"].notna(), congl["exp_lump"].notna())
    rate = num / den
    congl["rate_flag"] = 0
    congl.loc[attempted & (num <= 0), "rate_flag"] = 2
    congl.loc[attempted & (num > 0) & (~np.isfinite(rate) | (rate > RATE_Q_CAP)), "rate_flag"] = 3
    fail = pd.Series(False, index=congl.index)
    for c, years in RATE_FAIL:
        fail |= congl["CodConglomeradoPrudencial"].eq(_cnpj_to_code().get(int(c))) & congl["year"].isin(years)
    congl.loc[attempted & fail & ~is_post, "rate_flag"] = 3
    log.info("Marked failed (RATE_FAIL): %d conglomerate-quarters", int((attempted & fail & ~is_post).sum()))
    bench_out = (attempted & ~is_post & (congl["rate_a4_src"] == "benchmarked") & (congl["rate_flag"] == 0)
                 & ~(rate / congl["cdi_qoq"]).between(*BENCH_RANGE))
    congl.loc[bench_out, "rate_flag"] = 4
    gap_pp = 100 * ((1 + rate) ** 4 - (1 + congl["cdi_qoq"]) ** 4)
    imp_out = (attempted & ~is_post & (congl["rate_a4_src"] == "imputed") & (congl["rate_flag"] == 0)
               & ((gap_pp > IMPUTED_GAP_PP) | (rate < IMPUTED_FLOOR * congl["cdi_qoq"])))
    congl.loc[imp_out, "rate_flag"] = 5
    log.info("Imputed rates more than %g pp a year above the CDI or below %g x CDI (rate_flag 5): %d "
             "conglomerate-quarters at %d conglomerates (above %d, below %d)", IMPUTED_GAP_PP, IMPUTED_FLOOR,
             int(imp_out.sum()), congl.loc[imp_out, "CodConglomeradoPrudencial"].nunique(),
             int((imp_out & (gap_pp > IMPUTED_GAP_PP)).sum()), int((imp_out & (gap_pp <= IMPUTED_GAP_PP)).sum()))
    log.info("Benchmarked rates outside %s x CDI (rate_flag 4): %s", BENCH_RANGE,
             {c: sorted(f"{a // 100}Q{(a % 100) // 3}" for a in g)
              for c, g in congl.loc[bench_out].groupby("CodConglomeradoPrudencial")["AnoMes"]})
    congl["rate_a4_corrected_qoq"] = rate.where(congl["rate_flag"] == 0)
    congl["rate_a4_corrected_ann"] = annualize(congl["rate_a4_corrected_qoq"])
    out = congl.loc[attempted, ["CodConglomeradoPrudencial", "AnoMes", "rate_a4_corrected_qoq",
                                "rate_a4_corrected_ann", "tier", "rate_flag", "rate_a4_vintage_src",
                                "rate_a4_src"]].copy()
    log.info("Corrected rate: %d attempted conglomerate-quarters; strip or leaf <= 0: %d; above cap: %d; "
             "benchmarked out of range: %d; imputed out of band: %d; source %s", len(out),
             int((out["rate_flag"] == 2).sum()), int((out["rate_flag"] == 3).sum()),
             int((out["rate_flag"] == 4).sum()), int((out["rate_flag"] == 5).sum()),
             out["rate_a4_src"].value_counts().to_dict())
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
