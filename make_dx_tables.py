"""
make_dx_tables.py
=================
Tables of the `dx` robustness variant of the demand side: the D-type dummy (1 for a digital firm,
`is_B == false`) added to the linear part of the logit and of the RC-BLP, and the BBL cost
estimation re-run on that fit. Two appendix tables, main specification against the variant:

  tab_dx_demand_robustness   per routine, four columns: Logit, Logit + D, RC, RC + D.
                             Rows: the price coefficient, the D-type coefficient, Segment S5 (the
                             one characteristic only D firms have), the random-coefficient
                             parameters, the mean own-price elasticity, N, Q, G*.
                             `--rows all` prints every mean-utility coefficient.
  tab_dx_bbl_comparison      per routine and firm type, Main against D: c-bar (compounded annual
                             pp) with its 95% interval, omega and zeta with their profile
                             intervals, the share of violated inequalities, firms and rows.

WHAT IT READS (never writes there)
  logit, both columns   ESTIMATION_OUTPUT/BLP_RESULTS/logit/logit_summary_spec_12.json
                        (E{k}_full, E{k}_full_dtype: the logit step estimates both sub-models, so
                        the two logit columns exist before any cluster run of the variant)
  RC, main              BLP_RESULTS/cluster_raw/blp_results_E{k}_spec_12_{stage}.json
  RC, variant           <dx>/blp/blp_results_E{k}_spec_12_{stage}_dx.json and the sidecar
                        blp_meta_E{k}_spec_12_{stage}_dx.json (the SE guard's verdict)
  BBL, main             BBL_OUTPUT/cluster_processed/cost_params_E{k}_spec_12_extended{tag}.json
  BBL, variant          <dx>/bbl/cost_params_E{k}_spec_12_extended_dx{tag}.json
  where <dx> = ESTIMATION_OUTPUT/DX_VARIANT, the folder cluster_ingest_dx.py fills.

WHAT IT WRITES
  <dx>/tables/tab_dx_demand_robustness.{tex,md}, tab_dx_bbl_comparison.{tex,md}, dx_numbers.json.
  Every file name and every \\label carries `dx`; nothing else is written. `--to-drafts` also
  copies the two .tex files, under the same names, into the paper folder (it refuses a name that
  does not start with `tab_dx_`). The published tables (est*_logit*.tex, blp_*.tex,
  tab_bbl_*.tex) belong to other generators. This script calls none of them; from
  make_blp_rc_table it takes the label maps and the number formatting only.

A column whose source is not on disk yet prints a dash and is listed as pending; the script still
writes the table (today: the two logit columns and the main RC column).

Usage
-----
  python make_dx_tables.py                         # both tables, RC stage ext1, BBL tag _ms982
  python make_dx_tables.py --only demand --rows all
  python make_dx_tables.py --dx-dir <folder> --out-dir <folder>      # a sandbox
  python make_dx_tables.py --to-drafts
"""
from utils.venv_guard import ensure_project_venv
ensure_project_venv(__file__)

import argparse
import json
import math
import pathlib
import shutil
import sys

import make_blp_rc_table as rc   # label maps, decode_theta2, fmt_coef, fmt3, note_cell, est_ref
from utils import paths as _paths
from utils import routines as _routines

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

DX_SUFFIX   = "_dx"
DX_COL      = "dummy_D_type"
DX_LABEL    = "D Type"
SPEC        = 12
SE_STAGE    = "ext1"          # the stage whose RC standard errors the variant reports
BBL_STAGE   = "extended"
DEFAULT_TAG = "_ms982"
DEFAULT_ESTS = list(_routines.LINK_ESTS)

DX_DIR_DEFAULT = _paths.estimation_output() / "DX_VARIANT"
LOGIT_SUMMARY  = _paths.blp_results_dir() / "logit" / f"logit_summary_spec_{SPEC}.json"
MAIN_RC_DIR    = _paths.blp_results_dir() / "cluster_raw"
MAIN_BBL_DIR   = _paths.bbl_output_dir() / "cluster_processed"

DEMAND_BASE = "tab_dx_demand_robustness"
BBL_BASE    = "tab_dx_bbl_comparison"
THETA1_LABELS = dict(rc.THETA1_LABELS, **{DX_COL: DX_LABEL})
KEY_ROWS = ["alpha", DX_COL, "seg_S5"]


def _load(path: pathlib.Path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return d or None
    except (OSError, ValueError):
        return None


def _nan(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def _clean(o):
    """A JSON-ready copy: non-finite floats become null."""
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


# ══════════════════════════════════════════════════════════════════════════════════════════════
# Demand: logit and RC, main against D
# ══════════════════════════════════════════════════════════════════════════════════════════════
def load_logit(est: int, sub: str):
    """Entry E{est}_{sub} of the logit summary, in the RC result layout (theta1 / theta1_se /
    theta1_pval / param_names_theta1), or None."""
    d = _load(LOGIT_SUMMARY)
    e = (d or {}).get(f"E{est}_{sub}")
    if not e:
        return None
    return {"param_names_theta1": e.get("param_names", []), "theta1": e.get("theta1", []),
            "theta1_se": e.get("se", []), "theta1_pval": e.get("pval", []),
            "Q_value": e.get("Q_value"), "n_obs": e.get("n_obs"), "G_star": e.get("G_star"),
            "se_method": e.get("se_method"), "source": f"{LOGIT_SUMMARY.name}:E{est}_{sub}"}


def load_rc(est: int, stage: str, dx_dir: pathlib.Path, variant: bool):
    """The RC result JSON of a stage: the main one from cluster_raw, the variant's from <dx>/blp.
    The variant's entry also carries `se_guard`, the verdict in its sidecar (None when absent)."""
    if not variant:
        p = MAIN_RC_DIR / f"blp_results_E{est}_spec_{SPEC}_{stage}.json"
        d = _load(p)
        if d:
            d["source"] = p.name
        return d
    p = dx_dir / "blp" / f"blp_results_E{est}_spec_{SPEC}_{stage}{DX_SUFFIX}.json"
    d = _load(p)
    if not d:
        return None
    names = d.get("param_names_theta1") or []
    if len(names) != len(d.get("theta1", [])) or DX_COL not in names:
        raise SystemExit(f"[dx-tables] {p.name}: expected the variant's ten theta1 names with "
                         f"{DX_COL}; found {names}. Not a variant result.")
    meta = _load(dx_dir / "blp" / f"blp_meta_E{est}_spec_{SPEC}_{stage}{DX_SUFFIX}.json") or {}
    d["se_guard"] = ((meta.get("dx") or {}).get("se_guard"))
    d["source"] = p.name
    return d


def demand_columns(est: int, stage: str, dx_dir: pathlib.Path):
    """[(header, entry-or-None, is_rc)] for one routine."""
    return [("Logit",       load_logit(est, "full"),                  False),
            ("Logit + D",   load_logit(est, "full_dtype"),            False),
            ("RC",          load_rc(est, stage, dx_dir, False),       True),
            ("RC + D",      load_rc(est, stage, dx_dir, True),        True)]


def _coef(entry, name):
    """(value, se, pval) of theta1 parameter `name`, or None when the entry has no such row."""
    if not entry:
        return None
    names = entry.get("param_names_theta1") or []
    if name not in names:
        return None
    i = names.index(name)
    t1, se, pv = entry.get("theta1", []), entry.get("theta1_se", []), entry.get("theta1_pval", [])
    guard = entry.get("se_guard")
    no_se = bool(guard) and guard.get("ran") and not guard.get("ok")
    return (t1[i] if i < len(t1) else float("nan"),
            0.0 if no_se else (se[i] if i < len(se) else 0.0),
            None if no_se else (pv[i] if i < len(pv) else None))


def demand_numbers(ests, stage, dx_dir, rows):
    """Everything the demand table prints, as plain numbers: {E: {column: {...}}}."""
    out = {}
    for e in ests:
        cols = {}
        for hdr, entry, is_rc in demand_columns(e, stage, dx_dir):
            if not entry:
                cols[hdr] = None
                continue
            c = {"source": entry.get("source"), "Q": entry.get("Q_value"), "n_obs": entry.get("n_obs"),
                 "G_star": entry.get("G_star"), "theta1": {}, "theta2": []}
            for nm in entry.get("param_names_theta1") or []:
                v = _coef(entry, nm)
                c["theta1"][nm] = {"coef": v[0], "se": v[1] or None, "pval": v[2]}
            if is_rc:
                for lbl, v, se, pv in rc.decode_theta2(entry):
                    c["theta2"].append({"label": lbl, "coef": v, "se": se or None, "pval": pv})
                c["se_guard"] = entry.get("se_guard")
            rho = rc.mean_rho_one_minus_s(e)
            a = rc.alpha_of(entry)
            c["mean_own_price_elasticity"] = (a * rho) if (rho is not None and a is not None) else None
            cols[hdr] = c
        out[f"E{e}"] = cols
    return out


def build_demand_tex(ests, stage, dx_dir, rows):
    data = {e: demand_columns(e, stage, dx_dir) for e in ests}
    ncols = 4
    hdr = " & ".join(h for h, _, _ in data[ests[0]])
    pending = [f"E{e} {h}" for e in ests for h, d, _ in data[e] if not d]
    guard_bad = [rc.est_ref(e) for e in ests for h, d, r in data[e]
                 if d and r and (d.get("se_guard") or {}).get("ran") and not (d.get("se_guard") or {}).get("ok")]
    stage_lbl = rc.STAGE_LABELS.get(stage, stage)
    methods = {d.get("se_method") for e in ests for _, d, _ in data[e] if d and d.get("se_method")}
    se_txt = (r"Wild cluster bootstrap standard errors (conglomerate clusters) in parentheses. "
              if methods <= {"wcb"} else
              r"Cluster-robust standard errors (conglomerate clusters) in parentheses. ")
    note = (r"Main specification against the variant that adds a D-type dummy, equal to one for a "
            r"digital firm, to the mean utility. The dummy enters the linear part only: the random "
            r"coefficients, the simulation draws and the instruments are those of the main "
            r"specification, so each \textit{Logit + D} column is the $\theta_2 = 0$ case of the "
            r"\textit{RC + D} column beside it. The model has no constant, so the D-type coefficient "
            r"is the intercept of digital firms and brick-and-mortar firms have none. Segment S5 "
            r"contains digital firms only. "
            + (r"The remaining firm characteristics are included and not reported. " if rows != "all" else "")
            + (r"RC columns are the specification of Table~\ref{tab:blp_demand_comparison_noseg_spec12}. "
               if stage == SE_STAGE else rf"RC columns report the {stage_lbl} specification. ")
            + se_txt +
            r"$^{\dagger}$ on the bound $\Sigma \geq 0$; the other standard errors of that column "
            r"are conditional on it. $Q$ is the one-step GMM criterion."
            + (r" No standard errors for RC + D in " + ", ".join(guard_bad) +
               r": the parameters outnumber the moments at this stage." if guard_bad else "")
            + r" $^{*}p<0.10$, $^{**}p<0.05$, $^{***}p<0.01$.")
    W = r"\textwidth"
    lines = [
        r"\begin{spacing}{1.0}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\setlength{\LTleft}{\fill}",
        r"\setlength{\LTright}{\fill}",
        rf"\begin{{xltabular}}{{{W}}}{{>{{\raggedright\arraybackslash}}p{{5.2cm}} "
        rf"*{{{ncols}}}{{>{{\centering\arraybackslash}}X}}}}",
        r"    \caption{Demand Estimation with a D-Type Dummy}",
        rf"    \label{{tab:dx_demand_robustness}} \\",
        r"    \toprule",
        rf"     & {hdr} \\",
        r"    \midrule",
        r"    \endfirsthead",
        "",
        rf"    \multicolumn{{{ncols + 1}}}{{c}}{{\bfseries Table \thetable\ continued}} \\",
        r"    \toprule",
        rf"     & {hdr} \\",
        r"    \midrule",
        r"    \endhead",
        "",
        r"    \midrule",
        rf"    \multicolumn{{{ncols + 1}}}{{r}}{{\textit{{Continued on next page}}}} \\",
        r"    \endfoot",
        "",
        r"    \bottomrule",
        r"    \multicolumn{" + str(ncols + 1) + r"}{@{}p{\dimexpr" + W + r"-2\tabcolsep\relax}@{}}{"
        + rc.note_cell(note) + r"} \\",
        r"    \endlastfoot",
        "",
    ]
    for pi, e in enumerate(ests):
        cols = data[e]
        if pi:
            lines.append(r"    \midrule")
        lines.append(r"    \multicolumn{" + str(ncols + 1) + r"}{l}{\textit{Panel " + "ABCDEFGH"[pi] +
                     r": Estimation " + rc.est_ref(e) + r"}} \\*")
        lines.append(r"    \addlinespace[0.3ex]")
        ref_names = next((d.get("param_names_theta1") for _, d, _ in reversed(cols) if d), [])
        names = [n for n in ref_names] if rows == "all" else [n for n in KEY_ROWS]
        if rows == "all" and DX_COL not in names:
            names.append(DX_COL)
        for nm in names:
            cv, sv = [THETA1_LABELS.get(nm, nm.replace("_", r"\_"))], [""]
            for _, d, _ in cols:
                v = _coef(d, nm)
                if not d:
                    c, s_ = "---", ""
                elif v is None:
                    c, s_ = "", ""
                else:
                    c, s_ = rc.fmt_coef(v[0], v[1], d.get("G_star"), v[2])
                cv.append(c); sv.append(s_)
            lines.append("    " + " & ".join(cv) + r" \\*")
            if any(s.strip() for s in sv[1:]):
                lines.append("    " + " & ".join(sv) + r" \\*")
            lines.append(r"    \addlinespace[0.15ex]")
        labels = []
        for _, d, is_rc in cols:
            if d and is_rc:
                for lbl, *_ in rc.decode_theta2(d):
                    if lbl not in labels:
                        labels.append(lbl)
        for lbl in labels:
            cv, sv = [lbl], [""]
            for _, d, is_rc in cols:
                if not is_rc:
                    c, s_ = "", ""
                elif not d:
                    c, s_ = "---", ""
                else:
                    dec = {l: (v, se, pv) for l, v, se, pv in rc.decode_theta2(d)}
                    if lbl not in dec:
                        c, s_ = "", ""
                    else:
                        v, se, pv = dec[lbl]
                        guard = d.get("se_guard") or {}
                        if guard.get("ran") and not guard.get("ok"):
                            se, pv = 0.0, None
                        ob = lbl.startswith(r"$\Sigma$") and not _nan(v) and abs(v) < rc.SIGMA_BOUND_TOL
                        c, s_ = rc.fmt_coef(v, se, d.get("G_star"), pv, on_bound=ob)
                cv.append(c); sv.append(s_)
            lines.append("    " + " & ".join(cv) + r" \\*")
            if any(s.strip() for s in sv[1:]):
                lines.append("    " + " & ".join(sv) + r" \\*")
            lines.append(r"    \addlinespace[0.15ex]")

        def stat(name, fn):
            return "    " + name + " & " + " & ".join((fn(d) if d else "---") for _, d, _ in cols) + r" \\*"
        lines += [
            r"    \addlinespace[0.3ex]",
            stat(rc.ELAST_ROW_LABEL, lambda d: rc.semi_elast_cell(d, e)),
            stat(r"Observations", lambda d: f"{d['n_obs']:,}" if d.get("n_obs") is not None else "---"),
            stat(rc.Q_ROW_LABEL, lambda d: f"${rc.fmt3(d['Q_value'])}$" if d.get("Q_value") is not None else "---"),
            stat(rc.GSTAR_ROW_LABEL, lambda d: rc.fmt3(d["G_star"]) if d.get("G_star") is not None else "---"),
        ]
    lines += [r"\end{xltabular}", r"\end{spacing}"]
    return "\n".join(lines), pending


def build_demand_md(nums, rows):
    L = ["# Demand estimation with a D-type dummy (main vs dx)", ""]
    for ek, cols in nums.items():
        hdrs = list(cols)
        L += [f"## {ek}", "", "| | " + " | ".join(hdrs) + " |", "|:--" + "|--:" * len(hdrs) + "|"]
        names = []
        for h in hdrs:
            for nm in ((cols[h] or {}).get("theta1") or {}):
                if nm not in names:
                    names.append(nm)
        if rows != "all":
            names = [n for n in KEY_ROWS if n in names]

        def cell(c, nm):
            if c is None:
                return "pending"
            v = c["theta1"].get(nm)
            if v is None:
                return ""
            return rc.fmt3(v["coef"]) + (f" ({rc.fmt3(v['se'])})" if v["se"] else "")
        for nm in names:
            L.append(f"| {nm} | " + " | ".join(cell(cols[h], nm) for h in hdrs) + " |")
        labels = []
        for h in hdrs:
            for t in ((cols[h] or {}).get("theta2") or []):
                if t["label"] not in labels:
                    labels.append(t["label"])
        for lbl in labels:
            row = []
            for h in hdrs:
                c = cols[h]
                t = next((t for t in ((c or {}).get("theta2") or []) if t["label"] == lbl), None)
                row.append("pending" if c is None and h.startswith("RC") else "" if t is None else
                           rc.fmt3(t["coef"]) + (f" ({rc.fmt3(t['se'])})" if t["se"] else ""))
            L.append(f"| {lbl} | " + " | ".join(row) + " |")
        for key, lab, fmt in (("mean_own_price_elasticity", "Mean own-price elasticity", rc.fmt3),
                              ("n_obs", "Observations", lambda v: f"{v:,}"),
                              ("Q", "Q", rc.fmt3), ("G_star", "G*", rc.fmt3)):
            L.append(f"| {lab} | " + " | ".join(
                "pending" if cols[h] is None else (fmt(cols[h][key]) if cols[h].get(key) is not None else "")
                for h in hdrs) + " |")
        L.append("")
    return "\n".join(L)


# ══════════════════════════════════════════════════════════════════════════════════════════════
# BBL: main against D
# ══════════════════════════════════════════════════════════════════════════════════════════════
def _ann(x) -> float:
    """Quarterly rate -> compounded annual pp."""
    return ((1.0 + float(x)) ** 4 - 1.0) * 100.0


def load_cost(est: int, tag: str, dx_dir: pathlib.Path, variant: bool):
    if variant:
        p = dx_dir / "bbl" / f"cost_params_E{est}_spec_{SPEC}_{BBL_STAGE}{DX_SUFFIX}{tag}.json"
    else:
        p = MAIN_BBL_DIR / f"cost_params_E{est}_spec_{SPEC}_{BBL_STAGE}{tag}.json"
    d = _load(p)
    if d:
        want = f"E{est}_spec_{SPEC}_{BBL_STAGE}{DX_SUFFIX if variant else ''}{tag}"
        got = (d.get("run") or {}).get("tag")
        if got is not None and got != want:
            raise SystemExit(f"[dx-tables] {p.name}: the solve recorded tag '{got}', expected '{want}'.")
        d["source"] = p.name
    return d


def bbl_block(d, kappa):
    """The numbers the comparison shows for one firm type of one cost_params file."""
    if not d or kappa not in d:
        return None
    b = d[kappa]
    ci = ((b.get("c_bar_ci_sqrtn") or {}).get("0.95") or {})

    def prof(key):
        c = b.get(key) or {}
        if c.get("empty") or _nan(c.get("ci_lo")) or _nan(c.get("ci_hi")):
            return None
        return [c["ci_lo"], c["ci_hi"]]
    return {"cbar_q": b.get("c_bar"), "cbar_ann": None if _nan(b.get("c_bar")) else _ann(b["c_bar"]),
            "cbar_ann_ci": None if (_nan(ci.get("lo")) or _nan(ci.get("hi"))) else [_ann(ci["lo"]), _ann(ci["hi"])],
            "omega": b.get("omega"), "omega_ci": prof("ci_omega"),
            "zeta": b.get("zeta"), "zeta_ci": prof("ci_zeta"),
            "identified": bool(b.get("identified_split")),
            "violated_share": b.get("frac_bind"), "n_firms": b.get("n_firms"), "n_rows": b.get("n_rows"),
            "source": d.get("source")}


def bbl_numbers(ests, tag, dx_dir):
    out = {}
    for e in ests:
        m, v = load_cost(e, tag, dx_dir, False), load_cost(e, tag, dx_dir, True)
        out[f"E{e}"] = {k: {"main": bbl_block(m, k), "dx": bbl_block(v, k)} for k in ("B", "D")}
    return out


def build_bbl_tex(ests, tag, nums):
    cols = [(e, w) for e in ests for w in ("main", "dx")]
    ncols = len(cols)
    grp = " & ".join(rf"\multicolumn{{2}}{{c}}{{{rc.est_ref(e)}}}" for e in ests)
    cmid = " ".join(rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(ests)))
    hdr = " & ".join("Main" if w == "main" else "D" for _, w in cols)
    pending = sorted({f"E{e} {w}" for e, w in cols for k in ("B", "D") if nums[f"E{e}"][k][w] is None})
    unid = any((nums[f"E{e}"][k][w] or {}).get("identified") is False for e, w in cols for k in ("B", "D"))
    note = (r"The BBL cost estimation on the main demand fit (\textit{Main}) and on the fit with the "
            r"D-type dummy in the mean utility (\textit{D}); same forward simulation, policy "
            r"function and launch quarters. The demand fit enters through the mean utilities, the "
            r"price coefficient and the random-coefficient parameters only. "
            r"$\bar{c}$ in compounded annual pp, $((1+x)^4-1)\times100$, with its 95\% subsampling "
            r"interval in brackets; $\omega$ and $\zeta$ with 95\% profile intervals. Violated share "
            r"is the fraction of deviation inequalities violated at the estimate."
            + (r" $^{\ddagger}$ $\omega$ and $\zeta$ are not separately identified in that cell." if unid else ""))
    W = r"\textwidth"
    lines = [
        r"\begin{spacing}{1.0}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\setlength{\LTleft}{\fill}",
        r"\setlength{\LTright}{\fill}",
        rf"\begin{{xltabular}}{{{W}}}{{>{{\raggedright\arraybackslash}}p{{4.2cm}} "
        rf"*{{{ncols}}}{{>{{\centering\arraybackslash}}X}}}}",
        r"    \caption{Cost Estimates under the Demand Fit with a D-Type Dummy}",
        r"    \label{tab:dx_bbl_comparison} \\",
        r"    \toprule",
        rf"     & {grp} \\",
        rf"    {cmid}",
        rf"     & {hdr} \\",
        r"    \midrule",
        r"    \endfirsthead",
        "",
        rf"    \multicolumn{{{ncols + 1}}}{{c}}{{\bfseries Table \thetable\ continued}} \\",
        r"    \toprule",
        rf"     & {grp} \\",
        rf"    {cmid}",
        rf"     & {hdr} \\",
        r"    \midrule",
        r"    \endhead",
        "",
        r"    \midrule",
        rf"    \multicolumn{{{ncols + 1}}}{{r}}{{\textit{{Continued on next page}}}} \\",
        r"    \endfoot",
        "",
        r"    \bottomrule",
        r"    \multicolumn{" + str(ncols + 1) + r"}{@{}p{\dimexpr" + W + r"-2\tabcolsep\relax}@{}}{"
        + rc.note_cell(note) + r"} \\",
        r"    \endlastfoot",
        "",
    ]

    def cells(kappa, fn):
        return [("---" if nums[f"E{e}"][kappa][w] is None else fn(nums[f"E{e}"][kappa][w])) for e, w in cols]

    def num(key, dag=False):
        def f(b):
            if _nan(b.get(key)):
                return "---"
            return f"${rc.fmt3(b[key])}" + (r"^{\ddagger}" if dag and not b["identified"] else "") + "$"
        return f

    def ci(key):
        def f(b):
            c = b.get(key)
            return "" if not c else f"$[{rc.fmt3(c[0])},\\ {rc.fmt3(c[1])}]$"
        return f

    def count(key):
        return lambda b: "---" if b.get(key) is None else f"{int(b[key]):,}"
    for pi, (kappa, title) in enumerate((("B", "Brick-and-mortar firms (B)"), ("D", "Digital firms (D)"))):
        if pi:
            lines.append(r"    \midrule")
        lines.append(r"    \multicolumn{" + str(ncols + 1) + r"}{l}{\textit{Panel " + "AB"[pi] + ": " + title + r"}} \\*")
        lines.append(r"    \addlinespace[0.3ex]")
        for label, fn, cifn in ((r"$\bar{c}$ (annual pp)", num("cbar_ann"), ci("cbar_ann_ci")),
                                (r"$\omega$", num("omega", dag=True), ci("omega_ci")),
                                (r"$\zeta$", num("zeta", dag=True), ci("zeta_ci"))):
            lines.append("    " + " & ".join([label] + cells(kappa, fn)) + r" \\*")
            c2 = cells(kappa, cifn)
            if any(x.strip() and x != "---" for x in c2):
                lines.append("    " + " & ".join([""] + [("" if x == "---" else x) for x in c2]) + r" \\*")
            lines.append(r"    \addlinespace[0.15ex]")
        lines.append("    " + " & ".join(["Violated share"] + cells(kappa, num("violated_share"))) + r" \\*")
        lines.append("    " + " & ".join(["Firms"] + cells(kappa, count("n_firms"))) + r" \\*")
        lines.append("    " + " & ".join(["Observations"] + cells(kappa, count("n_rows"))) + r" \\*")
    lines += [r"\end{xltabular}", r"\end{spacing}"]
    return "\n".join(lines), pending


def build_bbl_md(nums, tag):
    L = [f"# BBL cost estimates, main ({tag}) vs dx ({DX_SUFFIX}{tag})", ""]
    for ek, blocks in nums.items():
        for kappa in ("B", "D"):
            m, v = blocks[kappa]["main"], blocks[kappa]["dx"]
            L += [f"## {ek}, {kappa} firms", "", "| | Main | D |", "|:--|--:|--:|"]

            def cell(b, key, cikey=None, integer=False):
                if b is None:
                    return "pending"
                if _nan(b.get(key)):
                    return ""
                s = f"{int(b[key]):,}" if integer else rc.fmt3(b[key])
                if cikey and b.get(cikey):
                    s += f" [{rc.fmt3(b[cikey][0])}, {rc.fmt3(b[cikey][1])}]"
                return s
            for lab, key, cik, integer in (("c-bar (annual pp)", "cbar_ann", "cbar_ann_ci", False),
                                           ("omega", "omega", "omega_ci", False),
                                           ("zeta", "zeta", "zeta_ci", False),
                                           ("Violated share", "violated_share", None, False),
                                           ("Firms", "n_firms", None, True),
                                           ("Observations", "n_rows", None, True)):
                L.append(f"| {lab} | {cell(m, key, cik, integer)} | {cell(v, key, cik, integer)} |")
            L.append("")
    return "\n".join(L)


# ══════════════════════════════════════════════════════════════════════════════════════════════
def _write(path: pathlib.Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text + "\n")
    print(f"  saved: {path}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Tables of the dx demand variant (main vs D).")
    ap.add_argument("--routines", default=_routines.csv(DEFAULT_ESTS))
    ap.add_argument("--only", choices=["demand", "bbl"], default=None)
    ap.add_argument("--stage", default=SE_STAGE, help="RC stage of the demand table (default ext1)")
    ap.add_argument("--rows", choices=["key", "all"], default="key",
                    help="mean-utility rows of the demand table: the price, D-type and S5 "
                         "coefficients (key) or every coefficient (all)")
    ap.add_argument("--psi-tag", default=DEFAULT_TAG, help="psi tag of the MAIN BBL run; the "
                    "variant's is _dx + this")
    ap.add_argument("--dx-dir", default=str(DX_DIR_DEFAULT), help="the variant's folder")
    ap.add_argument("--out-dir", default=None, help="default: <dx-dir>/tables")
    ap.add_argument("--to-drafts", action="store_true",
                    help="also copy the two .tex files into the paper folder")
    a = ap.parse_args(argv)
    ests = [int(x) for x in a.routines.split(",") if x.strip()]
    dx_dir = pathlib.Path(a.dx_dir)
    out = pathlib.Path(a.out_dir) if a.out_dir else dx_dir / "tables"
    numbers = {"generator": "make_dx_tables.py", "variant": "dx", "suffix": DX_SUFFIX,
               "rc_stage": a.stage, "psi_tag_main": a.psi_tag, "psi_tag_dx": DX_SUFFIX + a.psi_tag}
    written = []
    if a.only in (None, "demand"):
        nums = demand_numbers(ests, a.stage, dx_dir, a.rows)
        tex, pending = build_demand_tex(ests, a.stage, dx_dir, a.rows)
        _write(out / f"{DEMAND_BASE}.tex", tex); written.append(out / f"{DEMAND_BASE}.tex")
        _write(out / f"{DEMAND_BASE}.md", build_demand_md(nums, a.rows))
        numbers["demand"] = nums
        numbers["demand_pending"] = pending
        print("  demand table: " + ("every column on disk" if not pending else "PENDING columns: " + ", ".join(pending)))
    if a.only in (None, "bbl"):
        nums = bbl_numbers(ests, a.psi_tag, dx_dir)
        tex, pending = build_bbl_tex(ests, a.psi_tag, nums)
        _write(out / f"{BBL_BASE}.tex", tex); written.append(out / f"{BBL_BASE}.tex")
        _write(out / f"{BBL_BASE}.md", build_bbl_md(nums, a.psi_tag))
        numbers["bbl"] = nums
        numbers["bbl_pending"] = pending
        print("  BBL table: " + ("every column on disk" if not pending else "PENDING columns: " + ", ".join(pending)))
    _write(out / "dx_numbers.json", json.dumps(_clean(numbers), indent=1, allow_nan=False))
    if a.to_drafts:
        dest = _paths.drafts_dir()
        if not dest.is_dir():
            raise SystemExit(f"[dx-tables] --to-drafts: {dest} does not exist")
        for p in written:
            if not p.name.startswith("tab_dx_"):
                raise SystemExit(f"[dx-tables] refusing to copy {p.name}: not a variant table name")
            shutil.copyfile(p, dest / p.name)
            print(f"  copied: {dest / p.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
