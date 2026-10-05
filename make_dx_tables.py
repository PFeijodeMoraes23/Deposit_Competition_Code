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

CHECKS BEFORE A NUMBER IS PRINTED
  RC + D standard errors   printed only when the variant's result carries JOINT standard errors
                           (sidecar `se_joint`, and the result itself: se_method wcb | sandwich
                           with p-values). Otherwise the column shows point estimates alone and
                           the note gives the reason (parameters outnumber the moments; singular
                           Jacobian; the joint computation did not complete).
  main RC column           when the cluster's main-vs-dx record blp_compare_E{k}_spec_12_dx.json
                           has been ingested, the local main result must be the fit the variant
                           was compared with (alpha, Q, theta2). A mismatch stops the script;
                           `--allow-main-mismatch` prints the table with a sentence saying so.
  BBL, Main against D      the two cost_params files must record the same design: beta, T, launch
                           quarters, shard count, forward curves, the three model switches, the
                           simulation version and the sha256 of the policy, the transitions and
                           the sleep link. A difference stops the script;
                           `--allow-design-mismatch` prints the table with the differing items in
                           the note. Items one run did not record are listed, and the note then
                           no longer says the simulation is the same.
  cost_params against fit  the suite records the sha256 of the fit each BBL run of the variant was
                           simulated on (<dx>/blp/blp_bblfit_E{k}_spec_12_extended_dx{tag}.json,
                           written at the BBL launch). When the variant's cost_params are on disk,
                           the variant's `extended` fit on disk must be that file. A difference
                           stops the script before any table is written, demand or BBL: the two
                           tables would pair demand estimates from one fit with cost estimates
                           from another. There is no override; re-ingest the two archives of one
                           run, or move the cost_params out of <dx>/bbl. Without a record, or
                           without the fit on disk, the console says "not checked".
                           That ties the fit to the record. The record is tied to the cost_params
                           through the job id of the BBL archive: the record lists it, and
                           cluster_ingest_dx.py notes beside the cost_params the archive it landed
                           them from (cost_params_<key>.ingest.json). A cost_params file from an
                           archive the record does not list stops the script the same way. Only
                           when both links hold does the console say "pairing E{k}: bound". When
                           the second cannot be told (no archive job in the record, cost_params
                           not landed by the ingest, a renamed archive) it says "pairing E{k}: fit
                           and record bound; cost params not verifiable", and the BBL table's note
                           carries the same words; "not checked" goes into the note as well.

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
import hashlib
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


# Why a variant result has no joint standard errors, as the note words it: the sidecar's
# `se_status` (blp_dx.jl `dx_stamp!`) starts with one of these.
SE_REASONS = (("guard: K > L", "the parameters outnumber the moments at this stage"),
              ("guard: singular", "the moment Jacobian is numerically singular at this stage"),
              ("failed", "the joint standard-error computation did not complete"),
              ("not requested", "no standard errors were requested"))


def se_state(d: dict, meta: dict):
    """(ok, reason) for a variant RC result: whether its standard errors are the joint GMM ones.

    The sidecar's `se_joint` decides when it is there; a sidecar written before that field
    existed is read through its guard verdict. Either way the result itself must agree: a joint
    method in `se_method` and one p-value per theta1 (the engine's failure branch leaves the
    p-values empty, and a result without joint SEs is stamped se_method = none)."""
    dxr = meta.get("dx") or {}
    guard = dxr.get("se_guard") or {}
    tripped = bool(guard.get("ran")) and not guard.get("ok")
    pv = d.get("theta1_pval") or []
    in_result = d.get("se_method") in ("wcb", "sandwich") and len(pv) == len(d.get("theta1", []))
    if "se_joint" in dxr:
        joint = bool(dxr["se_joint"]) and in_result
        status = str(dxr.get("se_status") or "")
        if dxr["se_joint"] and not in_result:
            status = "failed"
    else:
        joint = in_result and not tripped
        status = ("guard: " + str(guard.get("reason", ""))) if tripped else ("" if joint else "failed")
    if joint:
        return True, ""
    for key, txt in SE_REASONS:
        if status.startswith(key):
            return False, txt
    return False, "no joint standard errors were recorded"


def load_rc(est: int, stage: str, dx_dir: pathlib.Path, variant: bool):
    """The RC result JSON of a stage: the main one from cluster_raw, the variant's from <dx>/blp.
    The variant's entry also carries `se_ok` / `se_reason` (`se_state`) and `se_guard`, the
    guard's verdict in its sidecar (None when absent)."""
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
    d["se_ok"], d["se_reason"] = se_state(d, meta)
    d["source"] = p.name
    return d


def main_rc_check(est: int, stage: str, dx_dir: pathlib.Path, tol: float = 1e-6):
    """The local main RC result against the main fit the cluster compared the variant with
    (blp_compare_E{est}_spec_12_dx.json, written by blp_dx_compare.jl at the hand-off).
    -> {"status": "match" | "MISMATCH" | <why it could not be checked>, "differences": [...]}.
    The variant's own alpha is checked the same way against the comparison's `alpha_dx`."""
    cmp_ = _load(dx_dir / "blp" / f"blp_compare_E{est}_spec_{SPEC}{DX_SUFFIX}.json")
    if not cmp_:
        return {"status": "not checked: no cluster comparison on disk", "differences": []}
    st = next((x for x in cmp_.get("stages", []) if x.get("stage") == stage), None)
    if st is None:
        return {"status": f"not checked: the cluster comparison has no stage {stage}", "differences": []}
    loc = load_rc(est, stage, dx_dir, False)
    if not loc:
        return {"status": "not checked: no local main result", "differences": []}

    def close(a, b):
        return (not _nan(a)) and (not _nan(b)) and abs(a - b) <= tol * max(1.0, abs(a), abs(b))
    diffs = []
    if not close(rc.alpha_of(loc), st.get("alpha_main")):
        diffs.append(f"alpha: local {rc.alpha_of(loc)}, cluster {st.get('alpha_main')}")
    if not close(loc.get("Q_value"), st.get("Q_main")):
        diffs.append(f"Q: local {loc.get('Q_value')}, cluster {st.get('Q_main')}")
    t2l, t2c = loc.get("theta2") or [], st.get("theta2_main") or []
    if len(t2l) != len(t2c) or any(not close(x, y) for x, y in zip(t2l, t2c)):
        diffs.append(f"theta2: local {t2l}, cluster {t2c}")
    var = load_rc(est, stage, dx_dir, True)
    if var and not close(rc.alpha_of(var), st.get("alpha_dx")):
        diffs.append(f"alpha of the variant: ingested {rc.alpha_of(var)}, cluster comparison {st.get('alpha_dx')}")
    return {"status": "MISMATCH" if diffs else "match", "differences": diffs,
            "cluster_main_file": st.get("main_file"), "local_main_file": loc.get("source")}


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
    no_se = entry.get("se_ok") is False
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
                no_se = entry.get("se_ok") is False
                for lbl, v, se, pv in rc.decode_theta2(entry):
                    c["theta2"].append({"label": lbl, "coef": v, "se": None if no_se else (se or None),
                                        "pval": None if no_se else pv})
                c["se_guard"] = entry.get("se_guard")
                c["se_joint"] = entry.get("se_ok", True)
                c["se_reason"] = entry.get("se_reason", "")
            rho = rc.mean_rho_one_minus_s(e)
            a = rc.alpha_of(entry)
            c["mean_own_price_elasticity"] = (a * rho) if (rho is not None and a is not None) else None
            cols[hdr] = c
        out[f"E{e}"] = cols
    return out


def build_demand_tex(ests, stage, dx_dir, rows, main_mismatch=()):
    data = {e: demand_columns(e, stage, dx_dir) for e in ests}
    ncols = 4
    hdr = " & ".join(h for h, _, _ in data[ests[0]])
    pending = [f"E{e} {h}" for e in ests for h, d, _ in data[e] if not d]
    no_se = {}
    for e in ests:
        for h, d, r in data[e]:
            if d and r and d.get("se_ok") is False:
                no_se.setdefault(d.get("se_reason") or "no joint standard errors were recorded", []).append(rc.est_ref(e))
    no_se_txt = "".join(" No standard errors for RC + D in " + ", ".join(v) + ": " + k + "." for k, v in no_se.items())
    if main_mismatch:
        no_se_txt += (" The RC column of " + ", ".join(rc.est_ref(e) for e in main_mismatch) + " is the local copy of "
                      "the main fit, which is not the fit the variant was compared with on the cluster.")
    stage_lbl = rc.STAGE_LABELS.get(stage, stage)
    # The sentence describes the columns that print standard errors. A variant column without
    # joint ones prints none (its result says se_method "none") and has its own sentence above.
    methods = {d.get("se_method") for e in ests for _, d, _ in data[e]
               if d and d.get("se_ok") is not False and d.get("se_method") not in (None, "", "none")}
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
            + no_se_txt
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
                        if d.get("se_ok") is False:
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


# What "the same forward simulation" means, read from each solve's own record (bbl_solve.py
# `run`): these items must agree between the main and the variant cost_params.
DESIGN_RUN_KEYS = ("beta", "T", "starts", "n_shards_expected", "rf_sources")
DESIGN_SIM_KEYS = ("phi_path", "z_path", "rdep_timing", "sim_version", "spread_units", "phi_d_rows",
                   "sleep_link_sha256", "state_transitions_sha256", "policy_csv_sha256")


def design_of(d: dict) -> dict:
    run = (d or {}).get("run") or {}
    sim = run.get("sim_paths") or {}
    out = {k: run.get(k) for k in DESIGN_RUN_KEYS}
    out.update({f"sim_paths.{k}": sim.get(k) for k in DESIGN_SIM_KEYS})
    return out


def design_check(m: dict, v: dict) -> dict:
    """Main against variant cost_params: which design items agree, differ, or were not recorded
    by one of the two runs. status: pending | same | same where recorded | DIFFERENT."""
    if not m or not v:
        return {"status": "pending", "equal": [], "differ": {}, "not_recorded": []}
    dm, dv = design_of(m), design_of(v)
    both = [k for k in dm if dm[k] is not None and dv[k] is not None]
    differ = {k: {"main": dm[k], "dx": dv[k]} for k in both if dm[k] != dv[k]}
    missing = [k for k in dm if k not in both]
    status = "DIFFERENT" if differ else ("same" if not missing else "same where recorded")
    return {"status": status, "equal": [k for k in both if k not in differ], "differ": differ,
            "not_recorded": missing}


def bbl_design(ests, tag, dx_dir):
    return {f"E{e}": design_check(load_cost(e, tag, dx_dir, False), load_cost(e, tag, dx_dir, True)) for e in ests}


def _sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


# What the pairing check reports when the first link holds and the second cannot be told.
UNVERIFIABLE = "fit and record bound; cost params not verifiable"


def fit_binding_check(est: int, tag: str, dx_dir: pathlib.Path) -> dict:
    """The variant's cost_params against the variant's demand fit on disk, in two links.

    FIT <-> RECORD. dx_suite_20261001.sh records the sha256 of the fit a BBL run was simulated on
    when it submits the launch (blp_bblfit_E{est}_spec_12_extended_dx{tag}.json in the blp folder:
    the `.jls` the forward simulation reads, and the engine's `.json` of the same stage). Every
    recorded file that is on disk here must have the recorded sha256.

    RECORD <-> COST PARAMS. Nothing inside a BBL archive says which launch it comes from (the cost
    params and the psi files record no job id, launch time or fit). The tie is the job id of the
    archive job, which is in the archive's name: the suite lists it in the record
    (`bbl_archive_jobs`), and cluster_ingest_dx.py notes beside each cost_params file it lands the
    archive it took it from (cost_params_<key>.ingest.json, with the file's sha256). The cost params
    on disk must be the file the ingest landed, from an archive the record lists.

    -> {"status": ..., "differences": [...], "why": <for the unverifiable case>}
       pending                 no variant cost_params on disk, so nothing is paired
       bound                   both links hold
       UNVERIFIABLE            the first link holds, the second cannot be told (see "why")
       not checked: <why>      no record on disk, or the recorded fit is not on disk
       MISMATCH                a link is broken: the tables are not written"""
    key = f"E{est}_spec_{SPEC}_{BBL_STAGE}{DX_SUFFIX}{tag}"
    cp = dx_dir / "bbl" / f"cost_params_{key}.json"
    if not cp.is_file():
        return {"status": "pending", "differences": []}
    rec_p = dx_dir / "blp" / f"blp_bblfit_{key}.json"
    rec = _load(rec_p)
    if not rec:
        return {"status": f"not checked: no fit record {rec_p.name} on disk", "differences": []}
    if rec.get("bbl_key") != key:
        return {"status": "MISMATCH", "record": rec_p.name,
                "differences": [f"{rec_p.name} is the record of '{rec.get('bbl_key')}', not of {key}"]}
    checked, diffs = [], []
    for name_key, sha_key in (("fit", "fit_sha256"), ("fit_json", "fit_json_sha256")):
        name, want = rec.get(name_key), rec.get(sha_key)
        if not name or not want:
            continue
        p = dx_dir / "blp" / name
        if not p.is_file():
            continue
        got = _sha256(p)
        checked.append(name)
        if got != want:
            diffs.append(f"{name}: sha256 {got[:12]}... on disk, the BBL was simulated on {want[:12]}... "
                         f"(recorded {rec.get('recorded')})")
    if diffs:
        return {"status": "MISMATCH", "record": rec_p.name, "checked": checked, "differences": diffs}
    if not checked:
        return {"status": f"not checked: the recorded fit ({rec.get('fit')}) is not on disk", "differences": []}
    out = {"record": rec_p.name, "checked": checked, "recorded": rec.get("recorded"), "differences": []}
    jobs = str(rec.get("bbl_archive_jobs") or "").split()
    src_p = cp.with_name(f"cost_params_{key}.ingest.json")
    src = _load(src_p)
    why = None
    if not jobs:
        why = f"{rec_p.name} lists no BBL archive job"
    elif not src:
        why = f"no {src_p.name} beside the cost params (they were not landed by cluster_ingest_dx.py)"
    elif src.get("sha256") != _sha256(cp):
        why = f"{cp.name} is not the file cluster_ingest_dx.py landed (it changed since)"
    elif not src.get("archive_job"):
        why = f"the archive {src.get('archive')} carries no job id in its name"
    elif str(src["archive_job"]) not in jobs:
        return dict(out, status="MISMATCH", differences=[
            f"{cp.name} came from the BBL archive of job {src['archive_job']} ({src.get('archive')}); {rec_p.name} "
            f"lists the archive job(s) {', '.join(jobs)} for the fit it records: another launch"])
    if why:
        return dict(out, status=UNVERIFIABLE, why=why)
    return dict(out, status="bound", archive_job=str(src["archive_job"]))


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


def pairing_note(ests, pairing) -> str:
    """The sentence(s) the BBL table's note carries when the pairing of column D with the demand
    fit is anything short of bound: the status up to its first colon, per group of routines."""
    groups = {}
    for e in ests:
        st = ((pairing or {}).get(f"E{e}") or {}).get("status")
        if st and st not in ("bound", "pending"):
            groups.setdefault(st.split(":")[0], []).append(rc.est_ref(e))
    return "".join(rf" Pairing of column \textit{{D}} with the demand fit, {', '.join(refs)}: {st}."
                   for st, refs in groups.items())


def build_bbl_tex(ests, tag, nums, design=None, pairing=None):
    cols = [(e, w) for e in ests for w in ("main", "dx")]
    ncols = len(cols)
    grp = " & ".join(rf"\multicolumn{{2}}{{c}}{{{rc.est_ref(e)}}}" for e in ests)
    cmid = " ".join(rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(ests)))
    hdr = " & ".join("Main" if w == "main" else "D" for _, w in cols)
    pending = sorted({f"E{e} {w}" for e, w in cols for k in ("B", "D") if nums[f"E{e}"][k][w] is None})
    unid = any((nums[f"E{e}"][k][w] or {}).get("identified") is False for e, w in cols for k in ("B", "D"))
    design = design or {}
    checked = [x for x in design.values() if x.get("status") != "pending"]
    all_same = bool(checked) and all(x["status"] == "same" for x in checked)
    differ = sorted({k for x in checked for k in x.get("differ", {})})
    same_txt = "; same forward simulation, policy function and launch quarters" if (all_same or not checked) else ""
    differ_txt = ((" The two runs differ in their recorded design: " + ", ".join(k.replace("_", r"\_") for k in differ) + ".")
                  if differ else "")
    note = (r"The BBL cost estimation on the main demand fit (\textit{Main}) and on the fit with the "
            r"D-type dummy in the mean utility (\textit{D})" + same_txt + ". "
            r"The demand fit enters through the mean utilities, the "
            r"price coefficient and the random-coefficient parameters only. "
            r"$\bar{c}$ in compounded annual pp, $((1+x)^4-1)\times100$, with its 95\% subsampling "
            r"interval in brackets; $\omega$ and $\zeta$ with 95\% profile intervals. Violated share "
            r"is the fraction of deviation inequalities violated at the estimate."
            + differ_txt
            + pairing_note(ests, pairing)
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
    ap.add_argument("--stage", default=SE_STAGE, choices=list(rc.STAGES),
                    help="RC stage of the demand table (default ext1)")
    ap.add_argument("--rows", choices=["key", "all"], default="key",
                    help="mean-utility rows of the demand table: the price, D-type and S5 "
                         "coefficients (key) or every coefficient (all)")
    ap.add_argument("--psi-tag", default=DEFAULT_TAG, help="psi tag of the MAIN BBL run; the "
                    "variant's is _dx + this")
    ap.add_argument("--dx-dir", default=str(DX_DIR_DEFAULT), help="the variant's folder")
    ap.add_argument("--out-dir", default=None, help="default: <dx-dir>/tables")
    ap.add_argument("--to-drafts", action="store_true",
                    help="also copy the two .tex files into the paper folder")
    ap.add_argument("--allow-main-mismatch", action="store_true",
                    help="print the demand table although the local main RC result is not the fit "
                         "the cluster compared the variant with (the note then says so)")
    ap.add_argument("--allow-design-mismatch", action="store_true",
                    help="print the BBL table although the main and the variant run record "
                         "different designs (the note then lists the differing items)")
    a = ap.parse_args(argv)
    ests = [int(x) for x in a.routines.split(",") if x.strip()]
    dx_dir = pathlib.Path(a.dx_dir)
    out = pathlib.Path(a.out_dir) if a.out_dir else dx_dir / "tables"
    numbers = {"generator": "make_dx_tables.py", "variant": "dx", "suffix": DX_SUFFIX,
               "rc_stage": a.stage, "psi_tag_main": a.psi_tag, "psi_tag_dx": DX_SUFFIX + a.psi_tag}
    # Before anything is written: the variant's cost_params on disk and the variant's fit on disk
    # must belong to each other, whichever table is asked for.
    fit = {f"E{e}": fit_binding_check(e, a.psi_tag, dx_dir) for e in ests}
    for ek, x in fit.items():
        print(f"  pairing {ek}: {x['status']}" + (f" ({x['why']})" if x.get("why") else "")
              + "".join("\n      " + d for d in x["differences"]))
    unbound = [ek for ek, x in fit.items() if x["status"] == "MISMATCH"]
    if unbound:
        raise SystemExit("[dx-tables] the variant's cost parameters of " + ", ".join(unbound) + " were not "
                         "estimated on the variant's demand fit that is on disk (above). The tables would pair "
                         "demand estimates from one fit with cost estimates from another, so none is written. "
                         "Ingest the BLP and the BBL archive of the same run (cluster_ingest_dx.py --force), or "
                         "move the cost_params file out of " + str(dx_dir / "bbl") + " to print the demand table alone.")
    numbers["bbl_fit_check"] = fit
    written = []
    if a.only in (None, "demand"):
        checks = {f"E{e}": main_rc_check(e, a.stage, dx_dir) for e in ests}
        bad = [e for e in ests if checks[f"E{e}"]["status"] == "MISMATCH"]
        for e in ests:
            print(f"  main RC check E{e}: {checks[f'E{e}']['status']}"
                  + ("".join("\n      " + x for x in checks[f"E{e}"]["differences"])))
        if bad and not a.allow_main_mismatch:
            raise SystemExit("[dx-tables] the local main RC result of " + ", ".join(f"E{e}" for e in bad) +
                             " is not the fit the cluster compared the variant with (above). Re-ingest the "
                             "main BLP results, or pass --allow-main-mismatch to print the table with a "
                             "sentence saying so.")
        numbers["main_rc_check"] = checks
        nums = demand_numbers(ests, a.stage, dx_dir, a.rows)
        tex, pending = build_demand_tex(ests, a.stage, dx_dir, a.rows, main_mismatch=bad)
        _write(out / f"{DEMAND_BASE}.tex", tex); written.append(out / f"{DEMAND_BASE}.tex")
        _write(out / f"{DEMAND_BASE}.md", build_demand_md(nums, a.rows))
        numbers["demand"] = nums
        numbers["demand_pending"] = pending
        print("  demand table: " + ("every column on disk" if not pending else "PENDING columns: " + ", ".join(pending)))
    if a.only in (None, "bbl"):
        design = bbl_design(ests, a.psi_tag, dx_dir)
        for ek, x in design.items():
            print(f"  BBL design check {ek}: {x['status']}"
                  + (f"; differ: {x['differ']}" if x["differ"] else "")
                  + (f"; not recorded by one of the runs: {', '.join(x['not_recorded'])}" if x["not_recorded"] else ""))
        bad = [ek for ek, x in design.items() if x["status"] == "DIFFERENT"]
        if bad and not a.allow_design_mismatch:
            raise SystemExit("[dx-tables] the main and the variant BBL runs of " + ", ".join(bad) +
                             " record different designs (above), so the comparison is not of the demand "
                             "fit alone. Pass --allow-design-mismatch to print the table with the "
                             "differing items in the note.")
        numbers["bbl_design_check"] = design
        nums = bbl_numbers(ests, a.psi_tag, dx_dir)
        tex, pending = build_bbl_tex(ests, a.psi_tag, nums, design, pairing=fit)
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
