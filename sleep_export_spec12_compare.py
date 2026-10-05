import os
import sys
import shutil
import pickle
import json
import tempfile
from datetime import datetime, timezone

from utils import paths as _paths_mod
from utils import routines as _routines

_DRAFTS_DIR = _paths_mod.drafts_dir()
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import stats

# Row labels and display units come from the SHARED registry, so the sleepiness tables,
# the BBL policy functions and the descriptives cannot state different units for the same
# variable.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
from utils import state_transform as _st  # noqa: E402
from utils.sleep_links import band_row as _band_row  # noqa: E402
from utils import se_national as _sen  # noqa: E402
from utils import sleep_notes as _notes
from sleep_export_link import (clean_name as _clean_name, fmt3 as _fmt3,  # noqa: E402
                               eff_f_cell as _eff_f_cell)

# Mock NonLinearResults for unpickling estimation_3_sleep pickles.
# Must match the real class's __init__ signature so pickle restores __dict__ correctly.
class NonLinearResults:
    def __init__(self, params, bse, tvalues, pvalues, df_resid,
                 params_native=None, nobs=None, rsquared=None,
                 fvalue=None, f_pvalue=None, G_nominal=None, cov_ame=None,
                 nlls_status=None, nlls_message=None):
        self.params = params
        self.bse = bse
        self.tvalues = tvalues
        self.pvalues = pvalues
        self.df_resid = df_resid
        self.params_native = params_native if params_native is not None else params
        self.G_star = df_resid
        self.nobs = nobs
        self.rsquared = rsquared
        self.fvalue = fvalue
        self.f_pvalue = f_pvalue
        self.G_nominal = G_nominal
        self.cov_ame = cov_ame
        self.nlls_status = nlls_status
        self.nlls_message = nlls_message

    def cov_params(self):
        """Return full AME covariance matrix as a DataFrame (mirrors statsmodels interface)."""
        if self.cov_ame is not None:
            return pd.DataFrame(self.cov_ame, index=self.params.index, columns=self.params.index)
        return pd.DataFrame(np.diag(self.bse ** 2), index=self.params.index, columns=self.params.index)

# Fake module so pickles that reference estimation_3_sleep.NonLinearResults unpickle here.
sys.modules['estimation_3_sleep'] = type('FakeModule', (), {'NonLinearResults': NonLinearResults})

# Try to respect the project's venv guard
try:
    from utils.venv_guard import ensure_project_venv  # type: ignore[import-untyped]
    ensure_project_venv(__file__)
except ImportError:
    pass

def get_stars(pval):
    if pd.isna(pval): return ""
    if pval < 0.01: return "***"
    elif pval < 0.05: return "**"
    elif pval < 0.1: return "*"
    return ""

def format_value(coef, se, pval, digits=3, mark="", ci=None, stars=None):
    """`mark` flags a row whose SE comes from a non-default clustering scheme. These cells are
    TEXT mode (no surrounding $), so the marker must carry its own math delimiters -- unlike the
    sleepiness exporters, where the dagger goes inside the existing $...$.

    `ci=(lo,hi)` prints a bracketed interval on the second line in place of the parenthesised
    SE, for the two-stage AME bootstrap whose reported object is a bias-corrected percentile
    interval rather than a Wald SE. `stars` overrides the p-value-derived stars so they come
    from the same interval that is printed."""
    if pd.isna(coef):
        return "-", "-"
    st = get_stars(pval) if stars is None else stars
    if ci is not None:
        return f"{_fmt3(coef, digits)}{st}", f"[{_fmt3(ci[0], digits)}, {_fmt3(ci[1], digits)}]{mark}"
    return f"{_fmt3(coef, digits)}{st}", f"({_fmt3(se, digits)}){mark}"


AME_CI_TOKEN = "%%AME_CI_NOTE%%"


def ame_ci_note(ci_cols, results_dict):
    """LaTeX sentence for a table whose single-index columns print an interval where its linear
    columns print a standard error. Emitted only for the columns that actually carried a band,
    so the note can never describe a construction that did not run."""
    if not ci_cols:
        return ""
    meta = {}
    for c in ci_cols:
        m = getattr(results_dict.get(c), "ame_2s_meta", None) or {}
        meta.setdefault("B", m.get("B"))
        meta.setdefault("scheme", m.get("scheme"))
    lbl = ", ".join(sorted(str(c) for c in ci_cols))
    b_s = (f" $B={meta['B']}$ {meta['scheme']} draws," if meta.get("B") else "")
    return (r"Column(s) " + lbl + r" report a 95\% bias-corrected percentile interval "
            r"(\textcite{efron1987better}) in brackets, not a standard error: their "
            r"inference is a TWO-STAGE WCB (\textcite{klinesantos2012}) "
            r"in which the index direction is re-solved and the link re-profiled at "
            r"every draw," + b_s + r" and the reported object is a tangent-cone interval "
            r"rather than a Wald statistic, because the link's shape constraints are active at "
            r"the estimate: on that boundary the bootstrap of the constrained estimator is "
            r"inconsistent (\textcite{andrews2000inconsistency}) and the average marginal "
            r"effect is only directionally differentiable, so the cone construction is the "
            r"numerical delta method of \textcite{fangsantos2019} and \textcite{hongli2018}. "
            + _notes.reversed_draws_note(results_dict.get(c) for c in ci_cols))


# The band lookup lives in utils.sleep_links.band_row: one reader for both column families,
# since the two-stage AME band and the linear WCB band are built by the same function and
# carry the same columns. It reads only -- see the division of labour in its docstring.


def pastelize_color(color, blend=0.7):
    """Blend color toward white for CI bands."""
    rgb = np.array(mcolors.to_rgb(color))
    return tuple((1 - blend) * rgb + blend * np.array([1.0, 1.0, 1.0]))

def _t_crit(res, alpha=0.025):
    """Exact two-tailed 95% CI critical value: t_{G*-1, 1-alpha} from effective cluster df."""
    if res is None:
        return float(stats.norm.ppf(1 - alpha))
    g_star = getattr(res, 'G_star', getattr(res, 'df_resid', None))
    try:
        g_star = float(g_star)
    except (TypeError, ValueError):
        return float(stats.norm.ppf(1 - alpha))
    if np.isnan(g_star) or g_star <= 1:
        return float(stats.norm.ppf(1 - alpha))
    return float(stats.t.ppf(1 - alpha, df=g_star - 1))

# market_panel_phis.csv stores demographic/infrastructure columns already
# in the rescaled units used for estimation (gdp/10k, cadunico/100, etc.)
_SCALE_COLS: dict = {}


def _build_phi_regressors(df_sub: pd.DataFrame, phi_params) -> np.ndarray:
    """
    Build regressor matrix X (n_obs × len(phi_params)) from market panel columns.
    Maps:  'nr_lagged_dep'      → constant 1.0
           'interaction_{sv}'   → column sv (scaled)
           other param names    → matching column if present, else 0.0
    """
    n = len(df_sub)
    X = np.zeros((n, len(phi_params)))
    for i, pname in enumerate(phi_params):
        if pname == 'nr_lagged_dep':
            X[:, i] = 1.0
        elif pname.startswith('interaction_'):
            sv = pname[len('interaction_'):]
            if sv == 'pix_exists':
                if sv in df_sub.columns:
                    v = df_sub[sv].values.astype(float)
                elif 'year' in df_sub.columns and 'quarter' in df_sub.columns:
                    # Reconstruct the RAW indicator, then map it onto the levels the
                    # estimation actually used. Emitting raw {0,1} here would inject an
                    # uncentered column into a centered index -- silently, and the resulting
                    # phi would still look like a plausible number in [0,1].
                    raw = ((df_sub['year'] > 2020) |
                           ((df_sub['year'] == 2020) & (df_sub['quarter'] == 4))
                           ).astype(float).values
                    _lv = _st.dummy_levels('pix_exists')
                    v = raw if _lv is None else np.where(raw > 0.5, _lv[1], _lv[0])
                else:
                    v = np.zeros(n)
            elif sv in df_sub.columns:
                v = df_sub[sv].fillna(0.0).values.astype(float) * _SCALE_COLS.get(sv, 1.0)
            else:
                v = np.zeros(n)
            X[:, i] = v
        elif pname in df_sub.columns:
            X[:, i] = df_sub[pname].fillna(0.0).values.astype(float)
        # else: leave column as 0.0
    return X


def calc_agg_delta(d_sub: pd.DataFrame, col: str, res, is_logistic: bool) -> tuple:
    """
    Compute market_size-weighted national phi_t and its analytical (delta-method) SE.

    SE²_t = X̄_t' Σ X̄_t  where X̄_t = pop-weighted mean regressor vector for period t.

    Linear models  : Σ = cov_OLS (params = OLS beta)
    Logistic models: Σ = cov_AME (params = AME). The phi*(1-phi) scaling is already
                     embedded in cov_AME via the delta-method Jacobian in
                     get_nlls_ame_and_se; the gradient of phi_t wrt AME is plain X̄.
    Phi-specific params = all params whose name does NOT start with 'v_hat'.
    """
    if len(d_sub) == 0 or col not in d_sub.columns:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    # WEIGHT = MARKET POPULATION (2026-08-05 fix, matching the national_phi_t.csv writers).
    # Row-level pop weighting implicitly multiplies by the bank count of each market (every
    # bank-row carries the market's pop), reproducing the pop x n_banks convention this fix
    # retires. Collapse to (quarter, market) cells first -- cell-mean phi, cell pop weight --
    # exactly like calculate_phis/_calculate_phis. Falls back to row weighting only when no
    # market key exists in the frame.
    mkey = next((k for k in ('mca_code', 'CODMUN_IBGE') if k in d_sub.columns), None)
    if mkey is not None and 'market_size' in d_sub.columns:
        cells = d_sub.groupby(['year_quarter', mkey], observed=True).agg(
            _phi=(col, 'mean'), _w=('market_size', 'mean')).reset_index()
        mean = ((cells['_phi'] * cells['_w']).groupby(cells['year_quarter']).sum()
                / cells['_w'].groupby(cells['year_quarter']).sum())
        # per-row weights for the delta-method X-bar below: pop / n_rows-in-cell, so each
        # market contributes its population once regardless of how many banks sit in it.
        _n = d_sub.groupby(['year_quarter', mkey], observed=True)[col].transform('size')
        w = (d_sub['market_size'] / _n).astype(float)
    else:
        w = d_sub['market_size'] if 'market_size' in d_sub.columns else pd.Series(1.0, index=d_sub.index)
        mean = (d_sub[col] * w).groupby(d_sub['year_quarter']).sum() / w.groupby(d_sub['year_quarter']).sum()
    w_arr = w.values.astype(float)

    if res is None:
        return mean, pd.Series(0.0, index=mean.index)

    all_idx = res.params.index
    phi_mask = ~pd.Series(list(all_idx)).str.startswith('v_hat').values
    phi_params = all_idx[phi_mask]

    full_cov = res.cov_params()
    p_idx = [j for j, p in enumerate(all_idx) if p in set(phi_params)]
    Sigma = full_cov.values[np.ix_(p_idx, p_idx)]

    X = _build_phi_regressors(d_sub, phi_params)
    yq_arr = d_sub['year_quarter'].values

    G = X

    se_vals: dict = {}
    for t in mean.index:
        mask = yq_arr == t
        w_t = w_arr[mask]
        g_t = G[mask]
        w_sum = w_t.sum()
        if w_sum == 0:
            se_vals[t] = 0.0
            continue
        g_bar = (w_t[:, None] * g_t).sum(axis=0) / w_sum   # (k,)
        var_t = float(g_bar @ Sigma @ g_bar)
        se_vals[t] = np.sqrt(max(var_t, 0.0))

    return mean, pd.Series(se_vals)

def nice_var_name(var):
    """Row label from the SHARED registry (units included). The one local override is
    'nr_lagged_dep', which this table heads as the Constant rather than as a regressor."""
    v = str(var).replace('interaction_', '')
    if v in ('nr_lagged_dep', 'const', 'constant'):
        return 'Constant'
    return _clean_name(v)

# This table's column/series keys, one per routine. They are ordinary dict keys -- `mapping`
# in main() and the figure's linestyle/legend maps all key off them -- so they are spelled
# once here and the rest of the file derives from this map.
EST_KEYS = {
    1: '1 Local',
    2: '2 Pooled Linear',
    3: '3 Single-Index',
    4: '4 Single-Index Time',
}

# Column headers reference the estimation-strategy enumeration in V_Main.tex
# (\item\label{estimation:*} at lines ~402-408), so a column reads as its item number
# ((1)-(6)) rather than a name -- thinner columns, and the strategy is defined once in
# the text. \ref resolves inside V_Main; standalone/test compiles show "(??)".
# The refs come from config/routines.toml via est_ref, which yields a plain E{id} for an id
# with no live label -- the column loses its strategy number rather than raising.
REF_LABELS = {k: _routines.est_ref(e) for e, k in EST_KEYS.items()}
# Column key -> routine id. `band_row` needs the id to find a linear routine's band sidecar
# when the result carries no attached band of its own.
EST_OF_KEY = {k: e for e, k in EST_KEYS.items()}

# The columns of these tables are estimation strategies under ONE specification, so the title
# names the strategies as the varying dimension and states which specification is held fixed,
# in the words of the specification guide (Table tab:sleep_specifications_guide).
_IV_WORDS = {'OLS': 'No Instruments (OLS)',
             'IV_CostShifters': 'Cost-Shifter Instruments',
             'IV_Wholesale': 'Cost-Shifter and Wholesale Instruments',
             'IV_HausmanFull': 'Hausman Instruments'}


def spec12_title(stage):
    iv, state = (p.strip() for p in _routines.SPEC12.split(" x "))
    return (rf"{stage} Results by Estimation Strategy, Specification "
            rf"({_routines.SPEC12_ID}): {state} State Vector, {_IV_WORDS.get(iv, iv)}")


def n_clusters(res):
    """Cluster count exactly as the Clusters ($G$) row reads it; 0 when unknown."""
    if hasattr(res, 'cov_kwds') and res.cov_kwds.get('groups', None) is not None:
        groups = res.cov_kwds.get('groups', None)
        return int(groups.nunique() if hasattr(groups, 'nunique') else len(set(groups)))
    g = getattr(res, 'G_nominal', np.nan)
    return int(g) if g is not None and not pd.isna(g) else 0


def _span(refs):
    """'(I)--(II)' for two adjacent column refs, a comma list otherwise."""
    return rf"{refs[0]}--{refs[-1]}" if len(refs) == 2 else ", ".join(refs)


def _cols(order_keys, ests):
    return [REF_LABELS[k] for k in order_keys if EST_OF_KEY.get(k) in ests]


def _local_exclusion(results_dict, order_keys):
    """The sample clause's caveat for column (I), which drops the D firms the pooled columns
    keep, with the observation and cluster gap read off the fits; "" when (I) is absent or its
    sample is not smaller."""
    loc = next((k for k in order_keys if EST_OF_KEY.get(k) == 1), None)
    pooled = [k for k in order_keys if k != loc and results_dict.get(k) is not None]
    r1 = results_dict.get(loc) if loc else None
    if r1 is None or not pooled:
        return ""
    n_p = {int(getattr(results_dict[k], 'nobs', 0) or 0) for k in pooled}
    g_p = {n_clusters(results_dict[k]) for k in pooled}
    n1, g1 = int(getattr(r1, 'nobs', 0) or 0), n_clusters(r1)
    if len(n_p) != 1 or len(g_p) != 1 or not n1 or n1 >= min(n_p):
        return ""
    dn, dg = min(n_p) - n1, min(g_p) - g1
    gap = f"{dn:,} observations" + (f", {dg} clusters" if dg > 0 else "")
    return rf"; {REF_LABELS[loc]} excludes $\mathrm{{D}}$ firms ({gap} fewer)"


def compare_note(results_dict, order_keys, mean_phi=None, first_stage=False):
    """The note body of these tables, assembled from the shared clauses in utils/sleep_notes so
    it states each convention in the words the appendix tables use. What belongs to this table
    alone -- column (I)'s smaller sample, what Mean phi-hat averages, that it is unbounded under
    the linear link -- is read off the fits, so the note cannot describe a sample it does not
    show."""
    extra = _local_exclusion(results_dict, order_keys)
    if first_stage:
        return _notes.first_stage_note(pooled=True, extra=extra) + _sen.NOTE_TOKEN
    lin, lnk = _cols(order_keys, (1, 2)), _cols(order_keys, (3, 4))
    effects = (_notes.effects_clause("both", _span(lin), _span(lnk)) if lin and lnk
               else _notes.effects_clause("coef" if lin else "ame"))
    caveats = ""
    if mean_phi and any(mean_phi.get(k) is not None for k in order_keys):
        caveats += (r"Mean $\hat{\phi}$: population-weighted national $\hat{\phi}_t$ averaged "
                    r"over quarters. ")
    if lin:
        caveats += _notes.linear_link_caveat(r"$\hat{\phi}$", where=f" of {_span(lin)}")
    return _notes.second_stage_note(_notes.sample_clause(True, extra), effects, caveats)


def two_stage_tag(order_keys):
    """The inference clause's qualifier naming the columns bootstrapped in two stages."""
    lnk = _cols(order_keys, (3, 4))
    return rf", two-stage for {_span(lnk)}" if lnk else ""


# ---------------------------------------------------------------------------
# Unrounded sidecar: a JSON record of the exact numbers a comparison table's cells were built
# from, written by the same loop that formats those cells to 3dp text. Exists because a reader
# that needs a value to more than 3 decimals (the paper-numbers collector's star-and-interval
# parser cannot always recover a -0.755-class estimate at 2dp from its printed text) has to get
# it from the generator rather than by inverting the rounded string.
# ---------------------------------------------------------------------------

def _empty_cell_record():
    """The sidecar record for a cell the table prints as '-' (no result, or the row absent
    from that column's params) -- same shape as a populated cell so a reader need not branch."""
    return {
        "estimate": None, "ci_lo": None, "ci_hi": None, "se": None, "pvalue": None,
        "stars": "", "stars_source": None, "dagger": False, "scheme": None,
        "printed_estimate": "-", "printed_interval": "-",
    }


def _atomic_json_dump(obj, path):
    """Write a JSON sidecar that is either complete or absent, never truncated.

    Same scheme as sleep_est_single.py's _atomic_pickle_dump: a temp file in the SAME
    directory, fsynced, then renamed into place with os.replace, which is atomic on Windows
    and POSIX alike. Written through os.fdopen(fd, "wb") as encoded UTF-8 bytes rather than
    opened in Python's text mode, so the LF line endings json.dumps produces are the bytes
    that reach disk -- text mode on Windows would otherwise rewrite every \\n to \\r\\n."""
    path = str(path)
    d = os.path.dirname(path)
    fd, tmp = tempfile.mkstemp(dir=d, prefix="." + os.path.basename(path) + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(json.dumps(obj, indent=2).encode("utf-8"))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _write_table_sidecar(sidecar_path, tex_path, label, order_keys, rows, mean_phi, nobs,
                         r_squared, clusters, pkl_meta=None):
    """Write one comparison table's unrounded cell values to `sidecar_path`.

    `rows`, `mean_phi`, `nobs`, `r_squared` and `clusters` are collected by the caller during
    the same pass that builds the .tex rows, so this cannot describe a table other than the one
    that was just written. `pkl_meta` carries the input pickles' paths and mtimes, so a reader
    can tell which vintage of estimation_results.pkl the numbers came from."""
    payload = {
        "schema": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generator": "sleep_export_spec12_compare.py:build_latex_table",
        "tex_file": os.path.basename(str(tex_path)),
        "label": label,
        "spec": _routines.SPEC12,
        "columns": list(order_keys),
        "column_est_id": {k: EST_OF_KEY.get(k) for k in order_keys},
        "column_ref_label": {k: REF_LABELS.get(k, k) for k in order_keys},
        "inputs": pkl_meta or {},
        "rows": rows,
        "table": {
            "mean_phi_hat_pp": mean_phi,
            "observations": nobs,
            "r_squared": r_squared,
            "clusters": clusters,
        },
    }
    _atomic_json_dump(payload, sidecar_path)
    print(f"Exported table sidecar (unrounded values) to {sidecar_path}")


def build_latex_table(results_dict, order_keys, target_vars, out_path, title="", label="",
                      mean_phi=None, sidecar_path=None, pkl_meta=None):
    tex = []
    is_first_stage = ("first_stage" in str(out_path).lower() or "stage1" in str(out_path).lower())

    # \setstretch{1.0} matches the paper's other tables (est1_first_stage_table.tex
    # etc.), all of which open with \setstretch{1.0} and no wrapping group.
    # Font size (\footnotesize) and \arraystretch (1.08) are applied automatically
    # by the document preamble via \AtBeginEnvironment{xltabular}{\footnotesize}
    # and \renewcommand{\arraystretch}{1.08}, so we do not override them locally.
    tex.append(r"\setstretch{1.0}")

    # xltabular pins the table to \textwidth and distributes the remaining width
    # equally among the X data columns (same as tabularx but supports longtable
    # headers/footers). The first column is a fixed-width raggedright p column so
    # long labels (e.g. "Mobile Lines (per 100 inhabitants)") wrap rather
    # than forcing the table past the text block. 0.26\textwidth leaves each X column room
    # for a bracketed interval on one line.
    n_data = len(order_keys)
    col_def = (r">{\raggedright\arraybackslash}p{0.26\textwidth} "
               r"*{" + str(n_data) + r"}{>{\centering\arraybackslash}X}")
    tex.append(r"\begin{xltabular}{\textwidth}{" + col_def + "}")
    tex.append(r"\caption{" + title + r"}\label{" + label + r"} \\")

    # First Header
    tex.append(r"\toprule")
    headers = ["Variable"] + [REF_LABELS.get(k, k) for k in order_keys]
    tex.append(" & ".join(headers) + r" \\")
    tex.append(r"\midrule")
    tex.append(r"\endfirsthead")

    # Next Headers
    # NOTE: keep this continuation line short. longtable computes column widths
    # from *every* head/foot row, so embedding the full (long) title here forces
    # an unbreakable \multicolumn wider than the page, which balloons the columns
    # and pushes the rules off the right margin even on the first page.
    tex.append(r"\multicolumn{" + str(len(order_keys) + 1) + r"}{c}{{\bfseries \tablename\ \thetable{} (continued from previous page)}} \\")
    tex.append(r"\toprule")
    tex.append(" & ".join(headers) + r" \\")
    tex.append(r"\midrule")
    tex.append(r"\endhead")

    # Footers
    tex.append(r"\midrule")
    tex.append(r"\multicolumn{" + str(len(order_keys) + 1) + r"}{r}{{Continued on next page}} \\")
    tex.append(r"\endfoot")

    # Last Footer: the note in \footnotesize, the size every sleepiness table note uses.
    tex.append(r"\bottomrule")
    # \dimexpr\textwidth-2\tabcolsep\relax is exactly the usable width of a
    # full-span multicolumn in a \textwidth-wide xltabular: the table occupies
    # \textwidth, but the outer \tabcolsep margins on left and right eat 2*3.5pt=7pt,
    # leaving \textwidth-7pt for the cell content.
    # The first-stage note carries _sen.NOTE_TOKEN, which expands to nothing while no DISPLAYED
    # row is national; if one ever is, it names the clustering that row actually used.
    notes_str = (r"\multicolumn{" + str(len(order_keys) + 1) + r"}{p{\dimexpr\textwidth-2\tabcolsep\relax}}"
                 r"{\footnotesize\textit{Notes:} "
                 + compare_note(results_dict, order_keys, mean_phi, first_stage=is_first_stage)
                 + r"}")
    tex.append(notes_str)
    tex.append(r"\endlastfoot")

    # Collect all unique vars from loaded results
    vars_to_print = []
    for col in order_keys:
        res = results_dict.get(col)
        if res is not None:
            params = getattr(res, 'params', pd.Series(dtype=float))
            for v in params.index:
                if v not in vars_to_print:
                    vars_to_print.append(v)

    # Target vars first, then any remaining (excluding CF nuisance term)
    ordered_vars = [v for v in target_vars if v in vars_to_print]
    _nat_schemes = set()   # what select_se ACTUALLY returned on the national rows
    _sen.reset_ame_se_realised()   # and which AME variance it actually delivered
    _ci_cols = set()       # columns whose second line is an interval rather than an SE
    # Unrounded companions to the tex rows below, built alongside them regardless of whether
    # sidecar_path is set (the bookkeeping is a handful of dicts, not another pass over the
    # results) and written out only if it is.
    _sidecar_rows: list = []
    _sidecar_mean_phi: dict = {}
    _sidecar_nobs: dict = {}
    _sidecar_r2: dict = {}
    _sidecar_clusters: dict = {}

    if not is_first_stage:
        other_vars = [v for v in vars_to_print if v not in ordered_vars]
        exclude_patterns = ["v_hat"]
        other_vars = [v for v in other_vars if not any(pattern in v.lower() for pattern in exclude_patterns)]
        ordered_vars += other_vars
        # Never show the control-function term in the second-stage table
        ordered_vars = [v for v in ordered_vars if 'v_hat' not in v.lower()]

    for v in ordered_vars:
        # Two table rows per variable: label spans both via \multirow[t]{2} so
        # it stays anchored even when it wraps. We give the explicit column width
        # (0.26\textwidth, matching the p-column spec) rather than = (infer) because
        # in V_Main.tex's \doublespacing context = computes 7pt wider than the column.
        # \\* on the coeff row forbids a page break between coefficient and SE.
        label_cell = r"\multirow[t]{2}{0.26\textwidth}{\raggedright " + nice_var_name(v) + r"}"
        row_cf = [label_cell]
        row_se = [""]
        _row_cells: dict = {}   # this row's unrounded companion, keyed by column
        for col in order_keys:
            res = results_dict.get(col)
            if res is None:
                row_cf.append("-")
                row_se.append("-")
                _row_cells[col] = _empty_cell_record()
                continue

            params = getattr(res, 'params', pd.Series(dtype=float))
            bse = getattr(res, 'bse', pd.Series(dtype=float))
            pvalues = getattr(res, 'pvalues', pd.Series(dtype=float))

            if v in params.index:
                # Display units from the shared registry: coefficient and SE only --
                # p-values and stars are invariant to a change of units.
                m = _st.display_mult(v, lhs=_st.SPREAD_DISPLAY if is_first_stage
                                     else _st.PHI_DISPLAY)
                # National rows (Pix, Selic) report the quarter-clustered bootstrap; all other
                # rows keep the conglomerate one. This table mixes linear (E1/E2) and nonlinear
                # (E3/E4) columns, so the same scheme must hold across a row -- which is why the
                # reported cell is quarter-WCB rather than the wider-of-two (DK is linear-only).
                _se, _pv, _sch = _sen.select_se(res, v)
                if _sen.is_national(v):
                    _nat_schemes.add(_sch)
                _mark = r"$^{\dagger}$" if _sch != "congl" else ""
                # Every column prints a bias-corrected percentile interval: the single-index
                # ones from the attached two-stage AME band, the linear ones from their band
                # sidecar. Both come from the same builder, so widths are comparable across a
                # row -- which they were not while half the row reported a standard error. Same
                # display multiplier on both endpoints; stars from the same interval. A column
                # whose band is missing falls back to the SE and the note says which did.
                _bd = _band_row(res, v, est=EST_OF_KEY.get(col), spec=_routines.SPEC12)
                _ci = (_bd[0] * m, _bd[1] * m) if _bd else None
                # Same selection format_value makes internally (stars=None there falls back to
                # get_stars(pval)); computed again here, off the same _bd/_pv, so the sidecar's
                # star string is provably the one that got typeset rather than a second guess.
                _stars = _bd[2] if _bd else get_stars(_pv)
                c_str, se_str = format_value(params[v] * m, _se * m, _pv, digits=3, mark=_mark,
                                             ci=_ci, stars=(_bd[2] if _bd else None))
                if _bd:
                    _ci_cols.add(col)
                row_cf.append(c_str)
                row_se.append(se_str)
                _est_val = params[v] * m
                _row_cells[col] = {
                    "estimate": float(_est_val) if pd.notna(_est_val) else None,
                    "ci_lo": float(_ci[0]) if _ci is not None else None,
                    "ci_hi": float(_ci[1]) if _ci is not None else None,
                    "se": float(_se * m) if pd.notna(_se) else None,
                    "pvalue": float(_pv) if pd.notna(_pv) else None,
                    "stars": _stars,
                    "stars_source": "band" if _bd else "pvalue",
                    "dagger": bool(_sch != "congl"),
                    "scheme": _sch,
                    "printed_estimate": c_str,
                    "printed_interval": se_str,
                }
            else:
                row_cf.append("-")
                row_se.append("-")
                _row_cells[col] = _empty_cell_record()

        tex.append(" & ".join(row_cf) + r" \\*")
        tex.append(" & ".join(row_se) + r" \\")
        tex.append(r"\addlinespace")
        _sidecar_rows.append({"var": v, "row_label": nice_var_name(v), "cells": _row_cells})

    tex.append(r"\midrule")

    # Implied national mean phi: the comparable "level" across linear / single-index / joint.
    # The single-index estimators carry no constant AME (the level is in the monotone
    # link), so this row gives the interpretable baseline sleepiness for every column.
    if mean_phi is not None and not is_first_stage:
        # phi is a LEVEL here, so it carries PHI_DISPLAY too -- otherwise this would be
        # the one row still in [0,1] while every coefficient above it is in pp.
        row_meanphi = [r"Mean $\hat{\phi}$ (pp)"]
        for col in order_keys:
            mp = mean_phi.get(col)
            _has_mp = mp is not None and pd.notna(mp)
            row_meanphi.append(_fmt3(mp * _st.PHI_DISPLAY) if _has_mp else "-")
            _sidecar_mean_phi[col] = {
                "unrounded": float(mp * _st.PHI_DISPLAY) if _has_mp else None,
                "printed": row_meanphi[-1],
            }
        tex.append(" & ".join(row_meanphi) + r" \\")

    row_nobs = ["Observations"]
    row_r2 = ["$R^2$"]
    # The two F rows are first-stage rows: the fit's own F over every slope, and under it the
    # effective F of the excluded instruments read from Rout/sleep_first_stage_strength.json.
    row_fstat = [_notes.ROW_F_ALL]
    row_efff = [_notes.ROW_F_EFF]
    row_cluster = ["Clusters ($G$)"]

    # E3/E4 (single-index) carry their own nobs/rsquared; no fallback needed.
    _nlls_fallback: dict = {}

    for col in order_keys:
        res = results_dict.get(col)
        if res is None:
            row_nobs.append("-"); row_r2.append("-"); row_fstat.append("-")
            row_efff.append("")
            row_cluster.append("-")
            _sidecar_nobs[col] = {"value": None, "printed": "-"}
            _sidecar_r2[col] = {"unrounded": None, "printed": "-"}
            _sidecar_clusters[col] = {"value": None, "printed": "-"}
            continue

        nobs  = getattr(res, 'nobs', np.nan)
        r2    = getattr(res, 'rsquared', np.nan)
        fstat = getattr(res, 'fvalue', np.nan)
        fpval = getattr(res, 'f_pvalue', np.nan)

        if pd.isna(nobs) or pd.isna(r2):
            fallback_col = _nlls_fallback.get(col)
            if fallback_col and fallback_col in results_dict:
                f_res = results_dict[fallback_col]
                if f_res is not None:
                    if pd.isna(nobs):  nobs  = getattr(f_res, 'nobs', np.nan)
                    if pd.isna(r2):    r2    = getattr(f_res, 'rsquared', np.nan)
                    if pd.isna(fstat): fstat = getattr(f_res, 'fvalue', np.nan)
                    if pd.isna(fpval): fpval = getattr(f_res, 'f_pvalue', np.nan)

        clusters = "-"
        if hasattr(res, 'cov_kwds') and res.cov_kwds.get('groups', None) is not None:
            groups = res.cov_kwds.get('groups', None)
            clusters = str(groups.nunique() if hasattr(groups, 'nunique') else len(set(groups)))
        elif hasattr(res, 'G_nominal') and not pd.isna(getattr(res, 'G_nominal', np.nan)):
            clusters = str(int(res.G_nominal))

        fstat_str = f"{_fmt3(fstat)}{get_stars(fpval)}" if pd.notna(fstat) else "-"

        row_nobs.append(f"{nobs:,.0f}" if pd.notna(nobs) else "-")
        row_r2.append(_fmt3(r2) if pd.notna(r2) else "-")
        row_fstat.append(fstat_str)
        row_efff.append(_eff_f_cell(EST_OF_KEY.get(col), _routines.SPEC12, math=False)
                        if is_first_stage else "")
        row_cluster.append(clusters)
        _sidecar_nobs[col] = {"value": float(nobs) if pd.notna(nobs) else None,
                              "printed": row_nobs[-1]}
        _sidecar_r2[col] = {"unrounded": float(r2) if pd.notna(r2) else None,
                            "printed": row_r2[-1]}
        _sidecar_clusters[col] = {"value": int(clusters) if clusters != "-" else None,
                                  "printed": clusters}

    tex.append(" & ".join(row_nobs) + r" \\")
    tex.append(" & ".join(row_r2) + r" \\")
    if is_first_stage:
        tex.append(" & ".join(row_fstat) + r" \\")
        tex.append(" & ".join(row_efff) + r" \\")
    tex.append(" & ".join(row_cluster) + r" \\")

    tex.append(r"\end{xltabular}")
    # Restore the document's double spacing; \setstretch{1.0} at the top of the
    # table suppressed it, so without this the following body text stays single-spaced.
    tex.append(r"\doublespacing")

    # Which opening the note gets is decided by what the cells actually printed: every data
    # column carrying a band, none, or some. `_ci_cols` was filled during rendering.
    _bands = True if len(_ci_cols) == n_data else (False if not _ci_cols else "mixed")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(tex).replace(
            _notes.OPEN_TOKEN, _notes.note_open(_bands, two_stage=two_stage_tag(order_keys))
            + _notes.reversed_draws_note(results_dict.get(c) for c in _ci_cols)).replace(
            _sen.NOTE_TOKEN, _sen.national_note(_nat_schemes)).replace(
            AME_CI_TOKEN, ame_ci_note(_ci_cols, results_dict)).replace(
            _sen.AME_SE_TOKEN, _sen.ame_se_note()))

    if sidecar_path is not None:
        _write_table_sidecar(
            sidecar_path=sidecar_path, tex_path=out_path, label=label, order_keys=order_keys,
            rows=_sidecar_rows, mean_phi=_sidecar_mean_phi, nobs=_sidecar_nobs,
            r_squared=_sidecar_r2, clusters=_sidecar_clusters, pkl_meta=pkl_meta,
        )


def build_latex_table_landscape(results_dict, order_keys, target_vars, out_path,
                                title="", label="", mean_phi=None, placement="ht",
                                first_stage=False):
    """Landscape variant of the stage-2 comparison for V_Main inclusion.

    Follows the house landscape style (blp_compare_E*_spec12.tex): a real
    `table` float wrapped in pdflscape's `landscape`, with booktabs rules and
    threeparttable notes -- all already loaded by V_Main.tex. The float takes a
    SOFT placement specifier (default [ht], NOT a hard [H]) so it settles around
    the \\input location. Carries the SAME \\label as the portrait table, so
    swapping the \\input in V_Main keeps every \\ref resolving. Natural centered
    columns (not xltabular's X) + two-line \\shortstack headers keep the seven
    columns readable across the rotated page. Column headers are the estimation-strategy
    item numbers via \\ref (see REF_LABELS), not names.
    """
    n = len(order_keys)
    headers = [""] + [REF_LABELS.get(k, k) for k in order_keys]

    # Variable order: target_vars first, then any extras, never the CF nuisance term.
    vars_to_print = []
    for col in order_keys:
        res = results_dict.get(col)
        if res is not None:
            for v in getattr(res, 'params', pd.Series(dtype=float)).index:
                if v not in vars_to_print:
                    vars_to_print.append(v)
    ordered_vars = [v for v in target_vars if v in vars_to_print]
    _nat_schemes = set()   # what select_se ACTUALLY returned on the national rows
    _sen.reset_ame_se_realised()   # and which AME variance it actually delivered
    _ci_cols = set()       # columns whose second line is an interval rather than an SE
    if not first_stage:   # second stage also lists any extra coefs; first stage = instruments only
        ordered_vars += [v for v in vars_to_print if v not in ordered_vars and 'v_hat' not in v.lower()]
        ordered_vars = [v for v in ordered_vars if 'v_hat' not in v.lower()]

    tex = [r"\begin{landscape}",
           r"\begin{table}[" + placement + r"]",
           r"\centering",
           r"\begin{threeparttable}",
           r"\caption{" + title + r"}",
           r"\label{" + label + r"}",
           r"\footnotesize",
           r"\setlength{\tabcolsep}{6pt}",
           r"\renewcommand{\arraystretch}{1.15}",
           r"\begin{tabular}{>{\raggedright\arraybackslash}p{4.8cm} *{" + str(n) + r"}{c}}",
           r"\toprule",
           " & ".join(headers) + r" \\",
           r"\midrule"]

    for v in ordered_vars:
        row_cf = [nice_var_name(v)]
        row_se = [""]
        for col in order_keys:
            res = results_dict.get(col)
            if res is None:
                row_cf.append("-"); row_se.append("-"); continue
            params = getattr(res, 'params', pd.Series(dtype=float))
            bse = getattr(res, 'bse', pd.Series(dtype=float))
            pvalues = getattr(res, 'pvalues', pd.Series(dtype=float))
            if v in params.index:
                m = _st.display_mult(v, lhs=_st.SPREAD_DISPLAY if first_stage
                                     else _st.PHI_DISPLAY)
                # See the portrait generator above: quarter-clustered WCB on the national rows.
                _se, _pv, _sch = _sen.select_se(res, v)
                if _sen.is_national(v):
                    _nat_schemes.add(_sch)
                _mark = r"$^{\dagger}$" if _sch != "congl" else ""
                # Every column prints a bias-corrected percentile interval: the single-index
                # ones from the attached two-stage AME band, the linear ones from their band
                # sidecar. Both come from the same builder, so widths are comparable across a
                # row -- which they were not while half the row reported a standard error. Same
                # display multiplier on both endpoints; stars from the same interval. A column
                # whose band is missing falls back to the SE and the note says which did.
                _bd = _band_row(res, v, est=EST_OF_KEY.get(col), spec=_routines.SPEC12)
                _ci = (_bd[0] * m, _bd[1] * m) if _bd else None
                c_str, se_str = format_value(params[v] * m, _se * m, _pv, digits=3, mark=_mark,
                                             ci=_ci, stars=(_bd[2] if _bd else None))
                if _bd:
                    _ci_cols.add(col)
                row_cf.append(c_str); row_se.append(se_str)
            else:
                row_cf.append("-"); row_se.append("-")
        tex.append(" & ".join(row_cf) + r" \\")
        tex.append(" & ".join(row_se) + r" \\")
        tex.append(r"\addlinespace[0.2ex]")

    tex.append(r"\midrule")
    if mean_phi is not None:
        row_mp = [r"Mean $\hat{\phi}$ (pp)"]
        for col in order_keys:
            mp = mean_phi.get(col)
            row_mp.append(_fmt3(mp * _st.PHI_DISPLAY)
                          if mp is not None and pd.notna(mp) else "-")
        tex.append(" & ".join(row_mp) + r" \\")

    row_nobs = ["Observations"]; row_r2 = ["$R^2$"]; row_fstat = [_notes.ROW_F_ALL]
    row_efff = [_notes.ROW_F_EFF]
    row_cl = [r"Clusters ($G$)"]
    for col in order_keys:
        res = results_dict.get(col)
        if res is None:
            for r_ in (row_nobs, row_r2, row_fstat, row_cl): r_.append("-")
            row_efff.append("")
            continue
        nobs = getattr(res, 'nobs', np.nan); r2 = getattr(res, 'rsquared', np.nan)
        fstat = getattr(res, 'fvalue', np.nan); fpval = getattr(res, 'f_pvalue', np.nan)
        clusters = "-"
        if hasattr(res, 'cov_kwds') and res.cov_kwds.get('groups', None) is not None:
            g = res.cov_kwds.get('groups', None)
            clusters = str(g.nunique() if hasattr(g, 'nunique') else len(set(g)))
        elif hasattr(res, 'G_nominal') and not pd.isna(getattr(res, 'G_nominal', np.nan)):
            clusters = str(int(res.G_nominal))
        row_nobs.append(f"{nobs:,.0f}" if pd.notna(nobs) else "-")
        row_r2.append(_fmt3(r2) if pd.notna(r2) else "-")
        row_fstat.append(f"{_fmt3(fstat)}{get_stars(fpval)}" if pd.notna(fstat) else "-")
        row_efff.append(_eff_f_cell(EST_OF_KEY.get(col), _routines.SPEC12, math=False)
                        if first_stage else "")
        row_cl.append(clusters)
    diag_rows = [row_nobs, row_r2] + ([row_fstat, row_efff] if first_stage else []) + [row_cl]
    for r_ in diag_rows:
        tex.append(" & ".join(r_) + r" \\")

    tex += [r"\bottomrule",
            r"\end{tabular}",
            r"\begin{tablenotes}[flushleft]",
            r"\footnotesize",
            r"\item \textit{Notes:} "
            + compare_note(results_dict, order_keys, mean_phi, first_stage=first_stage),
            r"\end{tablenotes}",
            r"\end{threeparttable}",
            r"\end{table}",
            r"\end{landscape}"]

    _bands = (True if len(_ci_cols) == len(order_keys)
              else (False if not _ci_cols else "mixed"))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(tex).replace(
            _notes.OPEN_TOKEN, _notes.note_open(_bands, two_stage=two_stage_tag(order_keys))
            + _notes.reversed_draws_note(results_dict.get(c) for c in _ci_cols)).replace(
            _sen.NOTE_TOKEN, _sen.national_note(_nat_schemes)).replace(
            AME_CI_TOKEN, ame_ci_note(_ci_cols, results_dict)).replace(
            _sen.AME_SE_TOKEN, _sen.ame_se_note()))


import argparse

def main():
    parser = argparse.ArgumentParser(description="Analyze Specification 12 Results (Est 1-3)")
    parser.add_argument('--skip-est2', action='store_true', help='Skip estimation 2 (Pooled Linear)')
    parser.add_argument('--tables-only', action='store_true',
                        help="write the tables and the model bundle, skip the phi_t plots (the "
                             "linear routines' plot reads each ~2 GB market_panel_phis.csv)")
    args = parser.parse_args()

    print(f"Collecting Estimation results for Spec 12 ({_routines.SPEC12})...")
    # demand_prep_root/est_dir/rout_dir all follow SLEEP_OUT_ROOT, so a sandboxed run
    # compares the fits it just produced instead of whatever sits in the production tree.
    sleep_root = _paths_mod.demand_prep_root()

    if not sleep_root.exists():
        print(f"ERROR: Cannot find {sleep_root}")
        sys.exit(1)

    # The lineup is E1/E2 (linear) + E3/E4 (bounded single-index).
    mapping = {EST_KEYS[e]: _paths_mod.est_dir(e) for e in _routines.ACTIVE}

    if getattr(args, 'skip_est2', False):
        mapping.pop('2 Pooled Linear', None)

    # All estimators store spec 12 under the same key.
    target_keys = {k: _routines.SPEC12 for k in mapping}

    stage1_res = {}
    stage2_res = {}
    phi_data = {}
    phi_csv_paths = {}  # lazy: market_panel_phis.csv is ~2GB/estimator, only read for the
                        # linear estimators (est1/est2) that actually plot from it (see below)
    mean_phi = {}      # implied national mean phi_t level per estimator (the comparable "level" row)
    models_dict = {}
    pkl_input_meta = {}   # label -> path/mtime/size of the estimation_results.pkl it came from,
                          # for the stage-2 sidecar's provenance (see build_latex_table)

    for label, d in mapping.items():
        pkl_path = d / "estimation_results.pkl"
        if not pkl_path.exists():
            print(f"  [Warning] Missing {pkl_path}, skipping {label}.")
            continue

        with open(pkl_path, 'rb') as f:
            try:
                res_dict = pickle.load(f)
            except Exception as e:
                print(f"  [Error] Failed to load {pkl_path}: {e}")
                continue

        tk = target_keys.get(label, _routines.SPEC12)
        spec_data = res_dict.get(tk)

        if not spec_data:
            print(f"  [Warning] '{tk}' not found in {label}, skipping.")
            continue

        models_dict[label] = spec_data
        _st_ = pkl_path.stat()
        pkl_input_meta[label] = {
            "path": str(pkl_path),
            "mtime_utc": datetime.fromtimestamp(_st_.st_mtime, tz=timezone.utc)
                                 .isoformat(timespec="seconds"),
            "size_bytes": _st_.st_size,
        }

        if spec_data.get('first_stage') is not None:
            stage1_res[label] = spec_data['first_stage']
        if spec_data.get('second_stage') is not None:
            stage2_res[label] = spec_data['second_stage']

        csv_path = d / "market_panel_phis.csv"
        if csv_path.exists():
            phi_csv_paths[label] = csv_path

        natl_path = d / "national_phi_t.csv"
        if natl_path.exists():
            try:
                nd = pd.read_csv(natl_path)
                pc = f'phi_t_{_routines.SPEC12_TAG}'
                if pc in nd.columns:
                    mean_phi[label] = float(nd[pc].mean())
            except Exception as e:
                print(f"  [Warning] Could not load national phi for {label}: {e}")

    # Per-estimator warnings above are skips, so with none of them loading the run would go
    # on to write empty comparison tables and an empty model pickle over the real ones.
    if not models_dict:
        print(f"ERROR: no estimator supplied spec 12 under {sleep_root}. "
              "Run the sleep estimators first.")
        sys.exit(1)

    # ---- 1) Export LaTeX Tables ----
    # ONE destination, chosen by environment. cluster_archive.sh packages only
    # data/output/sleep/Rout, so on the cluster a fragment written to the skeleton Drafts
    # dir never reaches the download; locally the paper directory is the only copy that
    # matters and Rout would just be a second one free to diverge from it. A LOCAL run with
    # SLEEP_OUT_ROOT pointed at a sandbox gets the same treatment: rout_dir() already follows
    # SLEEP_OUT_ROOT, but _DRAFTS_DIR never does, so without this a sandboxed run overwrote the
    # paper's own tables with the sandbox's numbers.
    _sandboxed = _paths_mod.sleep_out_root_set() and not _paths_mod.on_cluster()
    out_dir = _paths_mod.rout_dir() if (_paths_mod.on_cluster() or _sandboxed) else _DRAFTS_DIR
    if _sandboxed:
        print(f"  [sandbox] SLEEP_OUT_ROOT redirects this run: exhibits go to {out_dir}, "
              f"not Drafts.")
    # EXHIBITS vs DATA. .tex/.png are what the paper reads, so they follow out_dir. The
    # .pkl files are intermediates -- a serialized model bundle and the per-routine band
    # pickles -- and they belong in Rout on BOTH sides: they are not exhibits, the paper
    # never \input's them, and the band pickles are READ back below, so pointing them at
    # the paper directory both misfiles ~1.1 GB and makes the read miss.
    data_dir = _paths_mod.rout_dir()
    data_dir.mkdir(parents=True, exist_ok=True)

    order = list(mapping.keys())

    target_vars = [
        'nr_lagged_dep',
        'interaction_gdp_per_capita',
        'interaction_cadunico_families_per1000',
        'interaction_fraction_65plus',
        'interaction_fraction_young',
        'interaction_risk_free_qoq_lag',
        'interaction_connections_per100',
        'interaction_pix_exists',
        'interaction_gdp_growth_yoy',
        'v_hat_x_lagged_dep',
    ]

    first_stage_target_vars = [
        'tax_cost_ratio_lag', 'personnel_cost_ratio_lag', 'admin_cost_ratio_lag',
        'indice_basileia_lag', 'lci_lca_ratio_lag', 'wholesale_ratio_lag',
        'leave_one_out_mean_spread',
    ]

    build_latex_table(
        stage1_res, order, first_stage_target_vars,
        out_dir / "est1-4_spec12_stage1_comparison.tex",
        title=spec12_title("First Stage"),
        label="tab:spec12_stage1_comparison",
    )
    build_latex_table(
        stage2_res, order, target_vars,
        out_dir / "est1-4_spec12_stage2_comparison.tex",
        title=spec12_title("Second Stage"),
        label="tab:spec12_stage2_comparison",
        mean_phi=mean_phi,
        # Unrounded companion to this table's cells, written beside the band pickles rather
        # than into Drafts: it is DATA (a downstream reader's input), not an exhibit.
        sidecar_path=data_dir / "est1-4_spec12_stage2_comparison.json",
        pkl_meta=pkl_input_meta,
    )
    # Landscape variants of BOTH stages for V_Main inclusion (7 columns need the rotated
    # page). Same \labels as the portrait versions, so V_Main only switches which file it
    # \inputs (\input{est1-4_spec12_stage{1,2}_comparison_landscape.tex}).
    build_latex_table_landscape(
        stage1_res, order, first_stage_target_vars,
        out_dir / "est1-4_spec12_stage1_comparison_landscape.tex",
        title=spec12_title("First Stage"),
        label="tab:spec12_stage1_comparison",
        placement="ht", first_stage=True,
    )
    # No sidecar here: this reads the same stage2_res through the same select_se/band_row
    # calls as the portrait table above, so it prints the identical cells under a different
    # LaTeX wrapper -- the portrait table's sidecar already covers every number in it.
    build_latex_table_landscape(
        stage2_res, order, target_vars,
        out_dir / "est1-4_spec12_stage2_comparison_landscape.tex",
        title=spec12_title("Second Stage"),
        label="tab:spec12_stage2_comparison",
        mean_phi=mean_phi, placement="ht",
    )

    # No nonlinear-only cut is emitted: with the lineup at E1/E2 + E3/E4 it would be the
    # E3/E4 columns of the main table restated under a second \label.

    print(f"Exported LaTeX tables to {out_dir}")

    # ---- 2) Pickle Model Information ----
    with open(data_dir / "est1-4_spec12_all_models.pkl", "wb") as f:
        pickle.dump(models_dict, f)
    print(f"Exported combined model instances to {data_dir / 'est1-4_spec12_all_models.pkl'}")
    if args.tables_only:
        return

    # ---- 3) Plot Implied National Phi_t (single panel, all four strategies) ----
    #
    # Two band sources, by estimator family:
    #   * Linear E1/E2: aggregate per-market phi_mt to national with a delta-method
    #     SE band on the market panel (calc_agg_delta). Columns are saved as
    #     phi_mt_{safe_key}: SPEC12 -> phi_mt_{SPEC12_TAG} (utils/routines.py).
    #   * Single-index E3/E4: plot the bootstrap point path and
    #     the score/multiplier wild-cluster-bootstrap CI band cached in
    #     Rout/ts_link_band_est{N}.pkl (cols time_id, phi_t, lo, hi, _d). Point and
    #     band come from the SAME fit, so the line sits inside its band by
    #     construction; national_phi_t.csv tracks the same *level* (means agree to
    #     ~1e-3) but its per-quarter wiggle is not what the tight band bounds.
    phi_col_map = {k: f'phi_mt_{_routines.SPEC12_TAG}' for k in mapping}

    base_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    color_map = {lbl: base_colors[i % len(base_colors)] for i, lbl in enumerate(mapping.keys())}
    linestyle_map = {
        '1 Local':             '-',
        '2 Pooled Linear':     '-',
        '3 Single-Index':      '-',
        '4 Single-Index Time': '--',
    }
    label_rename = {
        '1 Local':             'Local Linear (E1)',
        '2 Pooled Linear':     'Pooled Linear (E2)',
        '3 Single-Index':      'Single-Index (E3)',
        '4 Single-Index Time': 'Single-Index +Time (E4)',
    }

    fig, ax = plt.subplots(figsize=(12, 6))
    ci_list: list = []

    for label in mapping.keys():
        est_num = int(str(label).split()[0])
        c = color_map[label]
        ls = linestyle_map.get(label, '-')
        plot_label = label_rename.get(label, label)

        band_pkl = data_dir / f"ts_link_band_est{est_num}.pkl"
        if est_num >= 3 and band_pkl.exists():
            # Single-index: bootstrap point path + wild-cluster band.
            try:
                with open(band_pkl, "rb") as fb:
                    bd = pickle.load(fb)
                bd = bd.sort_values("_d")
                ax.plot(bd["_d"], bd["phi_t"], label=plot_label, color=c, linewidth=2, linestyle=ls)
                ci_list.append((bd["_d"].values, bd["lo"].values, bd["hi"].values, pastelize_color(c)))
                continue
            except Exception as e:
                print(f"  [Warning] band load failed for {label} ({e}); falling back to delta band.")

        # Linear E1/E2: aggregate per-market phi to national with a delta-method band.
        # Also serves as the fallback for est_num>=3 if the band pkl was missing/unreadable.
        df_phi = phi_data.get(label)
        if df_phi is None:
            csv_path = phi_csv_paths.get(label)
            if csv_path is None:
                continue
            try:
                df_phi = pd.read_csv(csv_path, low_memory=False)
                phi_data[label] = df_phi
            except Exception as e:
                print(f"  [Warning] Could not load phi CSV for {label}: {e}")
                continue
        if 'year_quarter' not in df_phi.columns:
            continue
        tar_col = next((cc for cc in (f'phi_mt_{_routines.SPEC12_TAG}', 'phi_mt_Tech')
                        if cc in df_phi.columns), None)
        if tar_col is None:
            print(f"  [Warning] No phi column found for {label}, skipping plot.")
            continue

        res = stage2_res.get(label)
        crit_val = _t_crit(res)
        agg, se = calc_agg_delta(df_phi, tar_col, res, False)
        if agg.empty:
            continue

        idx_dates = pd.PeriodIndex(agg.index.str.replace('_', 'Q'), freq='Q').to_timestamp()
        ax.plot(idx_dates, agg.values, label=plot_label, color=c, linewidth=2, linestyle=ls)
        ci_list.append((
            idx_dates,
            agg.values - crit_val * se.values,
            agg.values + crit_val * se.values,
            pastelize_color(c),
        ))

    ax.set_title(r"Implied National $\hat{\phi}_t$ — Spec 12 (IV Hausman $\times$ Tech)", fontsize=14, pad=12)
    ax.set_ylabel(r"National $\hat{\phi}_t$")
    ax.set_ylim(0, 1.05)
    ax.axhline(1.0, color='gray', linestyle=':', linewidth=1.2, alpha=0.7)
    ax.grid(alpha=0.4)
    ax.legend(loc='best')
    for x, lo, hi, pc in ci_list:
        ax.fill_between(x, lo, hi, color=pc, alpha=0.45)
    fig.tight_layout()

    # The figure carries the pastel-toned confidence-interval bands (fill_between with
    # pastelize_color above), and "_ci_pastel" is the canonical output name. The plain
    # name is written as a copy for references that omit the suffix. Neither name is
    # cited by V_Main.tex, so renaming these is a generator-local decision.
    plot_name = "est1-4_spec12_phi_t_comparison_ci_pastel.png"
    plot_path = out_dir / plot_name
    plt.savefig(plot_path, dpi=300)
    plt.close(fig)

    shutil.copy(plot_path, out_dir / "est1-4_spec12_phi_t_comparison.png")
    print(f"Exported phi_t plot (pastel CIs) to {plot_path}")

    # ---- 3b) Base-vs-+Time pair graph (alongside the cross-strategy plot above) ----
    # The single index, E3 vs E4. The non-time estimator is a solid BLUE line, the
    # +Time estimator a dashed RED line; both carry their wild-cluster-bootstrap CI
    # bands; the axis is fixed to [0,1].
    def _make_pair_plot(nt_est, t_est, nt_label, t_label, title, out_name):
        figp, axp = plt.subplots(figsize=(11, 5.5))
        drew = False
        for est_num, lab, col, lstyle in ((nt_est, nt_label, "tab:blue", "-"),
                                          (t_est, t_label, "tab:red", "--")):
            bpkl = data_dir / f"ts_link_band_est{est_num}.pkl"
            if not bpkl.exists():
                print(f"  [Warning] no band pkl for est{est_num}; skipping in {out_name}")
                continue
            with open(bpkl, "rb") as fb:
                bd = pickle.load(fb).sort_values("_d")
            axp.plot(bd["_d"], bd["phi_t"], label=lab, color=col, linewidth=2, linestyle=lstyle)
            axp.fill_between(bd["_d"].values, bd["lo"].values, bd["hi"].values, color=col, alpha=0.18, linewidth=0)
            drew = True
        if not drew:
            plt.close(figp)
            return
        axp.set_title(title, fontsize=14, pad=12)
        axp.set_ylabel(r"National $\hat{\phi}_t$")
        axp.set_ylim(0, 1)
        axp.grid(alpha=0.4)
        axp.legend(loc="best")
        figp.tight_layout()
        pp = out_dir / out_name
        plt.savefig(pp, dpi=300)
        plt.close(figp)
        print(f"Exported pair phi_t plot -> {pp}")

    _make_pair_plot(3, 4, "Single-Index", "Single-Index + Time",
                    r"Implied National $\hat{\phi}_t$: Single-Index",
                    "est_phi_t_single_index_pair.png")

if __name__ == "__main__":
    main()
