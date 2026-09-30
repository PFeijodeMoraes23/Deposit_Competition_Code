"""utils/paths.py — single source of truth for every data path in the
Egan_et_al_2025_Rep pipeline.

Author: Pedro Feijó de Moraes

Every scrape_*/panel_*/estimation_*/analysis_* script imports its data
locations from here instead of recomputing ``parents[2]`` /
``os.path.join(..., "..", "..")`` inline.  Move a data tree once, edit it once
here, and the whole pipeline follows.

Layout anchor
-------------
    OPEN_FINANCE = .../Open-Finance          shared data root (3 levels up)
    DATA_ROOT    = OPEN_FINANCE/BCB/Egan_et_al_2025_Rep
    RAW          = DATA_ROOT/raw             consolidated *raw downloads*
    PROCESSED    = DATA_ROOT/processed       pipeline outputs (deposits_panel,
                                             market_panel, PANEL_INTERMED, ...)

Raw downloads for every source live under ``RAW/<SOURCE>``.  Per-source
*processed* panel CSVs (anatel_mca_panel, cadunico_mca_panel, IBGE demographics
panels, BCB Inclusion panels, PIX ``pix_*_panel``) stay in their historical
per-source directories and are exposed below as ``<SOURCE>_DIR``.

The anchor can be overridden with the ``OPEN_FINANCE_ROOT`` environment variable
(useful on HPC / a different machine).  This is independent of, and composes
with, the per-script ``resolve_script_paths`` (TOON) override layer: scripts
pass the constants below as *defaults*, which TOON context may still override.

All values are :class:`pathlib.Path`.  ``os.path.join`` accepts Path objects, so
both ``os.path``-style and ``pathlib``-style call sites keep working.
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Anchor
# ---------------------------------------------------------------------------
# utils/paths.py -> utils -> Egan_et_al_2025_Rep -> Code -> Open-Finance
_ENV_ROOT = os.environ.get("OPEN_FINANCE_ROOT", "").strip()
OPEN_FINANCE: Path = (
    Path(_ENV_ROOT).expanduser().resolve()
    if _ENV_ROOT
    else Path(__file__).resolve().parents[3]
)

BCB: Path       = OPEN_FINANCE / "BCB"
DATA_ROOT: Path = BCB / "Egan_et_al_2025_Rep"
RAW: Path       = DATA_ROOT / "raw"
PROCESSED: Path = DATA_ROOT / "processed"

# Shared cross-project datasets (ESTBAN / IF_DATA / COSIF / PIX) resolve via the
# repo-wide config (Open-Finance/Code/of_paths.py) so they follow the per-project +
# shared/ reorg (flip with OF_USE_SHARED). Falls back to the in-repo raw/ locations
# if of_paths isn't importable (e.g. a standalone checkout of this repo).
try:
    import sys as _sys
    _CODE_DIR = str(OPEN_FINANCE / "Code")
    if _CODE_DIR not in _sys.path:
        _sys.path.insert(0, _CODE_DIR)
    import of_paths as _ofp
except Exception:
    _ofp = None


def data_root() -> str:
    """Backwards-compatible accessor: the shared data root (…/Open-Finance) as a
    string, matching the historical ``disclosure_common.data_root(__file__)``."""
    return str(OPEN_FINANCE)


def demand_prep_root() -> Path:
    """Directory holding the per-estimator sleepiness outputs (``est1`` … ``est4``).

    Normally ``PROCESSED/ESTIMATION_OUTPUT/DEMAND_PREP``. Setting ``SLEEP_OUT_ROOT``
    redirects the whole tree somewhere else, which is how a re-estimation can be run
    WITHOUT touching production results.

    That matters here for two reasons beyond ordinary caution. Adding a column to a
    stored band requires re-fitting the cell, and the index direction has several
    near-equivalent optima (2026-08-03 probe), so a rerun can land in a different basin
    than the fit the uploaded cluster parquets descend from. And the estimators write
    ~2 GB of derived CSV per grid, which is better kept off OneDrive. Point a sandbox run
    at a local disk, compare, and promote by explicit copy only if the comparison is clean.

    INPUTS ARE UNAFFECTED -- they resolve through the other anchors in this module, so
    a redirected run reads exactly the same panels as production.
    """
    override = os.environ.get("SLEEP_OUT_ROOT", "").strip()
    if override:
        return Path(override).expanduser()
    return PROCESSED / "ESTIMATION_OUTPUT" / "DEMAND_PREP"


def demand_parquet_dir() -> Path:
    """Directory holding the demand-prep parquets (``demand_{k}[_tag]_spec_{s}.parquet``).

    ``DEMAND_PREP_DIR`` names the cluster's ``data/output/demand_prep`` step folder, the one
    place the on-cluster prep step writes them. Unset — every local run — this is
    :func:`demand_prep_root`, where the parquets sit beside the ``est{k}`` fits they are
    built from.

    It is a separate accessor from :func:`demand_prep_root` because the two coincide only
    locally: on the cluster the ``est{k}`` pickles belong to the sleep step folder and the
    parquets to their own. Resolving the parquets through one name with one producer is what
    keeps the exporter and the CF/BBL stack from reading two vintages of the same routine.
    """
    override = os.environ.get("DEMAND_PREP_DIR", "").strip()
    if override:
        return Path(override).expanduser()
    return demand_prep_root()


def estimation_output() -> Path:
    """Root of the estimation output tree (``PROCESSED/ESTIMATION_OUTPUT``).

    Production location in every run: it also holds the BLP, counterfactual and cost
    trees, which a sleepiness sandbox has no business redirecting. The objects a
    sleepiness run does own resolve through ``demand_prep_root``/``est_dir``/``rout_dir``.
    """
    return PROCESSED / "ESTIMATION_OUTPUT"


def est_dir(n: int | str) -> Path:
    """Per-estimator sleepiness directory ``est{n}`` of the ACTIVE vintage.

    Resolves under :func:`demand_prep_root`, so ``SLEEP_OUT_ROOT`` carries the whole
    lineup with it. Resolve ``est{n}`` this way rather than off ``PROCESSED``: a script
    that pins production reads an unrelated vintage during a sandboxed run, and since a
    missing pickle is a skip in most of the export/demand-prep chain, it reports success
    on empty output.
    """
    return demand_prep_root() / f"est{n}"


def rout_dir() -> Path:
    """``Rout`` — TeX fragments, figures and cached bands — for the ACTIVE vintage.

    ``estimation_output()/Rout``, or ``<SLEEP_OUT_ROOT>/Rout`` beside the per-estimator
    directories, so a sandboxed run's exhibits describe its own fits instead of
    overwriting the production ones. Created on demand.
    """
    override = os.environ.get("SLEEP_OUT_ROOT", "").strip()
    d = (Path(override).expanduser() / "Rout") if override else (estimation_output() / "Rout")
    d.mkdir(parents=True, exist_ok=True)
    return d


def blp_results_dir() -> Path:
    """``ESTIMATION_OUTPUT/BLP_RESULTS`` — the demand-estimation tree (``logit/``,
    ``cluster_raw/``, ``cluster_processed/``).

    Production location in every run. These are CLUSTER artifacts and are not written per
    sleepiness vintage, so they resolve through :func:`estimation_output` and a sandboxed
    run reads the same results the paper reports.
    """
    return estimation_output() / "BLP_RESULTS"


def bbl_output_dir() -> Path:
    """``ESTIMATION_OUTPUT/BBL_OUTPUT`` — the dynamic-cost tree (``cluster_raw/psi_cost.zip``,
    ``cluster_processed/cost_params_*.json``). Cluster artifacts, so production as above."""
    return estimation_output() / "BBL_OUTPUT"


def cf_foundation_dir() -> Path:
    """``ESTIMATION_OUTPUT/CF_FOUNDATION`` — the counterfactual inputs (``upsilon_pix_E*.json``,
    ``phi_nopix_E*.parquet``). Production CF input, not sandbox output.

    ``CF_FOUNDATION_DIR`` points it at the cluster's ``data/output/counterfactuals`` step
    folder, where the Υ_pix export writes and CF4 reads. Unset locally, so the default stands.
    """
    override = os.environ.get("CF_FOUNDATION_DIR", "").strip()
    if override:
        return Path(override).expanduser()
    return estimation_output() / "CF_FOUNDATION"


def cost_fwd_dir() -> Path:
    """``ESTIMATION_OUTPUT/COST_FWD`` — the forward-cost input the BBL simulation reads
    (``forward_rf_qoq.csv``) and the ψ deviation shards it writes for the solve step.

    ``CF_COST_FWD`` points it at the cluster's ``data/output/bbl`` step folder; the BBL jobs
    export that name, and bbl_solve.py reads the same variable. Unset locally,
    so the default stands.
    """
    override = os.environ.get("CF_COST_FWD", "").strip()
    if override:
        return Path(override).expanduser()
    return estimation_output() / "COST_FWD"


def polfunc_dir() -> Path:
    """``ESTIMATION_OUTPUT/COST_POLFUNC`` — the fitted pricing policy function
    (``polfunc_fitted.csv`` and its pkl/json/tex siblings) the BBL forward simulation reads.

    ``COST_POLFUNC_DIR`` points it at the cluster's ``data/output/bbl`` step folder, where the
    polfunc pre-step writes and the forward simulation picks it up via ``--policy-csv``. Unset
    locally, and pinned to :func:`estimation_output` rather than ``SLEEP_OUT_ROOT``: the policy
    function is spec-invariant, so it belongs to no sleepiness vintage.
    """
    override = os.environ.get("COST_POLFUNC_DIR", "").strip()
    if override:
        return Path(override).expanduser()
    return estimation_output() / "COST_POLFUNC"


def diag_dir(name: str) -> Path:
    """``ESTIMATION_OUTPUT/DIAG_{name}`` — one diagnostic battery's output tree.

    ``name`` is the suffix, so ``diag_dir("PHI_SEPARATION")`` resolves the identification
    battery's CSVs and figures and ``diag_dir("K4_SPREAD")`` the type-4 spread probes. These
    are archival diagnostics, keyed to the production fits the notes and the paper quote, so
    they resolve through :func:`estimation_output`.

    The per-vintage ``DIAGNOSTICS`` tree is a DIFFERENT object: it holds cross-checks of a
    particular run's bands and resolves under :func:`demand_prep_root`, beside the bands it
    checks, so it follows ``SLEEP_OUT_ROOT``.
    """
    return estimation_output() / f"DIAG_{name}"


def drafts_dir() -> Path:
    """Paper directory the exporters mirror their .tex fragments and figures into.

    Follows the ``OPEN_FINANCE`` anchor (so ``OPEN_FINANCE_ROOT`` moves it too) and is
    never created here: the paper directory is authored, and a copy into a path that does
    not exist should fail rather than scatter fragments into a fresh empty tree.
    """
    return OPEN_FINANCE / "Drafts" / "Deposit Competition"


# ---------------------------------------------------------------------------
# Raw download trees (consolidated under RAW/)
# ---------------------------------------------------------------------------
ESTBAN_DIR: Path   = (_ofp.ESTBAN if _ofp else RAW / "ESTBAN")    # whole ESTBAN tree (raw monthly CSVs + ESTBAN.csv)
IF_DATA_ROOT: Path = (_ofp.IF_DATA if _ofp else RAW / "IF_DATA")  # whole IF-Data tree
COSIF_RAW: Path    = (_ofp.COSIF if _ofp else RAW / "COSIF_RAW")
SGS_RAW: Path      = RAW / "SGS_RAW"
ANATEL_RAW: Path   = RAW / "ANATEL"      # raw acessos_telefonia_movel_* downloads only
CADUNICO_RAW: Path = RAW / "CadUnico"    # raw cadunico_YYYYMM.csv downloads only
PIX_RAW: Path      = RAW / "PIX"         # raw Pix source files only
IBGE_RAW: Path     = RAW / "IBGE"        # raw IBGE downloads (zips + census raw CSVs)

# ESTBAN sub-paths
ESTBAN_CSV: Path     = ESTBAN_DIR / "ESTBAN.csv"
ESTBAN_RAW_MUN: Path = ESTBAN_DIR / "Relatório por município"
ESTBAN_RAW_AG: Path  = ESTBAN_DIR / "Relatório por município e agência"

# IF-Data sub-paths
IF_DATA_LIST: Path       = IF_DATA_ROOT / "List"
IF_DATA_PRUDENTIAL: Path = IF_DATA_ROOT / "Prudential Conglomerates"
IF_DATA_FINANCIAL: Path  = IF_DATA_ROOT / "Financial Conglomerates"
IF_DATA_INDIVIDUAL: Path = IF_DATA_ROOT / "Individual Institutions"
IF_DATA_AGG: Path        = IF_DATA_ROOT / "Aggregated Data"

# ---------------------------------------------------------------------------
# Per-source directories that KEEP their processed panels in place
# (only the raw downloads above were consolidated into RAW/)
# ---------------------------------------------------------------------------
ANATEL_DIR: Path    = OPEN_FINANCE / "ANATEL"     # anatel_mca_panel.csv / anatel_muni_panel.csv
CADUNICO_DIR: Path  = OPEN_FINANCE / "CadUnico"   # cadunico_mca_panel.csv / cadunico_muni_panel.csv
IBGE_DIR: Path      = OPEN_FINANCE / "IBGE"       # *_demographics_panel.csv, muni/region panels
INCLUSION_DIR: Path = BCB / "Inclusion"           # bcb_inclusion_* / bcb_banked_* panels (pure processed)
PIX_DIR: Path       = (_ofp.PIX if _ofp else BCB / "PIX")         # pix_*_panel.csv processed outputs

# BCB Tarifas (not moved; raw/ was empty — real data is in processed/ + cache/)
TARIFAS_DIR: Path   = BCB / "Tarifas"
TARIFAS_PROC: Path  = TARIFAS_DIR / "processed"
TARIFAS_CACHE: Path = TARIFAS_DIR / "cache"
TARIFAS_RAW: Path   = TARIFAS_DIR / "raw"

# Service quality: BCB complaints ranking, app-store ratings, Reclame Aqui.
# Raw subtrees are created by their scrapers: complaints/, app_play/, app_ios/,
# reclameaqui/, wayback_cdx/, wayback_html/.
QUALITY_DIR: Path  = BCB / "Quality"
QUALITY_RAW: Path  = QUALITY_DIR / "raw"
QUALITY_PROC: Path = QUALITY_DIR / "processed"

# Awareness, advertising and perception series: the COSIF advertising accounts, advertising
# lines from securities filings, and state-owned banks' own advertising disclosures.
AWARENESS_DIR: Path  = BCB / "Awareness"
AWARENESS_RAW: Path  = AWARENESS_DIR / "raw"
AWARENESS_PROC: Path = AWARENESS_DIR / "processed"

# ---------------------------------------------------------------------------
# Stage-6 firm disclosures & other sources (unchanged location)
# ---------------------------------------------------------------------------
FIRM_DISCLOSURES: Path = OPEN_FINANCE / "FirmDisclosures"
WORLDBANK: Path        = OPEN_FINANCE / "WorldBank"
FGC: Path              = OPEN_FINANCE / "FGC"
ACCOUNTS: Path         = BCB / "Accounts"


def market_panel_csv(processed: Path | None = None) -> Path:
    """The market panel every estimation script reads. ONE source of truth.

    Returns `market_panel.csv` — the panel that panel_3/…/panel_10/panel_12 build and that
    carries every estimated regressor and instrument, including `estban_rival_branches_lag`.

    HISTORY (2026-08-04). Five scripts previously resolved this as
    `market_panel_with_fees.csv if it exists else market_panel.csv`. That silently pinned the
    whole pipeline to whichever vintage of the fees panel happened to sit on disk: on
    2026-08-04 the base panel was 6 days NEWER (08-03 vs 07-28), so every demand-prep parquet,
    the logit, the BLP and the sleepiness estimates were built from stale data. The fees panel
    is `market_panel.csv` PLUS 15 fee columns and nothing else (verified by column diff), and
    those 15 columns are referenced by no `.jl`, no sleepiness estimator, and neither X_COLS
    nor D_COLS — they reach the parquets only via `IV_FEE ⊂ EXTRA_KEEP_COLS`, which is guarded
    by `if c in df_base.columns` and so degrades safely when they are absent.

    Set `USE_FEE_PANEL=1` to opt back into `market_panel_with_fees.csv` (e.g. to estimate a fee
    specification). It is an explicit opt-in, never an implicit fallback: if you ask for it and
    it is missing, this raises rather than quietly handing back a different file.
    """
    d = PROCESSED if processed is None else Path(processed)
    base = d / "market_panel.csv"
    if os.environ.get("USE_FEE_PANEL", "").strip() in ("1", "true", "True", "yes"):
        fees = d / "market_panel_with_fees.csv"
        if not fees.exists():
            raise FileNotFoundError(
                f"USE_FEE_PANEL=1 but {fees} does not exist. Build it with "
                "`python panel_fee_merge.py --patch-market` (it must run AFTER "
                "panel_7/panel_10/panel_12), or unset USE_FEE_PANEL to use market_panel.csv.")
        return fees
    return base


def on_cluster() -> bool:
    """True when running inside the cluster's step-directory layout.

    cluster_lib.sh's ``cl_export_step_dirs`` exports ``CL_DATA_OUT`` for every job and
    nothing off-cluster sets it. The table exporters branch on this to decide where a
    fragment belongs: on the cluster the archiver packages ``data/output/sleep/Rout``, so a
    .tex written anywhere else never comes home in the download; locally the paper
    directory is the only copy that matters.
    """
    return bool(os.environ.get("CL_DATA_OUT", "").strip())


def sleep_out_root_set() -> bool:
    """True when ``SLEEP_OUT_ROOT`` is exported to a non-empty value -- the same check
    :func:`demand_prep_root`/:func:`rout_dir` make internally to decide whether they are
    pointed at a sandbox.

    ``cl_export_step_dirs`` exports it too, to the cluster step's own ``data/output/sleep``
    folder, at the same time as ``CL_DATA_OUT`` (so :func:`on_cluster` is also true there).
    A caller whose own default is :func:`drafts_dir` -- which never follows
    ``SLEEP_OUT_ROOT`` -- therefore cannot treat this flag alone as "write beside the
    sandbox instead of Drafts": on the cluster that would change behaviour that has nothing
    to do with sandboxing. Combine it with :func:`on_cluster` at the call site to keep the
    two cases distinct, the way each exporter's existing cluster routing requires.
    """
    return bool(os.environ.get("SLEEP_OUT_ROOT", "").strip())


__all__ = [
    "OPEN_FINANCE", "BCB", "DATA_ROOT", "RAW", "PROCESSED", "data_root", "market_panel_csv",
    "demand_prep_root", "demand_parquet_dir", "estimation_output", "est_dir", "rout_dir",
    "drafts_dir", "blp_results_dir", "bbl_output_dir", "cf_foundation_dir", "cost_fwd_dir",
    "polfunc_dir", "diag_dir", "on_cluster", "sleep_out_root_set",
    "ESTBAN_DIR", "ESTBAN_CSV", "ESTBAN_RAW_MUN", "ESTBAN_RAW_AG",
    "IF_DATA_ROOT", "IF_DATA_LIST", "IF_DATA_PRUDENTIAL", "IF_DATA_FINANCIAL", "IF_DATA_INDIVIDUAL", "IF_DATA_AGG",
    "COSIF_RAW", "SGS_RAW",
    "ANATEL_RAW", "CADUNICO_RAW", "PIX_RAW", "IBGE_RAW",
    "ANATEL_DIR", "CADUNICO_DIR", "IBGE_DIR", "INCLUSION_DIR", "PIX_DIR",
    "TARIFAS_DIR", "TARIFAS_PROC", "TARIFAS_CACHE", "TARIFAS_RAW",
    "QUALITY_DIR", "QUALITY_RAW", "QUALITY_PROC",
    "AWARENESS_DIR", "AWARENESS_RAW", "AWARENESS_PROC",
    "FIRM_DISCLOSURES", "WORLDBANK", "FGC", "ACCOUNTS",
]
