---
name: post-run-refresh
description: Bring a downloaded cluster run into the local tree and refresh everything downstream in the right order - ingest, tables and figures, paper-number macros, paper-equation macros, slide visuals - and rebuild the advertising register PDF. Use whenever the user says a run finished, landed or was downloaded, asks to ingest, to refresh or regenerate tables, figures or macros, asks whether an exhibit or a quoted number is stale, wants a new number the prose can quote, or has edited ADVERTISING_DATA.md.
---

# Refresh after a run

Each step below reads files the step before it wrote. A skipped step raises no error; the paper
simply keeps quoting the previous vintage. Run only the families that changed, but keep the order.

## Ground rules

- Run repo Python with `C:/venvs/egan/Scripts/python.exe`. Most modules call the venv guard at
  import, and under any other interpreter they re-launch themselves as scripts. Set
  `MPLBACKEND=Agg` for anything that draws.
- Never edit a `.tex` file. A wrong exhibit is fixed in its generator and re-rendered (the
  `paper-tables` skill has the conventions). `V_Main.tex` and the deck are the user's: give
  paste-ready lines instead. The deck is usually open in TeXstudio and saved every few minutes.
- Judge provenance by values or hashes, never by modification times. OneDrive touches mtimes.
- Before generators overwrite tables in `Drafts/Deposit Competition`, keep a copy of the ones
  being replaced (the session scratchpad, or the checkpoint folder in use), so the report can show
  before and after.
- One heavy step at a time. Two CPU-bound jobs on this machine both finish later.

## The chain

### 1. Land the download

The user downloads the cluster's `data/output/download` folder whole into
`<processed>/ESTIMATION_OUTPUT/CLUSTER_IN/<date>/`. The cluster is scratch, so a partial download
cannot be repaired later; say so if a family is missing.

### 2. Ingest

`python cluster_ingest.py --dry-run`, read it, then run it without the flag. `--in <folder>` and
`--family <name>` narrow it. It reassembles split parts, verifies against `sha256SUMS`, and routes
each family:

| Family | Lands in | Written by |
|---|---|---|
| `sleep` | `DEMAND_PREP/` (`est{k}/`, `Rout/`, `DIAGNOSTICS/`) | `cluster_ingest.py` |
| `demand_prep` | `DEMAND_PREP/` | `cluster_ingest.py` |
| `logit` | `BLP_RESULTS/logit/`, the `.tex` exhibits into Drafts | `cluster_ingest.py` |
| `blp` | `BLP_RESULTS/cluster_raw/`, then `cluster_processed/` | `cluster_ingest_blp.py` |
| `bbl` | `BBL_OUTPUT/cluster_raw/` and `cluster_processed/`, policy function to `COST_POLFUNC/` | `cluster_ingest_bbl_cf.py --kind bbl` |
| `counterfactuals` | `CF_OUTPUT/cluster_raw/`, `CF_FOUNDATION/` | `cluster_ingest_bbl_cf.py --kind cf` |
| `gates`, `logs` | `ESTIMATION_OUTPUT/CLUSTER_META/<tag>/` | `cluster_ingest.py` |

Things that have gone wrong here:

- **Two archives of one family in the folder.** Discovery takes the highest job id. A BBL run with
  a multi-start and a single-curve tag has two archives, so ingest each by name:
  `python cluster_ingest_bbl_cf.py --kind bbl --zip <archive> --dry-run`, then without the flag.
- **The policy-function guard.** If the archive's `polfunc_fitted.csv` is byte-identical to the
  local one, the local files are kept. If it differs, the ingest refuses and writes nothing.
  `--accept-cluster-polfunc` replaces the local fit and with it the policy-function tables
  (`polfunc_k4`, `polfunc_k5`), so it is the user's call.
- **BLP collisions.** After a BLP ingest, `BLP_RESULTS/cluster_raw` should hold no `*__*` files
  beyond the two inert relics `blp__blp_summary_E{3,4}_sigma_gpu_ift.json`. Compare one flat
  result json with its zip member. An earlier version of the flattening kept the old file on every
  name collision, and the tables showed the previous run for ten days.
- **The `dx` variant.** `python cluster_ingest_dx.py --zip <blp archive> --zip <bbl archive>`,
  both archives of one run in the same call, under the names they were downloaded with. The
  demand fit, its record and the cost parameters land together or not at all.
- Source archives are only read or copied. Never move or delete one: some are the only copy.

### 3. Check the run before rendering it

- BBL: in each `cost_params_*.json`, `n_shards_found` equals `n_shards_expected`, and the `run`
  block carries the beta, T and tag the paper expects.
- The gate jsons under `CLUSTER_META/<tag>/` all passed; G1's phi-hat matches the promoted values.
- Whatever the ingest reported as refused or missing is resolved or named in the report.

### 4. Render the exhibits

| Family | Commands, in order |
|---|---|
| Sleepiness | `sleep_export_all.py`; `sleep_desc_clusters.py`. `sleep_export_spec12_compare.py` loads four 1 GB pickles: ask first |
| Logit | Nothing: its `.tex` exhibits arrive with the ingest |
| BLP | `make_blp_demand_comparison_table.py`; `make_blp_rc_table.py` |
| Policy function | `bbl_polfunc.py --from-pkl`. It renders from the stored fit. Never re-fit to refresh a table |
| BBL costs | `make_bbl_cost_tables.py --psi-tag <ms tag> --compare-single-curve --single-curve-tag <sc tag>`; then `make_bbl_cost_tables.py --from-psi --psi-tag <ms tag> --psi-zip <that run's archive in BBL_OUTPUT/cluster_raw>`; then `make_bbl_ratio_figure.py --tag <ms tag>` |
| Counterfactuals | `make_cf1_franchise_table.py` |
| `dx` variant | `make_dx_tables.py` (its tables stay in the variant's folder unless `--to-drafts`) |

Read each generator's stdout. `make_bbl_cost_tables.py` lists under `ZERO_PRINTS` every nonzero
value that prints as 0.000; pass that list on rather than rescaling.

### 5. Paper numbers

`python make_paper_numbers.py` must end with return code 0, `0 pending`, and every cross-check
matching the rendered tables. It writes `paper_numbers_macros.tex` (the `\pn[fmt]{key}` macros) to
`ESTIMATION_OUTPUT/PAPER_NUMBERS` and to the paper folder.

- When a new BBL tag becomes the paper's run, the defaults `--bbl-target` and `--sc-target` in the
  script move with it.
- A number the prose should quote gets a key in the collector, never a typed value.
- If the deposit-types table changed, run `make_slide_figures.py --only table` before this step:
  it writes the json the `slide.types.*` keys are read from.

### 6. Compile, then equations

The user compiles `V_Main.tex`. To check page breaks or overfull boxes yourself, compile a copy in
the scratchpad. Then `python make_paper_equations.py`: it takes equation numbers and labels from
`V_Main.aux`, so it must follow the compile, and it never writes the paper.

### 7. Slide visuals

`python make_slide_figures.py`, or `--only <names>` for the ones that changed. The figures that
read `paper_numbers.json` (violations, wedge, coefficients) come after step 5. If the Figure 1
drawing code in `make_desc_compressed_tables.py` (`MS_PANELS`, `ms_series`, `ms_style`) was
touched, re-render the paper figure and compare pixels with the previous one.

### 8. Report

- The headline numbers before and after, with sign and significance changes named.
- Which claims in the paper or the deck no longer match. Those edits are the user's; list them
  with line references rather than making them.
- The `ZERO_PRINTS` list, anything the ingest refused, and every step skipped with the reason.

## Routing a new cluster output

Answer three questions in order before touching the ingest: which `output/` path does the writer
resolve, which family's archive patterns cover that path, and does that family strip the prefix
or flatten. Only `sleep`, `demand_prep`, `logit`, `gates` and `logs` pass through `member_dest()`
in `cluster_ingest.py`; `blp`, `bbl` and `counterfactuals` have their own modules, so a rule added
to `member_dest()` for them never runs. Test through the real entry point (`main()` with
`sys.argv` set and the destination roots pointed at temp folders), with a zip that keeps the
archive's prefix. A direct call to the helper proves nothing about whether it is on the path.

## The advertising register PDF

`ADVERTISING_DATA.md` in this folder is the master. Copies of the md and a compiled PDF sit next
to `V_Main.tex`, and go stale after any edit of the master. The build recipe and its checks are in
`references/register-pdf.md`; the pandoc filter it needs is `scripts/breaks.lua`.
