#!/usr/bin/env bash
# ===========================================================================================
# run_sleep_recompute.sh — post-panel-switch sleepiness recompute + demand prep + tables
# ===========================================================================================
# WHY THIS EXISTS (2026-08-04).  Until ~20:00 today, five estimation scripts resolved their
# input as "market_panel_with_fees.csv if it exists else market_panel.csv".  The fees panel was
# SIX DAYS OLDER than the base panel (07-28 vs 08-03), so every production sleepiness estimate
# and every demand-prep parquet was built from a stale panel.  utils/paths.market_panel_csv()
# now returns market_panel.csv (USE_FEE_PANEL=1 opts back in), so everything downstream of the
# panel has to be rebuilt once.  See identification_notes.md §10.
#
# WHY NOT JUST `python run_sleep_pipeline.py`.  Three reasons:
#   1. E3/E4 are deliberately EXCLUDED (user decision 2026-08-04) — the orchestrator has no flag
#      for that, and running them would cost ~2h for outputs that are not used.
#   2. This box is a 15 W i5-1334U with ~32 GB.  The orchestrator gives no memory pacing; a
#      spike meeting a thin margin is what crashed a run this morning.  Here each estimator is
#      its OWN process (memory fully reclaimed between steps) behind a free-RAM gate.
#   3. Restartability: --from lets you resume at a step instead of redoing hours of work.
#
# ORDERING IS LOAD-BEARING: sleepiness writes est{N}/market_panel_phis.csv -> demand prep
# consumes phi_mt from it -> the parquets feed logit/BLP.  Do not reorder.
#
# USAGE
#   bash run_sleep_recompute.sh                 # everything: E1,E2,E5-E8 -> exports -> prep -> tables
#   bash run_sleep_recompute.sh --from prep     # resume at demand prep (sleepiness already done)
#   bash run_sleep_recompute.sh --dry-run       # print the plan, run nothing
#   MIN_FREE_GB=9 bash run_sleep_recompute.sh   # raise the gate (default 7)
#
# DOES NOT run the logit/BLP; that is run_blp_pipeline.py --logit, deliberately separate.
# ===========================================================================================
set -uo pipefail
cd "$(dirname "$0")" || exit 1

export MPLBACKEND=Agg            # no display on a batch run (a missing font/icon breaks plots)
export PYTHONIOENCODING=utf-8    # Windows console is cp1252; the logs contain ψ, φ, ω
export PYTHONUNBUFFERED=1        # so a tailed log is live, not block-buffered
export DEMAND_PREP_JOBS=1        # 1 estimator at a time: 2-way concurrency OOM'd E4 tonight
unset SLEEP_OUT_ROOT             # explicit: write to PRODUCTION, never a sandbox

MIN_FREE_GB="${MIN_FREE_GB:-7}"
MAX_WAIT_MIN="${MAX_WAIT_MIN:-120}"
FROM="all"; DRY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --from) FROM="${2:-all}"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

free_gb() {
  powershell -NoProfile -Command \
    '[math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB,2)' 2>/dev/null \
    || echo 99
}

# Wait until the machine has room. Guards against a spike meeting a thin margin rather than
# against a single large allocation — that distinction is what actually bit us.
gate() {
  local label="$1" waited=0 f
  while :; do
    f="$(free_gb)"
    if [ "$(awk -v a="$f" -v b="$MIN_FREE_GB" 'BEGIN{print (a+0>=b+0)?1:0}')" = "1" ]; then
      echo "[recompute] ${label}: ${f} GB free — proceeding"; return 0
    fi
    if [ "$waited" -ge "$MAX_WAIT_MIN" ]; then
      echo "[recompute] ${label}: still ${f} GB after ${waited} min — PROCEEDING ANYWAY (gate=${MIN_FREE_GB})"
      return 0
    fi
    echo "[recompute] ${label}: ${f} GB < ${MIN_FREE_GB} GB — waiting 60s (${waited}/${MAX_WAIT_MIN} min)"
    sleep 60; waited=$((waited + 1))
  done
}

# step <key> <label> <script> [args...]
run_step() {
  local key="$1" label="$2"; shift 2
  case "$FROM" in
    all) ;;
    "$key") FROM="all" ;;                       # resume point reached
    *) echo "[recompute] SKIP  ${label}  (--from ${FROM})"; return 0 ;;
  esac
  if [ "$DRY" = "1" ]; then echo "[recompute] would run: ${label}  ->  python -u $*"; return 0; fi
  gate "$label"
  echo "[recompute] ===== ${label}  START $(date '+%F %H:%M:%S') ====="
  python -u "$@"
  local rc=$?
  echo "[recompute] ===== ${label}  END   $(date '+%F %H:%M:%S')  rc=${rc} ====="
  if [ "$rc" -ne 0 ]; then
    echo "[recompute] !! ${label} FAILED (rc=${rc})."
    case "$key" in
      # A failed ESTIMATOR poisons everything downstream (prep reads its phis) -> stop.
      e1|e2|e5|e6|e7|e8) echo "[recompute] !! aborting: downstream steps consume this estimator's phi."; exit "$rc" ;;
      *) echo "[recompute] continuing (non-fatal step)." ;;
    esac
  fi
}

echo "[recompute] start $(date '+%F %H:%M:%S')"
echo "[recompute] panel   : $(python -c 'from utils import paths; print(paths.market_panel_csv())' 2>/dev/null)"
echo "[recompute] gate    : ${MIN_FREE_GB} GB free, max wait ${MAX_WAIT_MIN} min | free now $(free_gb) GB"
echo "[recompute] from    : ${FROM}   dry-run: ${DRY}"
echo "[recompute] NOTE    : E3/E4 excluded by design (2026-08-04)."

# --- 1. sleepiness, one process each (E3/E4 excluded) --------------------------------------
run_step e1 "E1 local B-type"            estimation_1_sleep.py
run_step e2 "E2 pooled B+D linear"       estimation_2_sleep.py
run_step e5 "E5 pooled single-index"     estimation_sleep_common.py --est 5
run_step e6 "E6 single-index + time"     estimation_sleep_common.py --est 6
run_step e7 "E7 joint sieve"             estimation_sleep_common.py --est 7
run_step e8 "E8 joint sieve + time"      estimation_sleep_common.py --est 8

# --- 2. exports, demand prep, tables -------------------------------------------------------
# export_results and demand prep take --estimation all; E3/E4 simply keep their existing (07-28)
# outputs, which is the intended behaviour since we are not recomputing them.
run_step exports "Export 1st/2nd stage summaries" export_results.py --estimation all
run_step prep    "Demand prep (all estimators)"   estimation_demand_1_prep.py --estimation all --spec all
run_step spec12  "Analyze spec 12"                export_analyze_spec12.py
run_step desc3   "Cluster-imbalance table"        desc_3.py

echo "[recompute] ALL DONE $(date '+%F %H:%M:%S')"
echo "[recompute] next: python run_blp_pipeline.py --logit"
