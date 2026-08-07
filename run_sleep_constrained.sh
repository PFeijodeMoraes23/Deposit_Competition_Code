#!/usr/bin/env bash
# ===========================================================================================
# run_sleep_constrained.sh — re-estimate E1/E2/E5-E8 with the shape-constrained link
# ===========================================================================================
# WHAT THIS IS FOR.  phi is a probability, so the single-index link G must be a CDF: monotone
# and valued in [0,1] (V_Main.tex eq.(6)/:296, :408, :414, and :507 which divides by 1-phi).
# The link is now estimated under those constraints — a monotone I-spline with beta >= 0 and
# sum(beta) <= 1 — instead of being clipped at the reporting layer.  This script produces a
# COMPLETE parallel vintage of the sleepiness outputs under that estimator so it can be
# compared against the existing one cell by cell.
#
# SANDBOX ONLY, BY CONSTRUCTION.  SLEEP_OUT_ROOT is set to a scratch root and is never unset,
# so nothing here can write to PROCESSED/ESTIMATION_OUTPUT/DEMAND_PREP.  It also never writes
# to C:\egan_trimmed, which is the read-only comparator.  Contrast run_sleep_recompute.sh,
# which deliberately `unset SLEEP_OUT_ROOT` to target production.
#
# USAGE
#   bash run_sleep_constrained.sh                     # E1,E2,E5-E8 into C:/egan_constrained
#   OUT=C:/egan_other bash run_sleep_constrained.sh   # different sandbox root
#   bash run_sleep_constrained.sh --from e5           # force the start point
#   bash run_sleep_constrained.sh --dry-run           # print the plan, run nothing
#   ONLY="e5 e6" bash run_sleep_constrained.sh        # just those steps
#   bash run_sleep_constrained.sh --bands             # also build the unconditional bands
#   bash run_sleep_constrained.sh --fresh             # recompute everything, ignore what is on disk
#
# STOP AND RESUME.  Ctrl-C (or a reboot) at any point, then re-run the SAME command tomorrow:
# it continues where it stopped, at SPEC granularity, with no flags to remember.
#   * each estimator writes its pickle after EVERY state block, so an interruption costs at
#     most the block in flight (<= 4 specs), not the estimator;
#   * SLEEP_RESUME=1 (set below) makes an estimator skip specs already on disk, and a stored
#     spec still feeds the warm-start chain, so a resumed grid follows the same path as an
#     uninterrupted one;
#   * a spec fitted under a DIFFERENT SLEEP_LINK_CONSTRAINED setting is recomputed rather than
#     reused, so a resume can never blend two estimators inside one pickle;
#   * completed estimators are skipped wholesale by the marker check below.
# `--fresh` turns all of that off. Use it for a clean reproduction run.
#
# E3/E4 are excluded, as in run_sleep_recompute.sh: they are not reported, and E5/E6 derive
# their direction from an internally-fitted logit rather than from a stored est3/est4.
# ===========================================================================================
set -uo pipefail
cd "$(dirname "$0")" || exit 1

OUT="${OUT:-C:/egan_constrained}"
export SLEEP_OUT_ROOT="${OUT}/DEMAND_PREP"
export SLEEP_LINK_CONSTRAINED=1   # the point of this run
export SLEEP_WCB_MODE="${SLEEP_WCB_MODE:-wcu-t}"   # bootstrap-t reference, not normal
# KEEP THE LS LOSS. estimation_sleep_common defaults SLEEP_DROP_LS=1, which computes the robust
# (Cauchy-direction) fit only. Every comparison this run exists to support is robust-vs-LS: the
# stored vintage carries second_stage_ls for each spec, the loss ordering is measured across the
# 12 instrumented cells, and the briefing's headline cell is E5/LS. A robust-only vintage cannot
# be compared with it. Computing both in ONE pass also shares the frame build and first stage,
# which a robust pass followed by an SLEEP_LS_ONLY pass would pay for twice.
export SLEEP_DROP_LS="${SLEEP_DROP_LS:-0}"
export MPLBACKEND=Agg             # a dehydrated matplotlib icon breaks batch runs
export PYTHONIOENCODING=utf-8     # the logs contain phi/theta
export PYTHONUNBUFFERED=1         # so a tailed log is live
export DEMAND_PREP_JOBS=1

MIN_FREE_GB="${MIN_FREE_GB:-7}"
MAX_WAIT_MIN="${MAX_WAIT_MIN:-120}"
FROM="all"; DRY=0; BANDS=0; FRESH=0
while [ $# -gt 0 ]; do
  case "$1" in
    --from) FROM="${2:-all}"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    --bands) BANDS=1; shift ;;
    --fresh) FRESH=1; shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
export SLEEP_RESUME=$([ "$FRESH" = "1" ] && echo 0 || echo 1)

# A step is COMPLETE when its estimator wrote the artifact that only exists after the whole
# grid finished (the phi CSV, written last). The pickle alone is not evidence of completion --
# per-block checkpointing means a partial grid also has one.
step_done() {
  case "$1" in
    e1|e2|e5|e6|e7|e8) [ -s "${SLEEP_OUT_ROOT}/est${1#e}/market_panel_phis.csv" ] ;;
    band5) [ -s "${SLEEP_OUT_ROOT}/Rout/ts_link_band_est5_uncond_ls.pkl" ] ;;
    band6) [ -s "${SLEEP_OUT_ROOT}/Rout/ts_link_band_est6_uncond_ls.pkl" ] ;;
    *) return 1 ;;
  esac
}

free_gb() {
  powershell -NoProfile -Command \
    '[math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB,2)' 2>/dev/null \
    || echo 99
}

gate() {
  local label="$1" waited=0 f
  while :; do
    f="$(free_gb)"
    if [ "$(awk -v a="$f" -v b="$MIN_FREE_GB" 'BEGIN{print (a+0>=b+0)?1:0}')" = "1" ]; then
      echo "[constrained] ${label}: ${f} GB free — proceeding"; return 0
    fi
    if [ "$waited" -ge "$MAX_WAIT_MIN" ]; then
      echo "[constrained] ${label}: still ${f} GB after ${waited} min — PROCEEDING ANYWAY"
      return 0
    fi
    echo "[constrained] ${label}: ${f} GB < ${MIN_FREE_GB} GB — waiting 60s (${waited}/${MAX_WAIT_MIN})"
    sleep 60; waited=$((waited + 1))
  done
}

_run() {
  local key="$1" label="$2"; shift 2
  if [ -n "${ONLY:-}" ]; then
    case " ${ONLY} " in *" ${key} "*) ;; *) echo "[constrained] SKIP  ${label} (ONLY)"; return 0 ;; esac
  fi
  case "$FROM" in
    all) ;;
    "$key") FROM="all" ;;
    *) echo "[constrained] SKIP  ${label}  (--from ${FROM})"; return 0 ;;
  esac
  if [ "$FRESH" != "1" ] && step_done "$key"; then
    echo "[constrained] DONE  ${label}  (already complete — re-run with --fresh to redo)"
    return 0
  fi
  if [ "$DRY" = "1" ]; then echo "[constrained] would run: ${label}  ->  $*"; return 0; fi
  gate "$label"
  echo "[constrained] ===== ${label}  START $(date '+%F %H:%M:%S') ====="
  "$@"
  local rc=$?
  echo "[constrained] ===== ${label}  END   $(date '+%F %H:%M:%S')  rc=${rc} ====="
  if [ "$rc" -ne 0 ]; then
    echo "[constrained] !! ${label} FAILED (rc=${rc})."
    case "$key" in
      e1|e2|e5|e6|e7|e8) echo "[constrained] !! aborting: later steps consume this estimator."; exit "$rc" ;;
      *) echo "[constrained] continuing (non-fatal step)." ;;
    esac
  fi
}
run_step() { _run "$1" "$2" python -u "${@:3}"; }

mkdir -p "${SLEEP_OUT_ROOT}"
echo "[constrained] start $(date '+%F %H:%M:%S')"
echo "[constrained] out root : ${SLEEP_OUT_ROOT}   (production is NOT touched)"
echo "[constrained] link      : SLEEP_LINK_CONSTRAINED=${SLEEP_LINK_CONSTRAINED} (monotone I-spline, beta>=0, sum(beta)<=1)"
echo "[constrained] wcb mode  : SLEEP_WCB_MODE=${SLEEP_WCB_MODE}"
echo "[constrained] losses    : SLEEP_DROP_LS=${SLEEP_DROP_LS} (0 = compute BOTH robust and LS)"
echo "[constrained] panel     : $(python -c 'from utils import paths; print(paths.market_panel_csv())' 2>/dev/null)"
echo "[constrained] gate      : ${MIN_FREE_GB} GB free | free now $(free_gb) GB"
echo "[constrained] resume    : SLEEP_RESUME=${SLEEP_RESUME} (stop any time; re-run this same"
echo "[constrained]             command to continue at spec granularity. --fresh to start over)"
echo "[constrained] from      : ${FROM}   dry-run: ${DRY}   bands: ${BANDS}   fresh: ${FRESH}"

# --- estimators, one process each ----------------------------------------------------------
# E1/E2 are LINEAR and carry no link, so the constrained flag does not change them. They are
# re-run anyway so the whole vintage is same-frame and same-inference (WCU-t) for comparison.
run_step e1 "E1 local B-type"        estimation_1_sleep.py
run_step e2 "E2 pooled B+D linear"   estimation_2_sleep.py
run_step e5 "E5 single-index"        estimation_sleep_common.py --est 5
run_step e6 "E6 single-index + time" estimation_sleep_common.py --est 6

# E7/E8: SPEC 12 ONLY by default, matching the comparator vintage in C:\egan_trimmed (its est7
# and est8 pickles hold exactly one spec, IV_HausmanFull x Tech, while est5/est6 hold 8). The
# joint sieve runs a Julia theta search per spec at roughly 50 min, so a full 8-spec grid is
# ~7 h per estimator and would produce cells the comparator has nothing to compare against.
# E78_FULL=1 runs the whole grid instead.
_E78_ARGS=$([ "${E78_FULL:-0}" = "1" ] && echo "" || echo "--spec12")
run_step e7 "E7 joint sieve${_E78_ARGS:+ (spec 12)}"        estimation_sleep_common.py --est 7 ${_E78_ARGS}
run_step e8 "E8 joint sieve + time${_E78_ARGS:+ (spec 12)}" estimation_sleep_common.py --est 8 ${_E78_ARGS}

# --- unconditional bands (opt-in: the expensive part) ---------------------------------------
# Each cell re-profiles the constrained link once per draw, and additionally emits the
# tangent-cone band, which is the valid construction when shape constraints are active
# (Andrews 2000) — they are active in every cell measured so far.
if [ "$BANDS" = "1" ]; then
  for e in 5 6; do
    run_step "band${e}" "Unconditional band E${e} (both losses)" \
      estimation_uncond_band.py --est "${e}" --loss both
  done
fi

echo "[constrained] ALL DONE $(date '+%F %H:%M:%S')"
echo "[constrained] outputs under ${SLEEP_OUT_ROOT}"
echo "[constrained] compare against C:/egan_trimmed/DEMAND_PREP (read-only comparator)"
