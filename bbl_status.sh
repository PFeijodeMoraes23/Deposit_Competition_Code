#!/bin/bash
# ==============================================================================
# bbl_status.sh — one screen of BBL state, one line per routine.
#
# LOGIN-NODE SAFE. It runs squeue, sacct, ls, grep, sed and awk over small text files, and
# nothing else: no Python, no Julia, no sbatch, and it writes nothing.
#
#   bash bbl_status.sh                        the newest run's routines (psi tag auto-detected)
#   bash bbl_status.sh --routines "3 4" --psi-tag _ms1
#   bash bbl_status.sh --rss                  + MaxRSS of the COMPLETED fwd jobs (sacct .batch)
#   bash bbl_status.sh --probe                = bash bbl_sizing.sh: the memory probe -> MEM/PACK + launch
#
# COLUMNS
#   shards   psi_dev files of this run on disk / N_SHARDS. Names only: the sweep is what opens
#            them, so a truncated file counts here and is caught there.
#   fwd      R/P/F = running / pending / failed-timed-out-OOM fwd jobs since the launch (array
#            tasks; a packed task is k shards).
#   sweep    the last attempt and its outcome (events.log), or its queue state when live.
#   solve    queue state (and reason), else its last sacct state.
#   next     what, if anything, a person has to do. "wait" means the chain is handling it.
#
#   --rss reuses the recipe that produced the 2026-09 baseline (median 164 / max 179 GiB):
#     ids=$(sacct -u $USER -S <date> -X -n -P --name=<fwd names> --format=JobID,Partition,State \
#           | awk -F'|' '$2 ~ /gpu/ && $3=="COMPLETED"{print $1}' | paste -sd,)
#     sacct -j "$ids" -n -P --format=JobID,MaxRSS          # keep the .batch rows; units K/M/G
#   A PACKED job's .batch MaxRSS covers all k shards of the job; the per-shard peak is each
#   shard's own "[BBL] shard i peak RSS X GiB" line, which every packed job also summarises as
#   "[pack] ... peak_rss_gib=", reported beside it.
# ==============================================================================
set -uo pipefail
CL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${CL_DIR}/cluster_lib.sh"

ROUTINES="${CL_ROUTINES_CF}"; STAGE="${CF_STAGE:-extended}"
TAG=""; TAG_SET=0; DO_RSS=0; DO_PROBE=0; ROUTINES_SET=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --routines) ROUTINES="$2"; ROUTINES_SET=1; shift ;;
        --psi-tag)  TAG="$2"; TAG_SET=1; shift ;;
        --stage)    STAGE="$2"; shift ;;
        --rss)      DO_RSS=1 ;;
        --probe)    DO_PROBE=1 ;;
        -h|--help)  sed -n '2,29p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown option: $1 (see -h)" >&2; exit 2 ;;
    esac
    shift
done

ME="${USER:-$(id -un)}"
BD="${CL_STEP_BBL}"
DROOT="${CL_BBL_DISPATCH_ROOT:-${BD}/.dispatch}"
LOGS="${BBL_LOG_DIR:-${CL_ROOT}/logs}"
HAVE_SLURM=0; command -v squeue >/dev/null 2>&1 && HAVE_SLURM=1

# One value out of a context.env (printf %q output: plain tokens, or '' for empty).
_ctxval () { sed -n "s/^$2=//p" "$1" 2>/dev/null | tail -1 | sed "s/^''\$//"; }
_gib () {  # sacct MaxRSS (123456K / 12M / 179G / 1.2T / bare bytes) -> GiB
    awk -v v="$1" 'BEGIN{ u = substr(v, length(v)); n = v + 0
        if (u == "K") g = n / 1048576; else if (u == "M") g = n / 1024; else if (u == "G") g = n;
        else if (u == "T") g = n * 1024; else g = n / 1073741824; printf "%.1f", g }'
}

# ── the probe report: bbl_sizing.sh owns it ──────────────────────────────────
if [[ "${DO_PROBE}" == "1" ]]; then
    _a=(); [[ "${TAG_SET}" == "1" ]] && _a=(--psi-tag "${TAG}")
    exec bash "${CL_DIR}/bbl_sizing.sh" --stage "${STAGE}" ${_a[@]+"${_a[@]}"}
fi

# ── the per-routine table ─────────────────────────────────────────────────────
if [[ "${TAG_SET}" == "0" ]]; then
    # The newest dispatched run that is not a memory probe (a probe dir holds probe.env).
    _newest="$(ls -td "${DROOT}"/E* 2>/dev/null | while read -r _d; do [[ -f "${_d}/probe.env" ]] || { printf '%s\n' "${_d}"; break; }; done)"
    if [[ -n "${_newest}" ]]; then TAG="$(_ctxval "${_newest}/context.env" PSI_TAG)"; fi
fi

NAMES=""; SINCE_EPOCH=""
for k in ${ROUTINES}; do
    rt="$(cl_bbl_rtag "${k}" "${STAGE}" "${TAG}")"
    NAMES="${NAMES:+${NAMES},}bbl_fwd_${rt},bbl_sweep_${rt},bbl_solve_${rt}"
    e="$(_ctxval "${DROOT}/$(cl_bbl_key "${k}" "${STAGE}" "${TAG}")/context.env" RUN_EPOCH)"
    if [[ -n "${e}" ]] && [[ -z "${SINCE_EPOCH}" || "${e}" -lt "${SINCE_EPOCH}" ]]; then SINCE_EPOCH="${e}"; fi
done
NAMES="${NAMES},bbl_tables${TAG},bbl_zip${TAG}"
if [[ -n "${SINCE_EPOCH}" ]]; then SINCE="$(date -d "@$(( SINCE_EPOCH - 3600 ))" +%Y-%m-%dT%H:%M:%S 2>/dev/null || date -d '3 days ago' +%Y-%m-%d)"
else SINCE="$(date -d '3 days ago' +%Y-%m-%d 2>/dev/null || echo now-3days)"; fi

SQ=""; SA=""
if [[ "${HAVE_SLURM}" == "1" ]]; then
    SQ="$(squeue -h -r -u "${ME}" -o '%i|%j|%T|%r' 2>/dev/null | awk -F'|' '$2 ~ /^bbl_/' || true)"
    SA="$(sacct -u "${ME}" -X -n -P -S "${SINCE}" --name="${NAMES}" -o JobID,JobName,State 2>/dev/null || true)"
fi
_q () {   # _q <jobname> -> "R P" counts of live tasks
    awk -F'|' -v n="$1" '$2 == n { if ($3 == "RUNNING" || $3 == "COMPLETING") r++; else if ($3 == "PENDING") p++ }
        END { printf "%d %d", r, p }' <<< "${SQ}"
}
_live () { awk -F'|' -v n="$1" '$2 == n { print $1 "|" $3 "|" $4; exit }' <<< "${SQ}"; }   # first live job: id|state|reason
_last () { awk -F'|' -v n="$1" '$2 == n { l = $1 "|" $3 } END { print l }' <<< "${SA}"; }    # newest sacct row: id|state

printf 'BBL status %s | tag %s | stage %s | %s%s\n' "$(date +%Y-%m-%d\ %H:%M)" "'${TAG}'" "${STAGE}" "${BD}" \
    "$([[ "${HAVE_SLURM}" == "1" ]] || echo '  [no SLURM here: disk only]')"
printf '%-7s %-9s %-9s %-24s %-24s %s\n' routine shards "fwd R/P/F" sweep solve next
for k in ${ROUTINES}; do
    key="$(cl_bbl_key "${k}" "${STAGE}" "${TAG}")"; rt="$(cl_bbl_rtag "${k}" "${STAGE}" "${TAG}")"
    dd="${DROOT}/${key}"
    N="$(_ctxval "${dd}/context.env" N_SHARDS)"
    if [[ -z "${N}" ]]; then
        N="$(ls "${BD}" 2>/dev/null | sed -n "s/^psi_dev_${key}_shard[0-9]*of\([0-9]*\)\.parquet$/\1/p" | sort | uniq -c | sort -rn | awk 'NR==1{print $2}')"
    fi
    have=0
    if [[ -n "${N}" ]]; then have="$(ls "${BD}" 2>/dev/null | grep -c "^psi_dev_${key}_shard[0-9]*of${N}\.parquet$" || true)"; fi
    read -r fR fP <<< "$(_q "bbl_fwd_${rt}")"
    fF="$(awk -F'|' -v n="bbl_fwd_${rt}" '$2 == n && $3 ~ /FAILED|TIMEOUT|OUT_OF_ME|NODE_FAIL|CANCELLED|PREEMPTED/ { c++ } END { print c + 0 }' <<< "${SA}")"
    # sweep
    sl="$(_live "bbl_sweep_${rt}")"
    ev="$(grep ' SWEEP ' "${dd}/events.log" 2>/dev/null | tail -1)"
    ev_try="$(sed -n 's/.* retry=\([0-9]*\).*/\1/p' <<< "${ev}")"; ev_res="$(sed -n 's/.* result=\([a-z]*\).*/\1/p' <<< "${ev}")"
    # Retries used: re-submissions since the last LAUNCH, of the run's BBL_MAX_RETRIES.
    used="$(awk '/ LAUNCH /{ n = 0 } / SWEEP .*result=resubmit/{ n++ } END { print n + 0 }' "${dd}/events.log" 2>/dev/null || echo 0)"
    maxr="$(_ctxval "${dd}/context.env" BBL_MAX_RETRIES)"
    if [[ -n "${sl}" ]]; then
        IFS='|' read -r sj ss sr <<< "${sl}"
        if [[ "${sr}" == *Held* ]]; then sweep="HELD ${sj}"; else sweep="${ss,,} ${sj}"; fi
        sweep="${sweep} r${used:-0}/${maxr:-3}"
    elif [[ -n "${ev}" ]]; then sweep="${ev_res} r${used:-0}/${maxr:-3}"
    else sweep="-"; fi
    # solve
    vl="$(_live "bbl_solve_${rt}")"; va="$(_last "bbl_solve_${rt}")"
    if [[ -n "${vl}" ]]; then IFS='|' read -r vj vs vr <<< "${vl}"; solve="${vs} (${vr})"
    elif [[ -n "${va}" ]]; then IFS='|' read -r vj vs <<< "${va}"; solve="${vs%% *} ${vj}"
    else vs=""; vj=""; solve="-"; fi
    cp="no"; [[ -f "${BD}/cost_params_${key}.json" ]] && cp="yes"
    # next action
    repair="bash bbl_run.sh --routines ${k} --shards ${N:-<N>} --psi-tag '${TAG}' --repair"
    if [[ "${ev_res}" =~ ^(exhausted|problem|stopped)$ && -z "${sl}" && "${fR}${fP}" == "00" ]]; then
        next="ACT: read logs/bbl_sweep_${rt}_*.out, fix, then: ${repair}"
    elif (( fR + fP > 0 )); then next="wait (fwd running; the sweep re-runs gaps itself)"
    elif [[ -n "${sl}" && "${sweep}" == HELD* ]]; then next="ACT: scontrol release ${sj}"
    elif [[ -n "${sl}" ]]; then next="wait (sweep)"
    elif [[ -n "${vl}" ]]; then next="wait (solve)"
    elif [[ "${vs}" == COMPLETED* && "${cp}" == "yes" ]]; then next="done: cost_params_${key}.json"
    elif [[ -n "${vs}" && "${vs}" != COMPLETED* ]]; then next="ACT: read logs/bbl_solve_${rt}_${vj}_*.{out,err}; then: ${repair}"
    elif [[ -z "${N}" ]]; then next="not launched"
    else next="ACT: nothing queued, ${have}/${N} shards: ${repair}"; fi
    printf '%-7s %-9s %-9s %-24s %-24s %s\n' "E${k}" "${have}/${N:-?}" "${fR}/${fP}/${fF}" "${sweep:0:24}" "${solve:0:24}" "${next}"
done
for n in "bbl_tables${TAG}" "bbl_zip${TAG}"; do
    l="$(_live "${n}")"; a="$(_last "${n}")"
    if [[ -n "${l}" ]]; then IFS='|' read -r j s r <<< "${l}"; printf '%s: %s %s (%s)   ' "${n}" "${s}" "${j}" "${r}"
    elif [[ -n "${a}" ]]; then IFS='|' read -r j s <<< "${a}"; printf '%s: %s %s   ' "${n}" "${s%% *}" "${j}"
    else printf '%s: -   ' "${n}"; fi
done
echo

if [[ "${DO_RSS}" == "1" && "${HAVE_SLURM}" == "1" ]]; then
    FWDN="$(for k in ${ROUTINES}; do printf 'bbl_fwd_%s,' "$(cl_bbl_rtag "${k}" "${STAGE}" "${TAG}")"; done)"
    MAP="$(sacct -u "${ME}" -S "${SINCE}" -X -n -P --name="${FWDN%,}" --format=JobID,JobName,Partition,State 2>/dev/null \
           | awk -F'|' '$3 ~ /gpu/ && $4 == "COMPLETED" { print $1 "|" $2 }')"
    ids="$(cut -d'|' -f1 <<< "${MAP}" | paste -sd, -)"
    if [[ -n "${ids}" ]]; then
        RSS="$(sacct -j "${ids}" -n -P --format=JobID,MaxRSS 2>/dev/null | awk -F'|' '$1 ~ /\.batch$/ { sub(/\.batch$/, "", $1); print $1 "|" $2 }')"
        echo "MaxRSS of COMPLETED GPU fwd jobs (.batch; a packed job's covers all its shards), GiB:"
        awk -F'|' 'NR == FNR { name[$1] = $2; next }
            { v = $2; u = substr(v, length(v)); x = v + 0
              g = (u == "K" ? x / 1048576 : u == "M" ? x / 1024 : u == "G" ? x : u == "T" ? x * 1024 : x / 1073741824)
              n = name[$1]; if (n == "") next; c[n]++; s[n, c[n]] = g }
            END { for (n in c) { m = c[n]
                    for (i = 1; i <= m; i++) for (j = i + 1; j <= m; j++) if (s[n, j] < s[n, i]) { t = s[n, i]; s[n, i] = s[n, j]; s[n, j] = t }
                    med = (m % 2 ? s[n, (m + 1) / 2] : (s[n, m / 2] + s[n, m / 2 + 1]) / 2)
                    printf "  %-22s n=%-4d median %6.1f  max %6.1f\n", n, m, med, s[n, m] } }' \
            <(printf '%s\n' "${MAP}") <(printf '%s\n' "${RSS}")
    else
        echo "MaxRSS: no COMPLETED GPU fwd job of these routines since ${SINCE}"
    fi
    # Per-shard peaks of packed jobs, from their own [pack] lines.
    for k in ${ROUTINES}; do
        rt="$(cl_bbl_rtag "${k}" "${STAGE}" "${TAG}")"
        cat "${LOGS}"/bbl_fwd_"${rt}"_pack_*.out 2>/dev/null | sed -n 's/^\[pack\] shard=.* rc=0 .*peak_rss_gib=\([0-9.]*\) .*/\1/p' \
            | sort -n | awk -v n="E${k}" '{ a[NR] = $1 } END { if (NR) printf "  %-22s per-shard peak (packed [pack] lines) n=%d median %.1f max %.1f\n", n, NR, (NR % 2 ? a[(NR + 1) / 2] : (a[NR / 2] + a[NR / 2 + 1]) / 2), a[NR] }'
    done
fi
exit 0
