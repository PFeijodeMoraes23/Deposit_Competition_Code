#!/bin/bash
# ==============================================================================
# bbl_sizing.sh — turn the memory probe (bbl_run.sh --probe) into MEM, PACK_H200, PACK_H100
# and the exact launch command.
#
# LOGIN-NODE SAFE. It reads the probe's dispatch files and SLURM logs with ls/grep/sed/awk (and
# sacct for the job states when SLURM is there). No Python, no Julia, no sbatch; it writes nothing.
#
#   bash bbl_sizing.sh                                   the newest probe
#   bash bbl_sizing.sh --psi-tag _probe250               a named probe
#   bash bbl_sizing.sh --launch-tag _ms2 --routines "3 4"   the tag/routines the printed command launches
#
# THE RULES (BBL_RUNBOOK.md section 1)
#   MEM     = ceil(max per-shard peak RSS over every probe shard x 1.15) GiB
#             (per-shard peaks come from each shard's own "[BBL] shard i peak RSS X GiB" line:
#             sacct's MaxRSS of a packed job sums its children)
#   cap_P   = min(GPUs per node, floor(0.85 x node MB / MEM), floor(CPUs per node / 8 threads))
#   PACK_P  = the k <= cap_P that runs the most shards at once under the per-user QOS,
#             k x min(MaxJobsPU, floor(MaxGPU / k)); ties go to the smaller k
#   packing GO when the slowest packed shard <= 1.3 x the unpacked shard (same T, same partition);
#             NO-GO -> PACK 1 on both partitions (one shard per job, the configuration validated
#             against CPU to 3.4e-14)
#   gpu_h100 is used only if cap_h100 >= 1 and the probe's peak GPU memory <= 85% of an H100
#   --shard-time is added only when the measured shard x 1.5 (rounded up to 30 min) exceeds the
#             wall bbl_run.sh would derive at that T
# Constants: BBL_MEM_HEADROOM (1.15), BBL_NODE_FRACTION (0.85), BBL_PACK_MAX_RATIO (1.3),
#   BBL_THREADS_PER_SHARD (8), BBL_WALL_SAFETY (1.5); node and QOS shapes from cl_bbl_part.
# ==============================================================================
set -uo pipefail
CL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${CL_DIR}/cluster_lib.sh"

TAG=""; TAG_SET=0; STAGE="${CF_STAGE:-extended}"; LAUNCH_TAG="${LAUNCH_TAG:-_ms2}"; LROUTINES="3 4"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --psi-tag)    TAG="$2"; TAG_SET=1; shift ;;
        --stage)      STAGE="$2"; shift ;;
        --launch-tag) LAUNCH_TAG="$2"; shift ;;
        --routines)   LROUTINES="$2"; shift ;;
        -h|--help)    sed -n '2,27p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown option: $1 (see -h)" >&2; exit 2 ;;
    esac
    shift
done

DROOT="${CL_BBL_DISPATCH_ROOT:-${CL_STEP_BBL}/.dispatch}"
LOGS="${BBL_LOG_DIR:-${CL_ROOT}/logs}"
_val () { sed -n "s/^$2=//p" "$1" 2>/dev/null | tail -1 | sed "s/^'\(.*\)'\$/\1/"; }

# ── which probe ──────────────────────────────────────────────────────────────
if [[ "${TAG_SET}" == "1" ]]; then
    PD="$(ls -d "${DROOT}"/E*_spec_12_"${STAGE}${TAG}" 2>/dev/null | head -1)"
else
    PD="$(ls -td "${DROOT}"/*/ 2>/dev/null | while read -r d; do [[ -f "${d}probe.env" ]] && { printf '%s\n' "${d%/}"; break; }; done)"
fi
if [[ -z "${PD}" || ! -f "${PD}/probe.env" ]]; then
    echo "no probe found under ${DROOT}$([[ "${TAG_SET}" == "1" ]] && printf ' for tag %s' "${TAG}")."
    echo "Submit one first (BBL_RUNBOOK.md section 1): bbl_run.sh --probe ..."
    exit 1
fi
PE="${PD}/probe.env"
PJ="$(_val "${PE}" PACKED_JID)"; CJ="$(_val "${PE}" CONTROL_JID)"; CS="$(_val "${PE}" CONTROL_SHARD)"
PK="$(_val "${PE}" PACKED_K)"; PSH="$(_val "${PE}" PACKED_SHARDS)"; PMEM="$(_val "${PE}" PACKED_MEM)"
CMEM="$(_val "${PE}" CONTROL_MEM)"; PPART="$(_val "${PE}" PARTITION)"
T="$(_val "${PE}" T)"; B="$(_val "${PE}" BETA)"; NS="$(_val "${PE}" N_SHARDS)"; RTAG="$(_val "${PE}" RTAG)"

echo "PROBE ${PD##*/}   T=${T} beta=${B} N_SHARDS=${NS} on ${PPART}"
echo "  (a) packed   job ${PJ}: shards ${PSH}, ${PK} x ${PMEM}"
echo "  (b) unpacked job ${CJ:-none}: shard ${CS:-none}, 1 x ${CMEM}"
if command -v sacct >/dev/null 2>&1; then
    sacct -j "${PJ}${CJ:+,${CJ}}" -X -n -P -o JobID,State,Elapsed,MaxRSS 2>/dev/null \
        | awk -F'|' '{ printf "      sacct %-16s %-12s %s\n", $1, $2, $3 }'
fi
if command -v squeue >/dev/null 2>&1; then
    _live="$(squeue -h -j "${PJ}${CJ:+,${CJ}}" -o '%i %T' 2>/dev/null | paste -sd' ' -)"
    if [[ -n "${_live}" ]]; then
        echo "  still in the queue (${_live}): size from the probe once both jobs have left squeue."
        exit 0
    fi
fi

# ── one line per probe shard: kind shard rc elapsed_s gpu rss ────────────────
# Packed: the job's own [pack] summary lines; a shard missing there (the job was killed before it
# summarised) is read from its child log instead. Unpacked: the job log itself.
LINES=""; GMAX=0
_from_child () {   # _from_child <log> <shard>: kind line from a child's own log
    local f="$1" s="$2" st en rc rss gpu
    st="$(sed -n 's/^\[child\] shard=[0-9]* device=.* start_epoch=\([0-9]*\).*/\1/p' "${f}" | head -1)"
    en="$(sed -n 's/^\[child\] shard=[0-9]* end_epoch=\([0-9]*\).*/\1/p' "${f}" | tail -1)"
    rc="$(sed -n 's/^\[child\] shard=[0-9]* end_epoch=[0-9]* rc=\([0-9]*\).*/\1/p' "${f}" | tail -1)"
    rss="$(sed -n 's/.*\[BBL\] shard [0-9]* peak RSS \([0-9.]*\) GiB.*/\1/p' "${f}" | tail -1)"
    gpu=no; grep -q "GPU share kernel enabled" "${f}" && gpu=yes
    printf 'packed %s %s %s %s %s\n' "${s}" "${rc:-?}" "$([[ -n "${st}" && -n "${en}" ]] && echo $(( en - st )) || echo '?')" "${gpu}" "${rss:-?}"
}
PLOG="$(ls "${LOGS}"/bbl_fwd_"${RTAG}"_pack_"${PJ}"_*.out 2>/dev/null | head -1)"
for s in $(cl_expand_spec "${PSH}"); do
    l=""
    if [[ -n "${PLOG}" ]]; then
        l="$(grep -h "^\[pack\] shard=${s} " "${PLOG}" | tail -1 | awk '{
            for (i = 2; i <= NF; i++) { split($i, a, "="); v[a[1]] = a[2] }
            printf "packed %s %s %s %s %s\n", v["shard"], v["rc"], v["elapsed_s"], v["gpu"], v["peak_rss_gib"] }')"
    fi
    if [[ -z "${l}" ]]; then
        cf="${LOGS}/bbl_fwd_${RTAG}_shard${s}_${PJ}.out"
        if [[ -f "${cf}" ]]; then l="$(_from_child "${cf}" "${s}")"; else l="packed ${s} ? ? ? ?"; fi
    fi
    LINES="${LINES}${l}"$'\n'
done
if [[ -n "${CJ}" ]]; then
    CLOG="${LOGS}/bbl_fwd_${RTAG}_${CJ}_${CS}.out"
    if [[ -f "${CLOG}" ]]; then
        e="$(sed -n 's/^\[pack\] shard=[0-9]* rc=[0-9]* elapsed_s=\([0-9]*\).*/\1/p' "${CLOG}" | tail -1)"
        rc="$(sed -n 's/^\[pack\] shard=[0-9]* rc=\([0-9]*\).*/\1/p' "${CLOG}" | tail -1)"
        rss="$(sed -n 's/.*\[BBL\] shard [0-9]* peak RSS \([0-9.]*\) GiB.*/\1/p' "${CLOG}" | tail -1)"
        g=no; grep -q "GPU share kernel enabled" "${CLOG}" && g=yes
        LINES="${LINES}control ${CS} ${rc:-?} ${e:-?} ${g} ${rss:-?}"$'\n'
    else
        LINES="${LINES}control ${CS} ? ? ? ?"$'\n'
    fi
fi
for f in "${PLOG:-}" "${CLOG:-}"; do
    [[ -n "${f}" && -f "${f}" ]] || continue
    m="$(sed -n 's/^\[gpumem\] device=[^ ]* peak_mib=\([0-9]*\).*/\1/p' "${f}" | sort -n | tail -1)"
    if [[ -n "${m}" ]] && (( m > GMAX )); then GMAX="${m}"; fi
done

printf '%s' "${LINES}" | awk '{ printf "      %-8s shard %-4s rc=%-2s elapsed %6s s  GPU kernel %-3s  peak RSS %s GiB\n", $1, $2, $3, $4, $5, $6 }'

# ── the decision ─────────────────────────────────────────────────────────────
H2="gpu_h200"; H1="gpu_h100"
REC="$(printf '%s' "${LINES}" | awk \
    -v head="${BBL_MEM_HEADROOM:-1.15}" -v frac="${BBL_NODE_FRACTION:-0.85}" -v rmax="${BBL_PACK_MAX_RATIO:-1.3}" \
    -v thr="${BBL_THREADS_PER_SHARD:-8}" -v gmax="${GMAX}" \
    -v n2mb="$(cl_bbl_part "${H2}" node_mem_mb)" -v n2g="$(cl_bbl_part "${H2}" node_gpus)" -v n2c="$(cl_bbl_part "${H2}" node_cpus)" \
    -v q2j="$(cl_bbl_part "${H2}" max_jobs)" -v q2g="$(cl_bbl_part "${H2}" max_gpus)" \
    -v n1mb="$(cl_bbl_part "${H1}" node_mem_mb)" -v n1g="$(cl_bbl_part "${H1}" node_gpus)" -v n1c="$(cl_bbl_part "${H1}" node_cpus)" \
    -v q1j="$(cl_bbl_part "${H1}" max_jobs)" -v q1g="$(cl_bbl_part "${H1}" max_gpus)" -v g1mib="$(cl_bbl_part "${H1}" gpu_mem_mib)" '
    function cap(nmb, ng, nc, mb,   a, b, c) { a = ng; b = int(frac * nmb / mb); c = int(nc / thr)
        return (a < b ? (a < c ? a : c) : (b < c ? b : c)) }
    function conc(k, qj, qg,   j) { j = int(qg / k); if (j > qj) j = qj; return j * k }
    function best(c, qj, qg,   k, bk, bv, v) { bk = 0; bv = -1
        for (k = 1; k <= c; k++) { v = conc(k, qj, qg); if (v > bv) { bv = v; bk = k } } return bk }
    { n++; kind = $1; rc = $3; e = $4; g = $5; r = $6
      if (rc != "0" || r !~ /^[0-9.]+$/ || e !~ /^[0-9]+$/) { bad++; badl = badl " " kind ":" $2 }
      if (r ~ /^[0-9.]+$/ && r + 0 > peak) peak = r + 0
      if (kind == "packed" && e ~ /^[0-9]+$/ && e + 0 > pmax) pmax = e + 0
      if (kind == "control" && e ~ /^[0-9]+$/) cmax = e + 0
      if (kind == "packed" && g != "yes") nogpu++
      if (kind == "control") hasc = 1 }
    END {
      if (!n) { print "STATUS=EMPTY"; exit }
      if (bad) { print "STATUS=INCOMPLETE"; print "BAD=" badl; print "PEAK=" peak; exit }
      mem = peak * head; memg = int(mem); if (memg < mem) memg++
      mb = memg * 1024
      c2 = cap(n2mb, n2g, n2c, mb); c1 = cap(n1mb, n1g, n1c, mb)
      ratio = (hasc && cmax > 0) ? pmax / cmax : -1
      go = (ratio >= 0 && ratio <= rmax && !nogpu)
      k2 = (go ? best(c2, q2j, q2g) : (c2 >= 1 ? 1 : 0)); k1 = (go ? best(c1, q1j, q1g) : (c1 >= 1 ? 1 : 0))
      h1gpu = (gmax > 0 ? (gmax <= frac * g1mib) : -1)
      use1 = (k1 >= 1 && h1gpu != 0)
      print "STATUS=OK"; print "PEAK=" peak; print "MEMG=" memg; print "MEM_MB=" mb
      print "CAP2=" c2; print "CAP1=" c1; print "K2=" k2; print "K1=" k1
      print "C2=" (k2 ? conc(k2, q2j, q2g) : 0); print "C1=" (use1 ? conc(k1, q1j, q1g) : 0)
      printf "RATIO=%.3f\n", ratio; print "GO=" go; print "NOGPU=" nogpu + 0; print "USE1=" use1; print "H1GPU=" h1gpu
      print "PMAX=" pmax; print "CMAX=" cmax + 0
      printf "CAP2WHY=min(GPUs %d, floor(%.2f x %d / %d) = %d, floor(CPUs %d / %d) = %d) = %d\n", n2g, frac, n2mb, mb, int(frac * n2mb / mb), n2c, thr, int(n2c / thr), c2
      printf "CAP1WHY=min(GPUs %d, floor(%.2f x %d / %d) = %d, floor(CPUs %d / %d) = %d) = %d\n", n1g, frac, n1mb, mb, int(frac * n1mb / mb), n1c, thr, int(n1c / thr), c1
    }')"
_r () { sed -n "s/^$1=//p" <<< "${REC}" | tail -1; }

echo
echo "SIZING (T=${T})"
case "$(_r STATUS)" in
    EMPTY)
        echo "  no probe shard has reported yet: the jobs are still queued or running (squeue -u \$USER)."
        exit 0 ;;
    INCOMPLETE)
        echo "  PROBE INCOMPLETE:$(_r BAD) has no rc=0 / elapsed / peak-RSS line. No sizing from a partial probe."
        echo "  Read the logs:  ls ${LOGS}/bbl_fwd_${RTAG}_*${PJ}* ${LOGS}/bbl_fwd_${RTAG}_${CJ:-X}_*"
        echo "  OOM (sacct State OUT_OF_MEMORY): re-probe with more memory, e.g. PROBE_MEM_UNPACKED=900G PROBE_MEM_PACKED=480G."
        echo "  TIMEOUT: re-probe with a longer wall, e.g. BBL_PROBE_WALL_MULT=4 (BBL_RUNBOOK.md section 1)."
        exit 1 ;;
esac
PEAK="$(_r PEAK)"; MEMG="$(_r MEMG)"; K2="$(_r K2)"; K1="$(_r K1)"; GO="$(_r GO)"; USE1="$(_r USE1)"
echo "  peak RSS, max over every probe shard     ${PEAK} GiB"
echo "  MEM = ceil(${PEAK} x ${BBL_MEM_HEADROOM:-1.15})              = ${MEMG}G per shard"
echo "  cap gpu_h200 = $(_r CAP2WHY)"
echo "  cap gpu_h100 = $(_r CAP1WHY)"
_rat="$(_r RATIO)"
if [[ "${_rat}" == -* ]]; then
    echo "  [NO-GO] packing: no unpacked control finished, so the contention ratio cannot be judged -> PACK 1"
elif [[ "${GO}" == "1" ]]; then
    echo "  [GO   ] packing: slowest packed shard / unpacked = $(_r PMAX) / $(_r CMAX) s = ${_rat} <= ${BBL_PACK_MAX_RATIO:-1.3}"
else
    echo "  [NO-GO] packing: slowest packed shard / unpacked = $(_r PMAX) / $(_r CMAX) s = ${_rat}$([[ "$(_r NOGPU)" != "0" ]] && printf ' (and %s packed shard(s) ran WITHOUT the GPU kernel)' "$(_r NOGPU)") -> PACK 1"
fi
if [[ "${K2}" == "0" ]]; then
    echo "  [NO-GO] gpu_h200: ${MEMG}G per shard does not fit ${BBL_NODE_FRACTION:-0.85} of a node. Nothing to launch at T=${T}."
    exit 1
fi
echo "  PACK_H200 = ${K2}  -> $(_r C2) shards at once (QOS: 6 jobs / 16 GPUs; the throughput-best k <= cap)"
case "$(_r H1GPU)" in
    0)  echo "  [NO-GO] gpu_h100: peak GPU memory $(( GMAX / 1024 )) GiB > ${BBL_NODE_FRACTION:-0.85} of an H100 -> gpu_h200 only" ;;
    -1) echo "  [ ?   ] gpu_h100: no nvidia-smi samples in the probe logs; GPU memory not checked" ;;
    *)  echo "  [GO   ] gpu_h100 GPU memory: peak $(( GMAX / 1024 )) GiB <= ${BBL_NODE_FRACTION:-0.85} of an H100" ;;
esac
if [[ "${USE1}" == "1" ]]; then
    echo "  PACK_H100 = ${K1}  -> $(_r C1) shards at once (QOS: 12 jobs / 32 GPUs)"
elif [[ "${K1}" == "0" ]]; then
    echo "  gpu_h100: ${MEMG}G per shard does not fit ${BBL_NODE_FRACTION:-0.85} of a 1,000,000 MB node -> gpu_h200 only"
fi

# The wall: the derived one at this T, unless the probe measured a slower shard.
HORIZON="${T}"; SHOCKS="${SHOCKS:-50}"; N_SHARDS="${NS}"
_meas="$(_r PMAX)"; [[ "${GO}" == "1" ]] || _meas="$(_r CMAX)"
_wd="$(cl_bbl_wall_minutes "${H2}" "${K2}" 1)"; _wd="${_wd%%|*}"
_wm="$(awk -v e="${_meas}" -v s="${BBL_WALL_SAFETY:-1.5}" -v rd="${BBL_WALL_ROUND_MIN:-30}" 'BEGIN{ v = e / 60 * s; i = int(v); if (i < v) i++; i = int((i + rd - 1) / rd) * rd; print i }')"
ST=""
if (( _wm > _wd )); then
    ST=" --shard-time $(cl_min_to_wall "${_wm}")"
    echo "  wall: measured $(( _meas / 60 )) min x ${BBL_WALL_SAFETY:-1.5} = ${_wm} min > derived ${_wd} min -> pass${ST}"
else
    echo "  wall: measured $(( _meas / 60 )) min x ${BBL_WALL_SAFETY:-1.5} = ${_wm} min <= derived $(cl_min_to_wall "${_wd}") -> keep the derived wall"
fi

if [[ "${GO}" == "1" ]]; then PFLAGS="--pack-h200 ${K2}"; else PFLAGS="--pack 1"; fi
if [[ "${USE1}" == "1" ]]; then
    PARTS="--multi-partition"; MFLAGS="--mem-h200 ${MEMG}G --mem-h100 ${MEMG}G"
    [[ "${GO}" == "1" ]] && PFLAGS="${PFLAGS} --pack-h100 ${K1}"
else
    PARTS="--partitions gpu_h200"; MFLAGS="--mem-h200 ${MEMG}G"
fi
CMD="bash bbl_run.sh --routines '${LROUTINES}' --fwd-gpu ${PARTS} --shards ${NS} --multi-start --n-paths 1 --psi-tag ${LAUNCH_TAG} --no-warmup --no-polfunc ${MFLAGS} ${PFLAGS}${ST}"
echo
echo "LAUNCH (beta and T come from bbl_discount.env; the banner prints both and their source):"
echo "  sbatch -p day -t 00:30:00 -c 1 --mem=4G -J bbl_submit -o logs/bbl_submit_%j.out \\"
echo "    --wrap \"${CMD}\""
echo "BBL_LAUNCH_CMD=${CMD}"
exit 0
