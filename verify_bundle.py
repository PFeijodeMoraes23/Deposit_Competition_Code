"""verify_bundle.py -- read-only checks of a staged stage C upload, as the cluster will see it.

    python -I verify_bundle.py BUNDLE_DIR

BUNDLE_DIR holds stagec_code_<stamp>.zip, stagec_data_<stamp>.zip and stagec_upload_<stamp>.md5.
Unpacks both zips into BUNDLE_DIR/verify/ (the project root as on the cluster) and checks:
  V1 the upload md5 list (md5sum -c of the two zips)
  V2 the arrival list: sha256 of every member it names (what the wrapper checks at launch)
  V3 every code member byte-equals the LF-normalised repo file (or the wrapper of this folder);
     every data member equals its local source; no sleepiness estimate ships (made on the cluster)
  V4 no CR in .sh/.jl/.py/.env members; bash -n on every .sh member
  V5 required members present (the chain's job scripts, engines, preps, config, Project/Manifest)
  V6 no forbidden (demand_repair*, blp_dx*, dx_*) or excluded (weak-IV battery) member
  V7 the pin file: CHARS_SHA256, SIGMA_SHA256 (the 10-15 sigma rebuild, never the 08-26 file),
     JL_MANIFEST_SHA256 = the shipped Manifest-v1.11.toml (= the validated one); no sleepiness
     estimate and no EXPECT value ships, the conda create line conda-forge only; the shipped panel md5 6613228b...; the chars
     parquet schema metadata records that panel md5
Exit 0 only when every check passes.
"""
from __future__ import annotations

import hashlib
import re
import subprocess
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_stagec_bundle as B  # noqa: E402  (the same lists and sources the build used)

REQUIRED = ["stagec_suite_20261012.sh", "stagec_pins_20261012.env", "stagec_requests.env", "blp_smoke.jl", "cluster_lib.sh", "env_job.sh",
            "cluster_preflight.sh", "sleep_run.sh", "sleep_job.sh", "logit_job.sh", "blp_draws_job.sh",
            "blp_run.sh", "blp_stage_job.sh", "bbl_run.sh", "bbl_job.sh", "cluster_archive.sh",
            "bbl_cancel.sh", "bbl_status.sh", "bbl_discount.env", "Project.toml", "Manifest.toml", "Manifest-v1.11.toml",
            "blp_rc.jl", "blp_engine_gpu.jl", "blp_engine_cpu.jl", "blp_se_common.jl", "demand_inputs.jl",
            "blp_logit.jl", "blp_draws.jl", "blp_build_sysimage.jl", "blp_build_sysimage_workload.jl",
            "of_root.jl", "routines.jl", "bbl_fwd_sim.jl", "cf_demand_eval.jl", "cf_deposit_sim.jl",
            "cf_phi_path.jl", "cf_psi_basis.jl", "bbl_solve.py", "bbl_shards.py", "sleep_pipeline.py",
            "sleep_demand_prep.py", "sleep_demand_prep_link.py", "sleep_demand_prep_e1.py",
            "sleep_demand_prep_e2.py", "sleep_est_e2.py", "utils/__init__.py", "utils/demand_chars.py",
            "utils/paths.py", "utils/routines.py", "utils/state_transform.py", "utils/window.py",
            "utils/sleep_links.py", "utils/venv_guard.py", "config/routines.toml",
            "cluster/upload_manifest.txt", "make_bbl_cost_tables.py"]


def sha(b: bytes, algo="sha256") -> str:
    return hashlib.new(algo, b).hexdigest()


def main() -> int:
    bd = Path(sys.argv[1]); ok = True
    def chk(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'ok  ' if cond else 'FAIL'}] {name}{(' -- ' + str(detail)) if detail else ''}")
    md5f = sorted(bd.glob("stagec_upload_*.md5"))
    chk("V1 one upload md5 list", len(md5f) == 1, [p.name for p in md5f])
    # the bundle's own stamp (a rebuild with --stamp keeps an earlier one): the chars member name carries it
    stamp = re.match(r"stagec_upload_(\d+)\.md5$", md5f[0].name).group(1)
    if stamp != B.STAMP:
        old = B.CHARS_NAME
        B.STAMP, B.CHARS_NAME = stamp, f"demand_chars_{stamp}.parquet"
        B.DATA[B.CHARS_NAME] = B.DATA.pop(old)
    for line in md5f[0].read_text().splitlines():
        want, name = line.split(None, 1)
        chk(f"V1 md5 {name}", sha((bd / name).read_bytes(), "md5") == want)
    root = bd / "verify"; root.mkdir(exist_ok=True)
    members = {}
    for z in sorted(bd.glob("stagec_*.zip")):
        with zipfile.ZipFile(z) as zf:
            for info in zf.infolist():
                members[info.filename] = zf.read(info)
    arr = [m for m in members if re.match(r"scripts/stagec_arrival_\d+\.sha256$", m)]
    chk("V2 one arrival list in the code zip", len(arr) == 1, arr)
    lines = members[arr[0]].decode().splitlines()
    bad = [l for l in lines if (lambda w, p: p not in members or sha(members[p]) != w)(*l.split("  ", 1))]
    chk(f"V2 arrival: {len(lines)} members hash as listed", not bad, bad[:3])
    listed = {l.split("  ", 1)[1] for l in lines}
    unlisted = [m for m in members if m not in listed and m != arr[0]]
    chk("V2 every member except the list itself is listed", not unlisted, unlisted[:5])
    # V3
    diff = []
    for m, b in members.items():
        if m.startswith("scripts/"):
            rel = m[len("scripts/"):]
            if rel == B.WRAPPER.name:
                src = B.lf(B.WRAPPER)
            elif rel == B.JL_MANIFEST:
                src = (B.REPO / B.JL_MANIFEST).read_bytes()
            elif rel in (B.PINS_NAME,) or rel.startswith("stagec_arrival_"):
                continue
            else:
                src = B.lf(B.REPO / rel)
            if src != b:
                diff.append(m)
        elif m.startswith("data/input/"):
            name = m[len("data/input/"):]
            src_p = B.DATA.get(name)
            if src_p is None or src_p.read_bytes() != b:
                diff.append(m)
    chk("V3 every member equals its local source (code LF-normalised)", not diff, diff[:5])
    # V4
    crs = [m for m, b in members.items() if m.endswith((".sh", ".jl", ".py", ".env")) and b"\r" in b]
    chk("V4 no CR in .sh/.jl/.py/.env members", not crs, crs[:5])
    for m, b in members.items():
        if m.endswith(".sh"):
            p = root / m; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(b)
    shs = sorted(root.glob("scripts/*.sh"))
    badsh = [p.name for p in shs if subprocess.run(["bash", "-n", str(p)], capture_output=True).returncode != 0]
    chk(f"V4 bash -n on {len(shs)} shell members", not badsh, badsh)
    # V5, V6
    missing = [r for r in REQUIRED if f"scripts/{r}" not in members]
    chk(f"V5 {len(REQUIRED)} required members present", not missing, missing)
    forb = [m for m in members if B.FORBIDDEN.search(m) or Path(m).name in B.EXCLUDE]
    chk("V6 no forbidden or excluded member", not forb, forb)
    # V7
    pins = members.get("scripts/stagec_pins_20261012.env", b"").decode()
    chk("V7 pin CHARS_SHA256 = the pinned chars file", f"CHARS_SHA256={B.CHARS_SHA256}" in pins)
    chk("V7 no sleepiness estimate and no local reference value ships (Pedro 10-15)",
        not any("sleep_est_" in m for m in members) and "EST_UPLOADS" not in pins and "EXPECT_" not in pins)
    jm = members.get(f"scripts/{B.JL_MANIFEST}", b"")
    chk("V7 JL_MANIFEST_SHA256 = the shipped Manifest-v1.11.toml = the validated set",
        f"JL_MANIFEST_SHA256={sha(jm)}" in pins and sha(jm) == B.JL_MANIFEST_SHA256 and b'julia_version = "1.11.4"' in jm)
    sig = sha(members.get(f"data/input/{B.SIGMA_NAME}", b""))
    chk("V7 demographics_sigma.parquet = the pinned 10-15 rebuild, pinned in the pin file, not the stale 08-26 file",
        sig == B.SIGMA_SHA256 and f"SIGMA_SHA256={sig}" in pins and sig not in B.SIGMA_STALE_SHA256)
    chk("V7 conda create line: conda-forge only", "--override-channels --strict-channel-priority -c conda-forge" in pins)
    chk("V7 shipped panel md5 = 6613228b...", sha(members.get("data/input/market_panel.parquet", b""), "md5") == B.PANEL_MD5)
    import pyarrow.parquet as pq, io
    meta = pq.ParquetFile(io.BytesIO(members[f"data/input/{B.CHARS_NAME}"])).schema_arrow.metadata or {}
    chk("V7 chars parquet records the shipped panel md5", meta.get(b"demand_chars.market_panel_md5", b"").decode() == B.PANEL_MD5)
    print("VERIFY: ALL PASS" if ok else "VERIFY: FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
