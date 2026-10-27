"""
make_tr5_upload.py
==================
The cluster upload for TR5 (tr5_variants.sh): one zip that unzips from the project root into
    upload_tr5/scripts/   the TR5 code the sandboxes overlay on their copy of scripts/
                          (utils/deposit_types.py, the sleepiness sample builder, the three demand
                          preps, pipeline_all.sh with --sleep-args)
    scripts/tr5_variants.sh   the launcher (a new file in the production scripts/; the production
                          tree's other files are untouched)
and a sha256 list of every member for `sha256sum -c` from the project root. Scripts are written
with LF line endings. The zip carries the files as they are on disk when it is built, so build it
after every change it should carry (and after R2 has finished on the cluster).

Usage:
    python make_tr5_upload.py            list the members and their sizes; write nothing
    python make_tr5_upload.py --stage    write ESTIMATION_OUTPUT/CLUSTER_OUT/upload_tr5_<YYYYMMDD>.zip + .sha256
"""
from utils.venv_guard import ensure_project_venv
ensure_project_venv(__file__)

import datetime
import hashlib
import sys
import zipfile
from pathlib import Path

from utils import paths as _paths

HERE = Path(__file__).resolve().parent
OVERLAY = ["utils/deposit_types.py", "sleep_est_e2.py", "sleep_demand_prep_e1.py",
           "sleep_demand_prep_e2.py", "sleep_demand_prep_link.py", "pipeline_all.sh"]
LAUNCHER = "tr5_variants.sh"


def members():
    out = [(HERE / f, f"upload_tr5/scripts/{f}") for f in OVERLAY]
    out.append((HERE / LAUNCHER, f"scripts/{LAUNCHER}"))
    missing = [str(p) for p, _ in out if not p.is_file()]
    if missing:
        raise SystemExit(f"missing: {missing}")
    return out


def data_of(p: Path) -> bytes:
    b = p.read_bytes()
    return b.replace(b"\r\n", b"\n") if p.suffix in (".py", ".sh") else b


def main():
    ms = members()
    for p, arc in ms:
        print(f"  {len(data_of(p)):>9,}  {arc}")
    if "--stage" not in sys.argv:
        print("listing only; --stage writes the zip")
        return
    stamp = datetime.date.today().strftime("%Y%m%d")
    out = _paths.estimation_output() / "CLUSTER_OUT" / f"upload_tr5_{stamp}.zip"
    out.parent.mkdir(parents=True, exist_ok=True)
    sums = []
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p, arc in ms:
            b = data_of(p)
            info = zipfile.ZipInfo(arc, date_time=datetime.datetime.now().timetuple()[:6])
            info.external_attr = ((0o100755 if p.suffix == ".sh" else 0o100644) << 16)
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, b)
            sums.append(f"{hashlib.sha256(b).hexdigest()}  {arc}")
    out.with_suffix(".sha256").write_text("\n".join(sums) + "\n", encoding="utf-8", newline="\n")
    zsha = hashlib.sha256(out.read_bytes()).hexdigest()
    print(f"wrote {out} ({out.stat().st_size / 1e3:.1f} kB, {len(ms)} files), sha256 {zsha}, and its .sha256")


if __name__ == "__main__":
    main()
