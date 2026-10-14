"""Build the two cluster zips of the switch-cost investigation (scripts, data) with manifests.

scripts zip  every switch_cost_* file of the repo (root and utils/), plus README and
             MANIFEST.md5. Only NEW names: unpacking into the project's scripts checkout cannot
             overwrite a file of the shared project. REQUIRES.md5 lists the shared modules the
             copies import (found by parsing their imports, recursively) with their LOCAL md5,
             so the cluster side can check it runs the same versions before the first job.
data zip     the inputs listed in DATA_FILES (local path -> path inside the zip), plus
             MANIFEST.sha256. Nothing is uploaded or unpacked by this script.

usage: python -I claude/switch_cost/build_switch_cost_bundle.py <repo root> <out dir> [--data-list FILE] [--stamp S]
The data list is a two-column TSV (local path, path inside the zip); without it only the
scripts zip is built.
"""
from __future__ import annotations

import ast
import hashlib
import sys
import zipfile
from datetime import datetime
from pathlib import Path

REPO = Path(sys.argv[1]).resolve()
OUT = Path(sys.argv[2]).resolve()
DATA_LIST = Path(sys.argv[sys.argv.index("--data-list") + 1]) if "--data-list" in sys.argv else None
STAMP = sys.argv[sys.argv.index("--stamp") + 1] if "--stamp" in sys.argv else datetime.now().strftime("%Y%m%d_%H%M")
# utils/__init__.py defines load_panel_cached, which the copies import from the package
EXTRA_REQUIRES = ["config/routines.toml", "utils/__init__.py"]


def digest(p: Path, algo: str) -> str:
    h = hashlib.new(algo)
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def project_imports(py: Path) -> set[str]:
    """Repo modules imported by a file: utils.X, top-level repo modules."""
    try:
        tree = ast.parse(py.read_text(encoding="utf-8"))
    except SyntaxError:
        return set()
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module == "utils":
                mods |= {f"utils.{a.name}" for a in node.names}
            else:
                mods.add(node.module)
        elif isinstance(node, ast.Import):
            mods |= {a.name for a in node.names}
    out = set()
    for m in mods:
        rel = Path(*m.split("."))
        for cand in (REPO / rel.with_suffix(".py"), REPO / rel / "__init__.py"):
            if cand.exists():
                out.add(str(cand.relative_to(REPO)).replace("\\", "/"))
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted({p for pat in ("switch_cost_*", "utils/switch_cost_*") for p in REPO.glob(pat)
                    if p.is_file() and p.suffix in (".py", ".sh")})
    names = [str(p.relative_to(REPO)).replace("\\", "/") for p in files]
    # shared modules the copies need, recursively
    seen, todo = set(), [n for n in names if n.endswith(".py")]
    while todo:
        n = todo.pop()
        for dep in project_imports(REPO / n):
            if dep not in seen and dep not in names:
                seen.add(dep)
                todo.append(dep)
    # non-Python shared files the copies read: utils/routines.py loads the routine table
    requires = sorted(seen | {r for r in EXTRA_REQUIRES if (REPO / r).exists()})
    readme = REPO / "claude" / "switch_cost" / "README_CLUSTER.md"
    zpath = OUT / f"switch_cost_scripts_{STAMP}.zip"
    zpath.unlink(missing_ok=True)   # rebuilding a stamp replaces this script's own zip
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for p, n in zip(files, names):
            z.write(p, n)
        z.writestr("SWITCH_COST_MANIFEST.md5", "".join(f"{digest(p, 'md5')}  {n}\n" for p, n in zip(files, names)))
        z.writestr("SWITCH_COST_REQUIRES.md5", "".join(f"{digest(REPO / r, 'md5')}  {r}\n" for r in requires))
        if readme.exists():
            z.write(readme, "SWITCH_COST_README.md")
    print(f"scripts zip: {zpath} ({len(files)} files, {len(requires)} shared modules required)")
    for n in names:
        print("   ", n)
    print("  requires:", ", ".join(requires))

    if DATA_LIST is None:
        return
    rows = [ln.split("\t") for ln in DATA_LIST.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.startswith("#")]
    dpath = OUT / f"switch_cost_data_{STAMP}.zip"
    dpath.unlink(missing_ok=True)
    sha = []
    with zipfile.ZipFile(dpath, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as z:
        for local, inner in rows:
            lp = Path(local)
            if not lp.exists():
                raise SystemExit(f"data file missing: {lp}")
            z.write(lp, inner)
            sha.append(f"{digest(lp, 'sha256')}  {inner}\n")
        z.writestr("SWITCH_COST_DATA.sha256", "".join(sha))
    print(f"data zip: {dpath} ({len(rows)} files, {dpath.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
