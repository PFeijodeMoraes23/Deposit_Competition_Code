#!/usr/bin/env python
"""
Assert that every path seam resolves where the cluster tree says it should — and that the
local tree is untouched by the cluster's environment.

The pipeline decides paths in two places: ``utils/paths.py`` (Python) and ``of_root.jl``
(Julia). Both branch on the cluster's per-step layout::

    data/input                       uploads
    data/output/sleep                est{k}/, Rout/
    data/output/demand_prep          demand_{k}[_tag]_spec_{s}.parquet
    data/output/logit                logit deltas, .jls fits, .tex
    data/output/blp                  RC results/checkpoints   (+ blp/draws)
    data/output/bbl                  polfunc, psi shards, cost params
    data/output/counterfactuals      upsilon/phi_nopix, cf1, cf4
    data/output/download             archives

A writer and a reader that disagree about one of these directories do not raise: the writer
writes, the reader searches somewhere else and reports "not found", or — worse — finds an
older vintage and returns a wrong number. That failure is silent and survives review, which
is why the seams get an audit rather than a comment.

Two halves, and the second one is the one that catches leaks:

  1. CLUSTER — build a throwaway ``HEAD/{scripts,data/{input,output/*}}`` tree, export exactly
     what ``cl_export_step_dirs`` (cluster_lib.sh) exports, and require every accessor to land
     INSIDE its step folder.
  2. LOCAL — clear those variables and require that no accessor resolves anywhere inside the
     throwaway tree. An override that survives its unset is a path leak: it would silently
     redirect a local run at whatever the last cluster job happened to set.

Each Python phase runs in a SUBPROCESS. ``utils/paths.py`` reads ``OPEN_FINANCE_ROOT`` once at
import time, so mutating ``os.environ`` in-process and re-importing would test the module cache
rather than the module.

Usage:
    python audit_paths.py              # both halves
    python audit_paths.py --no-julia   # skip the Julia half
    python audit_paths.py -v           # print every resolved path, not just failures

Exit status is 0 only when every assertion passes.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent

# The step folders cl_export_step_dirs creates (cluster_lib.sh CL_STEP_*).
STEP_DIRS = ["sleep", "demand_prep", "logit", "blp", "blp/draws", "bbl",
             "counterfactuals", "download"]

# Accessor -> the step folder its result must sit inside, under the cluster environment.
# "" means the accessor is probed and reported but not constrained to one step folder.
EXPECT_IN = {
    "demand_prep_root":   "sleep",
    "demand_parquet_dir": "demand_prep",
    "est_dir_3":          "sleep",
    "rout_dir":           "sleep",
    "cf_foundation_dir":  "counterfactuals",
    "cost_fwd_dir":       "bbl",
    "polfunc_dir":        "bbl",
    "estimation_output":  "",
    "blp_results_dir":    "",
    "bbl_output_dir":     "",
    "data_root":          "",
}

# Run inside the subprocess: import utils.paths under whatever environment we hand it and
# report every accessor as an absolute path.
PROBE = r"""
import json, sys
sys.path.insert(0, sys.argv[1])
from utils import paths

def g(fn, *a):
    try:
        return str(fn(*a))
    except Exception as exc:          # an accessor that raises is a finding, not a crash
        return "ERROR: %s: %s" % (type(exc).__name__, exc)

print(json.dumps({
    "demand_prep_root":   g(paths.demand_prep_root),
    "demand_parquet_dir": g(paths.demand_parquet_dir),
    "est_dir_3":          g(paths.est_dir, 3),
    "rout_dir":           g(paths.rout_dir),
    "estimation_output":  g(paths.estimation_output),
    "blp_results_dir":    g(paths.blp_results_dir),
    "bbl_output_dir":     g(paths.bbl_output_dir),
    "cf_foundation_dir":  g(paths.cf_foundation_dir),
    "cost_fwd_dir":       g(paths.cost_fwd_dir),
    "polfunc_dir":        g(paths.polfunc_dir),
    "data_root":          g(paths.data_root),
}))
"""

# Julia half. of_root.jl has no package dependencies, so it includes standalone.
JULIA_PROBE = r"""
include(joinpath(ARGS[1], "of_root.jl"))
cl_out = ARGS[2]                      # <HEAD>/data/output   -> cluster-shaped
loc_out = ARGS[3]                     # <...>/BLP_RESULTS    -> local-shaped
loc_in  = ARGS[4]                     # <...>/DEMAND_PREP
j(p...) = joinpath(p...)
res = Dict{String,Any}(
  "is_cluster_out_cluster" => is_cluster_out(cl_out),
  "is_cluster_out_local"   => is_cluster_out(loc_out),
  "blp_dir_cluster"        => blp_dir(cl_out),
  "blp_dir_local"          => blp_dir(loc_out),
  "logit_dir_cluster"      => logit_dir(cl_out),
  "logit_dir_local"        => logit_dir(loc_out),
  "draws_dir_cluster"      => draws_dir(cl_out),
  "draws_dir_local"        => draws_dir(loc_out),
  "cf_out_foundation"      => cf_out_dir(cl_out, "CF_FOUNDATION"),
  "cf_out_costfwd"         => cf_out_dir(cl_out, "COST_FWD"),
  "cf_out_polfunc"         => cf_out_dir(cl_out, "COST_POLFUNC"),
  "cf_in_cluster"          => cf_in_dir(cl_out),
  "demand_search_cluster"  => demand_search_dirs(loc_in, cl_out),
  "demand_search_local"    => demand_search_dirs(loc_in, loc_out),
)
# Flat one-key-per-line output: no JSON dependency in a bare Julia process.
for k in sort(collect(keys(res)))
    v = res[k]
    println(k, "\t", v isa Vector ? join(v, "|") : string(v))
end
"""


class Audit:
    def __init__(self, verbose: bool) -> None:
        self.verbose = verbose
        self.passed = 0
        self.failed: list[str] = []

    def check(self, ok: bool, label: str, detail: str = "") -> None:
        if ok:
            self.passed += 1
            if self.verbose:
                print(f"    ok    {label}" + (f"  [{detail}]" if detail else ""))
        else:
            self.failed.append(f"{label}: {detail}")
            print(f"    FAIL  {label}  {detail}")


def build_tree(root: Path) -> tuple[Path, Path]:
    """The cluster layout, as cl_bootstrap_tree + cl_export_step_dirs leave it."""
    head = root / "HEAD"
    (head / "scripts").mkdir(parents=True, exist_ok=True)
    (head / "data" / "input").mkdir(parents=True, exist_ok=True)
    out = head / "data" / "output"
    for d in STEP_DIRS:
        (out / d).mkdir(parents=True, exist_ok=True)
    return head, out


def cluster_env(head: Path, out: Path) -> dict:
    """Exactly the exports in cl_export_step_dirs (cluster_lib.sh)."""
    env = dict(os.environ)
    env.update({
        "OPEN_FINANCE_ROOT":    str(head / "data" / "root" / "Open-Finance"),
        "SLEEP_OUT_ROOT":       str(out / "sleep"),
        "DEMAND_PREP_DIR":      str(out / "demand_prep"),
        "CF_FOUNDATION_DIR":    str(out / "counterfactuals"),
        "CF_COST_FWD":          str(out / "bbl"),
        "COST_POLFUNC_DIR":     str(out / "bbl"),
        "STATE_CENTERING_JSON": str(head / "data" / "input" / "state_centering_means.json"),
        "CL_ROOT":              str(head / "scripts"),
        "CL_DATA_IN":           str(head / "data" / "input"),
        "CL_DATA_OUT":          str(out),
    })
    return env


def local_env() -> dict:
    """Every override cleared — what a plain local run sees."""
    env = dict(os.environ)
    for k in ("OPEN_FINANCE_ROOT", "SLEEP_OUT_ROOT", "DEMAND_PREP_DIR", "CF_FOUNDATION_DIR",
              "CF_COST_FWD", "COST_POLFUNC_DIR", "STATE_CENTERING_JSON",
              "CL_ROOT", "CL_DATA_IN", "CL_DATA_OUT"):
        env.pop(k, None)
    return env


def probe(env: dict) -> dict:
    r = subprocess.run([sys.executable, "-c", PROBE, str(REPO)],
                       env=env, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"probe failed ({r.returncode}):\n{r.stderr.strip()}")
    return json.loads(r.stdout.strip().splitlines()[-1])


def norm(p: str) -> str:
    return os.path.normcase(os.path.normpath(p))


def inside(child: str, parent: Path) -> bool:
    c, p = norm(child), norm(str(parent))
    return c == p or c.startswith(p + os.sep)


def audit_python(a: Audit, head: Path, out: Path) -> None:
    print("\n[1/3] Python accessors under the CLUSTER environment")
    res = probe(cluster_env(head, out))
    for name, step in EXPECT_IN.items():
        got = res[name]
        if got.startswith("ERROR:"):
            a.check(False, name, got)
            continue
        if step:
            a.check(inside(got, out / step), name,
                    f"-> {got}" if not inside(got, out / step) else f"in {step}/")
        elif a.verbose:
            print(f"    --    {name}  -> {got}")

    print("\n[2/3] Python accessors with the cluster environment CLEARED")
    loc = probe(local_env())
    for name in EXPECT_IN:
        got = loc[name]
        if got.startswith("ERROR:"):
            a.check(False, f"{name} (local)", got)
            continue
        leaked = inside(got, head)
        a.check(not leaked, f"{name} (local)",
                f"LEAKED into the fake tree: {got}" if leaked else got)


def audit_julia(a: Audit, out: Path) -> None:
    print("\n[3/3] Julia seams in of_root.jl")
    julia = shutil.which("julia")
    if not julia:
        print("    SKIP  julia not on PATH")
        return

    loc_out = REPO / "fake_local" / "BLP_RESULTS"
    loc_in = REPO / "fake_local" / "DEMAND_PREP"
    r = subprocess.run([julia, "--startup-file=no", "-e", JULIA_PROBE,
                        str(REPO), str(out), str(loc_out), str(loc_in)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        a.check(False, "julia probe", r.stderr.strip()[:400])
        return

    g = {}
    for line in r.stdout.strip().splitlines():
        if "\t" in line:
            k, v = line.split("\t", 1)
            g[k] = v

    def eq(key: str, want: str) -> None:
        got = g.get(key, "<missing>")
        a.check(norm(got) == norm(want), key,
                got if norm(got) == norm(want) else f"got {got!r}, want {want!r}")

    a.check(g.get("is_cluster_out_cluster") == "true", "is_cluster_out(data/output)",
            g.get("is_cluster_out_cluster", "<missing>"))
    a.check(g.get("is_cluster_out_local") == "false", "is_cluster_out(BLP_RESULTS)",
            g.get("is_cluster_out_local", "<missing>"))

    eq("blp_dir_cluster",       str(out / "blp"))
    eq("blp_dir_local",         str(loc_out))
    eq("logit_dir_cluster",     str(out / "logit"))
    eq("logit_dir_local",       str(loc_out / "logit"))
    eq("draws_dir_cluster",     str(out / "blp" / "draws"))
    eq("draws_dir_local",       str(loc_out.parent / "BLP_DRAWS"))
    eq("cf_out_foundation",     str(out / "counterfactuals"))
    eq("cf_out_costfwd",        str(out / "bbl"))
    eq("cf_out_polfunc",        str(out / "bbl"))
    eq("cf_in_cluster",         str(out.parent / "input"))
    # The single-producer rule: on the cluster demand_search_dirs offers ONE directory, so a
    # stale uploaded vintage cannot shadow the one the prep step just wrote.
    eq("demand_search_cluster", str(out / "demand_prep"))
    eq("demand_search_local",   str(loc_in))

    # blp_logit.jl's delta_dir()/tex_out_dir() need a full run context (get_paths()), so they
    # are not invoked here. Both delegate to of_root.jl — delta_dir() forwards to logit_dir(),
    # tex_out_dir() branches on is_cluster_out — so the delegation is checked statically and the
    # behaviour it delegates to is covered by the assertions above.
    src = (REPO / "blp_logit.jl").read_text(encoding="utf-8", errors="replace")
    a.check("delta_dir() = logit_dir()" in src,
            "blp_1_logit delta_dir delegates to logit_dir")
    a.check("is_cluster_out(out_dir) ? logit_dir(out_dir)" in src,
            "blp_1_logit tex_out_dir branches on is_cluster_out")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-julia", action="store_true", help="skip the Julia half")
    ap.add_argument("-v", "--verbose", action="store_true", help="print every resolved path")
    args = ap.parse_args()

    a = Audit(args.verbose)
    tmp = Path(tempfile.mkdtemp(prefix="audit_paths_"))
    try:
        head, out = build_tree(tmp)
        print(f"fake cluster tree: {head}")
        audit_python(a, head, out)
        if not args.no_julia:
            audit_julia(a, out)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\n" + "=" * 70)
    if a.failed:
        print(f"FAILED — {len(a.failed)} assertion(s) failed, {a.passed} passed")
        for f in a.failed:
            print(f"  - {f}")
        return 1
    print(f"PASS — {a.passed} assertions, every seam resolves where it should")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
