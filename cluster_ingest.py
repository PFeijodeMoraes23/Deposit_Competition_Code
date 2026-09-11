#!/usr/bin/env python3
"""
cluster_ingest.py -- the ONE front door for a finished cluster run.

cluster_archive.sh packages the Bouchet run into data/output/download as eight
archives, one per step folder plus gates and logs:

    sleep_outputs  demand_prep_outputs  logit_outputs  blp_outputs
    bbl_outputs    counterfactuals_outputs            gates   logs

Each is either <BASE>[_<tag>].zip or, over SPLIT_BYTES, a run of
<BASE>[_<tag>].zip.part.NN pieces plus a <BASE>[_<tag>].zip.sha256 of the whole
reassembled zip. One sha256SUMS in the same folder covers every emitted file under
its plain relative name. Download that folder whole into CLUSTER_IN and run this.

WHAT IT DOES
  1. scans --in recursively and groups the files it finds into families, newest tag
     per family (the tag is the archiving job's SLURM id, so numeric-descending;
     an untagged family falls back to mtime);
  2. reassembles parts in NN order and verifies the result against the family's
     .zip.sha256, or the parts against their sha256SUMS lines when no .zip.sha256
     came down; whole zips are verified against their sha256SUMS line;
  3. routes each family to the local tree, STRIPPING the archive's domain prefix
     (output/<step>/ for data sets, logs/ for the log set) so members land where
     their local readers already look:

        sleep            -> DEMAND_PREP/            (est{k}/, Rout/, DIAGNOSTICS/)
                            ESTIMATION_OUTPUT/      (DIAG_PHI_SEPARATION/ — see
                                                     SLEEP_ESTOUT_DIRS)
        demand_prep      -> DEMAND_PREP/            (the demand_*_spec_*.parquet)
        logit            -> BLP_RESULTS/logit/  (.tex exhibits -> Drafts/Deposit Competition)
        blp              -> BLP_RESULTS/cluster_raw/ then process_blp_outputs
        bbl              -> process_cluster_outputs --kind bbl
        counterfactuals  -> process_cluster_outputs --kind cf
        gates            -> ESTIMATION_OUTPUT/CLUSTER_META/<tag>/
        logs             -> ESTIMATION_OUTPUT/CLUSTER_META/<tag>/logs/

A checksum MISMATCH refuses THAT FAMILY ONLY: the other seven still ingest and the
run exits 1. A missing checksum is reported, not fatal -- an archive can legitimately
arrive without its sha256SUMS line (a hand-made single-set zip, a partial download of
the folder), and refusing it would make the common case need a flag.

IT WRAPS THE TWO CONSOLIDATORS, it does not reimplement them. cluster_ingest_blp.py
(zip -> cluster_raw + cluster_processed/blp_E{k}_spec_12.jls + INDEX.json) and
cluster_ingest_bbl_cf.py (--kind bbl | cf) stay the authority on what a BLP/BBL/CF
artifact is and where it goes; both remain callable on their own. process_blp_outputs
writes its own SUMMARY.md next to the paper draft -- that is its behaviour, unchanged,
and the only thing in this path that touches Drafts. Nothing here writes there:
promotion into the draft stays a separate reviewed step.

SOURCE ARCHIVES ARE ONLY EVER READ OR COPIED -- never moved, never deleted. Some of
them are the only copy of a cluster artifact (the cluster is scratch), and leaving
them untouched is exactly what makes this script re-runnable: run it twice and the
second run does the same work against the same inputs and lands the same bytes.

BACKUP BEFORE OVERWRITE (default on, --no-backup disables). Anything the sleep family
is about to overwrite under DEMAND_PREP/est{k} or DEMAND_PREP/Rout is copied into
<--in>/_backup_<UTCstamp>.zip first. Concretely: est3/est4 estimation_results.pkl
carry two-stage AME attributes (bse_2s / pvalues_2s / cov_ame) attached locally after
the cluster wrote them, and a cluster pickle of the same name does not have them.

Usage:
    python cluster_ingest.py                       # everything in CLUSTER_IN
    python cluster_ingest.py --dry-run             # every check, writes nothing
    python cluster_ingest.py --family sleep --family gates
    python cluster_ingest.py --in D:/downloads/20260826 --no-backup
"""
from __future__ import annotations

import argparse
import bisect
import datetime
import hashlib
import io
import os
import re
import shutil
import sys
import zipfile
import zlib
from pathlib import Path

# Windows consoles default to cp1252 and the processors this drives print Greek (sigma, phi,
# delta) in their summaries. Without this an ingest that transported and verified perfectly is
# reported as a REFUSED family, sending the operator to re-download a sound archive. Same
# reconfigure the CF scripts use.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

REPO = Path(__file__).resolve().parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from utils import paths                                      # noqa: E402

# ── The eight families, in ingest order ──────────────────────────────────────────
# The order is the pipeline's own: sleep feeds demand_prep feeds logit feeds blp, and
# a run that refuses one family still ingests the rest, so ingesting upstream first
# means the tree is never left with a later step's outputs and not the earlier ones.
#
# BASE is cluster_archive.sh's, verbatim: the data sets are named <set>_outputs, the
# two meta sets are named for the set alone. Renaming one means renaming it there in
# the same edit -- the names are the whole contract between the two scripts.
FAMILY_BASE = {
    "sleep":           "sleep_outputs",
    "demand_prep":     "demand_prep_outputs",
    "logit":           "logit_outputs",
    "blp":             "blp_outputs",
    "bbl":             "bbl_outputs",
    "counterfactuals": "counterfactuals_outputs",
    "gates":           "gates",
    "logs":            "logs",
}
FAMILY_ORDER = tuple(FAMILY_BASE)
BASE_FAMILY = {b: f for f, b in FAMILY_BASE.items()}

# Member prefixes to strip. The archiver cds into the domain root before it zips, so
# a data member arrives as output/<step>/<path> and a log member as logs/<name>. gates
# is the one data set with no step folder of its own (its pattern is output/.gate_*),
# so it strips output/ alone. The bare "output/" fallback in _strip_prefix covers a
# member that carries the domain but not the step dir.
STRIP_PREFIXES = {
    "sleep":           ("output/sleep/",),
    "demand_prep":     ("output/demand_prep/",),
    "logit":           ("output/logit/",),
    "blp":             ("output/blp/",),
    "bbl":             ("output/bbl/",),
    "counterfactuals": ("output/counterfactuals/",),
    "gates":           ("output/",),
    "logs":            ("logs/",),
}

# <BASE>[_<tag>].zip[.part.NN] and <BASE>[_<tag>].zip.sha256. The tag excludes '.' so
# the .zip / .part / .sha256 boundaries can never be eaten by it. No BASE is a prefix
# of another, so the alternation is unambiguous.
_BASES_RE = "|".join(sorted(FAMILY_BASE.values(), key=len, reverse=True))
ARCHIVE_RE = re.compile(
    rf"^(?P<base>{_BASES_RE})(?:_(?P<tag>[^.]+))?\.zip(?:\.part\.(?P<part>\d+))?$")
SHA_RE = re.compile(rf"^(?P<base>{_BASES_RE})(?:_(?P<tag>[^.]+))?\.zip\.sha256$")

# Our own working directories inside --in, skipped by the scan so a kept
# _reassembled/ from a --keep-parts run is never mistaken for a second archive.
REASSEMBLED_DIR = "_reassembled"
BACKUP_PREFIX = "_backup_"

# Mirrors cluster_archive.sh's STORE_SUFFIXES: float payloads do not deflate, and the
# backup zip is written while the user waits for the ingest.
STORE_SUFFIXES = {".pkl", ".parquet", ".jls", ".npz", ".zip", ".gz", ".so"}

CHUNK = 1 << 20


# ── small helpers ────────────────────────────────────────────────────────────────
def fmt_size(n):
    for unit, div in (("GB", 1 << 30), ("MB", 1 << 20), ("KB", 1 << 10)):
        if n >= div:
            return f"{n / div:.1f} {unit}"
    return f"{n} B"


def sha256_stream(fh):
    h = hashlib.sha256()
    for blk in iter(lambda: fh.read(CHUNK), b""):
        h.update(blk)
    return h.hexdigest()


def sha256_file(path: Path):
    with open(path, "rb") as fh:
        return sha256_stream(fh)


def safe_rel(member: str):
    """Reject absolute members and any '..' traversal before writing anything to disk.
    Same rule as process_cluster_outputs._safe_rel, and for the same reason: a zip is
    an untrusted name list until each name has been proved relative."""
    rel = member.replace("\\", "/").lstrip("/")
    parts = [p for p in rel.split("/") if p not in ("", ".")]
    if any(p == ".." for p in parts) or (len(member) > 1 and member[1] == ":"):
        return None
    return "/".join(parts)


def strip_prefix(rel: str, family: str):
    """Drop the archive's domain prefix so members land at the path their local reader
    uses. output/sleep/est3/x.pkl -> est3/x.pkl; logs/pipe.out -> pipe.out."""
    for pre in STRIP_PREFIXES[family] + ("output/", "logs/"):
        if rel.startswith(pre):
            return rel[len(pre):]
    return rel


class ConcatReader(io.RawIOBase):
    """Read-only seekable view over an ordered run of .part.NN files, as though they
    had already been concatenated.

    It is what lets --dry-run do the FULL check -- hash the whole zip and list its
    members -- without writing the reassembled archive anywhere. A dry run that could
    not read a split family would be checking the one shape most likely to be broken.
    """

    def __init__(self, parts):
        super().__init__()
        self._parts = [Path(p) for p in parts]
        self._sizes = [p.stat().st_size for p in self._parts]
        self._offs, total = [], 0
        for s in self._sizes:
            self._offs.append(total)
            total += s
        self._size = total
        self._pos = 0
        self._fh = None
        self._idx = -1

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self._pos

    def seek(self, off, whence=io.SEEK_SET):
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self._pos, io.SEEK_END: self._size}[whence]
        self._pos = max(0, min(base + off, self._size))
        return self._pos

    def readinto(self, buf):
        want = len(buf)
        if want == 0 or self._pos >= self._size:
            return 0
        i = bisect.bisect_right(self._offs, self._pos) - 1
        if i != self._idx:
            if self._fh is not None:
                self._fh.close()
            self._fh = open(self._parts[i], "rb")
            self._idx = i
        within = self._pos - self._offs[i]
        self._fh.seek(within)
        data = self._fh.read(min(want, self._sizes[i] - within))
        buf[:len(data)] = data
        self._pos += len(data)
        return len(data)

    def close(self):
        if self._fh is not None:
            self._fh.close()
            self._fh = None
        super().close()


def open_parts(parts):
    """Buffered file object over the parts, ready to hand to zipfile.ZipFile."""
    return io.BufferedReader(ConcatReader(parts))


# ── discovery ────────────────────────────────────────────────────────────────────
class Archive:
    """One family at one tag: the whole zip or its parts, plus the sidecar checksum."""

    def __init__(self, family, base, tag):
        self.family = family
        self.base = base
        self.tag = tag
        self.whole = None
        self.parts = {}          # NN -> Path
        self.sha_file = None

    @property
    def name(self):
        return f"{self.base}{'_' + self.tag if self.tag else ''}.zip"

    @property
    def ordered_parts(self):
        return [self.parts[k] for k in sorted(self.parts)]

    @property
    def files(self):
        return ([self.whole] if self.whole else []) + self.ordered_parts

    @property
    def mtime(self):
        return max((p.stat().st_mtime for p in self.files), default=0.0)

    @property
    def size(self):
        return sum(p.stat().st_size for p in self.files)

    def sort_key(self):
        """Newest tag wins: a numeric tag is the archiving job's SLURM id and orders
        directly; anything else (no tag, a hand-written one) falls back to mtime."""
        numeric = self.tag is not None and self.tag.isdigit()
        return (1 if numeric else 0, int(self.tag) if numeric else 0, self.mtime)


def scan(in_dir: Path):
    """-> ({family: {tag: Archive}}, {dir: {name: sha}}). One pass over --in."""
    groups, sums = {}, {}
    for dirpath, dirnames, filenames in os.walk(in_dir):
        dirnames[:] = [d for d in dirnames if d != REASSEMBLED_DIR]
        here = Path(dirpath)
        for name in filenames:
            if name == "sha256SUMS":
                sums[here] = parse_sums(here / name)
                continue
            if name.startswith(BACKUP_PREFIX):
                continue
            m = ARCHIVE_RE.match(name) or SHA_RE.match(name)
            if not m:
                continue
            fam = BASE_FAMILY[m.group("base")]
            arc = groups.setdefault(fam, {}).get(m.group("tag"))
            if arc is None:
                arc = Archive(fam, m.group("base"), m.group("tag"))
                groups[fam][m.group("tag")] = arc
            if m.re is SHA_RE:
                arc.sha_file = here / name
            elif m.group("part") is None:
                arc.whole = here / name
            else:
                arc.parts[int(m.group("part"))] = here / name
    return groups, sums


def parse_sums(path: Path):
    """sha256sum output -> {name: sha}. Accepts both the text ('  name') and binary
    ('*name') markers, and ignores the directory the names were written relative to."""
    out = {}
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line or len(line) < 66:
                    continue
                sha, rest = line.split(None, 1)
                if len(sha) != 64:
                    continue
                out[os.path.basename(rest.lstrip("*").strip())] = sha.lower()
    except OSError:
        pass
    return out


def sums_lookup(sums, name, near: Path):
    """-> (sha, source dir) for `name`. The sha256SUMS sitting beside the archive wins;
    a download split across dated folders can still resolve against another one."""
    if near in sums and name in sums[near]:
        return sums[near][name], near
    for d, table in sorted(sums.items(), key=lambda kv: str(kv[0])):
        if name in table:
            return table[name], d
    return None, None


# ── preparing one archive: reassemble + verify ───────────────────────────────────
class Refused(Exception):
    """A check this family cannot pass. Refuses the family, never the run."""


def prepare(arc: Archive, sums, in_dir: Path, dry: bool, note):
    """-> (path_or_None, opener, n_parts). `opener` returns a file object for the whole
    archive; for a dry run over parts that is the concatenating reader, so nothing is
    written. Raises Refused on a checksum mismatch."""
    if arc.whole is not None:
        if arc.parts:
            note(f"both a whole zip and {len(arc.parts)} part(s) present; "
                 "using the whole zip")
        expected, src = sums_lookup(sums, arc.name, arc.whole.parent)
        if expected is None and arc.sha_file is not None:
            expected, src = first_sha(arc.sha_file), arc.sha_file.parent
        if expected is None:
            note("no checksum for this archive (not verified)")
        else:
            got = sha256_file(arc.whole)
            if got != expected:
                raise Refused(f"sha256 mismatch vs {src}{os.sep}sha256SUMS "
                              f"(expected {expected[:12]}..., got {got[:12]}...)")
            note("sha256 OK (whole zip)")
        return arc.whole, (lambda: open(arc.whole, "rb")), 0

    if not arc.parts:
        raise Refused("no zip and no parts")
    parts = arc.ordered_parts
    missing = [n for n in range(len(parts)) if n not in arc.parts]
    if missing:
        raise Refused(f"parts {missing} missing from the .part.NN run")

    # The .sha256 the archiver writes beside a split zip covers the REASSEMBLED whole,
    # which is the only checksum that proves the parts were concatenated in the right
    # order. Without it, per-part sha256SUMS lines prove transport but not assembly, so
    # the zip's own central directory becomes the assembly check further down.
    expected = first_sha(arc.sha_file) if arc.sha_file is not None else None
    if expected is None:
        expected, _src = sums_lookup(sums, arc.name, parts[0].parent)
    if expected is not None:
        with open_parts(parts) as fh:
            got = sha256_stream(fh)
        if got != expected:
            raise Refused(f"sha256 mismatch on the reassembled zip "
                          f"(expected {expected[:12]}..., got {got[:12]}...)")
        note(f"sha256 OK (reassembled from {len(parts)} parts)")
    else:
        checked = 0
        for p in parts:
            exp, _src = sums_lookup(sums, p.name, p.parent)
            if exp is None:
                continue
            if sha256_file(p) != exp:
                raise Refused(f"sha256 mismatch on part {p.name}")
            checked += 1
        if checked:
            note(f"sha256 OK ({checked}/{len(parts)} parts; "
                 "no whole-zip checksum came down)")
        else:
            note(f"no checksum for {len(parts)} part(s) (not verified)")

    if dry:
        return None, (lambda: open_parts(parts)), len(parts)

    out_dir = in_dir / REASSEMBLED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    whole = out_dir / arc.name
    with open(whole, "wb") as dst:
        for p in parts:
            with open(p, "rb") as src:
                shutil.copyfileobj(src, dst, CHUNK)
    note(f"reassembled {len(parts)} parts -> {whole.name} "
         f"({fmt_size(whole.stat().st_size)})")
    return whole, (lambda: open(whole, "rb")), len(parts)


def first_sha(path: Path):
    """The single '<sha>  <name>' line of a .zip.sha256 sidecar."""
    for sha in parse_sums(path).values():
        return sha
    return None


# ── backup of anything about to be overwritten ───────────────────────────────────
class Backup:
    """One zip per run, created lazily at <--in>/_backup_<UTCstamp>.zip.

    Only files that (a) already exist and (b) would be replaced are stored, so a first
    ingest into an empty tree writes no backup at all and the zip's presence is itself
    the signal that something was replaced.
    """

    def __init__(self, in_dir: Path, stamp: str, enabled: bool, dry: bool):
        self.path = in_dir / f"{BACKUP_PREFIX}{stamp}.zip"
        self.enabled = enabled
        self.dry = dry
        self._zf = None
        self.n = 0
        self.n_same = 0
        self._root = paths.demand_prep_root()
        self._root_resolved = self._root.resolve()

    def guard(self, dest: Path):
        """True for a destination whose local copy can hold work the cluster never had:
        the est{k} pickles (two-stage AME attributes are attached locally) and the Rout
        exports built from them."""
        if not self.enabled:
            return False
        try:
            rel = dest.resolve().relative_to(self._root_resolved)
        except (ValueError, OSError):
            return False
        head = rel.parts[0] if rel.parts else ""
        return head == "Rout" or head.startswith("est")

    @staticmethod
    def _same(dest: Path, info):
        """The file on disk already IS the member about to replace it (size + CRC32,
        both of which the zip carries in its directory)."""
        try:
            if dest.stat().st_size != info.file_size:
                return False
            crc = 0
            with open(dest, "rb") as fh:
                for blk in iter(lambda: fh.read(CHUNK), b""):
                    crc = zlib.crc32(blk, crc)
            return crc == info.CRC
        except OSError:
            return False

    def add(self, dest: Path, info=None):
        """-> True when `dest` was stored. A destination whose bytes already equal the
        incoming member is NOT stored: overwriting it loses nothing, and on a re-ingest
        that is every file -- which is what stops a second run from spending gigabytes
        zipping the est{k} pickles into a copy of themselves."""
        if not dest.is_file() or not self.guard(dest):
            return False
        if info is not None and self._same(dest, info):
            self.n_same += 1
            return False
        self.n += 1
        if self.dry:
            return True
        if self._zf is None:
            self._zf = zipfile.ZipFile(self.path, "w", zipfile.ZIP_DEFLATED)
        rel = os.path.relpath(dest, self._root)
        comp = (zipfile.ZIP_STORED if dest.suffix.lower() in STORE_SUFFIXES
                else zipfile.ZIP_DEFLATED)
        self._zf.write(dest, arcname=f"DEMAND_PREP/{rel}".replace("\\", "/"),
                       compress_type=comp)
        return True

    def close(self):
        if self._zf is not None:
            self._zf.close()
            self._zf = None


# ── routing ──────────────────────────────────────────────────────────────────────
def dest_root(family, tag):
    """The one directory this family's members belong in, resolved through utils.paths
    so a redirected tree (OPEN_FINANCE_ROOT, SLEEP_OUT_ROOT) carries the ingest with it."""
    if family == "sleep":
        return paths.demand_prep_root()
    if family == "demand_prep":
        return paths.demand_parquet_dir()
    if family == "logit":
        return paths.blp_results_dir() / "logit"
    if family == "blp":
        return paths.blp_results_dir() / "cluster_raw"
    if family == "bbl":
        return paths.bbl_output_dir() / "cluster_processed"
    if family == "counterfactuals":
        return paths.cf_foundation_dir()
    meta = paths.estimation_output() / "CLUSTER_META" / tag
    return meta / "logs" if family == "logs" else meta


def short(p):
    """A destination relative to ESTIMATION_OUTPUT, which the header prints in full.
    The absolute paths run past 130 characters and wrap the summary table into noise."""
    try:
        return str(Path(p).relative_to(paths.estimation_output()))
    except ValueError:
        return str(p)


def meta_tag(arc: Archive):
    """Folder name for the gates/logs snapshot. The SLURM tag when there is one; an
    untagged archive is stamped with its own date so two hand-made downloads of the
    same set do not land on top of each other."""
    if arc.tag:
        return arc.tag
    return f"untagged_{datetime.datetime.fromtimestamp(arc.mtime):%Y%m%d}"


# Sleep-family members that belong BESIDE DEMAND_PREP under ESTIMATION_OUTPUT, not inside it.
# The D6 entry job resolves its output through utils.paths.estimation_output(), which is
# deliberately not redirectable per sleepiness vintage, so on the cluster these sit next to the
# step folders rather than under output/sleep. Named one by one on purpose: output/sleep's own
# DIAGNOSTICS/ is a genuine child of the step folder and must keep landing in DEMAND_PREP, so a
# prefix rule like "starts with DIAG" would send it to the wrong tree.
SLEEP_ESTOUT_DIRS = ("DIAG_PHI_SEPARATION",)


def member_dest(family, rel: "str | Path", root: Path) -> Path:
    """Where one member belongs. Families land in one directory, with two exceptions.

    The logit family carries two different KINDS of product: the delta warm-starts, .jls fits
    and summary json, which belong with the BLP results the RC engine reads, and three .tex
    fragments that are paper exhibits. Sending the fragments to BLP_RESULTS leaves the paper
    reading whatever it read last: the cluster recomputes them every run and V_Main never sees
    the new numbers. Table generators write their .tex to the paper directory and nowhere else,
    and a downloaded exhibit is the same kind of object, so it follows the same rule.

    The sleep family carries SLEEP_ESTOUT_DIRS for the same reason in the other direction:
    those are written one level above the step folder, and routing them into DEMAND_PREP with
    the rest of the family would put them where no local reader looks.
    """
    rel = Path(rel)   # extract() hands over the slash-joined str from strip_prefix
    if family == "logit" and rel.suffix.lower() == ".tex":
        return paths.drafts_dir() / rel.name
    if family == "sleep" and rel.parts and rel.parts[0] in SLEEP_ESTOUT_DIRS:
        return paths.estimation_output() / rel
    return root / rel


def extract(opener, family, root: Path, backup: Backup, dry: bool, note):
    """Extract every member into `root` with the domain prefix stripped.
    -> (n_files, n_bytes, n_backed_up)."""
    n = nbytes = nbak = 0
    if not dry:
        root.mkdir(parents=True, exist_ok=True)
    with opener() as fh, zipfile.ZipFile(fh) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            rel = safe_rel(info.filename)
            if rel is None:
                note(f"SKIPPED unsafe member name: {info.filename}")
                continue
            rel = strip_prefix(rel, family)
            if not rel:
                continue
            dest = member_dest(family, rel, root)
            if backup.add(dest, info):
                nbak += 1
            if not dry:
                dest.parent.mkdir(parents=True, exist_ok=True)
                with z.open(info) as src, open(dest, "wb") as dst:
                    shutil.copyfileobj(src, dst, CHUNK)
            n += 1
            nbytes += info.file_size
    return n, nbytes, nbak


def count_members(opener):
    with opener() as fh, zipfile.ZipFile(fh) as z:
        return sum(1 for i in z.infolist() if not i.is_dir())


def call_main(mod, argv):
    """Run an importable script's main() with a pinned argv. Both consolidators are
    argparse scripts whose main() reads sys.argv, so this is their entry point --
    in-process, so they resolve the same utils.paths this run resolved."""
    old = sys.argv
    sys.argv = list(argv)
    try:
        rc = mod.main()
    except SystemExit as exc:                        # both call sys.exit() on bad input
        rc = exc.code
    finally:
        sys.argv = old
    if rc in (None, 0):
        return 0
    if not isinstance(rc, int):
        # sys.exit("ERROR: ...") carries the message as the code. Nothing prints it once
        # the SystemExit is caught, so it would vanish exactly when it is needed.
        print(f"     {rc}")
        return 1
    return rc


def route(arc: Archive, zip_path, opener, dry: bool, backup: Backup, note):
    """-> (n_files, destination). Raises Refused when a wrapped consolidator fails."""
    fam = arc.family
    if fam in ("sleep", "demand_prep", "logit", "gates", "logs"):
        root = dest_root(fam, meta_tag(arc))
        same_before = backup.n_same
        n, nbytes, nbak = extract(opener, fam, root, backup, dry, note)
        if nbak:
            note(f"{nbak} existing file(s) {'to be ' if dry else ''}saved to "
                 f"{backup.path.name}")
        if backup.n_same > same_before:
            note(f"{backup.n_same - same_before} destination(s) already hold these exact "
                 "bytes (nothing to back up)")
        note(f"{fmt_size(nbytes)} uncompressed")
        return n, root

    n = count_members(opener)
    if fam == "blp":
        # The zip itself is the artifact process_blp_outputs consumes: it extracts into
        # cluster_raw/, flattens the output/blp/ prefix away and builds cluster_processed.
        raw = dest_root("blp", None)
        if dry:
            note("would copy the zip into cluster_raw/ and run process_blp_outputs")
            return n, raw
        raw.mkdir(parents=True, exist_ok=True)
        target = raw / arc.name
        if zip_path.resolve() != target.resolve():
            shutil.copy2(zip_path, target)
        note(f"copied {arc.name} -> {raw}")
        import cluster_ingest_blp as blp_mod
        if call_main(blp_mod, ["cluster_ingest_blp.py"]) != 0:
            raise Refused("process_blp_outputs failed")
        return n, raw

    kind = "bbl" if fam == "bbl" else "cf"
    if zip_path is None:                             # dry run over an unassembled split
        note(f"would run process_cluster_outputs --kind {kind}")
        return n, dest_root(fam, None)
    import cluster_ingest_bbl_cf as cco
    argv = ["cluster_ingest_bbl_cf.py", "--kind", kind, "--zip", str(zip_path)]
    if dry:
        argv.append("--dry-run")
    if call_main(cco, argv) != 0:
        raise Refused(f"process_cluster_outputs --kind {kind} failed")
    return n, dest_root(fam, None)


# ── main ─────────────────────────────────────────────────────────────────────────
def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Reassemble, verify and route one cluster download into the local tree.")
    ap.add_argument("--in", dest="in_dir", default=None,
                    help="folder holding the download (default: ESTIMATION_OUTPUT/CLUSTER_IN)")
    ap.add_argument("--family", action="append", default=[], metavar="NAME",
                    help=f"ingest only this family (repeatable): {', '.join(FAMILY_ORDER)}")
    ap.add_argument("--dry-run", action="store_true",
                    help="run every check, including checksums, and write nothing")
    ap.add_argument("--no-backup", action="store_true",
                    help="do not save DEMAND_PREP files that are about to be overwritten")
    ap.add_argument("--keep-parts", action="store_true",
                    help="keep the reassembled zips under <--in>/_reassembled/")
    args = ap.parse_args(argv)

    in_dir = Path(args.in_dir).expanduser() if args.in_dir else \
        paths.estimation_output() / "CLUSTER_IN"
    if not in_dir.is_dir():
        print(f"ERROR: --in not found: {in_dir}")
        return 1

    wanted = []
    for name in args.family:
        fam = name if name in FAMILY_BASE else BASE_FAMILY.get(name)
        if fam is None:
            print(f"ERROR: unknown --family {name!r}; expected one of "
                  f"{', '.join(FAMILY_ORDER)}")
            return 2
        wanted.append(fam)
    selected = tuple(f for f in FAMILY_ORDER if f in wanted) if wanted else FAMILY_ORDER

    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = Backup(in_dir, stamp, not args.no_backup, args.dry_run)

    print("=" * 96)
    print(f"ingest_cluster_downloads{'  (DRY RUN, nothing written)' if args.dry_run else ''}")
    print("=" * 96)
    print(f"  from     : {in_dir}")
    print(f"  into     : {paths.estimation_output()}   (destinations below are relative to it)")
    print(f"  families : {', '.join(selected)}")
    print(f"  backup   : {'off (--no-backup)' if args.no_backup else backup.path.name}")

    groups, sums = scan(in_dir)
    if sums:
        print(f"  checksums: {sum(len(t) for t in sums.values())} name(s) from "
              f"{len(sums)} sha256SUMS file(s)")
    else:
        print("  checksums: no sha256SUMS found under --in")
    print()

    def note(msg):
        print(f"     {msg}")

    rows, refused = [], 0
    for fam in selected:
        tags = groups.get(fam)
        if not tags:
            print(f"-- {fam:<16} not present in this download")
            rows.append((fam, "-", "-", "-", "-", "absent"))
            continue
        arc = max(tags.values(), key=Archive.sort_key)
        shape = f"{len(arc.parts)} parts" if arc.parts and not arc.whole else "whole"
        print(f"-- {fam:<16} {arc.name}  {fmt_size(arc.size)} ({shape})")
        if len(tags) > 1:
            others = ", ".join(sorted(t or "(untagged)" for t in tags if t != arc.tag))
            note(f"{len(tags)} tags present ({others}); taking "
                 f"{arc.tag or '(untagged)'} (newest)")
        zip_path = None
        try:
            zip_path, opener, _n_parts = prepare(arc, sums, in_dir, args.dry_run, note)
            n_files, dest = route(arc, zip_path, opener, args.dry_run, backup, note)
            verdict = "OK"
            note(f"{n_files} file(s) -> {short(dest)}")
        except Refused as exc:
            verdict = f"REFUSED: {exc}"
            n_files, dest = 0, dest_root(fam, meta_tag(arc))
            note(verdict)
            refused += 1
        except Exception as exc:                     # one family's failure, not the run's
            verdict = f"REFUSED: {type(exc).__name__}: {exc}"
            n_files, dest = 0, dest_root(fam, meta_tag(arc))
            note(verdict)
            refused += 1
        rows.append((fam, arc.name, fmt_size(arc.size), str(n_files), short(dest), verdict))

        # The reassembled zip is a transient: the parts it was built from stay in --in,
        # and every consumer that needed a real file on disk has already read it. The
        # parent test is what keeps this from ever reaching a SOURCE archive -- only a
        # path this run wrote under _reassembled/ is eligible.
        if (zip_path is not None and not args.keep_parts
                and zip_path.parent == in_dir / REASSEMBLED_DIR):
            try:
                zip_path.unlink()
            except OSError:
                pass

    backup.close()
    if not args.keep_parts:
        try:
            (in_dir / REASSEMBLED_DIR).rmdir()          # only ever when it emptied itself
        except OSError:
            pass
    if backup.n:
        verb = "would be saved to" if args.dry_run else "saved to"
        print(f"\n{backup.n} file(s) {verb} {backup.path}")

    head = ("family", "archive", "size", "files", "destination", "verdict")
    widths = [max(len(str(r[i])) for r in (rows + [head])) for i in range(6)]
    print()
    print("=" * 96)
    print("  ".join(h.ljust(widths[i]) for i, h in enumerate(head)))
    print("  ".join("-" * widths[i] for i in range(6)))
    for r in rows:
        print("  ".join(str(r[i]).ljust(widths[i]) for i in range(6)))
    print()

    ok = sum(1 for r in rows if r[5] == "OK")
    verb = "would ingest" if args.dry_run else "ingested"
    print(f"{ok}/{len(rows)} families {verb}"
          f"{f'; {refused} REFUSED' if refused else ''}.")
    print("Source archives were left where they were found; nothing was moved or deleted.")
    if refused:
        print("The rest of the tree is current. A checksum mismatch means re-download that")
        print("family; any other error is local processing — the archive on disk is intact.")
    return 1 if refused else 0


if __name__ == "__main__":
    sys.exit(main())
