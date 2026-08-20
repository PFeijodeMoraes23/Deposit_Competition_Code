"""Per-column fingerprint of every DEMAND_PREP parquet.

Run once BEFORE re-running demand prep (--out baseline.json) and once after (--out after.json),
then --compare to prove which columns actually changed. This is what decides whether the logit /
BLP stages need re-running: if every pre-existing column is byte-identical and the only delta is
the two ADDED level columns, nothing downstream that selects columns by name can have moved.
"""
import argparse, hashlib, json, pathlib, sys
import pandas as pd

from utils import paths as _paths

sys.stdout.reconfigure(encoding="utf-8")

# demand_prep_root() honours SLEEP_OUT_ROOT, so a fingerprint taken around a sandboxed
# demand-prep run describes the tree that run actually wrote.
D = _paths.demand_prep_root()


def col_hash(s: pd.Series) -> str:
    # hash_pandas_object is NaN-stable and dtype-aware; fold to one digest per column.
    h = pd.util.hash_pandas_object(s, index=False).values
    return hashlib.blake2b(h.tobytes(), digest_size=16).hexdigest()


def fingerprint():
    out = {}
    files = sorted(D.glob("*.parquet"))
    for i, f in enumerate(files, 1):
        df = pd.read_parquet(f)
        out[f.name] = {"n_rows": int(len(df)), "cols": {c: col_hash(df[c]) for c in df.columns}}
        print(f"  [{i:2d}/{len(files)}] {f.name:44s} {len(df):>8,} rows x {df.shape[1]:>3} cols")
        del df
    return out


def compare(a_path, b_path):
    A = json.loads(pathlib.Path(a_path).read_text())
    B = json.loads(pathlib.Path(b_path).read_text())

    only_a = sorted(set(A) - set(B));  only_b = sorted(set(B) - set(A))
    if only_a: print(f"files only in BEFORE ({len(only_a)}): {only_a[:5]}")
    if only_b: print(f"files only in AFTER  ({len(only_b)}): {only_b[:5]}")

    n_same = n_changed = 0
    added_cols, removed_cols, changed_cols, row_deltas = {}, {}, {}, {}
    for name in sorted(set(A) & set(B)):
        ca, cb = A[name]["cols"], B[name]["cols"]
        add = sorted(set(cb) - set(ca)); rem = sorted(set(ca) - set(cb))
        chg = sorted(c for c in set(ca) & set(cb) if ca[c] != cb[c])
        if A[name]["n_rows"] != B[name]["n_rows"]:
            row_deltas[name] = (A[name]["n_rows"], B[name]["n_rows"])
        if add: added_cols[name] = add
        if rem: removed_cols[name] = rem
        if chg: changed_cols[name] = chg; n_changed += 1
        else: n_same += 1

    print(f"\n{'='*78}")
    print(f"{n_same} file(s) with EVERY shared column identical; {n_changed} with changes")
    print("=" * 78)

    if row_deltas:
        print(f"\n!! ROW COUNT CHANGED in {len(row_deltas)} file(s):")
        for k, (x, y) in list(row_deltas.items())[:10]: print(f"   {k}: {x:,} -> {y:,}")
    else:
        print("\nrow counts: IDENTICAL in every file")

    if added_cols:
        allad = sorted({c for v in added_cols.values() for c in v})
        print(f"\nADDED columns ({len(added_cols)} files): {allad}")
    if removed_cols:
        allrm = sorted({c for v in removed_cols.values() for c in v})
        print(f"\n!! REMOVED columns ({len(removed_cols)} files): {allrm}")
    if changed_cols:
        allch = sorted({c for v in changed_cols.values() for c in v})
        print(f"\n!! CHANGED columns ({len(changed_cols)} files), {len(allch)} distinct:")
        for c in allch[:40]:
            k = [f for f, v in changed_cols.items() if c in v]
            print(f"   {c:38s} in {len(k)} file(s)")
    else:
        print("\nCHANGED columns: NONE — every pre-existing column is bit-identical.")

    verdict = (not changed_cols) and (not removed_cols) and (not row_deltas)
    print(f"\nVERDICT: {'SAFE' if verdict else 'NOT SAFE'} — downstream stages that select columns "
          f"by name are {'UNAFFECTED' if verdict else 'POTENTIALLY AFFECTED'}.")
    return 0 if verdict else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"))
    a = ap.parse_args()
    if a.compare:
        sys.exit(compare(*a.compare))
    fp = fingerprint()
    pathlib.Path(a.out).write_text(json.dumps(fp, indent=1))
    ncols = sum(len(v["cols"]) for v in fp.values())
    print(f"\nwrote {a.out}: {len(fp)} files, {ncols:,} column hashes")
