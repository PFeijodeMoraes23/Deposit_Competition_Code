"""make_ccs_flows.py -- the BCB's CCS relationship flows (LAI 18810.025400/2026-04) as a tidy
institution x person type x month file.

Input: DATA_RAW/BCB/ccs_lai_18810_025400/raw/CCSEstatDTInicioTP_201601_a_202501.xlsx, read as delivered.
Output: DATA_RAW/BCB/ccs_lai_18810_025400/compiled/ccs_flows_institution_month.csv and, beside it,
ccs_flows_checks.json (every acceptance check with its value).

What the two sheets hold (verified against each other here, not inferred from the labels):
  Acumulado  per month, three measures: "Abertos no mes e encerrados posteriormente"
             (opened_closed_later), "Abertos no mes" (opened_active: opened in the month and still
             active at the extraction date) and "Encerrados no mes" (closed: closures in the month of
             relationships of any start date).
  Atual      per month, "Abertos no mes" only. It must equal Acumulado's middle measure in every
             institution x type x month cell (an institution absent from Atual counts as zero there).
opened_gross = opened_active + opened_closed_later.

Layout: column A is empty; an institution is a block of a PF row (CNPJ, name, segment filled), a PJ
row (those blank) and a Total row (segment "Total"); either type row can be absent. The last data row
is the grand total (CNPJ, name, segment and type blank). A row that carries a name but no CNPJ starts a
block of its own with cnpj_missing = 1 (BCO BEM in Acumulado, which repeats BCO BBI's PJ values and is
counted in the BCB's grand total). Blank cells are zero.

A relationship is one CPF or CNPJ at one institution: not a person and not an account. "Still active"
is measured at an extraction date after December 2024 that the BCB does not state.

The run fails (nonzero exit, nothing written) when any acceptance check breaks.
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import openpyxl
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import raw_paths  # noqa: E402

DATASET = raw_paths.BCB_CCS_LAI.parent
XLSX = raw_paths.BCB_CCS_LAI / "CCSEstatDTInicioTP_201601_a_202501.xlsx"
OUT = DATASET / "compiled" / "ccs_flows_institution_month.csv"
CHECKS = DATASET / "compiled" / "ccs_flows_checks.json"
XLSX_SHA256 = "a8742d4959a9a68f799c06d03c72f5f4340821b2f7f3b28e5780c83d8ea7abab"

MONTHS_PT = {"JAN": 1, "FEV": 2, "MAR": 3, "ABR": 4, "MAI": 5, "JUN": 6,
             "JUL": 7, "AGO": 8, "SET": 9, "OUT": 10, "NOV": 11, "DEZ": 12}
MONTH_RE = re.compile(r"^\s*([A-Z]{3})/(\d{4})\s*$")
MEASURES = {"Abertos no mês e encerrados posteriormente": "opened_closed_later",
            "Abertos no mês": "opened_active",
            "Encerrados no mês": "closed"}

# The acceptance checks of the handoff of 2026-10-30 (the main thread's probe of this file).
EXPECT = {
    "atual_blocks": 1190, "atual_rows": 2150, "atual_pf": 1148, "atual_pj": 1002,
    "acum_blocks_with_cnpj": 1557, "acum_blocks_missing_cnpj": 1,
    "acum_rows": 2864, "acum_pf": 1524, "acum_pj": 1340,
    "only_in_acum": 367, "months": 108,
    "equality_cells": 309204,
    "grand_opened_active": 875852059,
    "grand_opened_closed_later_m": 531.434, "grand_closed_m": 445.067,
}
NATIONAL_M = {  # (type, year): opened_gross, opened_active, opened_closed_later, closed, in millions
    ("PF", 2016): (31.08, 18.81, 12.27, 20.33), ("PF", 2017): (34.40, 20.75, 13.64, 10.28),
    ("PF", 2018): (49.68, 30.40, 19.29, 12.05), ("PF", 2019): (88.56, 49.05, 39.51, 14.45),
    ("PF", 2020): (209.14, 136.83, 72.31, 23.48), ("PF", 2021): (229.41, 127.37, 102.04, 37.29),
    ("PF", 2022): (258.25, 154.62, 103.63, 68.85), ("PF", 2023): (206.82, 137.53, 69.28, 77.80),
    ("PF", 2024): (231.14, 169.33, 61.80, 148.22),
    ("PJ", 2016): (2.48, 0.64, 1.84, 1.86), ("PJ", 2017): (1.97, 0.69, 1.28, 1.15),
    ("PJ", 2018): (2.42, 0.89, 1.52, 1.23), ("PJ", 2019): (4.23, 1.28, 2.95, 1.42),
    ("PJ", 2020): (7.27, 2.43, 4.84, 3.32), ("PJ", 2021): (9.77, 3.39, 6.39, 4.39),
    ("PJ", 2022): (12.71, 5.50, 7.21, 4.39), ("PJ", 2023): (13.17, 7.04, 6.14, 5.20),
    ("PJ", 2024): (14.79, 9.31, 5.48, 9.37),
}


def cell_text(v) -> str:
    return "" if v is None else str(v).strip()


def cell_num(v, where: str) -> int:
    """A count cell: blank is zero; anything that is not a whole number stops the run."""
    if v is None or (isinstance(v, str) and not v.strip()):
        return 0
    if isinstance(v, (int, float)) and float(v) == int(v):
        return int(v)
    raise SystemExit(f"non-integer count {v!r} at {where}")


def sheet_rows(ws) -> list:
    """Every row of a read-only sheet as a tuple of one width: read-only mode drops trailing empty cells."""
    rows = [tuple(r) for r in ws.iter_rows(values_only=True)]
    w = max(len(r) for r in rows)
    return [r + (None,) * (w - len(r)) for r in rows]


def ym(label: str) -> str:
    m = MONTH_RE.match(label)
    return f"{m.group(2)}-{MONTHS_PT[m.group(1)]:02d}"


def id_columns(row) -> dict:
    """{field: column index} of the identifier header row (CNPJ, Nome Instituicao, Segmento, Tipo)."""
    want = {"CNPJ": "cnpj", "Nome Instituição": "name", "Segmento": "segment", "Tipo": "type"}
    got = {want[cell_text(v)]: j for j, v in enumerate(row) if cell_text(v) in want}
    if set(got) != set(want.values()):
        raise SystemExit(f"identifier header not found: {got}")
    return got


def blocks(rows, ids, first_data: int, sheet: str):
    """Walk the data rows: yields ('type', key, type, row) for PF/PJ rows, ('total', key, None, row) for
    an institution's Total row and ('grand', None, None, row) for the grand total. key = (cnpj8, name,
    segment, cnpj_missing)."""
    key = None
    for i, r in enumerate(rows[first_data:], start=first_data + 1):
        cnpj, name = cell_text(r[ids["cnpj"]]), cell_text(r[ids["name"]])
        seg, typ = cell_text(r[ids["segment"]]), cell_text(r[ids["type"]])
        if not any((cnpj, name, seg, typ)):
            if all(v is None for v in r):
                continue                                      # a fully empty row
            yield "grand", None, None, r, i
            continue
        if cnpj or name:
            if cnpj and not re.fullmatch(r"\d{8}", cnpj):
                raise SystemExit(f"{sheet} row {i}: CNPJ {cnpj!r} is not 8 digits")
            key = (cnpj, name, seg, 0 if cnpj else 1)
        if key is None:
            raise SystemExit(f"{sheet} row {i}: data before any institution")
        if seg == "Total":
            yield "total", key, None, r, i
        elif typ in ("PF", "PJ"):
            yield "type", key, typ, r, i
        else:
            raise SystemExit(f"{sheet} row {i}: unrecognised row (segment {seg!r}, type {typ!r})")


def main():
    import hashlib
    sha = hashlib.sha256(XLSX.read_bytes()).hexdigest()
    if sha != XLSX_SHA256:
        raise SystemExit(f"{XLSX.name}: sha256 {sha} is not the delivered file's")
    wb = openpyxl.load_workbook(XLSX, read_only=True, data_only=True)
    checks, fails = {}, []

    def check(name, got, want, tol=0):
        ok = abs(got - want) <= tol if isinstance(want, (int, float)) else got == want
        checks[name] = {"got": got, "want": want, "ok": bool(ok)}
        if not ok:
            fails.append(name)

    # ── Acumulado: the three measures ─────────────────────────────────────────────────────────────
    acc = sheet_rows(wb["Acumulado"])
    ids = id_columns(acc[2])
    month_of, measure_of, cur = {}, {}, None
    for j, v in enumerate(acc[0]):
        t = cell_text(v)
        if MONTH_RE.match(t):
            cur = ym(t)
        if cur is not None and cell_text(acc[1][j]) in MEASURES:
            month_of[j], measure_of[j] = cur, MEASURES[cell_text(acc[1][j])]
    months = sorted(set(month_of.values()))
    check("months", len(months), EXPECT["months"])
    for m in months:
        got = sorted(measure_of[j] for j in month_of if month_of[j] == m)
        if got != sorted(MEASURES.values()):
            raise SystemExit(f"Acumulado {m}: measures {got}")
    order = [measure_of[j] for j in sorted(month_of)][:3]
    checks["acumulado_measure_order"] = order

    data = {}                                          # (key, type) -> {(ym, measure): count}
    grand_acc = None
    blk_totals = {}
    for kind, key, typ, r, i in blocks(acc, ids, 3, "Acumulado"):
        vals = {(month_of[j], measure_of[j]): cell_num(r[j], f"Acumulado row {i} col {j + 1}")
                for j in month_of}
        if kind == "type":
            if (key, typ) in data:
                raise SystemExit(f"Acumulado row {i}: second {typ} row for {key}")
            data[(key, typ)] = vals
        elif kind == "total":
            blk_totals[key] = vals
        else:
            grand_acc = vals
    if grand_acc is None:
        raise SystemExit("Acumulado: no grand-total row")
    keys_acc = {k for k, _ in data}
    check("acum_blocks_with_cnpj", sum(1 for k in keys_acc if not k[3]), EXPECT["acum_blocks_with_cnpj"])
    check("acum_blocks_missing_cnpj", sum(1 for k in keys_acc if k[3]), EXPECT["acum_blocks_missing_cnpj"])
    check("acum_rows", len(data), EXPECT["acum_rows"])
    check("acum_pf", sum(1 for _, t in data if t == "PF"), EXPECT["acum_pf"])
    check("acum_pj", sum(1 for _, t in data if t == "PJ"), EXPECT["acum_pj"])
    # each institution's Total row is the sum of its type rows
    bad_blk = [k for k, tot in blk_totals.items()
               if any(tot[c] != sum(data[(k, t)][c] for t in ("PF", "PJ") if (k, t) in data) for c in tot)]
    check("acum_block_totals_mismatch", len(bad_blk), 0)
    # the grand total row is the sum of every type row, in all 108 x 3 cells
    col_sum = defaultdict(int)
    for vals in data.values():
        for c, x in vals.items():
            col_sum[c] += x
    check("acum_grand_cells_mismatch", sum(1 for c in grand_acc if grand_acc[c] != col_sum[c]), 0)
    check("acum_grand_cells", len(grand_acc), 3 * EXPECT["months"])

    # ── Atual: opened and still active, with the BCB's Total column ───────────────────────────────
    atu = sheet_rows(wb["Atual"])
    ids_a = id_columns(atu[3])
    mcol = {j: ym(cell_text(v)) for j, v in enumerate(atu[2]) if MONTH_RE.match(cell_text(v))}
    tcol = [j for j, v in enumerate(atu[2]) if cell_text(v) == "Total"]
    if len(tcol) != 1:
        raise SystemExit(f"Atual: Total column {tcol}")
    check("atual_months", len(mcol), EXPECT["months"])
    atual, n_rowsum_bad = {}, 0
    for kind, key, typ, r, i in blocks(atu, ids_a, 4, "Atual"):
        if kind != "type":
            continue
        vals = {mcol[j]: cell_num(r[j], f"Atual row {i} col {j + 1}") for j in mcol}
        atual[(key, typ)] = vals
        n_rowsum_bad += sum(vals.values()) != cell_num(r[tcol[0]], f"Atual row {i} Total")
    keys_atu = {k for k, _ in atual}
    check("atual_blocks", len(keys_atu), EXPECT["atual_blocks"])
    check("atual_rows", len(atual), EXPECT["atual_rows"])
    check("atual_pf", sum(1 for _, t in atual if t == "PF"), EXPECT["atual_pf"])
    check("atual_pj", sum(1 for _, t in atual if t == "PJ"), EXPECT["atual_pj"])
    check("atual_rowsum_mismatch", int(n_rowsum_bad), 0)
    # Atual and Acumulado name their institutions alike: match on CNPJ and type.
    by_cnpj_acc = {(k[0], t): v for (k, t), v in data.items() if k[0]}
    by_cnpj_atu = {(k[0], t): v for (k, t), v in atual.items() if k[0]}
    check("atual_not_in_acum", len(set(by_cnpj_atu) - set(by_cnpj_acc)), 0)
    cells = mism = 0
    for kt, vals in by_cnpj_acc.items():
        a = by_cnpj_atu.get(kt, {})
        for m in months:
            cells += 1
            mism += vals[(m, "opened_active")] != a.get(m, 0)
    check("equality_cells", cells, EXPECT["equality_cells"])
    check("equality_mismatch", int(mism), 0)
    only = {k[0] for k in keys_acc if k[0]} - {k[0] for k in keys_atu}
    check("only_in_acum", len(only), EXPECT["only_in_acum"])
    check("only_in_acum_with_opened_active", sum(
        1 for (k, t), v in data.items() if k[0] in only and any(v[(m, "opened_active")] for m in months)), 0)

    # ── Tidy file ─────────────────────────────────────────────────────────────────────────────────
    recs = []
    for (k, t), vals in data.items():
        for m in months:
            oa, ocl, cl = vals[(m, "opened_active")], vals[(m, "opened_closed_later")], vals[(m, "closed")]
            recs.append((k[0], k[1], k[2], t, m, oa, ocl, oa + ocl, cl, k[3]))
    df = pd.DataFrame(recs, columns=["cnpj8", "name", "segment", "type", "ym", "opened_active",
                                     "opened_closed_later", "opened_gross", "closed", "cnpj_missing"])
    df = df.sort_values(["cnpj8", "cnpj_missing", "name", "type", "ym"]).reset_index(drop=True)
    check("grand_opened_active", int(df.opened_active.sum()), EXPECT["grand_opened_active"])
    check("grand_opened_closed_later_m", round(df.opened_closed_later.sum() / 1e6, 3),
          EXPECT["grand_opened_closed_later_m"], tol=0.0005)
    check("grand_closed_m", round(df.closed.sum() / 1e6, 3), EXPECT["grand_closed_m"], tol=0.0005)
    df["year"] = df.ym.str[:4].astype(int)
    nat = df.groupby(["type", "year"])[["opened_gross", "opened_active", "opened_closed_later", "closed"]].sum() / 1e6
    nat_bad = []
    for (t, y), want in NATIONAL_M.items():
        got = tuple(round(float(x), 2) for x in nat.loc[(t, y)])
        if any(abs(g - w) > 0.0051 for g, w in zip(got, want)):
            nat_bad.append({"type": t, "year": y, "got": got, "want": want})
    check("national_by_year_mismatch", len(nat_bad), 0)
    if nat_bad:
        checks["national_by_year_detail"] = nat_bad
    df = df.drop(columns="year")

    for name, c in checks.items():
        if isinstance(c, dict) and "ok" in c:
            print(f"  {'ok  ' if c['ok'] else 'FAIL'} {name:36s} {c['got']!s:>14s}  (want {c['want']})")
    print(f"  Acumulado measure order: {checks['acumulado_measure_order']}")
    if fails:
        raise SystemExit(f"{len(fails)} acceptance check(s) failed: {fails}. Nothing written.")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    CHECKS.write_text(json.dumps({"xlsx_sha256": sha, "rows": len(df), "checks": checks}, indent=2,
                                 ensure_ascii=False), encoding="utf-8")
    print(f"wrote {OUT} ({len(df):,} rows) and {CHECKS.name}")


if __name__ == "__main__":
    main()
