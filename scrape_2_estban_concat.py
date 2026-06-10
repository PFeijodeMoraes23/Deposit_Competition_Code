# scrape_2_estban_concat.py
# Last edited: 2026-06-02
# -----------------------------------------------------------------------------
# Concatenate the raw monthly ESTBAN municipality files into the processed
# BCB/ESTBAN/ESTBAN.csv that the rest of the pipeline consumes
# (panel_1_deposits.py, panel_5_flag_digital.py, scrape_9_bcb_inclusion.py).
#
# This is a faithful Python port of the now-DEPRECATED ESTBAN_Process_1.R.
# It reuses panel_1_deposits.process_raw_estban_csv() for the per-file
# VERBETE->V rename / V114 derivation (verified byte-identical to the old
# R-built ESTBAN.csv on the deposit columns), then:
#   * renames the four compound columns (V141_142, V144_..., V441_442, V444_...)
#     to their clean codes so the CSV is well-formed (the raw headers contain
#     embedded tabs/commas);
#   * applies the same NOME_INSTITUICAO cleaning the R script did;
#   * formats CNPJ (8-digit) / CODMUN (5-digit) zero-padded like the original;
#   * writes the canonical column order, including DATA_BASE.
#
# Year coverage: 2016..present (matches the old file's lower bound; panel_1
# still reads pre-2016 raw files directly via load_estban_raw_pre2016()).
# As new YYYYMM_ESTBAN.CSV files are downloaded by scrape_1, this step extends
# the panel automatically -- no constants to bump.
#
# Usage:
#     python scrape_2_estban_concat.py            # rebuild ESTBAN.csv
#     python scrape_2_estban_concat.py --verify   # rebuild + compare vs .bak
# -----------------------------------------------------------------------------
import argparse
import glob
import logging
import os
import re
import shutil
import sys

import numpy as np
import pandas as pd

try:
    from utils.venv_guard import ensure_project_venv
except Exception:
    ensure_project_venv = None
if ensure_project_venv is not None:
    ensure_project_venv(__file__)

# Reuse the proven per-file transform from the deposits builder.
from panel_1_deposits import process_raw_estban_csv, coerce_cnpj
from utils import paths

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_MUN    = str(paths.ESTBAN_RAW_MUN)
OUT_CSV    = str(paths.ESTBAN_CSV)
BAK_CSV    = OUT_CSV + ".bak"

# Lower bound: the old R file started in 2016; panel_1 handles < 2016 from raw.
START_YEAR = 2016

# Canonical column order of the original (R-built) ESTBAN.csv.
CANONICAL_COLS = [
    "UF", "CODMUN", "MUNICIPIO", "CNPJ", "NOME_INSTITUICAO",
    "AGEN_ESPERADAS", "AGEN_PROCESSADAS",
    "V110", "V111", "V112", "V113", "V120", "V130", "V140",
    "V141_142", "V144_145_146_147_152", "V158", "V160", "V161", "V162",
    "V163", "V169", "V171", "V172", "V174", "V176", "V180", "V184", "V190",
    "V200", "V399", "V400_401", "V420", "V430", "V431", "V432", "V433",
    "V440", "V441_142_PLACEHOLDER",  # replaced below
    "V444_445_446_447_456_458", "V460", "V470", "V480", "V490_500",
    "V610", "V710", "V711", "V712", "V899",
    "CODMUN_IBGE", "YEAR", "MONTH", "DATA_BASE", "V114",
]
# Fix the placeholder (kept the list readable above)
CANONICAL_COLS[CANONICAL_COLS.index("V441_142_PLACEHOLDER")] = "V441_442"

# Raw VERBETE columns that process_raw_estban_csv leaves un-renamed
# (substring of the raw VERBETE name -> clean canonical code).
#   * The four compound columns (multiple verbetes summed into one).
#   * V433: panel_1's rename dict keys it as "...CAPTACOES" but BCB spells the
#     raw column "...CAPTACEOS" (their typo), so the substring match misses it.
#     Mapped here so the column is not silently dropped (the old file had it).
_COMPOUND_RENAME = {
    "VERBETE_141": "V141_142",
    "VERBETE_144": "V144_145_146_147_152",
    "VERBETE_441": "V441_442",
    "VERBETE_444": "V444_445_446_447_456_458",
    "VERBETE_433": "V433",
}

# Integer-valued columns (written without a trailing ".0").
_INT_COLS = ["AGEN_ESPERADAS", "AGEN_PROCESSADAS", "CODMUN_IBGE"] + [
    c for c in CANONICAL_COLS if re.match(r"^V\d", c)
]

# NOME_INSTITUICAO fixes, ported verbatim from ESTBAN_Process_1.R.
# Keys are built from latin-1 bytes so they match exactly what pandas reads
# from the latin-1-encoded raw files.
_NAME_FIXES = {
    b"ITA\xda UNIBANCO S.A.".decode("latin1"): "ITAU UNIBANCO SA",
    b"BCO SANTANDER (BRASIL) S.A.".decode("latin1"): "BCO SANTANDER BRASIL SA",
    b"SOCIAL BANK s/A".decode("latin1"): "SOCIAL BANK SA",
    b"BCO ITA\xda BBA S.A.".decode("latin1"): "BCO ITAU BBA SA",
    b"BANCO TOP\xc1ZIO S.A.".decode("latin1"): "BANCO TOPAZIO SA",
    b"NOVO BCO CONTINENTAL S.A. - BM".decode("latin1"): "NOVO BCO CONTINENTAL SA BM",
    b"BCO DO EST. DE SE S.A.".decode("latin1"): "BCO DO EST DE SE SA",
    b"BCO BRASILEIRO DE CR\xc9DITO S.A.".decode("latin1"): "BCO BRASILEIRO DE CREDITO SA",
    b"BANCO MASTER M\xdaLTIPLO".decode("latin1"): "BANCO MASTER MULTIPLO",
    b"BANCO ITA\xda CONSIGNADO S.A.".decode("latin1"): "BANCO ITAU CONSIGNADO SA",
    b"DEUTSCHE BANK S.A.BCO ALEMAO".decode("latin1"): "DEUTSCHE BANK SA BCO ALEMAO",
    b"BCO CR\xc9DIT AGRICOLE BR SA".decode("latin1"): "BCO CREDIT AGRICOLE BR SA",
    b"BCO CR\xe9DIT AGRICOLE BR SA".decode("latin1"): "BCO CREDIT AGRICOLE BR SA",
    b"BCO CR\xc8DIT AGRICOLE BR SA".decode("latin1"): "BCO CREDIT AGRICOLE BR SA",
    b"BCO CR\xe8DIT AGRICOLE BR SA".decode("latin1"): "BCO CREDIT AGRICOLE BR SA",
    b"BCO CREDIT SUISSE (BRL) S.A.".decode("latin1"): "BCO CREDIT SUISSE BRL SA",
    b"BCO M\xc1XIMA S.A.".decode("latin1"): "BCO MAXIMA SA",
    b"BCO ORIGINAL DO AGRO S/A".decode("latin1"): "BCO ORIGINAL DO AGRO SA",
    b"ITA\xda UNIBANCO HOLDING S.A.".decode("latin1"): "ITAU UNIBANCO HOLDING SA",
    b"COMMERZBANK BRASIL S.A. - BCO M\xdaLTIPLO".decode("latin1"): "COMMERZBANK BRASIL SA BCO MULTIPLO",
    b"BROU - BRASIL ADMINISTRA\xc7\xc3O DE BENS PR\xd3PRIOS LTDA.".decode("latin1"): "BROU BRASIL ADMINISTRACAO DE BENS PROPRIOS LTDA",
    b"BCV - BCO, CR\xc9DITO E VAREJO S.A.".decode("latin1"): "BCV BCO CREDITO E VAREJO SA",
    b"ING ADMINISTRA\xc7\xc3O S.A.".decode("latin1"): "ING ADMINISTRACAO SA",
    b"BANIF - BCO INTERNACIONAL DO FUNCHAL (BRASIL), S.A.".decode("latin1"): "BANIF BCO INTERNACIONAL DO FUNCHAL BRASIL SA",
    b"BRB - BCO DE BRASILIA S.A.".decode("latin1"): "BRB BCO DE BRASILIA SA",
    b"PICPAY BANK - BANCO M\xdaLTIPLO S.A.".decode("latin1"): "PICPAY BANK BANCO MULTIPLO SA",
    b"ING BANK N.V.".decode("latin1"): "ING BANK NV",
}


def _clean_name(s):
    if not isinstance(s, str):
        return s
    s = _NAME_FIXES.get(s, s)
    return s.replace(".", "")


def _list_raw_files() -> list[str]:
    files = glob.glob(os.path.join(RAW_MUN, "*_ESTBAN.CSV")) + \
            glob.glob(os.path.join(RAW_MUN, "*_ESTBAN.csv"))
    keep = []
    for fp in sorted(set(files)):
        m = re.match(r"^(\d{4})(\d{2})_ESTBAN", os.path.basename(fp), re.IGNORECASE)
        if m and int(m.group(1)) >= START_YEAR:
            keep.append(fp)
    return keep


def _finalize(df: pd.DataFrame) -> pd.DataFrame:
    """Rename compound cols, clean names, format keys, enforce canonical order."""
    # Rename the leftover compound VERBETE columns to clean codes.
    ren = {}
    for col in df.columns:
        up = col.upper()
        for substr, clean in _COMPOUND_RENAME.items():
            if substr in up and clean not in df.columns and clean not in ren.values():
                ren[col] = clean
                break
    if ren:
        df = df.rename(columns=ren)

    # Clean institution names (R parity) and zero-pad code columns.
    if "NOME_INSTITUICAO" in df.columns:
        df["NOME_INSTITUICAO"] = df["NOME_INSTITUICAO"].map(_clean_name)
    if "CNPJ" in df.columns:
        cnpj = coerce_cnpj(df["CNPJ"])
        df["CNPJ"] = cnpj.map(lambda v: f"{int(v):08d}" if pd.notna(v) else "")
    if "CODMUN" in df.columns:
        cod = pd.to_numeric(df["CODMUN"], errors="coerce").astype("Int64")
        df["CODMUN"] = cod.map(lambda v: f"{int(v):05d}" if pd.notna(v) else "")

    # DATA_BASE = first day of the reference month.
    df["DATA_BASE"] = pd.to_datetime(
        df["YEAR"].astype(int).astype(str) + df["MONTH"].astype(int).map("{:02d}".format) + "01",
        format="%Y%m%d", errors="coerce",
    ).dt.strftime("%Y-%m-%d")

    # Integer columns -> nullable Int64 (writes "157892", not "157892.0").
    for col in _INT_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").round().astype("Int64")

    # Enforce canonical column order; warn on anything missing.
    missing = [c for c in CANONICAL_COLS if c not in df.columns]
    if missing:
        logging.warning(f"Columns absent from raw file, filled NA: {missing}")
        for c in missing:
            df[c] = pd.NA
    return df[CANONICAL_COLS]


def build() -> pd.DataFrame:
    files = _list_raw_files()
    if not files:
        logging.error(f"No raw ESTBAN files (>= {START_YEAR}) found in {RAW_MUN}")
        sys.exit(1)
    logging.info(f"Concatenating {len(files)} raw ESTBAN files ({START_YEAR}+) ...")

    frames = []
    for i, fp in enumerate(files, 1):
        df = process_raw_estban_csv(fp)
        if df is None or df.empty:
            logging.warning(f"  skipped (empty/unreadable): {os.path.basename(fp)}")
            continue
        frames.append(_finalize(df))
        if i % 12 == 0 or i == len(files):
            logging.info(f"  processed {i}/{len(files)} files")

    out = pd.concat(frames, ignore_index=True)
    logging.info(
        f"Concatenated: {len(out):,} rows; "
        f"{out['YEAR'].min()}-{out['MONTH'].min():02d} .. "
        f"{out['YEAR'].max()}-{out[out['YEAR'] == out['YEAR'].max()]['MONTH'].max():02d}"
    )
    return out


def main():
    ap = argparse.ArgumentParser(description="Rebuild ESTBAN.csv from raw monthlies.")
    ap.add_argument("--verify", action="store_true",
                    help="After writing, compare deposit-column sums vs ESTBAN.csv.bak.")
    args = ap.parse_args()

    # Preserve the original R-built file once.
    if os.path.exists(OUT_CSV) and not os.path.exists(BAK_CSV):
        logging.info(f"Backing up existing ESTBAN.csv -> {os.path.basename(BAK_CSV)}")
        shutil.copy2(OUT_CSV, BAK_CSV)

    out = build()
    logging.info(f"Writing {OUT_CSV} ...")
    out.to_csv(OUT_CSV, index=False, encoding="latin-1")
    logging.info(f"Done. {len(out):,} rows, {len(out.columns)} columns.")

    if args.verify and os.path.exists(BAK_CSV):
        _verify(OUT_CSV, BAK_CSV)


def _verify(new_path: str, old_path: str):
    """Compare per-(YEAR,MONTH) deposit-column national sums on the overlap."""
    dep = ["V400_401", "V420", "V431", "V432"]
    usecols = ["YEAR", "MONTH"] + dep
    logging.info("VERIFY: loading old & new (deposit columns only) ...")
    new = pd.read_csv(new_path, encoding="latin1", usecols=usecols, low_memory=False)
    old = pd.read_csv(old_path, encoding="latin1", usecols=usecols, low_memory=False)
    for c in dep:
        new[c] = pd.to_numeric(new[c], errors="coerce")
        old[c] = pd.to_numeric(old[c], errors="coerce")
    gn = new.groupby(["YEAR", "MONTH"])[dep].sum()
    go = old.groupby(["YEAR", "MONTH"])[dep].sum()
    common = gn.index.intersection(go.index)
    diff = (gn.loc[common] - go.loc[common]).abs()
    maxdiff = diff.to_numpy().max() if len(common) else float("nan")
    print(f"\nVERIFY: {len(common)} overlapping (YEAR,MONTH) cells; "
          f"max abs deposit-sum diff = {maxdiff:,.2f}")
    print(f"VERIFY: new-only periods = {sorted(set(map(tuple, gn.index.difference(go.index))))}")
    if maxdiff == 0:
        print("VERIFY: PASS - overlap is byte-identical on deposit columns.")
    else:
        print("VERIFY: WARNING - differences found; inspect before trusting.")


if __name__ == "__main__":
    main()
