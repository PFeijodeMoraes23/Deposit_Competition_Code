###─────────────────────────────────────────────────────────────────────────────
# fees/scrape_bcb_tarifas_monthly.py
#
# Author: Pedro Feijó de Moraes
#
# PURPOSE
#   Download the BCB's monthly census of listed bank tariffs ("Instituições e suas respectivas
#   tarifas bancárias", file <YYYYMM>TARINST.ZIP), and parse it into a dated listed-price
#   history: one row per institution x customer type x service x date in force x maximum price.
#
#   Each monthly file lists, for every institution that reported to the BCB, every service with
#   its "Vigência" (date the price took effect) and "Valor Máximo" (listed ceiling), for
#   individuals (Pessoa F) and firms (Pessoa J). The account-keeping services are
#     F 15.01-15.04  Pacote Padronizado de Serviços I-IV   (Olinda codes 1501-1504)
#     J 04.02/04.03  Manutenção de conta ativa / inativa   (Olinda codes 0402/0403)
#   The Olinda API (fees/scrape_bcb_tariff_vigencia.py) serves only the current schedule; these
#   files are the history the BCB published before it moved the tariff pages to Olinda.
#
#   Not a pipeline step, and no pipeline step reads its output. Listed prices are ceilings.
#
# DATA SOURCE
#   listing  the old download page www.bcb.gov.br/fis/tarifas/htms/tarifdwl.asp, as captured by
#            the Internet Archive (raw "id_" capture; default 2018-07-13, the last full listing:
#            months 2002-06 to 2018-07). The page is gone from the live site. File names are
#            taken from its <option value=...> list exactly as served, never built, and their
#            case varies by month (.ZIP / .zip).
#   files    https://www.bcb.gov.br/fis/tarifas/ftp/<name>, still served by the BCB.
#   A download is accepted only if it is a valid zip, its size in KiB rounds to the size the
#   listing states (within 1 KiB), and the CSV inside starts with the census title line. The
#   month the listing was captured in (2018-07) was regenerated at month-end, after the
#   capture; its file is accepted when the census "Data:" date falls in that month, and the
#   manifest marks it saved_month_end_version.
#   Months after the listing (--unlisted, Pedro's OK of 2026-10-23) use the listed name
#   pattern <YYYYMM>TARINST.ZIP; they are accepted by the same census-date test and marked
#   saved_not_in_listing.
#
# OUTPUT
#   raw (default utils.raw_paths.BCB_TARIFAS_MONTHLY, or --out)
#     listing/tarifdwl_<capture>.html   the archived listing page, as served
#     zip/<name>                        one zip per month, as served
#     manifest.csv                      yyyymm, kind, name, url, accessed, bytes, listed_kib,
#                                       last_modified, sha256, csv_name, csv_date, status
#   processed (utils.paths.TARIFAS_PROC)
#     tariff_history_bcb_monthly.csv    cnpj8, pessoa, servico, codigo_servico, descricao,
#                                       vigencia, valor_maximo, moeda, unidade, periodicidade,
#                                       instituicao, first_file, last_file, n_files,
#                                       first_file_date, last_file_date
#     One row per distinct (cnpj8, pessoa, servico, vigencia, valor_maximo, unidade,
#     periodicidade), with the first and last monthly file in which it is listed.
#     codigo_servico is the 4-digit Olinda code (15.01 -> 1501).
#     tariff_census_bcb_monthly.csv     yyyymm, file_date, pessoa, n_institutions, n_rows
#   A month already on disk as a valid zip is not downloaded again unless --force.
#
# USAGE
#   python fees/scrape_bcb_tarifas_monthly.py                    # 2015-01 to 2018-07 (listed)
#   python fees/scrape_bcb_tarifas_monthly.py --end 202412 --unlisted 201808 ... 202412
#   python fees/scrape_bcb_tarifas_monthly.py --end 202412 --parse-only
#   python fees/scrape_bcb_tarifas_monthly.py --start 201601 --end 201612 --force
###─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import io
import os
import re
import sys
import time
import zipfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root: utils/

from utils import bcb_http
from utils import paths
from utils import raw_paths

WAYBACK = "https://web.archive.org/web"
LISTING_URL = "http://www.bcb.gov.br/fis/tarifas/htms/tarifdwl.asp"
LISTING_CAPTURE = "20180713002915"            # last capture with the full month list
FILE_BASE = "https://www.bcb.gov.br/fis/tarifas/ftp/"
DEFAULT_OUT = raw_paths.BCB_TARIFAS_MONTHLY
PROC_DIR = paths.TARIFAS_PROC
TITLE = "Institui"                            # first line: "Instituições e suas respectivas ..."
PAUSE = 2.0                                   # seconds between downloads

MANIFEST_COLS = ["yyyymm", "kind", "name", "url", "accessed", "bytes", "listed_kib",
                 "last_modified", "sha256", "csv_name", "csv_date", "status"]
SPELL_KEYS = ["cnpj8", "pessoa", "servico", "vigencia", "valor_maximo", "unidade",
              "periodicidade"]


# ---------------------------------------------------------------------------
# Listing: file names and sizes from the archived download page
# ---------------------------------------------------------------------------

def fetch_listing(session, out: Path, capture: str) -> Path:
    """Save the archived listing page once (raw capture, bytes as served) and return its path."""
    target = out / "listing" / f"tarifdwl_{capture}.html"
    if target.exists() and target.stat().st_size > 10_000:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    r = session.get(f"{WAYBACK}/{capture}id_/{LISTING_URL}", timeout=120)
    r.raise_for_status()
    if b"TARINST" not in r.content:
        raise RuntimeError(f"capture {capture} of tarifdwl.asp has no TARINST options")
    target.write_bytes(r.content)
    return target


def parse_listing(page: Path) -> list[dict]:
    """Every <option value=YYYYMMKIND.ZIP>Month/Year (N KB)</option> on the page.

    The page is windows-1252; sizes use a no-break space as the thousands separator."""
    html = page.read_bytes().decode("cp1252", errors="replace")
    items = []
    for value, label in re.findall(r'<option\s+value="?([^\s">]+)"?\s*>([^<]*)</option>',
                                   html, re.IGNORECASE):
        m = re.fullmatch(r"(\d{6})(TAR[A-Z]+)\.zip", value, re.IGNORECASE)
        if not m:
            continue
        kib = re.search(r"\(([\d\s\xa0.]+)\s*KB\)", label)
        items.append({"yyyymm": m.group(1), "kind": m.group(2).upper(), "name": value,
                      "listed_kib": int(re.sub(r"\D", "", kib.group(1))) if kib else None})
    return items


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def csv_header(path: Path) -> tuple[str, str]:
    """(name of the CSV inside the zip, its "Data:" line), or ("", "") if not a census file."""
    try:
        with zipfile.ZipFile(path) as z:
            name = z.namelist()[0]
            with z.open(name) as fh:
                head = fh.read(600).decode("cp1252", errors="replace").splitlines()
    except (zipfile.BadZipFile, IndexError, OSError):
        return "", ""
    if not head or not head[0].startswith(TITLE):
        return "", ""
    date = next((ln.split(":", 1)[1].strip() for ln in head[:4] if ln.startswith("Data:")), "")
    return name, date


def valid(path: Path, item: dict) -> bool:
    """The file on disk is the one the listing describes: a zip holding a census CSV whose
    size in KiB rounds to the listed size (within 1 KiB).

    The month in which the listing was captured is the exception: its file was regenerated at
    month-end after the capture, so its size cannot match. For that month the census's own
    "Data:" date must fall in the named month instead."""
    if not path.exists() or not zipfile.is_zipfile(path):
        return False
    name, date = csv_header(path)
    if not name:
        return False
    if not item.get("size_checked", True):
        return date[3:5] + date[6:10] == item["yyyymm"][4:] + item["yyyymm"][:4]
    return item["listed_kib"] is None or abs(path.stat().st_size / 1024 - item["listed_kib"]) <= 1


def download(items: list[dict], out: Path, force: bool) -> list[dict]:
    session = bcb_http.retry_session(total=4, backoff=3.0)
    today = dt.date.today().isoformat()
    (out / "zip").mkdir(parents=True, exist_ok=True)
    rows = []
    for it in items:
        target = out / "zip" / it["name"]
        url = FILE_BASE + it["name"]
        status, last_mod = "present", ""
        if force or not valid(target, it):
            status = "FAILED"
            for attempt in range(3):
                r = session.get(url, timeout=300)
                if r.status_code == 200 and len(r.content) > 1000:
                    tmp = target.with_name(target.name + ".part")
                    tmp.write_bytes(r.content)
                    if valid(tmp, it):
                        os.replace(tmp, target)
                        status, last_mod = "saved", r.headers.get("Last-Modified", "")
                        break
                    tmp.unlink()
                time.sleep(10 * (attempt + 1))
            time.sleep(PAUSE)
        size = target.stat().st_size if target.exists() else 0
        sha, name, date = "", "", ""
        if status != "FAILED":
            sha = hashlib.sha256(target.read_bytes()).hexdigest()
            name, date = csv_header(target)
        if status == "saved" and it.get("unlisted"):
            status = "saved_not_in_listing"
        elif status == "saved" and not it.get("size_checked", True):
            status = "saved_month_end_version"
        rows.append({"yyyymm": it["yyyymm"], "kind": it["kind"], "name": it["name"], "url": url,
                     "accessed": today, "bytes": size, "listed_kib": it["listed_kib"],
                     "last_modified": last_mod, "sha256": sha, "csv_name": name,
                     "csv_date": date, "status": status})
        print(f"  {it['yyyymm']}  {it['name']:20s} {status:7s} {size:>10,d} B  {date}")
    return rows


def write_manifest(rows: list[dict], out: Path) -> None:
    """Merge into manifest.csv: rows of this run replace earlier rows for the same file, and
    a present file keeps the Last-Modified recorded when it was saved."""
    path = out / "manifest.csv"
    old = {}
    if path.exists():
        with open(path, encoding="utf-8", newline="") as f:
            old = {r["name"]: r for r in csv.DictReader(f)}
    for r in rows:
        prev = old.get(r["name"])
        if prev and r["status"] == "present" and not r["last_modified"]:
            r["last_modified"], r["accessed"] = prev["last_modified"], prev["accessed"]
        old[r["name"]] = r
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_COLS)
        w.writeheader()
        w.writerows(sorted(old.values(), key=lambda r: (r["kind"], r["yyyymm"])))


# ---------------------------------------------------------------------------
# Parse
# ---------------------------------------------------------------------------

def read_census(path: Path) -> tuple[pd.DataFrame, str]:
    """One monthly TARINST file as a DataFrame with stripped text columns, and its "Data:" date.

    Layout: title, "Data:dd/mm/yyyy", legend, source, then a ';' header row
    (CNPJ; Instituição; Serviço; Descrição do Serviço; Vigência; Valor Máximo; Moeda; Pessoa;
    Unidade; Periodicidade). Fields are fixed-width padded; values use the Brazilian format."""
    with zipfile.ZipFile(path) as z:
        raw = z.read(z.namelist()[0]).decode("cp1252")
    lines = raw.splitlines()
    head = next(i for i, ln in enumerate(lines[:12]) if ln.startswith("CNPJ"))
    date = next((ln.split(":", 1)[1].strip() for ln in lines[:head] if ln.startswith("Data:")), "")
    cols = [_fold(c) for c in lines[head].split(";")]
    if cols != CENSUS_COLS:
        raise ValueError(f"{path.name}: unexpected header {cols}")
    df = pd.read_csv(io.StringIO("\n".join(lines[head + 1:])), sep=";", header=None,
                     names=OUT_COLS, dtype=str, keep_default_na=False)
    for c in OUT_COLS:
        df[c] = df[c].str.strip()
    return df, date


def _fold(s: str) -> str:
    """Header cell without padding, accents or case: "Valor Máximo " -> "valor maximo"."""
    table = str.maketrans("áàâãéêíóôõúüçÁÀÂÃÉÊÍÓÔÕÚÜÇ", "aaaaeeiooouucAAAAEEIOOOUUC")
    return s.strip().translate(table).lower()


CENSUS_COLS = ["cnpj", "instituicao", "servico", "descricao do servico", "vigencia",
               "valor maximo", "moeda", "pessoa", "unidade", "periodicidade"]
OUT_COLS = ["cnpj8", "instituicao", "servico", "descricao", "vigencia", "valor_maximo",
            "moeda", "pessoa", "unidade", "periodicidade"]


def build_history(out: Path, start: str, end: str) -> None:
    """Stack the TARINST files of start..end and collapse them into price spells.

    A spell is a distinct (institution, customer type, service, date in force, maximum price,
    unit, periodicity); first_file / last_file are the first and last months that list it.
    Months are the file's yyyymm; first_file_date / last_file_date are the files' own
    "Data:" dates (the day the BCB generated the census)."""
    files = sorted(p for p in (out / "zip").glob("*")
                   if re.fullmatch(r"\d{6}TARINST\.zip", p.name, re.IGNORECASE)
                   and start <= p.name[:6] <= end)
    if not files:
        print("no TARINST files in the window")
        return
    # Months are folded in one at a time (about 87,000 rows each), so memory stays at the size
    # of the spell table rather than the stacked months. Files are visited in date order, so
    # "last" keeps the latest institution name and service wording.
    spells, census, n_rows, bad_v, bad_p = None, [], 0, 0, 0
    for p in files:
        df, date = read_census(p)
        yyyymm = p.name[:6]
        for pessoa, g in df.groupby("pessoa"):
            census.append({"yyyymm": yyyymm, "file_date": date, "pessoa": pessoa,
                           "n_institutions": g["cnpj8"].nunique(), "n_rows": len(g)})
        v = df["valor_maximo"].str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
        df["valor_maximo"] = pd.to_numeric(v, errors="coerce")
        df["vigencia"] = pd.to_datetime(df["vigencia"], format="%d/%m/%Y", errors="coerce")
        bad_v += int(df["vigencia"].isna().sum())
        bad_p += int(df["valor_maximo"].isna().sum())
        n_rows += len(df)
        fdate = pd.to_datetime(date, format="%d/%m/%Y", errors="coerce")
        df = df.assign(first_file=yyyymm, last_file=yyyymm, n_files=1,
                       first_file_date=fdate, last_file_date=fdate)
        df = df.drop_duplicates(SPELL_KEYS)
        both = df if spells is None else pd.concat([spells, df], ignore_index=True)
        spells = (both.groupby(SPELL_KEYS, dropna=False, sort=False)
                  .agg(descricao=("descricao", "last"), moeda=("moeda", "last"),
                       instituicao=("instituicao", "last"),
                       first_file=("first_file", "min"), last_file=("last_file", "max"),
                       n_files=("n_files", "sum"),
                       first_file_date=("first_file_date", "min"),
                       last_file_date=("last_file_date", "max"))
                  .reset_index())
        print(f"  parsed {p.name}: {len(df):,} rows, {df['cnpj8'].nunique():,} institutions "
              f"(Data: {date}); {len(spells):,} spells so far")

    spells["codigo_servico"] = spells["servico"].str.replace(".", "", regex=False).str.zfill(4)
    spells = spells.sort_values(["cnpj8", "pessoa", "servico", "first_file", "vigencia"])
    spells = spells[["cnpj8", "pessoa", "servico", "codigo_servico", "descricao", "vigencia",
                     "valor_maximo", "moeda", "unidade", "periodicidade", "instituicao",
                     "first_file", "last_file", "n_files", "first_file_date", "last_file_date"]]
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    spells.to_csv(PROC_DIR / "tariff_history_bcb_monthly.csv", index=False, date_format="%Y-%m-%d")
    pd.DataFrame(census).to_csv(PROC_DIR / "tariff_census_bcb_monthly.csv", index=False)

    print(f"{len(files)} files, {n_rows:,} listed rows -> {len(spells):,} spells "
          f"({spells['cnpj8'].nunique():,} institutions); unparsed vigencia {bad_v:,}, "
          f"unparsed valor {bad_p:,}")
    print(f"  -> {PROC_DIR / 'tariff_history_bcb_monthly.csv'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="BCB monthly tariff census (TARINST), raw + history")
    ap.add_argument("--start", default="201501", help="first YYYYMM (default 201501)")
    ap.add_argument("--end", default="999912", help="last YYYYMM (default: latest listed)")
    ap.add_argument("--kinds", nargs="+", default=["TARINST"],
                    help="file kinds to download: TARINST (by institution, default), TARSERV "
                         "(same data by service), TARCONS (min/mean/max by segment)")
    ap.add_argument("--capture", default=LISTING_CAPTURE,
                    help="Internet Archive timestamp of the listing page")
    ap.add_argument("--unlisted", nargs="+", default=[], metavar="YYYYMM",
                    help="months after the archived listing, fetched as <YYYYMM>TARINST.ZIP; their "
                         "names follow the listed pattern but appear in no archived listing")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--force", action="store_true", help="re-download months already present")
    ap.add_argument("--parse-only", action="store_true", help="skip downloads; rebuild history")
    ap.add_argument("--download-only", action="store_true", help="skip the history rebuild")
    args = ap.parse_args()
    out = Path(args.out)

    if not args.parse_only:
        session = bcb_http.retry_session(total=4, backoff=5.0)
        page = fetch_listing(session, out, args.capture)
        listed = parse_listing(page)
        kinds = {k.upper() for k in args.kinds}
        items = [it for it in listed
                 if it["kind"] in kinds and args.start <= it["yyyymm"] <= args.end]
        # A month listed twice (upper- and lower-case name) keeps the first listing.
        seen, uniq = set(), []
        for it in sorted(items, key=lambda it: (it["kind"], it["yyyymm"])):
            if (it["kind"], it["yyyymm"]) not in seen:
                seen.add((it["kind"], it["yyyymm"]))
                uniq.append(it)
        for it in uniq:
            it["size_checked"] = it["yyyymm"] != args.capture[:6]
        listed_months = {it["yyyymm"] for it in listed if it["kind"] == "TARINST"}
        for m in sorted(set(args.unlisted) - listed_months):
            uniq.append({"yyyymm": m, "kind": "TARINST", "name": f"{m}TARINST.ZIP",
                         "listed_kib": None, "size_checked": False, "unlisted": True})
        print(f"listing {page.name}: {len(listed)} files; {len(uniq)} of kinds {sorted(kinds)} "
              f"in {args.start}..{args.end}")
        rows = download(uniq, out, args.force)
        write_manifest(rows, out)
        bad = [r["name"] for r in rows if r["status"] == "FAILED"]
        print(f"{len(rows) - len(bad)} of {len(rows)} files on disk in {out}"
              + (f"; FAILED: {bad}" if bad else ""))

    if not args.download_only:
        build_history(out, args.start, args.end)


if __name__ == "__main__":
    main()
