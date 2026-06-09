## scrape_3c_pix_roster.py
# Author: Pedro Feijó de Moraes
#
# Last edited: 2026-06-09
#
# Purpose: Roster-based leg of the institution-level Pix entry timeline. Parses
#          the BCB's dated "Lista de participantes do Pix" PDFs to recover, per
#          institution and per roster date:
#             - participation TYPE: OBRIGATÓRIA (mandatory) vs FACULTATIVA (voluntary)
#             - SPI participation:   DIRETA vs INDIRETA
#             - institution type, modalidade, authorisation status
#          Then derives a per-institution first_seen date and reconciles against
#          the key-based entry dates from scrape_3b_pix_participants.py.
#
#          BCB serves only the *current* roster; the history lives on the Wayback
#          Machine across three naming regimes (continuous May 2020 -> present):
#            1. Registration phase  /pix/ListadeparticipantesemprocessodeadesaoaoPIX{DD.MM}.pdf
#            2. Launch era          /pix/ListadeparticipantesdoPix-{DD.MM}.pdf  +
#                                   rolling /pix/ListadeparticipantesdoPix.pdf
#            3. Modern              /participantes_pix_pdf/lista-participantes-...-{YYYYMMDD}.pdf
#          Raw bytes of any capture: http://web.archive.org/web/<timestamp>id_/<url>
#
#   Outputs (BCB/PIX/):
#     1. pix_roster_history.csv     one row per (roster_date × institution)
#     2. pix_roster_entry.csv       one row per institution (first_seen, type history)
#     3. pix_participant_timeline.csv  unified panel: roster classification joined
#                                      to the 3b key-based entry month, keyed on ISPB
#
# Notes:
#   • The 2020-2022 rosters key on CNPJ only; the modern rosters carry BOTH ISPB
#     and CNPJ, so they double as a CNPJ-root -> ISPB crosswalk used to backfill
#     the early rows.
#   • The real roster date is the filename DD.MM / YYYYMMDD or the cover-page
#     "Atualizado em DD/MM/YYYY" — NOT the Wayback capture timestamp.
#   • PDF text has latin-1 mojibake (accented letters -> U+FFFD), but the tokens
#     we classify on (OBRIGAT…, FACULT…, DIRETA, INDIRETA, Provedor, Liquidante)
#     and CNPJ/ISPB are clean ASCII, so a value-based heuristic parser is robust
#     across both layouts. Display names come from the clean Olinda feed (3b).
###─────────────────────────────────────────────────────────────────────────────

import os
import re
import ssl
import time
import json
import logging
import urllib.request

try:
    from utils.venv_guard import ensure_project_venv
except Exception:
    ensure_project_venv = None

if ensure_project_venv is not None:
    ensure_project_venv(__file__)

import pandas as pd
import pdfplumber

try:
    from utils.toon_runtime import resolve_script_paths
except Exception:
    resolve_script_paths = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

## ─────────────────────────────────────────────────────────────────────────────
## 1) PATHS & CONSTANTS
## ─────────────────────────────────────────────────────────────────────────────
BASE          = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
PIX_DIR       = os.path.join(BASE, "BCB", "PIX")
CACHE_DIR     = os.path.join(PIX_DIR, "roster_cache")
HISTORY_CSV   = os.path.join(PIX_DIR, "pix_roster_history.csv")
ROSTER_ENTRY  = os.path.join(PIX_DIR, "pix_roster_entry.csv")
TIMELINE_CSV  = os.path.join(PIX_DIR, "pix_participant_timeline.csv")
KEYS_ENTRY    = os.path.join(PIX_DIR, "pix_participant_entry.csv")  # produced by 3b

if resolve_script_paths is not None:
    _paths = resolve_script_paths(
        "scrape_3c_pix_roster",
        {"pix_dir": PIX_DIR, "cache_dir": CACHE_DIR, "history_csv": HISTORY_CSV,
         "roster_entry": ROSTER_ENTRY, "timeline_csv": TIMELINE_CSV, "keys_entry": KEYS_ENTRY},
        script_dir=os.path.dirname(os.path.abspath(__file__)),
    )
    PIX_DIR, CACHE_DIR   = _paths["pix_dir"], _paths["cache_dir"]
    HISTORY_CSV          = _paths["history_csv"]
    ROSTER_ENTRY         = _paths["roster_entry"]
    TIMELINE_CSV         = _paths["timeline_csv"]
    KEYS_ENTRY           = _paths["keys_entry"]

CDX = "http://web.archive.org/cdx/search/cdx"
WB_RAW = "http://web.archive.org/web/{ts}id_/{url}"
UA = {"User-Agent": "Mozilla/5.0"}

# URL prefixes for the three regimes
PREFIX_MODERN = "bcb.gov.br/content/estabilidadefinanceira/participantes_pix_pdf"
PREFIX_LEGACY = "bcb.gov.br/content/estabilidadefinanceira/pix"

_SSL = ssl.create_default_context()
_SSL.check_hostname = False
_SSL.verify_mode = ssl.CERT_NONE


## ─────────────────────────────────────────────────────────────────────────────
## 2) WAYBACK ENUMERATION
## ─────────────────────────────────────────────────────────────────────────────

def _cdx(params: str) -> list[list[str]]:
    url = f"{CDX}?{params}"
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=90, context=_SSL) as r:
                text = r.read().decode("utf-8", "replace")
            return [ln.split(" ") for ln in text.splitlines() if ln.strip()]
        except Exception as e:
            if attempt == 3:
                logging.error(f"CDX failed: {e}")
                return []
            time.sleep(2)


def enumerate_rosters() -> list[dict]:
    """Return jobs: {url, ts, roster_date|None, regime}. One per distinct roster."""
    jobs: list[dict] = []

    # --- Modern: one job per dated filename, earliest 200 capture ---
    # Do NOT collapse: many URLs' representative capture is a non-200 (404/redirect);
    # we must scan every capture and keep the earliest 200 for each distinct file.
    rows = _cdx(f"url={PREFIX_MODERN}*&output=text&fl=timestamp,statuscode,original"
                f"&limit=40000")
    seen = {}
    for parts in rows:
        if len(parts) != 3:
            continue
        ts, status, original = parts
        if status != "200" or "lista-participantes" not in original.lower():
            continue
        if not re.search(r"(\d{8})\.pdf", original):
            continue
        if original not in seen or ts < seen[original]:
            seen[original] = ts
    for original, ts in seen.items():
        rdate = pd.to_datetime(re.search(r"(\d{8})\.pdf", original).group(1),
                               format="%Y%m%d").date()
        jobs.append({"url": original, "ts": ts, "roster_date": rdate, "regime": "modern"})

    # --- Legacy dated files (/pix/ folder), earliest 200 capture ---
    rows = _cdx(f"url={PREFIX_LEGACY}*&output=text&fl=timestamp,statuscode,original"
                f"&limit=20000&from=2020&to=20230701")
    seen = {}
    for parts in rows:
        if len(parts) != 3:
            continue
        ts, status, original = parts
        base = original.lower()
        if status != "200" or "listadeparticipantes" not in base:
            continue
        if original not in seen or ts < seen[original]:
            seen[original] = ts
    for original, ts in seen.items():
        name = original.rsplit("/", 1)[-1]
        if name.lower() == "listadeparticipantesdopix.pdf":
            continue  # rolling file handled separately (multiple dates)
        rdate = _date_from_legacy_name(name)
        jobs.append({"url": original, "ts": ts, "roster_date": rdate, "regime": "legacy"})

    # --- Rolling file: every distinct-content capture (date read from cover page) ---
    rolling = f"{PREFIX_LEGACY}/ListadeparticipantesdoPix.pdf"
    rows = _cdx(f"url={rolling}&output=text&fl=timestamp,statuscode&collapse=digest&limit=500")
    for parts in rows:
        if len(parts) >= 2 and parts[1] == "200":
            jobs.append({"url": f"https://www.{rolling}", "ts": parts[0],
                         "roster_date": None, "regime": "rolling"})

    logging.info(f"Enumerated {len(jobs)} roster jobs "
                 f"(modern={sum(j['regime']=='modern' for j in jobs)}, "
                 f"legacy={sum(j['regime']=='legacy' for j in jobs)}, "
                 f"rolling={sum(j['regime']=='rolling' for j in jobs)}).")
    return jobs


def _date_from_legacy_name(name: str):
    """Map a 2020 legacy filename (DD.MM or DDMM) to a date; year is always 2020."""
    m = re.search(r"(\d{2})\.(\d{2})", name)
    if not m:
        m = re.search(r"(\d{2})(\d{2})\.pdf", name, re.IGNORECASE)
    if not m:
        return None
    dd, mm = int(m.group(1)), int(m.group(2))
    try:
        return pd.Timestamp(year=2020, month=mm, day=dd).date()
    except ValueError:
        return None


## ─────────────────────────────────────────────────────────────────────────────
## 3) DOWNLOAD (cached)
## ─────────────────────────────────────────────────────────────────────────────

def download_job(job: dict) -> str | None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    fname = f"{job['ts']}_{job['url'].rsplit('/',1)[-1]}"
    path = os.path.join(CACHE_DIR, fname)
    if os.path.exists(path) and os.path.getsize(path) > 5000:
        return path
    raw = WB_RAW.format(ts=job["ts"], url=job["url"])
    for attempt in range(4):
        try:
            req = urllib.request.Request(raw, headers=UA)
            with urllib.request.urlopen(req, timeout=120, context=_SSL) as r:
                data = r.read()
            if len(data) < 5000 or data[:4] != b"%PDF":
                logging.warning(f"  Not a PDF: {raw[:90]}")
                return None
            with open(path, "wb") as f:
                f.write(data)
            return path
        except Exception as e:
            if attempt == 3:
                logging.warning(f"  Download failed {raw[:90]}: {e}")
                return None
            time.sleep(2)


## ─────────────────────────────────────────────────────────────────────────────
## 4) PDF PARSING (value-based heuristic; handles both layouts)
## ─────────────────────────────────────────────────────────────────────────────

CNPJ_RE  = re.compile(r"\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}")
ISPB_RE  = re.compile(r"^\d{8}$")
COVER_RE = re.compile(r"Atualizado em\s*(\d{2})/(\d{2})/(\d{4})")


def _norm(s) -> str:
    return re.sub(r"\s+", " ", str(s).replace("�", "")).strip()


def _classify_row(cells: list) -> dict | None:
    """Pull fields from a table row by value, robust to column order / layout."""
    vals = [_norm(c) for c in cells if c is not None and _norm(c)]
    joined = " ".join(vals)

    cnpj = CNPJ_RE.search(joined)
    cnpj = cnpj.group(0) if cnpj else None

    ispb = next((v for v in vals if ISPB_RE.match(v)), None)

    tipo_pix = None
    for v in vals:
        u = v.upper()
        if u.startswith("OBRIGAT"):
            tipo_pix = "obrigatoria"; break
        if u.startswith("FACULT"):
            tipo_pix = "facultativa"; break

    spi = next((v.lower() for v in vals
                if v.upper() in ("DIRETA", "INDIRETA")), None)

    # process/operational status (registration lists: "Em andamento" / "Concluído";
    # modern lists: "Ativo em operação plena", etc.)
    status = next((v for v in vals if re.match(
        r"(Em\s+andamento|Conclu|Ativo|Inativo|Suspens|Em\s+ades)", v, re.I)), None)

    modalidade = next((v for v in vals
                       if re.match(r"(PROV|Provedor|LIQUID|Liquidante)", v, re.I)), None)

    # institution type (modern only): long descriptive cell
    tipo_inst = next((v for v in vals if re.search(
        r"Institui|Sociedade|Banco|Cooperativa|Confedera|Corretora|Distribuidora|"
        r"Caixa|Agência|Câmara|B3|Financeira|Cr.dito", v, re.I)), None)

    # A genuine participant row needs an identifier and at least one classification.
    # Registration-phase lists (May–Sep 2020) carry SPI only (no mandatory/voluntary
    # column yet); launch and modern lists carry both.
    if not (cnpj or ispb) or not (tipo_pix or spi):
        return None

    name = None
    for v in vals:
        if v in (cnpj, ispb, modalidade, tipo_inst, status):
            continue
        if v.upper() in ("DIRETA", "INDIRETA", "SIM", "NÃO", "NAO", "N/A"):
            continue
        if v.upper().startswith(("OBRIGAT", "FACULT", "ATIVO", "CONCLU", "EM ANDAMENTO")):
            continue
        if re.fullmatch(r"\d+", v):  # sequence number
            continue
        name = v; break

    return {"razao_social": name, "cnpj": cnpj, "ispb": ispb,
            "tipo_pix": tipo_pix, "spi": spi, "status": status,
            "modalidade": modalidade, "tipo_instituicao": tipo_inst}


def parse_roster(path: str, fallback_date) -> tuple[list[dict], object]:
    rows, cover_date = [], None
    try:
        with pdfplumber.open(path) as pdf:
            txt0 = pdf.pages[0].extract_text() or ""
            m = COVER_RE.search(txt0)
            if m:
                cover_date = pd.Timestamp(year=int(m.group(3)), month=int(m.group(2)),
                                          day=int(m.group(1))).date()
            for page in pdf.pages:
                for tbl in page.extract_tables() or []:
                    for cells in tbl:
                        rec = _classify_row(cells)
                        if rec:
                            rows.append(rec)
    except Exception as e:
        logging.warning(f"  Parse failed {os.path.basename(path)}: {e}")
    return rows, (cover_date or fallback_date)


## ─────────────────────────────────────────────────────────────────────────────
## 5) BUILD HISTORY, CROSSWALK, ENTRY, TIMELINE
## ─────────────────────────────────────────────────────────────────────────────

def cnpj_root(c):
    if not isinstance(c, str):
        return None
    digits = re.sub(r"\D", "", c)
    return digits[:8].zfill(8) if len(digits) >= 8 else None


def build_history(jobs: list[dict]) -> pd.DataFrame:
    recs = []
    for i, job in enumerate(sorted(jobs, key=lambda j: (j["roster_date"] or pd.Timestamp.max.date(), j["ts"])), 1):
        path = download_job(job)
        if not path:
            continue
        rows, rdate = parse_roster(path, job["roster_date"])
        if not rows or rdate is None:
            logging.info(f"  [{i}/{len(jobs)}] {job['regime']:7s} {str(rdate):12} -> {len(rows)} rows (skipped)")
            continue
        for r in rows:
            r["roster_date"] = rdate
            r["regime"] = job["regime"]
        recs.extend(rows)
        logging.info(f"  [{i}/{len(jobs)}] {job['regime']:7s} {rdate} -> {len(rows)} participants")

    df = pd.DataFrame(recs)
    df["cnpj8"] = df["cnpj"].map(cnpj_root)
    df["ispb"] = df["ispb"].where(df["ispb"].notna(), None)
    # one row per (roster_date, institution): dedupe on best key
    df["key"] = df["ispb"].fillna(df["cnpj8"])
    df = (df.dropna(subset=["key"])
            .sort_values(["roster_date", "key"])
            .drop_duplicates(["roster_date", "key"]))
    return df


def add_ispb_crosswalk(df: pd.DataFrame) -> pd.DataFrame:
    """Modern rows carry ISPB+CNPJ; use them to backfill ISPB onto CNPJ-only rows."""
    xwalk = (df.dropna(subset=["ispb", "cnpj8"])
               .drop_duplicates("cnpj8")
               .set_index("cnpj8")["ispb"].to_dict())
    df["ispb_filled"] = df["ispb"]
    mask = df["ispb_filled"].isna() & df["cnpj8"].notna()
    df.loc[mask, "ispb_filled"] = df.loc[mask, "cnpj8"].map(xwalk)
    # unified institution key: ISPB when known, else cnpj8
    df["inst_id"] = df["ispb_filled"].fillna(df["cnpj8"])
    return df


def build_roster_entry(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("roster_date")
    g = df.groupby("inst_id")
    entry = pd.DataFrame({
        "first_roster_date": g["roster_date"].first(),
        "last_roster_date":  g["roster_date"].last(),
        "n_rosters":         g["roster_date"].nunique(),
        "first_tipo_pix":    g["tipo_pix"].first(),
        "last_tipo_pix":     g["tipo_pix"].last(),
        "ever_mandatory":    g["tipo_pix"].apply(lambda s: (s == "obrigatoria").any()),
        "first_spi":         g["spi"].first(),
        "last_spi":          g["spi"].last(),
        "razao_social":      g["razao_social"].apply(lambda s: s.dropna().iloc[-1] if s.notna().any() else None),
        "cnpj":              g["cnpj"].apply(lambda s: s.dropna().iloc[-1] if s.notna().any() else None),
        "tipo_instituicao":  g["tipo_instituicao"].apply(lambda s: s.dropna().iloc[-1] if s.notna().any() else None),
        "ispb":              g["ispb_filled"].apply(lambda s: s.dropna().iloc[-1] if s.notna().any() else None),
    }).reset_index()
    return entry


def reconcile(roster_entry: pd.DataFrame) -> pd.DataFrame:
    """Join roster classification to 3b key-based entry month, keyed on ISPB."""
    if not os.path.exists(KEYS_ENTRY):
        logging.warning(f"{KEYS_ENTRY} missing — run scrape_3b first; timeline will lack key dates.")
        keys = pd.DataFrame(columns=["ISPB", "Nome", "entry_anomes", "keys_at_entry"])
    else:
        keys = pd.read_csv(KEYS_ENTRY, dtype={"ISPB": str})
        keys["ISPB"] = keys["ISPB"].str.zfill(8)

    re2 = roster_entry.copy()
    re2["ispb"] = re2["ispb"].astype(str).str.zfill(8).where(re2["ispb"].notna())

    tl = keys.merge(re2, left_on="ISPB", right_on="ispb", how="outer")
    tl["ispb_final"] = tl["ISPB"].fillna(tl["ispb"]).fillna(tl["inst_id"])
    tl["nome_final"] = tl["Nome"].fillna(tl["razao_social"])

    tl["entry_anomes"] = pd.to_numeric(tl["entry_anomes"], errors="coerce")
    tl["entry_month_keys"] = tl["entry_anomes"]
    tl["roster_first_anomes"] = pd.to_datetime(tl["first_roster_date"], errors="coerce").dt.strftime("%Y%m").astype("Float64")

    out = tl[[
        "ispb_final", "nome_final", "cnpj", "tipo_instituicao",
        "entry_month_keys", "keys_at_entry",
        "first_roster_date", "last_roster_date", "n_rosters",
        "first_tipo_pix", "last_tipo_pix", "ever_mandatory",
        "first_spi", "last_spi",
    ]].rename(columns={"ispb_final": "ispb", "nome_final": "nome"})
    out = out.sort_values(["entry_month_keys", "first_roster_date"], na_position="last")
    return out.reset_index(drop=True)


## ─────────────────────────────────────────────────────────────────────────────
## 6) MAIN
## ─────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(PIX_DIR, exist_ok=True)
    jobs = enumerate_rosters()
    if not jobs:
        raise RuntimeError("No roster jobs enumerated from Wayback CDX.")

    hist = build_history(jobs)
    hist = add_ispb_crosswalk(hist)
    hist_out = hist[["roster_date", "regime", "inst_id", "ispb_filled", "cnpj", "cnpj8",
                     "razao_social", "tipo_pix", "spi", "status", "tipo_instituicao", "modalidade"]]
    hist_out.to_csv(HISTORY_CSV, index=False)
    logging.info(f"Saved roster history -> {HISTORY_CSV}  ({len(hist_out):,} rows, "
                 f"{hist['roster_date'].nunique()} roster dates, {hist['inst_id'].nunique()} institutions)")

    entry = build_roster_entry(hist)
    entry.to_csv(ROSTER_ENTRY, index=False)
    logging.info(f"Saved roster entry   -> {ROSTER_ENTRY}  ({len(entry):,} institutions)")

    timeline = reconcile(entry)
    timeline.to_csv(TIMELINE_CSV, index=False)
    logging.info(f"Saved unified timeline -> {TIMELINE_CSV}  ({len(timeline):,} rows)")

    # Console summary
    rd = sorted(hist["roster_date"].dropna().unique())
    print("\nPix roster history")
    print(f"  Roster dates:        {len(rd)}  ({rd[0]} … {rd[-1]})")
    print(f"  Institutions seen:   {hist['inst_id'].nunique()}")
    cls = entry["last_tipo_pix"].value_counts(dropna=False)
    print("  Latest classification (last_tipo_pix):")
    for k, v in cls.items():
        print(f"     {k}: {v}")
    print(f"  Ever mandatory:      {int(entry['ever_mandatory'].sum())}")
    print("\n  Earliest roster (first 6 institutions by first_roster_date):")
    print(entry.sort_values('first_roster_date').head(6)[
        ['razao_social', 'first_roster_date', 'first_tipo_pix', 'first_spi']].to_string(index=False))


if __name__ == "__main__":
    main()
