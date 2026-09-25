"""
scrape_ancine_ad_films.py
=========================
Advertising films registered with ANCINE (Agencia Nacional do Cinema), mapped to the market
panel's prudential conglomerates. Every advertising film shown on open or pay TV, in cinemas,
on home video or in other regulated segments needs a Certificado de Registro de Titulo (CRT)
before it airs, and ANCINE publishes every CRT since 2013 as open data. The advertiser's CNPJ is
on each CRT, so a bank's film output can be counted month by month for the whole panel window,
for every bank, from one source - which none of the spending sources can offer.

WHAT THE COUNT MEASURES, AND WHAT IT MISSES
-------------------------------------------
  - Ads shown ONLY online (websites, YouTube, social media) are EXEMPT from registration
    (ANCINE FAQ). Digital-first banks that advertise mostly online are therefore undercounted,
    and a bank that moves its budget from TV to online shows a fall in CRTs with no fall in
    advertising. The monthly table carries this as `coverage_note`.
  - One CRT covers up to 5 versions of a film (50 for retail). The count is of FILMS, not of
    airings, insertions or spend.
  - CONDECINE is charged per title per market segment, so one film can carry one CRT per
    segment. `n_titles_distinct` counts a (normalised title, advertiser CNPJ8) pair once per
    month, which undoes that multiplication.

THE FILE
--------
  https://dados.ancine.gov.br/dados-abertos/crt-obras-publicitarias.csv, refreshed monthly, no key.
  Measured on the 2026-09-01 vintage (313,757,397 bytes): UTF-8 without BOM, ';'-delimited,
  25 columns, 562,074 rows, 537,163 distinct CRT numbers. The published field list omits three
  columns the file carries (CRTS_VERSOES, REGISTRO_ANCINE_REQUERENTE, CNPJ_REQUERENTE). A CRT
  appears on several rows (14,941 CRTs do) when it names several producers, directors, agencies
  or advertisers. Measured over the whole file, the rows of one CRT differ in PRODUTOR, DIRETOR,
  AGENCIA, CNPJ_AGENCIA, ANUNCIANTE and CNPJ_ANUNCIANTE, and for a single CRT in the five
  requester columns; the other 13 columns (title, the three dates, status, segment, version
  count and duration among them) never vary within a CRT. Every count here is therefore of
  DISTINCT CRT numbers, never rows. Check 6 below re-tests title, request date and segment on
  the mapped CRTs on every run. The last request month is incomplete: this vintage's latest
  request and issue dates are both 2026-08-24, a week before its Last-Modified (2026-09-01).
  The encoding and delimiter are detected from the file's first bytes on every run and the
  whole file is decoded strictly, so a change of either stops the run instead of mis-reading.
  Every CRT status is counted (EXPIRADO, REGISTRADO, IRREGULAR, CADASTRADO); SITUACAO_CRT stays
  on the lines table for a consumer who wants to drop some. The file holds 494,885 distinct CRTs
  issued by 2025-05-31, 1.2% fewer than the 500,663 on ANCINE's dashboard for the same window;
  the gate below therefore tests the whole file against that figure, not the window.

MAPPING ADVERTISERS TO CONGLOMERATES
------------------------------------
CNPJ Anunciante is reduced to 14 digits (placeholders such as 'PESSOA FISICA' and 'NAO
INFORMADO' carry no CNPJ and are never zero-filled into one) and then to its 8-digit root. The
root is matched against the IF.data lists with build_advertising_crosswalk's own registry
loader and candidate rule, including its guard that only digit-bearing CNPJ fields are
compared: Banco do Brasil's root is 00000000, so a zero-filled blank would match it. For each
CRT the quarter of its REQUEST date decides the code, in this order:
  panel_leader       the root is the leader CNPJ of a market-panel conglomerate that exists in
                     that quarter, and the IF.data lists confirm that leadership. The panel's own
                     identity wins over a later registry move: the registry folds Banco Pan into
                     BTG's conglomerate from 2021Q2 and PicPay into Original's in 2019Q4-2024Q3,
                     while the market panel keeps both as their own entities with their own
                     deposits. `registry_code` and `code_conflict` keep the disagreement visible.
  registry_quarter   the IF.data conglomerate code whose dated span contains the quarter
  registry_nearest   the quarter lies before the first IF.data list that carries conglomerate
                     codes (201403) or after the last list (202512), where the lists cannot
                     speak: the nearest span's code. Inside that window a root with no span
                     covering the quarter had no conglomerate code then; its rows stay
                     unmapped with reason `no_code_in_quarter` instead of borrowing a
                     neighbouring span.
A code is kept only when it is a market-panel code. Quarters outside the panel window are
evaluated at the window's nearest edge (`map_quarter` records the quarter actually used). Both
window edges are read from the data on every run and logged.
Holding companies and group affiliates that the prudential registry does not contain (insurers,
card issuers, asset managers, foreign holdings) are NOT guessed into a bank: they are listed in
ancine_unmatched_financial for the user to decide.

OUTPUTS (paths.AWARENESS_PROC, parquet + csv, written atomically)
-----------------------------------------------------------------
  ancine_ad_films_lines        every CRT row whose advertiser maps to a panel conglomerate, all
                               25 original fields as published plus cnpj14, cnpj_valid, cnpj8,
                               the parsed request and issue dates, year/month/quarter of the
                               request, map_quarter, registry_code, panel_code, match_method,
                               code_conflict, outside_panel_window, in_panel, partial_month,
                               panel_name and title_norm
  ancine_ad_films_monthly      panel_code x year x month of the REQUEST date, balanced over
                               2013-01 to the last request month for every conglomerate with at
                               least one CRT: n_crt, n_titles_distinct, n_crt_seg_<segment>,
                               sum_qtd_versoes (over CRTs that report a version count),
                               n_crt_versoes_reported, coverage_note, and two flags:
                                 in_panel       the code has market-panel rows in that month's
                                                quarter. A month without a CRT is a zero only
                                                where in_panel is True; elsewhere the
                                                conglomerate did not exist in the panel (not
                                                yet formed, merged away, or after the panel's
                                                last quarter), and its zero is not a data point
                                 partial_month  the file cannot hold the whole month: its end,
                                                plus ISSUE_LAG_DAYS, falls after the file's
                                                data cutoff (the earlier of the manifest's
                                                Last-Modified and the day after the file's
                                                latest issue date). The row is kept.
  ancine_advertiser_map        cnpj8 -> panel_code, match_method, first/last quarter, names on
                               both sides and `names_agree` (a review flag only: renames such
                               as Aymore -> Santander SCFI share no word, and the CNPJ root, not
                               the name, is the legal identity)
  ancine_unmatched_financial   advertisers that did not map and whose ANCINE or registry name
                               looks financial (or whose reason is no_code_in_quarter), with
                               CRT counts and the reason:
                                 not_in_registry             root absent from the IF.data lists
                                                             (insurers, capitalizacao, card
                                                             schemes, holdings, retailers)
                                 no_conglomerate_code        in the lists but never inside a
                                                             prudential conglomerate (credit
                                                             cooperatives: the market panel
                                                             holds no cooperative)
                                 no_code_in_quarter          in a conglomerate in other
                                                             quarters, but no span covers the
                                                             request quarter inside the lists'
                                                             window; always queued, whatever
                                                             its name
                                 registry_code_not_in_panel  its conglomerate is not a market-
                                                             panel code (BNDES, no deposits)
                                 ambiguous                   overlapping registry spans, or two
                                                             spans equally near
                                 ambiguous_panel_leader      two panel codes led by the same root
                                 no_cnpj, invalid_cnpj       placeholder or failed check digits
                               plus `kind` and `brand_word`, review aids that sort the queue
  ancine_ad_films_provenance.json   sidecar: the source file's URL, source_last_modified,
                               source_sha256 (re-verified against the file on disk), size and
                               retrieval time, the data cutoff and the registry and panel
                               windows the run used, and each table's row count
The download is cached under paths.AWARENESS_RAW/ancine with a manifest (URL, status, bytes,
Content-Length, SHA-256, Last-Modified, retrieval time, file name).

VALIDATION (any failure aborts before writing)
----------------------------------------------
  1. The file on disk has exactly the server's Content-Length and the manifest's SHA-256.
  2. At least 500,663 rows and 500,663 distinct CRTs (ANCINE's dashboard count of advertising
     CRTs issued 2013 to May 2025); every CRT number has 14 digits.
  3. At least 95% of the request and the issue dates parse.
  4. No panel_code-month appears twice in the monthly table; segment counts add up to n_crt.
  5. Every mapped CNPJ8 exists in the IF.data registry; every panel_code exists in the market
     panel; for each of the crosswalk's anchors (Banco do Brasil, Caixa, Itau holding, Nu) whose
     root is on any row of the file (a valid CNPJ, counted before the registry filter), EVERY
     such row maps to its code (an unmapped row, or one that never reached the mapping, fails
     too), and Banco do Brasil, Caixa and Nu must have rows, so the check cannot pass by
     finding nothing.
  6. Title, request date and segment never vary within a mapped CRT.
Each check logs what it measured when it passes.

Usage
-----
  python scrape_ancine_ad_films.py                  # uses the cached download
  python scrape_ancine_ad_films.py --refresh        # downloads the current vintage first
  python scrape_ancine_ad_films.py --download-only
  python scrape_ancine_ad_films.py --out-dir <dir>  # writes the four tables and the
                                                    # provenance file elsewhere
"""

from __future__ import annotations

import argparse
import codecs
import csv
import hashlib
import io
import json
import logging
import os
import re
import time
import unicodedata
import urllib.request
from datetime import timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

import numpy as np
import pandas as pd

import build_advertising_crosswalk as xw
from utils import paths
from utils.disclosure_common import now_iso

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

URL = "https://dados.ancine.gov.br/dados-abertos/crt-obras-publicitarias.csv"
RAW_DIR = paths.AWARENESS_RAW / "ancine"
CSV_PATH = RAW_DIR / "crt-obras-publicitarias.csv"
MANIFEST = RAW_DIR / "manifest.json"
OUT_DIR = paths.AWARENESS_PROC
MARKET_PANEL = xw.MARKET_PANEL
# No contact address in the agent string: ANCINE asks for none, and the project's shared
# agent string carries one for the SEC's benefit only.
USER_AGENT = "deposit-competition-research (academic; bulk open-data download)"

DASHBOARD_CRTS = 500_663          # ANCINE dashboard: advertising CRTs issued 2013 to May 2025
MIN_DATE_PARSE = 0.95
# A CRT enters the file when it is issued, which can be days after its request: of the CRTs
# requested in 2025, 9.8% were issued after their request month ended and 2.6% more than 10
# days after it. A request month therefore counts as complete only once this many days past
# its end fall inside the file's data. The months just before it can still gain a few CRTs in
# a later vintage.
ISSUE_LAG_DAYS = 10
# Anchors that must have CRT rows. Their absence means the file or the CNPJ parsing changed,
# and the anchor check would otherwise pass by finding nothing to test. The Itau holding
# (60872504) is not required: Itau's films are registered under its operating bank's CNPJ.
REQUIRED_ANCHORS = ("00000000", "00360305", "18236120")   # Banco do Brasil, Caixa, Nu
ANCHOR_ROOTS = set(xw.ANCHORS)
DEP_COLS = ["dep_a1", "dep_a2", "dep_a4", "dep_a5"]
CHUNK_ROWS = 100_000

COL_TITLE, COL_CRT = "TITULO_ORIGINAL", "CRT"
COL_REQ, COL_ISSUE, COL_VALID = "DATA_REQUERIMENTO_CRT", "DATA_EMISSAO_CRT", "DATA_VALIDADE_CRT"
COL_SEG, COL_VERS = "SEGMENTO", "QTD_VERSOES"
COL_ADV, COL_ADV_CNPJ = "ANUNCIANTE", "CNPJ_ANUNCIANTE"
REQUIRED = [COL_TITLE, COL_CRT, COL_REQ, COL_ISSUE, COL_VALID, COL_SEG, COL_VERS, COL_ADV,
            COL_ADV_CNPJ]

COVERAGE_NOTE = ("CRT counts cover films for TV, cinema, home video and other regulated segments; "
                 "ads shown only online, on YouTube or on social media are exempt from "
                 "registration, so online-heavy (digital) banks are undercounted. One CRT covers "
                 "up to 5 versions (50 for retail) and a film may carry one CRT per market "
                 "segment: counts are films, not airings or spend.")

# Folded segment label -> column suffix. An unseen label gets a slug of its own text, so a new
# segment appears as a new column instead of vanishing.
SEGMENTS = {"RADIODIFUSAO DE SONS E IMAGENS (TV ABERTA)": "tv_aberta",
            "TODOS OS SEGMENTOS DE MERCADO": "todos_segmentos",
            "COMUNICACAO ELETRONICA DE MASSA POR ASSINATURA (TV PAGA)": "tv_paga",
            "OUTROS MERCADOS": "outros_mercados",
            "SALAS DE EXIBICAO": "salas_exibicao",
            "VIDEO DOMESTICO": "video_domestico"}

# Words that make an advertiser name look financial, matched as whole words on the folded name.
# Deliberately broad: the list is a review queue for the user, not a mapping. Insurance is
# matched by SEGURO/SEGURADORA/SEGURIDADE rather than the stem SEGUR, which would also pull in
# every private security firm (SEGURANCA). ELO (the card scheme of Banco do Brasil, Bradesco
# and Caixa), LIVELO (the loyalty programme of Banco do Brasil and Bradesco), VERO and
# BANRISUL SOLUCOES (Banrisul's card acquirer) and BOLSA/B3 (the exchange) advertise under
# names that carry none of the other words.
FINANCIAL = re.compile(
    r"\b(?:BANCO|BANCOS|BCO|BANK|BANKING|FINANCEIRA|FINANCIAMENTO\w*|FINANCE|FINANCAS|"
    r"CREDITO\w*|CREDIT|SEGURO|SEGUROS|SEGURADORA|SEGURIDADE|CARTAO|CARTOES|CARD|CARDS|"
    r"CAPITALIZACAO|INVEST\w*|PAGAMENTO\w*|PAY|NU|NUBANK|PICPAY|PREVIDENCIA|PREV|"
    r"CONSORCIO\w*|CORRETORA|DTVM|VALORES MOBILIARIOS|"
    r"LEASING|ARRENDAMENTO|COOPERATIVA DE CREDITO|SICOOB|SICREDI|UNICRED|CRESOL|ASSET|"
    r"GESTORA|GESTAO DE RECURSOS|CAIXA ECONOMICA|CAIXA SEGURIDADE|BRADESCO|ITAU\w*|UNIBANCO|"
    r"SANTANDER|BRASILPREV|HIPERCARD|CREDICARD|MASTERCARD|VISA|CIELO|GETNET|STONE|PAGSEGURO|"
    r"PAGBANK|MERCADO PAGO|MERCADOPAGO|BMG|SAFRA|BTG|XP|DAYCOVAL|BANRISUL|BANESTES|BRB|"
    r"CREFISA|NEON|AGIBANK|C6|INTER|BANPARA|BANESE|ALELO|SERASA|CAMBIO|FOMENTO|FACTORING|"
    r"EMPRESTIMO\w*|ELO|LIVELO|VERO|BOLSA|B3|BANRISUL SOLUCOES)\b")
# Names the financial words catch that are plainly not financial firms: public bodies,
# condominiums and donation banks (BANCO DE OLHOS is an eye bank with 535 CRTs).
NOT_FINANCIAL = re.compile(r"\b(?:MUNICIPIO|PREFEITURA|SECRETARIA|CONDOMINIO|COND|"
                           r"BANCO DE (?:OLHOS|ALIMENTOS|SANGUE|LEITE|TECIDOS))\b")

# Review aids for the unmatched list, never used to map. `kind` sorts the queue; `brand_word`
# names the panel-bank brand an affiliate carries (BRADESCO SEGUROS, CAIXA SEGURADORA), which is
# the decision the user has to make: whether such an affiliate's films count for the bank.
KINDS = (("cooperative", r"\b(?:COOPERATIVA|COOP|SICOOB|SICREDI|UNICRED|CRESOL|CONFEDERACAO)\b"),
         ("insurance_pension", r"\b(?:SEGURO|SEGUROS|SEGURADORA|SEGURIDADE|CAPITALIZACAO|"
                               r"PREVIDENCIA|PREVIDENCIAL|BRASILPREV|BRASILSEG)\b"),
         ("consorcio", r"\bCONSORCIO\w*\b"),
         ("cards_payments", r"\b(?:CARTAO|CARTOES|CARD|CARDS|PAGAMENTO\w*|PAY|MASTERCARD|VISA|"
                            r"CIELO|GETNET|STONE|PAGSEGURO|PAGBANK|MERCADO ?PAGO|ALELO|ELO|"
                            r"VERO)\b"),
         ("bank_or_credit", r"\b(?:BANCO|BANK|BCO|FINANCEIRA|CREDITO|FINANCIAMENTO|"
                            r"EMPRESTIMO\w*|DTVM|CORRETORA|INVEST\w*|ASSET|GESTORA)\b"))
# BANRISUL SOLUCOES precedes BANRISUL because the first alternative that matches is the one
# extracted.
BRANDS = re.compile(r"\b(BRADESCO|ITAU|UNIBANCO|SANTANDER|CAIXA|BB|BANCO DO BRASIL|BRASILPREV|"
                    r"BRASILSEG|NU|NUBANK|INTER|C6|PICPAY|BTG|SAFRA|XP|BANRISUL SOLUCOES|"
                    r"BANRISUL|BMG|VOTORANTIM|BV|MERCADO PAGO|PAGSEGURO|PAGBANK|NEON|AGIBANK|"
                    r"ORIGINAL|PAN|ELO|LIVELO|VERO|BOLSA|B3)\b")
# Several brand words are also ordinary words or other firms' names (SAFRA is a harvest, INTER
# and PAN start many company names). A name that also says it is a builder, a farm, a food
# maker or an industry is such a firm, so it gets no brand_word; it stays in the queue.
NOT_BRAND = re.compile(r"\b(?:INCORPORACAO|INCORPORACOES|CONSTRUTORA|AGROPECUARIA|ALIMENTICIOS|"
                       r"INDUSTRIA|INDUSTRIAS|FERTILIZANTES)\b")

# Words too common in institution names to show that two names refer to the same firm.
NAME_STOP = {"BANCO", "BCO", "BANK", "S", "A", "SA", "S/A", "LTDA", "HOLDING", "DO", "DA", "DE",
             "DOS", "DAS", "E", "EM", "FINANCEIRA", "LEASING", "ARRENDAMENTO", "MERCANTIL",
             "CREDITO", "FINANCIAMENTO", "INVESTIMENTO", "INVESTIMENTOS", "COOPERATIVA",
             "CENTRAL", "LIVRE", "ADMISSAO", "ASSOCIADOS", "POUPANCA", "ECONOMIA", "MUTUO",
             "INSTITUICAO", "PAGAMENTO", "PAGAMENTOS", "SOCIEDADE", "SERVICOS", "COMPANHIA",
             "BRASIL", "BRASILEIRA", "ADMINISTRADORA", "CARTOES", "DISTRIBUIDORA", "TITULOS",
             "VALORES", "MOBILIARIOS", "CORRETORA", "CAMBIO", "PRUDENCIAL", "ESTADO", "ME",
             "EPP", "EIRELI", "GRUPO", "PARTICIPACOES", "NACIONAL", "REGIONAL", "SUL", "NORTE",
             "DESENVOLVIMENTO", "SICOOB", "SICREDI", "CRESOL", "UNICRED", "RURAL"}


# ---------------------------------------------------------------------------
# Text and identifier helpers
# ---------------------------------------------------------------------------
def fold(text) -> str:
    """Upper-case ASCII fold: accents removed, whitespace collapsed."""
    t = unicodedata.normalize("NFKD", str(text))
    t = "".join(c for c in t if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", t).strip().upper()


def norm_title(text) -> str:
    """Title key: folded, punctuation and quote marks dropped. ANCINE titles arrive as
    '"SORT 03 FEV 13"', "''CORES DA SORTE''" or plain, for the same kind of film."""
    return re.sub(r"\s+", " ", re.sub(r"[^A-Z0-9 ]+", " ", fold(text))).strip()


LEGAL_FORM = {"S", "A", "SA", "LTDA", "ME", "EPP", "EIRELI"}


def words(name) -> set[str]:
    return {t for t in re.split(r"[^A-Z0-9]+", fold(name)) if t and t not in LEGAL_FORM}


def names_agree(ancine: list[str], registry: list[str]) -> bool:
    """True when an ANCINE advertiser name and a registry name look like the same firm.

    Two-character words count ('XP', 'C6', 'NU', 'BV' are the whole brand). Where one side has
    no distinctive word at all ('BANCO DO BRASIL S.A.', 'BANCO MERCANTIL DO BRASIL SA'), the
    full word sets are compared instead, so a name made only of common words is not reported
    as a mismatch with itself."""
    for a in ancine:
        wa = words(a)
        da = {t for t in wa if len(t) >= 2} - NAME_STOP
        for r in registry:
            wr = words(r)
            dr = {t for t in wr if len(t) >= 2} - NAME_STOP
            if da & dr:
                return True
            if (not da or not dr) and wa and wr and (wa <= wr or wr <= wa):
                return True
    return False


def cnpj_check_ok(d14: str) -> bool:
    """Modulo-11 check digits of a 14-digit CNPJ. A string of one repeated digit passes the
    arithmetic but is no CNPJ (00000000000000 would otherwise be Banco do Brasil's root)."""
    if len(d14) != 14 or not d14.isdigit() or len(set(d14)) == 1:
        return False
    digits = [int(c) for c in d14]
    w1 = [5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2]
    w2 = [6] + w1
    r1 = sum(a * b for a, b in zip(digits[:12], w1)) % 11
    r2 = sum(a * b for a, b in zip(digits[:13], w2)) % 11
    return digits[12] == (0 if r1 < 2 else 11 - r1) and digits[13] == (0 if r2 < 2 else 11 - r2)


def ym_of(year, quarter) -> int:
    """Quarter as the IF.data lists key it: YYYYMM of the quarter's last month."""
    return int(year) * 100 + int(quarter) * 3


def qindex(ym: int) -> int:
    return (ym // 100) * 4 + (ym % 100) // 3 - 1


def ym_label(ym: int) -> str:
    return f"{ym // 100}Q{(ym % 100) // 3}"


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------
def write_json_atomic(path: Path, obj: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False), encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def download(refresh: bool) -> dict:
    """Fetch the CRT file once and record what the server said about it.

    The body is streamed to a '.part' file and moved into place only after its byte count
    equals the server's Content-Length, so an interrupted transfer never leaves a truncated
    file under the real name. Compression is refused (Accept-Encoding: identity) because a
    compressed body would make the Content-Length comparison meaningless.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {}
    if (not refresh and manifest and CSV_PATH.exists()
            and CSV_PATH.stat().st_size == manifest.get("bytes")):
        log.info("cached %s (%d bytes, Last-Modified %s, retrieved %s)", CSV_PATH.name,
                 manifest["bytes"], manifest.get("last_modified"), manifest.get("retrieved_at"))
        return manifest

    part = CSV_PATH.with_name(CSV_PATH.name + ".part")
    headers = {"User-Agent": USER_AGENT, "Accept-Encoding": "identity", "Accept": "*/*"}
    for attempt in range(1, 4):
        try:
            req = urllib.request.Request(URL, headers=headers)
            digest, n = hashlib.sha256(), 0
            with urllib.request.urlopen(req, timeout=180) as resp, part.open("wb") as fh:
                status = getattr(resp, "status", None)
                length = resp.headers.get("Content-Length")
                last_mod = resp.headers.get("Last-Modified")
                encoding = resp.headers.get("Content-Encoding")
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    fh.write(chunk)
                    digest.update(chunk)
                    n += len(chunk)
            if encoding and encoding.lower() != "identity":
                raise RuntimeError(f"server sent Content-Encoding {encoding!r} despite identity")
            if length is None or n != int(length):
                raise RuntimeError(f"received {n} bytes, Content-Length {length}")
            break
        except Exception as exc:  # noqa: BLE001
            log.warning("download attempt %d/3 failed: %s: %s", attempt, type(exc).__name__, exc)
            if part.exists():
                part.unlink()
            if attempt == 3:
                raise SystemExit(f"could not download {URL}")
            time.sleep(10 * attempt)
    os.replace(part, CSV_PATH)
    manifest = {"url": URL, "status": status, "bytes": n, "content_length": int(length),
                "sha256": digest.hexdigest(), "last_modified": last_mod,
                "retrieved_at": now_iso(), "file": CSV_PATH.name}
    write_json_atomic(MANIFEST, manifest)
    log.info("GET %s %s: %d bytes, Last-Modified %s", status, URL, n, last_mod)
    return manifest


# ---------------------------------------------------------------------------
# Reading the file
# ---------------------------------------------------------------------------
def sniff(path: Path) -> tuple[str, str, list[str]]:
    """Encoding, delimiter and header from the first bytes of the file.

    Encoding: a BOM decides outright; otherwise the first 4 MB (cut at a line end) must decode
    as strict UTF-8, else cp1252 is used. The later full read is strict, so a wrong call here
    raises instead of producing mojibake. Delimiter: the candidate that occurs most in the
    header line, confirmed by the csv module giving every sampled line the header's field count.
    """
    with path.open("rb") as fh:
        head = fh.read(4 << 20)
    boms = ((codecs.BOM_UTF8, "utf-8-sig"), (codecs.BOM_UTF16_LE, "utf-16"),
            (codecs.BOM_UTF16_BE, "utf-16"))
    enc = next((e for bom, e in boms if head.startswith(bom)), None)
    sample = head[: head.rfind(b"\n") + 1] if b"\n" in head else head
    n_high = sum(b > 127 for b in sample[:1_000_000])
    if enc is None:
        try:
            sample.decode("utf-8")
            enc = "utf-8"
        except UnicodeDecodeError:
            enc = "cp1252"
    text = sample.decode(enc)
    lines = text.splitlines()
    header_line = lines[0]
    counts = {d: header_line.count(d) for d in (";", ",", "\t", "|")}
    delim = max(counts, key=counts.get)
    if counts[delim] == 0:
        raise SystemExit(f"no delimiter found in the header line: {header_line[:200]!r}")
    rows = list(csv.reader(io.StringIO("\n".join(lines[:2000])), delimiter=delim))
    width = len(rows[0])
    bad = sum(len(r) != width for r in rows[1:])
    if bad > 0.01 * (len(rows) - 1):
        raise SystemExit(f"delimiter {delim!r}: {bad} of {len(rows) - 1} sampled lines do not "
                         f"have the header's {width} fields")
    log.info("format: encoding %s (BOM %s, %d non-ASCII bytes in the first MB), delimiter %r, "
             "%d columns", enc, "yes" if enc.endswith("sig") else "no", n_high, delim, width)
    return enc, delim, rows[0]


class ReadStats:
    """Whole-file facts gathered chunk by chunk, for the summary and the validation."""

    def __init__(self):
        self.rows = 0
        self.crts: set[str] = set()
        self.crts_to_may2025: set[str] = set()
        self.bad_crt = 0
        self.dates = {c: {"filled": 0, "parsed": 0, "min": None, "max": None}
                      for c in (COL_REQ, COL_ISSUE, COL_VALID)}
        self.cnpj14 = 0
        self.cnpj_valid = 0
        self.cnpj_text: dict[str, int] = {}
        self.crts_with_cnpj: set[str] = set()
        # Rows per anchor root over the WHOLE file, counted before the registry filter, so the
        # anchor check can tell a row that never reached the mapping from a row that is absent.
        self.anchor_rows: dict[str, int] = {}

    def date(self, col: str, raw: pd.Series, parsed: pd.Series) -> None:
        d = self.dates[col]
        d["filled"] += int(raw.notna().sum())
        d["parsed"] += int(parsed.notna().sum())
        if parsed.notna().any():
            lo, hi = parsed.min(), parsed.max()
            d["min"] = lo if d["min"] is None else min(d["min"], lo)
            d["max"] = hi if d["max"] is None else max(d["max"], hi)


def read_crt(enc: str, delim: str, roots: set[str]) -> tuple[pd.DataFrame, pd.DataFrame,
                                                              ReadStats]:
    """Stream the file. Returns the full rows whose advertiser root is in the IF.data registry,
    a compact frame of the other rows whose advertiser name looks financial, and the stats."""
    st = ReadStats()
    fold_cache: dict[str, str] = {}
    valid_cache: dict[str, bool] = {}
    keep, fin = [], []
    reader = pd.read_csv(CSV_PATH, sep=delim, encoding=enc, encoding_errors="strict", dtype=str,
                         keep_default_na=False, na_values=[""], chunksize=CHUNK_ROWS)
    for chunk in reader:
        st.rows += len(chunk)
        crt = chunk[COL_CRT].fillna("")
        st.bad_crt += int((~crt.str.fullmatch(r"\d{14}")).sum())
        st.crts.update(crt)
        parsed = {}
        for col in (COL_REQ, COL_ISSUE, COL_VALID):
            parsed[col] = pd.to_datetime(chunk[col], format="%d/%m/%Y", errors="coerce")
            st.date(col, chunk[col], parsed[col])
        st.crts_to_may2025.update(crt[parsed[COL_ISSUE] <= "2025-05-31"])

        raw = chunk[COL_ADV_CNPJ]
        digits = raw.fillna("").str.replace(r"\D", "", regex=True)
        has14 = digits.str.len() == 14
        st.cnpj14 += int(has14.sum())
        st.crts_with_cnpj.update(crt[has14])
        for text, n in raw[~has14].fillna("<blank>").value_counts().items():
            st.cnpj_text[text] = st.cnpj_text.get(text, 0) + int(n)
        for d in digits[has14].unique():
            if d not in valid_cache:
                valid_cache[d] = cnpj_check_ok(d)
        cnpj14 = digits.where(has14)
        valid = cnpj14.map(valid_cache).astype("boolean")
        st.cnpj_valid += int(valid.fillna(False).sum())
        # A CNPJ that fails its check digits is not trusted for its root either.
        cnpj8 = cnpj14.str[:8].where(valid.fillna(False))
        for r, n in cnpj8[cnpj8.isin(ANCHOR_ROOTS)].value_counts().items():
            st.anchor_rows[r] = st.anchor_rows.get(r, 0) + int(n)

        chunk = chunk.assign(cnpj14=cnpj14, cnpj_valid=valid, cnpj8=cnpj8,
                             request_date=parsed[COL_REQ], issue_date=parsed[COL_ISSUE])
        in_reg = cnpj8.isin(roots)
        keep.append(chunk[in_reg])

        names = chunk.loc[~in_reg, COL_ADV].fillna("")
        for nm in names.unique():
            if nm not in fold_cache:
                fold_cache[nm] = fold(nm)
        looks = names.map(fold_cache).str.contains(FINANCIAL)
        fin.append(chunk.loc[looks[looks].index, [COL_CRT, COL_ADV, COL_ADV_CNPJ, "cnpj14",
                                                   "cnpj_valid", "cnpj8", "request_date"]])
        log.info("  read %7d rows", st.rows)
    return pd.concat(keep, ignore_index=True), pd.concat(fin, ignore_index=True), st


# ---------------------------------------------------------------------------
# Registry and market panel
# ---------------------------------------------------------------------------
def registry_roots(reg: pd.DataFrame) -> set[str]:
    """Every CNPJ8 the IF.data lists carry, as an institution or as a conglomerate leader.
    Only digit-bearing fields count, the crosswalk's guard against zero-filled blanks."""
    member = reg["CodInst"].str.fullmatch(r"\d{1,8}", na=False)
    leader = reg["CnpjInstituicaoLider"].str.fullmatch(r"\d{1,8}", na=False)
    return (set(reg.loc[member, "CodInst"].str.zfill(8))
            | set(reg.loc[leader, "CnpjInstituicaoLider"].str.zfill(8)))


def registry_subset(reg: pd.DataFrame, roots: set[str]) -> pd.DataFrame:
    """The registry rows that can concern `roots`, so the crosswalk's per-root candidate rule
    runs on a few thousand rows instead of the whole history."""
    member = reg["CodInst"].str.fullmatch(r"\d{1,8}", na=False) & \
        reg["CodInst"].str.zfill(8).isin(roots)
    leader = reg["CnpjInstituicaoLider"].str.fullmatch(r"\d{1,8}", na=False) & \
        reg["CnpjInstituicaoLider"].str.zfill(8).isin(roots)
    return reg[member | leader].copy()


def panel_identity() -> tuple[pd.DataFrame, pd.Series, pd.MultiIndex]:
    """Per market-panel code: first and last quarter, leader CNPJ8, name; 2024Q4 deposits; and
    every (code, year, quarter) the panel holds. The last is what `in_panel` tests: a code's
    span can have holes (20 of 536 codes do), so first-to-last is not the same thing."""
    mp = pd.read_parquet(MARKET_PANEL, columns=["CodConglomeradoPrudencial", "CNPJ_Lider",
                                                "NomeInstituicao", "year", "quarter"] + DEP_COLS)
    present = pd.MultiIndex.from_frame(
        mp[["CodConglomeradoPrudencial", "year", "quarter"]].drop_duplicates()
          .astype({"year": "int64", "quarter": "int64"}))
    mp["ym"] = mp["year"] * 100 + mp["quarter"] * 3
    spans = (mp.groupby("CodConglomeradoPrudencial")
               .agg(first=("ym", "min"), last=("ym", "max"),
                    leader=("CNPJ_Lider", "first"), n_leaders=("CNPJ_Lider", "nunique"),
                    panel_name=("NomeInstituicao", "first")))
    if (spans["n_leaders"] > 1).any():
        raise AssertionError("a market-panel code carries several CNPJ_Lider values: "
                             f"{spans[spans['n_leaders'] > 1].index[:5].tolist()}")
    spans["leader8"] = spans["leader"].astype("int64").astype(str).str.zfill(8)
    dep = mp[DEP_COLS].sum(axis=1, min_count=1)
    q4 = (mp.assign(dep=dep)[(mp["year"] == 2024) & (mp["quarter"] == 4)]
            .groupby("CodConglomeradoPrudencial")["dep"].sum())
    return spans, q4, present


def in_panel(code: pd.Series, year: pd.Series, quarter: pd.Series,
             present: pd.MultiIndex) -> np.ndarray:
    """True where the market panel has rows for `code` in that year-quarter."""
    keys = pd.MultiIndex.from_arrays([code.astype(str), year.astype("int64"),
                                      quarter.astype("int64")])
    return np.asarray(keys.isin(present))


def registry_window(reg: pd.DataFrame) -> tuple[int, int]:
    """First IF.data list that carries a conglomerate code, and the last list (YYYYMM)."""
    coded = reg["CodConglomeradoPrudencial"].fillna("").ne("")
    return int(reg.loc[coded, "Data"].min()), int(reg["Data"].max())


class Mapper:
    """Code for a (CNPJ8, request quarter), by the rule order in the module docstring."""

    def __init__(self, reg_sub: pd.DataFrame, spans: pd.DataFrame, roots: list[str],
                 reg_window: tuple[int, int]):
        self.spans = spans
        self.panel_codes = set(spans.index)
        self.panel_first, self.panel_last = int(spans["first"].min()), int(spans["last"].max())
        self.reg_first, self.reg_last = reg_window
        self.periods: dict[str, pd.DataFrame] = {}
        for r in roots:
            c = xw.candidates(reg_sub, r)
            c = c.assign(min=c["min"].astype(int), max=c["max"].astype(int))
            self.periods[r] = c.rename(columns={"CodConglomeradoPrudencial": "code"})
        # The panel's leader identity is accepted only where the IF.data lists name the same
        # root as that conglomerate's leader (or the panel keyed the bank by its own CNPJ): the
        # panel stores CNPJ_Lider as an integer, and a zero there would otherwise read as
        # Banco do Brasil's root.
        has_leader = reg_sub["CnpjInstituicaoLider"].str.fullmatch(r"\d{1,8}", na=False)
        confirmed = set(zip(reg_sub.loc[has_leader, "CnpjInstituicaoLider"].str.zfill(8),
                            reg_sub.loc[has_leader, "CodConglomeradoPrudencial"]))
        self.leader_codes: dict[str, list[str]] = {}
        for code, row in spans.iterrows():
            if (row["leader8"], code) in confirmed or code == f"CNPJ_{int(row['leader'])}":
                self.leader_codes.setdefault(row["leader8"], []).append(code)

    def registry_code(self, cnpj8: str, ym: int) -> tuple[str | None, str]:
        per = self.periods.get(cnpj8)
        if per is None or per.empty:
            return None, "no_conglomerate_code"
        live = per[(per["min"] <= ym) & (per["max"] >= ym)]
        if len(live) == 1:
            return live["code"].iloc[0], "registry_quarter"
        if len(live) > 1:
            return None, "ambiguous"
        # Inside the lists' window the lists are the record: a root with no span covering the
        # quarter belonged to no conglomerate then (before joining one, after leaving one, or
        # between two), and the nearest span would assign it a membership it did not have.
        # Only before the first coded list or after the last one is the nearest span the best
        # available guess.
        if self.reg_first <= ym <= self.reg_last:
            return None, "no_code_in_quarter"
        q = qindex(ym)
        dist = np.where(ym < per["min"], per["min"].map(qindex) - q, q - per["max"].map(qindex))
        best = np.flatnonzero(dist == dist.min())
        if len(best) > 1:
            return None, "ambiguous"
        return per["code"].iloc[best[0]], "registry_nearest"

    def assign(self, cnpj8: str, ym: int) -> dict:
        ym_p = min(max(ym, self.panel_first), self.panel_last)
        lead = [c for c in self.leader_codes.get(cnpj8, ())
                if self.spans.at[c, "first"] <= ym_p <= self.spans.at[c, "last"]]
        reg_code, how = self.registry_code(cnpj8, ym_p)
        out = {"cnpj8": cnpj8, "ym": ym, "map_quarter": ym_label(ym_p),
               "registry_code": reg_code, "panel_code": None, "match_method": None,
               "unmapped_reason": None}
        if len(lead) == 1:
            out.update(panel_code=lead[0], match_method="panel_leader")
        elif len(lead) > 1:
            out.update(unmapped_reason="ambiguous_panel_leader")
        elif reg_code is None:
            out.update(unmapped_reason=how)
        elif reg_code not in self.panel_codes:
            out.update(unmapped_reason="registry_code_not_in_panel")
        else:
            out.update(panel_code=reg_code, match_method=how)
        return out


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def segment_slug(label) -> str:
    if pd.isna(label):
        return "nao_informado"
    f = fold(label)
    return SEGMENTS.get(f) or re.sub(r"[^A-Z0-9]+", "_", f).strip("_").lower()


def data_cutoff(manifest: dict, st: ReadStats) -> pd.Timestamp:
    """The first day the file holds no data for: the earlier of the server's Last-Modified and
    the day after the latest issue date in the file. Last-Modified alone overstates the
    snapshot: the 2026-09-01 vintage's latest request and issue dates are both 2026-08-24."""
    latest = st.dates[COL_ISSUE]["max"].normalize() + pd.Timedelta(days=1)
    if not manifest.get("last_modified"):
        return latest
    lm = parsedate_to_datetime(manifest["last_modified"]).astimezone(timezone.utc)
    return min(pd.Timestamp(lm.replace(tzinfo=None)).normalize(), latest)


def partial_month(year: pd.Series, month: pd.Series, cutoff: pd.Timestamp) -> np.ndarray:
    """True for a request month whose end, plus ISSUE_LAG_DAYS, lies after the data cutoff."""
    y, m = year.astype("int64"), month.astype("int64")
    next_month = pd.to_datetime(pd.DataFrame({"year": y + (m == 12), "month": m % 12 + 1,
                                              "day": 1}))
    return np.asarray(next_month + pd.Timedelta(days=ISSUE_LAG_DAYS) > cutoff)


def monthly_table(lines: pd.DataFrame, spans: pd.DataFrame, present: pd.MultiIndex,
                  cutoff: pd.Timestamp) -> pd.DataFrame:
    keys = ["panel_code", "year", "month"]
    per_crt = lines.drop_duplicates(["panel_code", COL_CRT]).copy()
    per_crt["seg"] = per_crt[COL_SEG].map(segment_slug)
    per_crt["versoes"] = pd.to_numeric(per_crt[COL_VERS], errors="coerce")
    base = (per_crt.groupby(keys)
                   .agg(n_crt=(COL_CRT, "nunique"),
                        sum_qtd_versoes=("versoes", "sum"),
                        n_crt_versoes_reported=("versoes", "count")))
    titles = lines.groupby(keys)["title_key"].nunique().rename("n_titles_distinct")
    seg = (per_crt.pivot_table(index=keys, columns="seg", values=COL_CRT, aggfunc="nunique",
                               fill_value=0)
                  .reindex(columns=sorted(set(SEGMENTS.values()) | set(per_crt["seg"])),
                           fill_value=0)
                  .add_prefix("n_crt_seg_"))
    obs = base.join(titles).join(seg)

    last = int(lines["year"].max()) * 12 + int(lines.loc[lines["year"] == lines["year"].max(),
                                                         "month"].max()) - 1
    months = [(m // 12, m % 12 + 1) for m in range(2013 * 12, last + 1)]
    grid = pd.MultiIndex.from_tuples(
        [(c, y, m) for c in sorted(obs.index.get_level_values(0).unique()) for y, m in months],
        names=keys)
    out = obs.reindex(grid).fillna(0)
    early = obs.index[~obs.index.isin(grid)]
    if len(early):
        raise AssertionError(f"CRTs requested before 2013-01 fall outside the grid: {early[:5]}")
    for c in out.columns:
        if c != "sum_qtd_versoes":
            out[c] = out[c].astype("int64")
    out = out.reset_index()
    out.insert(1, "panel_name", out["panel_code"].map(spans["panel_name"]))
    # The grid runs every conglomerate over every month, including months when its code had no
    # market-panel rows; there a zero is not an observation, and in_panel says so.
    out.insert(4, "in_panel", in_panel(out["panel_code"], out["year"],
                                       (out["month"] - 1) // 3 + 1, present))
    out.insert(5, "partial_month", partial_month(out["year"], out["month"], cutoff))
    out["coverage_note"] = COVERAGE_NOTE
    return out


def advertiser_map(lines: pd.DataFrame, reg_sub: pd.DataFrame,
                   spans: pd.DataFrame) -> pd.DataFrame:
    g = (lines.groupby(["cnpj8", "panel_code", "match_method"])
              .agg(first_ym=("ym", "min"), last_ym=("ym", "max"), n_crt=(COL_CRT, "nunique"),
                   n_rows=(COL_CRT, "size"), code_conflict_rows=("code_conflict", "sum"),
                   names=(COL_ADV, lambda s: list(s.value_counts().index[:3])))
              .reset_index())
    g["first_quarter"] = g["first_ym"].map(ym_label)
    g["last_quarter"] = g["last_ym"].map(ym_label)
    reg_names = registry_names(reg_sub)
    g["advertiser_names"] = g["names"].map(" | ".join)
    g["registry_name"] = g["cnpj8"].map(lambda r: xw.registry_name(reg_sub, r))
    g["panel_name"] = g["panel_code"].map(spans["panel_name"])
    # Printed for review, never used to unmap: renames (Aymore -> Santander SCFI, SEAC ->
    # Mulvi) legitimately share no word, and the CNPJ root is the legal identity.
    g["names_agree"] = [names_agree(a, reg_names.get(r, []))
                        for a, r in zip(g["names"], g["cnpj8"])]
    return (g.drop(columns=["first_ym", "last_ym", "names"])
             .sort_values(["n_crt", "cnpj8"], ascending=[False, True]).reset_index(drop=True))


def registry_names(reg_sub: pd.DataFrame) -> dict[str, list[str]]:
    member = reg_sub["CodInst"].str.fullmatch(r"\d{1,8}", na=False)
    m = reg_sub[member]
    return (m.assign(r=m["CodInst"].str.zfill(8)).groupby("r")["NomeInstituicao"]
             .agg(lambda s: sorted(set(s.dropna()))).to_dict())


def unmatched_financial(unmapped_reg: pd.DataFrame, fin: pd.DataFrame, reg_sub: pd.DataFrame,
                        mapper: Mapper) -> pd.DataFrame:
    """Advertisers left unmapped whose ANCINE name, or registry name, looks financial."""
    names = registry_names(reg_sub)
    a = unmapped_reg.assign(in_registry=True)
    b = fin.assign(in_registry=False,
                   unmapped_reason=np.where(fin["cnpj14"].isna(), "no_cnpj",
                                            np.where(fin["cnpj8"].isna(), "invalid_cnpj",
                                                     "not_in_registry")))
    both = pd.concat([a, b], ignore_index=True)
    both["key"] = both["cnpj8"].fillna("name:" + both[COL_ADV].fillna("").map(fold))
    # A root the lists place in a conglomerate in other quarters is a supervised institution
    # whose rows were held back by the rule, not by its name, so it is queued whatever its name.
    both["held_back"] = both["unmapped_reason"].eq("no_code_in_quarter")
    g = (both.groupby("key")
             .agg(cnpj8=("cnpj8", "first"), n_crt=(COL_CRT, "nunique"),
                  held_back=("held_back", "any"),
                  n_rows=(COL_CRT, "size"), first_request=("request_date", "min"),
                  last_request=("request_date", "max"), in_registry=("in_registry", "first"),
                  unmapped_reason=("unmapped_reason",
                                   lambda s: "; ".join(f"{k} ({v})" for k, v in
                                                       s.value_counts().items())),
                  advertiser_names=(COL_ADV, lambda s: " | ".join(s.value_counts().index[:3])),
                  example_cnpj=(COL_ADV_CNPJ, "first"))
             .reset_index(drop=True))
    g["registry_name"] = g["cnpj8"].map(lambda r: "; ".join(names.get(r, [])) if pd.notna(r)
                                        else "")
    g["registry_candidates"] = g["cnpj8"].map(
        lambda r: "; ".join(f"{p.code} ({p.min}-{p.max})"
                            for p in mapper.periods.get(r, pd.DataFrame()).itertuples())
        if pd.notna(r) else "")
    ancine = g["advertiser_names"].map(fold)
    looks = ((ancine.str.contains(FINANCIAL) & ~ancine.str.contains(NOT_FINANCIAL))
             | g["registry_name"].map(fold).str.contains(FINANCIAL)
             | g["held_back"])
    both_names = ancine + " | " + g["registry_name"].map(fold)
    g["kind"] = "other"
    for kind, pattern in reversed(KINDS):          # earlier entries win
        g.loc[both_names.str.contains(pattern), "kind"] = kind
    g["brand_word"] = (ancine.str.extract(BRANDS, expand=False).fillna("")
                             .where(~ancine.str.contains(NOT_BRAND), ""))
    return (g[looks].drop(columns="held_back")
             .sort_values(["n_crt", "advertiser_names"], ascending=[False, True])
             .reset_index(drop=True))


# ---------------------------------------------------------------------------
# Validation and summary
# ---------------------------------------------------------------------------
def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate(manifest: dict, header: list[str], st: ReadStats, lines: pd.DataFrame,
             rows: pd.DataFrame, monthly: pd.DataFrame, roots: set[str],
             spans: pd.DataFrame) -> None:
    """`rows` is every row whose advertiser root is in the IF.data registry, mapped or not.
    Each check logs what it measured when it passes, so the run log is the validation record."""
    on_disk = CSV_PATH.stat().st_size
    if not (manifest["bytes"] == manifest["content_length"] == on_disk):              # check 1
        raise AssertionError(f"size mismatch: manifest {manifest['bytes']}, Content-Length "
                             f"{manifest['content_length']}, on disk {on_disk}")
    # The outputs cite the manifest's hash as their source, so it must be the file's hash.
    sha = file_sha256(CSV_PATH)
    if sha != manifest["sha256"]:
        raise AssertionError(f"SHA-256 mismatch: manifest {manifest['sha256']}, on disk {sha}")
    missing = [c for c in REQUIRED if c not in header]
    if missing:
        raise AssertionError(f"columns missing from the header: {missing}")
    log.info("check 1 passed: %d bytes on disk = manifest = Content-Length; SHA-256 %s matches "
             "the manifest; all %d required columns present", on_disk, sha, len(REQUIRED))
    if st.rows < DASHBOARD_CRTS or len(st.crts) < DASHBOARD_CRTS:                    # check 2
        raise AssertionError(f"{st.rows} rows and {len(st.crts)} distinct CRTs; ANCINE's "
                             f"dashboard counted {DASHBOARD_CRTS} by May 2025")
    if st.bad_crt:
        raise AssertionError(f"{st.bad_crt} rows whose CRT number is not 14 digits "
                             "(a delimiter or quoting problem shifts columns)")
    log.info("check 2 passed: %d rows and %d distinct CRTs >= %d; every CRT number has 14 "
             "digits", st.rows, len(st.crts), DASHBOARD_CRTS)
    shares = {}
    for col in (COL_REQ, COL_ISSUE):                                                  # check 3
        shares[col] = st.dates[col]["parsed"] / st.rows
        if shares[col] < MIN_DATE_PARSE:
            raise AssertionError(f"{col}: {shares[col]:.2%} of rows parse as dates "
                                 f"(floor {MIN_DATE_PARSE:.0%})")
    log.info("check 3 passed: dates parse on %s (floor %.0f%%)",
             ", ".join(f"{c} {s:.2%}" for c, s in shares.items()), 100 * MIN_DATE_PARSE)
    key = ["panel_code", "year", "month"]
    if monthly.duplicated(key).any():                                                 # check 4
        raise AssertionError(f"{int(monthly.duplicated(key).sum())} repeated panel_code-months")
    seg_cols = [c for c in monthly.columns if c.startswith("n_crt_seg_")]
    if not (monthly[seg_cols].sum(axis=1) == monthly["n_crt"]).all():
        raise AssertionError("segment counts do not add up to n_crt")
    log.info("check 4 passed: %d panel_code-months, none repeated; %d segment columns add up "
             "to n_crt on every row", len(monthly), len(seg_cols))
    stray = set(lines["cnpj8"]) - roots                                               # check 5
    if stray:
        raise AssertionError("mapped CNPJ8s absent from the IF.data registry: "
                             f"{sorted(stray)[:10]}")
    off_panel = set(lines["panel_code"]) - set(spans.index)
    if off_panel:
        raise AssertionError(f"panel codes absent from the market panel: {sorted(off_panel)}")
    log.info("check 5 passed: %d mapped CNPJ8s all in the IF.data registry; %d panel codes all "
             "in the market panel", lines["cnpj8"].nunique(), lines["panel_code"].nunique())
    # Tested on every row of the file that carries the anchor's root, not only on the mapped
    # lines: an anchor row that fails to map would be absent from `lines`, and a root missing
    # from the registry would be absent from `rows`; either would otherwise pass unseen.
    for cnpj8, code in xw.ANCHORS.items():
        n_raw = st.anchor_rows.get(cnpj8, 0)
        own = rows[rows["cnpj8"] == cnpj8]
        if n_raw == 0:
            if cnpj8 in REQUIRED_ANCHORS:
                raise AssertionError(f"anchor {cnpj8} ({code}) has no CRT row")
            log.info("check 5: anchor %s (%s) has no CRT row, nothing to test", cnpj8, code)
            continue
        if len(own) != n_raw:
            raise AssertionError(f"anchor {cnpj8}: {n_raw} rows in the file carry the root but "
                                 f"{len(own)} reached the mapping (root absent from the "
                                 "IF.data registry?)")
        wrong = own[own["panel_code"].fillna("") != code]
        if len(wrong):
            raise AssertionError(
                f"anchor {cnpj8}: {len(wrong)} of {len(own)} rows do not map to {code}:\n"
                + wrong.groupby(["map_quarter", "panel_code", "unmapped_reason"], dropna=False)
                       .size().to_string())
        log.info("check 5 passed: anchor %s -> %s on all %d of its rows (%d CRTs, %s)", cnpj8,
                 code, n_raw, own[COL_CRT].nunique(),
                 own["match_method"].value_counts().to_dict())
    for col in (COL_TITLE, COL_REQ, COL_SEG):                                         # check 6
        varying = lines.groupby(COL_CRT)[col].nunique(dropna=False)
        if (varying > 1).any():
            raise AssertionError(f"{col} varies within {int((varying > 1).sum())} CRTs")
    log.info("check 6 passed: title, request date and segment constant within each of the %d "
             "mapped CRTs", lines[COL_CRT].nunique())


def summary(manifest: dict, header: list[str], st: ReadStats, lines: pd.DataFrame,
            monthly: pd.DataFrame, amap: pd.DataFrame, unmatched: pd.DataFrame,
            spans: pd.DataFrame, dep_q4: pd.Series, mapper: Mapper,
            cutoff: pd.Timestamp) -> None:
    log.info("file: %s, %d bytes, Last-Modified %s, sha256 %s", manifest["url"],
             manifest["bytes"], manifest["last_modified"], manifest["sha256"][:16])
    log.info("IF.data lists with conglomerate codes %s-%s (nearest-span fallback only outside); "
             "market panel %s-%s", ym_label(mapper.reg_first), ym_label(mapper.reg_last),
             ym_label(mapper.panel_first), ym_label(mapper.panel_last))
    log.info("header (%d columns): %s", len(header), ";".join(header))
    log.info("rows %d, distinct CRTs %d (%d issued by 2025-05-31; dashboard %d)", st.rows,
             len(st.crts), len(st.crts_to_may2025), DASHBOARD_CRTS)
    for col, d in st.dates.items():
        log.info("  %-22s filled %.2f%%, parsed %.2f%%, %s to %s", col,
                 100 * d["filled"] / st.rows, 100 * d["parsed"] / st.rows,
                 d["min"].date() if d["min"] is not None else None,
                 d["max"].date() if d["max"] is not None else None)
    log.info("CNPJ Anunciante: 14 digits on %.2f%% of rows (%d), valid check digits %d; "
             "CRTs with at least one advertiser CNPJ %.2f%%; other values %s",
             100 * st.cnpj14 / st.rows, st.cnpj14, st.cnpj_valid,
             100 * len(st.crts_with_cnpj) / len(st.crts), st.cnpj_text)
    per_crt = lines.drop_duplicates(["panel_code", COL_CRT])
    log.info("mapped: %d rows, %d distinct CRTs (%.2f%% of all), %d advertiser CNPJ8s, "
             "%d conglomerates with at least one CRT", len(lines), lines[COL_CRT].nunique(),
             100 * lines[COL_CRT].nunique() / len(st.crts), lines["cnpj8"].nunique(),
             lines["panel_code"].nunique())
    log.info("match methods (rows): %s", lines["match_method"].value_counts().to_dict())
    conflict = lines[lines["code_conflict"]]
    if len(conflict):
        log.info("registry code differs from the panel code on %d rows (the panel wins by "
                 "design):\n%s", len(conflict),
                 conflict.groupby(["cnpj8", COL_ADV, "registry_code", "panel_code"])[COL_CRT]
                         .nunique().rename("n_crt").reset_index().to_string(index=False))
    off = per_crt[~per_crt["in_panel"]]
    log.info("CRTs requested in a month when their code has no market-panel rows: %d of %d "
             "code-CRT pairs, %d of them after the panel's last quarter", len(off), len(per_crt),
             int(off["outside_panel_window"].sum()))
    if len(off[~off["outside_panel_window"]]):
        log.info("  inside the panel window, by code:\n%s",
                 off[~off["outside_panel_window"]]
                 .groupby(["panel_code", "panel_name", "map_quarter"])[COL_CRT].nunique()
                 .rename("n_crt").reset_index().to_string(index=False))
    part = monthly.loc[monthly["partial_month"], ["year", "month"]].drop_duplicates()
    log.info("partial request months (data cutoff %s, issue lag %d days): %s; their CRTs: %d",
             cutoff.date(), ISSUE_LAG_DAYS,
             [f"{y}-{m:02d}" for y, m in part.itertuples(index=False)],
             int(monthly.loc[monthly["partial_month"], "n_crt"].sum()))
    # Months outside the panel are excluded, so a conglomerate is ranked only on the months it
    # can enter an estimation with.
    win = monthly[monthly["in_panel"] & monthly["year"].between(2016, 2024)]
    top = (win.groupby("panel_code")["n_crt"].sum().rename("n_crt_2016_2024")
              .sort_values(ascending=False).head(15).to_frame())
    top["in_panel_months"] = win.groupby("panel_code").size().reindex(top.index)
    top["panel_name"] = top.index.map(spans["panel_name"])
    log.info("top 15 conglomerates by CRTs requested 2016-2024, in-panel months only:\n%s",
             top.to_string())
    adv24 = set(per_crt.loc[per_crt["year"] == 2024, "panel_code"])
    total = float(dep_q4.sum())
    share = float(dep_q4[dep_q4.index.isin(adv24)].sum()) / total if total else float("nan")
    log.info("2024Q4 panel deposits (dep_a1+a2+a4+a5): %.1f%% held by the %d conglomerates with "
             "at least one CRT requested in 2024 (%d of them have 2024Q4 panel rows; %d "
             "conglomerates hold positive 2024Q4 deposits)", 100 * share, len(adv24),
             len(adv24 & set(dep_q4.index)), int((dep_q4 > 0).sum()))
    odd = amap[~amap["names_agree"]]
    if len(odd):
        log.info("mapped CNPJ8s whose ANCINE and registry names do not agree (renames or a "
                 "wrong CNPJ on the CRT; review):\n%s",
                 odd[["cnpj8", "advertiser_names", "registry_name", "panel_code", "n_crt"]]
                 .to_string(index=False))
    show = unmatched.assign(advertiser_names=unmatched["advertiser_names"].str[:60],
                            unmapped_reason=unmatched["unmapped_reason"].str[:40])
    cols = ["cnpj8", "advertiser_names", "n_crt", "kind", "brand_word", "unmapped_reason"]
    log.info("unmatched advertisers that look financial: %d (%s CRTs); by kind: %s",
             len(unmatched), f"{int(unmatched['n_crt'].sum()):,}",
             unmatched.groupby("kind")["n_crt"].agg(["size", "sum"]).to_dict("index"))
    log.info("top 40 by CRTs:\n%s", show.head(40)[cols].to_string(index=False))
    log.info("top 30 carrying a brand word:\n%s",
             show[show["brand_word"] != ""].head(30)[cols].to_string(index=False))
    log.info("monthly table: %d rows, %d conglomerates, %s to %s", len(monthly),
             monthly["panel_code"].nunique(),
             f"{monthly['year'].min()}-{monthly['month'].iloc[0]:02d}",
             f"{monthly['year'].max()}-{monthly['month'].iloc[-1]:02d}")


def write_atomic(frame: pd.DataFrame, out_dir: Path, name: str) -> None:
    for ext in ("parquet", "csv"):
        final = out_dir / f"{name}.{ext}"
        tmp = out_dir / f"{name}.{ext}.tmp"
        if ext == "parquet":
            frame.to_parquet(tmp, index=False)
        else:
            frame.to_csv(tmp, index=False)
        os.replace(tmp, final)
    log.info("%s: %d rows -> %s", name, len(frame), out_dir / f"{name}.parquet")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="ANCINE advertising-film registrations (CRT)")
    ap.add_argument("--refresh", action="store_true", help="download again even when cached")
    ap.add_argument("--download-only", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    a = ap.parse_args()

    manifest = download(a.refresh)
    if a.download_only:
        return
    enc, delim, header = sniff(CSV_PATH)

    reg = xw.load_registry()
    roots = registry_roots(reg)
    rows, fin, st = read_crt(enc, delim, roots)
    log.info("rows whose advertiser root is in the IF.data registry: %d (%d roots)", len(rows),
             rows["cnpj8"].nunique())

    spans, dep_q4, present = panel_identity()
    reg_sub = registry_subset(reg, set(rows["cnpj8"]))
    mapper = Mapper(reg_sub, spans, sorted(set(rows["cnpj8"])), registry_window(reg))
    cutoff = data_cutoff(manifest, st)

    rows["year"] = rows["request_date"].dt.year.astype("Int64")
    rows["month"] = rows["request_date"].dt.month.astype("Int64")
    rows["quarter"] = rows["request_date"].dt.quarter.astype("Int64")
    if rows["year"].isna().any():
        raise AssertionError(f"{int(rows['year'].isna().sum())} registry-advertiser rows have "
                             "no request date")
    rows["ym"] = rows["year"].astype(int) * 100 + rows["quarter"].astype(int) * 3
    pairs = rows[["cnpj8", "ym"]].drop_duplicates()
    assigned = pd.DataFrame([mapper.assign(c, int(y)) for c, y in pairs.itertuples(index=False)])
    rows = rows.merge(assigned, on=["cnpj8", "ym"], how="left", validate="many_to_one")

    lines = rows[rows["panel_code"].notna()].copy()
    lines["code_conflict"] = lines["registry_code"].notna() & (lines["registry_code"]
                                                               != lines["panel_code"])
    lines["outside_panel_window"] = ((lines["ym"] < mapper.panel_first)
                                     | (lines["ym"] > mapper.panel_last))
    lines["in_panel"] = in_panel(lines["panel_code"], lines["year"], lines["quarter"], present)
    lines["partial_month"] = partial_month(lines["year"], lines["month"], cutoff)
    lines["panel_name"] = lines["panel_code"].map(spans["panel_name"])
    lines["title_norm"] = lines[COL_TITLE].map(norm_title)
    lines["title_key"] = lines["title_norm"] + "|" + lines["cnpj8"]
    unmapped = rows[rows["panel_code"].isna()]
    log.info("registry-root rows left unmapped, by reason: %s",
             unmapped["unmapped_reason"].value_counts().to_dict())

    monthly = monthly_table(lines, spans, present, cutoff)
    amap = advertiser_map(lines, reg_sub, spans)
    unmatched = unmatched_financial(unmapped, fin, reg_sub, mapper)

    validate(manifest, header, st, lines, rows, monthly, roots, spans)
    summary(manifest, header, st, lines, monthly, amap, unmatched, spans, dep_q4, mapper, cutoff)

    a.out_dir.mkdir(parents=True, exist_ok=True)
    out_lines = lines.drop(columns=["ym", "title_key", "unmapped_reason"]).sort_values(
        ["panel_code", "request_date", COL_CRT]).reset_index(drop=True)
    tables = (("ancine_ad_films_lines", out_lines),
              ("ancine_ad_films_monthly", monthly),
              ("ancine_advertiser_map", amap),
              ("ancine_unmatched_financial", unmatched))
    # The previous run's sidecar goes first and the new one is written last, so a sidecar on
    # disk always describes four tables written in full by the same run. validate() has
    # checked the file's SHA-256 against the manifest's.
    sidecar = a.out_dir / "ancine_ad_films_provenance.json"
    sidecar.unlink(missing_ok=True)
    for name, frame in tables:
        write_atomic(frame, a.out_dir, name)
    provenance = {"source_url": manifest["url"],
                  "source_last_modified": manifest.get("last_modified"),
                  "source_sha256": manifest["sha256"],
                  "source_bytes": manifest["bytes"],
                  "source_retrieved_at": manifest.get("retrieved_at"),
                  "source_file": str(CSV_PATH),
                  "data_cutoff": str(cutoff.date()),
                  "issue_lag_days": ISSUE_LAG_DAYS,
                  "registry_coded_lists": [mapper.reg_first, mapper.reg_last],
                  "market_panel": str(MARKET_PANEL),
                  "market_panel_quarters": [ym_label(mapper.panel_first),
                                            ym_label(mapper.panel_last)],
                  "tables": {name: len(frame) for name, frame in tables},
                  "written_at": now_iso(),
                  "script": Path(__file__).name}
    write_json_atomic(sidecar, provenance)
    log.info("%s -> %s", sidecar.name, a.out_dir)


if __name__ == "__main__":
    main()
