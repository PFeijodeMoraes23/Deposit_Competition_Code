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
    segment. `n_titles_distinct` counts a (normalised title, advertiser) pair once per month,
    which undoes that multiplication.

THE FILE
--------
  https://dados.ancine.gov.br/dados-abertos/crt-obras-publicitarias.csv, refreshed monthly, no key.
  Measured on the 2026-09-01 vintage (313,757,397 bytes): UTF-8 without BOM, ';'-delimited,
  25 columns, 562,074 rows, 537,163 distinct CRT numbers. The published field list omits three
  columns the file carries (CRTS_VERSOES, REGISTRO_ANCINE_REQUERENTE, CNPJ_REQUERENTE). A CRT
  appears on several rows (14,941 CRTs do) when it names several producers, directors, agencies
  or advertisers. Measured over the whole file, the rows of one CRT differ in PRODUTOR, DIRETOR,
  AGENCIA, CNPJ_AGENCIA, ANUNCIANTE and CNPJ_ANUNCIANTE, and for a single CRT in the five
  requester columns; the other 13 columns (title, product, the three dates, status, segment,
  version count and duration among them) never vary within a CRT. Every count here is therefore
  of DISTINCT CRT numbers, never rows. Check 6 below re-tests title, request date and segment on
  the mapped CRTs on every run. The last request month is incomplete: this vintage's latest
  request and issue dates are both 2026-08-24, a week before its Last-Modified (2026-09-01).
  The encoding and delimiter are detected from the file's first bytes on every run and the
  whole file is decoded strictly, so a change of either stops the run instead of mis-reading.
  Every CRT status is counted (EXPIRADO, REGISTRADO, IRREGULAR, CADASTRADO); SITUACAO_CRT stays
  on the lines table for a consumer who wants to drop some. The file holds 494,885 distinct CRTs
  issued by 2025-05-31, 1.2% fewer than the 500,663 on ANCINE's dashboard for the same window;
  the gate below therefore tests the whole file against that figure, not the window.

MAPPING ADVERTISERS TO CONGLOMERATES (attribution 'own')
--------------------------------------------------------
CNPJ Anunciante is reduced to 14 digits (placeholders such as 'PESSOA FISICA' and 'NAO
INFORMADO' carry no CNPJ and are never zero-filled into one) and then to its 8-digit root. The
root is matched against the IF.data lists with build_advertising_crosswalk's own registry
loader and candidate rule, including its guard that only digit-bearing CNPJ fields are
compared: Banco do Brasil's root is 00000000, so a zero-filled blank would match it. For each
CRT the quarter of its REQUEST date decides the code, in this order:
  panel_leader       the root is the leader CNPJ of a market-panel conglomerate whose first-to-
                     last panel span contains that quarter, and the IF.data lists confirm that
                     leadership. The span is tested, not the per-quarter presence: 20 codes have
                     holes in their span, and a CRT requested in a hole keeps the code with
                     `in_panel` False (testing presence instead would move those rows to the
                     registry step, so the span is kept to leave every mapping as it is). The
                     panel's own identity wins over a later registry move: the registry folds
                     Banco Pan into BTG's conglomerate from 2021Q2 and PicPay into Original's in
                     2019Q4-2024Q3, while the market panel keeps both as their own entities with
                     their own deposits. `registry_code`, `registry_reason` and `code_conflict`
                     keep the disagreement visible; code_conflict is also True where the lists
                     give the root NO code in a quarter inside their window.
  registry_quarter   the IF.data conglomerate code whose dated span contains the quarter
  registry_nearest   the quarter lies before the first IF.data list that carries conglomerate
                     codes (201403) or after the last list (202512), where the lists cannot
                     speak. The nearest span is used only when it reaches that edge: before the
                     first list, a span that starts AT the first list; after the last list, a
                     span that ends AT the last list. A root whose nearest span starts later (or
                     ends earlier) had no code at the edge the lists do speak for, so it gets
                     `no_code_in_quarter` instead of a membership it did not yet (or no longer)
                     have. Inside the window a root with no span covering the quarter had no
                     conglomerate code then; its rows stay unmapped with `no_code_in_quarter`.
A code is kept only when it is a market-panel code. Quarters outside the panel window are
evaluated at the window's nearest edge (`map_quarter` records the quarter actually used). Both
window edges are read from the data on every run and logged.

ATTRIBUTION BEYOND THE ADVERTISER'S OWN CNPJ
--------------------------------------------
Films that promote a bank are also registered by firms outside its prudential conglomerate.
Rows that did not map as 'own' are attributed by the rules below; each rule maps through the
bank's own CNPJ root with the same Mapper, so the code follows the registry and the panel
quarter by quarter exactly as an own row would.
  affiliate        the advertiser is a bank-owned insurer or other bank affiliate in no
                   prudential conglomerate, listed in AFFILIATES by CNPJ root. It qualifies when
                   it carries the bank's brand or the bank's group controls it; `attach_basis`
                   records which (brand, control, brand_and_control) and `evidence` the source
                   URL or registry fact. Brand decides where the two differ, because awareness is
                   of the brand: Caixa Seguradora is attached to Caixa though CNP controls it
                   (and only while it sold under the Caixa brand), Itau Seguros de Auto e
                   Residencia to Itau though Porto controls it. Every CRT of the affiliate counts.
                   A registry root in the table (Redecard, Banco Bradesco Cartoes) is attributed
                   only in the quarters it maps to no code; its other rows stay own.
                   Foundations and institutes (Fundacao Bradesco, Fundacao Itau, Instituto
                   Porto Seguro, Instituto Banese) are attached with naming_rights_suspect True.
  holding          the advertiser is the bank's parent holding (HOLDINGS: J&F -> PicPay and
                   Original, C6 Holding -> C6, UOL -> PagSeguro, Itausa -> Itau, Votorantim ->
                   BV, Americanas -> Ame, Porto Seguro S.A. -> Porto, Banco Santander S.A. ->
                   Santander). Only CRTs whose product or title names that bank count: a holding
                   also advertises its other businesses (JBS, Shoptime, cement).
  media_sponsored  the advertiser is a broadcaster, media or production company (MEDIA, by
                   name) and the product or title names a bank. The bank paid the media for the
                   content, so it is the bank's advertising. `naming_rights_suspect` marks CRTs
                   whose product or title is an event, venue or institute name (EVENT_VENUE:
                   'COPA SANTANDER LIBERTADORES', 'CAIXA CULTURAL', 'FEIRAO DA CAIXA', 'ARENA
                   BANCO ORIGINAL'), where the brand may be only a sponsor's naming right.
  coop_system      the advertiser is a credit cooperative (single, central or confederation, by
                   its ANCINE or IF.data name) or a system company, and the CRT is Sicoob- or
                   Sicredi-branded. The system is decided per CRT from the most specific
                   evidence: the system its film names; else the one the advertiser's IF.data
                   name carries in that quarter; else the one its ANCINE name carries (ANCINE
                   keeps one name per firm, undated); else the single system all its films and
                   names carry. Cooperatives change systems (UNICRED CEARA CENTRO NORTE is SICREDI
                   in the lists from 2016), so each film keeps the system of its time, and an
                   advertiser whose evidence points two ways gets none. Sicoob CRTs go to BANCO SICOOB (C0080879), Sicredi CRTs to
                   BANCO COOPERATIVO SICREDI (C0080745), the only two cooperative banks in the
                   market panel. A media company's film naming Sicoob or Sicredi is coop_system
                   too: the payer is a cooperative, not the bank. This class is kept OUT of the
                   bank total. Cooperatives of other systems (Cresol, Unicred, Ailos) stay
                   unattached with their system recorded in the review queue.
Not attributed:
  joint ventures   multi-bank firms (Elo, Livelo, Alelo, Cielo) stay with no bank; their
                   verified owners are recorded (JOINT_VENTURES) in the advertiser map and the
                   review queue. Alelo and Cielo keep their own panel codes where the registry
                   gives them one.
  other filers     a filer that is neither the bank, its affiliate or holding, a media company
                   nor a cooperative of the bank's system (a retailer's co-branded card, a car
                   dealer's financing fair, an advertising agency, a public body, an employee
                   association) is listed for review as `other_filer_named_bank` and counts
                   nowhere; `filer_type` says which kind it is.
Brand words are matched in the folded product and title (BANK_BRANDS). Words that are also
ordinary words or other firms' names match only next to a word that makes them the bank: SAFRA
is a harvest ('PLANO SAFRA', 'PROMO SAFRA') unless 'BANCO SAFRA', 'SAFRAPAY' or 'J SAFRA'; INTER,
PAN, NEON, XP, ORIGINAL, C6, BRB, BMG, BV, NU, CAIXA, BB, NEXT, STONE and CIELO likewise
('BANCO INTER', 'BANCO PAN', 'XP INVESTIMENTOS', 'C6 BANK', 'CAIXA ECONOMICA'); ITAU does not
match the town (ITAU DE MINAS), the mall (ITAU POWER) or the coffee (CAFE ITAU). A film that
names its own advertiser's root is not re-attributed.
Each (panel code, CRT) is counted once, under the first class it has in the order own,
affiliate, holding, media_sponsored, coop_system (`counted_as`).

OUTPUTS (paths.AWARENESS_PROC, parquet + csv, written atomically)
-----------------------------------------------------------------
  ancine_ad_films_lines        every CRT row attributed to a panel conglomerate, all 25 original
                               fields as published plus cnpj14, cnpj_valid, cnpj8,
                               advertiser_key, the parsed request and issue dates, year/month/
                               quarter of the request, map_quarter, registry_code,
                               registry_reason, panel_code, match_method, code_conflict,
                               outside_panel_window, in_panel, partial_month, panel_name,
                               title_norm, and the attribution: `attribution` (own, affiliate,
                               holding, media_sponsored, coop_system), `counted_as`,
                               `target_cnpj8` (the root the code was mapped through),
                               `bank_named`, `filer_type`, `attach_basis`, `evidence`,
                               `naming_rights_suspect`. A row attributed to two banks appears
                               once per bank.
  ancine_ad_films_monthly      panel_code x year x month of the REQUEST date, balanced over
                               2013-01 to the last request month for every conglomerate with at
                               least one CRT: n_crt_own, n_crt_affiliate, n_crt_holding,
                               n_crt_media_sponsored, n_crt (their sum, the bank total),
                               n_crt_naming_rights_suspect (CRTs inside n_crt flagged as
                               possible naming rights), n_crt_coop_system (NOT in n_crt), and,
                               over the CRTs in n_crt: n_titles_distinct, n_crt_seg_<segment>,
                               sum_qtd_versoes (over CRTs that report a version count),
                               n_crt_versoes_reported; then coverage_note and two flags:
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
  ancine_advertiser_map        advertiser_key -> panel_code, attribution, match_method, first/
                               last quarter, CRTs (and CRTs counted under that attribution),
                               names on both sides, `names_agree` (own rows only; a review flag:
                               renames such as Aymore -> Santander SCFI share no word, and the
                               CNPJ root, not the name, is the legal identity), attach_basis,
                               evidence and `jv_owners` for joint ventures that map to their own
                               code
  ancine_brand_attributions    review file: every row attributed by product/title or by
                               cooperative system (holding, media_sponsored, coop_system) and
                               every other_filer_named_bank, with advertiser, filer type,
                               product, title, the bank named, the code (for
                               other_filer_named_bank, the code the named bank maps to; it
                               counts nowhere), class, counted_as and naming_rights_suspect
  ancine_unmatched_financial   advertisers that did not map as own and whose ANCINE or registry
                               name looks financial (or whose reason is no_code_in_quarter), with
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
                                                             request quarter (inside the lists'
                                                             window, or beyond an edge its span
                                                             does not reach); always queued,
                                                             whatever its name
                                 registry_code_not_in_panel  its conglomerate is not a market-
                                                             panel code (BNDES, no deposits)
                                 ambiguous                   overlapping registry spans, or two
                                                             spans equally near
                                 ambiguous_panel_leader      two panel codes led by the same root
                                 no_cnpj, invalid_cnpj       placeholder or failed check digits
                               plus `kind` and `brand_word`, review aids that sort the queue,
                               `n_crt_attributed` and `attributed_as` (CRTs the attribution rules
                               took, and where), `jv_owners`/`jv_evidence` and `coop_system`
  ancine_ad_films_provenance.json   sidecar: the source file's URL, source_last_modified,
                               source_sha256 (re-verified against the file on disk), size and
                               retrieval time, the data cutoff and the registry and panel
                               windows the run used, and each table's row count
The download is cached under paths.AWARENESS_RAW/ancine with a manifest (URL, status, bytes,
Content-Length, SHA-256, Last-Modified, retrieval time, file name, final URL and redirect hops).
It follows a redirect only to https://dados.ancine.gov.br on the default port; any other target
stops the download before that target is contacted.

VALIDATION (any failure aborts before writing)
----------------------------------------------
  1. The file on disk has exactly the server's Content-Length and the manifest's SHA-256.
  2. At least 500,663 rows and 500,663 distinct CRTs (ANCINE's dashboard count of advertising
     CRTs issued 2013 to May 2025); every CRT number has 14 digits.
  3. At least 95% of the request and the issue dates parse.
  4. No panel_code-month appears twice in the monthly table; segment counts add up to n_crt;
     n_crt is the sum of the four bank classes; the naming-rights count fits inside the
     non-own part of n_crt.
  5. Every CNPJ8 a line was mapped through (the own root, or the bank root an attribution
     names) exists in the IF.data registry; every panel_code exists in the market panel; for
     each of the crosswalk's anchors (Banco do Brasil, Caixa, Itau holding, Nu) whose root is on
     any row of the file (a valid CNPJ, counted before the registry filter), EVERY such row maps
     to its code as own (an unmapped row, or one that never reached the mapping, fails too),
     and Banco do Brasil, Caixa and Nu must have rows, so the check cannot pass by finding
     nothing.
  6. Title, request date and segment never vary within a mapped CRT.
  7. Attribution: every class is one of the five; affiliate and holding rows come from filers in
     AFFILIATES and HOLDINGS, a holding row names one of its holding's banks, media rows come
     from media filers, no joint venture is attributed, and each (panel code, CRT) has exactly
     one counted_as.
Each check logs what it measured when it passes.

Usage
-----
  python scrape_ancine_ad_films.py                  # uses the cached download
  python scrape_ancine_ad_films.py --refresh        # downloads the current vintage first
  python scrape_ancine_ad_films.py --download-only
  python scrape_ancine_ad_films.py --out-dir <dir>  # writes the five tables and the
                                                    # provenance file elsewhere
"""

from __future__ import annotations

import argparse
import codecs
import contextlib
import csv
import hashlib
import io
import json
import logging
import os
import re
import time
import unicodedata
import urllib.parse
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
ALLOWED_HOST = "dados.ancine.gov.br"
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
# A CRT enters the file when it is issued, which can be days after its request: of the 35,643
# CRTs requested in 2025, 10.9% were issued on or after the 1st of the following month and 3.3%
# on or after its 11th, i.e. more than this many days after the request month ended. A request
# month therefore counts as complete only once this many days past its end fall inside the
# file's data; the 3.3% is what a month so marked can still gain in a later vintage.
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
COL_PROD = "PRODUTO_SERVICO_ANUNCIADO"
REQUIRED = [COL_TITLE, COL_CRT, COL_REQ, COL_ISSUE, COL_VALID, COL_SEG, COL_VERS, COL_ADV,
            COL_ADV_CNPJ, COL_PROD]

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
# names the bank or bank-owned brand a name carries (BRADESCO SEGUROS, CAIXA SEGURADORA, ELO,
# LIVELO, B3), which is the decision the user has to make: whether such a firm's films count for
# a bank. It is a whole-word match, so short brands (ELO, VERO, BOLSA, B3) also tag unrelated
# firms that share the word; NOT_BRAND removes the tag from the ones that say what they are.
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
# and PAN start many company names, ELO and B3 name agencies). A name that also says it is a
# builder, a farm, a food maker, an industry, an advertising or communication agency, a
# pharmacy, an estate agent or an optician is such a firm, so it gets no brand_word; it stays in
# the queue. The names are accent-folded before the test (COMUNICACAO for COMUNICAÇÃO, OTICA
# for ÓTICA); opticians also spell it OPTICA/OPTICOS (the ELO optics shop does).
NOT_BRAND = re.compile(r"\b(?:INCORPORACAO|INCORPORACOES|CONSTRUTORA|AGROPECUARIA|ALIMENTICIOS|"
                       r"INDUSTRIA|INDUSTRIAS|FERTILIZANTES|COMUNICACAO|PUBLICIDADE|PROPAGANDA|"
                       r"DROGARIA|IMOVEIS|OTICA|OTICAS|OPTICA|OPTICAS|OPTICOS)\b")

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
# Attribution tables
# ---------------------------------------------------------------------------
# Bank brands a film's product or title can name: label, the CNPJ8 of the institution whose code
# the brand stands for, and the pattern on the folded text (brand_text). The code is found by
# mapping that root with Mapper, so a brand follows its bank's registry and panel history.
# Bare words that are also ordinary words or other firms' names match only beside a word that
# makes them the bank: SAFRA is a harvest ('PLANO SAFRA', 'OURO SAFRA' seeds, 'SAFRA COTRISAL'),
# PAN the Pan American games, CAIXA a box or a till ('NO CAIXA', 'CAIXA DE SOM'), C6 a GloboNews
# programme code ('GNEWS C6'), NEXT, INTER, ORIGINAL, BV, STONE and CIELO the start of many firm
# names, and BB, NU and XP letters in any title. ITAU excludes the town (ITAU DE MINAS), the mall
# (ITAU POWER) and a coffee brand (CAFE ITAU).
BANK_BRANDS = (
    ("Bradesco", "60746948", r"\bBRADESCO\b|\bBANCO NEXT\b|\bNEXT BANK\b|\b(?:CONTA|APP) NEXT\b"),
    ("Itau", "60701190", r"(?<!\bCAFE )\bITAU\b(?! DE MINAS\b| POWER\b)|\bITAUCARD\b|"
                         r"\bUNIBANCO\b|\bHIPERCARD\b|\bCREDICARD\b|\bREDECARD\b|\bPERSONNALITE\b"),
    ("Santander", "90400888", r"\bSANTANDER\b"),
    ("Caixa", "00360305", r"\bCAIXA (?:ECONOMICA|FEDERAL|SEGURADORA|SEGUROS|SEGURIDADE|VIDA|"
                          r"CAPITALIZACAO|CONSORCIO\w*|TEM|AQUI|CULTURAL|CARTO\w*|ASSET|"
                          r"RESIDENCIAL|POUPANCA|HABITACAO|PRA ELAS)\b|\bCAIXAPRAELAS|"
                          r"\bFEIRAO (?:DA )?CAIXA\b|\bLOTERIAS? (?:DA )?CAIXA\b|"
                          r"\b(?:CARTAO|APP|BANCO|CONTA|POUPANCA|FINANCIAMENTO|CREDITO|"
                          r"CONSORCIO|SEGURO) (?:DA )?CAIXA\b"),
    ("Banco do Brasil", "00000000", r"\bBANCO DO BRASIL\b|\bBB (?:SEGUROS|SEGURIDADE|MAPFRE|DTVM|"
                                    r"ASSET|CONSORCIO\w*|PREVIDENCIA|CARTO\w*|CREDITO|"
                                    r"INVESTIMENTOS|AGRO|PAY|CAPITALIZACAO|CONTA|APP)\b|"
                                    r"\b(?:CARTAO|APP|CONTA) BB\b|\bOUROCARD\b|\bOUROCAP\b|"
                                    r"\bBRASILPREV\b|\bBRASILCAP\b|\bBRASILSEG\b|\bCCBB\b"),
    ("Nubank", "18236120", r"\bNUBANK\b|\bNU (?:PAGAMENTOS|BANK|CONTA|INVEST|CARTAO)\b|"
                           r"\bNUCONTA\b|\bNUINVEST\b|\b(?:CARTAO|CONTA|APP) (?:DO )?NU\b"),
    ("Inter", "00416968", r"\bBANCO INTER\b|\bINTERMEDIUM\b|\bINTER ?& ?CO\b|"
                          r"\bINTER (?:BANK|INVEST\w*|SHOP|PAG|CONTA)\b|"
                          r"\b(?:CONTA|APP|CARTAO) (?:DO )?INTER\b"),
    ("C6", "31872495", r"\bC6 (?:BANK|CARBON|CONTA|INVEST\w*|PAY|FEST|CARTAO)\b|\bBANCO C6\b|"
                       r"\b(?:CONTA|CARTAO|APP) C6\b"),
    ("PicPay", "22896431", r"\bPICPAY\b|\bPIC PAY\b"),
    ("BTG", "30306294", r"\bBTG PACTUAL\b|\bBANCO BTG\b|\bBTG ?\+|\bBTG (?:BANK|INVEST\w*)\b"),
    ("Safra", "58160789", r"\bBANCO SAFRA\b|\bSAFRA ?PAY\b|\bSAFRA (?:NATIONAL|INVEST\w*|WEALTH|"
                          r"ASSET|BANK)\b|\bJ ?SAFRA\b|\b(?:CONTA|CARTAO|APP) SAFRA\b"),
    ("XP", "02332886", r"\bXP (?:INVESTIMENTOS|INVEST|INC|CORRETORA|BANK|CARTAO|VISA)\b|"
                       r"\bBANCO XP\b|\b(?:CARTAO|CONTA|APP) XP\b"),
    ("Banrisul", "92702067", r"\bBANRISUL\b|\bBANRICOMPRAS\b"),
    ("BMG", "61186680", r"\bBANCO BMG\b|\bBMG (?:CARD|BANK|CARTAO|ARMOR|CONSIGNADO)\b|"
                        r"\b(?:CARTAO|CONTA|APP) BMG\b"),
    ("BV", "59588111", r"\bBANCO VOTORANTIM\b|\bBANCO BV\b|\bBV FINANCEIRA\b|"
                       r"\b(?:CONTA|CARTAO|APP) BV\b"),
    ("Mercado Pago", "10573521", r"\bMERCADO ?PAGO\b"),
    ("PagSeguro", "08561701", r"\bPAGSEGURO\b|\bPAGBANK\b|\bPAG SEGURO\b"),
    ("Neon", "20855875", r"\bNEON (?:PAGAMENTOS|BANK)\b|\bBANCO NEON\b|"
                         r"\b(?:CONTA|CARTAO|APP) NEON\b"),
    ("Agibank", "10664513", r"\bAGIBANK\b|\bAGIPLAN\b"),
    ("Original", "92894922", r"\bBANCO ORIGINAL\b"),
    ("Pan", "59285411", r"\bBANCO PAN\b|\bBANCOPAN\b|\bBANCO PANAMERICANO\b"),
    ("Banese", "13009717", r"\bBANESE\b|\bBANESCARD\b"),
    ("Banestes", "28127603", r"\bBANESTES\b"),
    ("BRB", "00000208", r"\bBANCO BRB\b|\bBANCO DE BRASILIA\b|\bBRB (?:BANCO|CARD|CARTAO|"
                        r"MOBILIDADE|CONTA|SEGUROS|CORRETORA|FINANCEIRA|NACAO|FLA|PAY)\b|"
                        r"\b(?:CARTAO|CONTA|APP|ARENA|NACAO) BRB\b"),
    ("Banpara", "04913711", r"\bBANPARA\b"),
    ("BASA", "04902979", r"\bBANCO DA AMAZONIA\b"),
    ("BNB", "07237373", r"\bBANCO DO NORDESTE\b|\bCREDIAMIGO\b"),
    ("Daycoval", "62232889", r"\bDAYCOVAL\b"),
    ("Crefisa", "60779196", r"\bCREFISA\b"),
    ("Digio", "27098060", r"\bDIGIO\b"),
    ("Citibank", "33479023", r"\bCITIBANK\b"),
    ("Mercantil", "17184037", r"\bBANCO MERCANTIL\b"),
    ("Cielo", "01027058", r"\bCIELO (?:LIO|PAGAMENTOS|MAQUININHA)\b|\bMAQUININHA (?:DA )?CIELO\b"),
    ("Stone", "16501555", r"\bSTONE (?:PAGAMENTOS|CO|MAQUININHA|CONTA)\b|\bSTONECO\b|"
                          r"\bMAQUININHA (?:DA )?STONE\b"),
    ("Getnet", "10440482", r"\bGETNET\b"),
    ("SumUp", "16668076", r"\bSUMUP\b"),
    ("Ame", "32778350", r"\bAME DIGITAL\b"),
    ("Will", "36272465", r"\bWILL BANK\b|\bBANCO WILL\b"),
    ("Porto", "04862600", r"\bPORTO BANK\b|\bPORTOSEG\b|\bCARTAO PORTO\b|"
                          r"\bPORTO SEGURO (?:CARTAO|CARTOES|BANK|CONSORCIO\w*|SEGUROS?|AUTO|"
                          r"SAUDE|CAPITALIZACAO|RESIDENCIA|VIDA|CIA|COMPANHIA)\b"),
)
# Cooperative systems. Sicoob and Sicredi map to their banks (the market panel's only two
# cooperative banks) as coop_system; the others have no panel bank and stay unattached.
COOP_SYSTEMS = (("Sicoob", "02038232", r"\bSICOO+B\w*|\bBANCOOB\b|\bSIPAG\b"),
                ("Sicredi", "01181521", r"\bSICREDI\b|\bSICRED\b"),
                ("Cresol", None, r"\bCRESOL\b"),
                ("Unicred", None, r"\bUNICRED\w*"),
                ("Ailos", None, r"\bAILOS\b|\bCECRED\b"))
ATTACHED_SYSTEMS = {lab for lab, root, _ in COOP_SYSTEMS if root}
BRAND_RX = tuple((lab, root, re.compile(p)) for lab, root, p in BANK_BRANDS + COOP_SYSTEMS)
BRAND_ROOT = {lab: root for lab, root, _ in BANK_BRANDS + COOP_SYSTEMS}
BRAND_ANY = re.compile("|".join(f"(?:{p})" for _, _, p in BANK_BRANDS + COOP_SYSTEMS))
SYSTEM_RX = tuple((lab, re.compile(p)) for lab, _, p in COOP_SYSTEMS)

# A credit cooperative, by the words its legal name uses (singles, centrals, confederations),
# or by a system name in it. Medical and farm cooperatives say COOPERATIVA too but not CREDITO.
# Tested on the name with punctuation turned to spaces (name_words), because ANCINE abbreviates
# ('COOP. ECON. E CRED. MUTUO').
CREDIT_COOP = re.compile(
    r"\bCOOP\w* (?:DE |DOS? |DAS? )?(?:\w+ ){0,4}?(?:CREDITO|CRED|ECONOMIA E CREDITO|POUPANCA)\b|"
    r"\bCENTRAL DAS COOPERATIVAS DE (?:ECONOMIA E )?CREDITO\b|"
    r"\bCONFEDERACAO\b.*\bCOOPERATIVAS\b|\bCCPI\b|\bCREDICOOP\b|"
    r"\bSICOO+B\b|\bSICREDI\b|\bSICRED\b|\bCRESOL\b|\bUNICRED\w*|\bAILOS\b")
# Broadcasters, media groups and audiovisual or event producers, by name. An advertising agency
# is not media (AGENCY): the user decides on agencies from the review file.
MEDIA = re.compile(
    r"\b(?:TELEVISAO|TELEVISOES|TV|TVS|TVSBT|RADIO|RADIOS|RADIODIFUSAO|RADIODIFUSORA|EMISSORA|"
    r"EMISSORAS|PROGRAMADORA|CHANNELS?|NETWORKS?|BROADCAST\w*|CANAL|CANAIS|MIDIA|MEDIA|JORNAL|"
    r"EDITORA|PRODUCOES|PRODUTORA|FILMES|FILMS|CINEMATOGRAFICA|AUDIOVISUAL|ENTRETENIMENTO|"
    r"ESPETACULOS|EVENTOS|STUDIOS?|ESTUDIOS?)\b|"
    r"\bPRODUCAO (?:E |DE )?(?:\w+ )?(?:EVENTOS|AUDIOVISUAL|ARTISTICA|CULTURAL|FILMES|VIDEO)\b|"
    r"\bGLOBO COMUNICACAO\b|\bGLOBOSAT\b|\bINFOGLOBO\b|\bENDEMOL\b|\bTFCF\b|"
    r"\bRBS PARTICIPACOES\b|\bFUNDACAO ROBERTO MARINHO\b|\bSISTEMA MASSA\b")
AGENCY = re.compile(r"\b(?:PUBLICIDADE|PROPAGANDA|PUBLICITARIA|AGENCIA|MARKETING|COMUNICACAO|"
                    r"COMUNICACOES)\b")
# Public bodies say COMUNICACAO too ('SECRETARIA DE COMUNICACAO SOCIAL'); they are neither
# media nor agencies.
PUBLIC_BODY = re.compile(r"\b(?:MUNICIPIO|PREFEITURA|SECRETARIA|GOVERNO|MINISTERIO|TRIBUNAL|"
                         r"ASSEMBLEIA LEGISLATIVA|CAMARA MUNICIPAL)\b")
# Event, venue and institute words: where one is in the product or title of a sponsored film,
# the bank's name may be only a naming right (COPA SANTANDER LIBERTADORES, CAIXA CULTURAL,
# FEIRAO DA CAIXA, ARENA BANCO ORIGINAL, BRADESCO ESPORTES FM).
EVENT_VENUE = re.compile(
    r"\b(?:COPA|LIGA|LALIGA|CAMPEONATO|TORNEIO|TROFEU|TACA|CIRCUITO|CORRIDA|MARATONA|IRONMAN|"
    r"TRIATHLON|RALLY|TOUR|TURNE|FESTIVAL|FEST|FESTA|FEIRA|FEIRAO|SALAO|EXPO\w*|EVENTO|EVENTOS|"
    r"SHOW|SHOWS|CONCERTO|SINFONIA|ORQUESTRA|TEATRO|CINEMA|CINEMAS|ESPACO|ARENA|ESTADIO|HALL|"
    r"CULTURAL|MUSEU|EXPOSICAO|INSTITUTO|FUNDACAO|PREMIO|OLIMPIADA|RODEIO|CARNAVAL|FM)\b")

# Bank-owned insurers and other bank affiliates outside any prudential conglomerate, by CNPJ
# root (or 'name:' + folded name where the CRT carries no CNPJ). bank/root: the bank the films
# count for and its CNPJ8; basis: brand, control or brand_and_control; institute: a foundation
# or institute whose films are about itself, flagged naming_rights_suspect; until: the last
# request quarter (YYYYMM) the brand applies to.
_BRADESCO_GROUP = ("Grupo Bradesco Seguros companies (Bradesco Seguros, Vida e Previdencia, Saude, "
                   "Capitalizacao, Auto/RE), subsidiaries of Banco Bradesco: "
                   "https://www.bradescoseguros.com.br/clientes/institucional/empresas-do-grupo ; "
                   "https://en.wikipedia.org/wiki/Bradesco_Seguros")
_BB_SEGURIDADE = ("BB Seguridade (controlled by Banco do Brasil) holds it through BB Seguros: "
                  "https://www.bbseguridaderi.com.br/en/bb-security/corporate-structure/")
_PORTO = ("Porto Seguro S.A. controls Porto Seguro Cia de Seguros Gerais and, through Porto Bank, "
          "Portoseg (the panel's Porto conglomerate): https://en.wikipedia.org/wiki/"
          "Porto_Seguro_S.A. ; https://ri.portoseguro.com.br/en/the-company/"
          "corporate-presentation/")
AFFILIATES = {
    "33055146": dict(bank="Bradesco", root="60746948", basis="brand_and_control",
                     evidence=_BRADESCO_GROUP),
    "92682038": dict(bank="Bradesco", root="60746948", basis="brand_and_control",
                     evidence=_BRADESCO_GROUP),
    "92693118": dict(bank="Bradesco", root="60746948", basis="brand_and_control",
                     evidence=_BRADESCO_GROUP),
    "51990695": dict(bank="Bradesco", root="60746948", basis="brand_and_control",
                     evidence=_BRADESCO_GROUP),
    "33010851": dict(bank="Bradesco", root="60746948", basis="brand_and_control",
                     evidence=_BRADESCO_GROUP),
    "60701521": dict(bank="Bradesco", root="60746948", basis="brand", institute=True,
                     evidence="Fundacao Bradesco, part of Bradesco's controlling group (it holds "
                              "Bradesco shares directly and through Cidade de Deus and NCF): "
                              "https://www.bradescori.com.br/en/bradesco/corporate-governance/"
                              "ownership-structure/"),
    "15011336": dict(bank="Bradesco", root="60746948", basis="brand_and_control",
                     evidence="next, Bradesco's digital bank (brand), a Bradesco Organization "
                              "company: https://pt.wikipedia.org/wiki/Next_(fintech) ; "
                              "https://next.me/sobre-nos"),
    "59438325": dict(bank="Bradesco", root="60746948", basis="brand_and_control",
                     evidence="IF.data lists Banco Bradesco Cartoes in Bradesco's conglomerate "
                              "in the quarters around the one it has no code in"),
    # A registry root: only its rows that map to no code (2013-2016, before the lists put it
    # in Itau's conglomerate in 201703) are attributed here; its later rows are own.
    "01425787": dict(bank="Itau", root="60701190", basis="control",
                     evidence="Redecard, 98% Itau Unibanco after the 2012 tender offer, delisted "
                              "October 2012: https://exame.com/invest/mercados/itau-investiu-r-11-"
                              "3-bi-na-compra-de-acoes-da-redecard/"),
    "61557039": dict(bank="Itau", root="60701190", basis="brand_and_control",
                     evidence="Itau Seguros, an Itau Unibanco company: https://www.itau.com.br/"
                              "seguros"),
    "92661388": dict(bank="Itau", root="60701190", basis="brand_and_control",
                     evidence="Itau Vida e Previdencia, 'company of the Itau Unibanco Financial "
                              "Conglomerate' (management report 2024): https://www.itau.com.br/"
                              "download-file/v2/d/42787847-4cf6-4461-94a5-40ed237dca33/"
                              "2a625af4-fc67-eb67-7588-576e3346f0f7?origin=1"),
    "23025711": dict(bank="Itau", root="60701190", basis="brand",
                     evidence="name carries ITAU (Cia Itau de Capitalizacao); control not "
                              "verified"),
    "08816067": dict(bank="Itau", root="60701190", basis="brand",
                     evidence="Itau-branded auto and home insurer of the 2009 Porto Seguro-Itau "
                              "association, controlled by Porto: https://www.infomoney.com.br/"
                              "minhas-financas/acionistas-do-itau-unibanco-firmam-associacao-com-"
                              "a-porto-seguro/"),
    "59573030": dict(bank="Itau", root="60701190", basis="brand", institute=True,
                     evidence="Fundacao Itau Social / Fundacao Itau para a Educacao e Cultura "
                              "(Itau Cultural): name carries ITAU"),
    "42786803": dict(bank="Itau", root="60701190", basis="control",
                     evidence="iupp, Itau Unibanco's points programme, integrated into the Itau "
                              "apps: https://tecnoblog.net/noticias/iupp-programa-de-pontos-do-"
                              "itau-e-integrado-ao-app-e-fica-restrito-a-clientes/"),
    "04270778": dict(bank="Santander", root="90400888", basis="brand_and_control",
                     evidence="Santander Corretora de Seguros, Investimentos e Servicos, "
                              "controlled entirely by Banco Santander (Brasil): "
                              "https://www.dnb.com/"
                              "business-directory/company-profiles.santander_corretora_de_seguros_"
                              "investimentos_e_servicos_sa.1ece257424fce8b09523d1256de8a139.html"),
    "61472676": dict(bank="Santander", root="90400888", basis="brand",
                     evidence="IF.data lists 61472676 as BANCO SANTANDER BRASIL S.A. with no "
                              "prudential conglomerate code in any list 201303-202512; its CRTs "
                              "(2013-2014) advertise Santander products (Select, Conta Combinada)"),
    "34020354": dict(bank="Caixa", root="00360305", basis="brand", until=202103,
                     evidence="Caixa Seguradora: CNP 51.75%, Caixa Seguridade 48.25%; sold Caixa-"
                              "branded insurance in the Caixa network until 2021: "
                              "https://pt.wikipedia.org/wiki/Caixa_Seguridade ; "
                              "https://www.ri.caixaseguridade.com.br/en/a-companhia/empresas-do-"
                              "grupo/"),
    "01599296": dict(bank="Caixa", root="00360305", basis="brand",
                     evidence="name carries CAIXA (Caixa Capitalizacao, a Caixa Seguridade "
                              "partnership): https://www.ri.caixaseguridade.com.br/en/a-companhia/"
                              "empresas-do-grupo/"),
    "03730204": dict(bank="Caixa", root="00360305", basis="brand",
                     evidence="name carries CAIXA (Caixa Vida e Previdencia, a Caixa Seguridade "
                              "partnership): https://www.ri.caixaseguridade.com.br/en/a-companhia/"
                              "empresas-do-grupo/"),
    "28196889": dict(bank="Banco do Brasil", root="00000000", basis="brand_and_control",
                     evidence="Brasilseg, its films sold as BB Seguros; " + _BB_SEGURIDADE),
    "27665207": dict(bank="Banco do Brasil", root="00000000", basis="control",
                     evidence="Brasilprev: BB Seguros holds 74.99% of the economics (49.99% of "
                              "votes); " + _BB_SEGURIDADE),
    "15138043": dict(bank="Banco do Brasil", root="00000000", basis="control",
                     evidence="Brasilcap: BB Seguros holds 66.77% of the economics (49.99% of "
                              "votes); " + _BB_SEGURIDADE),
    "01984199": dict(bank="BRB", root="00000208", basis="brand_and_control",
                     evidence="Cartao BRB S.A. (BRBCARD), controlled by BRB - Banco de Brasilia: "
                              "https://novo.brb.com.br/sobre-o-brb/empresas-com-marca-brb/"
                              "brbcard/"),
    "42597575": dict(bank="BRB", root="00000208", basis="brand_and_control",
                     evidence="BRB's insurance broker, held through Cartao BRB: "
                              "https://novo.brb.com.br/servico-de-informacao-ao-cidadao/"
                              "coligadas/"),
    "44705886": dict(bank="BRB", root="00000208", basis="brand_and_control",
                     evidence="BRB's insurance broker, held through Cartao BRB: "
                              "https://novo.brb.com.br/servico-de-informacao-ao-cidadao/"
                              "coligadas/"),
    "27053230": dict(bank="Banestes", root="28127603", basis="brand_and_control",
                     evidence="Banestes S.A. holds 100% of Banestes Seguros: "
                              "https://ri.banestes.com.br/o-banestes/empresas-controladas"),
    "13180351": dict(bank="Banese", root="13009717", basis="brand_and_control",
                     evidence="Banese's insurance broker, a Banese group company: "
                              "https://www.banese.com.br/o-banese/sobre-nos"),
    "10645538": dict(bank="Banese", root="13009717", basis="brand", institute=True,
                     evidence="name carries BANESE (Instituto Banese)"),
    "61198164": dict(bank="Porto", root="04862600", basis="brand_and_control", evidence=_PORTO),
    "33448150": dict(bank="Porto", root="04862600", basis="control",
                     evidence="Azul Seguros, controlled by Porto Seguro since 2003: "
                              "https://www.azulseguros.com.br/institucional/quem-somos/"),
    "06864650": dict(bank="Porto", root="04862600", basis="brand_and_control", institute=True,
                     evidence="Instituto Porto Seguro, maintained by Porto: "
                              "https://gife.org.br/associados/instituto-porto-seguro/"),
    "19091996": dict(bank="Porto", root="04862600", basis="brand",
                     evidence="Porto Seguro Locadora de Veiculos, the company behind Porto's "
                              "Carro Facil: https://www.portoseguro.com.br/carro-facil/"),
    "08279191": dict(bank="BNP Paribas", root="01522368", basis="control",
                     evidence="BNP Paribas Cardif, a BNP Paribas subsidiary: "
                              "https://bnpparibascardif.com.br/quem-somos/quem-somos/"),
    "03546261": dict(bank="BNP Paribas", root="01522368", basis="control",
                     evidence="BNP Paribas Cardif, a BNP Paribas subsidiary: "
                              "https://bnpparibascardif.com.br/quem-somos/quem-somos/"),
    "name:INTER & CO PAYMENTS": dict(bank="Inter", root="00416968", basis="brand_and_control",
                                     evidence="Inter&Co Payments, Inc., a subsidiary of "
                                              "Inter&Co, Inc. (Banco Inter's parent): "
                                              "https://us.inter.co/compliance"),
}
# Parent holdings: only CRTs whose product or title names one of `banks` are attributed.
HOLDINGS = {
    "07570673": dict(banks=("PicPay", "Original"),
                     evidence="J&F Participacoes, the Batista family's holding that controls "
                              "PicPay: https://euqueroinvestir.com/acoes/ipo-picpay-estrutura-"
                              "acionaria-controle-fundadores ; https://jfinvest.com.br/en/"
                              "business/picpay/"),
    "00350763": dict(banks=("PicPay", "Original"),
                     evidence="J&F Investimentos controls Banco Original (and PicPay through J&F "
                              "Participacoes): https://pt.wikipedia.org/wiki/J%26F_Investimentos"),
    "29694063": dict(banks=("C6",),
                     evidence="C6 Holding S.A., indirect controller of Banco C6 (Banco C6 "
                              "financial statements 2019): https://cdn.c6bank.com.br/c6-site-docs/"
                              "demonstracoes-financeiras-31-12-2019-c6bank.pdf"),
    "01109184": dict(banks=("PagSeguro",),
                     evidence="UOL, controlling shareholder of PagSeguro Digital (85% of votes): "
                              "https://en.wikipedia.org/wiki/PagSeguro"),
    "61532644": dict(banks=("Itau",),
                     evidence="Itausa controls Itau Unibanco through IUPAR: "
                              "https://www.nordinvestimentos.com.br/blog/itau-ou-itausa/"),
    "03407049": dict(banks=("BV",),
                     evidence="Votorantim S.A. controls Banco BV through Votorantim Financas "
                              "(50.01% of votes): https://ri.bv.com.br/en/corporate-governance/"
                              "shareholding-structure/"),
    "00776574": dict(banks=("Ame",),
                     evidence="Americanas (Lojas Americanas and B2W) controls Ame Digital: "
                              "https://conteudos.xpi.com.br/acoes/relatorios/curtas-b2w-e-lojas-"
                              "americanas-anunciam-estrutura-societaria-da-ame-digital/"),
    "02149205": dict(banks=("Porto",),
                     evidence="Porto Seguro S.A., the listed holding over Porto Bank and Portoseg: "
                              "https://ri.portoseguro.com.br/en/the-company/corporate-"
                              "presentation/"),
    "name:SANTANDER": dict(banks=("Santander",),
                           evidence="Santander group (no CNPJ on the CRT); Banco Santander S.A. "
                                    "controls Santander Brasil: https://en.wikipedia.org/wiki/"
                                    "Santander_Brasil"),
    "name:BANCO SANTANDER SA": dict(banks=("Santander",),
                                    evidence="Banco Santander S.A. (Spain, no CNPJ on the CRT), "
                                             "controlling shareholder of Santander Brasil: "
                                             "https://en.wikipedia.org/wiki/Santander_Brasil"),
}
# Multi-bank joint ventures: never attributed to a bank; owners recorded for the user.
_ELOPAR = "https://pt.wikipedia.org/wiki/Elo_Participa%C3%A7%C3%B5es_S/A"
JOINT_VENTURES = {
    "09227084": dict(owners="EloPar (Bradesco 50.01%, Banco do Brasil 49.99%) 66.665%; Caixa "
                            "33.335%",
                     evidence="https://www.infomoney.com.br/minhas-financas/banco-do-brasil-"
                              "bradesco-e-caixa-formalizam-acordo-para-criar-elo-servicos/ ; "
                              + _ELOPAR),
    "12888241": dict(owners="Banco do Brasil and Bradesco (EloPar)", evidence=_ELOPAR),
    "04740876": dict(owners="Banco do Brasil and Bradesco (EloPar); brands Alelo, Veloe",
                     evidence=_ELOPAR),
    "01027058": dict(owners="Banco do Brasil and Bradesco (directly and through EloPar)",
                     evidence="https://www.cnnbrasil.com.br/economia/negocios/cielo-anuncia-"
                              "proposta-de-bradesco-e-bb-para-sair-da-bolsa/ ; " + _ELOPAR),
}
ATTRIBUTIONS = ("own", "affiliate", "holding", "media_sponsored", "coop_system")
BANK_CLASSES = ATTRIBUTIONS[:4]
RANK = {c: i for i, c in enumerate(ATTRIBUTIONS)}


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


def brand_text(product, title) -> str:
    """Product and title, folded, punctuation turned to spaces (keeping & and + for 'INTER&CO'
    and 'BTG+'), joined by ' | ' so no brand phrase spans the two fields."""
    def one(t):
        return re.sub(r"\s+", " ", re.sub(r"[^A-Z0-9&+ ]+", " ", fold(t))).strip()
    return one("" if pd.isna(product) else product) + " | " + one("" if pd.isna(title) else title)


def name_words(text) -> str:
    """Folded name with punctuation turned to spaces, for the name patterns (CREDIT_COOP, MEDIA,
    AGENCY, PUBLIC_BODY): 'COOP. ECON. E CRED.' -> 'COOP ECON E CRED'."""
    return re.sub(r"\s+", " ", re.sub(r"[^A-Z0-9&+ ]+", " ", fold(text))).strip()


def advertiser_key(cnpj8: pd.Series, name_folded: pd.Series) -> pd.Series:
    """The advertiser's identity: its CNPJ root, or 'name:' + folded name when the CRT carries
    no valid CNPJ."""
    return cnpj8.astype("object").where(cnpj8.notna(), "name:" + name_folded)


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


def ym_from_label(label: str) -> int:
    y, q = label.split("Q")
    return int(y) * 100 + int(q) * 3


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------
def write_json_atomic(path: Path, obj: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False), encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def url_refusal(url: str) -> str | None:
    """Why `url` may not be requested, or None when it may: only https on ALLOWED_HOST, with no
    user information and at most the default port. The last test asks urllib which host it would
    connect to (`Request.host`), so a user name ('dados.ancine.gov.br@elsewhere') or another port
    cannot make the connection differ from the name that was checked."""
    parts = urllib.parse.urlsplit(url)
    if parts.scheme.lower() != "https":
        return f"scheme {parts.scheme or '(none)'!r} is not https"
    host = (parts.hostname or "").lower()
    if host != ALLOWED_HOST:
        return f"host {host or '(none)'} is not {ALLOWED_HOST}"
    if parts.username is not None or parts.password is not None:
        return "it carries user information"
    try:
        port = parts.port
    except ValueError:
        return "its port is not a number"
    if port not in (None, 443):
        return f"port {port} is not 443"
    try:
        connect = (urllib.request.Request(url).host or "").lower()
    except ValueError as exc:
        return f"urllib cannot parse it ({exc})"
    if connect not in (host, f"{host}:443"):
        return f"urllib would connect to {connect!r}, not to {host}"
    return None


class UnapprovedRedirect(RuntimeError):
    """A redirect to a target `url_refusal` refuses. Raised before the target is contacted. Not
    an OSError on purpose: download() retries network errors, and a refusal is no transient
    failure."""

    def __init__(self, code: int, source: str, target: str, reason: str):
        self.code, self.source, self.target, self.reason = code, source, target, reason
        super().__init__(f"{source} redirected (HTTP {code}) to {target}: {reason}")


class _CheckedRedirects(urllib.request.HTTPRedirectHandler):
    """urllib's redirect handling, but a hop is followed only to a URL `url_refusal` accepts,
    and every hop followed is recorded. urllib's default handler follows a 30x to any host,
    port or scheme."""

    def __init__(self, hops: list):
        super().__init__()
        self.hops = hops

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # urllib calls this with the target already made absolute and before requesting it, so
        # a refusal here means the target is never contacted.
        why = url_refusal(newurl)
        if why:
            with contextlib.suppress(Exception):
                fp.close()
            raise UnapprovedRedirect(int(code), req.full_url, newurl, why)
        self.hops.append([int(code), newurl])
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def download(refresh: bool) -> dict:
    """Fetch the CRT file once and record what the server said about it.

    The body is streamed to a '.part' file and moved into place only after its byte count
    equals the server's Content-Length, so an interrupted transfer never leaves a truncated
    file under the real name. Compression is refused (Accept-Encoding: identity) because a
    compressed body would make the Content-Length comparison meaningless. Redirects are
    followed only to https://dados.ancine.gov.br (url_refusal); a refused one stops the run
    without a retry, since the server would answer the same way.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {}
    if (not refresh and manifest and CSV_PATH.exists()
            and CSV_PATH.stat().st_size == manifest.get("bytes")):
        log.info("cached %s (%d bytes, Last-Modified %s, retrieved %s)", CSV_PATH.name,
                 manifest["bytes"], manifest.get("last_modified"), manifest.get("retrieved_at"))
        return manifest

    why = url_refusal(URL)
    if why:
        raise SystemExit(f"{URL} may not be requested: {why}")
    part = CSV_PATH.with_name(CSV_PATH.name + ".part")
    headers = {"User-Agent": USER_AGENT, "Accept-Encoding": "identity", "Accept": "*/*"}
    for attempt in range(1, 4):
        hops: list = []
        try:
            opener = urllib.request.build_opener(_CheckedRedirects(hops))
            req = urllib.request.Request(URL, headers=headers)
            digest, n = hashlib.sha256(), 0
            with opener.open(req, timeout=180) as resp, part.open("wb") as fh:
                status = getattr(resp, "status", None)
                final_url = resp.geturl()
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
        except UnapprovedRedirect as exc:
            if part.exists():
                part.unlink()
            raise SystemExit(f"refused a redirect: {exc}")
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
                "retrieved_at": now_iso(), "file": CSV_PATH.name, "final_url": final_url,
                "redirects": hops}
    write_json_atomic(MANIFEST, manifest)
    log.info("GET %s %s: %d bytes, Last-Modified %s%s", status, URL, n, last_mod,
             f" (via {len(hops)} redirect(s) to {final_url})" if hops else "")
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
                                                              pd.DataFrame, ReadStats]:
    """Stream the file. Returns the full rows whose advertiser root is in the IF.data registry;
    the full rows outside it that the attribution rules can take (the advertiser is in
    AFFILIATES or HOLDINGS, is a credit cooperative, or the product or title names a bank); a
    compact frame of the other rows whose advertiser name looks financial; and the stats."""
    st = ReadStats()
    fold_cache: dict[str, str] = {}
    valid_cache: dict[str, bool] = {}
    coop_cache: dict[str, bool] = {}
    keep, cand, fin = [], [], []
    attach_keys = set(AFFILIATES) | set(HOLDINGS)
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

        names = chunk[COL_ADV].fillna("")
        for nm in names.unique():
            if nm not in fold_cache:
                fold_cache[nm] = fold(nm)
                coop_cache[nm] = bool(CREDIT_COOP.search(name_words(nm)))
        folded = names.map(fold_cache)
        # Titles are nearly all distinct, so the brand test is cached per chunk only.
        text = [brand_text(p, t) for p, t in zip(chunk[COL_PROD], chunk[COL_TITLE])]
        pt_hit = pd.Series([bool(BRAND_ANY.search(t)) for t in text], index=chunk.index)
        key = advertiser_key(cnpj8, folded)
        take = ~in_reg & (key.isin(attach_keys) | names.map(coop_cache) | pt_hit)
        cand.append(chunk[take])

        looks = folded[~in_reg].str.contains(FINANCIAL)
        fin.append(chunk.loc[looks[looks].index, [COL_CRT, COL_ADV, COL_ADV_CNPJ, "cnpj14",
                                                   "cnpj_valid", "cnpj8", "request_date"]])
        log.info("  read %7d rows", st.rows)
    return (pd.concat(keep, ignore_index=True), pd.concat(cand, ignore_index=True),
            pd.concat(fin, ignore_index=True), st)


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
        if self.reg_first <= ym <= self.reg_last:
            return None, "no_code_in_quarter"
        q = qindex(ym)
        dist = np.where(ym < per["min"], per["min"].map(qindex) - q, q - per["max"].map(qindex))
        best = np.flatnonzero(dist == dist.min())
        if len(best) > 1:
            return None, "ambiguous"
        span = per.iloc[best[0]]
        # Outside the window the nearest span stands in only if it reaches the edge the lists
        # speak for: a span that starts after the first coded list (Redecard's Itau span starts
        # 201703) says the root had no code at 201403, so it had none before either.
        if ym < self.reg_first and span["min"] != self.reg_first:
            return None, "no_code_in_quarter"
        if ym > self.reg_last and span["max"] != self.reg_last:
            return None, "no_code_in_quarter"
        return span["code"], "registry_nearest"

    def assign(self, cnpj8: str, ym: int) -> dict:
        ym_p = min(max(ym, self.panel_first), self.panel_last)
        # The panel span (first to last quarter) is tested, not per-quarter presence; in_panel
        # marks the CRTs that fall in a hole of that span.
        lead = [c for c in self.leader_codes.get(cnpj8, ())
                if self.spans.at[c, "first"] <= ym_p <= self.spans.at[c, "last"]]
        reg_code, how = self.registry_code(cnpj8, ym_p)
        out = {"cnpj8": cnpj8, "ym": ym, "map_quarter": ym_label(ym_p),
               "registry_code": reg_code, "registry_reason": how, "panel_code": None,
               "match_method": None, "unmapped_reason": None}
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
# Attribution beyond the advertiser's own CNPJ
# ---------------------------------------------------------------------------
def dated_registry_names(reg_sub: pd.DataFrame) -> dict[str, tuple[np.ndarray, list[str]]]:
    """Per CNPJ8, the IF.data lists' dates and the institution's name in each (folded, with
    punctuation as spaces). Names change: cooperatives move between systems (UNICRED
    MANTIQUEIRA is SICOOB MANTIQUEIRA in later lists), so a name is read at a CRT's quarter."""
    m = reg_sub[reg_sub["CodInst"].str.fullmatch(r"\d{1,8}", na=False)]
    m = m.assign(r=m["CodInst"].str.zfill(8), d=m["Data"].astype(int))
    out = {}
    for r, g in m.sort_values("d").drop_duplicates(["r", "d"], keep="last").groupby("r"):
        out[r] = (g["d"].to_numpy(), [name_words(n) for n in g["NomeInstituicao"].fillna("")])
    return out


def name_at(names: dict[str, tuple[np.ndarray, list[str]]], root, ym: int) -> str:
    """The registry name of `root` in the latest list up to `ym` (the first list before any)."""
    if not isinstance(root, str) or root not in names:
        return ""
    dates, texts = names[root]
    return texts[max(int(np.searchsorted(dates, ym, side="right")) - 1, 0)]


def systems_in(text: str) -> list[str]:
    return [lab for lab, rx in SYSTEM_RX if rx.search(text)]


def filer_type(key: str, names: str, registry_name: str, in_registry: bool) -> str:
    """What kind of firm filed the CRT, for the attribution rules: the tables first, then the
    advertiser's names (all its ANCINE names joined, and its latest registry name)."""
    if key in AFFILIATES:
        return "affiliate"
    if key in HOLDINGS:
        return "holding"
    if key in JOINT_VENTURES:
        return "joint_venture"
    if CREDIT_COOP.search(names) or CREDIT_COOP.search(registry_name):
        return "credit_coop"
    if PUBLIC_BODY.search(names):
        return "public_body"
    if in_registry:
        return "registry_institution"
    if MEDIA.search(names):
        return "media"
    if AGENCY.search(names):
        return "agency"
    return "other"


def attribute(u: pd.DataFrame, mapper: Mapper, roots: set[str],
              reg_sub: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """Attribution records for rows that did not map as own.

    `u` holds the unmapped registry rows and the non-registry candidate rows. Returns one record
    per (row, bank) with the class (affiliate, holding, media_sponsored, coop_system or
    other_filer_named_bank), the bank root it maps through and the Mapper's answer for it; and,
    per credit-cooperative advertiser, the systems its rows were given.

    A cooperative's system is decided per CRT, from the most specific evidence: the system its
    film names; else the one its IF.data name carries in that quarter; else the one its ANCINE
    name carries; else the single system all its films and ANCINE names carry. The IF.data name
    is dated and the ANCINE name is not: UNICRED CEARA CENTRO NORTE keeps that name on its CRTs
    of 2015-2026, while the lists call it SICREDI from 2016. A cooperative that changed systems
    therefore keeps each film with the system of its time, and one whose evidence points two
    ways gets no system."""
    u = u.copy()
    u["adv_fold"] = u[COL_ADV].fillna("").map(fold)
    u["adv_words"] = u[COL_ADV].fillna("").map(name_words)
    u["advertiser_key"] = advertiser_key(u["cnpj8"], u["adv_fold"])
    u["brand_text"] = [brand_text(p, t) for p, t in zip(u[COL_PROD], u[COL_TITLE])]
    reg_names = dated_registry_names(reg_sub)
    # One type per advertiser: a CNPJ root is one legal entity whatever name a row spells.
    per_key = u.groupby("advertiser_key").agg(
        names=("adv_words", lambda s: " | ".join(sorted(set(s)))),
        in_registry=("in_registry", "any"),
        texts=("brand_text", lambda s: " | ".join(sorted(set(s)))))
    ftype = {k: filer_type(k, r["names"], name_at(reg_names, k, 999999),
                           bool(r["in_registry"]))
             for k, r in per_key.iterrows()}
    u["filer_type"] = u["advertiser_key"].map(ftype)
    key_system = {}
    for k, r in per_key[per_key.index.map(ftype) == "credit_coop"].iterrows():
        found = set(systems_in(r["names"])) | set(systems_in(r["texts"]))
        key_system[k] = found.pop() if len(found) == 1 else None

    recs = []
    given: dict[str, set[str]] = {}
    for idx, r in zip(u.index, u.itertuples(index=False)):
        text = r.brand_text
        # A film that names its own advertiser's root is that advertiser's own film, left
        # unmapped by the registry rule; it is not re-attributed.
        named = [lab for lab, root, rx in BRAND_RX
                 if rx.search(text) and (root is None or root != r.cnpj8)]
        suspect = bool(EVENT_VENUE.search(text))
        ft = r.filer_type

        def add(cls, bank, basis=None, evidence=None, flag=False, target=None):
            recs.append({"row": idx, "attribution": cls, "bank_named": bank,
                         "target_cnpj8": target or BRAND_ROOT.get(bank),
                         "attach_basis": basis, "evidence": evidence,
                         "naming_rights_suspect": flag})

        taken: set[str] = set()
        if ft == "affiliate":
            a = AFFILIATES[r.advertiser_key]
            if a.get("until") is None or r.ym <= a["until"]:
                add("affiliate", a["bank"], a["basis"], a["evidence"], a.get("institute", False),
                    target=a["root"])
                taken.add(a["bank"])
        elif ft == "holding":
            h = HOLDINGS[r.advertiser_key]
            for lab in named:
                if lab in h["banks"]:
                    add("holding", lab, "control", h["evidence"], suspect)
                    taken.add(lab)
        elif ft == "credit_coop":
            in_text = systems_in(text)
            in_name = systems_in(r.adv_words)
            in_reg = systems_in(name_at(reg_names, r.cnpj8, r.ym))
            if len(in_text) == 1:
                sysname, how = in_text[0], "its film names " + in_text[0].upper()
            elif in_reg:
                sysname, how = in_reg[0], ("its IF.data name in that quarter carries "
                                           + in_reg[0].upper())
            elif in_name:
                sysname, how = in_name[0], "its ANCINE name carries " + in_name[0].upper()
            elif key_system.get(r.advertiser_key):
                sysname = key_system[r.advertiser_key]
                how = "its other films or names carry " + sysname.upper()
            else:
                sysname, how = None, None
            if sysname:
                given.setdefault(r.advertiser_key, set()).add(sysname)
            if sysname in ATTACHED_SYSTEMS:
                add("coop_system", sysname, "brand", f"{sysname} system cooperative: {how}")
            # Its own system named in its films is that same attribution, not another bank.
            taken.update(lab for lab, _ in SYSTEM_RX)
        elif ft == "media":
            for lab in named:
                if lab in ATTACHED_SYSTEMS:
                    add("coop_system", lab, "brand",
                        f"media film naming {lab.upper()} (a cooperative's sponsorship)", suspect)
                    taken.add(lab)
                elif BRAND_ROOT.get(lab):
                    add("media_sponsored", lab, None, None, suspect)
                    taken.add(lab)
        for lab in named:
            if lab not in taken:
                add("other_filer_named_bank", lab, None, None, suspect)
    rec = pd.DataFrame(recs, columns=["row", "attribution", "bank_named", "target_cnpj8",
                                      "attach_basis", "evidence", "naming_rights_suspect"])
    stray = set(rec["target_cnpj8"].dropna()) - roots
    if stray:
        raise AssertionError(f"attribution roots absent from the IF.data registry: {stray}")
    pairs = rec.loc[rec["target_cnpj8"].notna(), ["target_cnpj8"]].assign(
        ym=u.loc[rec.loc[rec["target_cnpj8"].notna(), "row"], "ym"].to_numpy()).drop_duplicates()
    assigned = pd.DataFrame([mapper.assign(c, int(y)) for c, y in pairs.itertuples(index=False)],
                            columns=["cnpj8", "ym", "map_quarter", "registry_code",
                                     "registry_reason", "panel_code", "match_method",
                                     "unmapped_reason"]).rename(columns={"cnpj8": "target_cnpj8"})
    body = u.drop(columns=["map_quarter", "registry_code", "registry_reason", "panel_code",
                           "match_method", "unmapped_reason", "adv_words"], errors="ignore")
    out = (rec.merge(body, left_on="row", right_index=True, how="left", validate="many_to_one")
              .merge(assigned, on=["target_cnpj8", "ym"], how="left", validate="many_to_one"))
    systems = {k: "; ".join(sorted(v)) for k, v in given.items()}
    return out.drop(columns="row"), systems


def with_counted_as(lines: pd.DataFrame) -> pd.DataFrame:
    """`counted_as`: the class each (panel code, CRT) is counted under, the first in
    ATTRIBUTIONS order among its rows."""
    rank = lines["attribution"].map(RANK)
    best = rank.groupby([lines["panel_code"], lines[COL_CRT]]).transform("min")
    return lines.assign(counted_as=best.map(dict(enumerate(ATTRIBUTIONS))))


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
    # One record per (code, CRT) under the class it is counted as; the flag is taken from the
    # rows of that class only.
    own_class = lines[lines["attribution"] == lines["counted_as"]]
    per_crt = (own_class.groupby(["panel_code", COL_CRT], sort=False)
                        .agg(year=("year", "first"), month=("month", "first"),
                             counted_as=("counted_as", "first"), seg_label=(COL_SEG, "first"),
                             versoes=(COL_VERS, "first"),
                             suspect=("naming_rights_suspect", "any"))
                        .reset_index())
    per_crt["seg"] = per_crt["seg_label"].map(segment_slug)
    per_crt["versoes"] = pd.to_numeric(per_crt["versoes"], errors="coerce")
    bank = per_crt[per_crt["counted_as"].isin(BANK_CLASSES)]
    base = (bank.groupby(keys).agg(n_crt=(COL_CRT, "nunique"),
                                   sum_qtd_versoes=("versoes", "sum"),
                                   n_crt_versoes_reported=("versoes", "count")))
    classes = (per_crt.pivot_table(index=keys, columns="counted_as", values=COL_CRT,
                                   aggfunc="nunique", fill_value=0)
                      .reindex(columns=list(ATTRIBUTIONS), fill_value=0)
                      .add_prefix("n_crt_"))
    suspect = (bank[bank["suspect"].fillna(False).astype(bool)]
               .groupby(keys)[COL_CRT].nunique().rename("n_crt_naming_rights_suspect"))
    bank_rows = own_class[own_class["counted_as"].isin(BANK_CLASSES)]
    titles = bank_rows.groupby(keys)["title_key"].nunique().rename("n_titles_distinct")
    seg = (bank.pivot_table(index=keys, columns="seg", values=COL_CRT, aggfunc="nunique",
                            fill_value=0)
               .reindex(columns=sorted(set(SEGMENTS.values()) | set(bank["seg"])), fill_value=0)
               .add_prefix("n_crt_seg_"))
    obs = classes.join(base).join(suspect).join(titles).join(seg)

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
    first = ["panel_code", "year", "month", "n_crt_own", "n_crt_affiliate", "n_crt_holding",
             "n_crt_media_sponsored", "n_crt", "n_crt_naming_rights_suspect", "n_crt_coop_system"]
    out = out[first + [c for c in out.columns if c not in first]]
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
    counted = lines["attribution"] == lines["counted_as"]
    g = (lines.assign(crt_counted=lines[COL_CRT].where(counted))
              .groupby(["advertiser_key", "panel_code", "attribution", "match_method"],
                       dropna=False)
              .agg(cnpj8=("cnpj8", "first"), first_ym=("ym", "min"), last_ym=("ym", "max"),
                   n_crt=(COL_CRT, "nunique"), n_crt_counted=("crt_counted", "nunique"),
                   n_rows=(COL_CRT, "size"), code_conflict_rows=("code_conflict", "sum"),
                   names=(COL_ADV, lambda s: list(s.value_counts().index[:3])),
                   attach_basis=("attach_basis", "first"), evidence=("evidence", "first"))
              .reset_index())
    g = g[g["n_rows"] > 0]
    g["first_quarter"] = g["first_ym"].map(ym_label)
    g["last_quarter"] = g["last_ym"].map(ym_label)
    reg_names = registry_names(reg_sub)
    g["advertiser_names"] = g["names"].map(" | ".join)
    g["registry_name"] = g["cnpj8"].map(lambda r: xw.registry_name(reg_sub, r) if pd.notna(r)
                                        else "")
    g["panel_name"] = g["panel_code"].map(spans["panel_name"])
    # Printed for review, never used to unmap: renames (Aymore -> Santander SCFI, SEAC ->
    # Mulvi) legitimately share no word, and the CNPJ root is the legal identity. Only an own
    # row claims the advertiser is the institution, so only own rows are compared.
    g["names_agree"] = [names_agree(a, reg_names.get(r, [])) if att == "own" else pd.NA
                        for a, r, att in zip(g["names"], g["cnpj8"], g["attribution"])]
    g["names_agree"] = g["names_agree"].astype("boolean")
    g["jv_owners"] = g["advertiser_key"].map(lambda k: JOINT_VENTURES.get(k, {}).get("owners"))
    return (g.drop(columns=["first_ym", "last_ym", "names"])
             .sort_values(["n_crt", "advertiser_key"], ascending=[False, True])
             .reset_index(drop=True))


def registry_names(reg_sub: pd.DataFrame) -> dict[str, list[str]]:
    member = reg_sub["CodInst"].str.fullmatch(r"\d{1,8}", na=False)
    m = reg_sub[member]
    return (m.assign(r=m["CodInst"].str.zfill(8)).groupby("r")["NomeInstituicao"]
             .agg(lambda s: sorted(set(s.dropna()))).to_dict())


def brand_attributions(att: pd.DataFrame, lines: pd.DataFrame,
                       spans: pd.DataFrame) -> pd.DataFrame:
    """The review file: every attribution made by product/title or by cooperative system, and
    every other filer's film that names a bank, with the class each (code, CRT) counts as."""
    keep = att[att["attribution"].isin(["holding", "media_sponsored", "coop_system",
                                        "other_filer_named_bank"])].copy()
    counted = (lines.drop_duplicates(["panel_code", COL_CRT])
                    .set_index(["panel_code", COL_CRT])["counted_as"])
    idx = pd.MultiIndex.from_arrays([keep["panel_code"], keep[COL_CRT]])
    keep["counted_as"] = counted.reindex(idx).to_numpy()
    keep.loc[keep["attribution"] == "other_filer_named_bank", "counted_as"] = pd.NA
    keep["panel_name"] = keep["panel_code"].map(spans["panel_name"])
    cols = [COL_CRT, "request_date", "year", "month", COL_ADV, COL_ADV_CNPJ, "advertiser_key",
            "filer_type", COL_PROD, COL_TITLE, COL_SEG, "bank_named", "target_cnpj8",
            "attribution", "panel_code", "panel_name", "counted_as", "naming_rights_suspect",
            "in_panel", "map_quarter", "match_method", "unmapped_reason", "evidence"]
    return (keep[cols].sort_values(["attribution", "bank_named", "request_date", COL_CRT])
                      .reset_index(drop=True))


def unmatched_financial(unmapped_reg: pd.DataFrame, fin: pd.DataFrame, reg_sub: pd.DataFrame,
                        mapper: Mapper, attributed: pd.DataFrame,
                        systems: dict[str, str]) -> pd.DataFrame:
    """Advertisers left unmapped as own whose ANCINE name, or registry name, looks financial,
    with what the attribution rules took of their CRTs."""
    names = registry_names(reg_sub)
    a = unmapped_reg.assign(in_registry=True)
    b = fin.assign(in_registry=False,
                   unmapped_reason=np.where(fin["cnpj14"].isna(), "no_cnpj",
                                            np.where(fin["cnpj8"].isna(), "invalid_cnpj",
                                                     "not_in_registry")))
    both = pd.concat([a, b], ignore_index=True)
    both["key"] = advertiser_key(both["cnpj8"], both[COL_ADV].fillna("").map(fold))
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
                  example_cnpj=(COL_ADV_CNPJ, "first")))
    took = attributed[attributed["advertiser_key"].isin(g.index)]
    g["n_crt_attributed"] = (took.groupby("advertiser_key")[COL_CRT].nunique()
                                 .reindex(g.index).fillna(0).astype("int64"))
    g["attributed_as"] = (took.groupby(["advertiser_key", "attribution", "panel_code"])[COL_CRT]
                              .nunique().reset_index()
                              .assign(t=lambda d: d["attribution"] + " -> " + d["panel_code"]
                                      + " (" + d[COL_CRT].astype(str) + ")")
                              .groupby("advertiser_key")["t"].agg("; ".join)
                              .reindex(g.index).fillna(""))
    g = g.reset_index(drop=True).assign(key=g.index.to_numpy())
    g["jv_owners"] = g["key"].map(lambda k: JOINT_VENTURES.get(k, {}).get("owners", ""))
    g["jv_evidence"] = g["key"].map(lambda k: JOINT_VENTURES.get(k, {}).get("evidence", ""))
    g["coop_system"] = g["key"].map(systems).fillna("")
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
    return (g[looks].drop(columns=["held_back", "key"])
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
    parts = monthly[[f"n_crt_{c}" for c in BANK_CLASSES]].sum(axis=1)
    if not (parts == monthly["n_crt"]).all():
        raise AssertionError("n_crt is not the sum of n_crt_own, _affiliate, _holding and "
                             "_media_sponsored")
    if not (monthly["n_crt_naming_rights_suspect"] <= monthly["n_crt"]
            - monthly["n_crt_own"]).all():
        raise AssertionError("n_crt_naming_rights_suspect exceeds the non-own part of n_crt")
    log.info("check 4 passed: %d panel_code-months, none repeated; %d segment columns add up "
             "to n_crt, and n_crt = own + affiliate + holding + media_sponsored, on every row "
             "(n_crt_coop_system %d CRT-months kept out)", len(monthly), len(seg_cols),
             int(monthly["n_crt_coop_system"].sum()))
    stray = set(lines["target_cnpj8"]) - roots                                        # check 5
    if stray:
        raise AssertionError("mapped CNPJ8s absent from the IF.data registry: "
                             f"{sorted(stray)[:10]}")
    off_panel = set(lines["panel_code"]) - set(spans.index)
    if off_panel:
        raise AssertionError(f"panel codes absent from the market panel: {sorted(off_panel)}")
    own = lines[lines["attribution"] == "own"]
    log.info("check 5 passed: %d CNPJ8s mapped through (%d own advertiser roots) all in the "
             "IF.data registry; %d panel codes all in the market panel",
             lines["target_cnpj8"].nunique(), own["cnpj8"].nunique(),
             lines["panel_code"].nunique())
    # Tested on every row of the file that carries the anchor's root, not only on the mapped
    # lines: an anchor row that fails to map would be absent from `lines`, and a root missing
    # from the registry would be absent from `rows`; either would otherwise pass unseen.
    for cnpj8, code in xw.ANCHORS.items():
        n_raw = st.anchor_rows.get(cnpj8, 0)
        own_rows = rows[rows["cnpj8"] == cnpj8]
        if n_raw == 0:
            if cnpj8 in REQUIRED_ANCHORS:
                raise AssertionError(f"anchor {cnpj8} ({code}) has no CRT row")
            log.info("check 5: anchor %s (%s) has no CRT row, nothing to test", cnpj8, code)
            continue
        if len(own_rows) != n_raw:
            raise AssertionError(f"anchor {cnpj8}: {n_raw} rows in the file carry the root but "
                                 f"{len(own_rows)} reached the mapping (root absent from the "
                                 "IF.data registry?)")
        wrong = own_rows[own_rows["panel_code"].fillna("") != code]
        if len(wrong):
            raise AssertionError(
                f"anchor {cnpj8}: {len(wrong)} of {len(own_rows)} rows do not map to {code}:\n"
                + wrong.groupby(["map_quarter", "panel_code", "unmapped_reason"], dropna=False)
                       .size().to_string())
        log.info("check 5 passed: anchor %s -> %s on all %d of its rows (%d CRTs, %s)", cnpj8,
                 code, n_raw, own_rows[COL_CRT].nunique(),
                 own_rows["match_method"].value_counts().to_dict())
    for col in (COL_TITLE, COL_REQ, COL_SEG):                                         # check 6
        varying = lines.groupby(COL_CRT)[col].nunique(dropna=False)
        if (varying > 1).any():
            raise AssertionError(f"{col} varies within {int((varying > 1).sum())} CRTs")
    log.info("check 6 passed: title, request date and segment constant within each of the %d "
             "mapped CRTs", lines[COL_CRT].nunique())
    bad = set(lines["attribution"]) - set(ATTRIBUTIONS)                               # check 7
    if bad:
        raise AssertionError(f"unknown attribution classes: {bad}")
    rules = {"affiliate": ~lines["advertiser_key"].isin(set(AFFILIATES)),
             "holding": ~lines["advertiser_key"].isin(set(HOLDINGS)),
             "media_sponsored": lines["filer_type"] != "media"}
    for cls, broken in rules.items():
        n = int((broken & (lines["attribution"] == cls)).sum())
        if n:
            raise AssertionError(f"{n} {cls} rows come from filers the rule does not cover")
    hold = lines[lines["attribution"] == "holding"]
    off = [b not in HOLDINGS[k]["banks"] for k, b in zip(hold["advertiser_key"],
                                                        hold["bank_named"])]
    if any(off):
        raise AssertionError(f"{sum(off)} holding rows name a bank outside their holding's list")
    jv = lines[lines["advertiser_key"].isin(set(JOINT_VENTURES))
               & (lines["attribution"] != "own")]
    if len(jv):
        raise AssertionError(f"{len(jv)} joint-venture rows attributed to a bank")
    per = lines.groupby(["panel_code", COL_CRT])["counted_as"].nunique()
    if (per != 1).any():
        raise AssertionError(f"{int((per != 1).sum())} (code, CRT) pairs without one counted_as")
    log.info("check 7 passed: attribution classes %s; affiliate/holding/media rows all from the "
             "filers their rule covers; no joint venture attributed; each of %d (code, CRT) "
             "pairs counted once", lines["attribution"].value_counts().to_dict(), len(per))


def summary(manifest: dict, header: list[str], st: ReadStats, lines: pd.DataFrame,
            monthly: pd.DataFrame, amap: pd.DataFrame, unmatched: pd.DataFrame,
            review: pd.DataFrame, spans: pd.DataFrame, dep_q4: pd.Series, mapper: Mapper,
            cutoff: pd.Timestamp) -> None:
    log.info("file: %s, %d bytes, Last-Modified %s, sha256 %s", manifest["url"],
             manifest["bytes"], manifest["last_modified"], manifest["sha256"][:16])
    log.info("IF.data lists with conglomerate codes %s-%s (nearest-span fallback only outside, "
             "and only to a span reaching that edge); market panel %s-%s",
             ym_label(mapper.reg_first), ym_label(mapper.reg_last),
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
    own = lines[lines["attribution"] == "own"]
    log.info("own: %d rows, %d distinct CRTs (%.2f%% of all), %d advertiser CNPJ8s; all "
             "classes: %d rows, %d (code, CRT) pairs, %d conglomerates", len(own),
             own[COL_CRT].nunique(), 100 * own[COL_CRT].nunique() / len(st.crts),
             own["cnpj8"].nunique(), len(lines), len(per_crt), lines["panel_code"].nunique())
    log.info("(code, CRT) pairs by counted_as: %s",
             per_crt["counted_as"].value_counts().reindex(list(ATTRIBUTIONS)).to_dict())
    log.info("own match methods (rows): %s", own["match_method"].value_counts().to_dict())
    conflict = lines[lines["code_conflict"]]
    if len(conflict):
        log.info("code_conflict on %d rows (the panel wins by design; registry_reason says "
                 "whether the lists gave another code or none):\n%s", len(conflict),
                 conflict.groupby(["target_cnpj8", "registry_reason", "registry_code",
                                   "panel_code"], dropna=False)[COL_CRT]
                         .nunique().rename("n_crt").reset_index().to_string(index=False))
    off = per_crt[~per_crt["in_panel"]]
    log.info("CRTs requested in a month when their code has no market-panel rows: %d of %d "
             "code-CRT pairs, %d of them after the panel's last quarter", len(off), len(per_crt),
             int(off["outside_panel_window"].sum()))
    part = monthly.loc[monthly["partial_month"], ["year", "month"]].drop_duplicates()
    log.info("partial request months (data cutoff %s, issue lag %d days): %s; their CRTs: %d",
             cutoff.date(), ISSUE_LAG_DAYS,
             [f"{y}-{m:02d}" for y, m in part.itertuples(index=False)],
             int(monthly.loc[monthly["partial_month"], "n_crt"].sum()))
    # Months outside the panel are excluded, so a conglomerate is ranked only on the months it
    # can enter an estimation with.
    win = monthly[monthly["in_panel"] & monthly["year"].between(2016, 2024)]
    cols = ["n_crt", "n_crt_own", "n_crt_affiliate", "n_crt_holding", "n_crt_media_sponsored",
            "n_crt_naming_rights_suspect", "n_crt_coop_system"]
    top = win.groupby("panel_code")[cols].sum().sort_values("n_crt", ascending=False).head(15)
    top["in_panel_months"] = win.groupby("panel_code").size().reindex(top.index)
    top["panel_name"] = top.index.map(spans["panel_name"])
    log.info("top 15 conglomerates by CRTs requested 2016-2024, in-panel months only:\n%s",
             top.to_string())
    att = lines[lines["attribution"] != "own"]
    log.info("attributed CRTs by class and filer:\n%s",
             att.groupby(["attribution", "advertiser_key", "panel_code"])
                .agg(n_crt=(COL_CRT, "nunique"), name=(COL_ADV, "first"))
                .reset_index().assign(advertiser_key=lambda d: d["advertiser_key"].str[:30],
                                      name=lambda d: d["name"].str[:60])
                .sort_values(["attribution", "n_crt"], ascending=[True, False])
                .to_string(index=False))
    log.info("review file: %s", review["attribution"].value_counts().to_dict())
    adv24 = set(per_crt.loc[(per_crt["year"] == 2024)
                            & per_crt["counted_as"].isin(BANK_CLASSES), "panel_code"])
    total = float(dep_q4.sum())
    share = float(dep_q4[dep_q4.index.isin(adv24)].sum()) / total if total else float("nan")
    log.info("2024Q4 panel deposits (dep_a1+a2+a4+a5): %.1f%% held by the %d conglomerates with "
             "at least one bank-class CRT requested in 2024 (%d of them have 2024Q4 panel rows; "
             "%d conglomerates hold positive 2024Q4 deposits)", 100 * share, len(adv24),
             len(adv24 & set(dep_q4.index)), int((dep_q4 > 0).sum()))
    odd = amap[amap["names_agree"].eq(False).fillna(False).astype(bool)]
    if len(odd):
        log.info("own CNPJ8s whose ANCINE and registry names do not agree (renames or a "
                 "wrong CNPJ on the CRT; review):\n%s",
                 odd[["cnpj8", "advertiser_names", "registry_name", "panel_code", "n_crt"]]
                 .to_string(index=False))
    show = unmatched.assign(advertiser_names=unmatched["advertiser_names"].str[:60],
                            unmapped_reason=unmatched["unmapped_reason"].str[:40])
    cols = ["cnpj8", "advertiser_names", "n_crt", "n_crt_attributed", "kind", "brand_word",
            "unmapped_reason"]
    log.info("unmatched advertisers that look financial: %d (%s CRTs, %s of them attributed); "
             "by kind: %s", len(unmatched), f"{int(unmatched['n_crt'].sum()):,}",
             f"{int(unmatched['n_crt_attributed'].sum()):,}",
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
def add_dates(frame: pd.DataFrame) -> pd.DataFrame:
    """Request year/month/quarter and the IF.data quarter key `ym`."""
    frame = frame.copy()
    frame["year"] = frame["request_date"].dt.year.astype("Int64")
    frame["month"] = frame["request_date"].dt.month.astype("Int64")
    frame["quarter"] = frame["request_date"].dt.quarter.astype("Int64")
    if frame["year"].isna().any():
        raise AssertionError(f"{int(frame['year'].isna().sum())} candidate rows have no request "
                             "date")
    frame["ym"] = frame["year"].astype(int) * 100 + frame["quarter"].astype(int) * 3
    return frame


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
    rows, cand, fin, st = read_crt(enc, delim, roots)
    log.info("rows whose advertiser root is in the IF.data registry: %d (%d roots); rows outside "
             "it the attribution rules can take: %d", len(rows), rows["cnpj8"].nunique(),
             len(cand))

    spans, dep_q4, present = panel_identity()
    targets = ({a_["root"] for a_ in AFFILIATES.values()}
               | {r for r in BRAND_ROOT.values() if r})
    map_roots = set(rows["cnpj8"]) | targets
    reg_sub = registry_subset(reg, map_roots)
    mapper = Mapper(reg_sub, spans, sorted(map_roots), registry_window(reg))
    cutoff = data_cutoff(manifest, st)

    rows = add_dates(rows)
    pairs = rows[["cnpj8", "ym"]].drop_duplicates()
    assigned = pd.DataFrame([mapper.assign(c, int(y)) for c, y in pairs.itertuples(index=False)])
    rows = rows.merge(assigned, on=["cnpj8", "ym"], how="left", validate="many_to_one")

    own = rows[rows["panel_code"].notna()].copy()
    own["adv_fold"] = own[COL_ADV].fillna("").map(fold)
    own = own.assign(attribution="own", target_cnpj8=own["cnpj8"], bank_named=None,
                     filer_type="own", attach_basis=None, evidence=None,
                     naming_rights_suspect=False,
                     advertiser_key=advertiser_key(own["cnpj8"], own["adv_fold"]))
    unmapped = rows[rows["panel_code"].isna()]
    log.info("registry-root rows left unmapped as own, by reason: %s",
             unmapped["unmapped_reason"].value_counts().to_dict())
    u = pd.concat([unmapped.assign(in_registry=True),
                   add_dates(cand).assign(in_registry=False)], ignore_index=True)
    att, systems = attribute(u, mapper, roots, reg_sub)
    attributed = att[att["attribution"].isin(ATTRIBUTIONS) & att["panel_code"].notna()]
    log.info("attribution records: %s; mapped to a panel code: %s",
             att["attribution"].value_counts().to_dict(),
             attributed["attribution"].value_counts().to_dict())

    lines = pd.concat([own, attributed], ignore_index=True)
    lines = lines.drop(columns=[c for c in ("adv_fold", "brand_text", "in_registry")
                                if c in lines.columns])
    map_ym = lines["map_quarter"].map(ym_from_label)
    # The panel wins over the registry by design; the flag keeps both disagreements in view:
    # the lists gave another code, or (inside their window) no code at all.
    lines["code_conflict"] = ((lines["registry_code"].notna()
                               & (lines["registry_code"] != lines["panel_code"]))
                              | ((lines["match_method"] == "panel_leader")
                                 & lines["registry_code"].isna()
                                 & map_ym.between(mapper.reg_first, mapper.reg_last)))
    lines["outside_panel_window"] = ((lines["ym"] < mapper.panel_first)
                                     | (lines["ym"] > mapper.panel_last))
    lines["in_panel"] = in_panel(lines["panel_code"], lines["year"], lines["quarter"], present)
    lines["partial_month"] = partial_month(lines["year"], lines["month"], cutoff)
    lines["panel_name"] = lines["panel_code"].map(spans["panel_name"])
    lines["title_norm"] = lines[COL_TITLE].map(norm_title)
    lines["title_key"] = lines["title_norm"] + "|" + lines["advertiser_key"]
    lines["naming_rights_suspect"] = lines["naming_rights_suspect"].fillna(False).astype(bool)
    lines = with_counted_as(lines)
    att = att.assign(in_panel=in_panel(att["panel_code"].fillna(""), att["year"],
                                       att["quarter"], present))

    monthly = monthly_table(lines, spans, present, cutoff)
    amap = advertiser_map(lines, reg_sub, spans)
    review = brand_attributions(att, lines, spans)
    unmatched = unmatched_financial(unmapped, fin, reg_sub, mapper,
                                    lines[lines["attribution"] != "own"], systems)

    validate(manifest, header, st, lines, rows, monthly, roots, spans)
    summary(manifest, header, st, lines, monthly, amap, unmatched, review, spans, dep_q4, mapper,
            cutoff)

    a.out_dir.mkdir(parents=True, exist_ok=True)
    out_lines = lines.drop(columns=["ym", "title_key", "unmapped_reason"]).sort_values(
        ["panel_code", "request_date", COL_CRT, "attribution"]).reset_index(drop=True)
    tables = (("ancine_ad_films_lines", out_lines),
              ("ancine_ad_films_monthly", monthly),
              ("ancine_advertiser_map", amap),
              ("ancine_brand_attributions", review),
              ("ancine_unmatched_financial", unmatched))
    # The previous run's sidecar goes first and the new one is written last, so a sidecar on
    # disk always describes tables written in full by the same run. validate() has checked the
    # file's SHA-256 against the manifest's.
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
                  "attribution_classes": list(ATTRIBUTIONS),
                  "bank_total_classes": list(BANK_CLASSES),
                  "tables": {name: len(frame) for name, frame in tables},
                  "written_at": now_iso(),
                  "script": Path(__file__).name}
    write_json_atomic(sidecar, provenance)
    log.info("%s -> %s", sidecar.name, a.out_dir)


if __name__ == "__main__":
    main()
