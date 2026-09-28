"""
scrape_caixa_scan_advertising.py
================================
Caixa's twenty scanned advertising months, read by OCR and admitted only on the table's own
arithmetic, counted in integer centavos.

Caixa publishes 20 months as image scans with no text layer - all of 2013, January to June 2014,
and April and May 2016 - and Caixa is 16 to 31% of panel deposits, so these are worth more than any
other gap in the advertising data. April and May 2016 alone move that year's coverage from 38.7% to
about 69.5%. Every file is already on disk; nothing here downloads.

Why this is a SEPARATE builder and not an extension of `scrape_caixa_advertising.py`: a scan needs
tolerant label matching, box splitting and glyph repairs, and that tolerance has no business inside
the parser that reads 142 months of clean text at 100% accuracy, which stays provably untouched. The
two builders do NOT write the same row shape: a scanned cell carries a column index, not an agency,
and the two lines tables share only `period`, `amount_brl` and `parse_method`. The interface a panel
merges, at month level, is the monthly table here: `period`, `amount_brl` (the recovered total),
`validated`, `validation_status`, `tolerance_flagged` and `max_gap_cents`. No panel reads it yet.

What OCR gets wrong on these scans
----------------------------------
`diag_ocr_benchmark.py` measured 100% amount recall at 150, 200 and 300 dpi on ~1,100 amounts whose
answer was already known, but on RENDERED DIGITAL pages, which are cleaner than a scan. On the real
scans OCR makes errors the benchmark never showed, and each is handled here:
  - MERGED BOXES. Adjacent cells, or a label and its figure, come back as one box: November 2013's
    whole TOTAL row is '11.996.341,75 9.736.540,75 7.200.055,6413.460.353,42', and labels arrive
    fused to amounts ('VIDEO/PROD.EMAILMARKETING 675.271,12'). The box is split on the amount
    pattern. Between two amounts the two-digit cents make the boundary unambiguous, so that split
    changes no character. The LEFT edge of an amount is less sure: up to three digits in front of
    it are read into it, whatever they belong to. A lone digit touching it is a seam glyph (below)
    and is tested as one. Anything else run straight into it - two or more digits, a digit after a
    dot or comma, a '-', ',', '/' or '(' - may be a tax number, a code, an unread figure or a sign,
    and so may a '%' anywhere in the box, so such a box is left unread, with a note. So is text in
    front of the first amount that itself reads as an amount ('1.000 - 2.345,67'): it is not a
    label, and not a figure the split can vouch for. A digit of a label read into a figure
    ('CANAL21.234,56') cannot be seen this way; unless it is a 0, which changes nothing, it moves
    the figure by at least R$ 10, and the gate below fails the month.
  - COMMA READ AS DOT. '3.070.118.35' for 3.070.118,35 - six to eight cells in each of February to
    April 2014. The dot before the cents is restored to a comma. A string holding a tax number is
    left alone, since tax numbers are dotted digit groups too: a CNPJ with its slash, or with the
    slash misread as a dot or a 1, or lost, and a CPF with or without its hyphen.
  - GLYPH DUPLICATED AT A BOX SEAM. Where two detections overlap, the edge glyph of one is read into
    the other as well: May 2016's '36.047.860,02' is 6.047.860,02 beside '5.861.407,03', and May
    2013's '60,00' is 0,00 beside 'BANNERS/GRANDES FORMATOS', whose S became the 6.
  - WORDS LOSE CHARACTERS. "TOTAL GERAL" comes back as "TOTALGERAL" or "TOTAL GERA", so labels are
    matched on whitespace-stripped text and only PROPOSE which rows are totals; arithmetic decides.

Layout, from the scans themselves
---------------------------------
Each table is a matrix: a leftmost column of category labels (TELEVISAO ABERTA, RADIO, JORNAL,
REVISTA, INTERNET, CINEMA, MERCHANDISING, ...), one column per agency, and at the foot a TOTAL row
(one figure per agency) and a TOTAL GERAL (one figure). The figures are left-aligned in their
column in every month except January 2013, a spreadsheet-grid scan whose figures are right-aligned.
Columns are found from LEFT edges all the same: the centres are the median left edge, position by
position, over the rows that carry a figure in every column, and each figure goes to the ONE nearest
centre. January 2013 still works, because each figure's left edge stays within half a column
spacing of its column's centre, the window a figure must fall in; a figure outside every window is
dropped with a note. The number of columns is decided once per month, because every page of the
table carries the same agencies; deciding it per page once gave June 2013's second page two columns
out of four.

A row's label is the text that starts left of the first column, less two things OCR adds to it. The
side headings FORNECEDORES, VEICULOS, PRODUCAO and SERVICOS run down the table's left edge, rotated
or as stacked letters, and come back as fragments ('FORNECE', 'VEi', 'S', 'R', '0'): a box lying
wholly left of the leftmost label word (three or more letters, wider than tall), or rotated and
starting left of it, is dropped. And a first-column '0,00' fused onto its label comes back as
'BANNERS/GRANDESFORMATOSO,00': that tail is dropped. `label_as_read` keeps the label as OCR
returned it. The gate below certifies the AMOUNTS - each column against its printed total, and the
TOTAL row against the TOTAL GERAL - and not which category each amount is paired with: a figure on
the wrong row of the right column adds up just the same.

The table is read from every page region that holds an image and no text layer: the scanned pages
whole, plus any image covering a quarter of a page on which no word of the text layer lies. That
second case is January 2014's page 2, which is double width - its left half is the scanned
continuation of the table, with the TOTAL rows, and its right half a text supplier list. December
2013's page 2 is double width too, but scanned whole, so it is read whole: its right half is the
supplier list, names and tax numbers with no amount among them, all of it outside every column. The
other pages have a text layer and carry the supplier lists.

The gate
--------
All arithmetic is in integer centavos, compared with ==. The smallest possible gap is one centavo,
about 1.4e-10 of the largest figure, so any relative tolerance is either exactly zero or a hole.

A month is CHECKED only when every agency column has a printed total in the TOTAL row and a TOTAL
GERAL is found; otherwise it is `not_validated` and `validation_reason` says what was missing. The
checks are each column's components against its printed total, and the TOTAL row's sum against the
TOTAL GERAL. Then, on the gaps in centavos:
  validated_exact              every gap is 0.
  validated_within_5_centavos  every gap is within 5 centavos and at least one is not 0. This is
                               registered, never silent: `tolerance_flagged` is True, and the
                               monthly row gives `max_gap_cents` and names each check and its gap.
  not_validated                any gap is above 5 centavos, or the month could not be checked.
`validated` is True for the first two statuses. A month that fails is still written, with every
gap recorded, so nothing is silently dropped and nothing unvalidated is silently used. Every gap is
recorded AS READ, and so is `max_gap_cents`; it and `grand_gap_cents` are None when the month could
not be checked, since a maximum over the checks that happened to be made would read as a verdict.
Where a refused seam repair would move a gap, the gap it would give is recorded beside it (2014-06).

TWO TOLERANCES, NOT ONE. The 5-centavo allowance is there to admit a month whose PRINTED totals
disagree with each other by a slip (May 2014, where every cell is checked exactly by its column and
only the TOTAL GERAL is 4 centavos off). Arithmetic alone cannot tell such a slip from a misread
final digit, which is why the month is flagged rather than folded into `validated_exact`. The
allowance never admits an OCR repair. A repair that changes a character - a comma restored for a
dot, a seam glyph dropped - is kept only if the check it sits in (its column's, or the grand check
for the TOTAL GERAL) then closes to EXACTLY 0 centavos; otherwise it is undone: a restored comma or
a split-off glyph leaves no amount, and a seam candidate leaves the figure as OCR read it (2014-06's
915.422.471,37). Were the allowance applied to repairs, a wrong repair landing within 5 centavos of
a total would pass as validated. A split between two amounts changes no character and needs no such
test. Every repair is recorded on its cell (`amount_as_read`, `amount_brl`, `repair`), and no value
is changed without one: before writing, the builder refuses any cell or printed total that carries
no repair and differs from what OCR read.

Months whose printed tables do not add up
-----------------------------------------
In these the cells are read as printed, and the publisher's own totals disagree with them:
  2013-04  column 2 (NovaSB): the components exceed the printed total by 2.00. Five different
           single-digit changes would each close it, so it is not correctable.
  2013-12  column 3: the components exceed the printed total by 246,240.00, which is exactly the
           printed cell PAINEL ELETRONICO 246.240,00 - the total seems to omit it (an inference).
  2014-05  every column closes exactly; the TOTAL row sums 4 centavos below the TOTAL GERAL. This
           is the month the 5-centavo allowance admits, flagged. Checked by eye on the scan, the
           disagreement is in the print: the printed TOTAL row adds up to 40.272.216,56 and the
           printed TOTAL GERAL is 40.272.216,60. Arithmetic alone could not have said so.
  2014-06  columns 0-2 close exactly. Column 3's printed total, 15.422.471,37, is read by OCR as
           915.422.471,37 - a '9' duplicated at the seam with its neighbour 9.506.178,29 - and the
           column's cells sum to 15,222,714.57, 199,756.80 short of the printed figure, so the seam
           repair does not close and is refused. The checks, `gap_detail` and `max_gap_cents` hold
           the gaps AS READ: column 3 at -90,019,975,680 centavos and the grand check at
           +90,000,000,005, an OCR artifact of some R$900m and not a gap in the document. What the
           refused repair would give is recorded beside them - column 3 at -19,975,680 and the grand
           check at +5 - in the checks' `gap_cents_if_seam_glyph_dropped` (with the suspect figure
           in `stated_suspect_text`), in `gap_detail`, and in the month's
           `max_gap_cents_if_seam_glyph_dropped`.

Outputs (paths.AWARENESS_PROC)
------------------------------
  caixa_scan_advertising_lines.{parquet,csv}     one row per recovered cell, with its repair, and
                                                 its label cleaned and as read
  caixa_scan_advertising_monthly.{parquet,csv}   one row per month, with the reconciliation; the
                                                 interface a panel merges at month level
  caixa_scan_advertising_checks.{parquet,csv}    every arithmetic check, with its gap in centavos
                                                 as read and, where a refused seam repair would
                                                 move it, the gap that repair would give

Usage
-----
  python scrape_caixa_scan_advertising.py                  # all twenty months
  python scrape_caixa_scan_advertising.py --periods 201301 201604
  python scrape_caixa_scan_advertising.py --dpi 300 --debug 201406
"""

from __future__ import annotations

import argparse
import logging
import re
import unicodedata
import zipfile
from dataclasses import dataclass, field, replace
from pathlib import Path

from utils.venv_guard import ensure_project_venv
ensure_project_venv(__file__)

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

from utils import paths

PROC = paths.AWARENESS_PROC
CAIXA_RAW = paths.AWARENESS_RAW / "caixa"
DPI = 300
# The allowance for a month whose PRINTED totals disagree with each other. It admits a month, and
# never an OCR repair: a repair must close its column to exactly 0 (see the module docstring).
TOLERANCE_CENTS = 5
ROW_TOL = 0.6                  # fraction of median glyph height that still counts as one row
# An image on a page that also has a text layer is read as table only when it covers at least this
# share of the page and no word of the text layer lies on it. The share keeps out the small tiles
# some scans are cut into, which can be wordless on a supplier-list page without holding a table.
TABLE_REGION_MIN_SHARE = 0.25
SEAM_OVERLAP_PX = 2            # two detections overlapping by more than this may share a glyph

AMOUNT_TOKEN = re.compile(r"-?\d[\d.]*,\d{2}")
# A dotted integer with no cents is a real amount in these documents, as it is in the gazettes.
INT_TOKEN = re.compile(r"\d{1,3}(?:\.\d{3})+")
# One printed amount inside a longer string. The cents are always two digits, so the end of each
# amount is unambiguous even with nothing between two of them: '7.200.055,6413.460.353,42'.
AMOUNT_IN_TEXT = re.compile(r"\d{1,3}(?:\.\d{3})*,\d{2}")
# The characters that, run straight into an amount's first digit, leave its left edge in doubt:
# digits and the punctuation of tax numbers, codes, dates and signs.
EDGE_RUN = re.compile(r"[\d.,/(\-]*$")
# Tax numbers are dotted digit groups too, and a comma repair would make one an amount
# ('123.456.789-01' into 123.456,78). A CNPJ is recognised with its slash, or with the slash misread
# as a dot or a 1, or lost; a CPF with or without its hyphen. The lookarounds keep the last two from
# matching inside merged amounts, which are dotted digit groups end to end ('2.190.740.00264...').
CNPJ = re.compile(r"\d{2}\.\d{3}\.\d{3}/\d{4}")
CNPJ_NO_SLASH = re.compile(r"(?<![\d.])\d{2}\.\d{3}\.\d{3}[./1]?\d{4}-?\d{2}(?![\d.,])")
CPF = re.compile(r"(?<![\d.])\d{3}\.\d{3}\.\d{3}-?\d{2}(?![\d.,])")
TAX_NUMBERS = (CNPJ, CNPJ_NO_SLASH, CPF)
# A first-column '0,00' OCR fused onto the end of a label ('BANNERS/GRANDESFORMATOSO,00').
ZERO_TAIL = re.compile(r"\s*[O0][.,][O0]{2}$")
# A decimal comma read as a dot. The strict form requires that no digit follow the cents; the loose
# form also accepts a glyph straight after them ('3.410.000.006'), and is used only where no valid
# amount already stands, so that it cannot cut into a figure that reads correctly.
DOT_CENTS_STRICT = re.compile(r"(\d{1,3}(?:\.\d{3})+)\.(\d{2})(?!\d)")
DOT_CENTS_LOOSE = re.compile(r"(?<![\d,])(\d{1,3}(?:\.\d{3})+)\.(\d{2})")

COMMA_AS_DOT = "comma_as_dot"
SEAM_GLYPH = "seam_glyph"
SPLIT_MERGED = "split_merged"
# The label a cell carries when several repairs touched it: the most invasive first. A seam repair
# drops a character, a comma repair changes one, a split changes none.
REPAIR_PRECEDENCE = (SEAM_GLYPH, COMMA_AS_DOT, SPLIT_MERGED)
# The repairs that change a character of what OCR read, and so must close their column exactly.
CHARACTER_REPAIRS = frozenset({SEAM_GLYPH, COMMA_AS_DOT})

EXACT = "validated_exact"
WITHIN = "validated_within_5_centavos"
NOT_VALIDATED = "not_validated"


def fold(text: str) -> str:
    s = unicodedata.normalize("NFD", str(text).upper())
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def flat(text: str) -> str:
    """Folded and stripped of whitespace: the form every label test uses."""
    return re.sub(r"\s", "", fold(text))


def cents_of(token: str) -> int | None:
    """The amount a box states, in integer centavos, or None when the box is not one amount."""
    t = token.replace("R$", "").replace(" ", "")
    neg = t.startswith("-") or (t.startswith("(") and t.endswith(")"))
    t = t.strip("-()")
    if AMOUNT_TOKEN.fullmatch(t):
        whole, cents = t.rsplit(",", 1)
        v = int(whole.replace(".", "")) * 100 + int(cents)
    elif INT_TOKEN.fullmatch(t):
        v = int(t.replace(".", "")) * 100
    else:
        return None
    return -v if neg else v


def brl(cents: int | None) -> float | None:
    return None if cents is None else cents / 100


def fmt(cents: int | None) -> str:
    return "-" if cents is None else f"{cents / 100:,.2f}"


@dataclass
class Box:
    text: str
    x0: float
    x1: float
    y0: float
    y1: float
    conf: float
    page: int
    uid: int = -1                        # the OCR detection this box came from
    part: int = 0                        # which piece of that detection, once it is split
    raw: str = ""                        # the detection's text exactly as OCR returned it
    repairs: tuple[str, ...] = ()
    detail: str = ""
    # Why the box is not read as a figure although its text may parse as one; empty when it is read.
    unread: str = ""

    @property
    def xmid(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def ymid(self) -> float:
        return (self.y0 + self.y1) / 2

    @property
    def cents(self) -> int | None:
        return None if self.unread else cents_of(self.text)

    @property
    def key(self) -> tuple[int, int]:
        return (self.uid, self.part)


@dataclass
class Check:
    kind: str                            # "column" or "grand"
    column: int                          # -1 for the grand check
    label: str
    stated: int
    components: int
    stated_box: Box | None

    @property
    def gap(self) -> int:
        return self.components - self.stated


def check_name(key: tuple[str, int]) -> str:
    return f"column {key[1]}" if key[0] == "column" else "the grand check"


@dataclass
class Table:
    k: int = 0
    labels: list[str] = field(default_factory=list)
    labels_read: list[str] = field(default_factory=list)
    pages: list[int] = field(default_factory=list)
    values: list[list[int | None]] = field(default_factory=list)
    boxes: list[list[Box | None]] = field(default_factory=list)
    components: list[int] = field(default_factory=list)
    total_row: int | None = None
    grand_row: int | None = None
    checks: dict[tuple[str, int], Check] = field(default_factory=dict)
    reason: str | None = None            # why the month cannot be checked, when it cannot
    notes: list[str] = field(default_factory=list)

    def check_for(self, row: int, col: int) -> tuple[str, int] | None:
        """The check a cell is tested by: its column's, or the grand check for the TOTAL GERAL."""
        if row == self.grand_row:
            return ("grand", -1)
        if row in self.components or row == self.total_row:
            return ("column", col)
        return None

    def where(self, key: tuple[int, int]) -> tuple[int, int] | None:
        for i, line in enumerate(self.boxes):
            for j, b in enumerate(line):
                if b is not None and b.key == key:
                    return i, j
        return None

    def gaps(self) -> dict[tuple[str, int], int]:
        return {k: c.gap for k, c in self.checks.items()}


@dataclass
class MonthParse:
    period: str
    cells: list[dict] = field(default_factory=list)
    checks: list[dict] = field(default_factory=list)
    repairs: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    summary: dict = field(default_factory=dict)
    n_boxes: int = 0
    mean_conf: float = float("nan")
    regions: str = ""


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------
def source_bytes(source_file: str) -> bytes | None:
    """PDF bytes for a `source_file`, which may name a member inside a zip."""
    name, _, member = str(source_file).partition("::")
    hit = next(CAIXA_RAW.rglob(name), None)
    if hit is None:
        return None
    if not member:
        return hit.read_bytes()
    with zipfile.ZipFile(hit) as zf:
        cand = [n for n in zf.namelist() if n == member or n.endswith("/" + member)] or \
               [n for n in zf.namelist() if n.lower().endswith(".pdf")]
        return zf.read(cand[0]) if cand else None


def table_regions(pdf: bytes) -> list[tuple[int, tuple[float, float, float, float] | None]]:
    """(page index, clip) for every region that holds the table; a clip of None is the whole page.

    A page with no text layer is a scan and is read whole. A page WITH a text layer can still carry
    part of the table as an image beside the text: January 2014's page 2 is double width, the
    scanned table's continuation on the left and a text supplier list on the right, and reading
    only text-less pages lost its TOTAL rows (the month recorded 38,118,015.49 of a printed
    73,783,391.67). Every text page here also sits on a full-page background image, so an image
    counts only when no word of the text layer lies on it.
    """
    import fitz
    doc = fitz.open(stream=pdf, filetype="pdf")
    out: list[tuple[int, tuple[float, float, float, float] | None]] = []
    try:
        for i in range(doc.page_count):
            page = doc[i]
            words = page.get_text("words")
            if not words:
                out.append((i, None))
                continue
            area = page.rect.get_area()
            seen = set()
            for info in page.get_image_info():
                r = fitz.Rect(info["bbox"]) & page.rect
                key = tuple(round(v, 1) for v in r)
                if key in seen or r.get_area() < TABLE_REGION_MIN_SHARE * area:
                    continue
                seen.add(key)
                if not any(r.contains(fitz.Point((w[0] + w[2]) / 2, (w[1] + w[3]) / 2))
                           for w in words):
                    out.append((i, (r.x0, r.y0, r.x1, r.y1)))
    finally:
        doc.close()
    return out


def read_boxes(engine, pdf: bytes, regions, dpi: int) -> list[Box]:
    import cv2
    import fitz
    doc = fitz.open(stream=pdf, filetype="pdf")
    out: list[Box] = []
    scale = dpi / 72
    try:
        for i, clip in regions:
            if i >= doc.page_count:
                continue
            rect = fitz.Rect(clip) if clip is not None else None
            pix = doc[i].get_pixmap(dpi=dpi, clip=rect)
            img = cv2.imdecode(np.frombuffer(pix.tobytes("png"), np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            # A clip's pixels start at its own corner. Shifting them back puts every box on the
            # page's pixel grid, so two regions of one page share one coordinate system.
            dx, dy = (rect.x0 * scale, rect.y0 * scale) if rect is not None else (0.0, 0.0)
            result, _ = engine(img)
            for quad, text, conf in result or []:
                xs = [p[0] for p in quad]
                ys = [p[1] for p in quad]
                out.append(Box(str(text), min(xs) + dx, max(xs) + dx, min(ys) + dy, max(ys) + dy,
                               float(conf), i + 1))
    finally:
        doc.close()
    return out


def describe_regions(regions) -> str:
    return ", ".join(f"p{i + 1}" if clip is None else
                     f"p{i + 1}[image {','.join(f'{v:.0f}' for v in clip)}]" for i, clip in regions)


# ---------------------------------------------------------------------------
# Repairs: every one is recorded on the box it changes
# ---------------------------------------------------------------------------
def comma_for_dot(text: str) -> tuple[str, list[int]]:
    """`text` with each decimal comma OCR read as a dot restored, and the positions changed.

    The replacement keeps the length, so a position in the result is a position in `text`. A string
    holding a tax number (a CNPJ, with or without its slash, or a CPF) is left alone, since tax
    numbers are dotted digit groups too.
    """
    if any(p.search(text) for p in TAX_NUMBERS):
        return text, []
    chars = list(text)
    changed = [m.start(2) - 1 for m in DOT_CENTS_STRICT.finditer(text)]
    for i in changed:
        chars[i] = ","
    strict = "".join(chars)
    # The loose form runs only on the stretches between amounts that already read correctly, each
    # searched as a string of its own.
    pos = 0
    spans = [(m.start(), m.end()) for m in AMOUNT_IN_TEXT.finditer(strict)]
    for s, e in spans + [(len(strict), len(strict))]:
        for m in DOT_CENTS_LOOSE.finditer(strict[pos:s]):
            i = pos + m.start(2) - 1
            chars[i] = ","
            changed.append(i)
        pos = e
    return "".join(chars), sorted(changed)


def edge_doubt(text: str, toks: list[re.Match]) -> str:
    """Why the left edge of an amount inside `text` is in doubt, or '' when every edge is sound.

    Up to three digits in front of an amount are read into it, whatever they belong to; between two
    amounts the cents of the first fix the boundary, but nothing fixes the left edge. A lone digit
    touching an amount is a seam glyph and is tested as one. Anything else run straight into it -
    two or more digits, a digit after a dot or comma, a '-', ',', '/' or '(' - may be a tax number,
    a code, an unread figure or a sign, and a '%' anywhere makes the figures percentages. A split
    there would read a figure the table may not print, with nothing to mark it but `split_merged`.
    """
    if "%" in text:
        return "it holds a '%', so its figures may be percentages"
    end = 0
    for t in toks:
        run = EDGE_RUN.search(text, end, t.start()).group()
        if run and not (len(run) == 1 and run.isdigit()):
            return (f"{run!r} runs into {t.group()!r}, so where that amount starts is in doubt "
                    "(a tax number, a code, an unread figure or a sign)")
        end = t.end()
    return ""


def split_box(b: Box, text: str, comma_at: list[int], refused: set, notes: list[str]) -> list[Box]:
    """The amounts inside a box that is not itself one amount, each as a box of its own.

    The label in front of the first amount is kept as a label box; x-positions are shared out in
    proportion to the characters, which is close enough for the nearest-centre assignment. A digit
    touching an amount that no amount takes - '1.008.076,882' - is a glyph duplicated from the
    neighbouring box at the seam. Dropping it changes what was read, so it is a seam repair, and a
    refused one leaves that cell unread. A box whose amounts have a doubtful left edge (see
    `edge_doubt`) is left unread whole, as OCR returned it.
    """
    toks = list(AMOUNT_IN_TEXT.finditer(text))
    if not toks:
        return [replace(b, text=text)]
    doubt = edge_doubt(text, toks)
    if doubt:
        notes.append(f"p{b.page} {b.raw!r} left unread: {doubt}")
        return [replace(b, unread=doubt)]
    glyphs: list[list[str]] = []
    for idx, t in enumerate(toks):
        prev_end = toks[idx - 1].end() if idx else 0
        next_start = toks[idx + 1].start() if idx + 1 < len(toks) else len(text)
        g = []
        if t.start() > prev_end and text[t.start() - 1].isdigit():
            g.append(f"leading {text[t.start() - 1]!r}")
        if t.end() < next_start and text[t.end()].isdigit():
            g.append(f"trailing {text[t.end()]!r}")
        glyphs.append(g)
    # The box was merged when it held more than one amount and its seam glyphs: a second amount, or
    # a label. A lone amount with a glyph stuck to it is a seam repair and nothing else.
    rest = re.sub(r"\s", "", AMOUNT_IN_TEXT.sub("", text))
    merged = len(toks) > 1 or len(rest) > sum(len(g) for g in glyphs)
    n = max(1, len(text))
    w = b.x1 - b.x0
    out: list[Box] = []
    pre = text[:toks[0].start()].strip()
    if pre:
        # A label piece that reads as an amount ('1.000 -' of '1.000 - 2.345,67') is not a label,
        # and no split vouches for it as a figure: it is kept, unread, so no value enters a cell
        # without a repair label saying where it came from.
        unread = ("text in front of the first amount reads as an amount, which no split vouches "
                  "for" if cents_of(pre) is not None else "")
        if unread:
            notes.append(f"p{b.page} {pre!r} of {b.raw!r} left unread: {unread}")
        out.append(replace(b, text=pre, x1=b.x0 + w * toks[0].start() / n, part=0,
                           repairs=(), detail="", unread=unread))
    for idx, t in enumerate(toks):
        s, e = t.start(), t.end()
        steps, said = [], []
        if any(s <= i < e for i in comma_at):
            steps.append(COMMA_AS_DOT)
            said.append(f"comma read as dot, restored: {b.raw!r}")
        if merged:
            steps.append(SPLIT_MERGED)
            said.append(f"split from {b.raw!r}")
        if glyphs[idx]:
            if (b.uid, SEAM_GLYPH) in refused:
                continue
            steps.append(SEAM_GLYPH)
            said.append(f"dropped {' and '.join(glyphs[idx])} touching {t.group()!r}: "
                        "a seam glyph")
        out.append(replace(b, text=t.group(), x0=b.x0 + w * s / n, x1=b.x0 + w * e / n,
                           part=idx + 1, repairs=tuple(steps), detail="; ".join(said)))
    return out


def prepare(raw: list[Box], refused: set, notes: list[str]) -> list[Box]:
    """The boxes with comma repairs and splits applied, less any repair already refused."""
    out: list[Box] = []
    for b in raw:
        if b.cents is not None:
            out.append(b)
            continue
        text, comma_at = b.text, []
        if (b.uid, COMMA_AS_DOT) not in refused:
            text, comma_at = comma_for_dot(b.text)
        out.extend(split_box(b, text, comma_at, refused, notes))
    return out


def seam_candidates(boxes: list[Box]) -> list[tuple[int, str, str]]:
    """(index, new text, why) for amounts overlapping a neighbour in their row by one edge glyph."""
    out = []
    index = {id(b): i for i, b in enumerate(boxes)}
    for row in group_rows(boxes):
        row = sorted(row, key=lambda b: b.x0)
        for a, b in zip(row, row[1:]):
            if b.x0 >= a.x1 - SEAM_OVERLAP_PX:
                continue
            if b.cents is not None and len(b.text) > 1 and cents_of(b.text[1:]) is not None:
                out.append((index[id(b)], b.text[1:],
                            f"dropped leading {b.text[0]!r} of {b.text!r}, which overlaps "
                            f"{a.text!r}: a seam glyph"))
            if a.cents is not None and len(a.text) > 1 and cents_of(a.text[:-1]) is not None:
                out.append((index[id(a)], a.text[:-1],
                            f"dropped trailing {a.text[-1]!r} of {a.text!r}, which overlaps "
                            f"{b.text!r}: a seam glyph"))
    return out


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------
def group_rows(boxes: list[Box]) -> list[list[Box]]:
    """Boxes grouped into visual rows by vertical overlap, one page at a time."""
    rows: list[list[Box]] = []
    for page in sorted({b.page for b in boxes}):
        page_boxes = sorted((b for b in boxes if b.page == page), key=lambda b: (b.ymid, b.x0))
        if not page_boxes:
            continue
        height = float(np.median([b.y1 - b.y0 for b in page_boxes])) or 1.0
        current = [page_boxes[0]]
        for b in page_boxes[1:]:
            if abs(b.ymid - float(np.mean([c.ymid for c in current]))) <= ROW_TOL * height:
                current.append(b)
            else:
                rows.append(sorted(current, key=lambda c: c.x0))
                current = [b]
        rows.append(sorted(current, key=lambda c: c.x0))
    return rows


def column_count(rows: list[list[Box]]) -> int:
    """How many agency columns the table has, taken over every page of the month at once.

    The widest recurring row, not the most common one: many category rows leave an agency blank, so
    in January 2013 the modal row carries two figures while the table has three columns. Requiring
    the width to recur ignores a lone row with a stray number in it. One count for the month,
    because the pages carry the same agencies; a page's own count once gave June 2013's second page
    two columns, and the month could not be stacked.
    """
    counts = [sum(1 for b in r if b.cents is not None) for r in rows]
    counts = [c for c in counts if c >= 2]
    if not counts:
        return 0
    tally = pd.Series(counts).value_counts()
    repeated = [int(c) for c in tally[tally >= 2].index]
    return max(repeated) if repeated else int(max(counts))


def column_centres(rows: list[list[Box]], page: int, k: int) -> list[float] | None:
    """The left edge of each column: position by position, the median over rows with k figures.

    The figures are left-aligned in every month except January 2013, whose right-aligned figures
    still keep their left edges within half a column spacing of the centre. Clustering right edges
    made the ranges of neighbouring columns overlap, and a figure falling in two ranges blanked its
    cell. A row that carries a figure in every column says which edge belongs to which column
    without any clustering at all.
    """
    seeds = [sorted(b.x0 for b in r if b.page == page and b.cents is not None) for r in rows]
    seeds = [s for s in seeds if len(s) == k]
    if not seeds:
        return None
    return [float(x) for x in np.median(np.array(seeds), axis=0)]


def is_sidebar(b: Box, frame_x: float | None) -> bool:
    """A fragment of the side headings (FORNECEDORES, VEICULOS, PRODUCAO, SERVICOS) down the left.

    They lie left of every label word, rotated ('FORNECE', 'VEi') or as stacked letters ('S', 'R',
    '0'). A box lying wholly left of the leftmost label word, or rotated and starting left of it, is
    one: no label of the table lies left of its own leftmost word.
    """
    if frame_x is None:
        return False
    rotated = b.y1 - b.y0 > b.x1 - b.x0 and len(b.text.strip()) >= 2
    return b.x1 <= frame_x or (rotated and b.x0 < frame_x)


def page_matrix(rows: list[list[Box]], page: int, centres: list[float], notes: list[str]):
    """One page's (labels, labels as read, values, boxes): a row per label, each figure in its
    nearest column. A label drops the sidebar fragments and a fused '0,00' tail; as read, it keeps
    them."""
    k = len(centres)
    spacing = float(min(np.diff(centres))) if k > 1 else 400.0
    first_x = centres[0] - 0.3 * spacing
    lines = [row for row in rows if any(b.page == page for b in row)
             and any(b.cents is not None for b in row)]
    # The left edge of the label area: the leftmost word of three or more letters, wider than
    # tall. Rotated and single-letter sidebar fragments are neither, so they cannot set it.
    words = [b.x0 for row in lines for b in row if b.x0 < first_x and b.cents is None
             and b.x1 - b.x0 >= b.y1 - b.y0 and sum(c.isalpha() for c in b.text) >= 3]
    frame_x = min(words) if words else None
    labels: list[str] = []
    labels_read: list[str] = []
    values: list[list[int | None]] = []
    boxes: list[list[Box | None]] = []
    for row in lines:
        # A label is any text that STARTS left of the first column. Testing where it ends instead
        # loses the labels OCR fused to a figure ('BANNERS/GRANDES FORMATOS0,00'), whose label
        # piece, once split off, reaches into the first column.
        texts = [b for b in row if b.x0 < first_x and b.cents is None]
        label_read = " ".join(b.text for b in texts).strip()
        label = " ".join(t for t in (ZERO_TAIL.sub("", b.text).strip() for b in texts
                                     if not is_sidebar(b, frame_x)) if t)
        line: list[int | None] = [None] * k
        where: list[Box | None] = [None] * k
        hits = [0] * k
        for b in row:
            if b.cents is None:
                continue
            d = [abs(b.x0 - x) for x in centres]
            j = int(np.argmin(d))
            if d[j] > 0.5 * spacing:
                # Recorded: if the figure belongs to the table, its column comes up short, and the
                # note says where the shortfall went.
                notes.append(f"p{page} {label[:30]!r}: {b.text!r} lies outside every column, so "
                             "it is not read")
                continue
            hits[j] += 1
            # Two figures in one cell is a misplaced box; leave the cell empty rather than guess,
            # and let the arithmetic report the shortfall.
            line[j], where[j] = (b.cents, b) if hits[j] == 1 else (None, None)
        for j in range(k):
            if hits[j] > 1:
                notes.append(f"p{page} {label[:30]!r}: {hits[j]} figures fell in column {j}, "
                             "so that cell is left empty")
        if all(v is None for v in line):
            continue
        labels.append(label)
        labels_read.append(label_read)
        values.append(line)
        boxes.append(where)
    return labels, labels_read, values, boxes


def evaluate(boxes: list[Box]) -> Table:
    """The month's table, its TOTAL and TOTAL GERAL rows, and every check, in integer centavos."""
    t = Table()
    rows = group_rows(boxes)
    t.k = column_count(rows)
    if t.k < 1:
        t.reason = "no row carries two or more figures, so no columns were found"
        return t
    pages = sorted({b.page for b in boxes})
    centres = {p: column_centres(rows, p, t.k) for p in pages}
    found = [c for c in centres.values() if c is not None]
    for p in pages:
        cen = centres[p]
        if cen is None:
            if not found:
                continue
            # A page with no row carrying every column borrows another page's columns.
            cen = found[0]
            t.notes.append(f"p{p}: no row carries all {t.k} figures; "
                           "columns taken from another page")
        labels, labels_read, values, where = page_matrix(rows, p, cen, t.notes)
        t.labels += labels
        t.labels_read += labels_read
        t.values += values
        t.boxes += where
        t.pages += [p] * len(labels)
    if not t.labels:
        t.reason = "no amount rows found"
        return t

    # The label PROPOSES the total rows and the arithmetic chooses among them. Every row labelled
    # TOTAL stays out of the components: the TOTAL row and the TOTAL GERAL both sit at the foot, and
    # whichever one a rule let through would inflate the column sums.
    proposed = [i for i, lab in enumerate(t.labels) if "TOTAL" in flat(lab)]
    t.components = [i for i in range(len(t.labels)) if i not in proposed]
    sums = [sum(t.values[i][j] for i in t.components if t.values[i][j] is not None)
            for j in range(t.k)]
    best = None
    for i in proposed:
        stated = t.values[i]
        present = [j for j in range(t.k) if stated[j] is not None]
        if len(present) < 2:                    # a single figure is the TOTAL GERAL, not the row
            continue
        score = (t.k - len(present), sum(abs(sums[j] - stated[j]) for j in present))
        if best is None or score < best[0]:
            best = (score, i)
    if best is None:
        t.reason = "no row labelled TOTAL carries two or more figures"
    else:
        t.total_row = best[1]
    singles = [i for i in proposed if i != t.total_row
               and sum(v is not None for v in t.values[i]) == 1]
    geral = [i for i in singles if "GERA" in flat(t.labels[i])]
    pick = geral if geral else singles
    if len(pick) == 1:
        t.grand_row = pick[0]

    if t.total_row is not None:
        stated = t.values[t.total_row]
        for j in range(t.k):
            if stated[j] is not None:
                t.checks[("column", j)] = Check("column", j, t.labels[t.total_row], stated[j],
                                                sums[j], t.boxes[t.total_row][j])
        missing = [j for j in range(t.k) if stated[j] is None]
        if missing:
            t.reason = (f"column(s) {', '.join(map(str, missing))} have no printed total in the "
                        f"TOTAL row, so the month cannot be checked on every column")
    if t.grand_row is None:
        why = (f"{len(pick)} single-figure TOTAL rows, none singled out as TOTAL GERAL" if pick
               else "no TOTAL GERAL found")
        t.reason = f"{t.reason}; {why}" if t.reason else why
    elif t.total_row is not None and all(v is not None for v in t.values[t.total_row]):
        j = next(j for j, v in enumerate(t.values[t.grand_row]) if v is not None)
        t.checks[("grand", -1)] = Check("grand", -1, t.labels[t.grand_row],
                                        t.values[t.grand_row][j],
                                        sum(t.values[t.total_row]), t.boxes[t.grand_row][j])
    return t


def unverified_repairs(t: Table) -> dict[tuple[int, str], str]:
    """(detection, repair) -> why, for each character repair whose own check does not close exactly.

    The 5-centavo allowance plays no part here: it admits a month whose printed totals disagree,
    never a change to what OCR read. A repair is kept only at a gap of exactly 0.
    """
    bad: dict[tuple[int, str], str] = {}
    gaps = t.gaps()
    for i, line in enumerate(t.boxes):
        for j, b in enumerate(line):
            if b is None or not CHARACTER_REPAIRS.intersection(b.repairs):
                continue
            key = t.check_for(i, j)
            gap = gaps.get(key) if key is not None else None
            if gap == 0:
                continue
            why = ("its cell is in no check" if key is None else
                   f"{check_name(key)} is not checked" if gap is None else
                   f"with it {check_name(key)} does not close exactly (gap {gap:+,d} centavos)")
            for r in CHARACTER_REPAIRS.intersection(b.repairs):
                bad[(b.uid, r)] = why
    return bad


@dataclass
class RefusedSeam:
    """A seam candidate that was not taken: the box, its text as it stood, and the candidate."""
    key: tuple[int, int]
    text: str
    new: str
    why: str


def settle_seams(boxes: list[Box], refused: set,
                 notes: list[str]) -> tuple[list[Box], Table, list[RefusedSeam]]:
    """Drop a duplicated seam glyph wherever doing so closes that cell's own check EXACTLY.

    A candidate is accepted only if the check of the cell it changes goes from non-zero to exactly
    0 centavos and no check that was 0 moves. Two candidates that would close the same check are
    ambiguous, and neither is taken. The 5-centavo allowance plays no part. The candidates not
    taken are returned, so that what each would have given can be recorded beside the gaps as read.
    """
    table = evaluate(boxes)
    tried: set[tuple[tuple[int, int], str]] = set()
    passed_over: list[RefusedSeam] = []
    while any(g != 0 for g in table.gaps().values()) or table.reason:
        gaps0 = table.gaps()
        passing: dict[tuple[str, int], list] = {}
        for idx, new, why in seam_candidates(boxes):
            b = boxes[idx]
            if (b.uid, SEAM_GLYPH) in refused or (b.key, new) in tried:
                continue
            trial = list(boxes)
            trial[idx] = replace(b, text=new,
                                 repairs=tuple(dict.fromkeys(b.repairs + (SEAM_GLYPH,))),
                                 detail=f"{b.detail}; {why}" if b.detail else why)
            t2 = evaluate(trial)
            cell = t2.where(b.key)
            check = t2.check_for(*cell) if cell else None
            gaps1 = t2.gaps()
            closes = (check is not None and gaps1.get(check) == 0 and gaps0.get(check) != 0)
            keeps = all(gaps1.get(c) == 0 for c, g in gaps0.items() if g == 0)
            if closes and keeps:
                passing.setdefault(check, []).append((idx, new, why, trial, t2))
                continue
            tried.add((b.key, new))
            passed_over.append(RefusedSeam(b.key, b.text, new, why))
            if check is None:
                notes.append(f"seam candidate refused, its cell is in no check: {why}")
            elif not closes:
                after = f"{gaps1[check]:+,d}" if check in gaps1 else "absent"
                notes.append(f"seam candidate refused, {check_name(check)} would not close "
                             f"exactly (gap {after} centavos): {why}")
            else:
                notes.append(f"seam candidate refused, it would move a check that closes: {why}")
        accepted = False
        for check, found in passing.items():
            if len(found) > 1:
                for idx, new, why, *_ in found:
                    tried.add((boxes[idx].key, new))
                    passed_over.append(RefusedSeam(boxes[idx].key, boxes[idx].text, new, why))
                notes.append(f"{len(found)} seam candidates would each close {check_name(check)}; "
                             "ambiguous, none taken")
                continue
            idx, new, why, trial, t2 = found[0]
            boxes, table = trial, t2
            accepted = True
            break
        if not accepted:
            break
    return boxes, table, passed_over


@dataclass
class Solved:
    boxes: list[Box]
    table: Table
    notes: list[str]
    refused: dict[tuple[int, str], str]   # (detection, repair) -> why it was refused
    seams: list[RefusedSeam]              # seam candidates not taken


def solve(raw: list[Box]) -> Solved:
    """Apply the repairs, keep only those that close exactly, and return the settled table."""
    refused: dict[tuple[int, str], str] = {}
    # Each pass refuses at least one new (detection, repair) pair, so this ends in bounded passes.
    for _ in range(2 * len(raw) + 1):
        notes: list[str] = []
        boxes, table, seams = settle_seams(prepare(raw, set(refused), notes), set(refused), notes)
        bad = unverified_repairs(table)
        if not bad:
            return Solved(boxes, table, notes, refused, seams)
        refused.update(bad)
    raise RuntimeError("repair refusal did not settle")


def seam_alternatives(boxes: list[Box], t: Table, seams: list[RefusedSeam]
                      ) -> dict[tuple[str, int], list[tuple[int, RefusedSeam]]]:
    """For each check a refused seam candidate would move: (the gap it would then have, candidate).

    The gaps as read stay the record; this is written beside them, so that a figure OCR misread at
    a seam (2014-06's 915.422.471,37 for 15.422.471,37) is not taken for a gap in the document.
    Each candidate is tried alone on the settled table.
    """
    gaps = t.gaps()
    moved: dict[tuple[str, int], list[tuple[int, RefusedSeam]]] = {}
    for s in seams:
        idx = next((i for i, b in enumerate(boxes) if b.key == s.key and b.text == s.text), None)
        if idx is None:
            continue
        trial = list(boxes)
        trial[idx] = replace(boxes[idx], text=s.new)
        after = evaluate(trial).gaps()
        for check, gap in gaps.items():
            if check in after and after[check] != gap:
                moved.setdefault(check, []).append((after[check], s))
    return moved


# ---------------------------------------------------------------------------
# Month
# ---------------------------------------------------------------------------
def repair_label(b: Box | None) -> str | None:
    if b is None or not b.repairs:
        return None
    return next(r for r in REPAIR_PRECEDENCE if r in b.repairs)


def parse_boxes(period: str, raw: list[Box], dpi: int, debug: bool = False) -> MonthParse:
    """Everything after OCR: the repairs, the table, the gate and the rows written out."""
    out = MonthParse(period=period, n_boxes=len(raw))
    out.mean_conf = float(np.mean([b.conf for b in raw])) if raw else float("nan")
    raw = [replace(b, uid=i, raw=b.text, repairs=(), detail="", unread="")
           for i, b in enumerate(raw)]
    solved = solve(raw)
    boxes, t = solved.boxes, solved.table
    out.notes += t.notes + solved.notes
    for (uid, kind), why in sorted(solved.refused.items()):
        out.notes.append(f"{kind} repair refused, {why}: {raw[uid].text!r} left as read")
    moved = seam_alternatives(boxes, t, solved.seams)

    def gap_list(checks) -> str:
        return "; ".join(f"column {c.column}: {c.gap:+,d}" if c.kind == "column" else
                         f"grand: {c.gap:+,d}" for c in checks)

    gaps = t.gaps()
    if t.reason:
        status, reason = NOT_VALIDATED, t.reason
    elif all(g == 0 for g in gaps.values()):
        status, reason = EXACT, ""
    elif all(abs(g) <= TOLERANCE_CENTS for g in gaps.values()):
        # Admitted, and registered as such. Whether the print or the OCR is off by those centavos
        # is not something the arithmetic can say (2014-05's was settled by eye: the print).
        status = WITHIN
        reason = (f"every gap within {TOLERANCE_CENTS} centavos "
                  f"({gap_list(c for c in t.checks.values() if c.gap != 0)}); admitted and "
                  "flagged; arithmetic does not say whether the print or the OCR is off")
    else:
        status = NOT_VALIDATED
        reason = "gap above 5 centavos: " + gap_list(
            c for c in t.checks.values() if abs(c.gap) > TOLERANCE_CENTS)
    validated = status in (EXACT, WITHIN)

    if debug:
        log.info("  %s: %d rows x %d cols | TOTAL row = %s | TOTAL GERAL = %s | %s", period,
                 len(t.labels), t.k,
                 repr(t.labels[t.total_row]) if t.total_row is not None else "NOT FOUND",
                 repr(t.labels[t.grand_row]) if t.grand_row is not None else "NOT FOUND", status)
        for i, (lab, vals) in enumerate(zip(t.labels, t.values)):
            mark = "T" if i == t.total_row else ("G" if i == t.grand_row else " ")
            log.info("    %s p%d %-36s %s", mark, t.pages[i], lab[:36],
                     " ".join(f"{fmt(v):>15}" for v in vals))

    method = f"rapidocr {dpi}dpi, columns by amount left edge"
    for i in t.components:
        for j in range(t.k):
            v, b = t.values[i][j], t.boxes[i][j]
            if v is None or (v == 0 and not (b and b.repairs)):
                continue
            as_read = cents_of(b.raw)
            out.cells.append({
                "period": period, "page": t.pages[i], "label_as_published": t.labels[i],
                "label_as_read": t.labels_read[i],
                "column_index": j, "amount_brl": brl(v), "amount_cents": v,
                "amount_as_read": brl(as_read), "text_as_read": b.raw,
                "repair": repair_label(b), "repair_detail": b.detail or None,
                "parse_method": method, "validated": validated, "validation_status": status})
            if b.repairs:
                out.repairs.append({"row": t.labels[i], "column": j, "text_as_read": b.raw,
                                    "as_read": as_read, "final": v, "repair": repair_label(b),
                                    "detail": b.detail})

    # The gap each check would have if the one refused seam candidate that moves it were taken.
    # With two or more candidates on one check there is no single alternative, so none is given.
    alt_gap = {key: alts[0][0] for key, alts in moved.items() if len(alts) == 1}
    for c in t.checks.values():
        sb = c.stated_box
        as_read = cents_of(sb.raw) if sb is not None else None
        key = (c.kind, c.column)
        alts = moved.get(key, [])
        suspect = next((s.text for _, s in alts if sb is not None and s.key == sb.key), None)
        out.checks.append({
            "period": period, "check": c.kind, "column_index": c.column, "label": c.label,
            "stated": brl(c.stated), "components": brl(c.components), "diff": brl(c.gap),
            "gap_cents": c.gap,
            "check_result": ("exact" if c.gap == 0 else
                             "within_5_centavos" if abs(c.gap) <= TOLERANCE_CENTS else
                             "outside_tolerance"),
            "exact": c.gap == 0, "within_tolerance": abs(c.gap) <= TOLERANCE_CENTS,
            "stated_as_read": brl(as_read), "stated_text_as_read": sb.raw if sb else None,
            "stated_repair": repair_label(sb), "month_status": status,
            "gap_cents_if_seam_glyph_dropped": alt_gap.get(key),
            "stated_suspect_text": suspect,
            "seam_candidate": "; ".join(f"refused: {s.why}; would give a gap of {g:+,d} centavos"
                                        for g, s in alts) or None})
        if sb is not None and sb.repairs:
            out.repairs.append({"row": c.label, "column": c.column, "text_as_read": sb.raw,
                                "as_read": as_read, "final": c.stated,
                                "repair": repair_label(sb), "detail": sb.detail})

    def gap_text(c: Check) -> str:
        text = (f"column {c.column} ({c.label}): components minus printed total = {c.gap:+,d} "
                "centavos" if c.kind == "column" else
                f"grand: TOTAL row sum minus TOTAL GERAL = {c.gap:+,d} centavos")
        alts = moved.get((c.kind, c.column), [])
        if not alts:
            return text
        return text + " as read, " + ", ".join(
            f"{g:+,d} if the seam glyph is dropped from {s.text!r} (reading {s.new!r})"
            for g, s in alts)

    nonzero = [c for c in t.checks.values() if c.gap != 0]
    grand = t.checks.get(("grand", -1))
    total_row = (sum(t.values[t.total_row]) if t.total_row is not None
                 and all(v is not None for v in t.values[t.total_row]) else None)
    geral = (next(v for v in t.values[t.grand_row] if v is not None)
             if t.grand_row is not None else None)
    amount = sum(c["amount_cents"] for c in out.cells)
    # A month that could not be checked has no largest gap: the maximum over the checks that
    # happened to be made could read 0 and pass for a verdict.
    checked = not t.reason
    out.summary = {
        "n_columns": t.k, "amount_cents": amount, "total_row_cents": total_row,
        "total_geral_cents": geral,
        "n_checks": len(t.checks), "n_checks_exact": sum(c.gap == 0 for c in t.checks.values()),
        "n_checks_within_tolerance": sum(abs(c.gap) <= TOLERANCE_CENTS
                                         for c in t.checks.values()),
        "validated": validated, "validation_status": status, "validation_reason": reason,
        "tolerance_flagged": status == WITHIN,
        "max_gap_cents": (max((abs(c.gap) for c in t.checks.values()), default=None)
                          if checked else None),
        "max_gap_cents_if_seam_glyph_dropped": (
            max(abs(alt_gap.get(k, c.gap)) for k, c in t.checks.items())
            if checked and alt_gap else None),
        "gap_detail": "; ".join(gap_text(c) for c in nonzero) or None,
        "grand_gap_cents": grand.gap if grand and checked else None,
    }
    return out


def parse_month(engine, period: str, pdf: bytes, dpi: int, debug: bool = False) -> MonthParse:
    regions = table_regions(pdf)
    if not regions:
        out = MonthParse(period=period)
        out.notes.append("no image region without a text layer: this month is not a scan")
        out.summary = {"validated": False, "validation_status": NOT_VALIDATED,
                       "validation_reason": "no scanned region", "tolerance_flagged": False}
        return out
    out = parse_boxes(period, read_boxes(engine, pdf, regions, dpi), dpi, debug)
    out.regions = describe_regions(regions)
    return out


def unlabelled_changes(frames: dict[str, pd.DataFrame]) -> list[str]:
    """Every cell or printed total whose value is not what OCR read and that carries no repair.

    The rule that no value changes without a repair label is kept by construction; this checks the
    rows about to be written, so that a path that breaks it stops the run instead of reaching the
    outputs. An unread value (as read None) with no repair is caught too, since NaN equals nothing.
    """
    bad = []
    for name, value, read, repair, what in (
            ("caixa_scan_advertising_lines", "amount_brl", "amount_as_read", "repair",
             "text_as_read"),
            ("caixa_scan_advertising_checks", "stated", "stated_as_read", "stated_repair",
             "stated_text_as_read")):
        df = frames[name]
        if df.empty or repair not in df:
            continue
        bare = df[df[repair].isna()]
        same = (bare[read] * 100).round().eq((bare[value] * 100).round())
        bad += [f"{r['period']} {name}: {r[what]!r} as read {r[read]}, written {r[value]}"
                for _, r in bare[~same].iterrows()]
    return bad


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--periods", nargs="*", help="default: every scanned month")
    ap.add_argument("--dpi", type=int, default=DPI)
    ap.add_argument("--debug", nargs="*", default=[], help="print the rebuilt matrix for a period")
    ap.add_argument("--out-dir", type=Path, default=PROC)
    a = ap.parse_args()

    month = pd.read_parquet(PROC / "caixa_advertising_monthly.parquet")
    scan = month[month.status == "unreadable_scan"].sort_values("period")
    if a.periods:
        scan = scan[scan.period.isin(a.periods)]
    if scan.empty:
        raise SystemExit("no scanned month selected")

    from rapidocr_onnxruntime import RapidOCR
    engine = RapidOCR()
    log.info("reading %d scanned months at %d dpi", len(scan), a.dpi)

    cells, checks, months = [], [], []
    for r in scan.itertuples():
        pdf = source_bytes(r.source_file)
        if pdf is None:
            log.warning("  %s: %s not on disk", r.period, r.source_file)
            continue
        p = parse_month(engine, str(r.period), pdf, a.dpi, debug=str(r.period) in a.debug)
        s = p.summary
        cells.extend(p.cells)
        checks.extend(p.checks)
        months.append({
            "period": p.period, "n_columns": s.get("n_columns"), "n_cells": len(p.cells),
            "amount_brl": brl(s.get("amount_cents")),
            "total_row_brl": brl(s.get("total_row_cents")),
            "total_geral_brl": brl(s.get("total_geral_cents")),
            "n_checks": s.get("n_checks", 0), "n_checks_exact": s.get("n_checks_exact", 0),
            "n_checks_within_tolerance": s.get("n_checks_within_tolerance", 0),
            "validated": s["validated"], "validation_status": s["validation_status"],
            "validation_reason": s["validation_reason"] or None,
            "tolerance_flagged": s["tolerance_flagged"], "max_gap_cents": s.get("max_gap_cents"),
            "max_gap_cents_if_seam_glyph_dropped": s.get("max_gap_cents_if_seam_glyph_dropped"),
            "grand_gap_cents": s.get("grand_gap_cents"), "gap_detail": s.get("gap_detail"),
            "n_repairs": len(p.repairs),
            "repairs": "; ".join(
                f"{x['row'][:30]!r} col {x['column']}: {x['text_as_read']!r} -> "
                f"{fmt(x['final'])} ({x['repair']})" for x in p.repairs) or None,
            "n_boxes": p.n_boxes, "mean_conf": round(p.mean_conf, 3), "regions_read": p.regions,
            "source_file": r.source_file, "notes": "; ".join(p.notes)})
        log.info("  %s cells=%3d total=%16s TOTAL GERAL=%16s checks %d/%d exact  %s%s", p.period,
                 len(p.cells), fmt(s.get("amount_cents")), fmt(s.get("total_geral_cents")),
                 s.get("n_checks_exact", 0), s.get("n_checks", 0), s["validation_status"],
                 f" | {s['validation_reason']}" if s["validation_reason"] else "")

    a.out_dir.mkdir(parents=True, exist_ok=True)
    frames = {"caixa_scan_advertising_lines": pd.DataFrame(cells),
              "caixa_scan_advertising_checks": pd.DataFrame(checks),
              "caixa_scan_advertising_monthly": pd.DataFrame(months)}
    # A run restricted with --periods replaces only the months it re-read. Writing the frames as
    # they stand would overwrite every other month with nothing: a four-month diagnostic run once
    # left the outputs holding four months and none of the validated ones. A requested month that
    # was not re-read (its PDF missing) keeps its earlier rows rather than vanishing.
    if a.periods:
        done = {m["period"] for m in months}
        for p in sorted({str(p) for p in a.periods} - done):
            log.warning("  %s was not re-read; its earlier rows are kept", p)
        for name in frames:
            old = a.out_dir / f"{name}.parquet"
            if old.exists():
                prev = pd.read_parquet(old)
                prev = prev[~prev["period"].astype(str).isin(done)] if "period" in prev else prev
                frames[name] = pd.concat([prev, frames[name]], ignore_index=True)
    bad = unlabelled_changes(frames)
    if bad:
        raise SystemExit("values changed without a repair label, nothing written:\n  "
                         + "\n  ".join(bad))
    for name, df in frames.items():
        if "period" in df:
            df = df.sort_values("period", kind="stable").reset_index(drop=True)
        df.to_parquet(a.out_dir / f"{name}.parquet", index=False)
        df.to_csv(a.out_dir / f"{name}.csv", index=False)
        log.info("%s: %d rows", name, len(df))
    m = frames["caixa_scan_advertising_monthly"]
    if len(m):
        counts = m.validation_status.value_counts().to_dict() if "validation_status" in m else {}
        log.info("validated %d of %d months %s; %d flagged within 5 centavos",
                 int(m.validated.sum()), len(m), counts,
                 int(m["tolerance_flagged"].eq(True).sum()) if "tolerance_flagged" in m else 0)


if __name__ == "__main__":
    main()
