"""
diag_ocr_benchmark.py
=====================
Is OCR accurate enough to read the advertising tables we cannot read as text?

Twenty of Caixa's months, Banrisul's November 2024 and thirty-nine BRB gazettes are image scans
with no text layer, and they are the largest remaining hole in the advertising data. Before any of
them is trusted, the OCR chain is SCORED on pages whose answer is already known.

Two modes, because they measure different things:

  --mode truth   Pages that DO have a text layer, whose amounts are already parsed and whose
                 published TOTAL GERAL reconciles (67 Caixa months qualify). Those pages are
                 rendered to images, OCR'd, and the amounts recovered are compared against the
                 amounts the text layer gave. This is the accuracy number: it isolates the OCR and
                 amount-parsing chain, because the truth is not in dispute. It flatters OCR
                 slightly - a rendered digital page is cleaner than a scan - so read it as an upper
                 bound, and read the per-amount misses as the realistic failure mode.

  --mode scans   The actual scans, which have NO external truth: `stated_total_brl` is missing for
                 all twenty months and no line was ever parsed. What can still be checked is the
                 document's own arithmetic, since the summary table prints a TOTAL GERAL: if OCR
                 misreads one digit of one component, the components stop summing to that total.
                 That self-check is the production gate, the same device the BRB parser uses.

Amounts are parsed with the PRODUCTION reader (`brl` from scrape_statebank_advertising), so the
benchmark scores the chain that would actually be used, not a proxy for it.

Usage
-----
  python diag_ocr_benchmark.py                          # truth mode, 6 months, 3 resolutions
  python diag_ocr_benchmark.py --months 12 --dpi 200
  python diag_ocr_benchmark.py --mode scans
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

from utils.venv_guard import ensure_project_venv
ensure_project_venv(__file__)

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

from utils import paths
from scrape_statebank_advertising import brl

PROC = paths.AWARENESS_PROC
CAIXA_RAW = paths.AWARENESS_RAW / "caixa"
AMOUNT_TOKEN = re.compile(r"-?\d[\d.]*,\d{2}")


def ocr_engine():
    from rapidocr_onnxruntime import RapidOCR
    return RapidOCR()


def source_bytes(source_file: str) -> bytes | None:
    """The PDF bytes for a `source_file`, which may name a member inside a zip.

    The 2019 months are published as zips, and the parser records them as
    "archive.zip::member.pdf"; resolving only plain names silently skipped four months of the
    benchmark's sample.
    """
    import zipfile
    name, _, member = str(source_file).partition("::")
    hit = next(CAIXA_RAW.rglob(name), None)
    if hit is None:
        return None
    if not member:
        return hit.read_bytes()
    with zipfile.ZipFile(hit) as zf:
        cand = [n for n in zf.namelist() if n == member or n.endswith("/" + member)]
        if not cand:
            cand = [n for n in zf.namelist() if n.lower().endswith(".pdf")]
        return zf.read(cand[0]) if cand else None


def page_images(pdf: bytes, pages: list[int] | None, dpi: int) -> list[np.ndarray]:
    """Render the wanted pages of a PDF, given as bytes, to BGR arrays at `dpi`."""
    import cv2
    import fitz
    doc = fitz.open(stream=pdf, filetype="pdf")
    want = pages if pages else range(doc.page_count)
    out = []
    for i in want:
        if i >= doc.page_count:
            continue
        pix = doc[i].get_pixmap(dpi=dpi)
        arr = np.frombuffer(pix.tobytes("png"), dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is not None:
            out.append(img)
    doc.close()
    return out


def ocr_amounts(engine, images: list[np.ndarray]) -> tuple[list[float], list[str]]:
    """Every Brazilian-format amount OCR finds, plus the raw text lines it read."""
    amounts, texts = [], []
    for img in images:
        result, _ = engine(img)
        for row in result or []:
            text = str(row[1])
            texts.append(text)
            for tok in AMOUNT_TOKEN.findall(text.replace(" ", "")):
                v = brl(tok)
                if v is not None:
                    amounts.append(round(v, 2))
    return amounts, texts


def image_only_pages(pdf: bytes) -> list[int]:
    """Zero-based indices of the pages with no text layer, i.e. the scanned ones."""
    import fitz
    doc = fitz.open(stream=pdf, filetype="pdf")
    try:
        return [i for i in range(doc.page_count)
                if not re.sub(r"\s", "", doc[i].get_text())]
    finally:
        doc.close()


def ocr_page(engine, images: list[np.ndarray]) -> tuple[list[float], list[str], list[float]]:
    """Amounts, text lines and per-box confidences."""
    amounts, texts, confs = [], [], []
    for img in images:
        result, _ = engine(img)
        for row in result or []:
            texts.append(str(row[1]))
            confs.append(float(row[2]))
            for tok in AMOUNT_TOKEN.findall(str(row[1]).replace(" ", "")):
                v = brl(tok)
                if v is not None:
                    amounts.append(round(v, 2))
    return amounts, texts, confs


def total_is_printed(pdf: bytes, pages: list[int], value: float) -> bool:
    """Whether `value` is printed on those pages, judged on digits alone.

    The text layer stores a long amount as several fragments while OCR returns one token, so a
    string comparison would report a printed number as absent. Comparing digit runs avoids that.
    """
    import fitz
    want = re.sub(r"[^0-9]", "", f"{value:.2f}")
    doc = fitz.open(stream=pdf, filetype="pdf")
    try:
        for i in pages or range(doc.page_count):
            if i < doc.page_count and want in re.sub(r"[^0-9]", "", doc[i].get_text()):
                return True
    finally:
        doc.close()
    return False


def multiset_recall(truth: list[float], got: list[float]) -> tuple[int, int, int]:
    """(matched, missed, spurious) comparing two multisets of amounts."""
    from collections import Counter
    t, g = Counter(truth), Counter(got)
    matched = sum((t & g).values())
    return matched, sum(t.values()) - matched, sum(g.values()) - matched


def truth_mode(a: argparse.Namespace) -> None:
    lines = pd.read_parquet(PROC / "caixa_advertising_lines.parquet")
    month = pd.read_parquet(PROC / "caixa_advertising_monthly.parquet")
    # `stated_total_source` must be "text_pdf", i.e. a total actually PRINTED in the document. Five
    # months carry "text_pdf:agency_total_row" instead, where the figure is the sum of the
    # per-agency rows and appears nowhere on the page: asking OCR to find it scores the benchmark's
    # own assumption, not the OCR.
    ok = month[(month.status == "parsed") & (month.parse_method == "text_pdf")
               & (month.stated_total_source == "text_pdf")
               & month.stated_total_brl.notna() & (month.total_matches == True)]  # noqa: E712
    if ok.empty:
        raise SystemExit("no text-layer month with a printed, reconciling published total")
    # Spread the sample across years rather than taking the first N, so one layout does not
    # stand in for all of them: the unruled 2014-2018 layout and the ruled one differ.
    ok = ok.sort_values("period")
    pick = ok.iloc[:: max(1, len(ok) // a.months)].head(a.months)
    log.info("scoring %d months at %s dpi, ground truth = the text layer's own amounts",
             len(pick), a.dpi)
    engine = ocr_engine()
    rows = []
    for r in pick.itertuples():
        pdf = source_bytes(r.source_file)
        if pdf is None:
            log.warning("  %s: file not found on disk (%s)", r.period, r.source_file)
            continue
        sub = lines[lines.period == r.period]
        pages = sorted({int(p[1:]) - 1 for p in sub.page_or_sheet.dropna().unique()
                        if str(p).startswith("p")})
        # Include the stated total in the truth set ONLY if that value is actually PRINTED on the
        # page. Several months print a total per agency and the stated figure is their sum (one
        # month's note says so outright), so the number exists nowhere in the document; demanding
        # OCR find it scored the benchmark's own assumption and produced every apparent miss in the
        # first run. Tested on the digits alone, because the text layer stores a long number as
        # fragments while OCR reads it as one token.
        truth = [round(v, 2) for v in sub.amount_brl.tolist()]
        printed = total_is_printed(pdf, pages, r.stated_total_brl)
        if printed:
            truth.append(round(r.stated_total_brl, 2))
        for dpi in a.dpi:
            got, texts = ocr_amounts(engine, page_images(pdf, pages, dpi))
            # The unruled layout prints 0,00 in every empty cell - 122 of one page's 172 amounts -
            # and the production parser does not keep them, so they are dropped on both sides
            # rather than counted as OCR inventing numbers.
            got = [v for v in got if v != 0]
            matched, missed, spurious = multiset_recall(truth, got)
            rows.append({"period": r.period, "dpi": dpi, "n_truth": len(truth),
                         "n_ocr": len(got), "matched": matched, "missed": missed,
                         # not a defect count: the pages also print per-agency subtotal rows,
                         # which the production parser records as a check rather than as lines
                         "extra_nonzero": spurious,
                         "recall_pct": 100 * matched / len(truth) if truth else np.nan,
                         "total_printed": printed,
                         "total_found": (not printed) or round(r.stated_total_brl, 2) in got,
                         "lines_read": len(texts)})
            log.info("  %s dpi=%-4d truth=%3d ocr=%3d matched=%3d MISSED=%3d extra=%3d "
                     "recall=%5.1f%% total=%s", r.period, dpi, len(truth), len(got),
                     matched, missed, spurious, rows[-1]["recall_pct"],
                     ("read" if rows[-1]["total_found"] else "NOT READ")
                     if printed else "not printed")
            if missed and dpi == a.dpi[-1]:
                from collections import Counter
                miss = list((Counter(truth) - Counter(got)).elements())[:4]
                log.info("       missed examples: %s", [f"{v:,.2f}" for v in miss])
    res = pd.DataFrame(rows)
    if res.empty:
        raise SystemExit("nothing scored")
    log.info("\nby resolution:\n%s", res.groupby("dpi").agg(
        months=("period", "nunique"), recall_pct=("recall_pct", "mean"),
        missed=("missed", "sum"), extra_nonzero=("extra_nonzero", "sum"),
        printed_total_read=("total_found", "sum")).round(1).to_string())
    out = a.out_dir / "ocr_benchmark_truth.csv"
    res.to_csv(out, index=False)
    log.info("\nwrote %s", out)


def scans_mode(a: argparse.Namespace) -> None:
    """What OCR recovers from the real scans, where there is no external truth.

    This reports readability, not reconciliation. Reconciling needs the table rebuilt first: the
    summary is a matrix of categories by agency with per-agency totals along the top and TOTAL rows
    at the foot, so adding up every amount on the page counts the totals twice and lands at two to
    three times the month's true spend. Rebuilding it means feeding these OCR boxes into the
    column-by-x logic the Caixa parser already uses for the unruled layouts, which is the next step
    and not a benchmark.

    The label match is deliberately tolerant. OCR returns "TOTAL GERA" for "TOTAL GERAL" - one
    character short, at 0.97 confidence - and an exact test rejected all twenty months. Amounts came
    through exactly; it is the words that lose a character, which is the opposite of the usual worry
    about OCR and money.
    """
    month = pd.read_parquet(PROC / "caixa_advertising_monthly.parquet")
    scan = month[month.status == "unreadable_scan"].sort_values("period")
    log.info("%d scanned months; reporting what OCR recovers from each", len(scan))
    engine = ocr_engine()
    rows = []
    for r in scan.itertuples():
        pdf = source_bytes(r.source_file)
        if pdf is None:
            continue
        # The scanned pages are exactly those with no text layer; the rest are supplier lists,
        # which carry names and tax numbers but no amounts. Sixteen of the twenty months have TWO
        # image pages, so the summary spans both: reading only the first lost the TOTAL row on
        # nineteen months and left February's largest amount at R$4.3m, too small to be a month.
        got, texts, confs = ocr_page(engine, page_images(pdf, image_only_pages(pdf), a.dpi[-1]))
        nonzero = [v for v in got if v != 0]
        # Match on whitespace-stripped text: OCR returns "TOTALGERAL" with no space (0.99
        # confidence) as often as "TOTAL GERAL", and requiring the space reported the label as
        # missing on sixteen of twenty months when it had in fact been read correctly.
        flat = [re.sub(r"\s", "", t.upper()) for t in texts]
        total_row = next((t for t, f in zip(texts, flat) if "TOTALGER" in f), None)
        heading = next((t for t, f in zip(texts, flat)
                        if "CUSTOS" in f or "PUBLICIDADE" in f), None)
        rows.append({"period": r.period, "boxes": len(texts), "n_amounts": len(nonzero),
                     "largest_amount": max(nonzero) if nonzero else None,
                     "total_row_read": total_row, "heading_read": heading,
                     "mean_conf": round(float(np.mean(confs)), 3) if confs else None,
                     "min_conf": round(float(np.min(confs)), 3) if confs else None})
        log.info("  %s boxes=%3d amounts=%3d largest=%16s conf mean=%.3f min=%.2f total row=%s",
                 r.period, len(texts), len(nonzero),
                 f"{max(nonzero):,.2f}" if nonzero else "-",
                 rows[-1]["mean_conf"] or 0.0, rows[-1]["min_conf"] or 0.0,
                 repr(total_row) if total_row else "NOT FOUND")
    res = pd.DataFrame(rows)
    log.info("\nread a TOTAL row on %d of %d months; heading on %d; mean confidence %.3f, "
             "worst box %.2f", int(res.total_row_read.notna().sum()), len(res),
             int(res.heading_read.notna().sum()), float(res.mean_conf.mean()),
             float(res.min_conf.min()))
    out = a.out_dir / "ocr_benchmark_scans.csv"
    res.to_csv(out, index=False)
    log.info("wrote %s", out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("truth", "scans"), default="truth")
    ap.add_argument("--months", type=int, default=6)
    ap.add_argument("--dpi", type=int, nargs="+", default=[150, 200, 300])
    ap.add_argument("--out-dir", type=Path, default=PROC)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    (truth_mode if a.mode == "truth" else scans_mode)(a)


if __name__ == "__main__":
    main()
