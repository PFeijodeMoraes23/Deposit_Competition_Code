"""vision_extractor.py
# Author: Pedro Feijó de Moraes
#
# Tier-4 fallback in the firm-disclosure extraction cascade.
# Takes a PDF (bytes) or image file, renders pages to images with PyMuPDF,
# and asks Claude claude-haiku-4-5 to extract a specific numeric metric like a
# client/customer count. Returns a structured dict (or None on failure).
#
# GATED: only fires when the cheaper tiers (pdfplumber regex, OCR) return
# nothing. Charges the Anthropic API; keep VISION_MAX_PAGES small.
#
# Usage:
#   from utils.vision_extractor import extract_metric_from_pdf
#   result = extract_metric_from_pdf(pdf_bytes, firm="Banco Pan",
#               metric="total active clients", period="3Q2024")
#   # result -> {"value": 30000000, "unit": "count", "period": "3Q2024",
#   #            "source_quote": "30 milhões de clientes ativos", "confidence": "high"}
#
# Requires: ANTHROPIC_API_KEY environment variable, or pass api_key kwarg.
"""

from __future__ import annotations
import base64
import json
import logging
import os
import io
import re

log = logging.getLogger(__name__)

# Load .env file from the project root if it exists, so the key is available
# to Claude Code's bash tool without needing to restart the session.
def _load_dotenv() -> None:
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env_path = os.path.join(here, ".env")
    if not os.path.exists(env_path):
        return
    with open(env_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            k = k.strip(); v = v.strip().strip('"').strip("'")
            if k and k not in os.environ:
                os.environ[k] = v

_load_dotenv()

VISION_MAX_PAGES = 6      # max pages to send per PDF (cost control)
VISION_MODEL = "claude-haiku-4-5-20251001"
VISION_MAX_TOKENS = 256

_SYSTEM_PROMPT = (
    "You are a financial data extraction assistant. You will be given one or "
    "more pages from a Brazilian bank earnings release, results presentation, "
    "or investor deck. Extract ONLY the specific numeric metric requested. "
    "Respond with a single valid JSON object and nothing else."
)

_USER_TEMPLATE = (
    "From the pages shown, extract: {metric} for {firm}, period {period}.\n"
    "Rules:\n"
    "- Look for the specific KPI (e.g. total clients, active clients, correntistas, base de clientes).\n"
    "- Report the number as an absolute integer (e.g. 30000000 for 30 million).\n"
    "- If the value appears as 'X mi' or 'X milhoes', multiply by 1,000,000.\n"
    "- If not found, return null for value.\n"
    "Respond ONLY with JSON in this exact format (no markdown fences):\n"
    '{{"value": <int or null>, "unit": "count", "period": "<quarter or year>", '
    '"source_quote": "<exact text from the slide you used>", "confidence": "high|medium|low"}}'
)


def _pdf_to_page_images(pdf_bytes: bytes, max_pages: int = VISION_MAX_PAGES) -> list[bytes]:
    """Render the first `max_pages` pages of a PDF to JPEG bytes using PyMuPDF."""
    try:
        import fitz
    except ImportError:
        log.warning("PyMuPDF (fitz) not installed — cannot render PDF pages for vision")
        return []
    images = []
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for i in range(min(max_pages, len(doc))):
            page = doc[i]
            mat = fitz.Matrix(1.5, 1.5)  # 1.5x zoom -> ~108 DPI (good for slides)
            pix = page.get_pixmap(matrix=mat)
            images.append(pix.tobytes("jpeg"))
        doc.close()
    except Exception as e:
        log.debug(f"PDF render error: {e}")
    return images


def extract_metric_from_pdf(
    pdf_bytes: bytes,
    firm: str,
    metric: str = "total active clients",
    period: str = "latest",
    api_key: str | None = None,
    max_pages: int = VISION_MAX_PAGES,
) -> dict | None:
    """
    Send rendered PDF pages to Claude claude-haiku-4-5 vision and extract a numeric metric.
    Returns a dict with keys: value, unit, period, source_quote, confidence.
    Returns None if the API is unavailable, the metric is not found, or anthropic is not installed.
    """
    try:
        import anthropic as _anthropic
    except ImportError:
        log.warning("anthropic SDK not installed — Tier-4 vision extraction unavailable. "
                    "Install with: pip install anthropic")
        return None

    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        log.warning("ANTHROPIC_API_KEY not set — Tier-4 vision extraction skipped. "
                    "Set the environment variable to enable.")
        return None

    images = _pdf_to_page_images(pdf_bytes, max_pages)
    if not images:
        log.debug("No renderable pages — Tier-4 skipped")
        return None

    # Build the message: system + user text + image blocks
    content: list = [
        {"type": "text", "text": _USER_TEMPLATE.format(
            metric=metric, firm=firm, period=period)}
    ]
    for img_bytes in images:
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": base64.standard_b64encode(img_bytes).decode("utf-8"),
            }
        })

    try:
        client = _anthropic.Anthropic(api_key=key)
        msg = client.messages.create(
            model=VISION_MODEL,
            max_tokens=VISION_MAX_TOKENS,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": content}],
        )
        raw = msg.content[0].text.strip()
        # strip any markdown fences
        raw = re.sub(r"^```[a-z]*\n?", "", raw); raw = re.sub(r"\n?```$", "", raw)
        result = json.loads(raw)
        if not isinstance(result, dict) or result.get("value") is None:
            return None
        log.info(f"[vision] {firm} {period}: value={result['value']} "
                 f"({result.get('confidence','?')} confidence) — {result.get('source_quote','')[:60]}")
        return result
    except json.JSONDecodeError as e:
        log.debug(f"[vision] JSON parse error: {e} | raw={raw[:80]}")
    except Exception as e:
        log.warning(f"[vision] API error for {firm}: {type(e).__name__}: {e}")
    return None


def extract_metric_from_image(
    image_bytes: bytes,
    firm: str,
    metric: str = "total active clients",
    period: str = "latest",
    api_key: str | None = None,
) -> dict | None:
    """Same as extract_metric_from_pdf but takes raw image bytes (JPEG/PNG).
    Used for screenshots of JS dashboards (FGC, BCB RCF/REB)."""
    try:
        import anthropic as _anthropic
    except ImportError:
        log.warning("anthropic SDK not installed — Tier-4 unavailable")
        return None

    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        log.warning("ANTHROPIC_API_KEY not set — Tier-4 vision skipped")
        return None

    # detect media type
    media_type = "image/jpeg"
    if image_bytes[:8] == b"\x89PNG\r\n\x1a\n":
        media_type = "image/png"

    content = [
        {"type": "text", "text": _USER_TEMPLATE.format(
            metric=metric, firm=firm, period=period)},
        {"type": "image", "source": {
            "type": "base64",
            "media_type": media_type,
            "data": base64.standard_b64encode(image_bytes).decode("utf-8"),
        }}
    ]
    try:
        client = _anthropic.Anthropic(api_key=key)
        msg = client.messages.create(
            model=VISION_MODEL, max_tokens=VISION_MAX_TOKENS,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": content}],
        )
        raw = msg.content[0].text.strip()
        raw = re.sub(r"^```[a-z]*\n?", "", raw); raw = re.sub(r"\n?```$", "", raw)
        result = json.loads(raw)
        if not isinstance(result, dict) or result.get("value") is None:
            return None
        log.info(f"[vision] {firm} {period} (image): value={result['value']}")
        return result
    except Exception as e:
        log.warning(f"[vision] image API error for {firm}: {e}")
    return None
