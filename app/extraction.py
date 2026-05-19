"""
extraction.py — PDF text extraction and logical document grouping.

Pipeline:
  1. extract_pages()               — PyMuPDF + Tesseract OCR fallback
  2. classify_all_pages_parallel() — rule-based (instant) + Gemini (parallel)
  3. detect_boundaries_parallel()  — type-change fast path + parallel same-doc check
  4. group_pages_into_logical_docs() — merge consecutive pages into LogicalDocuments
"""

import io
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple

import fitz  # PyMuPDF
import pytesseract
from PIL import Image

from app.config import OCR_DPI, OCR_TEXT_THRESHOLD, PARALLEL_WORKERS
from app.models import PageInfo, LogicalDocument
from app.intelligence import (
    rule_based_classify,
    classify_doc_type,
    is_same_document,
)

logger = logging.getLogger(__name__)


# ── Step 1: Page extraction ───────────────────────────────────────────────────

def extract_pages(pdf_file) -> List[PageInfo]:
    """
    Extract raw text from every page using PyMuPDF.
    Falls back to Tesseract OCR for pages with < OCR_TEXT_THRESHOLD characters
    (i.e. scanned pages with no embedded text layer).
    """
    if isinstance(pdf_file, dict) and "content" in pdf_file:
        doc = fitz.open(stream=pdf_file["content"], filetype="pdf")
    elif hasattr(pdf_file, "read"):
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    else:
        doc = fitz.open(pdf_file)

    pages: List[PageInfo] = []
    scanned_count = 0

    for i, page in enumerate(doc):
        text = page.get_text().strip()

        if len(text) < OCR_TEXT_THRESHOLD:
            try:
                pix = page.get_pixmap(dpi=OCR_DPI)
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                text = pytesseract.image_to_string(img)
                scanned_count += 1
                logger.debug("Page %d: OCR extracted %d chars", i + 1, len(text))
            except Exception as exc:
                logger.warning("Page %d: OCR failed — %s", i + 1, exc)
                text = ""

        pages.append(PageInfo(page_num=i + 1, text=text))

    doc.close()

    if scanned_count:
        logger.info("%d scanned page(s) processed via OCR", scanned_count)
    logger.info("Extracted %d pages from PDF", len(pages))
    return pages


# ── Step 2: Parallel classification ──────────────────────────────────────────

def classify_all_pages_parallel(
    pages: List[PageInfo], max_workers: int = PARALLEL_WORKERS
) -> List[str]:
    """
    Two-pass classifier:
      Pass 1 — instant rule-based classification (zero API calls)
      Pass 2 — parallel Gemini calls for pages that rules couldn't resolve

    Returns a list of doc-type strings aligned with pages[].
    """
    results: List[str | None] = [None] * len(pages)
    needs_gemini: List[int] = []

    for i, page in enumerate(pages):
        result = rule_based_classify(page.text)
        if result:
            results[i] = result
        else:
            needs_gemini.append(i)

    rule_hits = len(pages) - len(needs_gemini)
    logger.info("Rule-based: %d/%d pages classified instantly", rule_hits, len(pages))

    if needs_gemini:
        logger.info("Gemini classifying %d uncertain page(s) in parallel...", len(needs_gemini))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(classify_doc_type, pages[i].text): i
                for i in needs_gemini
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception:
                    results[idx] = "Other"

    return results  # type: ignore[return-value]


# ── Step 3: Parallel boundary detection ──────────────────────────────────────

def detect_boundaries_parallel(
    pages: List[PageInfo],
    doc_types: List[str],
    max_workers: int = PARALLEL_WORKERS,
) -> List[bool]:
    """
    Determine document boundaries for all page transitions.

    Fast-path optimization:
    - Different classified types → boundary confirmed with zero API calls
    - Same classified types     → call is_same_document in parallel

    Returns is_new_doc[i] for each page index.
    """
    n = len(pages)
    is_new_doc = [False] * n
    is_new_doc[0] = True  # first page always starts a document

    type_changed: List[int] = []
    needs_check: List[int] = []

    for i in range(1, n):
        if doc_types[i] != doc_types[i - 1]:
            type_changed.append(i)
        else:
            needs_check.append(i)

    logger.info("Type-change boundaries (no API): %d", len(type_changed))
    logger.info("Same-type transitions to check (parallel): %d", len(needs_check))

    for i in type_changed:
        is_new_doc[i] = True

    if needs_check:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(
                    is_same_document,
                    pages[i - 1].text,
                    pages[i].text,
                    doc_types[i],
                ): i
                for i in needs_check
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    is_new_doc[idx] = not future.result()
                except Exception:
                    is_new_doc[idx] = False  # default: continuation

    return is_new_doc


# ── Step 3b: Apply results to PageInfo objects ────────────────────────────────

def detect_boundaries_and_classify(pages: List[PageInfo]) -> List[PageInfo]:
    """
    Orchestrates parallel classification + boundary detection,
    then writes results back onto each PageInfo object.
    """
    doc_types = classify_all_pages_parallel(pages)
    is_new    = detect_boundaries_parallel(pages, doc_types)

    page_in_doc = 1
    for i, page in enumerate(pages):
        page.doc_type   = doc_types[i]
        page.is_new_doc = is_new[i]
        if is_new[i]:
            page_in_doc = 1
            logger.info("Page %d: [NEW] %s", page.page_num, page.doc_type)
        else:
            page_in_doc += 1
        page.page_in_doc = page_in_doc

    return pages


# ── Step 4: Logical document grouping ────────────────────────────────────────

def group_pages_into_logical_docs(
    pages: List[PageInfo], filename: str = "Unknown"
) -> List[LogicalDocument]:
    """
    Merge consecutive pages that share a document into one LogicalDocument.
    Attaches the source filename so every downstream chunk carries provenance.
    """
    logical_docs: List[LogicalDocument] = []
    current: dict = {"text": "", "doc_type": None, "page_start": None, "doc_id": None}

    def _flush(page_end: int) -> None:
        if current["text"]:
            logical_docs.append(
                LogicalDocument(
                    doc_id=current["doc_id"],
                    doc_type=current["doc_type"],
                    page_start=current["page_start"],
                    page_end=page_end,
                    text=current["text"].strip(),
                    source_file=filename,
                )
            )

    for page in pages:
        if page.is_new_doc and current["text"]:
            _flush(page.page_num - 1)
            current = {"text": "", "doc_type": None, "page_start": None, "doc_id": None}

        if current["doc_id"] is None:
            safe_fname = "".join(c if c.isalnum() else "_" for c in filename)
            current["doc_id"]    = f"{safe_fname}_doc_{len(logical_docs)}"
            current["page_start"] = page.page_num

        current["text"]     += "\n\n" + page.text
        current["doc_type"]  = page.doc_type

    if current["text"]:
        _flush(pages[-1].page_num)

    logger.info("Identified %d logical document(s):", len(logical_docs))
    for ld in logical_docs:
        logger.info("  • %s (pages %d–%d)", ld.doc_type, ld.page_start, ld.page_end)

    return logical_docs


# ── Top-level entry point ─────────────────────────────────────────────────────

def extract_and_analyze_pdf(
    pdf_file, filename: str = "Unknown"
) -> Tuple[List[PageInfo], List[LogicalDocument]]:
    """
    Full pipeline: extract → classify (parallel) → detect boundaries (parallel) → group.
    Returns (pages, logical_docs).
    """
    logger.info("Starting PDF extraction and analysis for: %s", filename)
    pages = extract_pages(pdf_file)

    logger.info("Classifying pages and detecting document boundaries...")
    pages = detect_boundaries_and_classify(pages)

    logical_docs = group_pages_into_logical_docs(pages, filename)
    return pages, logical_docs
