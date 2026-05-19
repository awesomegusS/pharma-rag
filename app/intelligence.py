"""
intelligence.py — Document intelligence: classification, boundary detection, query routing.

Three-tier classification strategy:
  1. Rule-based (instant, zero API calls)   — covers ~70-80% of pages
  2. Gemini parallel (fast, structured JSON) — handles ambiguous pages
  3. Boundary detection (fast path + parallel) — avoids redundant API calls
"""

import json
import logging
from typing import Optional, Tuple

from app.config import VALID_DOC_TYPES, RULE_PATTERNS, HIGH_SPECIFICITY_TYPES
from app.llm import call_gemini

logger = logging.getLogger(__name__)


# ── Rule-based classifier (zero API calls) ────────────────────────────────────

def rule_based_classify(text: str) -> Optional[str]:
    """
    Instant keyword-based classification — no API call.
    Returns a doc type string if confident, None if the page is ambiguous.

    Logic:
    - Count matching keywords per doc type
    - High-specificity types (BSE/TSE, Chain of Custody, Packaging Spec) require 1+ match
    - All other types require 2+ matches to avoid false positives
    """
    text_lower = text[:1500].lower()
    scores: dict[str, int] = {}

    for doc_type, patterns in RULE_PATTERNS.items():
        score = sum(1 for p in patterns if p in text_lower)
        if score > 0:
            scores[doc_type] = score

    if not scores:
        return None

    best       = max(scores, key=scores.get)
    best_score = scores[best]
    threshold  = 1 if best in HIGH_SPECIFICITY_TYPES else 2
    return best if best_score >= threshold else None


# ── LLM-based classifier (Gemini, structured JSON prompt) ────────────────────

def clean_doc_type(raw: str) -> str:
    """Normalise a raw LLM response into one of the VALID_DOC_TYPES labels."""
    cleaned = raw.strip().lower().replace('"', "").replace("`", "").replace("*", "").replace(".", "")
    for label in VALID_DOC_TYPES:
        if label.lower() in cleaned:
            return label
    return "Other"


def classify_doc_type(text: str, max_chars: int = 1500) -> str:
    """
    Classify a page's document type using Gemini + structured JSON prompt.
    Falls back to 'Other' on any parse error.
    """
    sample = text[:max_chars]
    prompt = f"""You are an expert pharmaceutical document classifier.
Classify the page below into EXACTLY ONE type. Output ONLY valid JSON.

PERMITTED TYPES:
- "Cover Letter": Formal letter (often "To Whom It May Concern") about product info or storage.
- "Certificate of Quality": Contains lot numbers, manufacture/expiry dates, test results.
- "Packaging Specification": Packaging components, materials, part numbers, change history.
- "BSE/TSE Declaration": Animal-origin material declarations, TSE compliance.
- "Material Description": Materials of construction, sterilization compatibility, physical properties.
- "Supplier Qualification": Supplier audits, ISO 9001/13485 certifications, approved products.
- "Chain of Custody": Manufactured assemblies, traceability, shipment flow.
- "Other": Only if none of the above fit.

Page Content:
{sample}

Output ONLY: {{"reasoning": "<one sentence>", "document_type": "<type>"}}"""

    raw = call_gemini(prompt)
    try:
        start, end = raw.find("{"), raw.rfind("}") + 1
        parsed = json.loads(raw[start:end])
        return clean_doc_type(parsed.get("document_type", "Other"))
    except Exception:
        return clean_doc_type(raw)


# ── Boundary detection ────────────────────────────────────────────────────────

def is_same_document(
    prev_text: str, curr_text: str, current_doc_type: Optional[str] = None
) -> bool:
    """
    Determine whether two consecutive pages belong to the same logical document.

    Returns True (same document) or False (new document starts at curr).
    Defaults to True (continuation) on any API error to avoid false splits.
    """
    prev_sample = prev_text[-600:] if len(prev_text) > 600 else prev_text
    curr_sample = curr_text[:600]  if len(curr_text) > 600 else curr_text

    prompt = f"""Determine if these two consecutive pages are from the SAME pharmaceutical document.

Current document type: {current_doc_type or "Unknown"}

A NEW document starts when the page has:
- A different title/heading (e.g., "Certificate of Quality" vs "Packaging Specification")
- A completely different topic or subject matter
- Its own header with a new document number or reference

Pages belong to the SAME document when:
- The second page says "continued" or "page 2 of 2"
- The content directly continues the previous page's discussion
- They share the same document number or title

End of Previous Page:
...{prev_sample}

Start of Current Page:
{curr_sample}...

Answer ONLY 'Yes' (same document) or 'No' (different document)."""

    try:
        response = call_gemini(prompt)
        return response.strip().lower().startswith("yes")
    except Exception as exc:
        logger.warning("Boundary detection error: %s", exc)
        return True  # safe default: keep together


# ── Query routing ─────────────────────────────────────────────────────────────

def predict_doc_type_for_query(query: str) -> Tuple[str, float]:
    """
    Route a natural language query to the most likely pharmaceutical document type.
    Returns (predicted_type, confidence_score).
    """
    prompt = f"""Analyze this query and predict which pharmaceutical document type
would most likely contain the answer.

Query: "{query}"

Choose the MOST LIKELY type from:
- Cover Letter: Formal letters about product information or storage conditions
- Certificate of Quality: Lot numbers, manufacture/expiration dates, test results
- Packaging Specification: Packaging components, materials, part numbers
- BSE/TSE Declaration: Animal-origin material declarations, TSE compliance
- Material Description: Materials of construction, sterilization compatibility
- Supplier Qualification: Supplier audits, ISO certifications, approved products
- Chain of Custody: Manufactured assemblies, traceability, shipment flow
- Other: General or unclear queries, OR queries asking to compare/search ACROSS documents.

CRITICAL RULE: If the query uses phrases like "across these documents", "compare",
or asks for a general summary, you MUST select "Other".

Respond in JSON: {{"type": "DocumentType", "confidence": 0.85}}"""

    try:
        raw = call_gemini(prompt)
        start, end = raw.find("{"), raw.rfind("}") + 1
        parsed = json.loads(raw[start:end])
        predicted  = clean_doc_type(parsed.get("type", "Other"))
        confidence = float(parsed.get("confidence", 0.5))
        return predicted, confidence
    except Exception as exc:
        logger.warning("Query routing error: %s", exc)
        return "Other", 0.0
