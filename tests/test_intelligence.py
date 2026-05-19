"""
Tests for the rule-based document classifier and utility functions.
These tests run without any API calls or model weights.
"""

import pytest
from app.intelligence import rule_based_classify, clean_doc_type
from app.config import VALID_DOC_TYPES


class TestRuleBasedClassifier:
    def test_certificate_of_quality(self):
        text = "Certificate of Quality\nLot Number: 123456\nDate of Manufacture: 2024-01-01\nConforms"
        assert rule_based_classify(text) == "Certificate of Quality"

    def test_bse_tse_single_keyword(self):
        # BSE/TSE is a high-specificity type — single match should suffice
        text = "Transmissible spongiform encephalopathies statement for product XYZ"
        assert rule_based_classify(text) == "BSE/TSE Declaration"

    def test_cover_letter(self):
        text = "To Whom It May Concern,\nThis letter is provided for informational purposes.\nSincerely"
        assert rule_based_classify(text) == "Cover Letter"

    def test_chain_of_custody_single_keyword(self):
        text = "Chain of Custody document for assembled components"
        assert rule_based_classify(text) == "Chain of Custody"

    def test_packaging_specification(self):
        text = "Packaging Specification\nBlister tray: PETG\nLid film: Tyvek 1073B\nDrawing reference"
        assert rule_based_classify(text) == "Packaging Specification"

    def test_ambiguous_returns_none(self):
        # Completely generic text should return None (fall back to Gemini)
        text = "This is a general document with no pharmaceutical keywords."
        assert rule_based_classify(text) is None

    def test_supplier_qualification_requires_two_keywords(self):
        # Supplier qualification is NOT high-specificity — needs 2+ matches
        text = "ISO 9001 certified supplier"  # only 1 keyword
        result = rule_based_classify(text)
        assert result != "Supplier Qualification"

        text_two = "ISO 9001 certified supplier qualification record with quality agreement"
        assert rule_based_classify(text_two) == "Supplier Qualification"


class TestCleanDocType:
    def test_exact_match(self):
        assert clean_doc_type("Certificate of Quality") == "Certificate of Quality"

    def test_lowercase_with_noise(self):
        assert clean_doc_type('"bse/tse declaration"') == "BSE/TSE Declaration"

    def test_unknown_falls_back_to_other(self):
        assert clean_doc_type("completely unknown type") == "Other"

    def test_all_valid_types_are_recognised(self):
        for doc_type in VALID_DOC_TYPES:
            assert clean_doc_type(doc_type) == doc_type


class TestChunking:
    def test_short_document_produces_one_chunk(self):
        from app.models import LogicalDocument
        from app.chunking import chunk_logical_document

        ld = LogicalDocument(
            doc_id="test_doc_0",
            doc_type="Certificate of Quality",
            page_start=1, page_end=1,
            text="Short text with only a few words",
            source_file="test.pdf",
        )
        chunks = chunk_logical_document(ld, chunk_size=500, chunk_overlap=100)
        assert len(chunks) == 1
        assert chunks[0].doc_type == "Certificate of Quality"
        assert chunks[0].source_file == "test.pdf"

    def test_long_document_produces_multiple_chunks(self):
        from app.models import LogicalDocument
        from app.chunking import chunk_logical_document

        words = ["word"] * 1200
        ld = LogicalDocument(
            doc_id="test_doc_1",
            doc_type="Supplier Qualification",
            page_start=1, page_end=3,
            text=" ".join(words),
            source_file="test.pdf",
        )
        chunks = chunk_logical_document(ld, chunk_size=500, chunk_overlap=100)
        assert len(chunks) > 1
        # All chunks should carry correct provenance
        for c in chunks:
            assert c.doc_type == "Supplier Qualification"
            assert c.source_file == "test.pdf"
            assert c.doc_id == "test_doc_1"
