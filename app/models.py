"""
models.py — Core data structures for the Pharma RAG pipeline.

Tracks the full provenance chain:
  PDF page → LogicalDocument → ChunkMetadata → LlamaIndex Document
"""

from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class PageInfo:
    """One physical page extracted from the PDF."""
    page_num: int
    text: str
    doc_type: str = "Other"
    page_in_doc: int = 1      # 1-indexed position within its logical document
    is_new_doc: bool = False  # True when this page starts a new logical document


@dataclass
class LogicalDocument:
    """
    A group of consecutive pages that form one pharmaceutical sub-document.
    E.g. a 2-page Certificate of Quality becomes a single LogicalDocument.
    """
    doc_id: str
    doc_type: str
    page_start: int
    page_end: int
    text: str
    source_file: str = "Unknown"
    chunks: List[Dict] = field(default_factory=list)


@dataclass
class ChunkMetadata:
    """
    Rich metadata attached to every text chunk stored in the vector index.
    Enables MetadataFilter-based retrieval scoped to a specific document type.
    """
    chunk_id: str
    doc_id: str
    doc_type: str
    chunk_index: int
    page_start: int
    page_end: int
    text: str
    source_file: str = "Unknown"
