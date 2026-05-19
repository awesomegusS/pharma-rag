"""
chunking.py — Text chunking with rich metadata preservation.

Converts LogicalDocument objects into overlapping word-window chunks,
attaching full provenance metadata (doc_type, page range, source file)
to every chunk so MetadataFilter-based retrieval works correctly.
"""

import logging
from typing import List, Tuple

from llama_index.core import Document

from app.config import CHUNK_SIZE, CHUNK_OVERLAP
from app.models import LogicalDocument, ChunkMetadata

logger = logging.getLogger(__name__)


def chunk_logical_document(
    logical_doc: LogicalDocument,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[ChunkMetadata]:
    """
    Sliding-window word chunker with overlap.

    Strategy:
    - If the document fits in one chunk, store it as-is
    - Otherwise use a stride = chunk_size - chunk_overlap
    - Page position is approximated proportionally across the document's page range
    """
    words  = logical_doc.text.split()
    chunks: List[ChunkMetadata] = []

    if len(words) <= chunk_size:
        chunks.append(
            ChunkMetadata(
                chunk_id=f"{logical_doc.doc_id}_chunk_0",
                doc_id=logical_doc.doc_id,
                doc_type=logical_doc.doc_type,
                chunk_index=0,
                page_start=logical_doc.page_start,
                page_end=logical_doc.page_end,
                text=logical_doc.text,
                source_file=logical_doc.source_file,
            )
        )
        return chunks

    stride      = chunk_size - chunk_overlap
    total_words = len(words)
    page_span   = logical_doc.page_end - logical_doc.page_start

    for i, start in enumerate(range(0, total_words, stride)):
        end        = min(start + chunk_size, total_words)
        chunk_text = " ".join(words[start:end])

        frac             = start / total_words
        chunk_page_start = logical_doc.page_start + int(frac * page_span)
        chunk_page_end   = min(chunk_page_start + 1, logical_doc.page_end)

        chunks.append(
            ChunkMetadata(
                chunk_id=f"{logical_doc.doc_id}_chunk_{i}",
                doc_id=logical_doc.doc_id,
                doc_type=logical_doc.doc_type,
                chunk_index=i,
                page_start=chunk_page_start,
                page_end=chunk_page_end,
                text=chunk_text,
                source_file=logical_doc.source_file,
            )
        )

        if end >= total_words:
            break

    return chunks


def chunks_to_llama_documents(chunks: List[ChunkMetadata]) -> List[Document]:
    """
    Convert ChunkMetadata objects into LlamaIndex Document objects.
    The metadata dict is what MetadataFilters queries at retrieval time.
    """
    return [
        Document(
            text=c.text,
            metadata={
                "chunk_id":    c.chunk_id,
                "doc_id":      c.doc_id,
                "doc_type":    c.doc_type,
                "chunk_index": c.chunk_index,
                "page_start":  c.page_start,
                "page_end":    c.page_end,
                "source_file": c.source_file,
            },
            id_=c.chunk_id,
        )
        for c in chunks
    ]


def process_all_documents(
    logical_docs: List[LogicalDocument],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> Tuple[List[ChunkMetadata], List[Document]]:
    """
    Chunk every LogicalDocument and return both the metadata list
    and the LlamaIndex Document list ready for indexing.
    """
    all_chunk_meta: List[ChunkMetadata] = []

    for ld in logical_docs:
        chunks   = chunk_logical_document(ld, chunk_size, chunk_overlap)
        ld.chunks = chunks
        all_chunk_meta.extend(chunks)
        logger.debug("%s: %d chunk(s)", ld.doc_type, len(chunks))

    llama_docs = chunks_to_llama_documents(all_chunk_meta)
    logger.info("Total chunks created: %d", len(all_chunk_meta))
    return all_chunk_meta, llama_docs
