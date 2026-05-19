"""
store.py — EnhancedDocumentStore: the central orchestrator for one session.

Manages the full lifecycle:
  upload → extract → classify → group → chunk → index → retrieve → answer

Key design choices:
  - Incremental ingestion: new PDFs insert into the existing index (no full rebuild)
  - Deduplication by filename: the same file cannot be processed twice
  - query_stream() yields partial results for Gradio streaming
  - query() is the non-streaming fallback used by the evaluation notebook
"""

import logging
from datetime import datetime
from typing import Dict, Generator, List, Optional, Tuple

from llama_index.core import Document, VectorStoreIndex

from app.chunking import process_all_documents
from app.extraction import extract_and_analyze_pdf
from app.llm import stream_mistral
from app.models import ChunkMetadata, LogicalDocument, PageInfo
from app.retrieval import build_rag_prompt_with_budget, build_vector_index, retrieve_chunks
from app.config import MISTRAL_MAX_REPLY, MISTRAL_N_CTX, MISTRAL_TEMPERATURE

logger = logging.getLogger(__name__)


class EnhancedDocumentStore:
    """
    Stateful document store for one Gradio session.

    Public API:
      process_pdf(pdf_files)  → (success, stats_dict)
      query_stream(question)  → Generator[dict, None, None]
      query(question)         → dict
      get_document_structure() → List[dict]
    """

    def __init__(self) -> None:
        self.pages:           List[PageInfo]         = []
        self.logical_docs:    List[LogicalDocument]  = []
        self.chunk_meta:      List[ChunkMetadata]    = []
        self.llama_docs:      List[Document]         = []
        self.index:           Optional[VectorStoreIndex] = None
        self.is_ready:        bool = False
        self.stats:           Dict = {}
        self.processed_files: set  = set()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _normalize_filepaths(self, pdf_files) -> List[str]:
        if not pdf_files:
            return []
        if isinstance(pdf_files, str):
            return [pdf_files]
        return list(pdf_files)

    def _get_new_files(self, filepaths: List[str]) -> List[str]:
        import os
        return [p for p in filepaths if os.path.basename(p) not in self.processed_files]

    # ── Public API ────────────────────────────────────────────────────────────

    def process_pdf(self, pdf_files) -> Tuple[bool, Dict]:
        """
        Full ingestion pipeline for one or more PDF files.

        Behavior:
        - Only processes files not seen before (deduplication by basename)
        - First call builds the vector index from scratch
        - Subsequent calls insert new nodes incrementally (no full rebuild)

        Returns (success, stats_dict).
        """
        import os
        self.is_ready = False
        t0 = datetime.now()

        filepaths = self._normalize_filepaths(pdf_files)
        if not filepaths:
            return False, {"error": "No PDF files provided."}

        new_files = self._get_new_files(filepaths)
        if not new_files:
            self.is_ready = self.index is not None
            return True, {**self.stats, "message": "No new files to process."}

        try:
            batch_pages:     List[PageInfo]        = []
            batch_log_docs:  List[LogicalDocument] = []
            batch_chunks:    List[ChunkMetadata]   = []
            batch_llama:     List[Document]        = []

            for pdf_path in new_files:
                fname = os.path.basename(pdf_path)
                logger.info("Processing new file: %s", fname)

                pages, logical_docs       = extract_and_analyze_pdf(pdf_path, filename=fname)
                chunk_meta, llama_docs    = process_all_documents(logical_docs)

                batch_pages.extend(pages)
                batch_log_docs.extend(logical_docs)
                batch_chunks.extend(chunk_meta)
                batch_llama.extend(llama_docs)
                self.processed_files.add(fname)

            # Append to global store
            self.pages.extend(batch_pages)
            self.logical_docs.extend(batch_log_docs)
            self.chunk_meta.extend(batch_chunks)
            self.llama_docs.extend(batch_llama)

            # Build or incrementally update the index
            if self.index is None:
                self.index = build_vector_index(batch_llama)
            else:
                for doc in batch_llama:
                    self.index.insert(doc)
                logger.info("Incrementally inserted %d new nodes.", len(batch_llama))

            elapsed = (datetime.now() - t0).total_seconds()
            self.stats = {
                "filename":        f"{len(self.processed_files)} file(s)",
                "total_pages":     len(self.pages),
                "documents_found": len(self.logical_docs),
                "total_chunks":    len(self.chunk_meta),
                "document_types":  list(dict.fromkeys(ld.doc_type for ld in self.logical_docs)),
                "processing_time": f"{elapsed:.1f}s",
            }
            self.is_ready = True
            return True, self.stats

        except Exception as exc:
            import traceback
            traceback.print_exc()
            logger.error("Processing failed: %s", exc)
            return False, {"error": str(exc)}

    def retrieve_only(
        self,
        question: str,
        filter_doc_type: Optional[str] = None,
        auto_route: bool = True,
        k: int = 4,
    ) -> Tuple[List, Optional[str], float]:
        """Retrieval without answer generation. Useful for evaluation."""
        if not self.is_ready or self.index is None:
            return [], None, 0.0
        return retrieve_chunks(
            self.index, question,
            filter_doc_type=filter_doc_type,
            auto_route=auto_route,
            k=k,
        )

    def query_stream(
        self,
        question: str,
        *,
        filter_doc_type: Optional[str] = None,
        auto_route: bool = True,
        k: int = 4,
        max_new_tokens: int = MISTRAL_MAX_REPLY,
        temperature: float = MISTRAL_TEMPERATURE,
        max_ctx_tokens: int = MISTRAL_N_CTX,
    ) -> Generator[Dict, None, None]:
        """
        Streaming query: yields partial dicts as Mistral generates tokens.
        Each yielded dict: {answer_partial, sources, filter_used, confidence}
        """
        if not self.is_ready:
            yield {
                "answer_partial": "Please upload and process a pharmaceutical PDF first.",
                "sources": [],
                "filter_used": "none",
                "confidence": 0.0,
            }
            return

        nodes, routed_type, confidence = self.retrieve_only(
            question, filter_doc_type=filter_doc_type, auto_route=auto_route, k=k
        )

        if not nodes:
            yield {
                "answer_partial": (
                    "I couldn't find relevant information to answer your question. "
                    "Please ensure the PDF has been processed and try rephrasing your query."
                ),
                "sources": [],
                "filter_used": "none",
                "confidence": 0.0,
            }
            return

        prompt, sources, avg_score = build_rag_prompt_with_budget(
            question, nodes, max_ctx_tokens=max_ctx_tokens, max_new_tokens=max_new_tokens
        )

        acc = ""
        for token in stream_mistral(prompt, max_tokens=max_new_tokens, temperature=temperature):
            acc += token
            yield {
                "answer_partial": acc,
                "sources":        sources,
                "filter_used":    routed_type or "global",
                "confidence":     avg_score,
            }

    def query(
        self,
        question: str,
        filter_doc_type: Optional[str] = None,
        auto_route: bool = True,
        k: int = 4,
    ) -> Dict:
        """
        Non-streaming query. Collects full response before returning.
        Used by the evaluation notebook.
        """
        if not self.is_ready:
            return {
                "answer": "Please upload and process a pharmaceutical PDF first.",
                "sources": [], "confidence": 0.0, "filter_used": "none",
            }

        result = {"answer": "", "sources": [], "filter_used": "none", "confidence": 0.0}
        for partial in self.query_stream(
            question, filter_doc_type=filter_doc_type, auto_route=auto_route, k=k
        ):
            result["answer"]      = partial.get("answer_partial", "")
            result["sources"]     = partial.get("sources", [])
            result["filter_used"] = partial.get("filter_used", "none")
            result["confidence"]  = partial.get("confidence", 0.0)
        return result

    def get_document_structure(self) -> List[Dict]:
        """Summary list for the Document Info panel in the UI."""
        return [
            {
                "source_file": ld.source_file,
                "type":        ld.doc_type,
                "pages":       f"{ld.page_start}–{ld.page_end}",
                "chunks":      len(ld.chunks) if ld.chunks else 0,
            }
            for ld in self.logical_docs
        ]
