"""
retrieval.py — Vector indexing, metadata-filtered retrieval, and prompt assembly.

Key design decisions:
  - MetadataFilters scope retrieval to the predicted document type
  - Query routing cache prevents redundant Gemini calls for repeated queries
  - build_rag_prompt_with_budget() enforces context window limits before inference
"""

import logging
from typing import Dict, List, Optional, Tuple

from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.core.vector_stores import (
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)

from app.config import DEFAULT_TOP_K, MIN_CONFIDENCE_TO_FILTER, MISTRAL_MAX_REPLY, MISTRAL_N_CTX
from app.intelligence import predict_doc_type_for_query
from app.llm import estimate_tokens

logger = logging.getLogger(__name__)

# Per-process routing cache (query string → (doc_type, confidence))
_route_cache: Dict[str, Tuple[str, float]] = {}


# ── Indexing ──────────────────────────────────────────────────────────────────

def build_vector_index(llama_docs: List[Document]) -> VectorStoreIndex:
    """Build a new VectorStoreIndex from a list of LlamaIndex Documents."""
    logger.info("Building vector index from %d documents...", len(llama_docs))
    index = VectorStoreIndex.from_documents(
        llama_docs,
        embed_model=Settings.embed_model,
        show_progress=False,
    )
    logger.info("Index ready with %d nodes.", len(llama_docs))
    return index


# ── Retrieval ─────────────────────────────────────────────────────────────────

def retrieve_chunks(
    index: VectorStoreIndex,
    query: str,
    filter_doc_type: Optional[str] = None,
    auto_route: bool = True,
    k: int = DEFAULT_TOP_K,
    min_confidence: float = MIN_CONFIDENCE_TO_FILTER,
) -> Tuple[List, Optional[str], float]:
    """
    Retrieve relevant chunks with optional metadata filtering.

    Priority order:
    1. Hard filter — user explicitly selected a doc type in the UI
    2. Auto-routing — Gemini predicts the best doc type (cached)
    3. Global search — low confidence or "Other" prediction

    Returns (nodes, routed_doc_type, confidence).
    """
    # ── 1. Hard filter ────────────────────────────────────────────────────────
    if filter_doc_type and filter_doc_type != "All":
        retriever = index.as_retriever(
            similarity_top_k=k,
            filters=MetadataFilters(
                filters=[MetadataFilter(key="doc_type", value=filter_doc_type, operator=FilterOperator.EQ)]
            ),
        )
        logger.debug("Hard filter applied: %s", filter_doc_type)
        return retriever.retrieve(query), filter_doc_type, 1.0

    # ── 2. Auto-routing with cache ────────────────────────────────────────────
    if auto_route:
        cache_key = query.strip().lower()
        if cache_key in _route_cache:
            routed_type, confidence = _route_cache[cache_key]
            logger.debug("Route cache hit: %s (%.2f)", routed_type, confidence)
        else:
            routed_type, confidence = predict_doc_type_for_query(query)
            _route_cache[cache_key] = (routed_type, confidence)
            logger.debug("Auto-routed to: %s (%.2f)", routed_type, confidence)

        if routed_type != "Other" and confidence >= min_confidence:
            retriever = index.as_retriever(
                similarity_top_k=k,
                filters=MetadataFilters(
                    filters=[MetadataFilter(key="doc_type", value=routed_type, operator=FilterOperator.EQ)]
                ),
            )
        else:
            logger.debug("Low confidence or 'Other' → global search")
            retriever = index.as_retriever(similarity_top_k=k)

        return retriever.retrieve(query), routed_type, confidence

    # ── 3. Global (no routing) ────────────────────────────────────────────────
    retriever = index.as_retriever(similarity_top_k=k)
    return retriever.retrieve(query), None, 1.0


# ── Prompt building with context budget ──────────────────────────────────────

def build_rag_prompt_with_budget(
    query: str,
    nodes: List,
    *,
    max_ctx_tokens: int = MISTRAL_N_CTX,
    max_new_tokens: int = MISTRAL_MAX_REPLY,
    safety_margin: int = 128,
) -> Tuple[str, List[Dict], float]:
    """
    Assemble a RAG prompt from retrieved nodes, respecting the context window.

    Greedy strategy: add chunks from highest- to lowest-relevance until budget is full.
    Any chunk that would overflow the budget is silently dropped (a warning is logged).

    Returns (prompt_string, sources_list, avg_relevance_score).
    """
    if not nodes:
        return "", [], 0.0

    budget = max_ctx_tokens - max_new_tokens - safety_margin
    if budget < 512:
        budget = 512

    header = (
        "[INST] You are a pharmaceutical document assistant specialising in quality, "
        "packaging, and compliance documentation.\n"
        "Use ONLY the provided context to answer the question.\n"
        "Be specific — cite the source file, document type, and page(s) where information was found.\n"
        "If the context does not contain enough information, say so clearly.\n\n"
        "Context:\n"
    )
    footer = f"\n\nQuestion: {query} [/INST]"

    used           = estimate_tokens(header) + estimate_tokens(footer)
    context_blocks: List[str] = []
    sources:        List[Dict] = []
    dropped        = 0

    for node in nodes:
        meta     = node.metadata or {}
        doc_type = meta.get("doc_type", "Unknown")
        p_start  = meta.get("page_start", "?")
        p_end    = meta.get("page_end",   "?")
        src_file = meta.get("source_file", "Unknown File")
        score    = node.score if node.score is not None else 0.0

        block        = f"[Source File: {src_file} | Type: {doc_type} | Pages {p_start}–{p_end}]\n{node.text}\n"
        block_tokens = estimate_tokens(block)

        if used + block_tokens > budget:
            dropped += 1
            continue

        context_blocks.append(block)
        used += block_tokens
        sources.append(
            {
                "source_file": src_file,
                "doc_type":    doc_type,
                "pages":       f"{p_start}–{p_end}",
                "relevance":   f"{score:.2%}" if score else "N/A",
                "preview":     node.text[:120] + "...",
            }
        )

    if dropped:
        logger.warning(
            "Context budget: %d tokens used / %d available. "
            "Dropped %d chunk(s). Reduce 'Chunks to Retrieve' if answers seem incomplete.",
            used, budget, dropped,
        )

    prompt    = header + "\n".join(context_blocks) + footer
    avg_score = sum(n.score for n in nodes if n.score is not None) / max(len(nodes), 1)
    return prompt, sources, avg_score
