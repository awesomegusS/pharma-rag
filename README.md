# 💊 Pharma RAG — Pharmaceutical Document Q&A System

An end-to-end RAG (Retrieval-Augmented Generation) application for querying
pharmaceutical supporting documentation files (SDFs). Upload one or more pharmaceutical
blob PDFs, and ask questions in natural language. The system identifies sub-document
types, builds a metadata-enriched vector index, and streams grounded answers with
source citations.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Gradio Web Interface                            │
│   Upload PDFs │ Document Info Panel │ Retrieval Settings │ Chat         │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
               ┌────────────────▼────────────────┐
               │      EnhancedDocumentStore       │
               │  (app/store.py — orchestrator)   │
               └──┬───────────────────────────────┘
                  │
    ┌─────────────▼──────────────┐    ┌────────────────────────────────┐
    │     INGESTION PIPELINE     │    │       QUERY PIPELINE           │
    │                            │    │                                │
    │  1. extract_pages()        │    │  1. predict_doc_type_for_query │
    │     PyMuPDF + Tesseract    │    │     Gemini Flash (cached)      │
    │     OCR fallback           │    │                                │
    │                            │    │  2. retrieve_chunks()          │
    │  2. classify_all_pages     │    │     MetadataFilter scoped      │
    │     Rule-based (instant)   │    │     VectorStoreIndex search    │
    │     + Gemini (parallel)    │    │                                │
    │                            │    │  3. build_rag_prompt           │
    │  3. detect_boundaries      │    │     Token-budget enforcement   │
    │     Type-change fast path  │    │                                │
    │     + parallel same-type   │    │  4. stream_mistral()           │
    │     check                  │    │     Mistral-7B-Instruct GGUF   │
    │                            │    │     Streamed token output      │
    │  4. group_pages →          │    └────────────────────────────────┘
    │     LogicalDocuments       │
    │                            │
    │  5. chunk_logical_docs     │
    │     500-word sliding       │
    │     window, 100 overlap    │
    │                            │
    │  6. build_vector_index     │
    │     BGE-small embeddings   │
    │     LlamaIndex FAISS       │
    └────────────────────────────┘
```

### Technology Stack

| Layer | Component |
|---|---|
| **Answer LLM** | Mistral-7B-Instruct-v0.2 Q4_K_M (GGUF via llama-cpp-python) |
| **Classification LLM** | Gemini 2.5 Flash (API) |
| **Embeddings** | BAAI/bge-small-en-v1.5 (HuggingFace, local) |
| **Vector Index** | LlamaIndex VectorStoreIndex (in-memory FAISS) |
| **PDF Extraction** | PyMuPDF + Tesseract OCR fallback |
| **UI** | Gradio 4.x (streaming, multi-file upload) |

---

## Engineering Work: Latency Optimizations

Processing a 10-page pharmaceutical blob PDF previously took **60+ seconds**.
Three engineering changes reduced this to **~8 seconds**:

### 1. Rule-Based Pre-Filter (Zero API Calls)
Before the classifier prompt was sent to Gemini, every page was matched against a
keyword dictionary for all 8 document types. Pages with unambiguous keywords
(e.g. "Certificate of Quality", "Transmissible spongiform", "Chain of Custody")
are classified instantly with no API call.

- Coverage on the test blob: **7/10 pages classified by rules alone**
- API calls saved per processing run: **~70%**

### 2. Parallel Gemini Classification
The remaining ambiguous pages (those rules couldn't resolve) are sent to Gemini
**concurrently** using `ThreadPoolExecutor`, rather than sequentially.

- Sequential baseline (3 uncertain pages): ~6s
- Parallel (3 concurrent calls): ~2s

### 3. Type-Change Boundary Fast Path
Boundary detection previously called `is_same_document` (an LLM API call) for
**every** page transition. The optimized version skips the API call entirely when
adjacent pages have **different classified types** — a type change is unambiguous
evidence of a document boundary.

For the 10-page blob: 7 of 9 transitions were type-changes → only 2 API calls needed.

| Step | Before | After |
|---|---|---|
| Classification (10 pages) | ~20s (10 sequential) | ~2s (7 rules + 3 parallel) |
| Boundary detection (9 transitions) | ~18s (9 sequential) | ~2s (7 instant + 2 parallel) |
| **Total processing time** | **~60s** | **~8s** |

---

## Performance Metrics

Evaluated on 16 ground-truth queries across 2 test documents
(Sigma-Aldrich CoA T1503 and Sartorius BSE/TSE Declaration).

### Retrieval Performance

| Metric | Value | Target |
|---|---|---|
| Recall@4 | 100% | > 80% |
| Mean Reciprocal Rank (MRR) | 1.0 | > 0.70 |
| Precision@4 | 35.3% | > 70% |
| Hit Rate | 100% | 100% |

### End-to-End Accuracy

| Metric | Value | Notes |
|---|---|---|
| Answer Accuracy | 88.2% | Evaluated on 16 test questions |
| Citation Accuracy | 100.0% | Correct source attribution |
| Factual Consistency | 88.2% | Required keywords present (no hallucination proxy) |

### System Performance

| Metric | Value |
|---|---|
| Avg Response Time | 11.08s |
| Retrieval Latency | 7052ms |
| LLM Generation Time | 4.03s |
| Error Rate | 0.0% |

<!-- > **Note:** Fill in metric values from `eval_summary_<timestamp>.json`
> after running `notebooks/evaluation.ipynb`. -->

---

## Project Structure

```
pharma-rag/
├── app/
│   ├── config.py          # All constants (env-configurable)
│   ├── models.py          # PageInfo, LogicalDocument, ChunkMetadata dataclasses
│   ├── llm.py             # Mistral + Gemini wrappers, token budget utilities
│   ├── intelligence.py    # Rule classifier, Gemini classifier, boundary detection, query routing
│   ├── extraction.py      # PDF extraction, OCR fallback, parallel classification pipeline
│   ├── chunking.py        # Sliding-window chunker with metadata preservation
│   ├── retrieval.py       # Vector indexing, MetadataFilter retrieval, prompt assembly
│   ├── store.py           # EnhancedDocumentStore — session orchestrator
│   └── ui.py              # Gradio interface and all event handlers
├── tests/
│   └── test_intelligence.py  # Unit tests (no API / model required)
├── notebooks/
│   └── evaluation.ipynb   # Full evaluation pipeline (run in Colab)
├── models/                # GGUF weights (gitignored, downloaded at runtime)
├── main.py                # Entry point
├── requirements.txt
├── Dockerfile
├── .env.example
├── .gitignore
└── README.md
```

---

## Installation

### Prerequisites

- Python 3.11+
- NVIDIA GPU with CUDA 12.x (strongly recommended — CPU inference is very slow)
- `tesseract-ocr` system package for scanned PDF support
- A Gemini API key (free tier works)

### Option A: Local (without Docker)

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/pharma-rag.git
cd pharma-rag

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate       # Linux / macOS
# .venv\Scripts\activate        # Windows

# 3. Install Tesseract (system package)
sudo apt-get install tesseract-ocr    # Ubuntu / Debian
# brew install tesseract              # macOS

# 4. Install llama-cpp-python with the right backend for your hardware
# (must be done before pip install -r requirements.txt)

# Linux + NVIDIA GPU:
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python==0.3.23 --no-cache-dir

# macOS Apple Silicon (Metal):
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python==0.3.23 --no-cache-dir

# CPU only (slow):
pip install llama-cpp-python==0.3.23 --no-cache-dir

# 5. Install remaining dependencies
pip install -r requirements.txt

# 6. Configure environment
cp .env.example .env
# Edit .env and set GEMINI_API_KEY

# 7. Run
python main.py
```

Open your browser at `http://localhost:7860`.

The Mistral model (~4.4 GB) downloads automatically on first run to `./models/`.

### Option B: Docker (GPU)

```bash
# 1. Build the image
docker build -t pharma-rag .

# 2. Run with GPU and your .env file
docker run --gpus all \
  -p 7860:7860 \
  --env-file .env \
  -v $(pwd)/models:/app/models \
  pharma-rag
```

> Mount `./models` as a volume so the GGUF file persists across container restarts.

### Option C: CPU-only (no GPU)

```bash
docker run \
  -p 7860:7860 \
  --env-file .env \
  -e MISTRAL_N_GPU_LAYERS=0 \
  -v $(pwd)/models:/app/models \
  pharma-rag
```

Note: CPU-only inference is significantly slower (~30-60s per query).

---

## Usage

1. **Upload** one or more pharmaceutical PDFs using the file panel on the left
2. Click **Process Document** — the system will classify, group, chunk, and index
3. Watch the **Document Info** panel populate with identified sub-documents and types
4. **Ask questions** in the chat panel on the right
5. Optionally adjust **Retrieval Settings** (filter by doc type, change chunk count)
6. **Export Chat History** to download the full conversation as a text file

### Example Questions

- *What is the lot number?*
- *What sterilization method was used?*
- *Are any materials of animal origin present?*
- *What EU regulations are referenced?*
- *Summarise the packaging specification*

---

## Running Tests

```bash
pytest tests/ -v
```

The unit tests cover the rule-based classifier, `clean_doc_type`, and the chunking
logic. They run without any API keys or model weights.

---

## Future Improvements

- **Persistent vector store** (Pinecone / pgvector) so the index survives restarts
  and scales beyond a single session
- **BM25 hybrid retrieval** alongside dense embeddings to improve recall on
  pharmaceutical part numbers and lot IDs (exact-string queries)
- **Cross-encoder reranker** for multi-document summary queries that currently
  commit too early to one doc type
- **Async Gemini calls** (replace ThreadPoolExecutor with asyncio for cleaner
  concurrency and better error handling)
- **Structured output extraction** — a second LLM pass to extract specific fields
  (lot numbers, dates, certifications) into a structured JSON record
- **User authentication** for multi-user deployments
- **Precision@4 of retriever** 
- **Evaluation over more examples/test samples**
- **CI/CD**

---

## Acknowledgements

- [LlamaIndex](https://www.llamaindex.ai/) — RAG orchestration
- [TheBloke/Mistral-7B-Instruct-v0.2-GGUF](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF) — quantised model weights
- [Google Gemini](https://ai.google.dev/) — document classification and query routing
- [Gradio](https://www.gradio.app/) — web interface
