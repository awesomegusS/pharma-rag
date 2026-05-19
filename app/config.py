"""
config.py — Central configuration for the Pharma RAG pipeline.
All constants, paths, and model settings live here.
Override any value via environment variables or a .env file.
"""

import os
from typing import Dict, List
from dotenv import load_dotenv

load_dotenv()

# ── API Keys ──────────────────────────────────────────────────────────────────
GEMINI_API_KEY: str = os.environ.get("GEMINI_API_KEY", "")

# ── Model Paths & Settings ────────────────────────────────────────────────────
MISTRAL_PATH: str = os.environ.get("MISTRAL_PATH", "./models/mistral.gguf")
MISTRAL_DOWNLOAD_URL: str = (
    "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
    "/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
)
MISTRAL_N_CTX: int      = int(os.environ.get("MISTRAL_N_CTX", 16384))
MISTRAL_MAX_REPLY: int  = int(os.environ.get("MISTRAL_MAX_REPLY", 512))
MISTRAL_TEMPERATURE: float = float(os.environ.get("MISTRAL_TEMPERATURE", 0.1))
MISTRAL_N_GPU_LAYERS: int  = int(os.environ.get("MISTRAL_N_GPU_LAYERS", -1))  # -1 = all layers on GPU

GEMINI_MODEL: str = os.environ.get("GEMINI_MODEL", "models/gemini-2.5-flash")

# ── Embedding Model ───────────────────────────────────────────────────────────
EMBED_MODEL_NAME: str = os.environ.get("EMBED_MODEL_NAME", "BAAI/bge-small-en-v1.5")

# ── Chunking Defaults ─────────────────────────────────────────────────────────
CHUNK_SIZE: int    = int(os.environ.get("CHUNK_SIZE", 500))
CHUNK_OVERLAP: int = int(os.environ.get("CHUNK_OVERLAP", 100))

# ── Retrieval Defaults ────────────────────────────────────────────────────────
DEFAULT_TOP_K: int                   = int(os.environ.get("DEFAULT_TOP_K", 4))
MIN_CONFIDENCE_TO_FILTER: float      = float(os.environ.get("MIN_CONFIDENCE_TO_FILTER", 0.70))
PARALLEL_WORKERS: int                = int(os.environ.get("PARALLEL_WORKERS", 5))

# ── OCR Settings ──────────────────────────────────────────────────────────────
OCR_DPI: int              = int(os.environ.get("OCR_DPI", 300))
OCR_TEXT_THRESHOLD: int   = int(os.environ.get("OCR_TEXT_THRESHOLD", 50))

# ── App Settings ──────────────────────────────────────────────────────────────
CHAT_HISTORY_PATH: str = os.environ.get("CHAT_HISTORY_PATH", "./chat_history.txt")
GRADIO_SHARE: bool     = os.environ.get("GRADIO_SHARE", "false").lower() == "true"
GRADIO_PORT: int       = int(os.environ.get("GRADIO_PORT", 7860))
GRADIO_DEBUG: bool     = os.environ.get("GRADIO_DEBUG", "false").lower() == "true"

# ── Pharmaceutical Document Types ─────────────────────────────────────────────
VALID_DOC_TYPES: List[str] = [
    "Cover Letter",
    "Certificate of Quality",
    "Packaging Specification",
    "BSE/TSE Declaration",
    "Material Description",
    "Supplier Qualification",
    "Chain of Custody",
    "Other",
]

# ── Rule-Based Keyword Patterns (zero API calls) ──────────────────────────────
RULE_PATTERNS: Dict[str, List[str]] = {
    "Cover Letter": [
        "to whom it may concern", "this letter is provided",
        "storage temperature", "operating temperature", "sincerely",
        "recommended storage",
    ],
    "Certificate of Quality": [
        "certificate of quality", "certificate of analysis",
        "lot number", "date of manufacture", "expiration date",
        "release criteria", "conforms", "purity by",
    ],
    "Packaging Specification": [
        "packaging specification", "packaging component", "blister tray",
        "lid film", "secondary carton", "ecn number", "change history",
        "drawing reference", "effective date",
    ],
    "BSE/TSE Declaration": [
        "transmissible spongiform", "bse", "tse", "animal origin",
        "bovine", "encephalopathies", "prion",
    ],
    "Material Description": [
        "material description", "materials of construction",
        "sterilization compatibility", "operating pressure",
        "physical properties", "shelf life", "platinum-cured",
    ],
    "Supplier Qualification": [
        "supplier qualification", "supplier name", "supplier code",
        "on-site audit", "iso 9001", "iso 13485",
        "fda establishment", "qualification status", "quality agreement",
    ],
    "Chain of Custody": [
        "chain of custody", "manufactured at", "traceability",
        "lot traceable", "distribution center",
    ],
}

# Doc types where a single keyword match is sufficient (very specific vocabulary)
HIGH_SPECIFICITY_TYPES = {"BSE/TSE Declaration", "Chain of Custody", "Packaging Specification"}
