"""
main.py — Application entry point.

Usage:
    python main.py

Or with Docker:
    docker run --gpus all -p 7860:7860 pharma-rag
"""

import logging
import os

from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from app.config import EMBED_MODEL_NAME, GRADIO_DEBUG, GRADIO_PORT, GRADIO_SHARE
from app.llm import load_models
from app.ui import create_interface

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("=== Pharma RAG — Starting Up ===")

    # 1. Load embedding model (shared across all retrieval operations)
    logger.info("Loading embedding model: %s", EMBED_MODEL_NAME)
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)
    Settings.llm = None  # LLM calls are managed manually

    # 2. Load Mistral + Gemini
    logger.info("Loading LLMs...")
    load_models()

    # 3. Build and launch Gradio interface
    logger.info("Building Gradio interface...")
    demo = create_interface()

    logger.info("Launching on port %d (share=%s)...", GRADIO_PORT, GRADIO_SHARE)
    demo.launch(
        server_name="0.0.0.0",
        server_port=GRADIO_PORT,
        share=GRADIO_SHARE,
        debug=GRADIO_DEBUG,
    )


if __name__ == "__main__":
    main()
