"""
llm.py — LLM loading and inference wrappers.

  - Mistral-7B-Instruct (GGUF via llama-cpp-python): answer generation
  - Gemini Flash: document classification & boundary detection
"""

import os
import logging
from typing import Generator

from google import genai
from google.genai import types
from llama_cpp import Llama

from app.config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    MISTRAL_PATH,
    MISTRAL_DOWNLOAD_URL,
    MISTRAL_N_CTX,
    MISTRAL_MAX_REPLY,
    MISTRAL_TEMPERATURE,
    MISTRAL_N_GPU_LAYERS,
)

logger = logging.getLogger(__name__)

# ── Module-level singletons (initialised by load_models()) ───────────────────
_mistral_llm: Llama | None = None
_gemini_client = None


def load_models() -> None:
    """
    Initialise both LLMs. Call once at application startup (main.py).
    Downloading Mistral GGUF (~4.4 GB) only happens on first run.
    """
    global _mistral_llm, _gemini_client 
    # ── Gemini ────────────────────────────────────────────────────────────────
    if not GEMINI_API_KEY:
        raise EnvironmentError(
            "GEMINI_API_KEY is not set. Add it to your .env file or environment."
        )
    _gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    logger.info("Gemini model loaded: %s", GEMINI_MODEL)

    # ── Mistral ───────────────────────────────────────────────────────────────
    _mistral_llm = load_mistral()


def load_mistral(
    model_path: str = MISTRAL_PATH,
    n_gpu_layers: int = MISTRAL_N_GPU_LAYERS,
) -> Llama:
    """Download (if needed) and load Mistral-7B-Instruct Q4_K_M GGUF."""
    os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)

    if not os.path.exists(model_path):
        logger.info("Downloading Mistral GGUF to %s ...", model_path)
        ret = os.system(f"curl -L --progress-bar '{MISTRAL_DOWNLOAD_URL}' -o '{model_path}'")
        if ret != 0:
            raise RuntimeError(f"Failed to download Mistral model from {MISTRAL_DOWNLOAD_URL}")
        logger.info("Download complete.")

    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    logger.info(
        "Loading Mistral (%.0f MB) | n_ctx=%d | n_gpu_layers=%d",
        size_mb, MISTRAL_N_CTX, n_gpu_layers,
    )

    llm = Llama(
        model_path=model_path,
        n_gpu_layers=n_gpu_layers,
        n_ctx=MISTRAL_N_CTX,
        verbose=False,
    )
    logger.info("Mistral loaded. Context window: %d tokens.", MISTRAL_N_CTX)
    return llm


# ── Inference helpers ─────────────────────────────────────────────────────────

def call_mistral(prompt: str) -> str:
    """Synchronous Mistral inference. Returns plain text."""
    if _mistral_llm is None:
        raise RuntimeError("Models not loaded. Call load_models() first.")
    output = _mistral_llm.create_completion(
        prompt=prompt,
        max_tokens=MISTRAL_MAX_REPLY,
        temperature=MISTRAL_TEMPERATURE,
        stream=False,
    )
    return output["choices"][0]["text"].strip()


def stream_mistral(
    prompt: str,
    *,
    max_tokens: int = MISTRAL_MAX_REPLY,
    temperature: float = MISTRAL_TEMPERATURE,
) -> Generator[str, None, None]:
    """Stream token text from Mistral. Yields partial strings."""
    if _mistral_llm is None:
        raise RuntimeError("Models not loaded. Call load_models() first.")
    stream = _mistral_llm.create_completion(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        stream=True,
    )
    for chunk in stream:
        text = chunk["choices"][0].get("text", "")
        if text:
            yield text


def call_gemini(prompt: str) -> str:
    if _gemini_client is None:
        raise RuntimeError("Models not loaded. Call load_models() first.")
    try:
        response = _gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )
        return response.text.strip()
    except Exception as exc:
        logger.warning("Gemini API error: %s", exc)
        return ""


# ── Token budget utilities ────────────────────────────────────────────────────

def estimate_tokens(text: str) -> int:
    """Fast character-based token approximation (~1 token per 4 chars)."""
    return max(1, len(text) // 4)


def available_prompt_budget() -> int:
    """Maximum tokens available for the prompt (reserves space for reply)."""
    return MISTRAL_N_CTX - MISTRAL_MAX_REPLY
