"""OpenAI embeddings client wrapper."""
from __future__ import annotations

import os
from typing import Sequence
import logging

from lawqa_rag_studio.clients.openai_client import create_openai_client

logger = logging.getLogger(__name__)


def embed_texts(model: str, texts: Sequence[str], batch_size: int = 32) -> list[list[float]]:
    """Embed texts using OpenAI Embeddings API.

    Args:
        model: Embedding model name.
        texts: Sequence of input texts.
        batch_size: Maximum number of texts per API call.

    Returns:
        List of embedding vectors.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set; cannot embed texts.")

    client = create_openai_client(api_key=api_key, base_url=base_url)
    vectors: list[list[float]] = []

    def _chunks(seq: Sequence[str], size: int):
        for i in range(0, len(seq), size):
            yield seq[i : i + size]

    for batch in _chunks(list(texts), max(1, batch_size)):
        logger.info("Dense embedding batch size=%d model=%s", len(batch), model)
        resp = client.embeddings.create(model=model, input=batch)
        vectors.extend(record.embedding for record in resp.data)  # type: ignore[attr-defined]
    return vectors


__all__ = ["embed_texts"]
