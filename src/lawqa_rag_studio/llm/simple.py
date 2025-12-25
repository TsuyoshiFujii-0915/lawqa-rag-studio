"""Lightweight local LLM mock for offline testing."""
from __future__ import annotations

from typing import Sequence

from lawqa_rag_studio.chunking.base import Chunk
from lawqa_rag_studio.llm.base import LlmClient, LlmResponse


class SimpleLocalClient(LlmClient):
    """Rule-based lightweight responder for offline use."""

    def generate(self, query: str, context: Sequence[Chunk]) -> LlmResponse:
        """Return echo-style answer using available context.

        Args:
            query: User query string.
            context: Retrieved chunks.

        Returns:
            Generated answer text.
        """
        context_text = "\n\n".join(c["text"] for c in context[:3])
        return {
            "text": f"[Local mock answer]\nQuestion: {query}\nContext snippet:\n{context_text[:400]}",
            "usage": None,
        }


__all__ = ["SimpleLocalClient"]
