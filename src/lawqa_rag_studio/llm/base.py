"""LLM client interfaces."""
from __future__ import annotations

from typing import Protocol, Sequence, TypedDict, Any

from lawqa_rag_studio.chunking.base import Chunk


class LlmResponse(TypedDict, total=False):
    """LLM response payload with diagnostics."""

    text: str
    usage: dict[str, Any] | None


class LlmClient(Protocol):
    """Protocol for language model clients."""

    def generate(self, query: str, context: Sequence[Chunk]) -> LlmResponse:
        """Generate answer given query and retrieved context with diagnostics.

        Args:
            query: User query.
            context: Retrieved chunks.

        Returns:
            LlmResponse containing answer text and optional usage info.
        """
        ...


__all__ = ["LlmClient", "LlmResponse"]
