"""Chunking interfaces."""
from __future__ import annotations

from typing import Protocol, TypedDict

from lawqa_rag_studio.data.law_tree import LawNode


class Chunk(TypedDict):
    """Chunk of law text with metadata."""

    id: str
    text: str
    metadata: dict[str, str | int | float | list[str] | float]


class Chunker(Protocol):
    """Protocol for chunkers."""

    def create_chunks(self, root: LawNode) -> list[Chunk]:
        """Create chunks from a law tree.

        Args:
            root: Root law node to chunk.

        Returns:
            List of generated chunks.
        """
        ...


__all__ = ["Chunk", "Chunker"]
