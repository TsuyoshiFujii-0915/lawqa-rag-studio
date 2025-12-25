"""Fixed-size chunker implementation."""
from __future__ import annotations

from lawqa_rag_studio.chunking.base import Chunk, Chunker
from lawqa_rag_studio.data.law_tree import LawNode


def _iter_text(root: LawNode) -> str:
    """Concatenate all text fields in preorder.

    Args:
        root: Root law node.

    Returns:
        Concatenated text string.
    """
    parts: list[str] = []

    def _walk(node: LawNode) -> None:
        title = node.get("title")
        if title:
            parts.append(str(title))
        if node.get("text"):
            parts.append(node["text"] or "")
        for child in node.get("children", []):
            _walk(child)

    _walk(root)
    return "\n".join(parts)


class FixedChunker(Chunker):
    """Sliding-window chunker."""

    def __init__(self, max_chars: int, overlap_chars: int) -> None:
        """Initialize chunker.

        Args:
            max_chars: Maximum characters per chunk.
            overlap_chars: Character overlap between chunks.
        """
        self.max_chars = max_chars
        self.overlap_chars = overlap_chars

    def create_chunks(self, root: LawNode) -> list[Chunk]:
        """Create fixed-size chunks.

        Args:
            root: Root law node.

        Returns:
            List of chunks with metadata.
        """
        full_text = _iter_text(root)
        chunks: list[Chunk] = []
        start = 0
        idx = 0
        if self.max_chars <= 0:
            return []
        while start < len(full_text):
            end = min(start + self.max_chars, len(full_text))
            chunk_text = full_text[start:end]
            chunk_id = f"fixed-{idx}"
            chunks.append(
                Chunk(
                    id=chunk_id,
                    text=chunk_text,
                    metadata={"offset_start": start, "offset_end": end},
                )
            )
            idx += 1
            if end == len(full_text):
                break
            start = max(0, end - self.overlap_chars)
        return chunks


__all__ = ["FixedChunker"]
