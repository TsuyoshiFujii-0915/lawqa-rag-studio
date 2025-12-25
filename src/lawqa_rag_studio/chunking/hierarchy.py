"""Hierarchy-aware chunker implementation."""
from __future__ import annotations

from lawqa_rag_studio.chunking.base import Chunk, Chunker
from lawqa_rag_studio.config.constants import HierarchyLevel
from lawqa_rag_studio.data.law_tree import LawNode


class HierarchyChunker(Chunker):
    """Chunker that respects law hierarchy."""

    def __init__(self, level: HierarchyLevel, max_chars_per_chunk: int) -> None:
        """Initialize chunker.

        Args:
            level: Hierarchy level for splitting.
            max_chars_per_chunk: Maximum characters per chunk.
        """
        self.level = level
        self.max_chars_per_chunk = max_chars_per_chunk

    def create_chunks(self, root: LawNode) -> list[Chunk]:
        """Create hierarchy-based chunks.

        Args:
            root: Root law node.

        Returns:
            List of chunks with metadata.
        """
        chunks: list[Chunk] = []
        level_value = self._level_value()
        nodes_with_path = self._iter_level_nodes_with_path(root, level_value)
        nodes = [node for node, _ in nodes_with_path]
        if not nodes:
            stack = [root]
            available_types: set[str] = set()
            while stack:
                current = stack.pop()
                available_types.add(current["type"])
                stack.extend(reversed(current.get("children", [])))
            available = sorted(available_types)
            raise ValueError(
                f"No nodes found for hierarchy level '{level_value}'. "
                f"Check your parser output and config; available node types: {available}"
            )

        law_title = str(root.get("title") or "")
        for idx, (node, path_titles) in enumerate(nodes_with_path):
            title_lines = [law_title, *path_titles] if law_title else [*path_titles]
            base_text = self._collect_text(node, title_lines)
            if len(base_text) <= self.max_chars_per_chunk:
                chunk_id = f"hier-{level_value}-{idx}"
                chunks.append(
                    Chunk(
                        id=chunk_id,
                        text=base_text,
                        metadata={
                            "node_id": node["node_id"],
                            "law_title": root.get("title"),
                            "chunk_level": level_value,
                            "hierarchy_path": path_titles,
                            "split_index": 0,
                        },
                    )
                )
                continue

            split_chunks = self._split_long_text(node, title_lines)
            for part_idx, chunk_text in enumerate(split_chunks):
                chunk_id = f"hier-{level_value}-{idx}-{part_idx}"
                chunks.append(
                    Chunk(
                        id=chunk_id,
                        text=chunk_text,
                        metadata={
                            "node_id": node["node_id"],
                            "law_title": root.get("title"),
                            "chunk_level": level_value,
                            "hierarchy_path": path_titles,
                            "split_index": part_idx,
                        },
                    )
                )
        return chunks

    def _iter_level_nodes_with_path(
        self, root: LawNode, level_value: str
    ) -> list[tuple[LawNode, list[str]]]:
        """Collect nodes matching the target level with their title path.

        The returned title path contains titles from the first level below the law
        down to the node itself. The law title is intentionally excluded so it can
        be controlled independently in chunk text/metadata.

        Args:
            root: Root law node.
            level_value: Target level string value.

        Returns:
            List of (node, title_path) tuples.
        """
        target = level_value
        stack: list[tuple[LawNode, list[str]]] = [(root, [])]
        collected: list[tuple[LawNode, list[str]]] = []
        while stack:
            node, path = stack.pop()
            node_title = str(node.get("title") or "").strip()
            current_path = path
            if node["type"] != "law" and node_title:
                current_path = [*path, node_title]

            if node["type"] == target:
                collected.append((node, current_path))

            for child in reversed(node.get("children", [])):
                stack.append((child, current_path))
        return collected

    def _iter_level_nodes(self, root: LawNode, level_value: str) -> list[LawNode]:
        """Yield nodes that match target level.

        Args:
            root: Root law node.
            level_value: Target level string value.

        Returns:
            List of nodes at target level.
        """
        target = level_value
        stack = [root]
        collected: list[LawNode] = []
        while stack:
            node = stack.pop()
            if node["type"] == target:
                collected.append(node)
            stack.extend(reversed(node.get("children", [])))
        return collected

    def _collect_text(self, node: LawNode, title_lines: list[str]) -> str:
        """Collect concatenated text from a node subtree with titles.

        Args:
            node: Subtree root node.
            title_lines: Title lines to prepend (e.g., hierarchy path titles).

        Returns:
            Concatenated text suitable for embedding and retrieval.
        """
        parts: list[str] = [line for line in title_lines if line]
        stack = [node]
        while stack:
            current = stack.pop()
            if current.get("text"):
                parts.append(current["text"] or "")
            stack.extend(reversed(current.get("children", [])))
        return "\n".join(parts)

    def _split_long_text(self, node: LawNode, title_lines: list[str]) -> list[str]:
        """Split long text into fixed-size chunks with title prefix.

        Args:
            node: Subtree root node.
            title_lines: Title lines to prepend to each chunk.

        Returns:
            List of chunk texts, each with length <= max_chars_per_chunk.
        """
        prefix = "\n".join([line for line in title_lines if line])
        if prefix:
            prefix = f"{prefix}\n"

        body_text = self._collect_body_text(node)
        if not body_text:
            return [prefix[: self.max_chars_per_chunk]] if prefix else [""]

        available = self.max_chars_per_chunk - len(prefix)
        if available <= 0:
            return [prefix[: self.max_chars_per_chunk]]

        chunks: list[str] = []
        start = 0
        while start < len(body_text):
            segment = body_text[start : start + available]
            chunks.append(f"{prefix}{segment}" if prefix else segment)
            start += available
        return chunks

    def _collect_body_text(self, node: LawNode) -> str:
        """Collect concatenated text from a node subtree without titles.

        Args:
            node: Subtree root node.

        Returns:
            Concatenated body text without title lines.
        """
        parts: list[str] = []
        stack = [node]
        while stack:
            current = stack.pop()
            if current.get("text"):
                parts.append(current["text"] or "")
            stack.extend(reversed(current.get("children", [])))
        return "\n".join(parts)

    def _level_value(self) -> str:
        """Return the string value of the configured level."""

        value = getattr(self.level, "value", self.level)
        return getattr(value, "value", value)


__all__ = ["HierarchyChunker"]
