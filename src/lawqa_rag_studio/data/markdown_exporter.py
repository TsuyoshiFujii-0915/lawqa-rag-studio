"""Export law trees to Markdown for inspection."""
from __future__ import annotations

from pathlib import Path

from lawqa_rag_studio.data.law_tree import LawNode


def export_to_markdown(node: LawNode, output_path: Path) -> None:
    """Export a law tree to Markdown file.

    Args:
        node: Root law node to export.
        output_path: Destination Markdown file path.
    """
    lines: list[str] = []

    def _walk(current: LawNode, depth: int) -> None:
        heading = "#" * min(depth + 1, 6)
        title = current.get("title") or current.get("type", "node")
        lines.append(f"{heading} {title}")
        if current.get("text"):
            lines.append(current["text"] or "")
        for child in current.get("children", []):
            _walk(child, depth + 1)

    _walk(node, 0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n\n".join(lines), encoding="utf-8")


__all__ = ["export_to_markdown"]
