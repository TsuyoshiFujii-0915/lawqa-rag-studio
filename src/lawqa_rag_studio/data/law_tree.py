"""Law tree structures and helpers."""
from __future__ import annotations

from typing import Any, Literal, TypedDict


class LawNode(TypedDict):
    """Structured representation of a law node."""

    node_id: str
    type: Literal[
        "law",
        "suppl_provision",
        "chapter",
        "section",
        "subsection",
        "division",
        "article",
        "paragraph",
        "item",
    ]
    title: str | None
    text: str | None
    children: list["LawNode"]
    meta: dict[str, Any]


def flatten_law_nodes(root: LawNode) -> list[LawNode]:
    """Flatten law tree into a list preserving preorder traversal.

    Args:
        root: Root node of the law tree.

    Returns:
        List of `LawNode` in preorder.
    """
    nodes: list[LawNode] = []

    def _walk(node: LawNode) -> None:
        nodes.append(node)
        for child in node.get("children", []):
            _walk(child)

    _walk(root)
    return nodes


__all__ = ["LawNode", "flatten_law_nodes"]
