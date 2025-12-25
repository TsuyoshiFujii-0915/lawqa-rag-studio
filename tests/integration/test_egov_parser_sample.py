"""Smoke test for e-Gov XML parsing."""
from __future__ import annotations

from pathlib import Path

import pytest

from lawqa_rag_studio.data.egov_parser import parse_egov_xml


def test_parse_single_xml_produces_structure() -> None:
    """Parse one real XML and verify basic structure."""
    xml_dir = Path("data/egov/xml")
    if not xml_dir.exists():
        pytest.skip("data/egov/xml is not present; download fixtures first.")

    xml_files = sorted(xml_dir.glob("*.xml"))
    if not xml_files:
        pytest.skip("No XML files found in data/egov/xml.")

    target = xml_files[0]
    node = parse_egov_xml(target)

    assert node["type"] == "law"
    assert node["node_id"] == target.stem
    assert isinstance(node.get("children"), list)
    assert node["children"], "expected at least one top-level node"

    # Find an article node anywhere in the parsed tree.
    stack = [node]
    first_article = None
    while stack:
        cur = stack.pop()
        if cur["type"] == "article":
            first_article = cur
            break
        stack.extend(reversed(cur.get("children", [])))

    assert first_article is not None, "expected at least one article node"
    assert isinstance(first_article.get("children"), list)
    for para in first_article.get("children", []):
        assert para["type"] == "paragraph"
