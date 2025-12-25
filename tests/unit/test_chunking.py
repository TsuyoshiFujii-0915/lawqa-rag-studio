"""Tests for chunking strategies using a sample e-Gov-like XML."""
from __future__ import annotations

from pathlib import Path

import pytest

from lawqa_rag_studio.chunking.fixed import FixedChunker, _iter_text
from lawqa_rag_studio.chunking.hierarchy import HierarchyChunker
from lawqa_rag_studio.config.constants import HierarchyLevel
from lawqa_rag_studio.data.egov_parser import parse_egov_xml


FIXTURE_XML = Path("tests/fixtures/sample_law.xml")


def _load_sample_law():
    """Load the sample law fixture into a LawNode."""

    if not FIXTURE_XML.exists():
        raise FileNotFoundError(f"Fixture not found: {FIXTURE_XML}")
    return parse_egov_xml(FIXTURE_XML)


def test_fixed_chunker_produces_expected_windows() -> None:
    """Fixed chunker should slice concatenated text with overlap."""

    law = _load_sample_law()
    chunker = FixedChunker(max_chars=60, overlap_chars=5)
    chunks = chunker.create_chunks(law)

    full_text = _iter_text(law)
    expected = []
    start = 0
    idx = 0
    while start < len(full_text):
        end = min(start + 60, len(full_text))
        expected.append(
            {
                "id": f"fixed-{idx}",
                "text": full_text[start:end],
                "offset_start": start,
                "offset_end": end,
            }
        )
        idx += 1
        if end == len(full_text):
            break
        start = max(0, end - 5)

    assert len(chunks) == len(expected)
    for got, exp in zip(chunks, expected):
        assert got["id"] == exp["id"]
        assert got["text"] == exp["text"]
        assert got["metadata"]["offset_start"] == exp["offset_start"]
        assert got["metadata"]["offset_end"] == exp["offset_end"]


def test_hierarchy_chunker_collects_article_units() -> None:
    """Hierarchy chunker should produce one chunk per article with metadata."""

    law = _load_sample_law()
    chunker = HierarchyChunker(level=HierarchyLevel.ARTICLE, max_chars_per_chunk=200)
    chunks = chunker.create_chunks(law)

    articles = [child for child in law["children"] if child["type"] == "article"]
    assert len(chunks) == len(articles) == 2

    for idx, (chunk, article) in enumerate(zip(chunks, articles)):
        title_lines = [str(law.get("title") or ""), str(article.get("title") or "")]
        expected_text = chunker._collect_text(article, [t for t in title_lines if t])  # type: ignore[attr-defined]
        assert chunk["id"] == f"hier-article-{idx}"
        assert chunk["metadata"]["node_id"] == article["node_id"]
        assert chunk["metadata"]["law_title"] == law.get("title")
        assert chunk["metadata"]["chunk_level"] == HierarchyLevel.ARTICLE.value.value
        assert chunk["metadata"]["hierarchy_path"] == [str(article.get("title") or "")]
        assert chunk["text"] == expected_text
        assert len(chunk["text"]) <= 200


def test_hierarchy_chunker_raises_when_level_missing() -> None:
    """Hierarchy chunker should raise if the requested level does not exist in the tree."""

    law = _load_sample_law()
    chunker = HierarchyChunker(level=HierarchyLevel.CHAPTER, max_chars_per_chunk=200)
    with pytest.raises(ValueError, match="No nodes found for hierarchy level"):
        chunker.create_chunks(law)


def test_hierarchy_chunker_includes_ancestor_titles_in_text() -> None:
    """Lower-level chunks should include all ancestor titles in the chunk text."""

    law = {
        "node_id": "law",
        "type": "law",
        "title": "法",
        "text": None,
        "children": [
            {
                "node_id": "law-chapter-1",
                "type": "chapter",
                "title": "第一章 総則",
                "text": None,
                "children": [
                    {
                        "node_id": "law-chapter-1-article-1",
                        "type": "article",
                        "title": "第一条（目的）",
                        "text": None,
                        "children": [
                            {
                                "node_id": "law-chapter-1-article-1-para-1",
                                "type": "paragraph",
                                "title": "１",
                                "text": "本文",
                                "children": [],
                                "meta": {},
                            }
                        ],
                        "meta": {},
                    }
                ],
                "meta": {},
            }
        ],
        "meta": {},
    }

    chunker = HierarchyChunker(level=HierarchyLevel.PARAGRAPH, max_chars_per_chunk=200)
    chunks = chunker.create_chunks(law)

    assert len(chunks) == 1
    lines = chunks[0]["text"].splitlines()
    assert lines[:4] == ["法", "第一章 総則", "第一条（目的）", "１"]


def test_hierarchy_chunker_splits_long_nodes() -> None:
    """Hierarchy chunker should split oversized nodes into fixed-size chunks."""

    law = {
        "node_id": "law",
        "type": "law",
        "title": "法",
        "text": None,
        "children": [
            {
                "node_id": "law-chapter-1",
                "type": "chapter",
                "title": "第一章",
                "text": None,
                "children": [
                    {
                        "node_id": "law-chapter-1-article-1",
                        "type": "article",
                        "title": "第一条",
                        "text": None,
                        "children": [
                            {
                                "node_id": "law-chapter-1-article-1-para-1",
                                "type": "paragraph",
                                "title": "１",
                                "text": "あ" * 50,
                                "children": [],
                                "meta": {},
                            }
                        ],
                        "meta": {},
                    }
                ],
                "meta": {},
            }
        ],
        "meta": {},
    }

    chunker = HierarchyChunker(level=HierarchyLevel.ARTICLE, max_chars_per_chunk=35)
    chunks = chunker.create_chunks(law)

    assert len(chunks) == 2
    for chunk in chunks:
        assert chunk["text"].startswith("法\n第一章\n第一条\n")
        assert len(chunk["text"]) <= 35
