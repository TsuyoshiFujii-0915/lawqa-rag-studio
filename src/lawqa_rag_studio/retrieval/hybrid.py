"""Hybrid retrieval fusion methods."""
from __future__ import annotations

from typing import Iterable, Dict

from lawqa_rag_studio.chunking.base import Chunk
from lawqa_rag_studio.config.constants import HybridCombineMethod


def reciprocal_rank_fusion(dense: Iterable[Chunk], sparse: Iterable[Chunk], k: int) -> list[Chunk]:
    """Fuse dense and sparse rankings via RRF.

    Args:
        dense: Dense-ranked chunks.
        sparse: Sparse-ranked chunks.
        k: Number of results to return.

    Returns:
        Fused top-K chunks.
    """
    rank_dict: Dict[str, float] = {}
    for rank, chunk in enumerate(dense, start=1):
        rank_dict[chunk["id"]] = rank_dict.get(chunk["id"], 0.0) + 1.0 / (60 + rank)
    for rank, chunk in enumerate(sparse, start=1):
        rank_dict[chunk["id"]] = rank_dict.get(chunk["id"], 0.0) + 1.0 / (60 + rank)
    sorted_ids = sorted(rank_dict.items(), key=lambda kv: kv[1], reverse=True)
    top_ids = {cid for cid, _ in sorted_ids[:k]}
    merged: list[Chunk] = []
    for seq in (dense, sparse):
        for chunk in seq:
            if chunk["id"] in top_ids and chunk not in merged:
                merged.append(chunk)
                if len(merged) >= k:
                    return merged
    return merged


def linear_fusion(
    dense: Iterable[Chunk], sparse: Iterable[Chunk], weights: dict[str, float], k: int
) -> list[Chunk]:
    """Fuse rankings by weighted linear scores.

    Args:
        dense: Dense-ranked chunks with scores in metadata.
        sparse: Sparse-ranked chunks with scores in metadata.
        weights: Weight dictionary for sources.
        k: Number of results to return.

    Returns:
        Fused top-K chunks.
    """
    wd = weights.get("dense", 0.5)
    ws = weights.get("sparse", 0.5)
    scores: Dict[str, float] = {}
    chunk_map: Dict[str, Chunk] = {}
    for c in dense:
        score = float(c.get("metadata", {}).get("score", 0.0))
        scores[c["id"]] = scores.get(c["id"], 0.0) + wd * score
        chunk_map[c["id"]] = c
    for c in sparse:
        score = float(c.get("metadata", {}).get("score", 0.0))
        scores[c["id"]] = scores.get(c["id"], 0.0) + ws * score
        chunk_map[c["id"]] = c
    sorted_ids = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [chunk_map[cid] for cid, _ in sorted_ids[:k]]


def hybrid_search(
    method: HybridCombineMethod,
    dense_results: list[Chunk],
    sparse_results: list[Chunk],
    top_k: int,
    linear_weights: dict[str, float] | None = None,
) -> list[Chunk]:
    """Combine dense and sparse search results.

    Args:
        method: Fusion method to use.
        dense_results: Dense retrieval results.
        sparse_results: Sparse retrieval results.
        top_k: Number of fused results.
        linear_weights: Optional weights for linear fusion.

    Returns:
        Fused ranked chunks.
    """
    if method == HybridCombineMethod.RRF:
        return reciprocal_rank_fusion(dense_results, sparse_results, top_k)
    return linear_fusion(dense_results, sparse_results, linear_weights or {}, top_k)


__all__ = ["hybrid_search", "reciprocal_rank_fusion", "linear_fusion"]
