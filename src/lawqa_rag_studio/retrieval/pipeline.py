"""Retrieval pipeline orchestration."""
from __future__ import annotations

from lawqa_rag_studio.chunking.base import Chunk
from lawqa_rag_studio.config.schema import AppConfig
from lawqa_rag_studio.retrieval.dense import dense_search
from lawqa_rag_studio.retrieval.hybrid import hybrid_search
from lawqa_rag_studio.retrieval.rerank import rerank_chunks
from lawqa_rag_studio.retrieval.sparse import sparse_search
from lawqa_rag_studio.vectorstore.qdrant_client import VectorStoreClient


def retrieve(query: str, cfg: AppConfig, store: VectorStoreClient) -> list[Chunk]:
    """Retrieve chunks for a query based on configuration.

    Args:
        query: User query string.
        cfg: Application configuration.
        store: Vector store client.

    Returns:
        Ranked chunks ready for generation.
    """
    dense_results = dense_search(store, cfg.embedding.dense.model, query, cfg.retriever.dense.top_k)
    sparse_results: list[Chunk] = []
    if cfg.embedding.sparse.enabled:
        sparse_results = sparse_search(store, cfg.embedding.sparse.model, query, cfg.retriever.sparse.top_k)
    fused = hybrid_search(
        cfg.retriever.hybrid.combine,
        dense_results,
        sparse_results,
        cfg.retriever.hybrid.top_k,
        cfg.retriever.hybrid.linear_weights,
    )
    if cfg.retriever.rerank.enabled and cfg.retriever.rerank.mode == "cross_encoder":
        model_name = cfg.retriever.rerank.cross_encoder.get("model", "")
        return rerank_chunks(model_name, query, fused, cfg.retriever.rerank.top_k)
    return fused


__all__ = ["retrieve"]
