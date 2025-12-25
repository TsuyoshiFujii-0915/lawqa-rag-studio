"""Sparse retrieval logic."""
from __future__ import annotations

from lawqa_rag_studio.chunking.base import Chunk
from lawqa_rag_studio.embeddings.sparse.splade import SparseVector, embed_texts_splade
from lawqa_rag_studio.vectorstore.qdrant_client import VectorStoreClient


def sparse_search(store: VectorStoreClient, model: str, query: str, top_k: int) -> list[Chunk]:
    """Perform sparse vector search.

    Args:
        store: Vector store client supporting sparse search.
        model: Sparse model identifier.
        query: Query string.
        top_k: Number of desired results.

    Returns:
        Ranked chunks from sparse index.
    """
    query_vec: SparseVector = embed_texts_splade(model, [query])[0]
    return store.search_sparse(dict(query_vec), top_k)


__all__ = ["sparse_search"]
