"""Dense retrieval logic."""
from __future__ import annotations

from lawqa_rag_studio.chunking.base import Chunk
from lawqa_rag_studio.embeddings.dense.openai import embed_texts
from lawqa_rag_studio.vectorstore.qdrant_client import VectorStoreClient


def dense_search(
    store: VectorStoreClient, model: str, query: str, top_k: int
) -> list[Chunk]:
    """Perform dense vector search.

    Args:
        store: Vector store client.
        model: Embedding model name.
        query: Query string.
        top_k: Number of results.

    Returns:
        Ranked chunks.
    """
    query_vec = embed_texts(model, [query])[0]
    return store.search(query_vec, top_k)


__all__ = ["dense_search"]
