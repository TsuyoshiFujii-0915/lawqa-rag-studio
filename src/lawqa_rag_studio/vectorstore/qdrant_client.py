"""Qdrant client abstraction for LawQA-RAG-Studio."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Protocol

from lawqa_rag_studio.chunking.base import Chunk
from lawqa_rag_studio.embeddings.dense.openai import embed_texts
from lawqa_rag_studio.embeddings.sparse.splade import embed_texts_splade


class VectorStoreClient(Protocol):
    """Protocol for vector store operations."""

    def upsert_chunks(self, chunks: Iterable[Chunk]) -> None:
        """Upsert chunk payloads and vectors.

        Args:
            chunks: Iterable of chunks to register.
        """
        ...

    def search(self, query_vector: list[float], top_k: int) -> list[Chunk]:
        """Search by dense vector."""
        ...

    def search_sparse(self, query_vector: dict[int, float], top_k: int) -> list[Chunk]:
        """Search by sparse vector."""
        ...


class QdrantStore(VectorStoreClient):
    """Qdrant-backed vector store."""

    def __init__(
        self,
        collection: str,
        dense_model: str,
        sparse_model: str | None,
        dense_dim: int | None,
        location: str,
        server_url: str | None,
        server_api_key: str | None,
        storage_dir: str | None,
        dense_batch_size: int = 32,
        sparse_batch_size: int = 16,
    ) -> None:
        """Initialize store.

        Args:
            url: Qdrant endpoint URL.
            api_key: API key or None.
            collection: Collection name.
            dense_model: Dense embedding model name.
            sparse_model: Sparse embedding model name or None.
            dense_dim: Optional dense vector dimension override.
        """
        from qdrant_client import QdrantClient

        self.collection = collection
        self.dense_model = dense_model
        self.sparse_model = sparse_model
        self.dense_batch_size = dense_batch_size
        self.sparse_batch_size = sparse_batch_size
        if location == "server":
            if not server_url:
                raise ValueError("Qdrant url is required for server mode")
            self.client = QdrantClient(url=server_url, api_key=server_api_key)
        elif location == "local":
            if not storage_dir:
                raise ValueError("storage_dir is required for local mode")
            Path(str(storage_dir)).mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=str(storage_dir))
        elif location == "in-memory":
            self.client = QdrantClient(location=":memory:")
        else:
            raise ValueError(f"Unsupported Qdrant location: {location}")
        self._dense_dim = dense_dim

    def upsert_chunks(self, chunks: Iterable[Chunk]) -> None:
        """Upsert chunks into Qdrant with on-the-fly embeddings.

        Args:
            chunks: Iterable of chunk payloads to store.
        """
        from qdrant_client.http import models as rest
        from qdrant_client.http.models import Distance, SparseVectorParams, VectorParams

        chunk_list = list(chunks)
        logging.getLogger(__name__).info("Upserting %d chunks into Qdrant", len(chunk_list))
        dense_vectors = embed_texts(
            self.dense_model, [c["text"] for c in chunk_list], batch_size=self.dense_batch_size
        )
        sparse_vectors = (
            embed_texts_splade(
                self.sparse_model, [c["text"] for c in chunk_list], batch_size=self.sparse_batch_size
            )
            if self.sparse_model
            else [None] * len(chunk_list)
        )
        # lazily create collection when first vectors are available
        if self.collection not in [c.name for c in self.client.get_collections().collections]:
            if not dense_vectors:
                raise RuntimeError("No dense vectors to infer dimension from; cannot create collection.")
            dim = self._dense_dim or len(dense_vectors[0])
            vectors_conf = {"dense": VectorParams(size=dim, distance=Distance.COSINE)}
            sparse_conf = {"sparse": SparseVectorParams()} if self.sparse_model else None
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=vectors_conf,
                sparse_vectors_config=sparse_conf,
            )
        points = []
        for idx, (chunk, dvec, svec) in enumerate(zip(chunk_list, dense_vectors, sparse_vectors)):
            points.append(
                rest.PointStruct(
                    id=idx,
                    vector={
                        "dense": dvec,
                        **(
                            {"sparse": rest.SparseVector(indices=list(svec.keys()), values=list(svec.values()))}
                            if svec is not None
                            else {}
                        ),
                    },
                    payload={"text": chunk["text"], **chunk.get("metadata", {})},
                )
            )
        if points:
            self.client.upsert(collection_name=self.collection, points=points)

    def search(self, query_vector: list[float], top_k: int) -> list[Chunk]:
        """Search for nearest chunks.

        Args:
            query_vector: Dense embedding of the query.
            top_k: Number of results to return.

        Returns:
            Ranked list of matching chunks with scores in metadata.
        """
        results = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            using="dense",
            limit=top_k,
            with_payload=True,
        )
        chunks: list[Chunk] = []
        for hit in (results.points or []):
            payload = hit.payload or {}
            chunks.append(
                Chunk(
                    id=str(hit.id),
                    text=str(payload.get("text", "")),
                    metadata={**{k: v for k, v in payload.items() if k != "text"}, "score": float(hit.score)},
                )
            )
        return chunks

    def search_sparse(self, query_vector: dict[int, float], top_k: int) -> list[Chunk]:
        """Search using sparse vectors.

        Args:
            query_vector: Sparse representation mapping token index to weight.
            top_k: Number of results to return.

        Returns:
            Ranked list of matching chunks with scores in metadata.
        """
        from qdrant_client.http import models as rest

        if not self.sparse_model:
            raise RuntimeError("Sparse search requested but no sparse model configured.")

        sparse_query = rest.SparseVector(indices=list(query_vector.keys()), values=list(query_vector.values()))
        results = self.client.query_points(
            collection_name=self.collection,
            query=sparse_query,
            using="sparse",
            limit=top_k,
            with_payload=True,
        )
        chunks: list[Chunk] = []
        for hit in (results.points or []):
            payload = hit.payload or {}
            chunks.append(
                Chunk(
                    id=str(hit.id),
                    text=str(payload.get("text", "")),
                    metadata={**{k: v for k, v in payload.items() if k != "text"}, "score": float(hit.score)},
                )
            )
        return chunks




__all__ = ["VectorStoreClient", "QdrantStore"]
