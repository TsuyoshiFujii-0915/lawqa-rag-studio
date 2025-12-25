"""Sparse embedding backends."""

from lawqa_rag_studio.embeddings.sparse.splade import SparseVector, embed_texts_splade

__all__ = ["SparseVector", "embed_texts_splade"]
