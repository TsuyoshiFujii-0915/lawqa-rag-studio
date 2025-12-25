"""Embedding wrappers."""

from lawqa_rag_studio.embeddings.dense.openai import embed_texts
from lawqa_rag_studio.embeddings.sparse.splade import SparseVector, embed_texts_splade

__all__ = ["embed_texts", "SparseVector", "embed_texts_splade"]
