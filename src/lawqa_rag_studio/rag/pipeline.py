"""End-to-end RAG pipeline orchestration."""
from __future__ import annotations

from typing import TypedDict

from lawqa_rag_studio.config.schema import AppConfig
from lawqa_rag_studio.llm.base import LlmClient
from lawqa_rag_studio.retrieval.pipeline import retrieve
from lawqa_rag_studio.vectorstore.qdrant_client import VectorStoreClient


class RagResult(TypedDict):
    """Result of a RAG invocation."""

    answer: str
    llm_raw_answer: str
    used_chunks: list[dict]
    llm_calls: list[dict]
    retrieval_info: dict
    usage: dict | None


def answer_query(
    query: str,
    cfg: AppConfig,
    store: VectorStoreClient,
    llm: LlmClient,
    *,
    force_choice: bool = False,
    rag_mode: str = "plain",
) -> RagResult:
    """Run retrieval and generation for a query.

    Args:
        query: User query string.
        cfg: Application configuration.
        store: Vector store client.
        llm: Language model client.

    Returns:
        `RagResult` containing answer and diagnostics.
    """
    if force_choice:
        query = f"{query}\n\nPlease answer with a single lowercase letter: a, b, c, or d."

    # Agentic mode: let the LLM call the retriever via tools; skip upfront retrieval.
    if rag_mode == "agentic":
        retrieved: list[dict] = []
        response = llm.generate(query, [])
    else:
        retrieved = retrieve(query, cfg, store)
        if not retrieved:
            placeholder = {"id": "none", "text": "", "metadata": {}}
            retrieved = [placeholder]  # type: ignore[list-item]
        response = llm.generate(query, retrieved)
    answer_text = response.get("text", "") if isinstance(response, dict) else str(response)
    usage = response.get("usage") if isinstance(response, dict) else None
    raw_answer = response.get("raw_text", "") if isinstance(response, dict) else answer_text
    return RagResult(
        answer=answer_text,
        llm_raw_answer=raw_answer,
        used_chunks=[{**chunk} for chunk in retrieved],
        llm_calls=[],
        retrieval_info={"retrieved": len(retrieved)},
        usage=usage,
    )


__all__ = ["RagResult", "answer_query"]
