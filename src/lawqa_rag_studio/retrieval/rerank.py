"""Reranking utilities."""
from __future__ import annotations

import logging
from typing import Iterable, Any

from lawqa_rag_studio.chunking.base import Chunk

logger = logging.getLogger(__name__)

_RERANKER_CACHE: dict[str, tuple[Any, Any, str]] = {}


def _select_device() -> "torch.device":
    """Select best available torch device (cuda > mps > cpu)."""
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _get_reranker(model_name: str) -> tuple[Any, Any, "torch.device"]:
    """Load and cache reranker model/tokenizer on best device."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if model_name not in _RERANKER_CACHE:
        device = _select_device()
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model.to(device)
        model.eval()
        _RERANKER_CACHE[model_name] = (model, tokenizer, device.type)
        logger.info("Loaded reranker model=%s device=%s", model_name, device)
    model, tokenizer, device_type = _RERANKER_CACHE[model_name]
    device = torch.device(device_type)
    return model, tokenizer, device


def rerank_chunks(model: str, query: str, candidates: Iterable[Chunk], top_k: int) -> list[Chunk]:
    """Rerank retrieved chunks using a cross-encoder reranker.

    Args:
        model: Reranker model name.
        query: Original query string.
        candidates: Candidate chunks to rerank.
        top_k: Number of chunks to keep.

    Returns:
        Reranked top-K chunks with rerank scores in metadata.
    """
    import torch

    chunks = list(candidates)
    if not chunks or top_k <= 0:
        return []

    model_ce, tokenizer, device = _get_reranker(model)
    logger.info(
        "Reranking %d candidates with model=%s device=%s top_k=%d",
        len(chunks),
        model,
        device,
        top_k,
    )

    texts = [chunk["text"] for chunk in chunks]
    encoded = tokenizer(
        [query] * len(texts),
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        outputs = model_ce(**encoded)
        scores = outputs.logits.squeeze(-1)
    scores_list = scores.detach().cpu().tolist()

    ranked = sorted(zip(chunks, scores_list), key=lambda x: x[1], reverse=True)
    top = ranked[: min(top_k, len(ranked))]

    reranked: list[Chunk] = []
    for chunk, score in top:
        meta = dict(chunk["metadata"])
        meta["rerank_score"] = float(score)
        reranked.append(Chunk(id=chunk["id"], text=chunk["text"], metadata=meta))

    logger.debug(
        "Rerank top3 sample: %s",
        [(c["id"], c["metadata"].get("rerank_score")) for c in reranked[:3]],
    )
    return reranked


__all__ = ["rerank_chunks"]
