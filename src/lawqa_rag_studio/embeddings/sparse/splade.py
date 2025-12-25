"""SPLADE sparse embeddings wrapper."""
from __future__ import annotations

from typing import Mapping, Sequence, Any
import logging

SparseVector = Mapping[int, float]

# Lazily loaded to avoid ImportError when sparse is disabled.
_HF_SPLADE_CACHE: dict[str, tuple[Any, Any]] = {}
logger = logging.getLogger(__name__)


def _get_hf_splade(model_name: str) -> tuple[Any, Any]:
    """Load and cache Hugging Face SPLADE model/tokenizer."""
    if model_name not in _HF_SPLADE_CACHE:
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForMaskedLM.from_pretrained(model_name)
        model.eval()
        _HF_SPLADE_CACHE[model_name] = (model, tokenizer)
    return _HF_SPLADE_CACHE[model_name]


def _select_device() -> "torch.device":
    """Select best available torch device (cuda > mps > cpu)."""
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def embed_texts_splade(model: str, texts: Sequence[str], batch_size: int = 16) -> list[SparseVector]:
    """Embed texts using SPLADE.

    Args:
        model: Model name or identifier (Hugging Face hub ID).
        texts: Sequence of texts to embed.

    Returns:
        List of sparse vectors represented as index-value mappings.
    """

    if not texts:
        return []

    # Hugging Face SPLADE path (e.g., bizreach-inc/light-splade-japanese-56M)
    import torch

    model_hf, tokenizer = _get_hf_splade(model)
    device = _select_device()
    model_hf.to(device)
    logger.info(
        "Sparse embedding via HF SPLADE model=%s device=%s batch=%d",
        model,
        device,
        batch_size,
    )

    top_k = 200
    vectors: list[SparseVector] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        encoded = tokenizer(
            list(batch),
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            logits = model_hf(**encoded).logits  # (B, seq, vocab)
            scores = torch.log1p(torch.relu(logits))
            scores = scores * encoded["attention_mask"].unsqueeze(-1)
            scores, _ = torch.max(scores, dim=1)  # (B, vocab)

        scores_cpu = scores.cpu()
        for row in scores_cpu:
            vals, idxs = torch.topk(row, k=min(top_k, row.numel()))
            sparse = {int(i): float(v) for i, v in zip(idxs, vals) if v > 0}
            vectors.append(sparse)
    return vectors


__all__ = ["SparseVector", "embed_texts_splade"]
