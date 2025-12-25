"""Query transformation utilities (HyDE, multi-query)."""
from __future__ import annotations

from typing import Sequence


def generate_hyde_document(prompt_template: str, query: str, model: str) -> str:
    """Generate hypothetical document for HyDE.

    Args:
        prompt_template: Template identifier.
        query: Original user query.
        model: LLM model to use.

    Returns:
        Generated pseudo document text.
    """
    # Simple template-based pseudo answer.
    return f"{prompt_template}: Hypothetical answer for '{query}' using model {model}."


def generate_query_variants(query: str, num_variants: int) -> list[str]:
    """Generate query reformulations.

    Args:
        query: Original query string.
        num_variants: Number of reformulations to produce.

    Returns:
        List of variant queries.
    """
    variants = [query]
    for i in range(1, num_variants):
        variants.append(f"{query} (variant {i})")
    return variants


def compress_context(chunks: Sequence[str], target_tokens: int) -> list[str]:
    """Compress retrieved context to target tokens.

    Args:
        chunks: Sequence of chunk texts.
        target_tokens: Token budget for compressed context.

    Returns:
        Compressed chunk texts.
    """
    # Naive compression by truncation.
    compressed: list[str] = []
    for text in chunks:
        tokens = text.split()
        if len(tokens) > target_tokens:
            compressed.append(" ".join(tokens[:target_tokens]))
        else:
            compressed.append(text)
    return compressed


__all__ = [
    "generate_hyde_document",
    "generate_query_variants",
    "compress_context",
]
