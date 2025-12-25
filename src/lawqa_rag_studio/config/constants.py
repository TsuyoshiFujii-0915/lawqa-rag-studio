"""Constant enumerations for configuration options."""
from __future__ import annotations

from enum import Enum
from typing import NamedTuple


class ConfigOption(NamedTuple):
    """Metadata for a configuration option."""

    value: str
    description: str
    default: bool = False


class ChunkingStrategy(Enum):
    """Chunking strategies."""

    FIXED = ConfigOption("fixed", "Sliding window with fixed size chunks", True)
    HIERARCHY = ConfigOption("hierarchy", "Structure-aware chunking")


class HierarchyLevel(Enum):
    """Hierarchy levels for chunking."""

    LAW = ConfigOption("law", "Entire law as one chunk")
    SUPPL_PROVISION = ConfigOption("suppl_provision", "Chunk per supplementary provision block")
    CHAPTER = ConfigOption("chapter", "Chunk per chapter")
    SECTION = ConfigOption("section", "Chunk per section", True)
    SUBSECTION = ConfigOption("subsection", "Chunk per subsection")
    DIVISION = ConfigOption("division", "Chunk per division")
    ARTICLE = ConfigOption("article", "Chunk per article")
    PARAGRAPH = ConfigOption("paragraph", "Chunk per paragraph")


class DenseEmbeddingModel(Enum):
    """Dense embedding model options."""

    TEXT_EMBEDDING_3_SMALL = ConfigOption(
        "text-embedding-3-small", "OpenAI small embedding model (1536-dim)", True
    )
    TEXT_EMBEDDING_3_LARGE = ConfigOption(
        "text-embedding-3-large", "OpenAI large embedding model (3072-dim)"
    )


class SparseEmbeddingModel(Enum):
    """Sparse embedding model options."""

    LIGHT_SPLADE_JAPANESE = ConfigOption(
        "bizreach-inc/light-splade-japanese-56M",
        "Japanese-oriented lightweight SPLADE model",
        True,
    )


class HybridCombineMethod(Enum):
    """Hybrid score fusion methods."""

    RRF = ConfigOption("rrf", "Reciprocal Rank Fusion", True)
    LINEAR = ConfigOption("linear", "Normalized linear combination")


class RerankModel(Enum):
    """Reranker model options."""

    BGE_RERANKER_V2_M3 = ConfigOption(
        "BAAI/bge-reranker-v2-m3", "Multilingual reranker", True
    )
    JAPANESE_BGE_RERANKER = ConfigOption(
        "hotchpotch/japanese-bge-reranker-v2-m3-v1", "Japanese-specific reranker"
    )


class LLMProvider(Enum):
    """LLM provider options."""

    OPENAI = ConfigOption("openai", "OpenAI API", True)
    LMSTUDIO = ConfigOption("lmstudio", "LM Studio OpenAI-compatible API")


class LLMModel(Enum):
    """LLM model options."""

    GPT_5_2 = ConfigOption("gpt-5.2", "Latest GPT-5.2")
    GPT_5_1 = ConfigOption("gpt-5.1", "GPT-5.1")
    GPT_5 = ConfigOption("gpt-5", "GPT-5")
    GPT_5_MINI = ConfigOption("gpt-5-mini", "Lightweight GPT-5 Mini", True)
    GPT_5_NANO = ConfigOption("gpt-5-nano", "Smallest GPT-5 Nano")
    GPT_OSS_20B = ConfigOption("openai/gpt-oss-20b", "20B OSS model via LM Studio")


class LogLevel(Enum):
    """Logging level options."""

    DEBUG = ConfigOption("DEBUG", "Verbose debug logging")
    INFO = ConfigOption("INFO", "Standard info logging", True)
    WARNING = ConfigOption("WARNING", "Warnings only")
    ERROR = ConfigOption("ERROR", "Errors only")


class RagMode(Enum):
    """RAG execution modes for OpenAI Responses API."""

    PLAIN = ConfigOption(
        "plain", "Classic RAG: context stuffed into prompt", True
    )
    AGENTIC = ConfigOption(
        "agentic", "Agentic RAG: retrieve as tool for the model"
    )


ALL_CONFIG_ENUMS = {
    "chunking.strategy": ChunkingStrategy,
    "chunking.hierarchy.level": HierarchyLevel,
    "embedding.dense.model": DenseEmbeddingModel,
    "embedding.sparse.model": SparseEmbeddingModel,
    "retriever.hybrid.combine": HybridCombineMethod,
    "retriever.rerank.model": RerankModel,
    "llm.provider": LLMProvider,
    "llm.model": LLMModel,
    "llm.openai.rag_mode": RagMode,
    "logging.level": LogLevel,
}

__all__ = [
    "ConfigOption",
    "ChunkingStrategy",
    "HierarchyLevel",
    "DenseEmbeddingModel",
    "SparseEmbeddingModel",
    "HybridCombineMethod",
    "RerankModel",
    "LLMProvider",
    "LLMModel",
    "LogLevel",
    "RagMode",
    "ALL_CONFIG_ENUMS",
]
