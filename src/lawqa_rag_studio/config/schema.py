"""Configuration schema definitions using Pydantic."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, model_validator, field_validator

from lawqa_rag_studio.config.constants import ChunkingStrategy, HierarchyLevel, HybridCombineMethod, LogLevel, RagMode


class ExperimentConfig(BaseModel):
    """Top-level experiment settings."""

    name: str = Field(..., description="Experiment name")
    seed: int = Field(42, description="Random seed")
    output_dir: Path = Field(..., description="Output directory path")


class EgovConfig(BaseModel):
    """Configuration for e-Gov corpus."""

    enabled: bool = Field(True, description="Whether to use e-Gov corpus")
    xml_dir: Path = Field(..., description="Directory containing XML files")
    law_tree_cache: Path = Field(..., description="Cache path for parsed law tree")
    law_list_path: Path = Field(..., description="Path to law_list.json in lawqa_jp")
    api_base_url: str = Field(
        "https://laws.e-gov.go.jp/api/2/law_file/xml",
        description="Base URL for e-Gov law XML API",
    )


class LawQaConfig(BaseModel):
    """Configuration for lawqa_jp dataset."""

    path: Path = Field(..., description="Path to lawqa_jp JSON file")


class DataConfig(BaseModel):
    """Grouping of dataset configurations."""

    egov: EgovConfig
    lawqa_jp: LawQaConfig


class QdrantLocalConfig(BaseModel):
    """Local embedded Qdrant settings."""

    storage_dir: Path = Field(Path("./data/qdrant"), description="Storage directory for embedded Qdrant")


class QdrantServerConfig(BaseModel):
    """Remote/server Qdrant settings."""

    url: str = Field(..., description="Qdrant endpoint URL")
    api_key: Optional[str] = Field(None, description="API key or None")


class QdrantConfig(BaseModel):
    """Qdrant vector store settings."""

    collection_name: str = Field(..., description="Collection name")
    location: Literal["local", "server", "in-memory"] = Field(
        "local", description="Deployment location"
    )
    local: QdrantLocalConfig = Field(default_factory=QdrantLocalConfig)
    server: QdrantServerConfig = Field(
        default_factory=lambda: QdrantServerConfig(url="http://localhost:6333", api_key=None)
    )

    @model_validator(mode="after")
    def _validate_fields(self) -> "QdrantConfig":
        if self.location == "server":
            if not self.server.url:
                raise ValueError("qdrant.server.url is required when location=server")
        if self.location == "local":
            if not self.local.storage_dir:
                raise ValueError("qdrant.local.storage_dir is required when location=local")
        return self


class VectorStoreConfig(BaseModel):
    """Vector store configuration wrapper."""

    qdrant: QdrantConfig


class FixedChunkingConfig(BaseModel):
    """Fixed-size chunking parameters."""

    max_chars: int = Field(..., description="Maximum characters per chunk")
    overlap_chars: int = Field(..., description="Overlap between windows")


class HierarchyChunkingConfig(BaseModel):
    """Hierarchy-aware chunking parameters."""

    level: HierarchyLevel = Field(..., description="Target hierarchy level")
    max_chars_per_chunk: int = Field(..., description="Maximum characters per chunk")

    @field_validator("level", mode="before")
    @classmethod
    def _coerce_level(cls, v: object) -> HierarchyLevel:
        return _coerce_config_enum(HierarchyLevel, v)


class ChunkingConfig(BaseModel):
    """Chunking strategy configuration."""

    strategy: ChunkingStrategy = Field(..., description="Chunking strategy")
    fixed: FixedChunkingConfig
    hierarchy: HierarchyChunkingConfig

    @field_validator("strategy", mode="before")
    @classmethod
    def _coerce_strategy(cls, v: object) -> ChunkingStrategy:
        return _coerce_config_enum(ChunkingStrategy, v)


class DenseEmbeddingConfig(BaseModel):
    """Dense embedding parameters."""

    provider: Literal["openai"] = Field("openai", description="Dense embedding provider")
    model: str = Field(..., description="Dense embedding model name")
    batch_size: int = Field(32, description="Batch size for embedding requests")


class SparseEmbeddingConfig(BaseModel):
    """Sparse embedding parameters."""

    enabled: bool = Field(False, description="Whether sparse embedding is enabled")
    model: str = Field(..., description="Sparse embedding model name")
    batch_size: int = Field(16, description="Batch size for sparse embeddings")


class EmbeddingConfig(BaseModel):
    """Embedding configuration wrapper."""

    dense: DenseEmbeddingConfig
    sparse: SparseEmbeddingConfig


class DenseRetrieverConfig(BaseModel):
    """Dense retriever parameters."""

    top_k: int = Field(50, description="Top-K for dense search")


class SparseRetrieverConfig(BaseModel):
    """Sparse retriever parameters."""

    top_k: int = Field(100, description="Top-K for sparse search")


class HybridRetrieverConfig(BaseModel):
    """Hybrid retriever parameters."""

    combine: HybridCombineMethod = Field(HybridCombineMethod.RRF, description="Fusion method")
    top_k: int = Field(20, description="Top-K after fusion")
    linear_weights: dict[str, float] = Field(
        default_factory=lambda: {"dense": 0.7, "sparse": 0.3},
        description="Weights for linear fusion",
    )

    @field_validator("combine", mode="before")
    @classmethod
    def _coerce_combine(cls, v: object) -> HybridCombineMethod:
        return _coerce_config_enum(HybridCombineMethod, v)


class RerankConfig(BaseModel):
    """Reranker parameters."""

    enabled: bool = Field(False, description="Whether reranking is applied")
    mode: Literal["cross_encoder", "none"] = Field("cross_encoder", description="Rerank strategy")
    cross_encoder: dict[str, Any] = Field(
        default_factory=lambda: {"model": "BAAI/bge-reranker-v2-m3"},
        description="Cross-encoder reranker settings",
    )
    top_k: int = Field(5, description="Top-K after reranking")


class HydeConfig(BaseModel):
    """HyDE configuration."""

    enabled: bool = Field(False, description="Whether HyDE is used")
    prompt_template: str = Field("hyde_default", description="Prompt template id")


class MultiQueryConfig(BaseModel):
    """Multi-query configuration."""

    enabled: bool = Field(False, description="Whether multi-query is used")
    num_queries: int = Field(3, description="Number of reformulations")


class ContextCompressionConfig(BaseModel):
    """Context compression configuration."""

    enabled: bool = Field(False, description="Whether to compress retrieved context")
    target_tokens: int = Field(1500, description="Target token budget")


class ExtraRetrieverConfig(BaseModel):
    """Optional retriever extras."""

    multi_query: MultiQueryConfig
    context_compression: ContextCompressionConfig


class RetrieverConfig(BaseModel):
    """Retriever configuration wrapper."""

    dense: DenseRetrieverConfig
    sparse: SparseRetrieverConfig
    hybrid: HybridRetrieverConfig
    rerank: RerankConfig
    hyde: HydeConfig
    extra: ExtraRetrieverConfig


class OpenAIConfig(BaseModel):
    """OpenAI provider settings."""

    base_url: str = Field(..., description="OpenAI API base URL")
    api_key_env: str = Field(..., description="Environment variable containing API key")
    rag_mode: RagMode = Field(RagMode.PLAIN, description="RAG mode for Responses API")

    @field_validator("rag_mode", mode="before")
    @classmethod
    def _coerce_rag_mode(cls, v: object) -> RagMode:
        return _coerce_config_enum(RagMode, v)


class LmstudioConfig(BaseModel):
    """LM Studio OpenAI-compatible settings."""

    base_url: str = Field(..., description="LM Studio base URL")
    api_key_env: str = Field(..., description="Environment variable for token")


class LlmConfig(BaseModel):
    """LLM configuration wrapper."""

    provider: Literal["openai", "lmstudio"]
    model: str
    openai: OpenAIConfig
    lmstudio: LmstudioConfig


def _coerce_config_enum(enum_cls: type, value: object):
    """Coerce string or enum value to the given ConfigOption Enum."""

    if isinstance(value, enum_cls):
        return value
    if isinstance(value, str):
        for member in enum_cls:  # type: ignore[attr-defined]
            # member.value is ConfigOption; member.value.value is the string in config
            if getattr(member.value, "value", None) == value or member.name == value:
                return member
    raise ValueError(f"Invalid value {value!r} for enum {enum_cls.__name__}")


class EvalConfig(BaseModel):
    """Evaluation mode configuration."""

    split: Literal["validation", "test", "all"] = Field("test", description="Dataset split")
    max_examples: Optional[int] = Field(None, description="Maximum examples to evaluate")
    metrics: list[str] = Field(default_factory=lambda: ["accuracy"], description="Metric list")
    output_dir: Path = Field(..., description="Eval output directory")


class ServeConfig(BaseModel):
    """Serve mode configuration."""

    host: str = Field("0.0.0.0", description="Host for server")
    port: int = Field(8000, description="Port for server")


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: LogLevel = Field(LogLevel.INFO, description="Logging level")

    @field_validator("level", mode="before")
    @classmethod
    def _coerce_log_level(cls, v: object) -> LogLevel:
        return _coerce_config_enum(LogLevel, v)


class AppConfig(BaseModel):
    """Full application configuration."""

    experiment: ExperimentConfig
    data: DataConfig
    vector_store: VectorStoreConfig
    chunking: ChunkingConfig
    embedding: EmbeddingConfig
    retriever: RetrieverConfig
    llm: LlmConfig
    eval: EvalConfig
    serve: ServeConfig
    logging: LoggingConfig


__all__ = ["AppConfig"]
