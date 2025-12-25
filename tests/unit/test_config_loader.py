"""Tests for configuration loading and enum coercion."""
from __future__ import annotations

from pathlib import Path

from lawqa_rag_studio.config.constants import (
    ChunkingStrategy,
    HierarchyLevel,
    HybridCombineMethod,
    LogLevel,
    RagMode,
)
from lawqa_rag_studio.config.loader import load_config


def test_load_config_coerces_enums_and_paths(tmp_path: Path) -> None:
    """Ensure config strings are coerced into enums and paths are resolved."""

    config_path = tmp_path / "config.yaml"
    lawqa_path = tmp_path / "lawqa.json"
    lawqa_path.write_text("[]", encoding="utf-8")
    (tmp_path / "data" / "egov" / "xml").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data" / "egov" / "cache").mkdir(parents=True, exist_ok=True)

    yaml_text = f"""
experiment:
  name: unit-test
  seed: 123
  output_dir: {tmp_path}/outputs
data:
  egov:
    enabled: true
    xml_dir: {tmp_path}/data/egov/xml
    law_tree_cache: {tmp_path}/data/egov/cache/law_tree.jsonl
    law_list_path: {tmp_path}/data/egov/law_list.json
    api_base_url: https://laws.e-gov.go.jp/api/2/law_file/xml
  lawqa_jp:
    path: {lawqa_path}
vector_store:
  qdrant:
    collection_name: law_ja
    location: local
    local:
      storage_dir: {tmp_path}/data/qdrant
    server:
      url: http://localhost:6333
      api_key: null
chunking:
  strategy: fixed
  fixed:
    max_chars: 1200
    overlap_chars: 200
  hierarchy:
    level: section
    max_chars_per_chunk: 2000
embedding:
  dense:
    provider: openai
    model: text-embedding-3-small
    batch_size: 16
  sparse:
    enabled: false
    model: bizreach-inc/light-splade-japanese-56M
    batch_size: 8
retriever:
  dense:
    top_k: 10
  sparse:
    top_k: 20
  hybrid:
    combine: rrf
    top_k: 5
    linear_weights:
      dense: 0.6
      sparse: 0.4
  rerank:
    enabled: false
    mode: cross_encoder
    cross_encoder:
      model: BAAI/bge-reranker-v2-m3
    top_k: 3
  hyde:
    enabled: false
    prompt_template: hyde_default
  extra:
    multi_query:
      enabled: false
      num_queries: 2
    context_compression:
      enabled: false
      target_tokens: 500
llm:
  provider: openai
  model: gpt-5-mini
  openai:
    base_url: https://api.openai.com/v1
    api_key_env: OPENAI_API_KEY
    rag_mode: plain
  lmstudio:
    base_url: http://localhost:1234/v1
    api_key_env: LMSTUDIO_API_KEY
eval:
  split: test
  max_examples: 5
  metrics: [accuracy, macro_f1]
  output_dir: ${{experiment.output_dir}}/eval
serve:
  host: 0.0.0.0
  port: 9000
logging:
  level: INFO
"""
    config_path.write_text(yaml_text, encoding="utf-8")

    cfg = load_config(config_path)

    assert isinstance(cfg.chunking.strategy, ChunkingStrategy)
    assert cfg.chunking.strategy is ChunkingStrategy.FIXED
    assert isinstance(cfg.chunking.hierarchy.level, HierarchyLevel)
    assert cfg.chunking.hierarchy.level is HierarchyLevel.SECTION
    assert isinstance(cfg.retriever.hybrid.combine, HybridCombineMethod)
    assert cfg.retriever.hybrid.combine is HybridCombineMethod.RRF
    assert isinstance(cfg.logging.level, LogLevel)
    assert cfg.logging.level is LogLevel.INFO
    assert isinstance(cfg.llm.openai.rag_mode, RagMode)
    assert cfg.llm.openai.rag_mode is RagMode.PLAIN
    assert cfg.eval.output_dir == Path(tmp_path / "outputs" / "eval")
    assert cfg.vector_store.qdrant.local.storage_dir == Path(tmp_path / "data" / "qdrant")
