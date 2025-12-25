"""Interface contract tests for RAG components."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping

import pytest

from lawqa_rag_studio.config.loader import load_config
from lawqa_rag_studio.config.schema import AppConfig
from lawqa_rag_studio.config.constants import HybridCombineMethod
from lawqa_rag_studio.chunking.base import Chunk
from lawqa_rag_studio.retrieval.dense import dense_search
from lawqa_rag_studio.retrieval.sparse import sparse_search
from lawqa_rag_studio.retrieval.rerank import rerank_chunks
from lawqa_rag_studio.retrieval.pipeline import retrieve
from lawqa_rag_studio.rag.pipeline import answer_query


class DummyStore:
    """Minimal VectorStore stub."""

    def __init__(self, dense_chunk: Chunk | None = None, sparse_chunk: Chunk | None = None) -> None:
        self._dense_chunk = dense_chunk or Chunk(
            id="dense-1", text="text d", metadata={"score": 0.9, "source": "dense"}
        )
        self._sparse_chunk = sparse_chunk or Chunk(
            id="sparse-1", text="text s", metadata={"score": 0.8, "source": "sparse"}
        )

    def search(self, query_vector: list[float], top_k: int) -> list[Chunk]:
        assert isinstance(query_vector, list)
        return [self._dense_chunk][:top_k]

    def search_sparse(self, query_vector: Mapping[int, float], top_k: int) -> list[Chunk]:
        assert isinstance(query_vector, Mapping)
        return [self._sparse_chunk][:top_k]


def _mini_config(tmp_path: Path, *, sparse_enabled: bool = True) -> AppConfig:
    """Create minimal AppConfig for tests."""

    yaml_text = f"""
experiment:
  name: unit
  seed: 1
  output_dir: {tmp_path}/out
data:
  egov:
    enabled: false
    xml_dir: {tmp_path}/xml
    law_tree_cache: {tmp_path}/cache.json
    law_list_path: {tmp_path}/law_list.json
    api_base_url: https://example.com
  lawqa_jp:
    path: {tmp_path}/lawqa.json
vector_store:
  qdrant:
    collection_name: law
    location: in-memory
    local:
      storage_dir: {tmp_path}/qdrant
    server:
      url: http://localhost:6333
      api_key: null
chunking:
  strategy: fixed
  fixed:
    max_chars: 100
    overlap_chars: 0
  hierarchy:
    level: article
    max_chars_per_chunk: 200
embedding:
  dense:
    provider: openai
    model: text-embedding-3-small
    batch_size: 8
  sparse:
    enabled: {str(sparse_enabled).lower()}
    model: bizreach-inc/light-splade-japanese-56M
    batch_size: 4
retriever:
  dense:
    top_k: 5
  sparse:
    top_k: 5
  hybrid:
    combine: linear
    top_k: 3
    linear_weights:
      dense: 0.6
      sparse: 0.4
  rerank:
    enabled: false
    mode: cross_encoder
    cross_encoder:
      model: BAAI/bge-reranker-v2-m3
    top_k: 2
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
  max_examples: 1
  metrics: [accuracy]
  output_dir: {tmp_path}/out/eval
serve:
  host: 0.0.0.0
  port: 8000
logging:
  level: INFO
"""
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml_text, encoding="utf-8")
    return load_config(cfg_path)


def test_dense_search_returns_chunk_shape(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """dense_search should return Chunk list with required keys."""

    def fake_embed(model: str, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        assert model == "text-embedding-3-small"
        return [[0.1, 0.2, 0.3] for _ in texts]

    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    monkeypatch.setattr("lawqa_rag_studio.retrieval.dense.embed_texts", fake_embed)

    store = DummyStore()
    chunks = dense_search(store, "text-embedding-3-small", "質問", top_k=1)

    assert len(chunks) == 1
    chunk = chunks[0]
    assert set(chunk.keys()) == {"id", "text", "metadata"}
    assert isinstance(chunk["metadata"].get("score"), float)


def test_sparse_search_returns_chunk_shape(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """sparse_search should return Chunk list with required keys."""

    def fake_sparse_embed(model: str, texts: list[str], batch_size: int = 16) -> list[dict[int, float]]:
        assert model == "bizreach-inc/light-splade-japanese-56M"
        return [{1: 0.5, 3: 0.2} for _ in texts]

    monkeypatch.setattr("lawqa_rag_studio.retrieval.sparse.embed_texts_splade", fake_sparse_embed)
    store = DummyStore()
    chunks = sparse_search(store, "bizreach-inc/light-splade-japanese-56M", "質問", top_k=1)

    assert len(chunks) == 1
    chunk = chunks[0]
    assert set(chunk.keys()) == {"id", "text", "metadata"}
    assert isinstance(chunk["metadata"].get("score"), float)


def test_rerank_chunks_adds_scores(monkeypatch: pytest.MonkeyPatch) -> None:
    """rerank_chunks should attach rerank_score and truncate to top_k."""

    # Fake torch module minimal API
    class _NoGrad:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class FakeTorch:
        def no_grad(self):
            return _NoGrad()

    monkeypatch.setitem(sys.modules, "torch", FakeTorch())

    class FakeBatch(dict):
        def __init__(self, n: int):
            super().__init__({"input_ids": [0] * n, "attention_mask": [1] * n})

        def to(self, device: str) -> "FakeBatch":
            return self

    class FakeTokenizer:
        def __call__(self, queries, texts, padding=True, truncation=True, max_length=512, return_tensors="pt"):
            return FakeBatch(len(texts))

    class FakeTensor(list):
        def squeeze(self, dim: int = -1):
            return self

        def detach(self):
            return self

        def cpu(self):
            return self

        def tolist(self):
            return list(self)

    class FakeOutput:
        def __init__(self, scores: list[float]):
            self.logits = FakeTensor(scores)

    class FakeModel:
        def __call__(self, **encoded):
            n = len(encoded.get("input_ids", [])) or 1
            # ascending scores just for determinism
            scores = [0.1 * (i + 1) for i in range(n)]
            return FakeOutput(scores)

    def fake_get_reranker(model_name: str):
        return FakeModel(), FakeTokenizer(), "cpu"

    monkeypatch.setattr("lawqa_rag_studio.retrieval.rerank._get_reranker", fake_get_reranker)

    candidates: list[Chunk] = [
        Chunk(id="a", text="t1", metadata={"score": 0.3}),
        Chunk(id="b", text="t2", metadata={"score": 0.2}),
        Chunk(id="c", text="t3", metadata={"score": 0.1}),
    ]
    ranked = rerank_chunks("dummy", "query", candidates, top_k=2)

    assert len(ranked) == 2
    assert ranked[0]["metadata"]["rerank_score"] >= ranked[1]["metadata"]["rerank_score"]
    assert all("rerank_score" in c["metadata"] for c in ranked)


def test_retrieve_pipeline_respects_top_k(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """retrieve should fuse dense and sparse results into top_k chunks."""

    cfg = _mini_config(tmp_path, sparse_enabled=True)
    cfg.retriever.hybrid.combine = HybridCombineMethod.LINEAR
    cfg.retriever.hybrid.top_k = 2

    dense_chunks = [
        Chunk(id="d1", text="dense1", metadata={"score": 0.9}),
        Chunk(id="d2", text="dense2", metadata={"score": 0.5}),
    ]
    sparse_chunks = [
        Chunk(id="s1", text="sparse1", metadata={"score": 0.8}),
        Chunk(id="d1", text="dup", metadata={"score": 0.4}),
    ]

    monkeypatch.setattr("lawqa_rag_studio.retrieval.pipeline.dense_search", lambda store, model, q, k: dense_chunks)
    monkeypatch.setattr("lawqa_rag_studio.retrieval.pipeline.sparse_search", lambda store, model, q, k: sparse_chunks)

    store = DummyStore()
    fused = retrieve("質問", cfg, store)

    assert len(fused) == 2
    assert fused[0]["id"] in {"d1", "s1"}
    assert all(set(c.keys()) == {"id", "text", "metadata"} for c in fused)


class DummyLlm:
    """Minimal LlmClient stub."""

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, query: str, context: list[Chunk]) -> Mapping[str, Any]:
        self.calls += 1
        return {
            "text": "answer",
            "raw_text": "answer",
            "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
        }


def test_answer_query_returns_rag_result(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """answer_query should produce RagResult shape with used_chunks and usage."""

    cfg = _mini_config(tmp_path, sparse_enabled=False)
    # Skip real retrieval: return a single stub chunk.
    stub_chunk = Chunk(id="c1", text="ctx", metadata={"score": 1.0})
    monkeypatch.setattr("lawqa_rag_studio.rag.pipeline.retrieve", lambda q, c, s: [stub_chunk])

    llm = DummyLlm()
    store = DummyStore()
    result = answer_query("質問", cfg, store, llm, rag_mode="plain")

    assert set(result.keys()) == {"answer", "llm_raw_answer", "used_chunks", "llm_calls", "retrieval_info", "usage"}
    assert result["answer"] == "answer"
    assert result["llm_raw_answer"] == "answer"
    assert result["used_chunks"][0]["id"] == "c1"
    assert result["usage"]["prompt_tokens"] == 1
    assert llm.calls == 1
