"""Tests for session reset and chat session_id handling."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest

from lawqa_rag_studio.config.loader import load_config
from lawqa_rag_studio.serve import api as api_mod


def _mini_config(tmp_path: Path):
    """Build minimal config YAML and load it."""

    cfg_text = f"""
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
    enabled: false
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
    cfg_path.write_text(cfg_text, encoding="utf-8")
    return load_config(cfg_path)


class DummyStore:
    """Minimal stand-in for QdrantStore with collections metadata."""

    def __init__(self, collection: str) -> None:
        self.collection = collection
        self.client = SimpleNamespace(
            get_collections=lambda: SimpleNamespace(collections=[SimpleNamespace(name=collection)])
        )


def test_session_reset_and_chat_session_id(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Ensure /session/reset returns new id and chat echoes session_id."""

    cfg = _mini_config(tmp_path)

    # Bypass heavy dependencies
    monkeypatch.setattr(api_mod, "QdrantStore", lambda *args, **kwargs: DummyStore(collection=cfg.vector_store.qdrant.collection_name))
    monkeypatch.setattr(api_mod, "ingest_all", lambda *args, **kwargs: 0)
    monkeypatch.setattr(api_mod, "_build_llm", lambda cfg, store: object())
    monkeypatch.setattr(
        api_mod,
        "answer_query",
        lambda message, cfg, store, llm, rag_mode="plain": {
            "answer": "ok",
            "llm_raw_answer": "ok",
            "used_chunks": [],
            "llm_calls": [],
            "retrieval_info": {},
            "usage": {},
        },
    )

    app = api_mod.create_app(cfg, recreate_index=False)
    client = TestClient(app)

    reset_resp = client.post("/session/reset")
    assert reset_resp.status_code == 200
    session_id = reset_resp.json()["session_id"]
    assert session_id

    chat_resp = client.post("/chat", json={"message": "hi", "session_id": session_id})
    assert chat_resp.status_code == 200
    data = chat_resp.json()
    assert data["session_id"] == session_id
    assert data["answer"] == "ok"
