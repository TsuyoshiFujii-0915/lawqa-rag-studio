"""FastAPI application for LawQA-RAG-Studio."""
from __future__ import annotations

import logging
import os
import uuid
from typing import Any, Dict, List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from lawqa_rag_studio.config.schema import AppConfig
from lawqa_rag_studio.llm import (
    LlmClient,
    LmStudioOpenAIClient,
    OpenAIResponsesAgenticClient,
    OpenAIResponsesClient,
    SimpleLocalClient,
)
from lawqa_rag_studio.rag.pipeline import answer_query
from lawqa_rag_studio.serve.schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    SessionResetResponse,
)
from lawqa_rag_studio.vectorstore.qdrant_client import QdrantStore
from lawqa_rag_studio.ingest.pipeline import ingest_all


def create_app(cfg: AppConfig, recreate_index: bool = False) -> FastAPI:
    """Create FastAPI app wired with runtime dependencies.

    Args:
        cfg: Application configuration.

    Returns:
        Configured FastAPI application.
    """

    app = FastAPI(title="LawQA-RAG-Studio")
    cors_env = os.getenv("LAWQA_CORS_ORIGINS", "").strip()
    cors_origins = (
        [origin.strip() for origin in cors_env.split(",") if origin.strip()]
        if cors_env
        else [
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:4173",
            "http://127.0.0.1:4173",
        ]
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    sessions: Dict[str, List[dict[str, Any]]] = {}

    def _new_session() -> str:
        """Create and register a new session id."""

        sid = uuid.uuid4().hex
        sessions[sid] = []
        return sid

    store = QdrantStore(
        collection=cfg.vector_store.qdrant.collection_name,
        dense_model=cfg.embedding.dense.model,
        sparse_model=cfg.embedding.sparse.model if cfg.embedding.sparse.enabled else None,
        dense_dim=None,
        location=cfg.vector_store.qdrant.location,
        server_url=cfg.vector_store.qdrant.server.url if cfg.vector_store.qdrant.location == "server" else None,
        server_api_key=cfg.vector_store.qdrant.server.api_key if cfg.vector_store.qdrant.location == "server" else None,
        storage_dir=str(cfg.vector_store.qdrant.local.storage_dir) if cfg.vector_store.qdrant.location == "local" else None,
        dense_batch_size=cfg.embedding.dense.batch_size,
        sparse_batch_size=cfg.embedding.sparse.batch_size,
    )
    llm = _build_llm(cfg, store)

    collections = [c.name for c in store.client.get_collections().collections]  # type: ignore[attr-defined]
    if cfg.vector_store.qdrant.collection_name not in collections or recreate_index:
        ingest_all(cfg, store, recreate=True)

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            model=cfg.llm.model,
            collection=cfg.vector_store.qdrant.collection_name,
        )

    @app.post("/chat", response_model=ChatResponse)
    def chat(body: ChatRequest) -> ChatResponse:  # pragma: no cover - small wrapper
        rag_mode = cfg.llm.openai.rag_mode.value if hasattr(cfg.llm.openai.rag_mode, "value") else str(cfg.llm.openai.rag_mode)
        session_id = body.session_id or _new_session()
        if session_id not in sessions:
            sessions[session_id] = []
        result = answer_query(body.message, cfg, store, llm, rag_mode=rag_mode)
        sessions[session_id].extend(
            [
                {"role": "user", "content": body.message},
                {"role": "assistant", "content": result["answer"]},
            ]
        )
        return ChatResponse(
            answer=result["answer"],
            chunks=result["used_chunks"],
            config_hash=cfg.experiment.name,
            session_id=session_id,
        )

    @app.post("/session/reset", response_model=SessionResetResponse)
    def reset_session() -> SessionResetResponse:  # pragma: no cover - simple state reset
        """Reset conversation session and return new session id."""

        new_id = _new_session()
        return SessionResetResponse(session_id=new_id)

    return app


def _build_llm(cfg: AppConfig, store: VectorStoreClient) -> LlmClient:
    """Instantiate LLM client from config.

    Args:
        cfg: Application configuration.
        store: Vector store client instance.

    Returns:
        Configured LLM client.
    """

    if cfg.llm.provider == "openai":
        rag_mode = cfg.llm.openai.rag_mode.value if hasattr(cfg.llm.openai.rag_mode, "value") else str(cfg.llm.openai.rag_mode)
        api_key = os.getenv(cfg.llm.openai.api_key_env, "")
        logging.getLogger(__name__).info(
            "LLM setup provider=%s model=%s rag_mode=%s base_url=%s",
            cfg.llm.provider,
            cfg.llm.model,
            rag_mode,
            cfg.llm.openai.base_url,
        )
        if rag_mode == "agentic":
            return OpenAIResponsesAgenticClient(
                base_url=cfg.llm.openai.base_url,
                model=cfg.llm.model,
                api_key=api_key,
                cfg=cfg,
                store=store,
            )
        return OpenAIResponsesClient(
            base_url=cfg.llm.openai.base_url,
            model=cfg.llm.model,
            api_key=api_key,
        )
    if cfg.llm.provider == "lmstudio":
        logging.getLogger(__name__).info(
            "LLM setup provider=%s model=%s base_url=%s",
            cfg.llm.provider,
            cfg.llm.model,
            cfg.llm.lmstudio.base_url,
        )
        return LmStudioOpenAIClient(
            base_url=cfg.llm.lmstudio.base_url,
            model=cfg.llm.model,
            api_key=os.getenv(cfg.llm.lmstudio.api_key_env, ""),
        )
    logging.getLogger(__name__).info("LLM setup provider=%s model=%s (simple local)", cfg.llm.provider, cfg.llm.model)
    return SimpleLocalClient()


def run_server(cfg: AppConfig, host: str | None = None, port: int | None = None, recreate_index: bool = False) -> None:
    """Run the FastAPI server using uvicorn.

    Args:
        cfg: Application configuration.
        host: Optional host override.
        port: Optional port override.
    """
    import uvicorn

    app = create_app(cfg, recreate_index=recreate_index)
    uvicorn.run(app, host=host or cfg.serve.host, port=port or cfg.serve.port)


__all__ = ["create_app", "run_server"]
