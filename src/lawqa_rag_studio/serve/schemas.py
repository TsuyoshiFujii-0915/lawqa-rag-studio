"""API schemas for serve mode."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class ChatRequest(BaseModel):
    """Schema for chat requests."""

    message: str
    session_id: str | None = None
    history: list[dict[str, Any]] | None = None


class ChatResponse(BaseModel):
    """Schema for chat responses."""

    answer: str
    chunks: list[dict[str, Any]]
    config_hash: str
    session_id: str


class HealthResponse(BaseModel):
    """Schema for health check."""

    status: str
    model: str
    collection: str


class SessionResetResponse(BaseModel):
    """Schema for session reset responses."""

    session_id: str


__all__ = ["ChatRequest", "ChatResponse", "HealthResponse", "SessionResetResponse"]
