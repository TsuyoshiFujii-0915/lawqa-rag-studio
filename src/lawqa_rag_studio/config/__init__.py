"""Configuration utilities for LawQA-RAG-Studio."""

from lawqa_rag_studio.config.loader import load_config
from lawqa_rag_studio.config.schema import AppConfig

__all__ = ["load_config", "AppConfig"]
