"""Logging setup utilities."""
from __future__ import annotations

import logging
from pathlib import Path

from lawqa_rag_studio.config.schema import LoggingConfig
from lawqa_rag_studio.logging.formatters import default_formatter


def configure_logging(cfg: LoggingConfig) -> None:
    """Configure root logger.

    Args:
        cfg: Logging configuration.
    """
    fmt = default_formatter()
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)

    log_dir = Path("./logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_dir / "run.log", encoding="utf-8")
    file_handler.setFormatter(fmt)

    root = logging.getLogger()
    level_value = (
        cfg.level.value.value  # ConfigOption.value -> str
        if hasattr(cfg.level, "value") and hasattr(cfg.level.value, "value")
        else cfg.level.value
        if hasattr(cfg.level, "value")
        else str(cfg.level)
    )
    root.setLevel(level_value)
    root.handlers = [stream_handler, file_handler]


__all__ = ["configure_logging"]
