"""Logging formatters."""
from __future__ import annotations

import logging


def default_formatter() -> logging.Formatter:
    """Create default logging formatter.

    Returns:
        Configured logging formatter.
    """
    return logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")


__all__ = ["default_formatter"]
