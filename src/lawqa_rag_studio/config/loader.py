"""Config loader utilities."""
from __future__ import annotations

import yaml
from pathlib import Path
from typing import Any
import re

from lawqa_rag_studio.config.schema import AppConfig

PLACEHOLDER_PATTERN = re.compile(r"\$\{([^}]+)\}")


def load_config(path: Path) -> AppConfig:
    """Load application config from YAML.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        Parsed `AppConfig` object.
    """
    with path.open("r", encoding="utf-8") as fp:
        data: dict[str, Any] = yaml.safe_load(fp)
    resolved = _resolve_placeholders(data)
    return AppConfig(**resolved)


def _resolve_placeholders(data: dict[str, Any]) -> dict[str, Any]:
    """Resolve ${...} placeholders using loaded config values."""

    def get_by_path(root: dict[str, Any], dotted: str) -> Any:
        cur: Any = root
        for part in dotted.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                return None
        return cur

    def resolve(obj: Any, root: dict[str, Any]) -> Any:
        if isinstance(obj, dict):
            return {k: resolve(v, root) for k, v in obj.items()}
        if isinstance(obj, list):
            return [resolve(v, root) for v in obj]
        if isinstance(obj, str):
            def repl(match: re.Match[str]) -> str:
                key = match.group(1)
                val = get_by_path(root, key)
                return str(val) if val is not None else match.group(0)

            return PLACEHOLDER_PATTERN.sub(repl, obj)
        return obj

    return resolve(data, data)  # type: ignore[arg-type]


__all__ = ["load_config"]
