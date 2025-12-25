"""OpenAI client helpers."""
from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path
import httpx


def _valid_path(path: str | None) -> str | None:
    """Return the path if it exists on disk.

    Args:
        path: Candidate filesystem path.

    Returns:
        Path string if it exists, otherwise None.
    """
    if not path:
        return None
    return path if Path(path).exists() else None


def _resolve_verify_path() -> str | None:
    """Resolve a usable CA bundle path.

    Returns:
        Path to a CA bundle file or directory, or None if not available.
    """
    cert_file = _valid_path(os.getenv("SSL_CERT_FILE"))
    if cert_file:
        return cert_file

    cert_dir = _valid_path(os.getenv("SSL_CERT_DIR"))
    if cert_dir:
        return cert_dir

    try:
        import certifi
    except Exception:
        return None

    return certifi.where()


@lru_cache(maxsize=1)
def _get_http_client() -> httpx.Client | None:
    """Create a shared HTTP client with a stable SSL configuration.

    Returns:
        httpx.Client if a verify path is available, otherwise None.
    """
    verify_path = _resolve_verify_path()
    if not verify_path:
        return None
    return httpx.Client(verify=verify_path)


def create_openai_client(api_key: str, base_url: str) -> "OpenAI":
    """Create an OpenAI client with a robust TLS configuration.

    Args:
        api_key: OpenAI API key.
        base_url: OpenAI API base URL.

    Returns:
        Configured OpenAI client.
    """
    from openai import OpenAI

    http_client = _get_http_client()
    if http_client is None:
        return OpenAI(api_key=api_key, base_url=base_url)
    return OpenAI(api_key=api_key, base_url=base_url, http_client=http_client)


__all__ = ["create_openai_client"]
