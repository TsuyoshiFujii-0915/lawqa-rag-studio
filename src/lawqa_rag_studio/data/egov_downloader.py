"""Downloader for e-Gov XML corpus."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence
from urllib.parse import urlparse

import httpx
import logging

logger = logging.getLogger(__name__)

def download_egov_xml(output_dir: Path, url: str) -> list[Path]:
    """Download e-Gov XML archive (zip) and extract it.

    Args:
        output_dir: Destination directory for extracted XML files.
        url: Source URL of the bulk XML zip.

    Returns:
        List of extracted XML file paths.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = output_dir / "egov_bulk.zip"

    # Skip download if zip already exists
    if not zip_path.exists():
        with httpx.stream("GET", url, follow_redirects=True, timeout=None) as resp:
            resp.raise_for_status()
            with zip_path.open("wb") as f:
                for chunk in resp.iter_bytes():
                    f.write(chunk)

    # Extract
    import zipfile

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(output_dir)

    # Remove zip after extraction to save space
    try:
        zip_path.unlink()
    except Exception:
        pass

    # Collect XML paths
    xml_files = sorted(output_dir.glob("**/*.xml"))
    return xml_files


def list_downloaded_files(output_dir: Path) -> Iterable[Path]:
    """List already downloaded e-Gov XML files.

    Args:
        output_dir: Directory to scan.

    Returns:
        Iterable over file paths.
    """
    return output_dir.glob("**/*.xml")


def _extract_law_id(url: str) -> str:
    """Extract e-Gov law ID from law page URL.

    Args:
        url: e-Gov law page URL such as ``https://laws.e-gov.go.jp/law/323AC0000000025``.

    Returns:
        Law ID string usable for the API endpoint.
    """
    parsed = urlparse(url)
    if not parsed.path:
        raise ValueError(f"Invalid e-Gov law URL: {url}")
    law_id = parsed.path.rstrip("/").split("/")[-1]
    if not law_id:
        raise ValueError(f"Failed to extract law id from URL: {url}")
    return law_id


def load_law_ids_from_list(law_list_path: Path) -> list[str]:
    """Load e-Gov law IDs from ``law_list.json``.

    Args:
        law_list_path: Path to law_list.json.

    Returns:
        List of e-Gov law IDs.
    """
    with law_list_path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    if not isinstance(data, list):
        raise ValueError("law_list.json must be a JSON array.")

    urls: list[str] = []
    for entry in data:
        if isinstance(entry, str):
            urls.append(entry)
        elif isinstance(entry, dict) and "url" in entry and isinstance(entry["url"], str):
            urls.append(entry["url"])
        else:
            raise ValueError(f"Unsupported law_list entry format: {entry!r}")

    egov_urls = [url for url in urls if url.startswith("https://laws.e-gov.go.jp/law/")]
    if not egov_urls:
        raise ValueError("No e-Gov URLs found in law_list.json.")
    return [_extract_law_id(url) for url in egov_urls]


def download_laws_via_api(
    law_ids: Sequence[str],
    output_dir: Path,
    api_base_url: str = "https://laws.e-gov.go.jp/api/2/law_file/xml",
) -> list[Path]:
    """Download e-Gov XML via official API for specified law IDs.

    Args:
        law_ids: Sequence of e-Gov law IDs.
        output_dir: Directory to store downloaded XML files.
        api_base_url: Base URL for the e-Gov XML API.

    Returns:
        List of downloaded XML file paths (including those already present).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []
    for law_id in law_ids:
        dest = output_dir / f"{law_id}.xml"
        if dest.exists() and dest.stat().st_size > 0:
            downloaded.append(dest)
            continue

        url = f"{api_base_url}/{law_id}"
        logger.info("Downloading law %s", law_id)
        resp = httpx.get(url, follow_redirects=True, timeout=60.0)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        downloaded.append(dest)
    logger.info("Downloaded %d laws into %s", len(downloaded), output_dir)
    return downloaded


__all__ = [
    "download_egov_xml",
    "list_downloaded_files",
    "load_law_ids_from_list",
    "download_laws_via_api",
]
