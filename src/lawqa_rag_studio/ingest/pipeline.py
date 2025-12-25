"""Ingestion pipeline utilities."""
from __future__ import annotations

import logging

from lawqa_rag_studio.chunking.base import Chunker
from lawqa_rag_studio.chunking.fixed import FixedChunker
from lawqa_rag_studio.chunking.hierarchy import HierarchyChunker
from lawqa_rag_studio.config.constants import ChunkingStrategy
from lawqa_rag_studio.config.schema import AppConfig
from lawqa_rag_studio.data.egov_downloader import (
    download_laws_via_api,
    list_downloaded_files,
    load_law_ids_from_list,
)
from lawqa_rag_studio.data.egov_parser import parse_multiple
from lawqa_rag_studio.vectorstore.qdrant_client import QdrantStore

logger = logging.getLogger(__name__)


def build_chunker(cfg: AppConfig) -> Chunker:
    """Create chunker based on configuration.

    Args:
        cfg: Application configuration.

    Returns:
        Configured chunker instance.
    """

    if cfg.chunking.strategy == ChunkingStrategy.FIXED:
        return FixedChunker(cfg.chunking.fixed.max_chars, cfg.chunking.fixed.overlap_chars)
    return HierarchyChunker(cfg.chunking.hierarchy.level, cfg.chunking.hierarchy.max_chars_per_chunk)


def ingest_all(cfg: AppConfig, store: QdrantStore, recreate: bool = False) -> int:
    """Ingest e-Gov XML into the vector store.

    Args:
        cfg: Application configuration.
        store: Qdrant store instance.
        recreate: Whether to rebuild the collection (drops existing data).

    Returns:
        Number of chunks ingested.
    """

    # Drop collection if recreate is requested
    existing = [c.name for c in store.client.get_collections().collections]  # type: ignore[attr-defined]
    if recreate and cfg.vector_store.qdrant.collection_name in existing:
        store.client.delete_collection(collection_name=cfg.vector_store.qdrant.collection_name)  # type: ignore[attr-defined]
        existing.remove(cfg.vector_store.qdrant.collection_name)
        logger.info("Deleted existing collection %s", cfg.vector_store.qdrant.collection_name)

    if cfg.vector_store.qdrant.collection_name in existing and not recreate:
        # Already ingested
        logger.info("Collection %s already exists. Skipping ingest.", cfg.vector_store.qdrant.collection_name)
        return 0

    files = list(list_downloaded_files(cfg.data.egov.xml_dir))
    if not files:
        if not cfg.data.egov.enabled:
            raise RuntimeError("e-Gov ingestion requested while data.egov.enabled is false.")
        law_ids = load_law_ids_from_list(cfg.data.egov.law_list_path)
        logger.info("No XML found. Downloading %d laws via API.", len(law_ids))
        downloaded = download_laws_via_api(
            law_ids, cfg.data.egov.xml_dir, cfg.data.egov.api_base_url
        )
        if not downloaded:
            raise RuntimeError(
                f"Failed to download e-Gov XML via API into {cfg.data.egov.xml_dir}."
            )
        files = list(list_downloaded_files(cfg.data.egov.xml_dir))
        logger.info("Downloaded %d XML files.", len(downloaded))
    if not files:
        raise RuntimeError(
            f"No e-Gov XML files found in {cfg.data.egov.xml_dir} after API download."
        )

    trees = parse_multiple(files)
    logger.info("Parsed %d law XML files.", len(trees))
    chunker = build_chunker(cfg)
    total = 0
    for tree in trees:
        chunks = chunker.create_chunks(tree)
        store.upsert_chunks(chunks)
        total += len(chunks)
    logger.info("Ingest finished. Total chunks: %d", total)
    return total
