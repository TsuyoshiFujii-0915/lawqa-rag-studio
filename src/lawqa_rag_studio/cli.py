"""Command line interface for LawQA-RAG-Studio."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
import dotenv
import logging

from lawqa_rag_studio.config.loader import load_config
from lawqa_rag_studio.data.egov_downloader import download_egov_xml
from lawqa_rag_studio.ingest.pipeline import ingest_all
from lawqa_rag_studio.vectorstore.qdrant_client import QdrantStore
from lawqa_rag_studio.eval.runner import run_eval
from lawqa_rag_studio.serve.api import run_server
from lawqa_rag_studio.logging import configure_logging

app = typer.Typer(help="LawQA-RAG-Studio CLI")

# Ensure .env is loaded with highest priority
dotenv.load_dotenv(override=True)


@app.command()
def eval(
    config: Path = typer.Option(..., help="Path to YAML config."),
    recreate_index: bool = typer.Option(False, help="Recreate vector store before eval."),
) -> None:
    """Run evaluation pipeline.

    Args:
        config: Path to configuration YAML file.
        recreate_index: Whether to rebuild vector store before evaluation.
    """
    cfg = load_config(config)
    configure_logging(cfg.logging)
    logging.getLogger(__name__).info("Starting eval mode")
    result = run_eval(cfg, recreate_index=recreate_index, config_path=str(config))
    typer.echo(json.dumps(result.summary, ensure_ascii=False, indent=2))


@app.command()
def serve(
    config: Path = typer.Option(..., help="Path to YAML config."),
    host: Optional[str] = typer.Option(None, help="Override host."),
    port: Optional[int] = typer.Option(None, help="Override port."),
    recreate_index: bool = typer.Option(False, help="Recreate vector store before serving."),
) -> None:
    """Start HTTP API server.

    Args:
        config: Path to configuration YAML file.
        host: Optional host override.
        port: Optional port override.
        recreate_index: Whether to rebuild vector store before serving.
    """
    cfg = load_config(config)
    configure_logging(cfg.logging)
    logging.getLogger(__name__).info("Starting serve mode")
    run_server(cfg, host=host, port=port, recreate_index=recreate_index)


@app.command()
def ingest(
    config: Path = typer.Option(..., help="Path to YAML config."),
    recreate: bool = typer.Option(False, help="Drop and rebuild collection before ingest."),
) -> None:
    """Ingest source data and build vector store.

    Args:
        config: Path to configuration YAML file.
        recreate: Whether to drop existing collection and rebuild.
    """
    cfg = load_config(config)
    configure_logging(cfg.logging)
    logging.getLogger(__name__).info("Starting ingest")
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
    count = ingest_all(cfg, store, recreate=recreate)
    typer.echo(f"Ingest completed. Chunks ingested: {count}")


@app.command()
def fetch_egov(
    dest: Path = typer.Option(..., help="Destination directory for e-Gov XML (typically data/egov/xml)"),
    url: str = typer.Option(
        "https://laws.e-gov.go.jp/bulkdownload?file_section=1&only_xml_flag=true",
        help="Download URL for bulk e-Gov XML zip.",
    ),
) -> None:
    """Download e-Gov XML corpus once.

    Args:
        dest: Destination directory to place XML files.
        url: Download URL for bulk XML.
    """

    downloaded = download_egov_xml(dest, url)
    typer.echo(f"Downloaded {len(downloaded)} files into {dest}")


def main() -> None:
    """Entrypoint for console_scripts."""
    app()


if __name__ == "__main__":
    main()
