"""Evaluation runner."""
from __future__ import annotations

import os
import json
import yaml
import enum
from dataclasses import dataclass
from typing import Any, Mapping, Tuple
from pathlib import Path
import time
import logging

from lawqa_rag_studio.config.schema import AppConfig
from lawqa_rag_studio.data.lawqa_loader import LawQaExample, load_lawqa
from lawqa_rag_studio.eval import metrics as metrics_mod
from lawqa_rag_studio.eval.reporters import save_details, save_metrics
from lawqa_rag_studio.ingest.pipeline import ingest_all
from lawqa_rag_studio.llm import (
    LlmClient,
    LmStudioOpenAIClient,
    OpenAIResponsesAgenticClient,
    OpenAIResponsesClient,
    SimpleLocalClient,
)
from lawqa_rag_studio.rag.pipeline import answer_query
from lawqa_rag_studio.vectorstore.qdrant_client import QdrantStore

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """Container for evaluation results."""

    summary: Mapping[str, Any]
    details: list[Mapping[str, Any]]


def run_eval(cfg: AppConfig, recreate_index: bool = False, config_path: str | None = None) -> EvalResult:
    """Run evaluation mode.

    Args:
        cfg: Application configuration.

    Returns:
        Evaluation result containing summary metrics and per-example details.
    """
    cfg.eval.output_dir.mkdir(parents=True, exist_ok=True)
    dataset_all: list[LawQaExample] = load_lawqa(cfg.data.lawqa_jp.path, exclude_pdf_only=False)
    dataset: list[LawQaExample] = load_lawqa(cfg.data.lawqa_jp.path, exclude_pdf_only=True)
    pdf_only_count = len(dataset_all) - len(dataset)
    if not dataset:
        raise RuntimeError("No evaluation examples remain after excluding PDF-only references.")
    if cfg.eval.max_examples is not None:
        dataset = dataset[: cfg.eval.max_examples]
        logger.info(
            "Applying eval.max_examples=%d -> evaluating %d examples (PDF-only skipped=%d, available=%d)",
            cfg.eval.max_examples,
            len(dataset),
            pdf_only_count,
            len(dataset_all) - pdf_only_count,
        )

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
    collections = [c.name for c in store.client.get_collections().collections]  # type: ignore[attr-defined]
    if cfg.vector_store.qdrant.collection_name not in collections or recreate_index:
        ingest_all(cfg, store, recreate=True)
    llm = _build_llm(cfg, store)

    preds: list[int] = []
    rows: list[Mapping[str, Any]] = []
    total_prompt_tokens = 0
    total_completion_tokens = 0
    start_all = time.perf_counter()
    llm_log_path = Path("./logs/llm_responses.log")
    llm_log_path.parent.mkdir(parents=True, exist_ok=True)
    llm_log_path.write_text("", encoding="utf-8")  # truncate per run
    llm_log_snapshot = cfg.eval.output_dir / "llm_responses.log"
    llm_log_snapshot.parent.mkdir(parents=True, exist_ok=True)
    llm_log_snapshot.write_text("", encoding="utf-8")
    run_id = time.strftime("%Y%m%dT%H%M%S")

    for ex in dataset:
        options_text = "\n".join(f"{chr(ord('a') + idx)}. {opt}" for idx, opt in enumerate(ex["options"]))
        query = f"{ex['question']}\n\nOptions:\n{options_text}"
        t0 = time.perf_counter()
        rag_mode = cfg.llm.openai.rag_mode.value if hasattr(cfg.llm.openai.rag_mode, "value") else str(cfg.llm.openai.rag_mode)
        rag_result = answer_query(query, cfg, store, llm, force_choice=True, rag_mode=rag_mode)
        elapsed = time.perf_counter() - t0
        usage = rag_result.get("usage") or {}
        prompt_tokens, completion_tokens, total_tokens = _extract_token_usage(usage)
        total_prompt_tokens += prompt_tokens
        total_completion_tokens += completion_tokens
        normalized_answer = _normalize_choice_answer(rag_result["answer"])
        pred_idx = _extract_choice_index(normalized_answer, len(ex["options"]))
        preds.append(pred_idx)
        used_chunks = rag_result.get("used_chunks", [])
        rows.append(
            {
                "id": ex["id"],
                "pred": pred_idx,
                "label": ex["correct_index"],
                "answer": normalized_answer,
                "raw_answer": rag_result.get("llm_raw_answer", rag_result["answer"]),
                "elapsed_sec": round(elapsed, 3),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens if total_tokens else prompt_tokens + completion_tokens,
            }
        )
        # Log per-sample diagnostics.
        top_chunks = [
            (
                c.get("id"),
                c.get("metadata", {}).get("rerank_score", c.get("metadata", {}).get("score")),
                c.get("metadata", {}),
            )
            for c in used_chunks[:5]
        ]
        logger.info(
            "eval sample id=%s pred=%s gold=%s elapsed=%.3fs prompt_tokens=%s completion_tokens=%s chunks=%s answer=%s query=%s",
            ex["id"],
            pred_idx,
            ex["correct_index"],
            elapsed,
            prompt_tokens,
            completion_tokens,
            top_chunks,
            normalized_answer,
            query,
        )
        _append_llm_log(
            [llm_log_path, llm_log_snapshot],
            {
                "run_id": run_id,
                "id": ex["id"],
                "query": query,
                "answer": normalized_answer,
                "raw_answer": rag_result.get("llm_raw_answer", rag_result["answer"]),
                "pred": pred_idx,
                "label": ex["correct_index"],
                "elapsed_sec": round(elapsed, 3),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "used_chunks": used_chunks,
                "retrieval_info": rag_result.get("retrieval_info", {}),
                "usage": usage,
            },
        )

    labels = [ex["correct_index"] for ex in dataset]
    acc = metrics_mod.accuracy(preds, labels)
    f1 = metrics_mod.macro_f1(preds, labels)
    duration_all = time.perf_counter() - start_all
    summary = {
        "accuracy": acc,
        "macro_f1": f1,
        "num_examples": len(dataset),
        "pdf_only_skipped": pdf_only_count,
        "timing": {
            "total_duration_sec": round(duration_all, 3),
            "avg_response_time_sec": round(sum(r["elapsed_sec"] for r in rows) / len(rows), 3),
        },
        "token_usage": {
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "total_tokens": total_prompt_tokens + total_completion_tokens,
        },
    }
    save_metrics(summary, cfg.eval.output_dir / "metrics.json")
    save_details(rows, cfg.eval.output_dir / "details.csv")
    # Also write a per-sample log file for easier inspection.
    _write_sample_logs(cfg.eval.output_dir / "run_samples.log", rows)
    # Save config snapshot for reproducibility.
    snapshot_path = cfg.eval.output_dir / "config_snapshot.yaml"
    with snapshot_path.open("w", encoding="utf-8") as fp:
        yaml.safe_dump(_stringify_paths(cfg.model_dump()), fp, allow_unicode=True, sort_keys=False)
    if config_path:
        logger.info("Config snapshot saved to %s (source: %s)", snapshot_path, config_path)
    else:
        logger.info("Config snapshot saved to %s", snapshot_path)
    return EvalResult(summary=summary, details=rows)


def _build_llm(cfg: AppConfig, store: VectorStoreClient) -> LlmClient:
    """Instantiate LLM client from config for eval.

    Args:
        cfg: Application configuration.
        store: Vector store client instance.

    Returns:
        Configured LLM client.
    """

    if cfg.llm.provider == "openai":
        api_key = os.getenv(cfg.llm.openai.api_key_env, "")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for eval when provider=openai.")
        rag_mode = cfg.llm.openai.rag_mode.value if hasattr(cfg.llm.openai.rag_mode, "value") else str(cfg.llm.openai.rag_mode)
        logger.info(
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
        api_key = os.getenv(cfg.llm.lmstudio.api_key_env, "")
        if not api_key:
            raise RuntimeError("LMSTUDIO_API_KEY is required for eval when provider=lmstudio.")
        logger.info(
            "LLM setup provider=%s model=%s base_url=%s",
            cfg.llm.provider,
            cfg.llm.model,
            cfg.llm.lmstudio.base_url,
        )
        return LmStudioOpenAIClient(
            base_url=cfg.llm.lmstudio.base_url,
            model=cfg.llm.model,
            api_key=api_key,
        )
    logger.info("LLM setup provider=%s model=%s (simple local)", cfg.llm.provider, cfg.llm.model)
    return SimpleLocalClient()


def _extract_choice_index(answer_text: str, num_options: int) -> int:
    """Extract predicted choice index from model answer.

    Args:
        answer_text: Text response from LLM.
        num_options: Number of available options.

    Returns:
        Predicted option index (0-based). Returns -1 if parsing fails.
    """

    text = answer_text.strip().lower()
    for idx, label in enumerate(["a", "b", "c", "d", "e", "f"][:num_options]):
        if text.startswith(label) or f" {label}" in text:
            return idx
    digits = [int(ch) for ch in text if ch.isdigit()]
    if digits:
        cand = digits[0]
        if 0 <= cand < num_options:
            return cand
        if 1 <= cand <= num_options:
            return cand - 1
    return -1


def _normalize_choice_answer(answer_text: str) -> str:
    """Normalize model answer to a single lowercase letter when possible."""

    text = answer_text.strip().lower()
    for ch in text:
        if ch in {"a", "b", "c", "d", "e", "f"}:
            return ch
    return text


def _extract_token_usage(usage: Mapping[str, Any]) -> Tuple[int, int, int]:
    """Extract prompt/completion/total tokens from usage mapping."""

    prompt = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
    completion = usage.get("completion_tokens") or usage.get("output_tokens") or 0
    total = usage.get("total_tokens") or (prompt + completion if (prompt or completion) else 0)
    return int(prompt), int(completion), int(total)


def _write_sample_logs(path: Path, rows: list[Mapping[str, Any]]) -> None:
    """Write per-sample concise logs into a file for debugging."""

    import csv

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    field_set: set[str] = set()
    for row in rows:
        field_set.update(row.keys())
    field_set.add("raw_answer")
    fieldnames = [
        "id",
        "pred",
        "label",
        "answer",
        "raw_answer",
        "elapsed_sec",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
    ]
    # Ensure any extra keys are also captured to avoid DictWriter mismatch.
    for key in sorted(field_set):
        if key not in fieldnames:
            fieldnames.append(key)

    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            sanitized = dict(row)
            sanitized.setdefault("raw_answer", "")
            writer.writerow(sanitized)


def _append_llm_log(paths: list[Path], record: Mapping[str, Any]) -> None:
    """Append one JSON line with LLM interaction info to multiple paths."""

    line = json.dumps(record, ensure_ascii=False)
    for path in paths:
        with path.open("a", encoding="utf-8") as fp:
            fp.write(line + "\n")


def _stringify_paths(obj: Any) -> Any:
    """Recursively convert Path objects to str for YAML serialization."""

    if isinstance(obj, dict):
        return {k: _stringify_paths(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_stringify_paths(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "value") and hasattr(obj.value, "value"):
        # Config enums (ConfigOption)
        return obj.value.value
    if isinstance(obj, enum.Enum):
        return obj.value
    return obj


__all__ = ["EvalResult", "run_eval"]
