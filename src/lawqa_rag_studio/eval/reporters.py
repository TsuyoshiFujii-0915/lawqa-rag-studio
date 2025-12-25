"""Report generators for evaluation."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping


def save_metrics(metrics: Mapping[str, Any], output_path: Path) -> None:
    """Save metrics to JSON file.

    Args:
        metrics: Metrics dictionary.
        output_path: Destination file path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")


def save_details(rows: list[Mapping[str, Any]], output_path: Path) -> None:
    """Save detailed evaluation rows to CSV.

    Args:
        rows: Row dictionaries.
        output_path: Destination CSV path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return
    field_set: set[str] = set()
    for row in rows:
        field_set.update(row.keys())
    field_set.add("raw_answer")
    base_order = [
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
    fieldnames = base_order[:]
    for key in sorted(field_set):
        if key not in fieldnames:
            fieldnames.append(key)
    with output_path.open("w", encoding="utf-8", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            sanitized = dict(row)
            sanitized.setdefault("raw_answer", "")
            writer.writerow(sanitized)


__all__ = ["save_metrics", "save_details"]
