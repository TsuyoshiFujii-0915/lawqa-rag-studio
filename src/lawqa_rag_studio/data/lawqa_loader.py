"""Loader for lawqa_jp dataset subsets."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypedDict, cast


class LawQaExample(TypedDict):
    """Law QA example schema."""

    id: str
    question: str
    options: list[str]
    correct_index: int
    references: list[str]
    pdf_only: bool
    meta: dict[str, Any]


def load_lawqa(path: Path, *, exclude_pdf_only: bool = False) -> list[LawQaExample]:
    """Load lawqa_jp selection_randomized.json in its canonical schema.

    Args:
        path: Path to the JSON file (expected `samples` array with Japanese-keyed fields).
        exclude_pdf_only: When True, drop samples that do not reference any e-Gov URL.

    Returns:
        List of `LawQaExample` objects.

    Raises:
        ValueError: When required fields are missing or invalid.
    """

    def _require(cond: bool, msg: str) -> None:
        if not cond:
            raise ValueError(msg)

    with path.open("r", encoding="utf-8") as fp:
        loaded = json.load(fp)

    if isinstance(loaded, dict) and isinstance(loaded.get("samples"), list):
        data: list[dict[str, Any]] = loaded["samples"]
    else:
        raise ValueError("lawqa_jp data must be an object with key 'samples' as a list.")

    def _parse_options(raw_opts: Any) -> list[str]:
        """Parse options: string with line-separated a/b/c/d... or list of strings."""
        if isinstance(raw_opts, list):
            opts = [str(o).strip() for o in raw_opts if str(o).strip()]
        elif isinstance(raw_opts, str):
            lines = [ln.strip() for ln in raw_opts.splitlines() if ln.strip()]
            opts: list[str] = []
            for ln in lines:
                # strip leading bullet like "a " or "A " or "a．"
                if len(ln) >= 2 and ln[0].lower() in "abcdefghijklmnopqrstuvwxyz" and (ln[1].isspace() or ln[1] in "．."):
                    opts.append(ln[2:].strip())
                else:
                    opts.append(ln)
        else:
            raise ValueError("選択肢 must be a string or list of strings.")
        _require(len(opts) >= 2, "選択肢 must contain at least two options.")
        return opts

    def _correct_index(raw_output: Any, num_opts: int) -> int:
        if not isinstance(raw_output, str) or not raw_output:
            raise ValueError("output must be a non-empty string like 'a'/'b'/....")
        letter = raw_output.strip().lower()[0]
        _require(letter in "abcdefghijklmnopqrstuvwxyz", f"output must start with a letter, got {raw_output!r}")
        idx = ord(letter) - ord("a")
        _require(0 <= idx < num_opts, f"output index out of range for {num_opts} options.")
        return idx

    examples: list[LawQaExample] = []
    for raw in data:
        _require(isinstance(raw, dict), f"Each sample must be an object, got {type(raw)}")
        question = raw.get("問題文")
        _require(isinstance(question, str) and question.strip(), "問題文 is required and must be a string.")

        options = _parse_options(raw.get("選択肢"))
        correct_index = _correct_index(raw.get("output"), len(options))

        references = raw.get("references", [])
        _require(isinstance(references, list), "references must be a list.")
        has_egov = any(isinstance(ref, str) and ref.startswith("https://laws.e-gov.go.jp") for ref in references)
        pdf_only = not has_egov

        file_id = raw.get("ファイル名")
        _require(isinstance(file_id, str) and file_id.strip(), "ファイル名 is required and must be a string.")

        sample: LawQaExample = cast(
            LawQaExample,
            {
                "id": file_id,
                "question": question.strip(),
                "options": options,
                "correct_index": correct_index,
                "references": references,
                "pdf_only": pdf_only,
                "meta": raw,
            },
        )
        if exclude_pdf_only and pdf_only:
            continue
        examples.append(sample)
    return examples


__all__ = ["LawQaExample", "load_lawqa"]
