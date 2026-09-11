from __future__ import annotations

import csv
import json
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from backend.app.core.config import Settings
from backend.app.services.text_utils import normalise_text
from backend.app.stages.stage1_document_parser import Stage1DocumentParser


def evaluate_parser_boundaries(
    settings: Settings,
    manifest_path: Path,
    *,
    output_path: Path | None = None,
    match_threshold: float = 0.85,
) -> dict[str, Any]:
    """checks clause extraction against manually transcribed gold clauses - reads docs in place, never copies them into the project."""
    documents = _load_manifest(manifest_path)
    parser = Stage1DocumentParser(settings)
    per_document: list[dict[str, Any]] = []
    total_predicted = total_gold = total_matched = 0

    for item in documents:
        path = Path(str(item["path"])).expanduser()
        if not path.is_absolute():
            path = (manifest_path.parent / path).resolve()
        gold = [normalise_text(value) for value in item["gold_clauses"] if normalise_text(value)]
        predicted_rows = parser.parse(path)
        predicted = [normalise_text(row.text) for row in predicted_rows]
        matches = _greedy_matches(predicted, gold, match_threshold)
        matched = len(matches)
        precision = matched / len(predicted) if predicted else 0.0
        recall = matched / len(gold) if gold else 0.0
        per_document.append(
            {
                "path": str(path),
                "predicted_clauses": len(predicted),
                "gold_clauses": len(gold),
                "matched_clauses": matched,
                "precision": precision,
                "recall": recall,
                "f1": _f1(precision, recall),
                "extraction_methods": sorted({row.extraction_method for row in predicted_rows}),
                "matches": matches,
            }
        )
        total_predicted += len(predicted)
        total_gold += len(gold)
        total_matched += matched

    precision = total_matched / total_predicted if total_predicted else 0.0
    recall = total_matched / total_gold if total_gold else 0.0
    payload = {
        "match_threshold": match_threshold,
        "documents": per_document,
        "micro": {
            "predicted_clauses": total_predicted,
            "gold_clauses": total_gold,
            "matched_clauses": total_matched,
            "precision": precision,
            "recall": recall,
            "f1": _f1(precision, recall),
        },
        "definition": "Greedy one-to-one normalized text matching using SequenceMatcher ratio.",
    }
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".json":
        rows = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise ValueError("Parser manifest JSON must be a list")
        output = []
        for row in rows:
            if not isinstance(row, dict) or not row.get("path") or not isinstance(row.get("gold_clauses"), list):
                raise ValueError("Every JSON row requires path and gold_clauses[]")
            output.append(row)
        return output
    grouped: dict[str, list[str]] = {}
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            document = str(row.get("path") or "").strip()
            clause = str(row.get("gold_clause") or "").strip()
            if document and clause:
                grouped.setdefault(document, []).append(clause)
    if not grouped:
        raise ValueError("Parser manifest CSV requires path,gold_clause rows")
    return [{"path": key, "gold_clauses": values} for key, values in grouped.items()]


def _greedy_matches(predicted: list[str], gold: list[str], threshold: float) -> list[dict[str, Any]]:
    candidates = sorted(
        (
            (SequenceMatcher(None, pred, target).ratio(), pred_index, gold_index)
            for pred_index, pred in enumerate(predicted)
            for gold_index, target in enumerate(gold)
        ),
        reverse=True,
    )
    used_predicted: set[int] = set()
    used_gold: set[int] = set()
    output: list[dict[str, Any]] = []
    for score, pred_index, gold_index in candidates:
        if score < threshold:
            break
        if pred_index in used_predicted or gold_index in used_gold:
            continue
        used_predicted.add(pred_index)
        used_gold.add(gold_index)
        output.append(
            {
                "predicted_index": pred_index,
                "gold_index": gold_index,
                "similarity": round(score, 6),
            }
        )
    return sorted(output, key=lambda row: row["gold_index"])


def _f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0
