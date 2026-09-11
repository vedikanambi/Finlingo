from __future__ import annotations

import hashlib
import json
from pathlib import Path

from backend.app.core.config import Settings
from backend.app.services.dataset_streams import StreamingDatasetRepository


def _key(row: dict) -> str:
    dataset = str(row.get("dataset") or "unknown")
    document = str(row.get("document_id") or row.get("source_id") or "")
    return hashlib.sha256(f"{dataset}\0{document}".encode("utf-8", errors="ignore")).hexdigest()


def audit_split_contamination(
    settings: Settings,
    *,
    sample_per_split: int = 5_000,
    output_path: Path | None = None,
) -> dict:
    """Stream document IDs only and verify train/validation/test disjointness."""
    repository = StreamingDatasetRepository(settings)
    split_keys: dict[str, set[str]] = {"train": set(), "validation": set(), "test": set()}
    errors: dict[str, str] = {}
    try:
        for split in split_keys:
            try:
                for row in repository.mixed_simplification_sources(sample_per_split, split=split):
                    split_keys[split].add(_key(row))
            except Exception as exc:
                errors[split] = str(exc)
    finally:
        repository.close()
    overlaps = {
        "train_validation": len(split_keys["train"] & split_keys["validation"]),
        "train_test": len(split_keys["train"] & split_keys["test"]),
        "validation_test": len(split_keys["validation"] & split_keys["test"]),
    }
    payload = {
        "sampled_documents": {name: len(values) for name, values in split_keys.items()},
        "overlaps": overlaps,
        "passed": not any(overlaps.values()) and not errors,
        "errors": errors,
        "definition": "SHA-256(dataset_id + document_id); source text is never persisted",
    }
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
