from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from backend.app.core.config import Settings
from backend.app.services.text_utils import normalise_text
from backend.training.silver_labels import ProviderQuotaExhausted, SimplificationBatch, _retry_after_seconds
from backend.training.silver_label_providers import SilverLabelProvider, build_silver_label_provider

_PROMPT = (
    "Create faithful plain-English consumer-finance references. Preserve every number, date, percentage, entity, "
    "obligation, condition, exception and negation. Do not add facts. Return JSON only."
)


class CorrectedCacheResult(BaseModel):
    requested_records: int = 0
    accepted_labels: int = 0
    rejected_labels: int = 0
    malformed_responses: int = 0
    duplicates: int = 0
    source_hashes: list[str] = Field(default_factory=list)
    retry_after_seconds: float | None = None
    rejection_reasons: dict[str, int] = Field(default_factory=dict)


def source_hash(text: str) -> str:
    return hashlib.sha256(normalise_text(text).encode("utf-8")).hexdigest()


def output_hash(text: str) -> str:
    return hashlib.sha256(normalise_text(text).casefold().encode("utf-8")).hexdigest()


def validate_preservation(source: str, target: str) -> tuple[bool, list[str]]:
    """Require exact preservation of critical surface facts before accepting silver data."""
    months = r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    checks = {
        "number": r"(?<!\w)(?:[$£€])?\d[\d,]*(?:\.\d+)?",
        "date": rf"\b(?:\d{{1,2}}[/-]\d{{1,2}}[/-]\d{{2,4}}|{months}\s+\d{{1,2}}(?:,\s*)?\d{{4}}|\d{{1,2}}\s+{months}\s+\d{{4}})\b",
        "percentage": r"\b\d+(?:\.\d+)?\s*(?:%|percent)\b",
        "negation": r"\b(?:no|not|never|without|unless|cannot|can't|won't|mustn't|may not)\b",
    }
    failed: list[str] = []
    for name, pattern in checks.items():
        flags = 0 if name == "entity" else re.IGNORECASE
        before = {normalise_text(value).casefold() for value in re.findall(pattern, source, flags)}
        after = {normalise_text(value).casefold() for value in re.findall(pattern, target, flags)}
        if before != after:
            failed.append(name)
    if _entities(source) != _entities(target):
        failed.append("entity")
    if _obligation_polarity(source) != _obligation_polarity(target):
        failed.append("obligation")
    return not failed, failed


def _entities(text: str) -> set[str]:
    """Extract high-precision organisation names/acronyms without treating sentence starts as entities."""
    suffix = r"(?:Bank|Bureau|Company|Corp(?:oration)?|Credit Union|Financial|Inc|LLC|Ltd|Services)"
    names = re.findall(rf"\b(?:[A-Z][A-Za-z&.-]+\s+){{0,4}}{suffix}\b", text)
    acronyms = re.findall(r"\b[A-Z]{2,8}\b", text)
    return {normalise_text(value).casefold() for value in [*names, *acronyms]}


def _obligation_polarity(text: str) -> tuple[bool, bool]:
    required = bool(re.search(r"\b(?:must|shall|required|need(?:s)? to|has to|have to)\b", text, re.I))
    prohibited = bool(re.search(r"\b(?:must not|shall not|may not|cannot|can't|prohibited)\b", text, re.I))
    return required, prohibited


class CorrectedCFPBCache:
    def __init__(
        self, settings: Settings, root: Path | None = None, provider: SilverLabelProvider | None = None
    ) -> None:
        self.settings = settings
        self.root = settings.resolve(root or settings.corrected_cfpb_cache_dir)
        self.batch_dir = self.root / "batches"
        self.manifest_path = self.root / "manifest.json"
        self.provider = provider or build_silver_label_provider(settings)

    def identity(self, split: str) -> dict[str, Any]:
        return {
            "dataset_identifier": self.settings.cfpb_dataset,
            "dataset_revision": self.settings.dataset_revision(self.settings.cfpb_dataset),
            "split": split,
            "selected_narrative_field": self.settings.corrected_cfpb_narrative_field,
            "preprocessing_version": self.settings.corrected_cfpb_preprocessing_version,
            "prompt_template_version": self.settings.corrected_cfpb_prompt_template_version,
            "provider": self.settings.silver_label_provider,
            "model_identifier": (
                self.settings.silver_local_model
                if self.settings.silver_label_provider == "local_qwen"
                else self.settings.gemini_silver_label_model
                if self.settings.silver_label_provider == "gemini"
                else self.settings.groq_silver_label_model
            ),
            "model_revision": self.settings.model_revision(
                self.settings.silver_local_model
                if self.settings.silver_label_provider == "local_qwen"
                else self.settings.groq_silver_label_model
            ),
            "schema_version": self.settings.corrected_cfpb_schema_version,
        }

    def generate_one(self, rows: list[dict[str, Any]], split: str) -> CorrectedCacheResult:
        hashes = [source_hash(row["text"]) for row in rows]
        result = CorrectedCacheResult(requested_records=len(rows), source_hashes=hashes)
        identity = self.identity(split)
        batch_hash = hashlib.sha256(
            json.dumps({**identity, "normalized_source_hashes": hashes}, sort_keys=True).encode("utf-8")
        ).hexdigest()
        path = self.batch_dir / f"{batch_hash}.json"
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            result.accepted_labels = len(existing.get("items", []))
            result.duplicates = len(rows)
            self._write_manifest(identity, result)
            return result

        payload = [{"id": digest, "text": normalise_text(row["text"])} for digest, row in zip(hashes, rows)]
        try:
            provider_result = self.provider.generate(
                SimplificationBatch,
                _PROMPT,
                'Return exactly {"items":[{"id":"source hash","simplified":"..."}]}, one item per input.\n'
                + json.dumps(payload, ensure_ascii=False),
            )
        except Exception as exc:
            if getattr(getattr(exc, "response", None), "status_code", None) == 429:
                result.retry_after_seconds = _retry_after_seconds(exc)
                self._write_manifest(identity, result)
                raise ProviderQuotaExhausted(
                    "Groq quota exhausted; corrected cache remains intact.", result.retry_after_seconds
                ) from exc
            raise

        parsed = provider_result.parsed
        result.malformed_responses = provider_result.malformed_attempts
        if parsed is None:
            result.rejected_labels = len(rows)
            self._write_manifest(identity, result)
            return result

        expected = set(hashes)
        response_ids = [item.id for item in parsed.items]
        malformed_ids = set(response_ids) != expected or len(response_ids) != len(set(response_ids))
        if len(parsed.items) != len(rows) or malformed_ids:
            result.malformed_responses = 1
        by_hash = {item.id: normalise_text(item.simplified) for item in parsed.items if item.simplified.strip()}
        existing_sources = self._existing_hashes("source_hash")
        seen_outputs = self._existing_hashes("output_hash")
        accepted: list[dict[str, Any]] = []
        seen_sources: set[str] = set()
        for row, digest in zip(rows, hashes):
            if digest in existing_sources or digest in seen_sources:
                result.duplicates += 1
                continue
            seen_sources.add(digest)
            target = by_hash.get(digest, "")
            if not target:
                result.rejected_labels += 1
                continue
            target_digest = output_hash(target)
            if target_digest in seen_outputs:
                result.duplicates += 1
                continue
            valid, failures = validate_preservation(row["text"], target)
            if not valid:
                result.rejected_labels += 1
                for failure in failures:
                    result.rejection_reasons[failure] = result.rejection_reasons.get(failure, 0) + 1
                continue
            seen_outputs.add(target_digest)
            accepted.append(
                {
                    "source_hash": digest,
                    "output_hash": target_digest,
                    "target": target,
                    "validation": {"critical_fields_preserved": True, "failed_checks": failures},
                    "provider_metadata": provider_result.metadata,
                    "prompt_version": self.settings.corrected_cfpb_prompt_template_version,
                    "status": "accepted",
                }
            )
        result.accepted_labels = len(accepted)
        if accepted:
            self._atomic_json(
                path,
                {
                    "identity": identity,
                    "batch_hash": batch_hash,
                    "created_at": _now(),
                    "provider_metadata": provider_result.metadata,
                    "items": accepted,
                },
            )
        self._write_manifest(identity, result)
        return result

    def _existing_hashes(self, key: str) -> set[str]:
        values: set[str] = set()
        if not self.batch_dir.exists():
            return values
        for path in self.batch_dir.glob("*.json"):
            data = json.loads(path.read_text(encoding="utf-8"))
            values.update(str(item[key]) for item in data.get("items", []) if item.get(key))
        return values

    def _write_manifest(self, identity: dict[str, Any], result: CorrectedCacheResult) -> None:
        batches = sorted(self.batch_dir.glob("*.json")) if self.batch_dir.exists() else []
        accepted = 0
        hashes: set[str] = set()
        for path in batches:
            data = json.loads(path.read_text(encoding="utf-8"))
            accepted += len(data.get("items", []))
            hashes.update(item["source_hash"] for item in data.get("items", []))
        self._atomic_json(
            self.manifest_path,
            {
                "identity": identity,
                "updated_at": _now(),
                "batch_count": len(batches),
                "valid_label_count": accepted,
                "source_hashes": sorted(hashes),
                "last_request": result.model_dump(),
                "raw_records_persisted": False,
            },
        )

    @staticmethod
    def _atomic_json(path: Path, value: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
        temporary.replace(path)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
