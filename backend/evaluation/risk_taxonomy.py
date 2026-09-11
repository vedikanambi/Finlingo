from __future__ import annotations

import re
from pathlib import Path

import yaml


def _normalise_source_label(value: str) -> str:
    """Normalize source labels without using broad substring matching."""
    text = str(value or "").strip().lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


class RiskTaxonomy:
    def __init__(self, path: Path) -> None:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))

        self.version = str(raw.get("version", "unknown"))
        self.mapping_mode = str(raw.get("mapping_mode", "exact_source_label"))
        self.safe_policy = str(raw.get("safe_policy", ""))

        self.labels: dict[str, list[str]] = {
            str(label): [str(item) for item in source_labels] for label, source_labels in raw["labels"].items()
        }

        self._exact_mapping: dict[str, str] = {}
        duplicate_mappings: dict[str, set[str]] = {}

        for target_label, source_labels in self.labels.items():
            for source_label in source_labels:
                normalised = _normalise_source_label(source_label)

                existing = self._exact_mapping.get(normalised)
                if existing is not None and existing != target_label:
                    duplicate_mappings.setdefault(normalised, {existing}).add(target_label)

                self._exact_mapping[normalised] = target_label

        if duplicate_mappings:
            raise ValueError(
                f"Risk taxonomy contains source labels mapped to multiple target classes: {duplicate_mappings}"
            )

    def map_question(self, question: str) -> str | None:
        """Return a target risk only for an exact normalized source label."""
        return self._exact_mapping.get(_normalise_source_label(question))
