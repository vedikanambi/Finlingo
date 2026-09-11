from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml


@dataclass(frozen=True)
class RiskTaxonomy:
    labels: tuple[str, ...]
    mappings: dict[str, tuple[str, ...]]
    safe_policy: str


@lru_cache(maxsize=8)
def load_risk_taxonomy(path: str | Path) -> RiskTaxonomy:
    resolved = Path(path).resolve()
    if not resolved.exists():
        raise RuntimeError(f"Risk taxonomy file not found: {resolved}")
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    raw_labels = payload.get("labels") or {}
    if not isinstance(raw_labels, dict) or not raw_labels:
        raise RuntimeError("Risk taxonomy must define a non-empty 'labels' mapping")
    labels = tuple(str(label).strip() for label in raw_labels if str(label).strip())
    if "Safe" not in labels:
        labels = (*labels, "Safe")
    mappings = {
        str(label): tuple(str(item).strip() for item in (items or []) if str(item).strip())
        for label, items in raw_labels.items()
    }
    mappings.setdefault("Safe", tuple())
    return RiskTaxonomy(
        labels=labels,
        mappings=mappings,
        safe_policy=str(payload.get("safe_policy") or "").strip(),
    )
