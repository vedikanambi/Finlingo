from __future__ import annotations

import csv
import io
import random
from dataclasses import dataclass

import httpx

from backend.app.core.config import Settings


@dataclass(frozen=True)
class FaithBenchRecord:
    source: str
    summary: str
    supported: int
    worst_label: str
    best_label: str
    summarizer: str


def load_faithbench(settings: Settings, max_examples: int | None = None) -> list[FaithBenchRecord]:
    """pulls the official FaithBench CSV straight into memory, nothing written to disk."""
    limit = max_examples or settings.faithbench_max_examples
    with httpx.Client(
        timeout=settings.faithbench_timeout,
        follow_redirects=True,
        headers={"User-Agent": "FinLingoPP-AcademicResearch/1.0"},
    ) as client:
        response = client.get(settings.faithbench_url)
        response.raise_for_status()
    reader = csv.DictReader(io.StringIO(response.text))
    rows: list[FaithBenchRecord] = []
    seen: set[tuple[str, str]] = set()
    for row in reader:
        source = str(row.get("source") or "").strip()
        summary = str(row.get("summary") or "").strip()
        worst = str(row.get("worst-label") or "").strip()
        best = str(row.get("best-label") or "").strip()
        summarizer = str(row.get("LLM") or "").strip()
        if not source or not summary or not worst:
            continue
        key = (source, summary)
        if key in seen:
            continue
        seen.add(key)
        label = worst.lower()
        if label in {"questionable", "unwanted"}:
            supported = 0
        elif label in {"consistent", "benign"}:
            supported = 1
        else:
            continue
        rows.append(FaithBenchRecord(source, summary, supported, worst, best, summarizer))

    if not rows:
        raise RuntimeError("The official FaithBench endpoint returned no usable rows")
    if limit >= len(rows):
        return rows

    # stratified so both classes and a mix of summarizer models stay represented after downsampling
    rng = random.Random(settings.random_seed)
    buckets: dict[tuple[int, str], list[FaithBenchRecord]] = {}
    for row in rows:
        buckets.setdefault((row.supported, row.summarizer), []).append(row)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    selected: list[FaithBenchRecord] = []
    active = sorted(buckets)
    while active and len(selected) < limit:
        next_active = []
        for key in active:
            bucket = buckets[key]
            if bucket and len(selected) < limit:
                selected.append(bucket.pop())
            if bucket:
                next_active.append(key)
        active = next_active
    rng.shuffle(selected)
    return selected[:limit]
