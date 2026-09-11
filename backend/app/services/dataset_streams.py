from __future__ import annotations

import hashlib
import itertools
import logging
import random
import tempfile
from collections.abc import Iterable, Iterator
from typing import Any

from backend.app.core.config import Settings

logger = logging.getLogger(__name__)


class StreamingDatasetRepository:
    """Reads dataset rows directly from HF iterable streams with streaming=True, so nothing is ever materialised to disk."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        # HF still wants a scratch cache dir even in streaming mode - this one's temp-only, wiped on close, never a real dataset copy
        self._ephemeral_cache = tempfile.TemporaryDirectory(prefix="finlingo_hf_stream_")
        self._split_cache: dict[tuple[str, str | None], tuple[str, ...]] = {}

    def close(self) -> None:
        cache = getattr(self, "_ephemeral_cache", None)
        if cache is not None:
            cache.cleanup()
            self._ephemeral_cache = None

    def __del__(self) -> None:  # pragma: no cover - defensive cleanup
        try:
            self.close()
        except Exception:
            pass

    def _load(self, dataset_id: str, split: str, config: str | None = None):
        from datasets import load_dataset

        kwargs: dict[str, Any] = {
            "path": dataset_id,
            "split": split,
            "streaming": True,
            "token": self.settings.hf_token,
            "cache_dir": self._ephemeral_cache.name,
        }
        if config:
            kwargs["name"] = config
        revision = self.settings.dataset_revision(dataset_id)
        if revision:
            kwargs["revision"] = revision
        if dataset_id == self.settings.edgar_dataset and self.settings.edgar_trust_remote_code:
            kwargs["trust_remote_code"] = True
        logger.info("Streaming dataset=%s config=%s split=%s", dataset_id, config, split)
        if dataset_id == "theatticusproject/cuad-qa":
            kwargs["trust_remote_code"] = True

        return load_dataset(**kwargs)

    def stream_raw(
        self,
        dataset_id: str,
        *,
        split: str,
        config: str | None = None,
        shuffle: bool = False,
        seed: int | None = None,
        limit: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        dataset = self._load(dataset_id, split, config)
        if shuffle:
            dataset = dataset.shuffle(
                seed=seed or self.settings.random_seed, buffer_size=self.settings.dataset_shuffle_buffer
            )
        iterator: Iterable[dict[str, Any]] = dataset
        if limit is not None:
            iterator = itertools.islice(iterator, limit)
        yield from iterator

    def available_splits(self, dataset_id: str, config: str | None = None) -> tuple[str, ...]:
        key = (dataset_id, config)
        if key not in self._split_cache:
            from datasets import get_dataset_split_names

            split_kwargs: dict[str, Any] = {
                "path": dataset_id,
                "config_name": config,
                "token": self.settings.hf_token,
            }
            revision = self.settings.dataset_revision(dataset_id)
            if revision:
                split_kwargs["revision"] = revision
            names = get_dataset_split_names(**split_kwargs)
            self._split_cache[key] = tuple(str(name) for name in names)
        return self._split_cache[key]

    def stream_partitioned(
        self,
        dataset_id: str,
        *,
        requested_split: str,
        config: str | None = None,
        key_fields: tuple[str, ...] = ("document_id", "title", "id"),
        shuffle: bool = False,
        limit: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Use official splits when present, else derive a deterministic split (hashed doc id) to avoid train/test leakage on single-split sources."""
        try:
            splits = self.available_splits(dataset_id, config)
        except Exception as exc:
            logger.warning("Could not inspect splits for %s/%s: %s", dataset_id, config, exc)
            splits = ("train",)
        if requested_split in splits:
            yield from self.stream_raw(dataset_id, split=requested_split, config=config, shuffle=shuffle, limit=limit)
            return
        if "train" not in splits:
            raise RuntimeError(f"Dataset {dataset_id} config={config} has splits {splits}, not {requested_split}")
        emitted = 0
        for row in self.stream_raw(dataset_id, split="train", config=config, shuffle=shuffle):
            key = next((str(row.get(field)) for field in key_fields if row.get(field) not in (None, "")), None)
            key = key or repr(sorted(row.items()))[:500]
            # hash the doc id instead of shuffling - keeps the same document out of train and test forever
            bucket = int(hashlib.sha256(key.encode("utf-8", errors="ignore")).hexdigest()[:8], 16) % 100
            selected = (
                (requested_split == "train" and bucket < 80)
                or (requested_split == "validation" and 80 <= bucket < 90)
                or (requested_split == "test" and bucket >= 90)
            )
            if not selected:
                continue
            yield row
            emitted += 1
            if limit is not None and emitted >= limit:
                return

    def cuad_clauses(self, split: str = "train", limit: int | None = None) -> Iterator[dict[str, Any]]:
        emitted = 0
        for row in self.stream_partitioned(
            self.settings.cuad_dataset,
            requested_split=split,
            config=self.settings.cuad_config,
            key_fields=("file_name", "title", "document_id", "id"),
            shuffle=split == "train",
        ):
            text = _first_text(row, "clause", "text", "answers")
            if not text or isinstance(text, dict):
                answers = row.get("answers") or {}
                texts = answers.get("text") if isinstance(answers, dict) else None
                if texts:
                    for index, t in enumerate(texts):
                        t = str(t).strip()
                        if len(t.split()) < 5:
                            continue
                        yield {
                            "source_id": f"{row.get('id', 'cuad')}::{index}",
                            "document_id": str(row.get("file_name") or row.get("title") or row.get("id") or "unknown"),
                            "text": t,
                            "question": str(row.get("question") or ""),
                            "dataset": self.settings.cuad_dataset,
                            "split": split,
                        }
                        emitted += 1
                        if limit is not None and emitted >= limit:
                            return
                continue
            text = str(text).strip()
            if len(text.split()) < 5:
                continue
            yield {
                "source_id": str(row.get("file_name") or row.get("id") or f"cuad-{emitted}"),
                "document_id": str(row.get("file_name") or row.get("id") or "unknown"),
                "text": text,
                "question": str(row.get("label") or ""),
                "dataset": self.settings.cuad_dataset,
                "split": split,
            }
            emitted += 1
            if limit is not None and emitted >= limit:
                return

    def contractnli_pairs(
        self, split: str = "train", config: str | None = None, limit: int | None = None
    ) -> Iterator[dict[str, Any]]:
        emitted = 0
        try:
            for row in self.stream_partitioned(
                self.settings.contractnli_dataset,
                requested_split=split,
                config=None,
                key_fields=("id", "pid", "document_id"),
                shuffle=split == "train",
            ):
                for item in _normalise_contractnli_row(row, "default"):
                    if item["label"] not in {"entailment", "neutral", "contradiction"}:
                        continue
                    yield item
                    emitted += 1
                    if limit is not None and emitted >= limit:
                        return
        except Exception as exc:
            logger.warning("ContractNLI split=%s unavailable: %s", split, exc)
        if emitted == 0:
            raise RuntimeError(f"No usable ContractNLI rows were streamed for split={split}")

    def snli_pairs(self, split: str = "train", limit: int | None = None) -> Iterator[dict[str, Any]]:
        label_map = {0: "entailment", 1: "neutral", 2: "contradiction"}
        emitted = 0
        for row in self.stream_raw(self.settings.snli_dataset, split=split, shuffle=split == "train"):
            label = label_map.get(row.get("label"), str(row.get("label", "")).lower())
            premise = str(row.get("premise") or row.get("sentence1") or "").strip()
            hypothesis = str(row.get("hypothesis") or row.get("sentence2") or "").strip()
            if label not in label_map.values() or not premise or not hypothesis:
                continue
            yield {
                "source_id": str(row.get("id") or f"snli-{emitted}"),
                "premise": premise,
                "hypothesis": hypothesis,
                "label": label,
                "evidence_chunk_id": None,
                "dataset": self.settings.snli_dataset,
                "split": split,
            }
            emitted += 1
            if limit is not None and emitted >= limit:
                return

    def financebench(self, split: str = "train", limit: int | None = None) -> Iterator[dict[str, Any]]:
        yield from self.stream_raw(self.settings.financebench_dataset, split=split, limit=limit)

    def cfpb_narratives(self, split: str = "train", limit: int | None = None) -> Iterator[dict[str, Any]]:
        emitted = 0
        rows = self.stream_partitioned(
            self.settings.cfpb_dataset,
            requested_split=split,
            key_fields=(
                "complaint_id",
                "id",
                "consumer_complaint_narrative",
                "complaint_what_happened",
            ),
            shuffle=split == "train",
        )

        for row in rows:
            text = _first_text(
                row,
                "consumer_narrative",
                "consumer_complaint_narrative",
                "complaint_what_happened",
                "narrative",
                "text",
            )
            if len(text.split()) < 10:
                continue
            yield {
                "source_id": str(row.get("complaint_id") or row.get("id") or f"cfpb-{emitted}"),
                "document_id": str(row.get("complaint_id") or row.get("id") or "unknown"),
                "text": text,
                "product": str(row.get("product") or ""),
                "issue": str(row.get("issue") or row.get("subissue") or ""),
                "sub_issue": str(row.get("sub_issue") or row.get("subissue") or ""),
                "dataset": self.settings.cfpb_dataset,
                "split": split,
            }
            emitted += 1
            if limit is not None and emitted >= limit:
                return

    def edgar_texts(self, split: str = "train", limit: int | None = None) -> Iterator[dict[str, Any]]:
        emitted = 0
        rows = self.stream_partitioned(
            self.settings.edgar_dataset,
            requested_split=split,
            config=self.settings.edgar_config,
            key_fields=("accession_number", "document_id", "id"),
            shuffle=split == "train",
        )
        for row in rows:
            text = _first_text(row, "text", "content", "section_text", "document")
            if not text:
                parts = []
                for field in ("instruction", "input", "output"):
                    val = str(row.get(field) or "").strip()
                    if val:
                        parts.append(val)
                text = " ".join(parts)
            if not text:
                section_keys = sorted(
                    key
                    for key, value in row.items()
                    if key.lower().startswith("section_") and isinstance(value, str) and value.strip()
                )
                text = "\n\n".join(str(row[key]).strip() for key in section_keys)
            if len(text.split()) < 20:
                continue
            yield {
                "source_id": str(row.get("id") or row.get("accession_number") or f"edgar-{emitted}"),
                "document_id": str(row.get("accession_number") or row.get("id") or "unknown"),
                "text": text,
                "year": row.get("year"),
                "cik": row.get("cik"),
                "dataset": self.settings.edgar_dataset,
                "split": split,
            }
            emitted += 1
            if limit is not None and emitted >= limit:
                return

    def mixed_simplification_sources(self, total: int, split: str = "train") -> Iterator[dict[str, Any]]:
        """Interleave legal/financial sources on the fly; validation/test use only explicit non-train splits to avoid leakage."""
        if split != "train":
            cuad_split = "test" if split == "validation" else split

            streams = [
                self.cuad_clauses(
                    cuad_split,
                    total,
                ),
                self._contractnli_texts(
                    split,
                    max(1, int(total * 0.3)),
                ),
            ]
        else:
            streams = [
                self.cuad_clauses("train", total),
                self._contractnli_texts("train", max(1, int(total * 0.15))),
                self.cfpb_narratives("train", max(1, int(total * 0.15))),
                self.edgar_texts("train", total),
            ]
        rng = random.Random(self.settings.random_seed + (0 if split == "train" else 1))
        active = list(streams)
        emitted = 0
        seen_sources: set[str] = set()

        while active and emitted < total:
            stream = rng.choice(active)

            try:
                row = next(stream)
                row = dict(row)

                words = str(row.get("text") or "").split()

                if len(words) > self.settings.simplification_source_max_words:
                    row["text"] = " ".join(words[: self.settings.simplification_source_max_words])

                source_id = str(
                    row.get("source_id") or (f"{row.get('dataset')}::{row.get('document_id')}::{row.get('text')}")
                )

                if source_id in seen_sources:
                    continue

                seen_sources.add(source_id)

                yield row
                emitted += 1
            except StopIteration:
                active.remove(stream)
            except Exception as exc:
                logger.warning(
                    "Removing unavailable simplification stream: %s",
                    exc,
                )
                if stream in active:
                    active.remove(stream)

    def _contractnli_texts(self, split: str, limit: int) -> Iterator[dict[str, Any]]:
        emitted, seen = 0, set()
        for row in self.contractnli_pairs(split=split):
            text = row["premise"].strip()
            if text in seen or len(text.split()) < 5:
                continue
            seen.add(text)
            yield {
                "source_id": row["source_id"],
                "document_id": row.get("document_id", row["source_id"]),
                "text": text,
                "dataset": self.settings.contractnli_dataset,
                "split": split,
            }
            emitted += 1
            if emitted >= limit:
                return


def _first_text(row: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _normalise_contractnli_row(row: dict[str, Any], config: str) -> Iterator[dict[str, Any]]:
    document_id = str(
        row.get("document_id") or row.get("id") or row.get("pid") or row.get("file_name") or "contractnli"
    )

    if "input" in row and "output" in row and "premise" not in row:
        raw_input = str(row.get("input") or "")
        raw_label = str(row.get("output") or "").strip().lower()
        premise, hypothesis = "", ""
        if "\n\nContext:" in raw_input:
            parts = raw_input.split("\n\nContext:", 1)
            hypothesis = parts[0].replace("Q:", "").strip()
            premise = parts[1].strip()
        elif "Context:" in raw_input:
            parts = raw_input.split("Context:", 1)
            hypothesis = parts[0].replace("Q:", "").strip()
            premise = parts[1].strip()
        else:
            hypothesis = raw_input[:300]
            premise = raw_input
        if "entail" in raw_label:
            label = "entailment"
        elif "contrad" in raw_label:
            label = "contradiction"
        else:
            label = "neutral"
        if premise and hypothesis:
            yield {
                "source_id": str(row.get("id") or row.get("pid") or f"{document_id}::{hypothesis[:30]}"),
                "document_id": document_id,
                "premise": premise,
                "hypothesis": hypothesis,
                "label": label,
                "evidence_chunk_id": None,
                "evidence": None,
                "dataset": "contract-nli",
                "config": config,
            }
        return

    premise = _first_text(row, "premise", "context", "document", "text")
    hypothesis = _first_text(row, "hypothesis", "statement", "claim")
    label = row.get("label") if row.get("label") is not None else row.get("choice") or row.get("gold_label")
    evidence = row.get("evidence") or row.get("evidence_text") or row.get("spans")
    if premise and hypothesis:
        yield {
            "source_id": str(row.get("id") or f"{document_id}::{hypothesis[:30]}"),
            "document_id": document_id,
            "premise": premise,
            "hypothesis": hypothesis,
            "label": _normalise_nli_label(label),
            "evidence_chunk_id": str(row.get("evidence_chunk_id") or "") or None,
            "evidence": evidence,
            "dataset": "contract-nli",
            "config": config,
        }
        return
    hypotheses = row.get("hypotheses") or row.get("claims")
    labels = row.get("labels") or row.get("choices")
    if premise and isinstance(hypotheses, list):
        for index, item in enumerate(hypotheses):
            if isinstance(item, dict):
                text = _first_text(item, "hypothesis", "text", "statement")
                item_label = item.get("label") if item.get("label") is not None else item.get("choice")
                item_evidence = item.get("evidence") or item.get("spans")
            else:
                text = str(item)
                item_label = labels[index] if isinstance(labels, list) and index < len(labels) else None
                item_evidence = None
            if text:
                yield {
                    "source_id": f"{document_id}::{index}",
                    "document_id": document_id,
                    "premise": premise,
                    "hypothesis": text,
                    "label": _normalise_nli_label(item_label),
                    "evidence_chunk_id": None,
                    "evidence": item_evidence,
                    "dataset": "contract-nli",
                    "config": config,
                }


def _normalise_nli_label(value: Any) -> str:
    if isinstance(value, int):
        return {0: "contradiction", 1: "entailment", 2: "neutral"}.get(value, "neutral")
    text = str(value or "neutral").lower().strip()
    if "entail" in text or text in {"yes", "true"}:
        return "entailment"
    if "contrad" in text or text in {"no", "false"}:
        return "contradiction"
    return "neutral"
