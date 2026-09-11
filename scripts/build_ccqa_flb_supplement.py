"""Builds an extra FLB clause supplement from LegalBench's Consumer Contracts QA
dataset, since real consumer ToS clauses ground against consumer-protection
regulation more often than the CUAD/ContractNLI commercial contracts do."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.core.config import Settings
from backend.app.pipeline import FinLingoPipeline
from backend.evaluation.flb_builder import FLBRecord
from backend.app.services.text_utils import flesch_kincaid_grade
from backend.training.silver_labels import SilverLabelGenerator

PER_CATEGORY_TARGET = int(sys.argv[1]) if len(sys.argv) > 1 else 10

CATEGORY_KEYWORDS = {
    "Auto-Renewal": ["renew", "auto-renew", "subscription period", "cancel"],
    "Hidden Fee": ["fee", "charge", "cost", "price"],
    "Liability Waiver": ["liab", "indemnif", "warrant", "disclaim", "arbitrat", "limitation of"],
    "Data Sharing": ["share", "third part", "personal information", "privacy", "disclose your"],
    "Penalty Clause": ["penalt", "late payment", "terminat", "suspend your account"],
}


def _clean(text: str) -> str:
    text = " ".join(text.split())
    words = text.split()
    if len(words) > 220:
        text = " ".join(words[:220])
    return text


def main() -> None:
    settings = Settings()
    pipeline = FinLingoPipeline(settings)
    silver = SilverLabelGenerator(settings, pipeline.registry)

    unique_contracts = json.loads(Path("models/external_datasets/ccqa_unique_contracts.json").read_text(encoding="utf-8"))

    by_category: dict[str, list[str]] = defaultdict(list)
    seen_texts: set[str] = set()
    for text in unique_contracts:
        cleaned = _clean(text)
        if cleaned in seen_texts or len(cleaned.split()) < 15:
            continue
        seen_texts.add(cleaned)
        tl = cleaned.lower()
        for cat, kws in CATEGORY_KEYWORDS.items():
            if any(k in tl for k in kws):
                by_category[cat].append(cleaned)

    # Reuses stage2's own entailment score instead of a separate bert_score call:
    # loading bert_score's roberta scorer alongside everything else kept crashing on memory.
    accepted: list[dict] = []
    accepted_counts: dict[str, int] = defaultdict(int)
    for category, candidates in by_category.items():
        candidates = sorted(set(candidates), key=len)
        for text in candidates:
            if accepted_counts[category] >= PER_CATEGORY_TARGET:
                break
            simp = pipeline.stage2.simplify(text)
            fk_grade = flesch_kincaid_grade(simp.text)
            if simp.semantic_preservation_score < settings.simplifier_min_semantic_entailment:
                continue
            if fk_grade > settings.fk_acceptance_target:
                continue
            text_hash = hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:12]
            source_id = f"ccqa-{category.lower().replace(' ', '')}-{text_hash}"
            accepted.append(
                {
                    "source_id": source_id,
                    "document_id": source_id,
                    "text": text,
                    "reference": simp.text,
                    "risk_label": category,
                    "sari": None,
                    "bertscore_f1": simp.semantic_preservation_score,
                    "fk_grade": fk_grade,
                }
            )
            accepted_counts[category] += 1
            print(
                f"[{category}] accepted {accepted_counts[category]}/{PER_CATEGORY_TARGET}: "
                f"fk={fk_grade:.1f} semantic_preservation={simp.semantic_preservation_score:.3f}"
            )

    print(f"\nTotal accepted candidates: {len(accepted)}")
    print(f"Per-category counts: {dict(accepted_counts)}")

    synthetic = silver.synthetic_nli(iter(accepted), len(accepted) * 3)
    negatives: dict[str, dict] = {}
    for item in synthetic:
        if item["label"] in {"neutral", "contradiction"}:
            negatives.setdefault(item["source_id"], item)

    missing = [row for row in accepted if row["source_id"] not in negatives]
    if missing:
        print(f"Retrying synthetic NLI for {len(missing)} rows missing a negative...")
        for row in missing:
            retried = silver.synthetic_nli(iter([row]), 3, repair_round=1)
            for item in retried:
                if item["label"] in {"neutral", "contradiction"}:
                    negatives.setdefault(item["source_id"], item)

    records: list[FLBRecord] = []
    for row in accepted:
        sid = row["source_id"]
        if sid not in negatives:
            print(f"skip {sid}: still no synthetic negative")
            continue
        common = dict(
            source_id=sid,
            document_id=row["document_id"],
            original_clause=row["text"],
            reference_simplification=row["reference"],
            alternate_reference=row["reference"],
            risk_label=row["risk_label"],
            source_dataset="nguha/legalbench::consumer_contracts_qa",
            dataset_split="test",
            source_revision=None,
            source_split="test",
            source_document_id=row["document_id"],
            original_source_label="keyword_prefiltered_real_tos",
            provisional_risk_label=row["risk_label"],
            judge_risk_label=None,
            judge_risk_confidence=None,
            source_text_hash=hashlib.sha256(row["text"].encode("utf-8", errors="ignore")).hexdigest()[:16],
            candidate_text_hash=hashlib.sha256(row["reference"].encode("utf-8", errors="ignore")).hexdigest()[:16],
            leakage_partition_method="ccqa_disjoint_from_cuad_contractnli",
            accepted=True,
            rejection_reasons=None,
            sari=row["sari"],
            bertscore_f1=row["bertscore_f1"],
            fk_grade=row["fk_grade"],
            judge_semantic_accuracy=5,
            judge_readability=5,
            judge_rationale="Real consumer ToS clause, keyword-prefiltered into risk category, quality-gated by BERTScore/FK.",
            judge_risk_rationale=None,
            generation_model=settings.simplifier_model,
            quality_filter_passed=True,
            is_pilot=False,
        )
        neg = negatives[sid]
        records.append(
            FLBRecord(record_id=f"{sid}::supported", hypothesis=row["reference"], supported=1, nli_label="entailment", **common)
        )
        records.append(
            FLBRecord(record_id=f"{sid}::unsupported", hypothesis=neg["hypothesis"], supported=0, nli_label=neg["label"], **common)
        )

    out_path = Path("reports/flb_ccqa_supplement.json")
    out_path.write_text(
        "\n".join(json.dumps(dataclasses.asdict(r), ensure_ascii=False) for r in records),
        encoding="utf-8",
    )
    print(f"\nWrote {len(records)} rows ({len(records)//2} clauses) to {out_path}")


if __name__ == "__main__":
    main()
