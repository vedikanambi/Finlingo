"""Builds a (chunk, hypothesis, grounded-or-not) training set for calibrating the
RQ3 faithfulness threshold, using the judge model as ground truth and the base
verifier's raw NLI scores as features."""

from __future__ import annotations

import dataclasses
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.core.config import Settings
from backend.app.pipeline import FinLingoPipeline
from backend.evaluation.evaluator import _provisional_evidence
from backend.evaluation.flb_builder import FLBRecord

TOP_K_FAST = 5


def main() -> None:
    export_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("reports/flb_ccqa_supplement.json")
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("reports/verifier_calibration_labels.jsonl")
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else 10_000

    settings = Settings()
    pipeline = FinLingoPipeline(settings)

    lines = export_path.read_text(encoding="utf-8").splitlines()
    field_names = {f.name for f in dataclasses.fields(FLBRecord)}
    records = []
    for line in lines:
        payload = json.loads(line)
        filtered = {k: v for k, v in payload.items() if k in field_names}
        records.append(FLBRecord(**filtered))
    records = records[:limit]

    out_f = out_path.open("a", encoding="utf-8")
    for i, record in enumerate(records, 1):
        started = time.time()
        query = f"LEGAL CLAUSE:\n{record.original_clause}\n\nPLAIN-LANGUAGE INTERPRETATION:\n{record.hypothesis}"
        retrieved = pipeline.stage3.retrieve(query, "bm25", original_clause=record.original_clause)
        evidence = retrieved[:TOP_K_FAST]
        if not evidence:
            print(f"[{i}/{len(records)}] {record.record_id}: no evidence retrieved, skipping")
            continue
        decision = _provisional_evidence(settings, pipeline.registry, record, evidence)
        chunk = next((c for c in evidence if c.chunk_id == decision.chunk_id), None)
        if chunk is None:
            print(f"[{i}/{len(records)}] {record.record_id}: judge picked no valid chunk, skipping")
            continue
        score = pipeline.stage6.nli.score_pairs(
            [(chunk.text, record.hypothesis)], settings.verifier_batch_size, settings.verifier_max_length
        )[0]
        row = {
            "record_id": record.record_id,
            "risk_label": record.risk_label,
            "judge_supported": int(decision.supported),
            "entailment": score.entailment,
            "neutral": score.neutral,
            "contradiction": score.contradiction,
            "premise": chunk.text,
            "hypothesis": record.hypothesis,
        }
        out_f.write(json.dumps(row) + "\n")
        out_f.flush()
        elapsed = time.time() - started
        print(
            f"[{i}/{len(records)}] {record.record_id}: judge_supported={decision.supported} "
            f"entailment={score.entailment:.4f} ({elapsed:.1f}s)"
        )
    out_f.close()
    print(f"Wrote labels to {out_path}")


if __name__ == "__main__":
    main()
