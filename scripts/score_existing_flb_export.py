"""Re-scores the frozen FLB export against the current pipeline (e.g. the
newly fine-tuned risk classifier) without rebuilding the corpus itself.
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.core.config import Settings
from backend.evaluation.evaluator import evaluate_records
from backend.evaluation.flb_builder import FLBRecord


def main() -> None:
    settings = Settings()
    export_path = settings.resolve(Path("reports/flb_achievable_export.json"))
    lines = export_path.read_text(encoding="utf-8").splitlines()
    field_names = {f.name for f in dataclasses.fields(FLBRecord)}
    records = []
    for line in lines:
        if not line.strip():
            continue
        payload = json.loads(line)
        filtered = {k: v for k, v in payload.items() if k in field_names}
        records.append(FLBRecord(**filtered))

    print(f"Loaded {len(records)} records from {export_path}")
    result = evaluate_records(settings, records, adjudicate_evidence=True, final_evaluation=False)

    output_path = settings.resolve(Path("reports/evaluation_rq2_trained.json"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
