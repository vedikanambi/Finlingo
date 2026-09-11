"""Runs the RQ4 failure analysis against an already-built FLB export instead
of letting it rebuild the benchmark from scratch, which takes hours.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.core.config import Settings
from backend.evaluation.flb_builder import FLBRecord
from backend.experiments.failure_analysis import run_failure_analysis

N_CLAUSES = 10


def main() -> None:
    export_path = Path("reports/benchmark_dataset/flb_final_check3_export.json")
    field_names = {f.name for f in dataclasses.fields(FLBRecord)}
    seen_source_ids: set[str] = set()
    records: list[FLBRecord] = []
    for line in export_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if payload["source_id"] in seen_source_ids:
            continue
        if len(seen_source_ids) >= N_CLAUSES:
            break
        seen_source_ids.add(payload["source_id"])
        filtered = {k: v for k, v in payload.items() if k in field_names}
        records.append(FLBRecord(**filtered))

    print(f"Using {len(records)} unique clauses from {export_path}")

    settings = Settings()
    result = run_failure_analysis(settings, max_examples=N_CLAUSES, output_path=None, records=records)

    out_path = Path("reports/failure_analysis_rq4.json")
    out_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {out_path}")

    for condition, data in result.items():
        if not isinstance(data, dict) or "pipeline" not in data:
            continue
        p = data["pipeline"]
        delta = p.get("delta_vs_clean", {})
        print(f"\n=== {condition} ===")
        print(f"risk_macro_f1={p['risk'].get('macro_f1'):.3f} (delta {delta.get('risk_macro_f1')})")
        print(f"mean_fk_grade={p.get('mean_fk_grade')}")
        print(f"mean_source_faithfulness={p.get('mean_generated_source_faithfulness')}")


if __name__ == "__main__":
    main()
