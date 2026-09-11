from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
import numpy as np

from backend.app.core.config import Settings
from backend.app.services.dataset_streams import StreamingDatasetRepository
from backend.evaluation.risk_taxonomy import RiskTaxonomy


def run_streaming_eda(settings: Settings, sample_size: int = 2000, output_path: Path | None = None) -> dict:
    repo = StreamingDatasetRepository(settings)
    taxonomy = RiskTaxonomy(settings.project_root / "configs" / "risk_taxonomy.yaml")
    sources = {
        "cuad": repo.cuad_clauses(limit=sample_size),
        "contractnli": repo.contractnli_pairs(limit=sample_size),
        "cfpb": repo.cfpb_narratives(limit=sample_size),
        "edgar": repo.edgar_texts(limit=sample_size),
    }
    result = {}
    try:
        for name, iterator in sources.items():
            lengths, labels, documents = [], Counter(), set()
            risk_labels, products, issues, years = Counter(), Counter(), Counter(), Counter()
            for row in iterator:
                text = row.get("text") or row.get("premise") or ""
                lengths.append(len(str(text).split()))
                documents.add(str(row.get("document_id") or row.get("source_id")))
                if row.get("label"):
                    labels[str(row["label"])] += 1
                mapped = taxonomy.map_question(str(row.get("question") or ""))
                if mapped:
                    risk_labels[mapped] += 1
                if row.get("product"):
                    products[str(row["product"])] += 1
                if row.get("issue"):
                    issues[str(row["issue"])] += 1
                if row.get("year") is not None:
                    years[str(row["year"])] += 1
            result[name] = {
                "sample_count": len(lengths),
                "document_count": len(documents),
                "word_length": {
                    "mean": float(np.mean(lengths)) if lengths else None,
                    "median": float(np.median(lengths)) if lengths else None,
                    "p95": float(np.quantile(lengths, 0.95)) if lengths else None,
                    "min": min(lengths) if lengths else None,
                    "max": max(lengths) if lengths else None,
                },
                "label_distribution": dict(labels),
                "mapped_risk_distribution": dict(risk_labels),
                "product_distribution": dict(products.most_common(20)),
                "issue_distribution": dict(issues.most_common(20)),
                "year_distribution": dict(sorted(years.items())),
                "raw_records_persisted": False,
            }
    finally:
        repo.close()
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
