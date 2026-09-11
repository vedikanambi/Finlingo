from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backend.app.core.config import Settings


@dataclass(frozen=True)
class ResultRow:
    section: str
    artifact: str
    metric: str
    value: str
    target: str
    status: str
    source: str


def consolidate_results(settings: Settings, output_dir: Path) -> dict[str, Any]:
    """builds the results table from whatever artifacts actually exist - missing ones show up as missing, not skipped."""
    output = settings.resolve(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows: list[ResultRow] = []

    adapters = {
        "S2 domain warm-up": settings.resolve(settings.domain_warmup_adapter or settings.domain_warmup_adapter_output),
        "S2 simplifier": settings.resolve(settings.simplifier_adapter_output),
        "S6 DeBERTa": settings.resolve(settings.verifier_adapter_output),
        "S6 DeBERTa NLI verifier": settings.resolve(Path("models/stage6_qlora_adapter")),
    }
    for name, directory in adapters.items():
        manifest = directory / "training_manifest.json"
        if manifest.exists():
            payload = _read_json(manifest)
            evidence = payload.get("training_evidence") or {}
            validation = evidence.get("final_validation") or {}
            rows.extend(
                [
                    _row("training", name, "status", "complete", "complete", manifest),
                    _row("training", name, "base_model", payload.get("base_model"), "declared model", manifest),
                    _row("training", name, "train_examples", payload.get("train_examples"), "> 0", manifest),
                    _row("training", name, "validation_examples", payload.get("validation_examples"), "> 0", manifest),
                    _row("training", name, "global_step", evidence.get("global_step"), "> 0", manifest),
                    _row("training", name, "train_loss", _last_log_value(evidence, "train_loss"), "reported", manifest),
                    _row("training", name, "eval_loss", validation.get("eval_loss"), "reported", manifest),
                    _row(
                        "training",
                        name,
                        "cuda_peak_allocated_gib",
                        (evidence.get("cuda_peak_memory") or {}).get("allocated_gib"),
                        "<= hardware",
                        manifest,
                    ),
                ]
            )
        else:
            rows.append(_row("training", name, "status", "missing", "complete", manifest))

    evaluation_files = {
        "FLB evaluation": settings.resolve(Path("reports/evaluation.json")),
        "Ablations": settings.resolve(Path("reports/ablations.json")),
        "Multi-verifier": settings.resolve(Path("reports/multiverifier.json")),
        "Reranker comparison": settings.resolve(Path("reports/reranker_comparison.json")),
        "Retrieval-query ablation": settings.resolve(Path("reports/retrieval_query_ablation.json")),
        "Failure analysis": settings.resolve(Path("reports/failure_analysis.json")),
        "FinanceBench": settings.resolve(Path("reports/financebench.json")),
    }
    for name, path in evaluation_files.items():
        if not path.exists():
            rows.append(_row("evaluation", name, "status", "missing", "complete", path))
            continue
        payload = _read_json(path)
        rows.append(_row("evaluation", name, "status", "complete", "complete", path))
        for metric, value in _flatten(payload):
            if isinstance(value, (str, int, float, bool)) or value is None:
                rows.append(_row("evaluation", name, metric, value, _target_for(metric, settings), path))

    csv_path = output / "consolidated_results.csv"
    md_path = output / "consolidated_results.md"
    _write_csv(csv_path, rows)
    _write_markdown(md_path, rows)
    summary = {
        "rows": len(rows),
        "complete_artifacts": sum(row.metric == "status" and row.value == "complete" for row in rows),
        "missing_artifacts": sum(row.metric == "status" and row.value == "missing" for row in rows),
        "csv": str(csv_path),
        "markdown": str(md_path),
    }
    (output / "consolidated_results_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _last_log_value(evidence: dict[str, Any], key: str) -> Any:
    for item in reversed(evidence.get("log_history") or []):
        if key in item:
            return item[key]
    return None


def _flatten(value: Any, prefix: str = ""):
    if isinstance(value, dict):
        for key, child in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            yield from _flatten(child, name)
    elif isinstance(value, list):
        if len(value) <= 8 and all(not isinstance(item, (dict, list)) for item in value):
            yield prefix, json.dumps(value, ensure_ascii=False)
    else:
        yield prefix, value


def _target_for(metric: str, settings: Settings) -> str:
    name = metric.lower()
    if name.endswith("sari"):
        return "> 40"
    if "bertscore" in name and name.endswith(("f1", "mean")):
        return "> 0.88"
    if "fk" in name and "grade" in name:
        return "<= 9"
    if "risk" in name and name.endswith("macro_f1"):
        return f"> {settings.target_risk_macro_f1}"
    if "faithfulness" in name and name.endswith("precision"):
        return f"> {settings.target_faithfulness_precision}"
    if "faithfulness" in name and name.endswith("recall"):
        return f"> {settings.target_faithfulness_recall}"
    if "attribution" in name and name.endswith("accuracy"):
        return f"> {settings.target_attribution_accuracy}"
    if "hallucination" in name and name.endswith("rate"):
        return f"< {settings.target_hallucination_rate}"
    return "reported"


def _row(section: str, artifact: str, metric: str, value: Any, target: str, source: Path) -> ResultRow:
    rendered = "" if value is None else str(value)
    status = _compare(rendered, target)
    return ResultRow(section, artifact, metric, rendered, target, status, str(source))


def _compare(value: str, target: str) -> str:
    if value == "missing":
        return "missing"
    if target == "complete":
        return "pass" if value == "complete" else "fail"
    try:
        number = float(value)
        operator, threshold = target.split(maxsplit=1)
        expected = float(threshold)
        if operator == ">":
            return "pass" if number > expected else "fail"
        if operator == "<":
            return "pass" if number < expected else "fail"
        if operator == "<=":
            return "pass" if number <= expected else "fail"
    except (ValueError, TypeError):
        pass
    return "reported" if value else "missing"


def _write_csv(path: Path, rows: list[ResultRow]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(ResultRow.__annotations__.keys())
        for row in rows:
            writer.writerow(row.__dict__.values())


def _write_markdown(path: Path, rows: list[ResultRow]) -> None:
    headers = list(ResultRow.__annotations__.keys())
    lines = [
        "# FinLingo++ consolidated results",
        "",
        "Generated only from persisted training/evaluation artifacts; missing work remains explicit.",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        values = [str(value).replace("|", "\\|").replace("\n", " ") for value in row.__dict__.values()]
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
