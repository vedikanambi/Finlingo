from __future__ import annotations

import json
from pathlib import Path

from backend.app.core.config import Settings
from backend.app.core.provenance import runtime_manifest
from backend.app.services.dataset_streams import StreamingDatasetRepository
from backend.app.services.model_registry import ModelRegistry
from backend.app.services.nli_service import NLIService


def calibrate_verifier(
    settings: Settings,
    *,
    max_examples: int = 2_000,
    output_path: Path = Path("models/stage6_calibration.json"),
    dataset: str = "contractnli",
) -> dict:
    """Fit one temperature on an untouched NLI validation stream."""
    import torch
    import torch.nn.functional as F

    repository = StreamingDatasetRepository(settings)
    try:
        if dataset == "contractnli":
            rows = list(repository.contractnli_pairs(split="validation", limit=max_examples))
        elif dataset == "snli":
            rows = list(repository.snli_pairs(split="validation", limit=max_examples))
        else:
            raise ValueError("dataset must be 'contractnli' or 'snli'")
    finally:
        repository.close()
    if not rows:
        raise RuntimeError("No validation examples were available for verifier calibration")

    registry = ModelRegistry(settings)
    service = NLIService(registry, require_adapter=settings.verifier_require_adapter)
    handle = service.handle
    label_to_index = {
        "entailment": handle.entailment_index,
        "neutral": handle.neutral_index,
        "contradiction": handle.contradiction_index,
    }
    if any(value is None for value in label_to_index.values()):
        raise RuntimeError(f"Verifier is not a three-way NLI model: {label_to_index}")

    logits_batches = []
    label_batches = []
    try:
        for start in range(0, len(rows), settings.verifier_batch_size):
            batch = rows[start : start + settings.verifier_batch_size]
            encoded = handle.tokenizer(
                [row["premise"] for row in batch],
                [row["hypothesis"] for row in batch],
                padding=True,
                truncation=True,
                max_length=settings.verifier_max_length,
                return_tensors="pt",
            )
            device = next(handle.model.parameters()).device
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.inference_mode():
                logits_batches.append(handle.model(**encoded).logits.float().cpu())
            label_batches.append(torch.tensor([int(label_to_index[row["label"]]) for row in batch], dtype=torch.long))
        logits = torch.cat(logits_batches)
        labels = torch.cat(label_batches)
        before_nll = float(F.cross_entropy(logits, labels))
        log_temperature = torch.nn.Parameter(torch.zeros(()))
        optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=100)

        def closure():
            optimizer.zero_grad()
            temperature = torch.exp(log_temperature).clamp(min=0.05, max=20.0)
            loss = F.cross_entropy(logits / temperature, labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        temperature = float(torch.exp(log_temperature).clamp(min=0.05, max=20.0).detach())
        after_nll = float(F.cross_entropy(logits / temperature, labels))
        probabilities = torch.softmax(logits / temperature, dim=-1)[:, int(handle.entailment_index)]
        supported = labels == int(handle.entailment_index)
        threshold_metrics = {}
        for step in range(20, 96):
            threshold = step / 100
            predicted = probabilities >= threshold
            true_positive = int((predicted & supported).sum())
            false_positive = int((predicted & ~supported).sum())
            false_negative = int((~predicted & supported).sum())
            precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
            recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            threshold_metrics[f"{threshold:.2f}"] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "predicted_supported": int(predicted.sum()),
            }
        eligible = [
            (float(threshold), metrics)
            for threshold, metrics in threshold_metrics.items()
            if metrics["precision"] >= 0.90
        ]
        best_unconstrained_threshold, best_unconstrained_metrics = max(
            [(float(t), m) for t, m in threshold_metrics.items()],
            key=lambda item: (item[1]["recall"], item[1]["f1"], -item[0]),
        )
        if eligible:
            selected_threshold, selected_metrics = max(
                eligible,
                key=lambda item: (item[1]["recall"], item[1]["f1"], -item[0]),
            )
        else:
            selected_threshold, selected_metrics = None, None
        payload = {
            "temperature": temperature,
            "nll_before": before_nll,
            "nll_after": after_nll,
            "examples": len(rows),
            "dataset": dataset,
            "split": "validation",
            "model": settings.verifier_model,
            "adapter": settings.verifier_adapter,
            "threshold_selection": {
                "objective": "maximize supported recall subject to precision >= 0.90",
                "selected_threshold": selected_threshold,
                "selected_metrics": selected_metrics,
                "precision_constraint_met": bool(eligible),
                "best_unconstrained_threshold": best_unconstrained_threshold,
                "best_unconstrained_metrics": best_unconstrained_metrics,
                "deployment_note": (
                    "No threshold is recommended when the precision constraint is unmet; "
                    "retain the existing deployment policy pending an in-domain calibration set."
                    if not eligible
                    else "Selected on held-out validation data."
                ),
                "sweep": threshold_metrics,
            },
            "runtime_manifest": runtime_manifest(settings),
            "note": (
                "Temperature was fitted on held-out NLI validation data. For the strongest thesis claim, also report "
                "calibration on the independently human-adjudicated FLB subset."
            ),
        }
        output = settings.resolve(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload
    finally:
        registry.close()
