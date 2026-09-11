"""checks the LoRA adapter is actually attached and doing something, not just loaded but ignored."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from backend.app.core.config import Settings
from backend.app.services.model_registry import ModelRegistry
from backend.app.services.nli_service import NLIService


def _hash_adapter_files(adapter_dir: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    if not adapter_dir.exists():
        return hashes
    for file in sorted(adapter_dir.rglob("*")):
        if file.is_file() and file.suffix in {".safetensors", ".bin", ".json"}:
            hashes[file.name] = hashlib.sha256(file.read_bytes()).hexdigest()
    return hashes


def _load_probe_pairs(settings: Settings) -> list[tuple[str, str]]:
    path = settings.resolve(settings.adapter_probe_pairs_path)
    if not path.exists():
        raise RuntimeError(f"Adapter probe pairs file not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    pairs = [(str(p["premise"]), str(p["hypothesis"])) for p in payload.get("pairs", [])]
    if len(pairs) < 5:
        raise RuntimeError("Adapter proof requires at least 5 probe pairs")
    return pairs


def prove_adapter_attachment(settings: Settings, output_path: Path | None = None) -> dict[str, Any]:
    if not settings.verifier_adapter:
        raise RuntimeError("VERIFIER_ADAPTER is not configured; nothing to prove.")
    pairs = _load_probe_pairs(settings)
    registry = ModelRegistry(settings)
    try:
        adapted = NLIService(
            registry, adapter_id=settings.verifier_adapter, require_adapter=True, use_verifier_calibration=False
        )
        adapted_scores = [
            s.entailment for s in adapted.score_pairs(pairs, settings.verifier_batch_size, settings.verifier_max_length)
        ]
        adapted_handle = adapted.handle

        peft_modules = [name for name, _ in adapted_handle.model.named_modules() if "lora" in name.lower()]
        peft_config: dict[str, Any] = {}
        peft_cfg_attr = getattr(adapted_handle.model, "peft_config", None)
        if peft_cfg_attr:
            first = next(iter(peft_cfg_attr.values()))
            peft_config = {k: v for k, v in first.to_dict().items() if isinstance(v, (str, int, float, bool, list))}
        registry.release_nli()

        base = NLIService(
            registry,
            model_id=settings.verifier_model,
            adapter_id=None,
            require_adapter=False,
            use_verifier_calibration=False,
        )
        base_scores = [
            s.entailment for s in base.score_pairs(pairs, settings.verifier_batch_size, settings.verifier_max_length)
        ]
    finally:
        registry.close()

    deltas = [round(a - b, 6) for a, b in zip(adapted_scores, base_scores)]
    max_abs_delta = max(abs(d) for d in deltas)
    adapter_dir = settings.resolve(settings.verifier_adapter)
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "adapter_id": settings.verifier_adapter,
        "base_model": settings.verifier_model,
        "structural_proof": {
            "lora_module_count": len(peft_modules),
            "lora_modules_sample": peft_modules[:8],
            "peft_config": peft_config,
            "adapter_file_sha256": _hash_adapter_files(adapter_dir),
        },
        "behavioural_proof": {
            "probe_pairs": len(pairs),
            "base_entailment": [round(x, 6) for x in base_scores],
            "adapter_entailment": [round(x, 6) for x in adapted_scores],
            "deltas": deltas,
            "max_abs_delta": round(max_abs_delta, 6),
            "min_required_delta": settings.adapter_proof_min_delta,
        },
        "attached": bool(peft_modules) and max_abs_delta >= settings.adapter_proof_min_delta,
    }
    if not result["attached"]:
        result["failure_reason"] = (
            "No LoRA modules found in the loaded verifier"
            if not peft_modules
            else f"Adapter outputs are indistinguishable from base (max |delta| "
            f"{max_abs_delta:.4f} < {settings.adapter_proof_min_delta})"
        )
    out = settings.resolve(output_path or settings.adapter_proof_output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if not result["attached"]:
        raise RuntimeError(f"Adapter attachment proof FAILED: {result['failure_reason']} (see {out})")
    return result
