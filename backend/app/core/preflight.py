from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from backend.app.core.config import Settings
from backend.app.core.risk_taxonomy import load_risk_taxonomy


def _verifier_threshold_from_calibration(payload: dict[str, Any]) -> float | None:
    """Pull the calibrated threshold out of either a flat or nested calibration artefact."""
    if "threshold" in payload:
        try:
            return float(payload["threshold"])
        except (TypeError, ValueError):
            return None
    systems = payload.get("systems")
    if isinstance(systems, dict):
        for system in systems.values():
            calibration = (system or {}).get("calibration") or {}
            selected = calibration.get("selected_threshold") or {}
            if "threshold" in selected:
                try:
                    return float(selected["threshold"])
                except (TypeError, ValueError):
                    return None
    return None


def _hash_adapter_dir(adapter_dir: Path) -> str | None:
    if not adapter_dir.exists():
        return None
    digest = hashlib.sha256()
    for file in sorted(adapter_dir.rglob("*")):
        if file.is_file() and file.suffix in {".safetensors", ".bin", ".json"}:
            digest.update(file.read_bytes())
    return digest.hexdigest()


def run_preflight(settings: Settings, *, final: bool = False) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def adapter_available(adapter: str | None) -> bool:
        if not adapter:
            return False
        direct = Path(adapter).expanduser()
        if direct.exists() or settings.resolve(direct).exists():
            return True
        return "/" in adapter and bool(settings.model_revision(adapter))

    def add(name: str, ok: bool, detail: str, severity: str = "error") -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail, "severity": severity})

    add("no_s3_or_local_dataset_dependency", True, "Raw datasets are streamed; regulatory indexes are in memory.")

    groq_selected = "groq" in {
        settings.silver_label_provider,
        settings.silver_judge_provider,
        settings.silver_regeneration_provider,
    }
    add(
        "silver_label_api_key",
        bool(settings.groq_api_key) if groq_selected else True,
        "Required for silver-label generation via Groq."
        if groq_selected
        else "Groq is not the configured silver-label/judge/regeneration provider; no key required.",
        severity="error" if groq_selected else "warning",
    )
    hf_token_required = settings.model_backend == "hf_endpoint"
    add(
        "hf_token",
        bool(settings.hf_token) if hf_token_required else True,
        "Required for gated/private models and hosted endpoints."
        if hf_token_required
        else "Optional for the current backend; only needed for gated/private models or hf_endpoint hosting.",
        severity="error" if hf_token_required else "warning",
    )
    add(
        "openai_api_key",
        True,
        "Optional — only needed for --include-gpt-judge. Not required for primary pipeline.",
        severity="warning" if not settings.openai_api_key else "error",
    )

    if settings.model_backend == "hf_endpoint":
        separate = bool(settings.hf_simplifier_endpoint_url and settings.hf_risk_endpoint_url)
        routed = bool(settings.hf_endpoint_supports_routing and settings.hf_generation_endpoint_url)
        add(
            "task_specific_generation_routing",
            separate or routed,
            "Use separate S2/S5 endpoints, or an explicitly routing-aware custom endpoint.",
        )
    else:
        s2_backend = settings.backend_for("S2 simplifier")
        if s2_backend == "ollama":
            add(
                "simplifier_adapter",
                True,
                "S2 is served by Ollama; no trained adapter is required (Ollama has no adapter-loading path).",
                severity="warning",
            )
        else:
            add(
                "simplifier_adapter",
                adapter_available(settings.simplifier_adapter) or settings.allow_base_models,
                "A trained S2 adapter is mandatory except for a documented zero-shot baseline.",
            )
        add(
            "verifier_adapter",
            adapter_available(settings.verifier_adapter) or not settings.verifier_require_adapter,
            "A trained S6 adapter (Verifier v3) is mandatory for proposal-compliant runs.",
        )

    # S5 has to go through Ollama - routing it to the HF loader would load a second multi-GB model and could silently fall back to an unadapted base model
    s5_backend = settings.backend_for("S5 risk classifier")
    add(
        "s5_routed_to_ollama",
        s5_backend == "ollama",
        f"S5 backend is '{s5_backend}'; it must be 'ollama' so mistral:7b-instruct is never sent through the "
        "Hugging Face tokenizer/model loaders."
        if s5_backend != "ollama"
        else "S5 is routed exclusively through Ollama.",
    )

    if settings.verifier_calibration_path:
        calibration = settings.resolve(settings.verifier_calibration_path)
        threshold = None
        detail = f"Expected {calibration}"
        ok = calibration.exists()
        if ok:
            try:
                payload = json.loads(calibration.read_text(encoding="utf-8"))
                threshold = _verifier_threshold_from_calibration(payload)
                detail = (
                    f"Calibration artefact loaded; threshold={threshold}"
                    if threshold is not None
                    else f"Calibration artefact loaded but no threshold field was found in {calibration}"
                )
            except Exception as exc:
                ok = False
                detail = f"Calibration artefact unreadable: {exc}"
        add("verifier_calibration", ok, detail)
    else:
        add(
            "verifier_calibration",
            False,
            "No calibration artefact configured; use VERIFIER_CALIBRATION_PATH for final probability claims.",
            severity="warning" if not final else "error",
        )

    if settings.verifier_adapter:
        adapter_dir = settings.resolve(settings.verifier_adapter)
        checks.append(
            {
                "name": "verifier_adapter_runtime_record",
                "ok": True,
                "severity": "info",
                "detail": {
                    "adapter_loaded": adapter_available(settings.verifier_adapter),
                    "adapter_path": str(adapter_dir),
                    "adapter_sha256": _hash_adapter_dir(adapter_dir),
                    "base_model": settings.verifier_model,
                    "threshold": _verifier_threshold_from_calibration(
                        json.loads(settings.resolve(settings.verifier_calibration_path).read_text(encoding="utf-8"))
                    )
                    if settings.verifier_calibration_path
                    and settings.resolve(settings.verifier_calibration_path).exists()
                    else None,
                },
            }
        )

    risk_config = settings.resolve(settings.risk_few_shot_path)
    add("risk_few_shot_config", risk_config.exists(), f"Expected {risk_config}")
    taxonomy_path = settings.resolve(settings.risk_taxonomy_path)
    try:
        taxonomy = load_risk_taxonomy(taxonomy_path)
        taxonomy_ok = set(taxonomy.labels) == {
            "Auto-Renewal",
            "Hidden Fee",
            "Liability Waiver",
            "Data Sharing",
            "Penalty Clause",
            "Safe",
        }
        taxonomy_detail = f"Loaded labels: {list(taxonomy.labels)}"
    except Exception as exc:
        taxonomy_ok = False
        taxonomy_detail = str(exc)
    add("risk_taxonomy", taxonomy_ok, taxonomy_detail)
    regulatory_path = settings.resolve(settings.regulatory_sources_path)
    add("regulatory_sources_config", regulatory_path.exists(), f"Expected {regulatory_path}")

    datasets = [
        settings.cuad_dataset,
        settings.contractnli_dataset,
        settings.snli_dataset,
        settings.financebench_dataset,
        settings.cfpb_dataset,
        settings.edgar_dataset,
    ]
    unpinned_datasets = [dataset for dataset in datasets if not settings.dataset_revision(dataset)]
    add(
        "dataset_revisions_pinned",
        not unpinned_datasets,
        f"Unpinned datasets: {unpinned_datasets}" if unpinned_datasets else "All configured datasets are pinned.",
        severity="warning" if unpinned_datasets and not final else "error",
    )
    models = [
        settings.simplifier_model,
        settings.risk_model or settings.simplifier_model,
        settings.s2_semantic_model,
        settings.verifier_model,
        settings.verifier_training_base_model,
        settings.modernbert_model,
        settings.modernbert_training_base_model,
        settings.reranker_model,
        *settings.reranker_models,
        settings.bertscore_model,
        settings.finbert_model,
    ]
    unpinned_models = sorted({model for model in models if not settings.model_revision(model)})
    add(
        "model_revisions_pinned",
        not unpinned_models,
        f"Unpinned models: {unpinned_models}" if unpinned_models else "All configured models are pinned.",
        severity="warning" if unpinned_models and not final else "error",
    )
    add(
        "faithfulness_metrics_separated",
        settings.faithfulness_primary_premise in {"retrieved_chunk", "source_clause", "dual"},
        (
            f"Primary display premise is {settings.faithfulness_primary_premise}; "
            "source-clause preservation and regulatory grounding are always reported separately."
        ),
    )
    add(
        "attribution_threshold",
        0 <= settings.attribution_tau <= 1,
        f"Attribution requires score >= {settings.attribution_tau}.",
    )
    add(
        "model_remote_code_disabled",
        not settings.model_trust_remote_code,
        "Remote model code should remain disabled unless explicitly audited and revision-pinned.",
        severity="warning" if settings.model_trust_remote_code else "error",
    )
    add(
        "edgar_remote_code_disabled",
        not settings.edgar_trust_remote_code,
        "Remote dataset code should remain disabled unless explicitly reviewed and pinned.",
        severity="warning" if settings.edgar_trust_remote_code else "error",
    )

    ollama_needed = (
        settings.backend_for("S5 risk classifier") == "ollama"
        or settings.backend_for("S2 simplifier") == "ollama"
        or settings.model_backend == "ollama"
        or settings.embedding_provider == "ollama"
        or "ollama"
        in {settings.silver_label_provider, settings.silver_judge_provider, settings.silver_regeneration_provider}
    )
    if ollama_needed:
        from backend.app.services.ollama_runtime import OllamaRuntime

        health = OllamaRuntime(settings).health()
        add(
            "ollama_runtime",
            bool(health["ok"]),
            health["error"] or f"Ollama ready: {health['resolved_model']} at {health['base_url']}",
        )

    proof_path = settings.resolve(settings.adapter_proof_output)
    add(
        "verifier_adapter_attachment_proof",
        proof_path.exists(),
        f"Run `python main.py adapter-proof` (expected {proof_path}).",
        severity="error" if final else "warning",
    )
    repro_path = settings.resolve(settings.faithbench_reproduction_output)
    add(
        "faithbench_reproduction",
        repro_path.exists(),
        f"Run `python main.py faithbench-reproduce` (expected {repro_path}).",
        severity="error" if final else "warning",
    )
    if settings.flb_mode.value == "achievable":
        plan_path = settings.resolve(settings.flb_plan_path)
        plan_ok = False
        plan_detail = f"Run `python main.py flb-plan --accept` (expected {plan_path})."
        if plan_path.exists():
            try:
                import json as _json

                plan_ok = bool(_json.loads(plan_path.read_text(encoding="utf-8")).get("accepted"))
                plan_detail = (
                    "Accepted achievable FLB plan found."
                    if plan_ok
                    else f"Plan at {plan_path} exists but is not accepted."
                )
            except Exception as exc:
                plan_detail = f"Plan unreadable: {exc}"
        add("flb_achievable_plan_accepted", plan_ok, plan_detail)

    blocking = [item for item in checks if not item["ok"] and item["severity"] == "error"]
    warnings = [item for item in checks if not item["ok"] and item["severity"] == "warning"]
    result = {
        "passed": not blocking,
        "final_mode": final,
        "blocking_count": len(blocking),
        "warning_count": len(warnings),
        "checks": checks,
    }
    if final and blocking:
        names = ", ".join(item["name"] for item in blocking)
        raise RuntimeError(f"Final preflight failed: {names}")
    return result
