"""works out a realistic per-class FLB quota when CUAD can't fully supply the balanced target, then freezes it once accepted."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.app.core.config import Settings
from backend.evaluation.flb_builder import FLBBuilder


def _taxonomy_sha(settings: Settings) -> str:
    path = settings.resolve(settings.risk_taxonomy_path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_flb_plan(settings: Settings, *, accept: bool = False, output_path: Path | None = None) -> dict[str, Any]:
    audit = FLBBuilder.feasibility_audit(settings, settings.flb_size)
    labels = list(settings.configured_risk_labels)
    balanced = dict(audit["quotas"])
    counts = dict(audit["class_counts"])
    multiplier = settings.flb_candidate_multiplier

    achievable: dict[str, int] = {}
    limited: dict[str, dict[str, int]] = {}
    for label in labels:
        supply = int(counts.get(label, 0))
        supply_limited_quota = supply // max(multiplier, 1)
        quota = min(int(balanced[label]), supply_limited_quota)
        if quota < int(balanced[label]):
            limited[label] = {"available": supply, "balanced_quota": int(balanced[label]), "achievable_quota": quota}
        achievable[label] = quota

    infeasible = {label: q for label, q in achievable.items() if q < settings.flb_plan_min_per_class}
    plan: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": settings.random_seed,
        "flb_balanced_target": settings.flb_size,
        "candidate_multiplier": multiplier,
        "taxonomy_sha256": _taxonomy_sha(settings),
        "dataset_revisions": settings.dataset_revisions,
        "class_counts_at_planning": counts,
        "balanced_quotas": balanced,
        "achievable_quotas": achievable,
        "achievable_total": sum(achievable.values()),
        "supply_limited_classes": limited,
        "min_per_class": settings.flb_plan_min_per_class,
        "plan_feasible": not infeasible,
        "infeasible_classes": infeasible,
        "accepted": bool(accept) and not infeasible,
        "note": (
            "Accepted achievable plan; FLB_MODE=achievable builds exactly these "
            "quotas (FLB-Silver). FLB-Gold is the human-adjudicated review of "
            "these records via the evaluate --review-file workflow."
            if accept and not infeasible
            else "Proposal only. Review quotas, then rerun `python main.py flb-plan --accept`."
        ),
    }
    if accept and infeasible:
        plan["note"] = (
            f"REFUSED to accept: classes below FLB_PLAN_MIN_PER_CLASS="
            f"{settings.flb_plan_min_per_class}: {sorted(infeasible)}. Expand external "
            "candidates (configs/flb_external_*.yaml) or lower the floor explicitly."
        )
    out = settings.resolve(output_path or settings.flb_plan_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    return plan


def load_accepted_plan(settings: Settings) -> dict[str, Any]:
    path = settings.resolve(settings.flb_plan_path)
    if not path.exists():
        raise RuntimeError(
            f"FLB_MODE=achievable requires an accepted plan at {path}. Run `python main.py flb-plan` then `--accept`."
        )
    plan = json.loads(path.read_text(encoding="utf-8"))
    if not plan.get("accepted"):
        raise RuntimeError(f"FLB plan at {path} has not been accepted (`flb-plan --accept`).")
    if plan.get("seed") != settings.random_seed:
        raise RuntimeError("FLB plan seed differs from RANDOM_SEED; regenerate the plan.")
    if plan.get("taxonomy_sha256") != _taxonomy_sha(settings):
        raise RuntimeError("Risk taxonomy changed since the FLB plan was accepted; regenerate it.")
    quotas = {str(k): int(v) for k, v in plan["achievable_quotas"].items()}
    if set(quotas) != set(settings.configured_risk_labels):
        raise RuntimeError("FLB plan labels do not match the configured taxonomy.")
    return plan
