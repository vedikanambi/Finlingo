"""Frozen environment report for the submission manifest."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

from backend.app.core.config import Settings

_SECRET_MARKERS = ("key", "token", "secret", "password")


def _redacted_settings(settings: Settings) -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    for name, value in settings.model_dump().items():
        if any(marker in name.lower() for marker in _SECRET_MARKERS):
            snapshot[name] = "***set***" if value else None
        else:
            snapshot[name] = str(value) if isinstance(value, Path) else value
    return snapshot


def _git_commit(root: Path) -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except Exception:
        return None


def environment_report(settings: Settings, output_path: Path | None = None) -> dict[str, Any]:
    packages = sorted(f"{d.metadata['Name']}=={d.version}" for d in metadata.distributions() if d.metadata["Name"])
    gpu: dict[str, Any] = {"cuda_available": False}
    torch_version = None
    try:
        import torch

        torch_version = torch.__version__
        gpu["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            gpu.update(
                {
                    "device_name": props.name,
                    "vram_gb": round(props.total_memory / 1024**3, 2),
                    "cuda_version": torch.version.cuda,
                    "tf32_enabled": bool(settings.enable_tf32),
                }
            )
    except Exception as exc:
        gpu["error"] = str(exc)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch_version,
        "gpu": gpu,
        "git_commit": _git_commit(settings.project_root),
        "package_count": len(packages),
        "packages_sha256": hashlib.sha256("\n".join(packages).encode()).hexdigest(),
        "packages": packages,
        "dataset_revisions": settings.dataset_revisions,
        "model_revisions": settings.model_revisions,
        "settings": _redacted_settings(settings),
    }
    if output_path is not None:
        out = settings.resolve(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
