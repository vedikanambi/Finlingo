from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

from backend.app.core.config import Settings

_TRACKED_PACKAGES = (
    "torch",
    "transformers",
    "peft",
    "bitsandbytes",
    "datasets",
    "sentence-transformers",
    "faiss-cpu",
    "openai",
    "fastapi",
    "pydantic",
    "bert-score",
    "scikit-learn",
)


def runtime_manifest(settings: Settings) -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for name in _TRACKED_PACKAGES:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None

    hardware: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }
    try:
        import torch

        hardware.update(
            torch_version=torch.__version__,
            cuda_available=torch.cuda.is_available(),
            cuda_version=torch.version.cuda,
            gpu=(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
        )
    except Exception:
        hardware.update(torch_version=None, cuda_available=None, cuda_version=None, gpu=None)

    return {
        "python": sys.version.split()[0],
        "git_commit": _git_commit(settings),
        "random_seed": settings.random_seed,
        "dataset_revisions": settings.dataset_revisions,
        "model_revisions": settings.model_revisions,
        "packages": packages,
        "hardware": hardware,
    }


def sha256_path(path: Path | None) -> str | None:
    """Hash a file or adapter directory deterministically without exposing it."""
    if path is None or not path.exists():
        return None
    digest = hashlib.sha256()
    if path.is_file():
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(str(child.relative_to(path)).replace("\\", "/").encode("utf-8"))
        with child.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def adapter_sha256(settings: Settings, adapter_id: str | None) -> str | None:
    if not adapter_id:
        return None
    direct = Path(adapter_id).expanduser()
    if direct.exists():
        return sha256_path(direct.resolve())
    project_relative = settings.resolve(direct)
    if project_relative.exists():
        return sha256_path(project_relative)
    # no local bytes for a hub adapter, so just hash its pinned revision instead
    revision = settings.model_revision(adapter_id)
    return hashlib.sha256(f"{adapter_id}@{revision or 'UNPINNED'}".encode("utf-8")).hexdigest()


def _git_commit(settings: Settings) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=settings.project_root,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2,
        ).strip()
    except Exception:
        return None
