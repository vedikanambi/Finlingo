from __future__ import annotations

from typing import Any


def reset_cuda_peak(torch_module) -> None:
    """Reset peak memory tracking before a run."""
    if torch_module.cuda.is_available():
        torch_module.cuda.empty_cache()
        torch_module.cuda.reset_peak_memory_stats()
        torch_module.cuda.synchronize()


def collect_training_evidence(trainer, torch_module) -> dict[str, Any]:
    """Collect final validation, trainer history, and peak CUDA memory."""
    if torch_module.cuda.is_available():
        torch_module.cuda.synchronize()
        peak_allocated = int(torch_module.cuda.max_memory_allocated())
        peak_reserved = int(torch_module.cuda.max_memory_reserved())
    else:
        peak_allocated = None
        peak_reserved = None
    evaluation = trainer.evaluate()
    return {
        "global_step": int(trainer.state.global_step),
        "epoch": float(trainer.state.epoch) if trainer.state.epoch is not None else None,
        "best_metric": trainer.state.best_metric,
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
        "final_validation": {key: _json_number(value) for key, value in evaluation.items()},
        "log_history": [{key: _json_number(value) for key, value in row.items()} for row in trainer.state.log_history],
        "cuda_peak_memory": {
            "measurement": "peak after reset immediately before trainer.train; model-load memory excluded",
            "allocated_bytes": peak_allocated,
            "reserved_bytes": peak_reserved,
            "allocated_gib": round(peak_allocated / 1024**3, 4) if peak_allocated is not None else None,
            "reserved_gib": round(peak_reserved / 1024**3, 4) if peak_reserved is not None else None,
        },
    }


def _json_number(value: Any) -> Any:
    try:
        if hasattr(value, "item"):
            return value.item()
    except Exception:
        pass
    return value
