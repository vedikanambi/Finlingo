from __future__ import annotations

import json
from dataclasses import dataclass

from backend.app.services.model_registry import ModelRegistry


@dataclass
class NLIScore:
    entailment: float
    contradiction: float | None = None
    neutral: float | None = None


class NLIService:
    """Lazy, temperature-calibrated NLI scorer."""

    def __init__(
        self,
        registry: ModelRegistry,
        model_id: str | None = None,
        adapter_id: str | None = None,
        *,
        require_adapter: bool | None = None,
        use_verifier_calibration: bool = True,
    ) -> None:
        self.registry = registry
        self.model_id = model_id
        self.adapter_id = adapter_id
        self.require_adapter = require_adapter
        self.use_verifier_calibration = use_verifier_calibration
        self._handle = None
        self._temperature: float | None = None

    @property
    def handle(self):
        if self._handle is None:
            self._handle = self.registry.nli(
                self.model_id,
                self.adapter_id,
                require_adapter=self.require_adapter,
            )
        return self._handle

    @property
    def temperature(self) -> float:
        if self._temperature is not None:
            return self._temperature
        settings = self.registry.settings
        value = float(settings.verifier_temperature)
        if self.use_verifier_calibration and settings.verifier_calibration_path:
            path = settings.resolve(settings.verifier_calibration_path)
            if not path.exists():
                raise RuntimeError(f"Verifier calibration file not found: {path}")
            payload = json.loads(path.read_text(encoding="utf-8"))
            value = float(payload.get("temperature", value))
        if value <= 0:
            raise RuntimeError("Verifier temperature must be positive")
        self._temperature = value
        return value

    def score_pairs(self, pairs: list[tuple[str, str]], batch_size: int, max_length: int) -> list[NLIScore]:
        if not pairs:
            return []
        import torch

        handle = self.handle
        output: list[NLIScore] = []
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start : start + batch_size]
            encoded = handle.tokenizer(
                [premise for premise, _ in batch],
                [hypothesis for _, hypothesis in batch],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            device = next(handle.model.parameters()).device
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.inference_mode():
                logits = handle.model(**encoded).logits.float() / self.temperature
                probs = torch.softmax(logits, dim=-1).cpu()
            for row in probs:
                output.append(
                    NLIScore(
                        entailment=float(row[handle.entailment_index]),
                        contradiction=(
                            float(row[handle.contradiction_index]) if handle.contradiction_index is not None else None
                        ),
                        neutral=(float(row[handle.neutral_index]) if handle.neutral_index is not None else None),
                    )
                )
        return output
