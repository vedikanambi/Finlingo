from __future__ import annotations

import json
import logging
from dataclasses import replace
from pathlib import Path

from backend.app.core.config import Settings
from backend.app.core.provenance import runtime_manifest, sha256_path
from backend.app.services.dataset_streams import StreamingDatasetRepository
from backend.app.services.model_registry import ModelRegistry
from backend.app.services.nli_service import NLIService
from backend.evaluation.metrics import sari_sentence
from backend.training.silver_labels import ProviderQuotaExhausted, SilverLabelGenerator, SimplificationPair
from backend.training.evidence import collect_training_evidence, reset_cuda_peak

logger = logging.getLogger(__name__)

_QWEN_LORA_TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]
_SYSTEM_MESSAGE = (
    "You simplify consumer financial and legal clauses into plain English at approximately "
    "Flesch-Kincaid Grade 8 or lower. Preserve every amount, date, deadline, obligation, "
    "exception, penalty, condition, negation, named party, and legal meaning exactly. "
    "Do not add information. Return only the simplified text."
)


class _Dataset:
    def __init__(self, pairs: list[SimplificationPair], tokenizer, max_length: int, target_max_length: int) -> None:
        self.pairs, self.tokenizer, self.max_length = pairs, tokenizer, max_length
        self.target_max_length = target_max_length

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict:
        pair = self.pairs[index]
        prompt_messages = [
            {"role": "system", "content": _SYSTEM_MESSAGE},
            {"role": "user", "content": pair.source},
        ]
        full_messages = [
            *prompt_messages,
            {"role": "assistant", "content": pair.target},
        ]
        prompt_ids = _template_input_ids(
            self.tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=True,
                add_generation_prompt=True,
            )
        )
        full_ids = _template_input_ids(
            self.tokenizer.apply_chat_template(
                full_messages,
                tokenize=True,
                add_generation_prompt=False,
            )
        )
        if full_ids[: len(prompt_ids)] != prompt_ids:
            raise RuntimeError("Qwen chat-template assistant boundary could not be identified")
        target_ids = full_ids[len(prompt_ids) :]
        target_ids = target_ids[: min(max(8, self.target_max_length + 1), self.max_length - 1)]
        prompt_ids = prompt_ids[: max(self.max_length - len(target_ids), 1)]
        input_ids = prompt_ids + target_ids
        labels = [-100] * len(prompt_ids) + target_ids
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
            "quality_weight": float(pair.quality_weight),
        }


def _template_input_ids(encoded) -> list[int]:
    """Normalise chat-template output across tokenizer versions."""
    if isinstance(encoded, dict) or hasattr(encoded, "keys"):
        encoded = encoded["input_ids"]
    if encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    return list(encoded)


class _QualityCollator:
    def __init__(self, tokenizer) -> None:
        from transformers import DataCollatorForSeq2Seq

        self.base = DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100)

    def __call__(self, features: list[dict]):
        import torch

        weights = [float(feature.pop("quality_weight", 1.0)) for feature in features]
        batch = self.base(features)
        batch["quality_weight"] = torch.tensor(weights, dtype=torch.float32)
        return batch


class _QualityAwareTrainerMixin:
    # SARI isn't differentiable, so instead of a real multi-objective loss this just weights
    # the per-example causal-LM loss by precomputed SARI/NLI scores


    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):  # noqa: ARG002
        import torch.nn.functional as functional

        quality_weight = inputs.pop("quality_weight")
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        shifted_logits = logits[:, :-1, :].contiguous()
        shifted_labels = labels[:, 1:].contiguous()
        token_loss = functional.cross_entropy(
            shifted_logits.view(-1, shifted_logits.size(-1)),
            shifted_labels.view(-1),
            reduction="none",
            ignore_index=-100,
        ).view(shifted_labels.shape)
        mask = shifted_labels.ne(-100)
        per_example = (token_loss * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        weights = quality_weight.to(per_example.device)
        loss = (per_example * weights).sum() / weights.sum().clamp(min=1e-6)
        return (loss, outputs) if return_outputs else loss


def _quality_pairs(
    settings: Settings,
    registry: ModelRegistry,
    silver: SilverLabelGenerator,
    rows: list[dict[str, str]],
    *,
    training: bool = True,
) -> list[SimplificationPair]:
    primary = silver.simplification_pairs(iter(rows), len(rows), alternate=False)
    if not settings.simplifier_quality_objective or not training:
        return primary

    alternate = {pair.source_id: pair for pair in silver.simplification_pairs(iter(rows), len(rows), alternate=True)}
    aligned = [pair for pair in primary if pair.source_id in alternate]
    if not aligned:
        raise RuntimeError("No aligned primary/alternate simplification references were generated")

    nli = NLIService(
        registry,
        model_id=settings.s2_semantic_model,
        adapter_id=settings.s2_semantic_adapter,
        require_adapter=settings.s2_semantic_require_adapter,
        use_verifier_calibration=False,
    )
    scores = nli.score_pairs(
        [(pair.source, pair.target) for pair in aligned],
        settings.verifier_batch_size,
        settings.verifier_max_length,
    )
    signal_total = settings.simplifier_quality_sari_weight + settings.simplifier_quality_nli_weight
    output: list[SimplificationPair] = []
    for pair, score in zip(aligned, scores):
        sari = sari_sentence(pair.source, pair.target, alternate[pair.source_id].target)
        sari_signal = min(max(sari / 100.0, 0.0), 1.0)
        nli_signal = min(max(score.entailment, 0.0), 1.0)
        combined = (
            settings.simplifier_quality_sari_weight * sari_signal + settings.simplifier_quality_nli_weight * nli_signal
        ) / signal_total
        weight = settings.simplifier_quality_min_weight + (1 - settings.simplifier_quality_min_weight) * combined
        output.append(
            replace(
                pair,
                quality_weight=float(weight),
                sari_score=float(sari),
                semantic_entailment=float(score.entailment),
            )
        )
    return output


def train_simplifier(settings: Settings) -> Path:
    """Fine-tune Qwen with QLoRA on streamed, RAM-only silver pairs."""
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    )
    from transformers.trainer_utils import get_last_checkpoint

    if not torch.cuda.is_available():
        raise RuntimeError("S2 QLoRA training requires an NVIDIA CUDA GPU (local or Colab)")

    repo = StreamingDatasetRepository(settings)
    registry = ModelRegistry(settings)
    silver = SilverLabelGenerator(settings, registry)
    try:
        train_rows = list(repo.mixed_simplification_sources(settings.simplifier_train_samples, "train"))
        validation_rows = list(repo.mixed_simplification_sources(settings.simplifier_validation_samples, "validation"))
        try:
            train_pairs = _quality_pairs(settings, registry, silver, train_rows, training=True)
            validation_pairs = _quality_pairs(settings, registry, silver, validation_rows, training=False)
        except ProviderQuotaExhausted:
            _log_silver_summary(settings, silver)
            raise
    finally:
        repo.close()
    # need to free this before loading Qwen or it won't fit on an 8GB GPU
    registry.release_nli(settings.s2_semantic_model, settings.s2_semantic_adapter)
    if (
        len(train_pairs) < settings.simplifier_min_training_pairs
        or len(validation_pairs) < settings.simplifier_min_validation_pairs
    ):
        raise RuntimeError(
            "Insufficient quality-controlled simplification pairs: "
            f"train={len(train_pairs)}/"
            f"{settings.simplifier_min_training_pairs}, "
            f"validation={len(validation_pairs)}/"
            f"{settings.simplifier_min_validation_pairs}. "
            f"Source rows: train={len(train_rows)}, "
            f"validation={len(validation_rows)}."
        )

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=settings.qlora_quant_type,
        bnb_4bit_use_double_quant=settings.qlora_use_double_quant,
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    )
    model_revision = settings.model_revision(settings.simplifier_model)
    tokenizer = AutoTokenizer.from_pretrained(
        settings.simplifier_model, token=settings.hf_token, use_fast=True, revision=model_revision
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        settings.simplifier_model,
        token=settings.hf_token,
        revision=model_revision,
        quantization_config=quant,
        device_map="auto",
        trust_remote_code=settings.model_trust_remote_code,
    )
    model = prepare_model_for_kbit_training(model)
    if settings.domain_warmup_adapter:
        model = PeftModel.from_pretrained(
            model,
            str(settings.resolve(settings.domain_warmup_adapter)),
            token=settings.hf_token,
            is_trainable=True,
        )
    else:
        model = get_peft_model(
            model,
            LoraConfig(
                r=settings.qlora_rank,
                lora_alpha=settings.qlora_alpha,
                lora_dropout=settings.qlora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=_QWEN_LORA_TARGETS,
            ),
        )
    model.config.use_cache = False

    output = settings.resolve(settings.simplifier_adapter_output)
    if settings.enable_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    arguments = TrainingArguments(
        group_by_length=settings.use_length_grouped_batching,
        dataloader_num_workers=settings.dataloader_num_workers,
        dataloader_pin_memory=torch.cuda.is_available(),
        output_dir=str(output),
        per_device_train_batch_size=settings.simplifier_batch_size,
        per_device_eval_batch_size=settings.eval_batch_size,
        gradient_accumulation_steps=settings.simplifier_gradient_accumulation_steps,
        learning_rate=settings.learning_rate,
        num_train_epochs=settings.num_train_epochs,
        max_steps=settings.max_train_steps or -1,
        warmup_ratio=settings.warmup_ratio,
        weight_decay=settings.weight_decay,
        logging_steps=settings.logging_steps,
        save_steps=settings.save_steps,
        eval_steps=settings.eval_steps,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=settings.simplifier_bf16 and torch.cuda.is_bf16_supported(),
        fp16=settings.simplifier_fp16 and not settings.simplifier_bf16,
        gradient_checkpointing=True,
        report_to="none",
        remove_unused_columns=False,
        seed=settings.random_seed,
    )
    trainer_class = type("QualityAwareTrainer", (_QualityAwareTrainerMixin, Trainer), {})
    trainer = trainer_class(
        model=model,
        args=arguments,
        train_dataset=_Dataset(
            train_pairs, tokenizer, settings.simplifier_max_input_tokens, settings.simplifier_max_new_tokens
        ),
        eval_dataset=_Dataset(
            validation_pairs, tokenizer, settings.simplifier_max_input_tokens, settings.simplifier_max_new_tokens
        ),
        data_collator=_QualityCollator(tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    reset_cuda_peak(torch)
    last_checkpoint = get_last_checkpoint(str(output)) if output.exists() else None
    if last_checkpoint:
        logger.info("Resuming simplifier training from checkpoint: %s", last_checkpoint)
    trainer.train(resume_from_checkpoint=last_checkpoint)
    training_evidence = collect_training_evidence(trainer, torch)
    output.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(output)
    tokenizer.save_pretrained(output)
    adapter_artifact_sha256 = sha256_path(output)

    quality = {
        "objective": "quality-weighted causal-LM loss using precomputed SARI and NLI semantic-preservation signals",
        "enabled": settings.simplifier_quality_objective,
        "minimum_weight": settings.simplifier_quality_min_weight,
        "sari_weight": settings.simplifier_quality_sari_weight,
        "nli_weight": settings.simplifier_quality_nli_weight,
        "mean_train_sari": _mean([pair.sari_score for pair in train_pairs if pair.sari_score is not None]),
        "mean_train_nli": _mean(
            [pair.semantic_entailment for pair in train_pairs if pair.semantic_entailment is not None]
        ),
        "mean_train_weight": _mean([pair.quality_weight for pair in train_pairs]),
    }
    (output / "training_manifest.json").write_text(
        json.dumps(
            {
                "base_model": settings.simplifier_model,
                "adapter_artifact_sha256_before_manifest": adapter_artifact_sha256,
                "domain_warmup_adapter": settings.domain_warmup_adapter,
                "train_examples": len(train_pairs),
                "validation_examples": len(validation_pairs),
                "raw_dataset_persisted": False,
                "dataset_revisions": settings.dataset_revisions,
                "quality_objective": quality,
                "runtime_manifest": runtime_manifest(settings),
                "training_evidence": training_evidence,
                "continuation_strategy": (
                    "continued training of the configured domain-warm-up PEFT adapter"
                    if settings.domain_warmup_adapter
                    else "new task-specific PEFT adapter"
                ),
                "qlora": {
                    "rank": settings.qlora_rank,
                    "alpha": settings.qlora_alpha,
                    "dropout": settings.qlora_dropout,
                    "quant_type": settings.qlora_quant_type,
                    "double_quant": settings.qlora_use_double_quant,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _log_silver_summary(settings, silver)
    return output


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _log_silver_summary(settings: Settings, silver: SilverLabelGenerator) -> None:
    logger.info(
        "Silver-label summary: cached labels reused=%d; new labels generated=%d; "
        "rejected labels=%d; total valid labels=%d; cache path=%s",
        silver.cached_labels_reused,
        silver.new_labels_generated,
        silver.rejected_labels,
        silver.cached_labels_reused + silver.new_labels_generated - silver.rejected_labels,
        settings.resolve(settings.silver_label_cache_dir) if settings.silver_label_cache_dir else "disabled",
    )
