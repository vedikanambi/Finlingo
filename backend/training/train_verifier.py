from __future__ import annotations

import inspect
import json
import logging
from pathlib import Path
from collections import Counter
from typing import Any

from backend.app.core.config import Settings
from backend.app.core.provenance import runtime_manifest, sha256_path
from backend.app.services.dataset_streams import StreamingDatasetRepository
from backend.app.services.model_registry import ModelRegistry
from backend.training.silver_labels import SilverLabelGenerator
from backend.training.evidence import collect_training_evidence, reset_cuda_peak

LABEL2ID = {"entailment": 0, "neutral": 1, "contradiction": 2}
logger = logging.getLogger(__name__)


class _Dataset:
    def __init__(self, rows, tokenizer, max_length):
        self.rows, self.tokenizer, self.max_length = rows, tokenizer, max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        encoded = self.tokenizer(row["premise"], row["hypothesis"], truncation=True, max_length=self.max_length)
        encoded["labels"] = LABEL2ID[row["label"]]
        return encoded


def _valid_unique(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    clean, seen = [], set()
    for row in rows:
        if row.get("label") not in LABEL2ID or not row.get("premise") or not row.get("hypothesis"):
            continue
        key = (str(row["premise"]), str(row["hypothesis"]))
        if key in seen:
            continue
        seen.add(key)
        clean.append(row)
    return clean


def _collect(settings, split, limit, silver, repo):
    """Collect the source mix, and fail loudly instead of silently backfilling shortages."""
    training = split == "train"
    snli_ratio = settings.verifier_snli_ratio if training else settings.verifier_validation_snli_ratio
    contract_ratio = settings.verifier_contractnli_ratio if training else settings.verifier_validation_contractnli_ratio
    requested = {
        "snli": int(limit * snli_ratio),
        "contractnli": int(limit * contract_ratio),
    }
    requested["synthetic"] = limit - requested["snli"] - requested["contractnli"] if training else 0
    rows: list[dict[str, Any]] = []
    actual = Counter()

    try:
        values = list(repo.snli_pairs(split, requested["snli"]))
        rows.extend(values)
        actual["snli"] += len(values)
    except Exception as exc:
        logger.exception("SNLI collection failed for split=%s: %s", split, exc)

    try:
        values = list(repo.contractnli_pairs(split, limit=requested["contractnli"]))
        rows.extend(values)
        actual["contractnli"] += len(values)
    except Exception as exc:
        logger.exception("ContractNLI collection failed for split=%s: %s", split, exc)

    if requested["synthetic"]:
        try:
            source_count = max(requested["synthetic"] * 4, requested["synthetic"] + 16)
            values = list(silver.synthetic_nli(repo.cuad_clauses("train", source_count), requested["synthetic"]))
            rows.extend(values)
            actual["synthetic"] += len(values)
        except Exception as exc:
            logger.exception("Synthetic financial NLI collection failed: %s", exc)

    rows = _valid_unique(rows)
    logger.info(
        "Verifier source mix split=%s requested=%s actual=%s usable=%d", split, requested, dict(actual), len(rows)
    )
    if training and limit:
        synthetic_fraction = actual["synthetic"] / limit
        if synthetic_fraction < settings.verifier_min_synthetic_fraction:
            message = (
                f"Synthetic financial NLI shortage: {actual['synthetic']}/{limit} "
                f"({synthetic_fraction:.1%}), minimum={settings.verifier_min_synthetic_fraction:.1%}."
            )
            if settings.verifier_fail_on_source_shortage:
                raise RuntimeError(message)
            logger.warning(message)
    minimum = max(1, int(limit * 0.80))
    if len(rows) < minimum:
        message = f"Verifier split={split} collected only {len(rows)}/{limit} usable examples"
        if settings.verifier_fail_on_source_shortage:
            raise RuntimeError(message)
        logger.warning(message)
    return rows[:limit]


def _peft_targets(model) -> tuple[list[str], list[str] | None]:
    """Find the attention-projection LoRA targets for whichever architecture this model is."""
    present = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules()}
    if {"query_proj", "key_proj", "value_proj"}.issubset(present):
        targets = ["query_proj", "key_proj", "value_proj"]
    elif {"q_proj", "k_proj", "v_proj"}.issubset(present):
        targets = ["q_proj", "k_proj", "v_proj"]
    elif "Wqkv" in present:
        targets = ["Wqkv"]
    else:
        raise RuntimeError(f"No supported attention projection set found; leaves={sorted(present)[:100]}")
    save = [name for name in ("classifier", "pooler", "head") if name in present]
    return targets, (save or None)


def train_verifier(settings: Settings, *, model_id: str | None = None, output_dir: Path | None = None) -> Path:
    """Fine-tune the NLI checkpoint with LoRA."""
    import numpy as np
    import torch
    from peft import LoraConfig, get_peft_model
    from sklearn.metrics import precision_recall_fscore_support
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    )
    from transformers.trainer_utils import get_last_checkpoint

    use_cuda = torch.cuda.is_available()
    model_id = model_id or settings.verifier_training_base_model
    output = settings.resolve(output_dir or settings.verifier_adapter_output)
    repo = StreamingDatasetRepository(settings)
    silver = SilverLabelGenerator(settings, ModelRegistry(settings))
    try:
        train = _collect(settings, "train", settings.verifier_train_samples, silver, repo)
        val = _collect(settings, "validation", settings.verifier_validation_samples, silver, repo)
    finally:
        repo.close()
    if len(train) < min(100, settings.verifier_train_samples) or not val:
        raise RuntimeError(f"Insufficient NLI examples: train={len(train)}, val={len(val)}")
    model_revision = settings.model_revision(model_id)
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        token=settings.hf_token,
        trust_remote_code=settings.model_trust_remote_code,
        revision=model_revision,
        use_fast=False,
    )
    compute_dtype = (
        torch.bfloat16
        if use_cuda and torch.cuda.is_bf16_supported()
        else (torch.float16 if use_cuda else torch.float32)
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        token=settings.hf_token,
        num_labels=3,
        revision=model_revision,
        label2id=LABEL2ID,
        id2label={v: k for k, v in LABEL2ID.items()},
        dtype=compute_dtype,
        trust_remote_code=settings.model_trust_remote_code,
    )
    targets, modules_to_save = _peft_targets(model)
    model = get_peft_model(
        model,
        LoraConfig(
            r=settings.qlora_rank,
            lora_alpha=settings.qlora_alpha,
            lora_dropout=settings.qlora_dropout,
            bias="none",
            task_type="SEQ_CLS",
            target_modules=targets,
            modules_to_save=modules_to_save,
        ),
    )
    model.print_trainable_parameters()

    def metrics(pred):
        labels = pred.label_ids
        predictions = np.argmax(pred.predictions, axis=-1)
        p, r, f1, _ = precision_recall_fscore_support(labels, predictions, average="macro", zero_division=0)
        return {
            "macro_precision": p,
            "macro_recall": r,
            "macro_f1": f1,
            "accuracy": float(np.mean(labels == predictions)),
        }

    counts = Counter(row["label"] for row in train)
    total = sum(counts.values())
    weights = torch.tensor(
        [total / (len(LABEL2ID) * max(1, counts[label])) for label in ("entailment", "neutral", "contradiction")],
        dtype=torch.float32,
    )

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            class_weights = (
                weights.to(outputs.logits.device, dtype=outputs.logits.dtype)
                if settings.verifier_use_class_weights
                else None
            )
            loss = torch.nn.functional.cross_entropy(outputs.logits, labels, weight=class_weights)
            return (loss, outputs) if return_outputs else loss

    if settings.enable_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    training_arguments_kwargs = {}
    if "group_by_length" in inspect.signature(TrainingArguments.__init__).parameters:
        # newer transformers versions dropped this arg - it's just a batching hint so safe to skip instead of pinning a version
        training_arguments_kwargs["group_by_length"] = settings.use_length_grouped_batching

    args = TrainingArguments(
        **training_arguments_kwargs,
        dataloader_num_workers=settings.dataloader_num_workers,
        dataloader_pin_memory=torch.cuda.is_available(),
        output_dir=str(output),
        per_device_train_batch_size=settings.train_batch_size,
        per_device_eval_batch_size=settings.eval_batch_size,
        gradient_accumulation_steps=settings.gradient_accumulation_steps,
        learning_rate=settings.verifier_learning_rate,
        num_train_epochs=settings.verifier_num_train_epochs,
        max_steps=settings.max_train_steps or -1,
        warmup_ratio=settings.warmup_ratio,
        weight_decay=settings.weight_decay,
        logging_steps=settings.logging_steps,
        save_steps=settings.save_steps,
        eval_steps=settings.eval_steps,
        eval_strategy="steps",
        save_strategy="steps",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_macro_f1",
        greater_is_better=True,
        bf16=use_cuda and torch.cuda.is_bf16_supported(),
        fp16=use_cuda and not torch.cuda.is_bf16_supported(),
        gradient_checkpointing=settings.verifier_gradient_checkpointing,
        report_to="none",
        seed=settings.random_seed,
        data_seed=settings.random_seed,
    )
    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=_Dataset(train, tokenizer, settings.verifier_max_length),
        eval_dataset=_Dataset(val, tokenizer, settings.verifier_max_length),
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=settings.verifier_early_stopping_patience)],
    )
    baseline_metrics = trainer.evaluate(metric_key_prefix="baseline")
    reset_cuda_peak(torch)
    last_checkpoint = get_last_checkpoint(str(output)) if output.exists() else None
    if last_checkpoint:
        logger.info("Resuming verifier training from checkpoint: %s", last_checkpoint)
    trainer.train(resume_from_checkpoint=last_checkpoint)
    final_metrics = trainer.evaluate(metric_key_prefix="best")
    training_evidence = collect_training_evidence(trainer, torch)
    output.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(output)
    tokenizer.save_pretrained(output)
    adapter_artifact_sha256 = sha256_path(output)
    (output / "training_manifest.json").write_text(
        json.dumps(
            {
                "base_model": model_id,
                "training_mode": settings.verifier_training_mode,
                "adapter_artifact_sha256_before_manifest": adapter_artifact_sha256,
                "train_examples": len(train),
                "validation_examples": len(val),
                "label2id": LABEL2ID,
                "sources": [settings.snli_dataset, settings.contractnli_dataset, settings.cuad_dataset],
                "dataset_revisions": settings.dataset_revisions,
                "model_revision": model_revision,
                "lora_target_modules": targets,
                "modules_to_save": modules_to_save,
                "raw_dataset_persisted": False,
                "train_label_distribution": dict(counts),
                "validation_label_distribution": dict(Counter(row["label"] for row in val)),
                "baseline_metrics": baseline_metrics,
                "best_metrics": final_metrics,
                "training_evidence": training_evidence,
                "runtime_manifest": runtime_manifest(settings),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return output
