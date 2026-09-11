from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

from backend.app.core.config import Settings
from backend.app.core.provenance import runtime_manifest, sha256_path
from backend.app.services.dataset_streams import StreamingDatasetRepository
from backend.evaluation.risk_taxonomy import RiskTaxonomy
from backend.training.evidence import collect_training_evidence, reset_cuda_peak

logger = logging.getLogger(__name__)

RISK_LABELS = ["Auto-Renewal", "Hidden Fee", "Liability Waiver", "Data Sharing", "Penalty Clause", "Safe"]
LABEL2ID = {label: i for i, label in enumerate(RISK_LABELS)}
_SAFE_SOURCE_CATEGORIES = {"Parties", "Document Name", "Effective Date", "Agreement Date", "Governing Law"}


class _Dataset:
    def __init__(self, rows: list[dict[str, Any]], tokenizer, max_length: int):
        self.rows, self.tokenizer, self.max_length = rows, tokenizer, max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        row = self.rows[i]
        encoded = self.tokenizer(row["text"], truncation=True, max_length=self.max_length)
        encoded["labels"] = LABEL2ID[row["label"]]
        return encoded


def _collect(settings: Settings, split: str, limit: int, taxonomy: RiskTaxonomy, repo: StreamingDatasetRepository) -> list[dict[str, Any]]:
    """Collect CUAD clause/label pairs using the same document-hash split as everywhere else, so no FLB document leaks into training."""
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    safe_count = 0
    for row in repo.stream_partitioned(
        settings.cuad_dataset,
        requested_split=split,
        config=settings.cuad_config,
        key_fields=("file_name",),
        shuffle=True,
    ):
        source_label = str(row.get("label") or "")
        clause_text = str(row.get("clause") or "").strip()
        if not clause_text or len(clause_text.split()) < 6:
            continue
        key = clause_text[:200]
        if key in seen:
            continue
        mapped = taxonomy.map_question(source_label)
        target_label = None
        if mapped:
            target_label = mapped
        elif source_label in _SAFE_SOURCE_CATEGORIES and safe_count < limit // 6:
            target_label = "Safe"
            safe_count += 1
        if not target_label:
            continue
        seen.add(key)
        rows.append({"text": clause_text, "label": target_label})
        if len(rows) >= limit:
            break
    logger.info("Risk classifier split=%s collected=%d label_counts=%s", split, len(rows), dict(Counter(r["label"] for r in rows)))
    return rows


def _collect_data_sharing_supplement(
    settings: Settings, split: str, limit: int, repo: StreamingDatasetRepository
) -> list[dict[str, Any]]:
    """Backfill Data Sharing, since CUAD has zero examples of it - pulled from LegalBench's OPP-115 task instead, a source that never touches FLB's own eval pool."""
    import hashlib

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in repo.stream_raw(
        settings.legalbench_dataset,
        split="test",
        config=settings.legalbench_data_sharing_config,
        shuffle=True,
    ):
        if str(row.get("answer") or "").strip().lower() != "yes":
            continue
        clause_text = str(row.get("text") or "").strip()
        if not clause_text or len(clause_text.split()) < 6:
            continue
        key = clause_text[:200]
        if key in seen:
            continue
        seen.add(key)
        bucket = int(hashlib.sha256(key.encode("utf-8", errors="ignore")).hexdigest()[:8], 16) % 100
        wants_validation = split == "validation"
        if (bucket >= 85) != wants_validation:
            continue
        rows.append({"text": clause_text, "label": "Data Sharing"})
        if len(rows) >= limit:
            break
    logger.info("Risk classifier Data Sharing supplement split=%s collected=%d", split, len(rows))
    return rows


def train_risk_classifier(settings: Settings, *, model_id: str | None = None, output_dir: Path | None = None) -> Path:
    """Fine-tune a sequence classifier for the six-class consumer-risk taxonomy - full fine-tuning instead of prompting, since the frozen-embedding baseline already beat the prompted classifier."""
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
    model_id = model_id or settings.risk_classifier_model
    output = settings.resolve(output_dir or settings.risk_classifier_adapter_output)
    taxonomy = RiskTaxonomy(settings.resolve(settings.risk_taxonomy_path))
    repo = StreamingDatasetRepository(settings)
    try:
        # CUAD's own validation split is too small and skewed for this taxonomy, so I carve my own
        # stratified slice out of the train bucket instead - see the val_per_class split below.
        pool = _collect(settings, "train", settings.risk_classifier_train_samples, taxonomy, repo)
        other_class_sizes = [
            n for label, n in Counter(r["label"] for r in pool).items() if label not in ("Data Sharing", "Safe")
        ]
        target_supplement = max(200, int(sum(other_class_sizes) / max(1, len(other_class_sizes))))
        pool += _collect_data_sharing_supplement(settings, "train", target_supplement, repo)

        by_label: dict[str, list[dict[str, Any]]] = {}
        for row in pool:
            by_label.setdefault(row["label"], []).append(row)
        rng = __import__("random").Random(settings.random_seed)
        val_per_class = 25
        train, val = [], []
        for label, label_rows in by_label.items():
            rng.shuffle(label_rows)
            split_at = min(val_per_class, max(1, len(label_rows) // 10))
            val.extend(label_rows[:split_at])
            train.extend(label_rows[split_at:])
        train_counts = Counter(r["label"] for r in train)
        # class weighting alone wasn't enough - Penalty Clause kept collapsing into Liability Waiver
        # (0/10 correct), so I oversample the minority classes up to full parity instead
        target_min = max(train_counts.values())
        oversampled = list(train)
        for label, rows in by_label.items():
            current = train_counts.get(label, 0)
            if current == 0 or current >= target_min:
                continue
            label_train_rows = [r for r in train if r["label"] == label]
            needed = target_min - current
            for i in range(needed):
                oversampled.append(dict(label_train_rows[i % len(label_train_rows)]))
        train = oversampled
        rng.shuffle(train)
        rng.shuffle(val)
    finally:
        repo.close()
    train_texts = {" ".join(row["text"].lower().split()) for row in train}
    val = [row for row in val if " ".join(row["text"].lower().split()) not in train_texts]
    if len(set(r["label"] for r in train)) < 2 or not val:
        raise RuntimeError(f"Insufficient risk-classifier training data: train={len(train)}, val={len(val)}")

    model_revision = settings.model_revision(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=settings.hf_token, revision=model_revision)
    compute_dtype = (
        torch.bfloat16
        if use_cuda and torch.cuda.is_bf16_supported()
        else (torch.float16 if use_cuda else torch.float32)
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        token=settings.hf_token,
        num_labels=len(RISK_LABELS),
        ignore_mismatched_sizes=True,
        revision=model_revision,
        label2id=LABEL2ID,
        id2label={v: k for k, v in LABEL2ID.items()},
        dtype=compute_dtype,
    )

    present = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules()}
    if {"query", "key", "value"}.issubset(present):
        targets = ["query", "key", "value"]
    elif {"query_proj", "key_proj", "value_proj"}.issubset(present):
        targets = ["query_proj", "key_proj", "value_proj"]
    elif {"q_proj", "k_proj", "v_proj"}.issubset(present):
        targets = ["q_proj", "k_proj", "v_proj"]
    else:
        raise RuntimeError(f"No supported attention projection set found; leaves={sorted(present)[:100]}")
    modules_to_save = [name for name in ("classifier", "pooler") if name in present] or None

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
        [total / (len(RISK_LABELS) * max(1, counts[label])) for label in RISK_LABELS],
        dtype=torch.float32,
    )

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            class_weights = (
                weights.to(outputs.logits.device, dtype=outputs.logits.dtype)
                if settings.risk_classifier_use_class_weights
                else None
            )
            loss = torch.nn.functional.cross_entropy(outputs.logits, labels, weight=class_weights)
            return (loss, outputs) if return_outputs else loss

    if settings.enable_tf32 and use_cuda:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    args = TrainingArguments(
        dataloader_num_workers=settings.dataloader_num_workers,
        dataloader_pin_memory=use_cuda,
        output_dir=str(output),
        per_device_train_batch_size=settings.train_batch_size,
        per_device_eval_batch_size=settings.eval_batch_size,
        gradient_accumulation_steps=settings.gradient_accumulation_steps,
        learning_rate=settings.risk_classifier_learning_rate,
        num_train_epochs=settings.risk_classifier_num_train_epochs,
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
        report_to="none",
        seed=settings.random_seed,
        data_seed=settings.random_seed,
    )
    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=_Dataset(train, tokenizer, settings.risk_classifier_max_length),
        eval_dataset=_Dataset(val, tokenizer, settings.risk_classifier_max_length),
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=settings.risk_classifier_early_stopping_patience)],
    )
    baseline_metrics = trainer.evaluate(metric_key_prefix="baseline")
    reset_cuda_peak(torch)
    last_checkpoint = get_last_checkpoint(str(output)) if output.exists() else None
    if last_checkpoint:
        logger.info("Resuming risk-classifier training from checkpoint: %s", last_checkpoint)
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
                "training_mode": "fp16_lora" if not (use_cuda and torch.cuda.is_bf16_supported()) else "bf16_lora",
                "adapter_artifact_sha256_before_manifest": adapter_artifact_sha256,
                "train_examples": len(train),
                "validation_examples": len(val),
                "label2id": LABEL2ID,
                "source": settings.cuad_dataset,
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
                "deviation_note": (
                    "Proposal specifies S5 as prompted-only; this adapter is a disclosed "
                    "deviation adopted after reports/ablations.json's FinBERT frozen-embedding "
                    "baseline (macro-F1 0.69) beat the deployed few-shot classifier (0.56)."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return output
