from __future__ import annotations

import json
from pathlib import Path

from backend.app.core.config import Settings
from backend.app.core.provenance import runtime_manifest, sha256_path
from backend.app.services.dataset_streams import StreamingDatasetRepository
from backend.training.evidence import collect_training_evidence, reset_cuda_peak


class _WarmupDataset:
    def __init__(self, texts: list[str], tokenizer, max_length: int) -> None:
        self.texts, self.tokenizer, self.max_length = texts, tokenizer, max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> dict:
        return self.tokenizer(
            self.texts[index],
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True,
        )


def train_domain_warmup(settings: Settings) -> Path:
    """QLoRA financial-domain warm-up on streamed EDGAR text - only the adapter and manifest get written to disk."""
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("EDGAR domain warm-up requires an NVIDIA CUDA GPU (local or Colab)")

    model_revision = settings.model_revision(settings.simplifier_model)
    tokenizer = AutoTokenizer.from_pretrained(
        settings.simplifier_model, token=settings.hf_token, use_fast=True, revision=model_revision
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    def collect_chunks(repo: StreamingDatasetRepository, split: str, target: int) -> list[str]:
        chunks: list[str] = []
        for row in repo.edgar_texts(split):
            token_ids = tokenizer(
                row["text"],
                add_special_tokens=False,
                truncation=False,
            ).input_ids
            for start in range(0, len(token_ids), settings.simplifier_max_input_tokens):
                window = token_ids[start : start + settings.simplifier_max_input_tokens]
                if len(window) < 128:
                    continue
                chunks.append(tokenizer.decode(window, skip_special_tokens=True))
                if len(chunks) >= target:
                    return chunks
        return chunks

    repo = StreamingDatasetRepository(settings)
    try:
        train_texts = collect_chunks(repo, "train", settings.domain_warmup_samples)
        validation_texts = collect_chunks(repo, "validation", settings.domain_warmup_validation_samples)
    finally:
        repo.close()
    if len(train_texts) < 100 or not validation_texts:
        raise RuntimeError(
            f"Insufficient streamed EDGAR chunks: train={len(train_texts)}, validation={len(validation_texts)}"
        )
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=settings.qlora_quant_type,
        bnb_4bit_use_double_quant=settings.qlora_use_double_quant,
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        settings.simplifier_model,
        token=settings.hf_token,
        revision=model_revision,
        quantization_config=quant,
        device_map="auto",
        trust_remote_code=settings.model_trust_remote_code,
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(
        model,
        LoraConfig(
            r=settings.qlora_rank,
            lora_alpha=settings.qlora_alpha,
            lora_dropout=settings.qlora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
    )

    output = settings.resolve(settings.domain_warmup_adapter_output)
    arguments = TrainingArguments(
        output_dir=str(output),
        per_device_train_batch_size=settings.train_batch_size,
        per_device_eval_batch_size=settings.eval_batch_size,
        gradient_accumulation_steps=settings.gradient_accumulation_steps,
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
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True,
        report_to="none",
        seed=settings.random_seed,
    )
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=_WarmupDataset(train_texts, tokenizer, settings.simplifier_max_input_tokens),
        eval_dataset=_WarmupDataset(validation_texts, tokenizer, settings.simplifier_max_input_tokens),
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    reset_cuda_peak(torch)
    trainer.train()
    training_evidence = collect_training_evidence(trainer, torch)
    output.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(output)
    tokenizer.save_pretrained(output)
    adapter_artifact_sha256 = sha256_path(output)
    (output / "training_manifest.json").write_text(
        json.dumps(
            {
                "stage": "S2 financial-domain warm-up",
                "adapter_artifact_sha256_before_manifest": adapter_artifact_sha256,
                "base_model": settings.simplifier_model,
                "dataset": settings.edgar_dataset,
                "dataset_revision": settings.dataset_revision(settings.edgar_dataset),
                "train_examples": len(train_texts),
                "validation_examples": len(validation_texts),
                "raw_dataset_persisted": False,
                "runtime_manifest": runtime_manifest(settings),
                "training_evidence": training_evidence,
                "qlora": {
                    "rank": settings.qlora_rank,
                    "alpha": settings.qlora_alpha,
                    "quant_type": settings.qlora_quant_type,
                    "double_quant": settings.qlora_use_double_quant,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return output
