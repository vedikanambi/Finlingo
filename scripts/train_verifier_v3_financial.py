from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

import backend.training.train_verifier as verifier_module
from backend.app.core.config import Settings
from backend.training.train_verifier import _valid_unique


SEED = 42
RULE_FILE = Path("reports/verifier_rule_nli_v2_clean.jsonl")
random.seed(SEED)


def read_rule_rows() -> list[dict]:
    rows = []

    for line in RULE_FILE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))

    return _valid_unique(rows)


def balanced_select(
    rows: list[dict],
    target: int,
) -> list[dict]:
    labels = {
        "entailment": [],
        "neutral": [],
        "contradiction": [],
    }

    for row in _valid_unique(rows):
        labels[row["label"]].append(row)

    base = target // 3
    targets = {
        "entailment": base + (1 if target % 3 >= 1 else 0),
        "neutral": base + (1 if target % 3 >= 2 else 0),
        "contradiction": base,
    }

    selected = []

    for label, required in targets.items():
        random.shuffle(labels[label])

        if len(labels[label]) < required:
            raise RuntimeError(f"Insufficient {label} examples: {len(labels[label])}/{required}")

        selected.extend(labels[label][:required])

    random.shuffle(selected)
    return selected


def custom_collect(settings, split, limit, silver, repo):
    if split == "train":
        rule_rows = read_rule_rows()

        snli = list(repo.snli_pairs("train", 3000))
        contract = list(repo.contractnli_pairs("train", limit=3000))

        combined = rule_rows + snli + contract
        selected = balanced_select(combined, limit)

    else:
        snli = list(repo.snli_pairs("validation", 1500))
        contract = list(repo.contractnli_pairs("validation", limit=1500))

        selected = balanced_select(snli + contract, limit)

    print(
        f"{split.upper()} DATA:",
        len(selected),
        Counter(row["label"] for row in selected),
    )

    return selected


verifier_module._collect = custom_collect

settings = Settings(
    verifier_train_samples=5000,
    verifier_validation_samples=999,
    verifier_snli_ratio=0.20,
    verifier_contractnli_ratio=0.40,
    verifier_synthetic_ratio=0.40,
    verifier_validation_snli_ratio=0.40,
    verifier_validation_contractnli_ratio=0.60,
    verifier_min_synthetic_fraction=0.0,
    verifier_fail_on_source_shortage=True,
    verifier_num_train_epochs=3.0,
    verifier_learning_rate=1e-5,
    verifier_use_class_weights=True,
)

output = verifier_module.train_verifier(
    settings,
    output_dir=Path("models/stage6_verifier_v3_financial"),
)

print("TRAINING COMPLETE:", output.resolve())
