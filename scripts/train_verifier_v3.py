from pathlib import Path

from backend.app.core.config import Settings
from backend.training.train_verifier import train_verifier

settings = Settings(
    verifier_train_samples=5000,
    verifier_validation_samples=1000,
    verifier_snli_ratio=0.40,
    verifier_contractnli_ratio=0.60,
    verifier_synthetic_ratio=0.0,
    verifier_validation_snli_ratio=0.40,
    verifier_validation_contractnli_ratio=0.60,
    verifier_min_synthetic_fraction=0.0,
    verifier_fail_on_source_shortage=True,
    verifier_num_train_epochs=3.0,
    verifier_learning_rate=1e-5,
    verifier_use_class_weights=True,
)

output = train_verifier(
    settings,
    output_dir=Path("models/stage6_verifier_v3"),
)

print(f"TRAINING COMPLETE: {output.resolve()}")
