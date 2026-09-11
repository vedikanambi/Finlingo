import json
from pathlib import Path

from backend.app.core.config import Settings
from backend.evaluation.parser_eval import evaluate_parser_boundaries


def test_parser_boundary_evaluation_on_labelled_text(tmp_path: Path):
    document = tmp_path / "contract.txt"
    document.write_text(
        "TERMS\n1. The borrower pays ten euro monthly.\n2. The agreement renews automatically.\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            [
                {
                    "path": document.name,
                    "gold_clauses": [
                        "1. The borrower pays ten euro monthly.",
                        "2. The agreement renews automatically.",
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    result = evaluate_parser_boundaries(Settings(_env_file=None), manifest)
    assert result["micro"]["precision"] == 1.0
    assert result["micro"]["recall"] == 1.0
