import csv
from pathlib import Path

from backend.app.core.config import Settings
from backend.evaluation.consolidate_results import consolidate_results


def test_consolidation_records_missing_artifacts_without_inventing_scores(tmp_path: Path):
    settings = Settings(_env_file=None, project_root=tmp_path)
    result = consolidate_results(settings, Path("reports/tables"))
    rows = list(csv.DictReader(Path(result["csv"]).open(encoding="utf-8")))
    assert result["missing_artifacts"] >= 1
    assert any(row["value"] == "missing" for row in rows)
    assert not any(row["value"] in {"0.9", "0.95"} for row in rows)
