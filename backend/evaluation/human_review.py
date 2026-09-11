from __future__ import annotations

import csv
import datetime
from collections import Counter
from pathlib import Path

from sklearn.metrics import cohen_kappa_score

from backend.evaluation.flb_builder import FLBRecord

MAX_REVIEWERS = 5

_YES_VALUES = {"yes", "y", "1", "true"}
_NO_VALUES = {"no", "n", "0", "false"}


def _binary(row: dict[str, str], *names: str) -> int | None:
    for name in names:
        value = str(row.get(name) or "").strip().lower()
        if value in _YES_VALUES:
            return 1
        if value in _NO_VALUES:
            return 0
    return None


def _binary_inverted(row: dict[str, str], *names: str) -> int | None:
    """like _binary but inverted - for questions where Yes means "something's wrong"."""
    value = _binary(row, *names)
    return None if value is None else 1 - value


def fleiss_kappa(rows: list[list[str]]) -> float | None:
    """Fleiss' kappa for 3+ raters per subject; subjects with under 2 ratings get skipped."""
    usable = [row for row in rows if len(row) >= 2]
    if not usable:
        return None
    categories = sorted({label for row in usable for label in row})
    if len(categories) < 2:
        return None
    n_subjects = len(usable)
    n_raters = len(usable[0])
    if any(len(row) != n_raters for row in usable):
        n_raters = min(len(row) for row in usable)
        usable = [row[:n_raters] for row in usable]
    category_index = {label: i for i, label in enumerate(categories)}
    counts = [[0] * len(categories) for _ in range(n_subjects)]
    for i, row in enumerate(usable):
        for label in row:
            counts[i][category_index[label]] += 1

    p_i = []
    for row_counts in counts:
        total = sum(row_counts)
        p_i.append((sum(c * c for c in row_counts) - total) / (total * (total - 1)))
    p_bar = sum(p_i) / len(p_i)

    p_j = []
    total_ratings = n_subjects * n_raters
    for j in range(len(categories)):
        p_j.append(sum(row_counts[j] for row_counts in counts) / total_ratings)
    p_bar_e = sum(p * p for p in p_j)

    if p_bar_e == 1:
        return None
    return (p_bar - p_bar_e) / (1 - p_bar_e)


def load_adjudicated_reviews(path: Path) -> dict[str, dict[str, object]]:
    """loads completed adjudications - old sheets with a single adjudicated_supported column still work, mapped to both labels."""
    output: dict[str, dict[str, object]] = {}
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            record_id = str(row.get("record_id") or "").strip()
            source_supported = _binary(row, "adjudicated_source_supported", "adjudicated_supported")
            evidence_supported = _binary(row, "adjudicated_evidence_supported", "adjudicated_supported")
            if not record_id or source_supported is None or evidence_supported is None:
                continue
            risk_label = str(row.get("adjudicated_risk_label") or "").strip() or None
            risk_score_raw = str(row.get("adjudicated_risk_score") or "").strip()
            risk_score = int(risk_score_raw) if risk_score_raw.isdigit() and 1 <= int(risk_score_raw) <= 5 else None
            allowed_risks = {"Auto-Renewal", "Hidden Fee", "Liability Waiver", "Data Sharing", "Penalty Clause", "Safe"}
            if risk_label is not None and risk_label not in allowed_risks:
                continue
            chunk_id = str(row.get("adjudicated_chunk_id") or "").strip() or None
            if evidence_supported == 1 and not chunk_id:
                continue
            if evidence_supported == 0:
                chunk_id = None
            output[record_id] = {
                "source_supported": source_supported,
                "evidence_supported": evidence_supported,
                "chunk_id": chunk_id,
                "risk_label": risk_label,
                "risk_score": risk_score,
            }
    return output


def apply_adjudicated_reviews(records: list[FLBRecord], path: Path) -> dict[str, int]:
    """swaps in reviewer-adjudicated labels over the provisional ones, in place."""
    reviews = load_adjudicated_reviews(path)
    applied = 0
    for record in records:
        review = reviews.get(record.record_id)
        if review is None:
            continue
        record.supported = int(review["source_supported"])
        record.evidence_supported = int(review["evidence_supported"])
        record.ground_truth_chunk_id = str(review["chunk_id"]) if review["chunk_id"] else None
        record.evidence_label_source = "human_adjudicated"
        if review.get("risk_label"):
            record.risk_label = str(review["risk_label"])
            record.human_risk_label = str(review["risk_label"])
            record.risk_label_source = "human_adjudicated"
        if review.get("risk_score") is not None:
            record.human_risk_score = int(review["risk_score"])
        record.human_reviewed = True
        applied += 1
    return {"available": len(reviews), "applied": applied, "unmatched": len(reviews) - applied}


def _detect_reviewer_count(fieldnames: list[str]) -> int:
    count = 0
    for i in range(1, MAX_REVIEWERS + 1):
        if any(name.startswith(f"reviewer_{i}_") for name in fieldnames):
            count = i
    return count


def score_review_file(path: Path) -> dict:
    """scores a review sheet - auto-detects how many reviewer columns exist, uses Cohen's kappa for 2 raters, Fleiss' for 3+."""
    with path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        n_reviewers = _detect_reviewer_count(fieldnames)
        rows = list(reader)

    total_rows = len(rows)
    source_rows: list[list[int]] = []
    evidence_rows: list[list[int]] = []
    risk_rows: list[list[str]] = []
    risk_score_rows: list[list[int]] = []
    adjudicated = 0
    conflicts_source = 0
    conflicts_evidence = 0
    conflicts_risk = 0
    adjudicated_class_distribution: Counter = Counter()

    for row in rows:
        source_vals: list[int] = []
        evidence_vals: list[int] = []
        risk_vals: list[str] = []
        risk_score_vals: list[int] = []
        for i in range(1, n_reviewers + 1):
            s = _binary(row, f"reviewer_{i}_source_supported", f"reviewer_{i}_supported", f"reviewer_{i}_meaning_preserved")
            e = _binary(row, f"reviewer_{i}_evidence_supported", f"reviewer_{i}_supported")
            if e is None:
                e = _binary_inverted(row, f"reviewer_{i}_missing_or_changed")
            r = str(row.get(f"reviewer_{i}_risk_label") or row.get(f"reviewer_{i}_risk_category") or "").strip()
            rs = str(row.get(f"reviewer_{i}_risk_score") or "").strip()
            if s is not None:
                source_vals.append(s)
            if e is not None:
                evidence_vals.append(e)
            if r:
                risk_vals.append(r)
            if rs.isdigit() and 1 <= int(rs) <= 5:
                risk_score_vals.append(int(rs))

        if len(source_vals) >= 2:
            source_rows.append(source_vals)
            if len(set(source_vals)) > 1:
                conflicts_source += 1
        if len(evidence_vals) >= 2:
            evidence_rows.append(evidence_vals)
            if len(set(evidence_vals)) > 1:
                conflicts_evidence += 1
        if len(risk_vals) >= 2:
            risk_rows.append(risk_vals)
            if len(set(risk_vals)) > 1:
                conflicts_risk += 1
        if len(risk_score_vals) >= 2:
            risk_score_rows.append(risk_score_vals)

        adjudicated_source = _binary(row, "adjudicated_source_supported", "adjudicated_supported")
        adjudicated_evidence = _binary(row, "adjudicated_evidence_supported", "adjudicated_supported")
        if adjudicated_source is not None and adjudicated_evidence is not None:
            adjudicated += 1
            adjudicated_label = str(row.get("adjudicated_risk_label") or "").strip()
            if adjudicated_label:
                adjudicated_class_distribution[adjudicated_label] += 1

    def _kappa(pairs_2: list[list[int]] | list[list[str]], is_binary: bool) -> float | None:
        if not pairs_2:
            return None
        if n_reviewers <= 2:
            a = [row[0] for row in pairs_2]
            b = [row[1] for row in pairs_2]
            return float(cohen_kappa_score(a, b))
        return fleiss_kappa(pairs_2)

    completion_fraction = (adjudicated / total_rows) if total_rows else 0.0

    return {
        "total_rows_in_sheet": total_rows,
        "reviewer_count_detected": n_reviewers,
        "agreement_method": "cohen_kappa" if n_reviewers <= 2 else "fleiss_kappa",
        "source_double_reviewed": len(source_rows),
        "source_kappa": _kappa(source_rows, is_binary=True),
        "evidence_double_reviewed": len(evidence_rows),
        "evidence_kappa": _kappa(evidence_rows, is_binary=True),
        "risk_double_reviewed": len(risk_rows),
        "risk_kappa": _kappa(risk_rows, is_binary=False),
        "risk_score_double_reviewed": len(risk_score_rows),
        "risk_score_quadratic_kappa": (
            float(cohen_kappa_score(risk_score_rows[0], risk_score_rows[1], weights="quadratic"))
            if risk_score_rows and n_reviewers <= 2 and len(risk_score_rows[0]) == 2
            else None
        ),
        "adjudicated": adjudicated,
        "adjudication_completion_fraction": completion_fraction,
        "conflict_count": {
            "source_supported": conflicts_source,
            "evidence_supported": conflicts_evidence,
            "risk_label": conflicts_risk,
        },
        "adjudicated_class_distribution": dict(adjudicated_class_distribution),
        "complete": bool(source_rows and evidence_rows and adjudicated >= max(len(source_rows), len(evidence_rows))),
        "source_cohen_kappa": _kappa(source_rows, is_binary=True) if n_reviewers <= 2 else None,
        "evidence_cohen_kappa": _kappa(evidence_rows, is_binary=True) if n_reviewers <= 2 else None,
        "risk_cohen_kappa": _kappa(risk_rows, is_binary=False) if n_reviewers <= 2 else None,
    }


REQUIRED_GOLD_FIELDS = (
    "adjudicated_source_supported",
    "adjudicated_evidence_supported",
    "adjudicated_risk_label",
)


def validate_gold_export_ready(path: Path) -> dict:
    """raises if any row is missing mandatory adjudication fields - no partial gold exports."""
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    missing_rows = []
    for row in rows:
        record_id = row.get("record_id", "<unknown>")
        missing = [field for field in REQUIRED_GOLD_FIELDS if not str(row.get(field) or "").strip()]
        if missing:
            missing_rows.append({"record_id": record_id, "missing_fields": missing})
    if missing_rows:
        raise RuntimeError(
            f"Cannot produce a frozen FLB-Gold export: {len(missing_rows)}/{len(rows)} rows are missing "
            f"mandatory adjudication fields. First few: {missing_rows[:5]}"
        )
    return {"status": "ready", "rows": len(rows), "checked_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
