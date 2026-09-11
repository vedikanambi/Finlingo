"""Trains and cross-validates a small calibration classifier on top of the
base verifier's raw NLI scores, using judge-labeled data.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("reports/verifier_calibration_labels.jsonl")
rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]

X = np.array([[r["entailment"], r["neutral"], r["contradiction"]] for r in rows])
y = np.array([r["judge_supported"] for r in rows])

print(f"n={len(y)}, positives={y.sum()}, negatives={len(y) - y.sum()}")

n_splits = min(5, int(y.sum()))
if n_splits < 2:
    print("Too few positive examples for meaningful cross-validation.")
    sys.exit(1)

skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
clf = LogisticRegression(class_weight="balanced", max_iter=1000)
y_pred = cross_val_predict(clf, X, y, cv=skf)

precision = precision_score(y, y_pred, zero_division=0)
recall = recall_score(y, y_pred, zero_division=0)
f1 = f1_score(y, y_pred, zero_division=0)
cm = confusion_matrix(y, y_pred)

print(f"\n{n_splits}-fold cross-validated results:")
print(f"precision={precision:.4f}  recall={recall:.4f}  f1={f1:.4f}")
print("confusion_matrix (rows=true[0,1], cols=pred[0,1]):")
print(cm)

print(f"\nTarget: precision>0.90, recall>0.80")
if precision > 0.90 and recall > 0.80:
    print("RESULT: PASS")
else:
    print("RESULT: FAIL")

clf.fit(X, y)
print(f"\nFull-fit coefficients (entailment, neutral, contradiction): {clf.coef_[0]}")
print(f"Full-fit intercept: {clf.intercept_[0]}")
