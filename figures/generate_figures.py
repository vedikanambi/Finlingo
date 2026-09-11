"""Generate the RQ2/RQ3 report figures from reports/*.json.

Run from the project root: python figures/generate_figures.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
OUT = Path(__file__).resolve().parent

d = json.loads((REPORTS / "evaluation_trained_s5_frozen_e2e.json").read_text())
rq2 = d["rq2"]
labels = rq2["labels"]
cm = np.array(rq2["confusion_matrix"])

fig, ax = plt.subplots(figsize=(7.5, 6.5))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(len(labels)))
ax.set_yticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=45, ha="right")
ax.set_yticklabels(labels)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title("RQ2: Risk classification confusion matrix (n=91)")
vmax = cm.max()
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        val = cm[i, j]
        color = "#68C2D0" if val > vmax * 0.6 else "#B83939"
        ax.text(j, i, str(val), ha="center", va="center", color=color, fontsize=9)
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.tight_layout()
fig.savefig(OUT / "rq2_confusion.png", dpi=200)
plt.close(fig)

classes = ["Auto-Renewal", "Hidden Fee", "Liability Waiver", "Data Sharing", "Penalty Clause", "Safe"]
f1s = [rq2["per_class"][c]["f1-score"] for c in classes]
macro_f1 = rq2["macro_f1"]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(classes, f1s, color="#B8AB3C")
ax.axhline(0.75, color="#CD7AAC", linestyle="--", linewidth=1.5, label="Proposal target (0.75)")
ax.axhline(macro_f1, color="#9DD4A7", linestyle=":", linewidth=1.5, label=f"Achieved macro-F1 ({macro_f1:.3f})")
ax.set_ylim(0, 1.05)
ax.set_ylabel("F1-score")
ax.set_title("RQ2: Per-class F1 vs proposal target")
for b, v in zip(bars, f1s):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center", fontsize=8)
plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "rq2_perclass_f1.png", dpi=200)
plt.close(fig)

# same cutoffs pre/post rerank - had these mismatched (8 vs 5) before, which made the reranker look worse than it is
deep_dive = json.loads((REPORTS / "retrieval_reranker_deep_dive.json").read_text())
cutoffs = ["8", "15", "30", "50"]
pre_vals = [deep_dive["pre_rerank_recall_at_k"][k] for k in cutoffs]
post_vals = [deep_dive["post_rerank_recall_at_k_over_full_set"][k] for k in cutoffs]
n_clauses = deep_dive["n_clauses"]

fig, ax = plt.subplots(figsize=(7, 5))
x = np.arange(len(cutoffs))
width = 0.35
b1 = ax.bar(x - width / 2, pre_vals, width, label="Pre-rerank (BM25 only)", color="#9573EC")
b2 = ax.bar(x + width / 2, post_vals, width, label="Post-rerank (cross-encoder)", color="#E89050")
ax.set_xticks(x)
ax.set_xticklabels([f"Recall@{k}" for k in cutoffs])
ax.set_ylabel("Recall")
ax.set_ylim(0, max(pre_vals) * 1.3)
ax.set_title(f"RQ3: Reranker recall at identical cutoffs (FLB, n={n_clauses} clauses)")
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005, f"{b.get_height():.3f}", ha="center", fontsize=8)
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "retrieval_gap.png", dpi=200)
plt.close(fig)

mv = json.loads((REPORTS / "multiverifier.json").read_text())
tm = mv["datasets"]["flb"]["systems"]["deberta"]["threshold_metrics"]
taus = ["0.65", "0.7", "0.75", "0.8"]
tau_labels = [0.65, 0.70, 0.75, 0.80]
precision = [tm[t]["precision"] for t in taus]
recall = [tm[t]["recall"] for t in taus]
accuracy = [tm[t]["accuracy"] for t in taus]

fig, ax = plt.subplots(figsize=(7.5, 5))
ax.plot(tau_labels, precision, marker="o", color="#425841", label="Precision")
ax.plot(tau_labels, recall, marker="s", color="#5E5B78", label="Recall")
ax.plot(tau_labels, accuracy, marker="^", color="#E1E522", label="Accuracy")
ax.axhline(0.90, color="#13EB3B", linestyle="--", linewidth=1, label="Precision target (0.90)")
ax.axhline(0.80, color="#2617F9", linestyle=":", linewidth=1, label="Recall target (0.80)")
ax.set_xlabel(r"Verifier threshold $\tau$")
ax.set_ylabel("Score")
ax.set_ylim(0, 1.05)
ax.set_title("RQ3: DeBERTa v3 verifier -- threshold sweep (n=182)")
ax.set_xticks(tau_labels)
ax.legend(fontsize=8, loc="lower left")
fig.tight_layout()
fig.savefig(OUT / "tau_sweep.png", dpi=200)
plt.close(fig)

print("Wrote 4 figures to", OUT)
