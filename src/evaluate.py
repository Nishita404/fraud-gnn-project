import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, roc_curve,
    auc, confusion_matrix, ConfusionMatrixDisplay, f1_score
)
from model import GraphSAGE

print("Loading graph...")
data = torch.load("data/processed/transaction_graph.pt", weights_only=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = GraphSAGE(input_dim=data.num_features).to(device)
model.load_state_dict(torch.load("best_model.pt", map_location=device))
model.eval()

data = data.to(device)

with torch.no_grad():
    out   = model(data.x, data.edge_index)
    probs = torch.exp(out[:, 1]).cpu().numpy()

y_true = data.y.cpu().numpy()

# ── CORE METRICS ─────────────────────────────────────────────────────────────
roc_auc = roc_auc_score(y_true, probs)
precision, recall, pr_thresholds = precision_recall_curve(y_true, probs)
pr_auc = auc(recall, precision)

print(f"ROC-AUC : {roc_auc:.4f}   (target: > 0.85)")
print(f"PR-AUC  : {pr_auc:.4f}   (target: > 0.40)")

# ── FIND BEST THRESHOLD (maximises F1 score) ─────────────────────────────────
f1_scores   = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-8)
best_idx    = np.argmax(f1_scores)
best_threshold = pr_thresholds[best_idx]
best_f1        = f1_scores[best_idx]

print(f"\nBest threshold : {best_threshold:.4f}")
print(f"Best F1 score  : {best_f1:.4f}")

y_pred = (probs >= best_threshold).astype(int)

# ── CONFUSION MATRIX ─────────────────────────────────────────────────────────
cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()

print(f"\nConfusion Matrix at threshold {best_threshold:.3f}:")
print(f"  True Negatives  (correctly caught legit) : {tn:,}")
print(f"  False Positives (legit flagged as fraud)  : {fp:,}")
print(f"  False Negatives (fraud missed)            : {fn:,}")
print(f"  True Positives  (fraud correctly caught)  : {tp:,}")
print(f"\n  Precision : {tp/(tp+fp):.3f}  (of flagged, how many were real fraud)")
print(f"  Recall    : {tp/(tp+fn):.3f}  (of all fraud, how many were caught)")
print(f"  F1 Score  : {best_f1:.3f}")

# ── PLOT 1: ROC CURVE ─────────────────────────────────────────────────────────
fpr, tpr, _ = roc_curve(y_true, probs)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color="#534AB7", linewidth=2, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
plt.fill_between(fpr, tpr, alpha=0.08, color="#534AB7")
plt.title("ROC Curve", fontsize=14)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("roc_curve.png", dpi=150)
plt.show()

# ── PLOT 2: PRECISION-RECALL CURVE ───────────────────────────────────────────
plt.figure(figsize=(6, 5))
plt.plot(recall, precision, color="#1D9E75", linewidth=2, label=f"PR-AUC = {pr_auc:.3f}")
plt.scatter(recall[best_idx], precision[best_idx],
            color="#D85A30", zorder=5, s=80,
            label=f"Best threshold = {best_threshold:.3f}")
plt.title("Precision-Recall Curve", fontsize=14)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.tight_layout()
plt.savefig("pr_curve.png", dpi=150)
plt.show()

# ── PLOT 3: CONFUSION MATRIX ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 4))
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Legitimate", "Fraud"]
)
disp.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title(f"Confusion Matrix\n(threshold = {best_threshold:.3f}, F1 = {best_f1:.3f})",
             fontsize=12)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()

# ── PLOT 4: THRESHOLD vs PRECISION / RECALL / F1 ─────────────────────────────
plt.figure(figsize=(7, 4))
plt.plot(pr_thresholds, precision[:-1], color="#534AB7", label="Precision")
plt.plot(pr_thresholds, recall[:-1],    color="#1D9E75", label="Recall")
plt.plot(pr_thresholds, f1_scores,      color="#D85A30", label="F1")
plt.axvline(best_threshold, color="gray", linestyle="--", linewidth=1,
            label=f"Best threshold = {best_threshold:.3f}")
plt.title("Threshold vs Precision / Recall / F1", fontsize=14)
plt.xlabel("Decision Threshold")
plt.ylabel("Score")
plt.legend()
plt.tight_layout()
plt.savefig("threshold_analysis.png", dpi=150)
plt.show()

print("\nAll plots saved:")
print("  roc_curve.png, pr_curve.png, confusion_matrix.png, threshold_analysis.png")