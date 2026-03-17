import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from model import GraphSAGE

print("Loading graph...")
data = torch.load("data/processed/transaction_graph.pt", weights_only=False)

# ── SHARED TRAIN / TEST SPLIT (same as train.py) ─────────────────────────────
X = data.x.numpy()
y = data.y.numpy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ── XGBOOST BASELINE ─────────────────────────────────────────────────────────
print("Training XGBoost baseline...")

n_legit = (y_train == 0).sum()
n_fraud = (y_train == 1).sum()
scale   = n_legit / n_fraud          # same class weighting as GNN

xgb = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    scale_pos_weight=scale,          # handles class imbalance
    eval_metric="aucpr",
    random_state=42,
    verbosity=0
)
xgb.fit(X_train, y_train)

xgb_probs = xgb.predict_proba(X_test)[:, 1]
xgb_roc   = roc_auc_score(y_test, xgb_probs)
p, r, _   = precision_recall_curve(y_test, xgb_probs)
xgb_pr    = auc(r, p)

print(f"XGBoost  →  ROC-AUC: {xgb_roc:.4f}  |  PR-AUC: {xgb_pr:.4f}")

# ── GNN (load saved best model) ───────────────────────────────────────────────
print("Loading GNN...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = GraphSAGE(input_dim=data.num_features).to(device)
model.load_state_dict(torch.load("best_model.pt", map_location=device))
model.eval()
data = data.to(device)

with torch.no_grad():
    out       = model(data.x, data.edge_index)
    gnn_probs = torch.exp(out[:, 1]).cpu().numpy()

# evaluate on same test indices
_, test_idx = train_test_split(
    np.arange(len(y)), test_size=0.2, stratify=y, random_state=42
)
gnn_probs_test = gnn_probs[test_idx]
gnn_roc = roc_auc_score(y_test, gnn_probs_test)
p, r, _ = precision_recall_curve(y_test, gnn_probs_test)
gnn_pr  = auc(r, p)

print(f"GNN      →  ROC-AUC: {gnn_roc:.4f}  |  PR-AUC: {gnn_pr:.4f}")

# ── COMPARISON TABLE ─────────────────────────────────────────────────────────
print("\n── Comparison ──────────────────────────────────────────")
print(f"{'Model':<12} {'ROC-AUC':>10} {'PR-AUC':>10} {'ROC lift':>12}")
print(f"{'XGBoost':<12} {xgb_roc:>10.4f} {xgb_pr:>10.4f} {'baseline':>12}")
print(f"{'GNN':<12} {gnn_roc:>10.4f} {gnn_pr:>10.4f} {((gnn_roc-xgb_roc)/xgb_roc*100):>+11.1f}%")
print("────────────────────────────────────────────────────────")

# ── COMPARISON PLOT ───────────────────────────────────────────────────────────
models  = ["XGBoost\n(no graph)", "GNN\n(GraphSAGE)"]
roc_scores = [xgb_roc, gnn_roc]
pr_scores  = [xgb_pr,  gnn_pr]
colors     = ["#B4B2A9", "#534AB7"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))

bars1 = ax1.bar(models, roc_scores, color=colors, width=0.4)
ax1.set_ylim(0, 1)
ax1.set_title("ROC-AUC Comparison", fontsize=13)
ax1.set_ylabel("ROC-AUC")
for bar, val in zip(bars1, roc_scores):
    ax1.text(bar.get_x() + bar.get_width()/2, val + 0.01,
             f"{val:.3f}", ha="center", fontsize=12, fontweight="bold")

bars2 = ax2.bar(models, pr_scores, color=colors, width=0.4)
ax2.set_ylim(0, 1)
ax2.set_title("PR-AUC Comparison", fontsize=13)
ax2.set_ylabel("PR-AUC")
for bar, val in zip(bars2, pr_scores):
    ax2.text(bar.get_x() + bar.get_width()/2, val + 0.01,
             f"{val:.3f}", ha="center", fontsize=12, fontweight="bold")

plt.suptitle("GNN vs XGBoost — Fraud Detection", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("baseline_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

print("Saved: baseline_comparison.png")