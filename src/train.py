import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from model import GraphSAGE

print("Loading graph...")
data = torch.load("data/processed/transaction_graph.pt", weights_only=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = GraphSAGE(input_dim=data.num_features).to(device)
data   = data.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ── WEIGHT: back to 1.2x (1.5x was too aggressive) ───────────────────────────
n_legit = (data.y == 0).sum().item()
n_fraud = (data.y == 1).sum().item()
fraud_weight  = (n_legit / n_fraud) * 1.2
class_weights = torch.tensor([1.0, fraud_weight], dtype=torch.float).to(device)
print(f"Class weights — legit: 1.0, fraud: {fraud_weight:.1f}x")

# ── TRAIN / VAL SPLIT ────────────────────────────────────────────────────────
indices = torch.arange(data.num_nodes)
train_idx, val_idx = train_test_split(
    indices, test_size=0.2, stratify=data.y.cpu(), random_state=42
)
train_idx = train_idx.to(device)
val_idx   = val_idx.to(device)

# ── TRAINING LOOP with early stopping ────────────────────────────────────────
best_auc      = 0
best_epoch    = 0
patience      = 20   # stop if no improvement for 20 checks (= 100 epochs)
no_improve    = 0

for epoch in range(1, 201):

    model.train()
    optimizer.zero_grad()
    out  = model(data.x, data.edge_index)
    loss = F.nll_loss(out[train_idx], data.y[train_idx], weight=class_weights)
    loss.backward()
    optimizer.step()

    if epoch % 5 == 0:
        model.eval()
        with torch.no_grad():
            val_out   = model(data.x, data.edge_index)
            val_probs = torch.exp(val_out[:, 1])[val_idx].cpu().numpy()
            val_true  = data.y[val_idx].cpu().numpy()

        val_auc = roc_auc_score(val_true, val_probs)

        if val_auc > best_auc:
            best_auc   = val_auc
            best_epoch = epoch
            no_improve = 0
            torch.save(model.state_dict(), "best_model.pt")
        else:
            no_improve += 1

        print(f"Epoch {epoch:3d} | Loss {loss:.4f} | Val AUC {val_auc:.4f}"
              + (" ← best" if epoch == best_epoch else f"  (no improve: {no_improve}/{patience})"))

        # ── EARLY STOPPING: quit when stuck ──────────────────────────────────
        if no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch} — no improvement for {patience} checks.")
            break

print(f"\nBest AUC: {best_auc:.4f} at epoch {best_epoch}")
print("Best model saved to best_model.pt")