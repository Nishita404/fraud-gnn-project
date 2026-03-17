import pandas as pd
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler

print("Loading dataset...")

df_transaction = pd.read_csv("data/raw/train_transaction.csv")
df_identity    = pd.read_csv("data/raw/train_identity.csv")
df = df_transaction.merge(df_identity, on="TransactionID", how="left")

df = df.sample(50000, random_state=42).reset_index(drop=True)
print("Sampled shape:", df.shape)

# ── MANY MORE V-COLUMNS (was 50, now 150) ─────────────────────────────────────
v_cols = [c for c in df.columns if c.startswith("V")][:150]

feature_cols = [
    "TransactionAmt",
    "card1", "card2", "card3", "card5",
    "addr1", "addr2",
    "dist1", "dist2",
    "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10",
    "C11", "C12", "C13", "C14",          # C-cols = counting features
    "D1", "D2", "D3", "D4",              # D-cols = time delta features
] + v_cols

# Only keep columns that actually exist in the dataframe
feature_cols = [c for c in feature_cols if c in df.columns]
print(f"Using {len(feature_cols)} features")

df_features = df[feature_cols].fillna(0)
scaler = StandardScaler()
node_features = torch.tensor(scaler.fit_transform(df_features), dtype=torch.float)
print("Node feature shape:", node_features.shape)

labels = torch.tensor(df["isFraud"].values, dtype=torch.long)

# ── EDGES (same as before) ───────────────────────────────────────────────────
print("Building edges...")
edges = []

def add_edges_for_column(col):
    if col not in df.columns:
        return
    valid  = df[col].dropna()
    groups = df.loc[valid.index].groupby(col).indices
    for _, indices in groups.items():
        if len(indices) < 2 or len(indices) > 100:
            continue
        for i in indices:
            for j in indices:
                if i != j:
                    edges.append([i, j])

add_edges_for_column("card1")
add_edges_for_column("card2")
add_edges_for_column("addr1")
add_edges_for_column("P_emaildomain")

print("Total edges:", len(edges))

edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

data = Data(x=node_features, edge_index=edge_index, y=labels)
print(data)

torch.save(data, "data/processed/transaction_graph.pt")
print("Graph saved!")