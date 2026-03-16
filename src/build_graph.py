import pandas as pd
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler

print("Loading dataset...")

# -----------------------------
# Load datasets
# -----------------------------

transaction_path = "data/raw/train_transaction.csv"
identity_path = "data/raw/train_identity.csv"

df_transaction = pd.read_csv(transaction_path)
df_identity = pd.read_csv(identity_path)

# Merge both datasets
df = df_transaction.merge(df_identity, on="TransactionID", how="left")

print("Original dataset shape:", df.shape)

# -----------------------------
# Sample smaller dataset
# -----------------------------

df = df.sample(50000, random_state=42).reset_index(drop=True)

print("Sampled dataset shape:", df.shape)

# -----------------------------
# Select node features
# -----------------------------

feature_cols = [
    "TransactionAmt",
    "card1",
    "card2",
    "card3",
    "card5"
]

df_features = df[feature_cols].fillna(0)

# Normalize features
scaler = StandardScaler()
df_features = scaler.fit_transform(df_features)

# Convert to tensor
node_features = torch.tensor(df_features, dtype=torch.float)

print("Node feature shape:", node_features.shape)

# -----------------------------
# Create labels
# -----------------------------

labels = torch.tensor(df["isFraud"].values, dtype=torch.long)

# -----------------------------
# Build graph edges
# -----------------------------

print("Building graph edges...")

edges = []

groups = df.groupby("card1").indices

for card, indices in groups.items():

    if len(indices) > 1:

        for i in indices:
            for j in indices:

                if i != j:
                    edges.append([i, j])

print("Total edges created:", len(edges))

# Convert edges to tensor
edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

print("Edge index shape:", edge_index.shape)

# -----------------------------
# Create graph object
# -----------------------------

data = Data(
    x=node_features,
    edge_index=edge_index,
    y=labels
)

print(data)

# -----------------------------
# Save graph
# -----------------------------

torch.save(data, "data/processed/transaction_graph.pt")

print("Graph saved successfully!")