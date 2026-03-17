# Graph Neural Network — Credit Card Fraud Detection

> Detecting fraudulent transactions by modelling the **relationship structure** between accounts, devices, and emails using a 3-layer GraphSAGE network.

---

## Results

| Model | ROC-AUC | PR-AUC |
|---|---|---|
| XGBoost (no graph, baseline) | 0.882 | 0.534 |
| **GraphSAGE GNN** | **0.911** | **0.498** |

> The GNN achieves higher ROC-AUC (0.911 vs 0.882) but lower PR-AUC than XGBoost.
> This is a known tradeoff: the graph structure helps the model rank fraud nodes
> higher overall, while XGBoost's engineered V-features give it an edge at
> precision-recall tradeoffs. A production system would ensemble both.

## Why a Graph Neural Network?

Traditional ML models (XGBoost, Random Forest) treat each transaction independently. They cannot see that two transactions share the same device, email, or billing address — which are strong fraud signals.

This project models transactions as a **graph**:
- **Nodes** = individual transactions (~50,000)
- **Edges** = two transactions share a card number, address, or email domain
- **Node label** = fraudulent (1) or legitimate (0)

A GNN learns that fraud nodes form suspicious clusters in this graph — a pattern invisible to flat feature-based models.

---

## Model Architecture

```
Input features (160 dims)
        ↓
SAGEConv Layer 1  →  256 units, ReLU, Dropout 0.3
        ↓
SAGEConv Layer 2  →  128 units, ReLU, Dropout 0.3
        ↓
SAGEConv Layer 3  →   64 units, ReLU
        ↓
Linear Classifier →    2 units, log-softmax
```

**Key design choices:**
- **GraphSAGE** over GCN — neighbourhood sampling scales to 500K+ nodes
- **3 layers** — each node sees its 3-hop neighbourhood context
- **Weighted cross-entropy loss** — fraud class upweighted ~27× to handle 96.5/3.5 class imbalance
- **Early stopping** — halts training when validation AUC stops improving (patience = 20 checks)

---

## Dataset

**IEEE-CIS Fraud Detection** — sourced from real e-commerce transactions by Vesta Corporation, hosted on [Kaggle](https://www.kaggle.com/c/ieee-fraud-detection).

| Property | Value |
|---|---|
| Total transactions | 590,540 |
| Fraud rate | 3.5% |
| Features used | 160 (TransactionAmt, card1-5, addr, dist, C1-C14, D1-D4, V1-V150) |
| Training sample | 50,000 (stratified) |

### Graph construction

Transactions are connected if they share any of the following:

| Edge type | Column | Fraud signal |
|---|---|---|
| Same card | `card1`, `card2` | High |
| Same billing address | `addr1` | High |
| Same email domain | `P_emaildomain` | Medium |

---

## Project Structure

```
fraud-gnn-project/
│
├── data/
│   ├── raw/                        ← original Kaggle CSVs (not committed)
│   │   ├── train_transaction.csv
│   │   └── train_identity.csv
│   └── processed/
│       └── transaction_graph.pt    ← built graph (PyG Data object)
│
├── notebooks/
│   ├── 01_eda.ipynb                ← exploratory data analysis
│   └── 02_graph_visualization.ipynb
│
├── src/
│   ├── build_graph.py              ← CSV → PyG graph
│   ├── model.py                    ← GraphSAGE architecture
│   ├── train.py                    ← training loop with early stopping
│   └── evaluate.py                 ← metrics + ROC / PR curves
│
├── best_model.pt                   ← saved best checkpoint
├── roc_curve.png                   ← ROC curve output
├── requirements.txt
└── README.md
```

---

## Reproducing the Results

### 1. Clone and set up environment

```bash
git clone https://github.com/YOUR_USERNAME/fraud-gnn-project.git
cd fraud-gnn-project

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac / Linux

pip install -r requirements.txt
```

### 2. Download the dataset

Go to [kaggle.com/c/ieee-fraud-detection](https://www.kaggle.com/c/ieee-fraud-detection), download the data, and place the files at:

```
data/raw/train_transaction.csv
data/raw/train_identity.csv
```

### 3. Build the graph

```bash
python src/build_graph.py
```

Outputs: `data/processed/transaction_graph.pt`
Expected: ~50,000 nodes, ~2M+ edges

### 4. Train the model

```bash
python src/train.py
```

Training stops automatically when validation AUC plateaus (early stopping).
Best checkpoint saved to `best_model.pt`.

### 5. Evaluate

```bash
python src/evaluate.py
```

Outputs: ROC-AUC, PR-AUC, `roc_curve.png`, `pr_curve.png`

---

## Requirements

```
torch>=2.0.0
torch-geometric>=2.3.0
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
matplotlib>=3.6.0
seaborn>=0.12.0
networkx>=3.0
```

Install all at once:

```bash
pip install -r requirements.txt
```

---

## What I Learned

This project was built from scratch as a learning exercise in graph-based machine learning. Key lessons:

- **Graph construction is the hardest part** — deciding which columns become edges, and capping hub nodes to prevent one node dominating aggregation
- **Class imbalance kills PR-AUC** — unweighted loss resulted in ROC-AUC of 0.63; weighted loss pushed it to 0.91
- **Early stopping prevents overfitting** — the model peaked at epoch 75-105 across runs; training beyond that consistently hurt PR-AUC
- **Feature richness matters more than architecture** — going from 5 features to 160 (including V-columns) had more impact than adding a third GNN layer

---

## Potential Extensions

- [ ] **GNNExplainer** — identify which edges caused each fraud prediction (model interpretability)
- [ ] **Heterogeneous graph** — separate node types for transactions, cards, and devices
- [ ] **XGBoost comparison** — quantify exactly how much the graph structure adds
- [ ] **FastAPI deployment** — wrap the model in a REST endpoint for real-time scoring

---

## References

- Hamilton et al. (2017) — [Inductive Representation Learning on Large Graphs](https://arxiv.org/abs/1706.02216) (GraphSAGE paper)
- [PyTorch Geometric Documentation](https://pytorch-geometric.readthedocs.io/)
- [IEEE-CIS Fraud Detection — Kaggle](https://www.kaggle.com/c/ieee-fraud-detection)
- [CS224W: Machine Learning with Graphs — Stanford](http://web.stanford.edu/class/cs224w/)

