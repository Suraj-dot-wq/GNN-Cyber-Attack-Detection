import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data


# -----------------------------
# Configuration
# -----------------------------
TRAIN_FILE = "data/train.csv"
OUTPUT_FILE = "data/train_graph.pt"

K = 5


# -----------------------------
# Load training data
# -----------------------------
print("Loading training data...")

df = pd.read_csv(TRAIN_FILE)

print(f"Dataset shape: {df.shape}")


# -----------------------------
# Separate features and labels
# -----------------------------
X = df.drop(columns=["label"]).values
y = df["label"].values

print(f"Number of nodes: {X.shape[0]}")
print(f"Number of features: {X.shape[1]}")


# -----------------------------
# Build k-NN graph
# -----------------------------
print(f"\nBuilding k-NN graph with k={K}...")

knn = NearestNeighbors(
    n_neighbors=K + 1,
    metric="euclidean",
    n_jobs=-1
)

knn.fit(X)

distances, indices = knn.kneighbors(X)


# -----------------------------
# Create edges
# -----------------------------
# The first neighbor is the node itself,
# so we skip index 0.

source_nodes = []
target_nodes = []

for node in range(X.shape[0]):
    for neighbor_position in range(1, K + 1):
        neighbor = indices[node, neighbor_position]

        source_nodes.append(node)
        target_nodes.append(neighbor)


# Convert to PyTorch tensor
edge_index = torch.tensor(
    [source_nodes, target_nodes],
    dtype=torch.long
)


# -----------------------------
# Convert features and labels
# -----------------------------
x = torch.tensor(
    X,
    dtype=torch.float
)

y = torch.tensor(
    y,
    dtype=torch.long
)


# -----------------------------
# Create PyTorch Geometric Data
# -----------------------------
graph = Data(
    x=x,
    edge_index=edge_index,
    y=y
)


# -----------------------------
# Print graph information
# -----------------------------
print("\nGraph created successfully!")

print(f"Nodes: {graph.num_nodes}")
print(f"Node features: {graph.num_node_features}")
print(f"Edges: {graph.num_edges}")

print(f"Feature tensor shape: {graph.x.shape}")
print(f"Edge index shape: {graph.edge_index.shape}")
print(f"Label tensor shape: {graph.y.shape}")


# -----------------------------
# Save graph
# -----------------------------
torch.save(graph, OUTPUT_FILE)

print(f"\nGraph saved to: {OUTPUT_FILE}")