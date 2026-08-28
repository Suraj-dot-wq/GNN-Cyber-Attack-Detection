import pandas as pd
import torch
from torch_geometric.data import Data
from sklearn.neighbors import NearestNeighbors


TEST_FILE = "data/test.csv"
GRAPH_FILE = "data/test_graph.pt"

K = 5


print("Loading test data...")

df = pd.read_csv(TEST_FILE)

X = df.drop(columns=["label"]).values
y = df["label"].values

print("Test samples:", len(df))
print("Features:", X.shape[1])


# Convert features and labels to tensors
x = torch.tensor(
    X,
    dtype=torch.float
)

y = torch.tensor(
    y,
    dtype=torch.long
)


# --------------------------------
# Build k-NN graph
# --------------------------------

print(f"\nBuilding k-NN graph with k={K}...")

knn = NearestNeighbors(
    n_neighbors=K + 1,
    metric="euclidean"
)

knn.fit(X)

distances, indices = knn.kneighbors(X)


# First neighbor is the node itself,
# so skip it.
indices = indices[:, 1:]


source_nodes = []
target_nodes = []


for i in range(len(df)):

    for j in indices[i]:

        source_nodes.append(i)
        target_nodes.append(j)


edge_index = torch.tensor(
    [source_nodes, target_nodes],
    dtype=torch.long
)


# --------------------------------
# Create PyG graph
# --------------------------------

graph = Data(
    x=x,
    edge_index=edge_index,
    y=y
)


print("\nGraph created successfully!")

print("Nodes:", graph.num_nodes)
print("Features:", graph.num_node_features)
print("Edges:", graph.num_edges)

print("Feature tensor shape:", graph.x.shape)
print("Edge index shape:", graph.edge_index.shape)
print("Label tensor shape:", graph.y.shape)


# --------------------------------
# Save
# --------------------------------

torch.save(
    graph,
    GRAPH_FILE
)

print(f"\nTest graph saved to: {GRAPH_FILE}")