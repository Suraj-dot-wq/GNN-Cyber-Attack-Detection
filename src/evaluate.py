import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from model import GCN


GRAPH_FILE = "data/test_graph.pt"
MODEL_FILE = "data/gnn_model.pt"

device = torch.device("cpu")


print("Loading test graph...")

graph = torch.load(
    GRAPH_FILE,
    map_location=device,
    weights_only=False
)

graph = graph.to(device)

print("Nodes:", graph.num_nodes)
print("Features:", graph.num_node_features)
print("Edges:", graph.num_edges)


print("\nLoading trained GCN model...")

model = GCN(
    input_features=graph.num_node_features,
    hidden_features=64,
    num_classes=2
).to(device)

model.load_state_dict(
    torch.load(
        MODEL_FILE,
        map_location=device,
        weights_only=True
    )
)

model.eval()

print("Model loaded successfully.")


print("\nRunning predictions...")

with torch.no_grad():

    output = model(
        graph.x,
        graph.edge_index
    )

    predictions = output.argmax(
        dim=1
    )

    predictions = predictions.cpu().numpy()
    labels = graph.y.cpu().numpy()


# --------------------------------
# Metrics
# --------------------------------

accuracy = accuracy_score(
    labels,
    predictions
)

precision = precision_score(
    labels,
    predictions,
    zero_division=0
)

recall = recall_score(
    labels,
    predictions,
    zero_division=0
)

f1 = f1_score(
    labels,
    predictions,
    zero_division=0
)

cm = confusion_matrix(
    labels,
    predictions
)


print("\n" + "=" * 60)
print("FINAL TEST RESULTS")
print("=" * 60)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")


print("\nConfusion Matrix:")
print(cm)


print("\nClassification Report:")

print(
    classification_report(
        labels,
        predictions,
        target_names=[
            "Benign",
            "Attack"
        ],
        zero_division=0
    )
)