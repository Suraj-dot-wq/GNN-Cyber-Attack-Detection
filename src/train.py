import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from torch_geometric.utils import to_undirected

from model import GCN


# -----------------------------
# Configuration
# -----------------------------
GRAPH_FILE = "data/train_graph.pt"
MODEL_FILE = "data/gnn_model.pt"

HIDDEN_FEATURES = 64
NUM_CLASSES = 2

LEARNING_RATE = 0.01
WEIGHT_DECAY = 5e-4
EPOCHS = 100

TRAIN_RATIO = 0.80


# -----------------------------
# Device
# -----------------------------
device = torch.device("cpu")

print("Using device:", device)


# -----------------------------
# Load graph
# -----------------------------
print("\nLoading graph...")

graph = torch.load(
    GRAPH_FILE,
    map_location=device,
    weights_only=False
)

graph = graph.to(device)

print(f"Nodes: {graph.num_nodes}")
print(f"Features: {graph.num_node_features}")
print(f"Edges: {graph.num_edges}")


# -----------------------------
# Make graph undirected
# -----------------------------
print("\nConverting graph to undirected...")

graph.edge_index = to_undirected(
    graph.edge_index
)

print(f"Edges after conversion: {graph.num_edges}")


# -----------------------------
# Create train/validation masks
# -----------------------------
num_nodes = graph.num_nodes

generator = torch.Generator().manual_seed(42)

random_indices = torch.randperm(
    num_nodes,
    generator=generator
)

train_size = int(
    TRAIN_RATIO * num_nodes
)

train_indices = random_indices[:train_size]
val_indices = random_indices[train_size:]


train_mask = torch.zeros(
    num_nodes,
    dtype=torch.bool
)

val_mask = torch.zeros(
    num_nodes,
    dtype=torch.bool
)

train_mask[train_indices] = True
val_mask[val_indices] = True

graph.train_mask = train_mask
graph.val_mask = val_mask


print("\nTraining/Validation split:")
print("Training nodes:", train_mask.sum().item())
print("Validation nodes:", val_mask.sum().item())


# -----------------------------
# Calculate class weights
# -----------------------------
train_labels = graph.y[train_mask]

class_counts = torch.bincount(
    train_labels,
    minlength=NUM_CLASSES
).float()

print("\nTraining class distribution:")

print("Benign:", int(class_counts[0].item()))
print("Attack:", int(class_counts[1].item()))


class_weights = class_counts.sum() / (
    NUM_CLASSES * class_counts
)

print("\nClass weights:")
print("Benign weight:", class_weights[0].item())
print("Attack weight:", class_weights[1].item())


class_weights = class_weights.to(device)


# -----------------------------
# Create model
# -----------------------------
model = GCN(
    input_features=graph.num_node_features,
    hidden_features=HIDDEN_FEATURES,
    num_classes=NUM_CLASSES
).to(device)


print("\nModel:")
print(model)


# -----------------------------
# Loss and optimizer
# -----------------------------
criterion = torch.nn.CrossEntropyLoss(
    weight=class_weights
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# -----------------------------
# Training
# -----------------------------
print("\nStarting training...\n")

best_val_f1 = 0.0

for epoch in range(1, EPOCHS + 1):

    model.train()

    optimizer.zero_grad()

    output = model(
        graph.x,
        graph.edge_index
    )

    loss = criterion(
        output[train_mask],
        graph.y[train_mask]
    )

    loss.backward()

    optimizer.step()


    # -------------------------
    # Validation
    # -------------------------
    model.eval()

    with torch.no_grad():

        output = model(
            graph.x,
            graph.edge_index
        )

        predictions = output.argmax(
            dim=1
        )

        val_predictions = predictions[
            val_mask
        ].cpu().numpy()

        val_labels = graph.y[
            val_mask
        ].cpu().numpy()


    val_accuracy = accuracy_score(
        val_labels,
        val_predictions
    )

    val_precision = precision_score(
        val_labels,
        val_predictions,
        zero_division=0
    )

    val_recall = recall_score(
        val_labels,
        val_predictions,
        zero_division=0
    )

    val_f1 = f1_score(
        val_labels,
        val_predictions,
        zero_division=0
    )


    if val_f1 > best_val_f1:

        best_val_f1 = val_f1

        torch.save(
            model.state_dict(),
            MODEL_FILE
        )


    if epoch == 1 or epoch % 10 == 0:

        print(
            f"Epoch {epoch:03d} | "
            f"Loss: {loss.item():.4f} | "
            f"Val Accuracy: {val_accuracy:.4f} | "
            f"Val Precision: {val_precision:.4f} | "
            f"Val Recall: {val_recall:.4f} | "
            f"Val F1: {val_f1:.4f}"
        )


# -----------------------------
# Load best model
# -----------------------------
print("\nLoading best model...")

model.load_state_dict(
    torch.load(
        MODEL_FILE,
        map_location=device,
        weights_only=True
    )
)

model.eval()


# -----------------------------
# Final validation evaluation
# -----------------------------
with torch.no_grad():

    output = model(
        graph.x,
        graph.edge_index
    )

    predictions = output.argmax(
        dim=1
    )

    final_predictions = predictions[
        val_mask
    ].cpu().numpy()

    final_labels = graph.y[
        val_mask
    ].cpu().numpy()


# -----------------------------
# Metrics
# -----------------------------
accuracy = accuracy_score(
    final_labels,
    final_predictions
)

precision = precision_score(
    final_labels,
    final_predictions,
    zero_division=0
)

recall = recall_score(
    final_labels,
    final_predictions,
    zero_division=0
)

f1 = f1_score(
    final_labels,
    final_predictions,
    zero_division=0
)

cm = confusion_matrix(
    final_labels,
    final_predictions
)


print("\n" + "=" * 60)
print("FINAL VALIDATION RESULTS")
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
        final_labels,
        final_predictions,
        target_names=["Benign", "Attack"],
        zero_division=0
    )
)

print("\nBest Validation F1:", best_val_f1)

print(f"\nModel saved to: {MODEL_FILE}")