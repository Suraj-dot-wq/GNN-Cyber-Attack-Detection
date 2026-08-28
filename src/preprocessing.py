import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os


# -----------------------------
# Configuration
# -----------------------------
INPUT_FILE = "CIC_IoT_2023_10000.csv"
OUTPUT_DIR = "data"

TRAIN_FILE = os.path.join(OUTPUT_DIR, "train.csv")
TEST_FILE = os.path.join(OUTPUT_DIR, "test.csv")
SCALER_FILE = os.path.join(OUTPUT_DIR, "scaler.pkl")


# -----------------------------
# Create output directory
# -----------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -----------------------------
# Load dataset
# -----------------------------
print("Loading dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Original shape: {df.shape}")


# -----------------------------
# Remove duplicate rows
# -----------------------------
duplicates = df.duplicated().sum()

print(f"Duplicate rows: {duplicates}")

df = df.drop_duplicates().reset_index(drop=True)

print(f"Shape after removing duplicates: {df.shape}")


# -----------------------------
# Define target
# -----------------------------
TARGET = "label"

# Metadata columns are not used as model features
METADATA_COLUMNS = [
    "Label",
    "attack_class",
    "label"
]


# -----------------------------
# Create feature matrix X
# -----------------------------
X = df.drop(columns=METADATA_COLUMNS)

y = df[TARGET]


print(f"\nNumber of features: {X.shape[1]}")
print(f"Number of samples: {X.shape[0]}")


# -----------------------------
# Train/Test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTrain/Test split:")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")


# -----------------------------
# Feature scaling
# -----------------------------
print("\nScaling features...")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Convert back to DataFrames
X_train_scaled = pd.DataFrame(
    X_train_scaled,
    columns=X.columns
)

X_test_scaled = pd.DataFrame(
    X_test_scaled,
    columns=X.columns
)


# Add target column
X_train_scaled["label"] = y_train.values
X_test_scaled["label"] = y_test.values


# -----------------------------
# Save processed datasets
# -----------------------------
X_train_scaled.to_csv(TRAIN_FILE, index=False)
X_test_scaled.to_csv(TEST_FILE, index=False)

joblib.dump(scaler, SCALER_FILE)


# -----------------------------
# Final information
# -----------------------------
print("\nPreprocessing completed successfully!")

print(f"Training data saved to: {TRAIN_FILE}")
print(f"Testing data saved to: {TEST_FILE}")
print(f"Scaler saved to: {SCALER_FILE}")

print("\nTraining label distribution:")
print(y_train.value_counts())

print("\nTesting label distribution:")
print(y_test.value_counts())