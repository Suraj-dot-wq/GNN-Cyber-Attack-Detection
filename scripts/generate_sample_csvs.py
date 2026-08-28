import pandas as pd
import os

# ============================================================
# CONFIGURATION
# ============================================================

RAW_FILE = "CIC_IoT_2023_10000.csv"
REFERENCE_FILE = "data/test.csv"

OUTPUT_DIR = "data"

NORMAL_FILE = os.path.join(OUTPUT_DIR, "normal_traffic.csv")
BENIGN_FILE = os.path.join(OUTPUT_DIR, "all_benign_traffic.csv")
MIXED_FILE = os.path.join(OUTPUT_DIR, "mixed_traffic.csv")


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("GENERATING SAMPLE CSV FILES")
print("=" * 60)

print("\nLoading raw CIC-IoT dataset...")
df = pd.read_csv(RAW_FILE)

print(f"Raw dataset shape: {df.shape}")
print(f"Raw dataset columns: {len(df.columns)}")


# ============================================================
# LOAD REFERENCE MODEL FEATURES
# ============================================================

print("\nLoading reference test.csv...")

reference = pd.read_csv(REFERENCE_FILE)

if "label" not in reference.columns:
    raise ValueError("ERROR: label column not found in test.csv.")

FEATURE_COLUMNS = reference.drop(columns=["label"]).columns.tolist()

print(f"Reference feature count: {len(FEATURE_COLUMNS)}")

print("\nModel feature columns:")
print(FEATURE_COLUMNS)


# ============================================================
# CHECK RAW DATA
# ============================================================

missing_columns = [
    col for col in FEATURE_COLUMNS
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Raw dataset is missing required columns: {missing_columns}"
    )


if "label" not in df.columns:
    raise ValueError("ERROR: label column not found in raw dataset.")


# ============================================================
# KEEP ONLY MODEL FEATURES + LABEL
# ============================================================

df = df[FEATURE_COLUMNS + ["label"]].copy()

print("\nAfter selecting model columns:")
print(f"Shape: {df.shape}")

if len(FEATURE_COLUMNS) != 39:
    raise ValueError(
        f"Expected 39 model features, found {len(FEATURE_COLUMNS)}"
    )


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# SHOW ORIGINAL LABEL DISTRIBUTION
# ============================================================

print("\nOriginal label distribution:")
print(df["label"].value_counts().to_dict())


# ============================================================
# SEPARATE BENIGN AND ATTACK TRAFFIC
# ============================================================

benign = df[df["label"] == 0].copy()
attacks = df[df["label"] == 1].copy()

print(f"\nAvailable benign records : {len(benign)}")
print(f"Available attack records : {len(attacks)}")


# ============================================================
# 1. NORMAL TRAFFIC
# ============================================================

NORMAL_COUNT = min(200, len(benign))

normal_sample = benign.sample(
    n=NORMAL_COUNT,
    random_state=123
).reset_index(drop=True)

normal_sample.to_csv(
    NORMAL_FILE,
    index=False
)

print("\n[1] NORMAL TRAFFIC")
print(f"File   : {NORMAL_FILE}")
print(f"Shape  : {normal_sample.shape}")
print(f"Labels : {normal_sample['label'].value_counts().to_dict()}")


# ============================================================
# 2. ALL BENIGN TRAFFIC
# ============================================================

BENIGN_COUNT = min(150, len(benign))

benign_sample = benign.sample(
    n=BENIGN_COUNT,
    random_state=456
).reset_index(drop=True)

benign_sample.to_csv(
    BENIGN_FILE,
    index=False
)

print("\n[2] ALL BENIGN TRAFFIC")
print(f"File   : {BENIGN_FILE}")
print(f"Shape  : {benign_sample.shape}")
print(f"Labels : {benign_sample['label'].value_counts().to_dict()}")


# ============================================================
# 3. MIXED TRAFFIC
# ============================================================

TOTAL_MIXED = min(200, len(df))

TARGET_BENIGN = int(TOTAL_MIXED * 0.70)
TARGET_ATTACK = TOTAL_MIXED - TARGET_BENIGN

BENIGN_COUNT = min(TARGET_BENIGN, len(benign))
ATTACK_COUNT = min(TARGET_ATTACK, len(attacks))

mixed_benign = benign.sample(
    n=BENIGN_COUNT,
    random_state=789
)

mixed_attacks = attacks.sample(
    n=ATTACK_COUNT,
    random_state=101112
)

mixed_sample = pd.concat(
    [
        mixed_benign,
        mixed_attacks
    ],
    ignore_index=True
)

# Shuffle attack and benign rows together
mixed_sample = mixed_sample.sample(
    frac=1,
    random_state=131415
).reset_index(drop=True)

mixed_sample.to_csv(
    MIXED_FILE,
    index=False
)

print("\n[3] MIXED TRAFFIC")
print(f"File   : {MIXED_FILE}")
print(f"Shape  : {mixed_sample.shape}")
print(f"Labels : {mixed_sample['label'].value_counts().to_dict()}")


# ============================================================
# FINAL VALIDATION
# ============================================================

print("\n" + "=" * 60)
print("FINAL VALIDATION")
print("=" * 60)

for file in [
    NORMAL_FILE,
    BENIGN_FILE,
    MIXED_FILE
]:

    check = pd.read_csv(file)

    feature_count = len(
        check.drop(columns=["label"]).columns
    )

    print(f"\n{os.path.basename(file)}")
    print(f"  Shape          : {check.shape}")
    print(f"  Features       : {feature_count}")
    print(f"  Labels         : {check['label'].value_counts().to_dict()}")
    print(f"  Columns correct: {list(check.columns) == FEATURE_COLUMNS + ['label']}")

print("\n" + "=" * 60)
print("SAMPLE GENERATION COMPLETE")
print("=" * 60)