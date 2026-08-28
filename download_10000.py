from datasets import load_dataset
import pandas as pd

print("Loading dataset...")

dataset = load_dataset(
    "lacg030175/CIC-IoT-2023-full",
    split="train",
    streaming=True
)

print("Taking 10,000 rows...")

rows = []

for i, row in enumerate(dataset):
    rows.append(row)

    if i + 1 == 10000:
        break

df = pd.DataFrame(rows)

df.to_csv("CIC_IoT_2023_10000.csv", index=False)

print("Done!")
print("Shape:", df.shape)
print("Saved as: CIC_IoT_2023_10000.csv")