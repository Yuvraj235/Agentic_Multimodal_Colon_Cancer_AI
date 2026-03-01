"""
DAY 12 – DATASET AUDIT
Counts images per class and subclass
Detects imbalance and rare categories
"""

import os
from collections import defaultdict
import pandas as pd

DATASET_ROOT = "data/processed/hyper_kvasir_clean"

print("🔍 Starting Dataset Audit...\n")

class_counts = defaultdict(int)
subclass_counts = defaultdict(int)
total_images = 0

for root, dirs, files in os.walk(DATASET_ROOT):
    for file in files:
        if file.endswith(".jpg") or file.endswith(".png"):
            total_images += 1

            path_parts = root.split(os.sep)

            # Expected structure:
            # hyper_kvasir_clean / tract / class / subclass

            if len(path_parts) >= 3:
                main_class = path_parts[-2]
                subclass = path_parts[-1]

                class_counts[main_class] += 1
                subclass_counts[f"{main_class}/{subclass}"] += 1

print("Total Images:", total_images)
print("\n==============================")
print("Top-Level Class Distribution")
print("==============================\n")

for cls, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"{cls:30s} : {count}")

print("\n==============================")
print("Subclass Distribution")
print("==============================\n")

for sub, count in sorted(subclass_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"{sub:60s} : {count}")

# Convert to DataFrame for saving
df_main = pd.DataFrame(class_counts.items(), columns=["Class", "Count"])
df_sub = pd.DataFrame(subclass_counts.items(), columns=["Subclass", "Count"])

df_main.to_csv("outputs/day12_class_distribution.csv", index=False)
df_sub.to_csv("outputs/day12_subclass_distribution.csv", index=False)

print("\n✅ Dataset Audit Complete")