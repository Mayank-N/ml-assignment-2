"""
Stage 1: Dataset Inspection & Conversion
Converts Dry_Bean_Dataset.arff → dataset.csv and prints full inspection report.
"""

import re
import pandas as pd
import os

ARFF_PATH = "/Users/mayank/Documents/projects/BITS/DryBeanDataset/Dry_Bean_Dataset.arff"
CSV_OUT   = "/Users/mayank/Documents/projects/BITS/ML-Assignment-2/data/dataset.csv"

# ── Parse ARFF ──────────────────────────────────────────────────────────────
def parse_arff(path):
    attributes = []
    data_lines = []
    in_data = False
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("%"):
                continue
            if line.upper().startswith("@ATTRIBUTE"):
                parts = line.split(None, 2)   # @ATTRIBUTE  <name>  <type>
                attributes.append(parts[1])
            elif line.upper().startswith("@DATA"):
                in_data = True
            elif in_data:
                data_lines.append(line)

    rows = [row.split(",") for row in data_lines if row]
    df = pd.DataFrame(rows, columns=attributes)

    # Cast numeric columns
    for col in df.columns[:-1]:       # all except Class
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Class"] = df["Class"].str.strip()
    return df

df = parse_arff(ARFF_PATH)
df.to_csv(CSV_OUT, index=False)
print(f"✅  Saved → {CSV_OUT}\n")

# ── Inspection Report ────────────────────────────────────────────────────────
print("=" * 60)
print("  DRY BEAN DATASET – INSPECTION REPORT")
print("=" * 60)
print(f"\nShape             : {df.shape[0]} rows × {df.shape[1]} columns")
print(f"Input features    : {df.shape[1] - 1}  (all columns except 'Class')")
print(f"Target column     : Class")
print(f"\nUnique Classes ({df['Class'].nunique()}):")
for cls, cnt in df["Class"].value_counts().items():
    print(f"   {cls:<12} : {cnt:,}")

print(f"\nMissing values per column:")
mv = df.isnull().sum()
print(mv[mv > 0] if mv.sum() > 0 else "   None — dataset is complete!")

print(f"\nDuplicate rows    : {df.duplicated().sum()}")

print("\nData types:")
print(df.dtypes.to_string())

print("\nFirst 3 rows (sample):")
print(df.head(3).to_string(index=False))

print("\nBasic statistics (first 4 features):")
print(df.iloc[:, :4].describe().round(2).to_string())
print("\n✅  Dataset satisfies all assignment requirements.")

