import pandas as pd
from pathlib import Path


# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "cs-training.csv"


# --------------------------------------------------
# Load dataset
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)


# --------------------------------------------------
# Basic dataset information
# --------------------------------------------------

print("=" * 60)
print("CODEALPHA - CREDIT SCORING MODEL")
print("=" * 60)

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
for column in df.columns:
    print(f"- {column}")


# --------------------------------------------------
# First few records
# --------------------------------------------------

print("\n" + "=" * 60)
print("FIRST 5 RECORDS")
print("=" * 60)

print(df.head())


# --------------------------------------------------
# Data types
# --------------------------------------------------

print("\n" + "=" * 60)
print("DATA TYPES")
print("=" * 60)

print(df.dtypes)


# --------------------------------------------------
# Missing values
# --------------------------------------------------

print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)

missing = df.isnull().sum()

missing_table = pd.DataFrame({
    "Missing Values": missing,
    "Missing Percentage": (missing / len(df) * 100).round(2)
})

print(missing_table)


# --------------------------------------------------
# Duplicate records
# --------------------------------------------------

print("\n" + "=" * 60)
print("DUPLICATE RECORDS")
print("=" * 60)

print(df.duplicated().sum())


# --------------------------------------------------
# Target distribution
# --------------------------------------------------

target_column = "SeriousDlqin2yrs"

print("\n" + "=" * 60)
print("TARGET DISTRIBUTION")
print("=" * 60)

print(df[target_column].value_counts())

print("\nTarget percentage:")
print(
    (df[target_column].value_counts(normalize=True) * 100)
    .round(2)
)


# --------------------------------------------------
# Numerical summary
# --------------------------------------------------

print("\n" + "=" * 60)
print("NUMERICAL SUMMARY")
print("=" * 60)

print(df.describe().T)