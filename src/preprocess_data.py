import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer


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
# Remove unnecessary ID column
# --------------------------------------------------

df = df.drop(columns=["Unnamed: 0"])


# --------------------------------------------------
# Separate features and target
# --------------------------------------------------

X = df.drop(columns=["SeriousDlqin2yrs"])
y = df["SeriousDlqin2yrs"]


# --------------------------------------------------
# Train-test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# --------------------------------------------------
# Missing value imputation
# --------------------------------------------------

imputer = SimpleImputer(strategy="median")

X_train_imputed = pd.DataFrame(
    imputer.fit_transform(X_train),
    columns=X_train.columns,
    index=X_train.index
)

X_test_imputed = pd.DataFrame(
    imputer.transform(X_test),
    columns=X_test.columns,
    index=X_test.index
)


# --------------------------------------------------
# Display preprocessing results
# --------------------------------------------------

print("=" * 60)
print("CODEALPHA - CREDIT SCORING MODEL")
print("PREPROCESSING")
print("=" * 60)

print("\nOriginal dataset:")
print(f"Rows: {len(df)}")
print(f"Features: {X.shape[1]}")

print("\nTrain-test split:")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples : {len(X_test)}")

print("\nTarget distribution - training data:")
print(y_train.value_counts())

print("\nTarget percentage - training data:")
print(
    (y_train.value_counts(normalize=True) * 100).round(2)
)

print("\nTarget distribution - testing data:")
print(y_test.value_counts())

print("\nMissing values before imputation:")
print(X_train.isnull().sum()[X_train.isnull().sum() > 0])

print("\nMissing values after imputation:")
print(X_train_imputed.isnull().sum().sum())

print("\nPreprocessing completed successfully.")