import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    accuracy_score,
    confusion_matrix
)


# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "cs-training.csv"
)


# --------------------------------------------------
# Load dataset
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

df = df.drop(
    columns=["Unnamed: 0"]
)


# --------------------------------------------------
# Features and target
# --------------------------------------------------

X = df.drop(
    columns=["SeriousDlqin2yrs"]
)

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
# Handle missing values
# --------------------------------------------------

imputer = SimpleImputer(
    strategy="median"
)

X_train = imputer.fit_transform(
    X_train
)

X_test = imputer.transform(
    X_test
)


# --------------------------------------------------
# Tuned Random Forest
# --------------------------------------------------

model = RandomForestClassifier(
    n_estimators=150,
    min_samples_split=5,
    min_samples_leaf=1,
    max_features="log2",
    max_depth=8,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)


print("=" * 60)
print("CODEALPHA - CREDIT SCORING MODEL")
print("THRESHOLD OPTIMIZATION")
print("=" * 60)

print("\nTraining tuned Random Forest...")


model.fit(
    X_train,
    y_train
)


# --------------------------------------------------
# Probability predictions
# --------------------------------------------------

y_probability = model.predict_proba(
    X_test
)[:, 1]


print("\nROC-AUC:")
print(
    f"{roc_auc_score(y_test, y_probability):.4f}"
)


# --------------------------------------------------
# Test different thresholds
# --------------------------------------------------

thresholds = np.arange(
    0.20,
    0.81,
    0.05
)

results = []


print("\n" + "=" * 60)
print("THRESHOLD COMPARISON")
print("=" * 60)

for threshold in thresholds:

    y_pred = (
        y_probability >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    results.append({
        "Threshold": threshold,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1
    })


results_df = pd.DataFrame(
    results
)


print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# --------------------------------------------------
# Find best threshold using F1
# --------------------------------------------------

best_row = results_df.loc[
    results_df["F1-Score"].idxmax()
]


best_threshold = best_row[
    "Threshold"
]


print("\n" + "=" * 60)
print("BEST THRESHOLD")
print("=" * 60)

print(
    f"Threshold: {best_threshold:.2f}"
)

print(
    f"Accuracy : {best_row['Accuracy']:.4f}"
)

print(
    f"Precision: {best_row['Precision']:.4f}"
)

print(
    f"Recall   : {best_row['Recall']:.4f}"
)

print(
    f"F1-Score : {best_row['F1-Score']:.4f}"
)


# --------------------------------------------------
# Final confusion matrix
# --------------------------------------------------

y_best_pred = (
    y_probability >= best_threshold
).astype(int)


cm = confusion_matrix(
    y_test,
    y_best_pred
)


print("\nConfusion Matrix:")
print(cm)


print("\nThreshold optimization completed successfully.")