import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
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

OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


# --------------------------------------------------
# Load dataset
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)


# --------------------------------------------------
# Remove ID column
# --------------------------------------------------

df = df.drop(
    columns=["Unnamed: 0"]
)


# --------------------------------------------------
# Separate features and target
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

X_train_imputed = imputer.fit_transform(
    X_train
)

X_test_imputed = imputer.transform(
    X_test
)


# --------------------------------------------------
# Final Random Forest model
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


# --------------------------------------------------
# Train final model
# --------------------------------------------------

print("=" * 60)
print("CODEALPHA - CREDIT SCORING MODEL")
print("FINAL MODEL EVALUATION")
print("=" * 60)

print("\nTraining final Random Forest...")

model.fit(
    X_train_imputed,
    y_train
)


# --------------------------------------------------
# Probability predictions
# --------------------------------------------------

y_probability = model.predict_proba(
    X_test_imputed
)[:, 1]


# --------------------------------------------------
# Final threshold
# --------------------------------------------------

threshold = 0.75

y_pred = (
    y_probability >= threshold
).astype(int)


# --------------------------------------------------
# Calculate metrics
# --------------------------------------------------

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

roc_auc = roc_auc_score(
    y_test,
    y_probability
)


# --------------------------------------------------
# Display results
# --------------------------------------------------

print("\n" + "=" * 60)
print("FINAL MODEL RESULTS")
print("=" * 60)

print(f"Decision Threshold: {threshold:.2f}")

print(f"\nAccuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-Score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")


# --------------------------------------------------
# Classification report
# --------------------------------------------------

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "No Serious Delinquency",
            "Serious Delinquency"
        ],
        zero_division=0
    )
)


# --------------------------------------------------
# Confusion matrix
# --------------------------------------------------

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print(cm)


# --------------------------------------------------
# Confusion matrix plot
# --------------------------------------------------

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "No Delinquency",
        "Serious Delinquency"
    ]
)

display.plot()

plt.title(
    "Final Model Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "final_confusion_matrix.png",
    dpi=300
)

plt.show()


# --------------------------------------------------
# Feature importance
# --------------------------------------------------

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)


print("\n" + "=" * 60)
print("FEATURE IMPORTANCE")
print("=" * 60)

print(
    feature_importance.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# --------------------------------------------------
# Feature importance plot
# --------------------------------------------------

plt.figure(figsize=(10, 6))

plt.barh(
    feature_importance["Feature"],
    feature_importance["Importance"]
)

plt.gca().invert_yaxis()

plt.title(
    "Random Forest Feature Importance"
)

plt.xlabel(
    "Importance"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "feature_importance.png",
    dpi=300
)

plt.show()


# --------------------------------------------------
# Save feature importance
# --------------------------------------------------

feature_importance.to_csv(
    OUTPUT_DIR / "feature_importance.csv",
    index=False
)


print("\nFeature importance saved to:")
print("outputs/feature_importance.csv")

print("\nFinal model evaluation completed successfully.")