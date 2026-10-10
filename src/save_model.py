import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier


# ============================================================
# CODEALPHA - CREDIT SCORING MODEL
# SAVE FINAL MODEL
# ============================================================

DATA_PATH = "data/raw/cs-training.csv"
MODEL_PATH = "models/credit_scoring_model.pkl"

RANDOM_STATE = 42
DECISION_THRESHOLD = 0.75


print("=" * 60)
print("CODEALPHA - SAVING FINAL CREDIT SCORING MODEL")
print("=" * 60)


# ------------------------------------------------------------
# 1. Load dataset
# ------------------------------------------------------------
df = pd.read_csv(DATA_PATH)

# Remove unnecessary ID column
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])


# ------------------------------------------------------------
# 2. Separate features and target
# ------------------------------------------------------------
X = df.drop(columns=["SeriousDlqin2yrs"])
y = df["SeriousDlqin2yrs"]


# ------------------------------------------------------------
# 3. Train/test split
# ------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)


# ------------------------------------------------------------
# 4. Final Random Forest model
# ------------------------------------------------------------
model = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    ),
    (
        "model",
        RandomForestClassifier(
            n_estimators=150,
            max_depth=8,
            min_samples_split=5,
            min_samples_leaf=1,
            max_features="log2",
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    )
])


# ------------------------------------------------------------
# 5. Train model
# ------------------------------------------------------------
print("\nTraining final model...")

model.fit(X_train, y_train)

print("Model training completed.")


# ------------------------------------------------------------
# 6. Create models directory
# ------------------------------------------------------------
os.makedirs("models", exist_ok=True)


# ------------------------------------------------------------
# 7. Save model
# ------------------------------------------------------------
joblib.dump(model, MODEL_PATH)


# ------------------------------------------------------------
# 8. Save decision threshold
# ------------------------------------------------------------
threshold_path = "models/decision_threshold.txt"

with open(threshold_path, "w") as file:
    file.write(str(DECISION_THRESHOLD))


# ------------------------------------------------------------
# 9. Confirmation
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("MODEL SAVED SUCCESSFULLY")
print("=" * 60)

print(f"Model file     : {MODEL_PATH}")
print(f"Threshold file : {threshold_path}")
print(f"Decision threshold: {DECISION_THRESHOLD}")

print("\nThe model is ready for the prediction system.")