"""
train.py – BITS Pilani M.Tech AIML | Machine Learning Assignment 2
Dry Bean Classification – Complete Training Pipeline

Stages:
  1. Load & inspect dataset
  2. Preprocess (remove duplicates, encode labels, split, scale)
  3. Train five classifiers inside Scikit-learn Pipelines
  4. Evaluate all models with six metrics + confusion matrices
  5. Save trained pipelines as .joblib files
  6. Export test_data.csv, model_comparison.csv, reports & plots

Author : BITS Pilani M.Tech AIML Student
Dataset: UCI Dry Bean Dataset (13,611 records, 16 features, 7 classes)
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0.  IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import os
import warnings
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend – safe for scripts
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 1.  PATHS
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_PATH       = os.path.join(BASE_DIR, "data", "dataset.csv")
MODEL_DIR       = os.path.join(BASE_DIR, "model")
RESULTS_DIR     = os.path.join(BASE_DIR, "results")
REPORTS_DIR     = os.path.join(RESULTS_DIR, "classification_reports")
TEST_CSV        = os.path.join(BASE_DIR, "test_data.csv")
COMPARISON_CSV  = os.path.join(RESULTS_DIR, "model_comparison.csv")

for d in [MODEL_DIR, RESULTS_DIR, REPORTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 2.  LOAD & INSPECT
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("  STAGE 1 – LOAD & INSPECT DATASET")
print("=" * 65)

df = pd.read_csv(DATA_PATH)
print(f"\nLoaded : {DATA_PATH}")
print(f"Shape  : {df.shape[0]} rows × {df.shape[1]} columns")
print(f"\nClass distribution:\n{df['Class'].value_counts().to_string()}")
print(f"\nMissing values : {df.isnull().sum().sum()}")
print(f"Duplicate rows : {df.duplicated().sum()}")

# ─────────────────────────────────────────────────────────────────────────────
# 3.  PREPROCESS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  STAGE 2 – DATA PREPROCESSING")
print("=" * 65)

# 3a. Remove duplicate rows
before = len(df)
df = df.drop_duplicates().reset_index(drop=True)
print(f"\nRemoved {before - len(df)} duplicate rows → {len(df)} remaining")

# 3b. Separate features (X) and target (y)
FEATURE_COLS = [
    "Area", "Perimeter", "MajorAxisLength", "MinorAxisLength",
    "AspectRation", "Eccentricity", "ConvexArea", "EquivDiameter",
    "Extent", "Solidity", "roundness", "Compactness",
    "ShapeFactor1", "ShapeFactor2", "ShapeFactor3", "ShapeFactor4",
]
TARGET_COL = "Class"

X = df[FEATURE_COLS].values
y_raw = df[TARGET_COL].values

# 3c. Encode target labels
le = LabelEncoder()
y = le.fit_transform(y_raw)
print(f"\nLabel mapping  : {dict(zip(le.classes_, le.transform(le.classes_)))}")

# 3d. Stratified train / test split  (80 / 20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
print(f"\nTrain set size : {X_train.shape[0]} samples")
print(f"Test  set size : {X_test.shape[0]} samples")
print(f"All 5 models will use the SAME split.")

# 3e. Save test_data.csv (raw features + true Class label)
test_df = pd.DataFrame(X_test, columns=FEATURE_COLS)
test_df[TARGET_COL] = le.inverse_transform(y_test)
test_df.to_csv(TEST_CSV, index=False)
print(f"\ntest_data.csv saved → {TEST_CSV}")

# Save label encoder separately (needed by app.py)
joblib.dump(le, os.path.join(MODEL_DIR, "label_encoder.joblib"))
print(f"label_encoder.joblib saved → {MODEL_DIR}")

# ─────────────────────────────────────────────────────────────────────────────
# 4.  MODEL DEFINITIONS
#     Each model is wrapped in a Pipeline with StandardScaler.
#     GaussianNB already works fine with scaled data.
#     Decision Tree is scale-invariant but we keep it in a Pipeline
#     for consistency and to prevent any accidental data leakage.
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  STAGE 3 – DEFINING CLASSIFIERS")
print("=" * 65)

N_CLASSES = len(le.classes_)

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            C=1.0,
            random_state=42
        )),
    ]),

    "Decision Tree": Pipeline([
        ("scaler", StandardScaler()),          # scale-invariant but uniform pipeline
        ("clf", DecisionTreeClassifier(
            max_depth=None,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )),
    ]),

    "KNN": Pipeline([
        ("scaler", StandardScaler()),          # KNN is distance-based → scaling critical
        ("clf", KNeighborsClassifier(
            n_neighbors=7,
            weights="distance",
            metric="minkowski",
            p=2
        )),
    ]),

    "Naive Bayes": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GaussianNB()),
    ]),

    "SVM": Pipeline([
        ("scaler", StandardScaler()),          # SVM is distance-based → scaling critical
        ("clf", SVC(
            kernel="rbf",
            C=10,
            gamma="scale",
            probability=True,                  # required for AUC with predict_proba
            random_state=42
        )),
    ]),
}

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree":       "decision_tree.joblib",
    "KNN":                 "knn.joblib",
    "Naive Bayes":         "naive_bayes.joblib",
    "SVM":                 "svm.joblib",
}

print(f"\nFive classifiers defined:")
for name in models:
    print(f"   • {name}")

# ─────────────────────────────────────────────────────────────────────────────
# 5.  TRAIN + EVALUATE EACH MODEL
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  STAGE 4 – TRAINING & EVALUATION")
print("=" * 65)

CLASS_NAMES = list(le.classes_)   # original string labels
results = {}                       # store metrics for all models

for name, pipeline in models.items():
    print(f"\n{'─'*60}")
    print(f"  Model : {name}")
    print(f"{'─'*60}")

    # ── Train ────────────────────────────────────────────────────────────────
    print("  Training …", end="", flush=True)
    pipeline.fit(X_train, y_train)
    print(" done.")

    # ── Predict ──────────────────────────────────────────────────────────────
    y_pred   = pipeline.predict(X_test)
    y_proba  = pipeline.predict_proba(X_test)   # shape (n_samples, n_classes)

    # ── Metrics ──────────────────────────────────────────────────────────────
    # Accuracy
    acc = accuracy_score(y_test, y_pred)

    # AUC – One-vs-Rest (OvR), macro averaging (standard for multiclass)
    auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")

    # Precision / Recall / F1 – weighted averaging accounts for class imbalance
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    # Matthews Correlation Coefficient – works natively for multiclass in sklearn
    mcc  = matthews_corrcoef(y_test, y_pred)

    results[name] = {
        "Accuracy":  round(acc,  4),
        "AUC":       round(auc,  4),
        "Precision": round(prec, 4),
        "Recall":    round(rec,  4),
        "F1 Score":  round(f1,   4),
        "MCC":       round(mcc,  4),
    }

    print(f"  Accuracy  : {acc:.4f}")
    print(f"  AUC (OvR) : {auc:.4f}   [macro, One-vs-Rest]")
    print(f"  Precision : {prec:.4f}  [weighted avg]")
    print(f"  Recall    : {rec:.4f}  [weighted avg]")
    print(f"  F1 Score  : {f1:.4f}  [weighted avg]")
    print(f"  MCC       : {mcc:.4f}")

    # ── Classification Report ─────────────────────────────────────────────────
    report = classification_report(
        y_test, y_pred, target_names=CLASS_NAMES, digits=4
    )
    report_path = os.path.join(REPORTS_DIR, f"{name.replace(' ', '_')}_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Classification Report – {name}\n")
        f.write("=" * 60 + "\n")
        f.write(f"Averaging: weighted (Precision, Recall, F1)\n")
        f.write(f"AUC      : macro OvR\n")
        f.write("=" * 60 + "\n\n")
        f.write(report)
    print(f"\n  Classification report → {report_path}")


    # ── Save Model ───────────────────────────────────────────────────────────
    model_path = os.path.join(MODEL_DIR, MODEL_FILES[name])
    joblib.dump(pipeline, model_path)
    print(f"  Model saved        → {model_path}")

# ─────────────────────────────────────────────────────────────────────────────
# 6.  COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  STAGE 5 – MODEL COMPARISON TABLE")
print("=" * 65)

comparison_df = pd.DataFrame(results).T
comparison_df.index.name = "Model"
comparison_df = comparison_df.sort_values("Accuracy", ascending=False)

print("\n" + comparison_df.to_string())

comparison_df.to_csv(COMPARISON_CSV)
print(f"\nComparison table saved → {COMPARISON_CSV}")

# ── Identify best model ───────────────────────────────────────────────────────
best_model = comparison_df["Accuracy"].idxmax()
print(f"\n🏆  Best Model : {best_model}")
print(f"    Accuracy  : {comparison_df.loc[best_model, 'Accuracy']:.4f}")
print(f"    AUC       : {comparison_df.loc[best_model, 'AUC']:.4f}")
print(f"    F1 Score  : {comparison_df.loc[best_model, 'F1 Score']:.4f}")
print(f"    MCC       : {comparison_df.loc[best_model, 'MCC']:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 7.  COMPARISON BAR CHART
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# 8.  SAVE RESULTS AS JSON (used by app.py for display)
# ─────────────────────────────────────────────────────────────────────────────
json_path = os.path.join(RESULTS_DIR, "metrics.json")
with open(json_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"metrics.json saved     → {json_path}")

# ─────────────────────────────────────────────────────────────────────────────
# 9.  FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  TRAINING COMPLETE – FILES GENERATED")
print("=" * 65)
print(f"""
  model/logistic_regression.joblib
  model/decision_tree.joblib
  model/knn.joblib
  model/naive_bayes.joblib
  model/svm.joblib
  model/label_encoder.joblib
  test_data.csv
  results/model_comparison.csv
  results/metrics.json
  results/classification_reports/*.txt

  Note: Visualizations (confusion matrices, comparison charts) are generated
  dynamically by the Streamlit app (app.py) when evaluating uploaded datasets.
""")
print(f"  🏆  Best Model : {best_model}  (Accuracy = {comparison_df.loc[best_model, 'Accuracy']:.4f})")
print("=" * 65)

