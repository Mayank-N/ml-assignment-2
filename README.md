# Dry Bean Classification — ML Assignment 2

## M.Tech AIML | Machine Learning | Assignment 2

---

## Problem Statement

Uniform seed classification and quality inspection in agriculture are vital for grain processing, market valuation, and seed certification. In dry bean production, distinguishing seed varieties with high visual resemblance manually is labor-intensive, error-prone, and subjective.

This project builds a **multiclass classification system** to predict the variety of a dry bean among **seven registered commercial varieties** (SEKER, BARBUNYA, BOMBAY, CALI, HOROZ, SIRA, and DERMASON) using 16 morphological features extracted from digitised images of bean grains.

The objective is to train, evaluate, and compare **five classical machine learning classifiers** using the Dry Bean dataset, deploy an interactive Streamlit application, and document all findings for academic submission.

---

## Dataset Description

| Property                 | Value                                                                                                |
| ------------------------ | ---------------------------------------------------------------------------------------------------- |
| **Name**           | Dry Bean Dataset                                                                                     |
| **Source**         | UCI Machine Learning Repository                                                                      |
| **Records**        | 13,611                                                                                               |
| **Features**       | 16 real-valued numeric features                                                                      |
| **Target Column**  | `Class`                                                                                            |
| **Classes**        | DERMASON (3546), SIRA (2636), SEKER (2027), HOROZ (1928), CALI (1630), BARBUNYA (1322), BOMBAY (522) |
| **Missing Values** | None                                                                                                 |
| **Duplicates**     | 68 (removed)                                                                                         |

**Features:** Sixteen morphological measurements — comprising 12 dimensional attributes (Area, Perimeter, MajorAxisLength, MinorAxisLength, AspectRation, Eccentricity, ConvexArea, EquivDiameter, Extent, Solidity, roundness, Compactness) and 4 shape factors (ShapeFactor1 through ShapeFactor4) — are computed from digital image segmentation across the 7 bean varieties.

---

## GitHub Repository

> 🔗 **GitHub Repository URL:** `https://github.com/Mayank-N/ml-assignment-2`

---

## Project Structure

```
ml-assignment-2/
├── app.py                  # Streamlit web application
├── train.py                # Training script (runs all 5 models)
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── test_data.csv           # Held-out test set (2,709 samples)
├── data/
│   ├── dataset.csv         # Dry Bean dataset (13,543 records)
│   └── inspect_dataset.py  # Dataset inspection script
├── model/
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── knn.joblib
│   ├── naive_bayes.joblib
│   ├── svm.joblib
│   └── label_encoder.joblib
└── results/
    ├── metrics.json
    ├── model_comparison.csv
    ├── model_comparison_chart.png
    ├── classification_reports/
    └── confusion_matrices/
```

---

## Five Machine Learning Models

| # | Model                            | Library Class              | Key Hyperparameters                                                      |
| - | -------------------------------- | -------------------------- | ------------------------------------------------------------------------ |
| 1 | **Logistic Regression**    | `LogisticRegression`     | `max_iter=1000, solver='lbfgs', C=1.0, random_state=42`                |
| 2 | **Decision Tree**          | `DecisionTreeClassifier` | `min_samples_split=5, min_samples_leaf=2, random_state=42`             |
| 3 | **K-Nearest Neighbors**    | `KNeighborsClassifier`   | `n_neighbors=7, weights='distance', metric='minkowski', p=2`           |
| 4 | **Gaussian Naive Bayes**   | `GaussianNB`             | default                                                                  |
| 5 | **Support Vector Machine** | `SVC`                    | `kernel='rbf', C=10, gamma='scale', probability=True, random_state=42` |

Each model is wrapped in a **scikit-learn Pipeline** with `StandardScaler` to prevent data leakage. The scaler is fit only on training data and applied to test data during evaluation.

---

## Data Preprocessing Steps

1. **Load** CSV with named columns (16 features + `Class`)
2. **Remove** duplicates → 68 found and removed (13,543 clean records)
3. **Encode** target: 7 classes mapped to integers (0 to 6) via `LabelEncoder`
4. **Split** features (X) and target (y)
5. **Stratified train/test split**: 80% train (10,834), 20% test (2,709), `random_state=42`
6. **Pipeline** with `StandardScaler` applied inside `Pipeline.fit()` on training data only → no leakage

---

## Evaluation Metrics — Model Comparison Table

> **Test Set:** 2,709 samples | Stratified across 7 classes | `random_state=42`

| Model                      | Accuracy         | AUC              | Precision        | Recall           | F1 Score         | MCC              |
| -------------------------- | ---------------- | ---------------- | ---------------- | ---------------- | ---------------- | ---------------- |
| Logistic Regression        | 0.9192           | **0.9947** | 0.9197           | 0.9192           | 0.9193           | 0.9023           |
| Decision Tree              | 0.8966           | 0.9573           | 0.8971           | 0.8966           | 0.8965           | 0.8752           |
| K-Nearest Neighbors        | 0.9136           | 0.9854           | 0.9144           | 0.9136           | 0.9138           | 0.8956           |
| Naive Bayes                | 0.8970           | 0.9911           | 0.8997           | 0.8970           | 0.8972           | 0.8762           |
| **SVM (RBF kernel)** | **0.9236** | 0.9944           | **0.9238** | **0.9236** | **0.9236** | **0.9076** |

> ✅ All metrics are computed on the held-out test set only. No data leakage. Models are never retrained on test data.

---

## Observations Per Model

### 1. Logistic Regression

- **Accuracy: 91.92%** — Second-highest accuracy, strong linear separator for this dataset.
- **AUC: 0.9947** — Highest AUC, indicating excellent discriminative ability across all thresholds.
- **Precision: 0.9197** — Very balanced precision across classes.
- **MCC: 0.9023** — Strong balanced performance even with class imbalance.
- **Strength:** Interpretable coefficients, fast training, reliable probability estimates.
- **Weakness:** Assumes linearity — slightly lower than SVM on curved boundaries.

### 2. Decision Tree

- **Accuracy: 89.66%** — Lowest accuracy among the five models.
- **AUC: 0.9573** — Lowest AUC; probability estimates from DT can be unreliable.
- **Precision = 0.8971, Recall = 0.8966** — Balanced but lower performance.
- **MCC: 0.8752** — Acceptable but weakest correlation coefficient.
- **Strength:** Fully interpretable (visualisable tree), no need for feature scaling.
- **Weakness:** Prone to overfitting; axis-aligned splits may miss curved boundaries.

### 3. K-Nearest Neighbors (KNN)

- **Accuracy: 91.36%** — Third-highest accuracy, solid non-parametric performance.
- **AUC: 0.9854** — Good probabilistic discrimination.
- **Precision: 0.9144** — Very few false positives.
- **MCC: 0.8956** — Good balanced score.
- **Strength:** Simple, no training phase, effective with 7 neighbours on this dataset.
- **Weakness:** Slow prediction for large datasets; sensitive to irrelevant features.

### 4. Naive Bayes (Gaussian)

- **Accuracy: 89.70%** — Second-lowest accuracy.
- **AUC: 0.9911** — Surprisingly high AUC despite lower accuracy; probabilistic model works well.
- **Recall: 0.8970** — Solid recall across dominant classes.
- **MCC: 0.8762** — Good baseline correlation coefficient.
- **Strength:** Extremely fast, works with small training sets, handles continuous features well.
- **Weakness:** Independence assumption often violated in real data.

### 5. Support Vector Machine (RBF kernel)

- **Accuracy: 92.36%** — **Highest accuracy** on the test set.
- **AUC: 0.9944** — Near-perfect discriminative performance.
- **Precision: 0.9238** — Highest weighted precision across all seven classes.
- **F1 Score: 0.9236** — Best F1 among all five models.
- **MCC: 0.9076** — Highest balanced correlation.
- **Strength:** Excellent with high-dimensional data; RBF kernel captures non-linear boundaries; maximum-margin classifier.
- **Weakness:** Slower to train on large datasets; less interpretable than LR or DT; hyperparameter tuning needed.

---

## 🏆 Best Performing Model

> **Support Vector Machine (SVM with RBF kernel)**

SVM achieves the highest test accuracy (92.36%), the best F1 Score (0.9236), and the highest MCC (0.9076). Crucially, the RBF kernel handles non-linear boundaries between overlapping bean classes effectively. While Logistic Regression achieves a slightly higher AUC (0.9947 vs 0.9944), SVM's superior accuracy, F1, and MCC make it the best overall model for deployment.

---

## Installation Instructions

### Prerequisites

- Python 3.9 or higher
- pip

### 1. Clone the repository

```bash
git clone https://github.com/Mayank-N/ml-assignment-2.git
cd ml-assignment-2
```

### 2. Create a virtual environment (recommended)

```bash
python3 -m venv venv
source venv/bin/activate       # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Execution Instructions

### Step 1 — Train all models

```bash
python train.py
```

This will:

- Load `data/dataset.csv`
- Preprocess the data (deduplicate, encode labels, split 80/20)
- Train all five pipelines
- Save `.joblib` files in `model/`
- Save `test_data.csv`
- Print evaluation metrics and the comparison table

### Step 2 — Run the Streamlit application

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`. Upload `test_data.csv` and select a model to evaluate.

---

## Streamlit Deployment

> 🌐 **Live App URL:** https://ml-assignment-2-2026aj05561.streamlit.app/

The application is deployed on Streamlit Community Cloud and loads pre-trained models from the `model/` directory — it never retrains models from the uploaded test data.
