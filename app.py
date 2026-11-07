"""
app.py – BITS Pilani M.Tech AIML | ML Assignment 2
Dry Bean Classification – Single-page Streamlit App
"""

import os, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import joblib
import streamlit as st

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report,
)

warnings.filterwarnings("ignore")

# ── Setup ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dry Bean Classifier | BITS AIML",
    page_icon="🫘",
    layout="wide",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FEATURES = [
    "Area", "Perimeter", "MajorAxisLength", "MinorAxisLength",
    "AspectRation", "Eccentricity", "ConvexArea", "EquivDiameter",
    "Extent", "Solidity", "roundness", "Compactness",
    "ShapeFactor1", "ShapeFactor2", "ShapeFactor3", "ShapeFactor4",
]

MODELS = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree":       "decision_tree.joblib",
    "KNN":                 "knn.joblib",
    "Naive Bayes":         "naive_bayes.joblib",
    "SVM":                 "svm.joblib",
}

BEAN_DESCRIPTIONS = {
    "SEKER": "Small, oval seed; smooth surface; uniform light/cream appearance.",
    "BARBUNYA": "Medium seed; speckled red-and-beige pattern; kidney-shaped.",
    "BOMBAY": "Largest bean grain; nearly spherical; thick dark-coloured skin.",
    "CALI": "Large elongated kidney bean; light brown to tan colouration.",
    "HOROZ": "Cylindrical elongated body; distinctive white colour with eye.",
    "SIRA": "Medium-sized seed; yellowish tone; rounded oval structure.",
    "DERMASON": "Smallest bean variety; flat oval form; creamy white seed coat.",
}

@st.cache_resource
def get_pipeline(name):
    return joblib.load(os.path.join(BASE_DIR, "model", MODELS[name]))

@st.cache_resource
def get_le():
    return joblib.load(os.path.join(BASE_DIR, "model", "label_encoder.joblib"))

@st.cache_data
def get_stored_metrics():
    p = os.path.join(BASE_DIR, "results", "metrics.json")
    return json.load(open(p)) if os.path.exists(p) else {}

# ── Read default test_data.csv for download ───────────────────────────────────
test_csv_path = os.path.join(BASE_DIR, "test_data.csv")
test_csv_bytes = open(test_csv_path, "rb").read() if os.path.exists(test_csv_path) else b""

# ── Reusable Component: Model Comparison Table & Charts ────────────────────────
def render_model_comparison():
    st.divider()
    st.markdown("### Model Comparison Table")
    st.caption("All models trained with identical 80% train split and evaluated on the same held-out 20% test dataset")

    stored = get_stored_metrics()
    if stored:
        comp = pd.DataFrame(stored).T.sort_values("Accuracy", ascending=False)
        comp.index.name = "Classifier"
        best_clf = comp["Accuracy"].idxmax()

        def highlight_best(row):
            if row.name == best_clf:
                return ["background-color: #dbeafe; font-weight: bold; color: #1e40af"] * len(row)
            return [""] * len(row)

        styled_comp = comp.style.format("{:.4f}").apply(highlight_best, axis=1).set_properties(**{"text-align": "center"})

        st.dataframe(
            styled_comp,
            use_container_width=True,
            height=215,
        )

        st.success(
            f"🏆 **Best Performing Model**: **{best_clf}** — "
            f"Accuracy: **{comp.loc[best_clf, 'Accuracy']:.4f}** · "
            f"AUC: **{comp.loc[best_clf, 'AUC']:.4f}** · "
            f"F1 Score: **{comp.loc[best_clf, 'F1 Score']:.4f}** · "
            f"MCC: **{comp.loc[best_clf, 'MCC']:.4f}**"
        )

        MODEL_COLORS = {
            "SVM": "#4F46E5",                  # Indigo
            "Logistic Regression": "#0284C7",  # Sky Blue
            "KNN": "#0D9488",                  # Teal
            "Naive Bayes": "#D97706",          # Amber
            "Decision Tree": "#E11D48",        # Rose
        }
        bar_colors = [MODEL_COLORS.get(m, "#4F46E5") for m in comp.index]
        metrics_plot = ["Accuracy", "AUC", "F1 Score", "MCC"]
        fig, axes = plt.subplots(1, 4, figsize=(14, 3.8), dpi=160)
        fig.patch.set_facecolor("#FFFFFF")

        for ax, metric in zip(axes, metrics_plot):
            ax.set_facecolor("#FFFFFF")
            vals = comp[metric].values
            bars = ax.bar(comp.index, vals, color=bar_colors, width=0.52, zorder=3)
            ax.set_title(metric, fontsize=11.5, fontweight="bold", color="#1E293B", pad=10)
            ax.set_ylim(max(0, min(vals) - 0.04), min(1.0, max(vals) + 0.035))
            ax.set_xticks(range(len(comp.index)))
            ax.set_xticklabels(comp.index, rotation=32, ha="right", fontsize=8.5, color="#475569", fontweight="500")
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
            ax.tick_params(axis="y", colors="#64748B", labelsize=8.5)
            ax.tick_params(axis="both", which="both", length=0)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.spines["bottom"].set_color("#CBD5E1")
            ax.spines["bottom"].set_linewidth(1.0)
            ax.grid(axis="y", linestyle="--", linewidth=0.7, color="#E2E8F0", alpha=0.9, zorder=0)
            for bar, val in zip(bars, vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.002,
                    f"{val:.3f}", ha="center", va="bottom",
                    fontsize=8.5, fontweight="bold", color="#1E293B",
                )

        plt.tight_layout(pad=2)
        st.pyplot(fig, use_container_width=True)
        plt.close()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🫘 Dry Bean Classifier")
    st.caption("BITS Pilani · M.Tech AIML · ML Assignment 2")
    st.divider()

    model_name = st.selectbox(
        "**1. Select Classifier**",
        list(MODELS.keys()),
        help="Choose one of the 5 trained classification pipelines"
    )
    st.divider()

    st.markdown("**2. Upload Dataset**")
    uploaded = st.file_uploader(
        "Upload test CSV for evaluation",
        type=["csv"],
        help="Upload a CSV with 16 morphological features + Class column"
    )

    st.download_button(
        label="📥 Download Sample test_data.csv",
        data=test_csv_bytes,
        file_name="test_data.csv",
        mime="text/csv",
        help="Download the held-out 20% test dataset (2,709 samples) to upload and evaluate",
        use_container_width=True,
    )

    st.divider()
    st.markdown("**3. Display Settings**")
    preview_limit = st.slider(
        "Preview Rows Count",
        min_value=5,
        max_value=100,
        value=15,
        step=5,
        help="Adjust the number of rows displayed in the data preview and sample prediction tables"
    )

    st.divider()
    st.markdown("**About System**")
    st.markdown("- **Dataset**: UCI Dry Bean (13,611 records)")
    st.markdown("- **Pipeline**: StandardScaler + Classifier")
    st.markdown("- **Status**: Pre-trained models only (no re-training)")

# ── Main Header ───────────────────────────────────────────────────────────────
st.title("🫘 Dry Bean Classification Dashboard")
st.caption("BITS Pilani M.Tech AIML — Machine Learning Assignment 2 · End-to-End Evaluation")

# ── Section: Project Introduction (Collapsible for Clean UX) ──────────────────
with st.expander("📖 Project Overview & 7 Bean Classes (Assignment Spec)", expanded=False):
    col_intro, col_classes = st.columns([1.1, 1], gap="large")
    with col_intro:
        st.markdown("#### Problem Statement & Dataset")
        st.markdown("""
        Dry bean varieties registered by agricultural authorities possess subtle dimensional
        and shape variations. Using image processing and computer vision techniques, **16 morphological
        features** (12 dimensional + 4 shape factors) were extracted from grain images.

        - **Target Task**: Multiclass classification across **7 registered dry bean varieties**.
        - **Source**: UCI Machine Learning Repository (Koklu & Ozkan, 2020).
        - **Dataset Size**: 13,611 total records · 0 missing values · 68 duplicates handled.
        - **Hold-out Split**: Stratified 80% train / 20% test split (`random_state=42`).
        - **Pipelines**: Integrated `StandardScaler` to prevent any data leakage.
        """)
    with col_classes:
        st.markdown("#### 7 Target Bean Classes")
        for bean_name, desc in BEAN_DESCRIPTIONS.items():
            st.markdown(f"- **`{bean_name}`**: {desc}")

# ── Upload Required State ─────────────────────────────────────────────────────
if not uploaded:
    st.info("👈 **Please upload a CSV file in the sidebar to run classification and model evaluation.**")

    c_box1, c_box2 = st.columns([3, 1])
    with c_box1:
        st.markdown("""
        **Don't have a test file ready?**  
        Download the sample held-out test dataset (`test_data.csv` · 2,709 records), then upload it in the sidebar:
        """)
    with c_box2:
        st.download_button(
            label="📥 Download test_data.csv",
            data=test_csv_bytes,
            file_name="test_data.csv",
            mime="text/csv",
            type="primary",
            use_container_width=True,
        )

    # Render benchmark training performance comparison table & charts
    render_model_comparison()
    st.stop()

# ── Load uploaded data ────────────────────────────────────────────────────────
df = pd.read_csv(uploaded)
st.success(f"✅  **Dataset Loaded**: `{uploaded.name}` ({len(df):,} records)")

# Validate feature columns
missing_cols = [c for c in FEATURES if c not in df.columns]
if missing_cols:
    st.error(f"❌  Uploaded dataset is missing required feature columns: {missing_cols}")
    st.stop()

has_labels = "Class" in df.columns

# ── Preview Section ───────────────────────────────────────────────────────────
show_preview = st.checkbox("👁️ Preview Uploaded Dataset & Summary Information", value=False)

if show_preview:
    p_col1, p_col2, p_col3, p_col4 = st.columns(4)
    p_col1.metric("Total Rows", f"{len(df):,}")
    p_col2.metric("Input Features", f"{len(FEATURES)}")
    p_col3.metric("Missing Values", f"{df[FEATURES].isnull().sum().sum()}")
    p_col4.metric("Target Available", "Yes (Class)" if has_labels else "No")

    st.markdown(f"**Sample Data Preview (First {preview_limit} Rows)**")
    st.dataframe(df.head(preview_limit), use_container_width=True, height=min(400, 40 + preview_limit * 35))

    if has_labels:
        st.markdown("**Class Distribution in Uploaded Dataset**")
        class_counts = df["Class"].value_counts().reset_index()
        class_counts.columns = ["Bean Class", "Number of Samples"]
        st.dataframe(class_counts, use_container_width=True)

# ── Run model inference ───────────────────────────────────────────────────────
pipe       = get_pipeline(model_name)
le         = get_le()
X          = df[FEATURES].values
y_pred_enc = pipe.predict(X)
y_pred     = le.inverse_transform(y_pred_enc)
y_proba    = pipe.predict_proba(X)

if has_labels:
    y_true = le.transform(df["Class"].values)
    acc    = accuracy_score(y_true, y_pred_enc)
    auc    = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
    prec   = precision_score(y_true, y_pred_enc, average="weighted", zero_division=0)
    rec    = recall_score(y_true, y_pred_enc, average="weighted", zero_division=0)
    f1     = f1_score(y_true, y_pred_enc, average="weighted", zero_division=0)
    mcc    = matthews_corrcoef(y_true, y_pred_enc)

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 — EVALUATION METRICS
# ═════════════════════════════════════════════════════════════════════════════
st.divider()
st.subheader(f"Evaluation Results — {model_name}")

if not has_labels:
    st.warning("⚠️ The uploaded CSV lacks a `Class` column. Displaying model predictions only (evaluation metrics require true class labels).")
else:
    col_m = st.columns(6)
    tiles = [
        ("Accuracy",  f"{acc:.4f}"),
        ("AUC Score", f"{auc:.4f}"),
        ("Precision", f"{prec:.4f}"),
        ("Recall",    f"{rec:.4f}"),
        ("F1 Score",  f"{f1:.4f}"),
        ("MCC",       f"{mcc:.4f}"),
    ]
    for c, (label, val) in zip(col_m, tiles):
        c.metric(label, val)

    st.caption("Averaging: Precision, Recall, F1 = weighted average | AUC = macro One-vs-Rest (OvR) | MCC = multiclass Matthews Correlation Coefficient")

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 — CONFUSION MATRIX & CLASSIFICATION REPORT
# ═════════════════════════════════════════════════════════════════════════════
if has_labels:
    st.divider()
    left, right = st.columns(2, gap="large")

    with left:
        st.markdown("### Confusion Matrix")
        cm = confusion_matrix(y_true, y_pred_enc)
        fig, ax = plt.subplots(figsize=(6.5, 5.2), dpi=160)
        fig.patch.set_facecolor("#FFFFFF")
        ax.set_facecolor("#FFFFFF")

        cmap_cm = LinearSegmentedColormap.from_list(
            "modern_blue", ["#F8FAFC", "#DBEAFE", "#93C5FD", "#3B82F6", "#1E40AF"]
        )
        sns.heatmap(
            cm, annot=True, fmt="d", cmap=cmap_cm,
            xticklabels=le.classes_, yticklabels=le.classes_,
            linewidths=1.5, linecolor="white", cbar=False, ax=ax,
            annot_kws={"fontsize": 9.5, "fontweight": "600"},
        )
        ax.set_xlabel("Predicted Variety", fontsize=10.5, fontweight="600", color="#334155", labelpad=8)
        ax.set_ylabel("True Variety", fontsize=10.5, fontweight="600", color="#334155", labelpad=8)
        ax.tick_params(axis="x", rotation=30, labelsize=9, colors="#475569")
        ax.tick_params(axis="y", rotation=0, labelsize=9, colors="#475569")
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(f"{model_name} (Accuracy: {acc:.4f})", fontsize=11.5, fontweight="bold", color="#1E293B", pad=12)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with right:
        st.markdown("### Classification Report")
        st.caption("Per-class precision, recall, F1-score and sample support")
        report_dict = classification_report(
            y_true, y_pred_enc,
            target_names=le.classes_, digits=4, output_dict=True
        )
        rows = []
        for cls in le.classes_:
            r = report_dict[cls]
            rows.append({
                "Class":     cls,
                "Precision": round(r["precision"], 4),
                "Recall":    round(r["recall"],    4),
                "F1-Score":  round(r["f1-score"],  4),
                "Support":   int(r["support"]),
            })
        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True,
            height=len(le.classes_) * 36 + 40,
        )

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 — ACTUAL VS PREDICTED TABLE
# ═════════════════════════════════════════════════════════════════════════════
st.divider()
st.markdown("### Actual vs Predicted Results")
st.caption(f"Displaying sample predictions comparing true bean label with model prediction (first {preview_limit} rows)")

pred_rows = {
    "Actual Class":    df["Class"].values[:preview_limit] if has_labels else ["—"] * preview_limit,
    "Predicted Class": y_pred[:preview_limit],
}
if has_labels:
    pred_rows["Prediction Status"] = [
        "✅ Correct" if a == p else "❌ Incorrect"
        for a, p in zip(df["Class"].values[:preview_limit], y_pred[:preview_limit])
    ]
st.dataframe(pd.DataFrame(pred_rows), use_container_width=True, hide_index=True, height=min(400, 40 + preview_limit * 35))

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4 — ALL-MODEL COMPARISON (TRAINING BENCHMARK)
# ═════════════════════════════════════════════════════════════════════════════
render_model_comparison()

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5 — DOWNLOAD PREDICTIONS
# ═════════════════════════════════════════════════════════════════════════════
st.divider()
st.markdown("### Export Predictions")

dl = pd.DataFrame()
if has_labels:
    dl["Actual_Class"] = df["Class"].values
dl["Predicted_Class"] = y_pred
if has_labels:
    dl["Correct"] = np.where(df["Class"].values == y_pred, "Yes", "No")

left_dl, right_dl = st.columns([2, 1])
with left_dl:
    if has_labels:
        correct_count = int((df["Class"].values == y_pred).sum())
        total_count = len(dl)
        acc_rate = (correct_count / total_count) * 100
        st.markdown(f"**Total Records**: `{total_count:,}` | **Correct**: `{correct_count:,}` | **Incorrect**: `{total_count - correct_count:,}` | **Test Accuracy**: `{acc_rate:.2f}%`")
    else:
        st.markdown(f"**Total Predictions**: `{len(dl):,}` records")

with right_dl:
    st.download_button(
        label=f"⬇️ Download Predictions CSV ({model_name})",
        data=dl.to_csv(index=False).encode(),
        file_name=f"predictions_{model_name.replace(' ', '_')}.csv",
        mime="text/csv",
        type="primary",
        use_container_width=True,
    )

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "BITS Pilani M.Tech AIML · Machine Learning Assignment 2 · "
    "UCI Dry Bean Dataset · Scikit-learn Pipelines · Streamlit Deployment"
)
