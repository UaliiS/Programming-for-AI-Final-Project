"""
Star-rating prediction from review text (topic focus: "TF-IDF baseline").

Split (chronological, no leakage):
    train      : reviews from years <= 2018   (sampled to `n_train`)
    validation : reviews from 2019            (used ONLY to choose the best model)
    test       : reviews from 2020 onwards    (reported once, never used for choices)
Exact duplicate texts were already removed in the ETL, so the same text cannot appear
in two splits. Every vectoriser is inside a Pipeline, so it is fitted on train only.

Models (text representations are what we compare, per Requirement 9):
    B0  Majority class (DummyClassifier)                     — trivial baseline
    B1  TF-IDF word 1-2 grams + Logistic Regression          — the topic's TF-IDF baseline
    A1  Bag-of-Words (binary counts) + Multinomial Naive Bayes
    A2  TF-IDF word 1-2 grams, 100K vocabulary + Linear SVM (class-balanced)
    A3  LSA dense embeddings (TF-IDF -> TruncatedSVD 300 dims) + Logistic Regression
Criteria: accuracy, macro-F1, MAE in stars, within-±1-star accuracy, training time,
inference latency, peak process memory while training (RSS sampled with psutil),
and model size on disk.

Authors: Uali Sembay & Arystan Aldiyev (AAI-2605)
"""

import gc
import json
import time
from typing import Any, Dict, List

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import TruncatedSVD
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, mean_absolute_error
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.svm import LinearSVC

from src.benchmarks import PeakRSSSampler
from src.config import BEST_MODEL_PATH, MODEL_METRICS_JSON, MODELS_DIR, PARQUET_REVIEWS_SINGLE, PLOTS_DIR

SEED = 42
TRAIN_MAX_YEAR = 2018
VAL_YEAR = 2019
TEST_MIN_YEAR = 2020


def build_pipelines() -> Dict[str, Pipeline]:
    return {
        "B0: Majority class (Dummy)": Pipeline([
            ("vec", CountVectorizer(max_features=10)),
            ("clf", DummyClassifier(strategy="most_frequent")),
        ]),
        "B1: TF-IDF + LogisticRegression (baseline)": Pipeline([
            ("vec", TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=50_000, sublinear_tf=True)),
            ("clf", LogisticRegression(C=2.0, max_iter=1000, random_state=SEED)),
        ]),
        "A1: Bag-of-Words + MultinomialNB": Pipeline([
            ("vec", CountVectorizer(ngram_range=(1, 2), min_df=3, max_features=50_000, binary=True)),
            ("clf", MultinomialNB(alpha=0.5)),
        ]),
        "A2: TF-IDF 1-2 grams (100K vocab) + LinearSVC, class-balanced": Pipeline([
            ("vec", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=100_000, sublinear_tf=True)),
            ("clf", LinearSVC(C=0.3, class_weight="balanced", random_state=SEED)),
        ]),
        "A3: LSA embeddings (SVD-300) + LogisticRegression": Pipeline([
            ("vec", TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=50_000, sublinear_tf=True)),
            ("svd", TruncatedSVD(n_components=300, random_state=SEED)),
            ("norm", Normalizer()),
            ("clf", LogisticRegression(C=2.0, max_iter=1000, random_state=SEED)),
        ]),
    }


def load_splits(n_train: int, n_val: int, n_test: int):
    df = pd.read_parquet(PARQUET_REVIEWS_SINGLE, columns=["review_id", "text", "stars", "year"])
    df["stars"] = df["stars"].astype(int)
    train = df[df["year"] <= TRAIN_MAX_YEAR]
    val = df[df["year"] == VAL_YEAR]
    test = df[df["year"] >= TEST_MIN_YEAR]
    train = train.sample(n=min(n_train, len(train)), random_state=SEED)
    val = val.sample(n=min(n_val, len(val)), random_state=SEED)
    test = test.sample(n=min(n_test, len(test)), random_state=SEED)
    del df
    gc.collect()
    return train, val, test


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro")), 4),
        "f1_weighted": round(float(f1_score(y_true, y_pred, average="weighted")), 4),
        "mae_stars": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "within_1_star": round(float(np.mean(np.abs(y_true - y_pred) <= 1)), 4),
    }


def train_and_evaluate_models(n_train: int = 120_000, n_val: int = 20_000, n_test: int = 30_000) -> Dict[str, Any]:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    train, val, test = load_splits(n_train, n_val, n_test)
    Xtr, ytr = train["text"].tolist(), train["stars"].to_numpy()
    Xva, yva = val["text"].tolist(), val["stars"].to_numpy()
    Xte, yte = test["text"].tolist(), test["stars"].to_numpy()

    results: List[Dict[str, Any]] = []
    fitted: Dict[str, Pipeline] = {}
    for name, pipe in build_pipelines().items():
        gc.collect()
        with PeakRSSSampler(interval=0.02) as mem:
            t0 = time.perf_counter()
            pipe.fit(Xtr, ytr)
            train_sec = time.perf_counter() - t0

        val_pred = pipe.predict(Xva)
        t1 = time.perf_counter()
        test_pred = pipe.predict(Xte)
        infer_sec = time.perf_counter() - t1

        tmp = MODELS_DIR / "_tmp_size.joblib"
        joblib.dump(pipe, tmp, compress=3)
        size_mb = tmp.stat().st_size / 2**20
        tmp.unlink(missing_ok=True)

        row = {
            "model_name": name,
            "validation": evaluate(yva, val_pred),
            "test": evaluate(yte, test_pred),
            "train_time_seconds": round(train_sec, 2),
            "inference_ms_per_1k_reviews": round(1000 * 1000 * infer_sec / len(Xte), 1),
            "train_memory_increase_mb": round(mem.peak_mb - mem.baseline_mb, 1),
            "model_size_mb": round(size_mb, 2),
        }
        results.append(row)
        fitted[name] = pipe
        print(f"  -> {name:<52} val F1 {row['validation']['f1_macro']:.4f} | test F1 {row['test']['f1_macro']:.4f} "
              f"| MAE {row['test']['mae_stars']:.3f} | {train_sec:6.1f}s")

    # model selection on VALIDATION macro-F1 only
    best = max(results, key=lambda r: r["validation"]["f1_macro"])
    best_pipe = fitted[best["model_name"]]
    joblib.dump(best_pipe, BEST_MODEL_PATH, compress=3)
    baseline = next(r for r in results if r["model_name"].startswith("B1"))

    best_pred = best_pipe.predict(Xte)
    cm = confusion_matrix(yte, best_pred, labels=[1, 2, 3, 4, 5])
    errors = int(cm.sum() - np.trace(cm))
    adjacent = int(sum(cm[i, j] for i in range(5) for j in range(5) if abs(i - j) == 1))
    per_class_f1 = f1_score(yte, best_pred, average=None, labels=[1, 2, 3, 4, 5])

    fig, ax = plt.subplots(figsize=(7.5, 6), dpi=150)
    cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100
    sns.heatmap(cm_pct, annot=cm, fmt="d", cmap="Blues", xticklabels=[1, 2, 3, 4, 5],
                yticklabels=[1, 2, 3, 4, 5], ax=ax, cbar_kws={"label": "% of true class"})
    ax.set_title(f"Confusion matrix on test set ({TEST_MIN_YEAR}+)\n{best['model_name']}", fontsize=11, fontweight="bold")
    ax.set_xlabel("Predicted stars"); ax.set_ylabel("True stars")
    fig.tight_layout()
    cm_path = PLOTS_DIR / "fig6_confusion_matrix.png"
    fig.savefig(cm_path); plt.close(fig)

    test_eval = test.assign(pred=best_pred)
    wrong = test_eval[test_eval["stars"] != test_eval["pred"]]
    far = wrong[(wrong["stars"] - wrong["pred"]).abs() >= 3].head(5)
    near = wrong[(wrong["stars"] - wrong["pred"]).abs() == 1].head(5)
    samples = [{"review_id": r.review_id, "true_stars": int(r.stars), "predicted_stars": int(r.pred),
                "text_snippet": str(r.text)[:200].replace("\n", " ") + "..."} for r in pd.concat([far, near]).itertuples()]

    summary = {
        "split_strategy": (f"Chronological: train years <= {TRAIN_MAX_YEAR} ({len(Xtr):,} sampled), "
                           f"validation {VAL_YEAR} ({len(Xva):,}), test >= {TEST_MIN_YEAR} ({len(Xte):,})"),
        "selection_rule": "best validation macro-F1 (test set used only for final reporting)",
        "train_samples": len(Xtr), "validation_samples": len(Xva), "test_samples": len(Xte),
        "test_class_distribution_percent": {int(k): round(float(v) * 100, 2)
                                            for k, v in pd.Series(yte).value_counts(normalize=True).sort_index().items()},
        "models_comparison": results,
        "best_model": best["model_name"],
        "best_vs_tfidf_baseline": {
            "macro_f1_gain": round(best["test"]["f1_macro"] - baseline["test"]["f1_macro"], 4),
            "mae_change": round(best["test"]["mae_stars"] - baseline["test"]["mae_stars"], 4),
        },
        "best_per_class_f1": {str(k): round(float(v), 4) for k, v in zip([1, 2, 3, 4, 5], per_class_f1)},
        "confusion_matrix": cm.tolist(),
        "errors_total": errors,
        "errors_adjacent_share": round(adjacent / max(errors, 1), 4),
        "confusion_matrix_plot": "outputs/plots/fig6_confusion_matrix.png",
        "error_analysis_samples": samples,
    }
    with open(MODEL_METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def predict_custom_review(review_text: str, model_path=None) -> Dict[str, Any]:
    """Predict the star rating of a user-typed review with the saved best model."""
    model_path = model_path or BEST_MODEL_PATH
    if not review_text or not str(review_text).strip():
        raise ValueError("Review text cannot be empty.")
    if not model_path.exists():
        raise FileNotFoundError(f"Trained model not found at {model_path}. Run the model training stage first.")
    pipe = joblib.load(model_path)
    text = str(review_text).strip()
    pred = int(pipe.predict([text])[0])
    probs: Dict[str, float] = {}
    if hasattr(pipe, "predict_proba"):
        try:
            for cls, pr in zip(pipe.classes_, pipe.predict_proba([text])[0]):
                probs[f"{int(cls)}_star"] = round(float(pr) * 100, 2)
        except AttributeError:
            probs = {}
    label = "Positive (4-5 stars)" if pred >= 4 else ("Neutral (3 stars)" if pred == 3 else "Negative (1-2 stars)")
    return {"input_text": text, "predicted_stars": pred, "sentiment_category": label, "class_probabilities_pct": probs}
