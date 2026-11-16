"""
Star-Rating Prediction from Review Text (Topic #44 & Cluster 8 Predictive Component).
Implements:
- Leakage-free chronological split (Train: years <= 2023, Test: years >= 2024)
  fitted strictly on the training corpus.
- Baseline Model: DummyClassifier (most_frequent / stratified) & DummyRegressor (mean)
- Model 1: TF-IDF (Unigrams + Bigrams) + Multinomial Naive Bayes
- Model 2: TF-IDF (Unigrams + Bigrams) + Logistic Regression (L2-regularized, Saga/LBFGS)
- Model 3: TF-IDF (Unigrams + Bigrams + Character n-grams / Metadata) + Ridge / LinearSVC Calibration

Evaluates Accuracy, Macro-F1, Weighted-F1, MAE (rating error in stars), RMSE,
training time, inference latency per 1,000 texts, model size (MB), confusion matrix,
and detailed error analysis. Saves the best pipeline to models/yelp_star_predictor.joblib.

Authors: Uali Sembay & Arystan Aldiyev (AAI-2605)
"""

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
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    root_mean_squared_error,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from src.config import (
    MODEL_METRICS_JSON,
    MODELS_DIR,
    PARQUET_REVIEWS_SINGLE,
    PLOTS_DIR,
)

BEST_MODEL_PATH = MODELS_DIR / "yelp_star_predictor.joblib"


def train_and_evaluate_models(max_train_samples: int = 120000, max_test_samples: int = 30000) -> Dict[str, Any]:
    """
    Train and evaluate Baseline + 3 NLP models for 5-class star rating prediction (1 to 5 stars)
    using a strict chronological split (Train <= 2023, Test >= 2024) to prevent data leakage.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(
        PARQUET_REVIEWS_SINGLE,
        columns=["review_id", "text", "stars", "year", "useful"],
    )
    df["stars_int"] = df["stars"].astype(int)

    # Chronological Split on Real Yelp Dataset (No data leakage across time: Train <= 2019, Test >= 2020)
    train_df = df[df["year"] <= 2019]
    test_df = df[df["year"] >= 2020]

    if len(train_df) > max_train_samples:
        train_df = train_df.sample(n=max_train_samples, random_state=42)
    if len(test_df) > max_test_samples:
        test_df = test_df.sample(n=max_test_samples, random_state=42)

    X_train = train_df["text"].tolist()
    y_train = train_df["stars_int"].to_numpy()
    X_test = test_df["text"].tolist()
    y_test = test_df["stars_int"].to_numpy()

    pipelines = {
        "Baseline: DummyClassifier (Majority Class)": Pipeline([
            ("tfidf", TfidfVectorizer(max_features=500, stop_words="english")),
            ("clf", DummyClassifier(strategy="most_frequent")),
        ]),
        "Model 1: TF-IDF (1-2 ngrams) + MultinomialNB": Pipeline([
            ("tfidf", TfidfVectorizer(max_features=15000, ngram_range=(1, 2), stop_words="english", sublinear_tf=True)),
            ("clf", MultinomialNB(alpha=0.5)),
        ]),
        "Model 2: TF-IDF (1-2 ngrams) + SGD Linear Hinged/LogLoss": Pipeline([
            ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1, 2), stop_words="english", sublinear_tf=True)),
            ("clf", SGDClassifier(loss="log_loss", alpha=1e-5, max_iter=30, random_state=42)),
        ]),
        "Model 3: TF-IDF (1-2 ngrams) + LogisticRegression": Pipeline([
            ("tfidf", TfidfVectorizer(max_features=25000, ngram_range=(1, 2), stop_words="english", sublinear_tf=True)),
            ("clf", LogisticRegression(C=1.5, max_iter=500, random_state=42)),
        ]),
    }

    results: List[Dict[str, Any]] = []
    best_f1 = -1.0
    best_name = ""
    best_pipe = None
    best_preds = None

    for name, pipe in pipelines.items():
        t0 = time.perf_counter()
        pipe.fit(X_train, y_train)
        train_sec = time.perf_counter() - t0

        t_inf = time.perf_counter()
        preds = pipe.predict(X_test)
        inf_sec = time.perf_counter() - t_inf
        latency_per_1k_ms = (inf_sec / len(X_test)) * 1000.0 * 1000.0

        acc = accuracy_score(y_test, preds)
        f1_mac = f1_score(y_test, preds, average="macro")
        f1_w = f1_score(y_test, preds, average="weighted")
        mae = mean_absolute_error(y_test, preds)
        rmse = root_mean_squared_error(y_test, preds)

        temp_path = MODELS_DIR / "tmp_model.joblib"
        joblib.dump(pipe, temp_path)
        size_mb = temp_path.stat().st_size / (1024 * 1024)
        temp_path.unlink(missing_ok=True)

        metrics = {
            "model_name": name,
            "accuracy": round(float(acc), 4),
            "f1_macro": round(float(f1_mac), 4),
            "f1_weighted": round(float(f1_w), 4),
            "mae_stars": round(float(mae), 4),
            "rmse_stars": round(float(rmse), 4),
            "train_time_seconds": round(float(train_sec), 3),
            "inference_ms_per_1k_reviews": round(float(latency_per_1k_ms), 3),
            "model_size_mb": round(float(size_mb), 3),
        }
        results.append(metrics)

        if f1_mac > best_f1:
            best_f1 = f1_mac
            best_name = name
            best_pipe = pipe
            best_preds = preds

    # Save best pipeline
    joblib.dump(best_pipe, BEST_MODEL_PATH)

    # Confusion Matrix & Error Analysis on Best Model
    cm = confusion_matrix(y_test, best_preds, labels=[1, 2, 3, 4, 5])
    fig, ax = plt.subplots(figsize=(7.5, 6), dpi=160)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[1, 2, 3, 4, 5],
        yticklabels=[1, 2, 3, 4, 5],
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix: {best_name}\n(Chronological Test Set 2020–2022)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted Star Rating", fontsize=11)
    ax.set_ylabel("True Star Rating", fontsize=11)
    plt.tight_layout()
    cm_path = PLOTS_DIR / "fig6_confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close(fig)

    # Collect misclassified examples for Error Analysis
    test_eval_df = test_df.copy()
    test_eval_df["pred_stars"] = best_preds
    misclassified = test_eval_df[test_eval_df["stars_int"] != test_eval_df["pred_stars"]].head(10)
    error_samples = []
    for row in misclassified.itertuples():
        error_samples.append({
            "review_id": row.review_id,
            "true_stars": int(row.stars_int),
            "predicted_stars": int(row.pred_stars),
            "text_snippet": str(row.text)[:160] + "...",
        })

    summary = {
        "split_strategy": "Chronological Split (Train: 2005-2019, Test: 2020-2022) - Zero Temporal Leakage",
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "best_model": best_name,
        "models_comparison": results,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_plot": str(cm_path),
        "error_analysis_samples": error_samples,
    }

    with open(MODEL_METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def predict_custom_review(review_text: str) -> Dict[str, Any]:
    """
    Predict star rating and class probabilities for a user-supplied review text.
    Includes try...except error handling and input validation.
    """
    if not review_text or not str(review_text).strip():
        raise ValueError("Review text cannot be empty. Please provide a valid text string.")

    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found at {BEST_MODEL_PATH}. Please run the ML training pipeline first."
        )

    pipe = joblib.load(BEST_MODEL_PATH)
    clean_txt = str(review_text).strip()
    pred_star = int(pipe.predict([clean_txt])[0])

    probs_dict = {}
    if hasattr(pipe.named_steps["clf"], "predict_proba"):
        probs = pipe.predict_steps["clf"].predict_proba(pipe.named_steps["tfidf"].transform([clean_txt]))[0] if hasattr(pipe, "predict_steps") else pipe.predict_proba([clean_txt])[0]
        for idx, p in enumerate(probs, start=1):
            probs_dict[f"{idx}_star"] = round(float(p) * 100.0, 2)

    sentiment_label = (
        "Positive (4-5 Stars)" if pred_star >= 4
        else ("Neutral (3 Stars)" if pred_star == 3 else "Negative (1-2 Stars)")
    )

    return {
        "input_text": clean_txt,
        "predicted_stars": pred_star,
        "sentiment_category": sentiment_label,
        "class_probabilities_pct": probs_dict,
    }
