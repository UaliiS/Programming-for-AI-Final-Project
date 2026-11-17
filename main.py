1#!/usr/bin/env python3
"""
Interactive Console Menu & CLI Entrypoint for Yelp Reviews at Scale Project (Topic #44).
Demonstrates all mandatory Technical Requirements (1–11):
- Functional programming (multiple modular functions across src/)
- Lists & Dictionaries for storing state and results
- Conditional logic (if / elif / else)
- While / For loops for interactive menu navigation and batch processing
- Persistent file I/O (.json, .csv, .parquet, .db, .joblib)
- Robust error handling (try...except for invalid user inputs & missing files)

Usage:
  python main.py          -> Launches interactive console menu
  python main.py --auto   -> Runs full end-to-end automated pipeline and exits

Authors: Uali Sembay & Arystan Aldiyev (AAI-2605)
"""

import argparse
import json
import sys
from typing import Dict

from src.benchmarks import run_all_benchmarks
from src.config import (
    AUDIT_REPORT_JSON,
    BENCHMARK_RESULTS_JSON,
    HYPOTHESIS_RESULTS_JSON,
    MODEL_METRICS_JSON,
    PROJECT_METADATA,
    ensure_directories,
)
from src.data_download import stream_extract_official_yelp_dataset
from src.database import build_sqlite_database, execute_all_sql_queries
from src.eda_analytics import generate_all_visualizations, load_analysis_frame, run_statistical_hypotheses
from src.er_diagram import draw_er_diagram
from src.etl_pipeline import run_out_of_core_reviews_etl
from src.ml_models import predict_custom_review, train_and_evaluate_models


def print_banner() -> None:
    """Display project title and team metadata."""
    print("=" * 78)
    print(f"  FINAL PROJECT — TOPIC #{PROJECT_METADATA['topic_number']}: {PROJECT_METADATA['title'].upper()}")
    print(f"  Cluster: {PROJECT_METADATA['cluster']}")
    print(f"  Authors: {', '.join(PROJECT_METADATA['authors'])} | Group: {PROJECT_METADATA['group']}")
    print("=" * 78)


def run_full_automated_pipeline(force_download: bool = False) -> Dict[str, object]:
    """Execute all 6 stages of the project sequentially and return a summary dictionary."""
    ensure_directories()
    summary_state: Dict[str, object] = {}

    print("\n[Stage 1/6] Official Yelp Open Dataset: stream + extract (skipped if already extracted)...")
    manifest = stream_extract_official_yelp_dataset(force=force_download)
    summary_state["download"] = manifest
    print(f"  -> {manifest['business_rows']:,} businesses | {manifest['review_rows_kept']:,} reviews "
          f"({manifest['review_sampling']}) | {manifest['user_rows_kept']:,} real users | {manifest['total_raw_mb']} MB raw JSONL")

    print("\n[Stage 2/6] Chunked JSONL -> Parquet ETL, dtype optimisation and data-quality audit...")
    etl = run_out_of_core_reviews_etl(chunk_size=100_000)
    summary_state["etl_audit"] = etl
    a, mem = etl["reviews_audit"], etl["reviews_memory_optimization"]
    print(f"  -> {a['raw_rows']:,} raw reviews -> {a['cleaned_rows']:,} clean ({etl['reviews_removed_percent']}% removed)")
    print(f"  -> Memory {mem['raw_chunks_in_memory_mb']} MB -> {mem['optimized_chunks_in_memory_mb']} MB "
          f"(-{mem['memory_reduction_percent']}%); disk {mem['raw_jsonl_disk_mb']} MB JSONL -> "
          f"{mem['cleaned_parquet_disk_mb']} MB Parquet")

    print("\n[Stage 3/6] Benchmark: JSONL baseline vs Pandas vs Dask vs Polars (isolated processes)...")
    bench = run_all_benchmarks()
    summary_state["benchmarks"] = bench
    print(f"  -> Results identical across engines: {bench['all_results_identical']} | fastest: {bench['fastest_engine']}")

    print("\n[Stage 4/6] SQLite database, ER diagram and 8 SQL queries...")
    db = build_sqlite_database()
    draw_er_diagram()
    sql_results = execute_all_sql_queries()
    summary_state["database"] = db
    print(f"  -> Tables: {json.dumps(db['row_counts'])} | FK violations: {db['foreign_key_violations']}")
    for q_name, df_q in sql_results.items():
        print(f"  -> {q_name}: {len(df_q)} rows -> outputs/sql_results/{q_name}.csv")

    print("\n[Stage 5/6] EDA, hypothesis tests and visualisations...")
    frame = load_analysis_frame()
    hyps = run_statistical_hypotheses(frame)
    plots = generate_all_visualizations(frame)
    del frame
    summary_state["hypotheses"] = hyps
    for h in hyps.values():
        print(f"  -> {h['title']}: {h['decision']}")
    print(f"  -> {len(plots)} figures saved in outputs/plots/")

    print("\n[Stage 6/6] Star-rating prediction from review text (chronological split)...")
    ml = train_and_evaluate_models()
    summary_state["ml_models"] = ml
    print(f"  -> Best model (chosen on validation): {ml['best_model']}")
    print("\n[SUCCESS] Pipeline finished. Results: outputs/, data/parquet/, data/db/, models/.")
    return summary_state


def show_saved_results() -> None:
    """Print a short summary of the results saved by earlier runs (reads the JSON files)."""
    files = {
        "Data-quality audit": AUDIT_REPORT_JSON,
        "Engine benchmark": BENCHMARK_RESULTS_JSON,
        "Hypothesis tests": HYPOTHESIS_RESULTS_JSON,
        "Model comparison": MODEL_METRICS_JSON,
    }
    for title, path in files.items():
        if not path.exists():
            print(f"[{title}] not computed yet ({path.name} missing).")
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        print(f"\n=== {title} ===")
        if title == "Data-quality audit":
            print(json.dumps(data["reviews_audit"], indent=2))
        elif title == "Engine benchmark":
            for b in data["benchmarks"]:
                print(f"  {b['engine']:<46} {b['time_seconds_median']:>8.3f}s  {b['peak_rss_mb']:>7.1f} MB")
        elif title == "Hypothesis tests":
            for h in data.values():
                print(f"  {h['title']}\n    {h['conclusion']}")
        else:
            for m in data["models_comparison"]:
                print(f"  {m['model_name']:<52} test macro-F1 {m['test']['f1_macro']:.4f}  MAE {m['test']['mae_stars']:.3f}")


def interactive_menu() -> None:
    """Run an interactive CLI menu loop with try...except input handling."""
    print_banner()
    menu_options = [
        ("1", "Download / verify the official Yelp data (streamed, ~4.3 GB archive)"),
        ("2", "Run Out-of-Core JSONL -> Parquet ETL & Data Quality Audit"),
        ("3", "Run Engine Benchmarks (JSONL Baseline vs Pandas vs Dask vs Polars)"),
        ("4", "Build Normalized SQLite DB & Run 8 Analytical SQL Queries"),
        ("5", "Run Statistical Hypothesis Tests & Generate Visualizations"),
        ("6", "Train & Evaluate NLP Star-Rating Prediction Models"),
        ("7", "Predict Star Rating for Your Custom Review Text (Interactive NLP Inference)"),
        ("8", "Run Complete End-to-End Automated Pipeline (Stages 1 to 6)"),
        ("9", "Show saved results (audit, benchmark, hypotheses, models)"),
        ("0", "Exit Program"),
    ]

    while True:
        print("\n" + "-" * 78)
        print("MAIN MENU — Select an operation:")
        for code, desc in menu_options:
            print(f"  [{code}] {desc}")
        print("-" * 78)

        try:
            choice = input("Enter choice (0-9): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting program. Goodbye!")
            break

        try:
            if choice == "1":
                stats = stream_extract_official_yelp_dataset()
                print(json.dumps(stats, indent=2))
            elif choice == "2":
                report = run_out_of_core_reviews_etl()
                print(json.dumps(report["reviews_audit"], indent=2))
                print(json.dumps(report["cleaning_rules_justification"], indent=2))
                print(json.dumps(report["reviews_memory_optimization"], indent=2))
            elif choice == "3":
                res = run_all_benchmarks()
                print(json.dumps(res["benchmarks"], indent=2))
            elif choice == "4":
                counts = build_sqlite_database()
                print("Database:", counts)
                print("ER diagram:", draw_er_diagram())
                queries = execute_all_sql_queries()
                for q_name, df_q in queries.items():
                    print(f"\n--- {q_name} ---")
                    print(df_q.head(5).to_string(index=False))
            elif choice == "5":
                frame = load_analysis_frame()
                hyps = run_statistical_hypotheses(frame)
                for h in hyps.values():
                    print(f"\n{h['title']} -> {h['decision']}\n  {h['conclusion']}")
                plots = generate_all_visualizations(frame)
                print("Saved plots:", json.dumps(plots, indent=2))
            elif choice == "6":
                ml_res = train_and_evaluate_models()
                for m in ml_res["models_comparison"]:
                    print(f"  {m['model_name']:<52} test macro-F1 {m['test']['f1_macro']:.4f}  MAE {m['test']['mae_stars']:.3f}")
                print("Best (validation):", ml_res["best_model"])
            elif choice == "7":
                user_txt = input("\nEnter a restaurant/business review text to classify: ").strip()
                pred = predict_custom_review(user_txt)
                print("\n--- NLP Star Rating Prediction ---")
                print(f"Predicted Rating : {'★' * pred['predicted_stars']} ({pred['predicted_stars']} / 5 Stars)")
                print(f"Sentiment Class  : {pred['sentiment_category']}")
                print(f"Class Probabilities (%): {pred['class_probabilities_pct']}")
            elif choice == "8":
                run_full_automated_pipeline()
            elif choice == "9":
                show_saved_results()
            elif choice == "0":
                print("Exiting Yelp Analytics CLI. Goodbye!")
                break
            else:
                print(f"[ERROR] Unrecognized option '{choice}'. Please enter a number from 0 to 9.")
        except FileNotFoundError as exc:
            print(f"[MISSING FILE] {exc}\n  Hint: run the earlier stages first (or option 8).")
        except ValueError as exc:
            print(f"[INVALID INPUT] {exc}")
        except Exception as exc:
            print(f"[RUNTIME ERROR CAUGHT]: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Yelp Reviews at Scale — Final Project CLI")
    parser.add_argument("--auto", action="store_true", help="Run full automated pipeline non-interactively")
    parser.add_argument("--force-download", action="store_true", help="Re-download the raw Yelp data even if present")
    args = parser.parse_args()

    if args.auto:
        print_banner()
        run_full_automated_pipeline(force_download=args.force_download)
    else:
        interactive_menu()
