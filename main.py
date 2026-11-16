#!/usr/bin/env python3
"""
Interactive Console Menu & CLI Entrypoint for Yelp Reviews at Scale Project (Topic #44).
Demonstrates all mandatory Technical Requirements (1–11):
- Functional programming (multiple modular functions across src/)
- Lists & Dictionaries for storing state and results
- Conditional logic (if / elif / else)
- While / For loops for interactive menu navigation and batch processing
- Persistent file I/O (.json, .parquet, .db, .joblib)
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
from src.data_generator import generate_dataset_files
from src.database import build_sqlite_database, execute_all_sql_queries
from src.eda_analytics import generate_all_visualizations, run_statistical_hypotheses
from src.etl_pipeline import run_out_of_core_reviews_etl
from src.ml_models import predict_custom_review, train_and_evaluate_models


def print_banner() -> None:
    """Display project title and team metadata."""
    print("=" * 78)
    print(f"  FINAL PROJECT — TOPIC #{PROJECT_METADATA['topic_number']}: {PROJECT_METADATA['title'].upper()}")
    print(f"  Cluster: {PROJECT_METADATA['cluster']}")
    print(f"  Authors: {', '.join(PROJECT_METADATA['authors'])} | Group: {PROJECT_METADATA['group']}")
    print("=" * 78)


def run_full_automated_pipeline() -> Dict[str, object]:
    """Execute all 6 stages of the project sequentially and return summary dictionary."""
    ensure_directories()
    summary_state: Dict[str, object] = {}

    print("\n[Stage 1/6] Checking / Extracting Official Yelp Open Dataset (>750K reviews, 150K businesses, ~747 MB)...")
    gen_stats = generate_dataset_files(num_reviews=750000, num_businesses=150346, num_users=444680, force=False)
    summary_state["generation"] = gen_stats
    print(f"  -> Raw JSONL Ready: Total Size = {gen_stats['total_mb']} MB (Reviews: {gen_stats['review_mb']} MB)")

    print("\n[Stage 2/6] Running Out-of-Core JSONL -> Partitioned Parquet ETL & Data Quality Audit...")
    etl_report = run_out_of_core_reviews_etl(chunk_size=100000)
    summary_state["etl_audit"] = etl_report
    rev_audit = etl_report["reviews_audit"]
    mem_opt = etl_report["reviews_memory_optimization"]
    print(
        f"  -> Audited {rev_audit['raw_total_rows']:,} raw reviews -> {rev_audit['cleaned_total_rows']:,} cleaned rows."
    )
    print(f"  -> Audit Summary: {json.dumps(rev_audit)}")
    print(
        f"  -> Memory Optimization: {mem_opt['raw_in_memory_mb']} MB -> {mem_opt['optimized_in_memory_mb']} MB "
        f"({mem_opt['memory_reduction_percent']}% RAM reduction); Disk: {mem_opt['raw_jsonl_disk_mb']} MB JSONL -> "
        f"{mem_opt['cleaned_parquet_disk_mb']} MB Parquet ({mem_opt['disk_compression_ratio']}x compression)."
    )

    print("\n[Stage 3/6] Benchmarking Engines: JSONL Chunked Baseline vs. Pandas vs. Dask vs. Polars Lazy...")
    bench_report = run_all_benchmarks()
    summary_state["benchmarks"] = bench_report
    for b in bench_report["benchmarks"]:
        print(
            f"  -> {b['engine']:<35} | Time: {b['time_seconds']:>6.3f}s | "
            f"Peak RAM: {b['peak_memory_mb']:>6.2f} MB | Mean Stars: {b['global_mean_stars']:.6f}"
        )
    print(
        f"  -> Numerical Equivalence Verified: {bench_report['consistency_check']['exact_equivalence_verified']} "
        f"(Polars Speedup vs Baseline: {bench_report['speedup_polars_vs_jsonl_baseline']}x)"
    )

    print("\n[Stage 4/6] Building Normalized 5-Table SQLite Database & Executing 8 Analytical SQL Queries...")
    db_counts = build_sqlite_database()
    sql_results = execute_all_sql_queries()
    summary_state["db_counts"] = db_counts
    print(f"  -> SQLite Tables Populated: {json.dumps(db_counts)}")
    for q_name, df_q in sql_results.items():
        print(f"  -> Executed {q_name}: returned {len(df_q)} rows x {len(df_q.columns)} cols.")

    print("\n[Stage 5/6] Running Statistical Hypothesis Tests & Generating Visualizations (Static + Plotly)...")
    hyp_results = run_statistical_hypotheses()
    plots = generate_all_visualizations()
    summary_state["hypotheses"] = hyp_results
    summary_state["plots"] = plots
    for k, h in hyp_results.items():
        print(f"  -> {h['title']}: {h['decision']}")
    print(f"  -> Generated {len(plots)} visualizations in outputs/plots/.")

    print("\n[Stage 6/6] Training & Evaluating NLP Star-Rating Prediction Models (Chronological Split)...")
    ml_report = train_and_evaluate_models()
    summary_state["ml_models"] = ml_report
    for m in ml_report["models_comparison"]:
        print(
            f"  -> {m['model_name']:<52} | Acc: {m['accuracy']:.4f} | "
            f"F1-Macro: {m['f1_macro']:.4f} | MAE: {m['mae_stars']:.4f} | Time: {m['train_time_seconds']:.2f}s"
        )
    print(f"  -> Best Model Saved: {ml_report['best_model']}")
    print("\n[SUCCESS] Full pipeline completed! All outputs saved to data/, models/, and outputs/.")
    return summary_state


def interactive_menu() -> None:
    """Run an interactive CLI menu loop with try...except input handling."""
    print_banner()
    menu_options = [
        ("1", "Generate / Verify Large-Scale Raw Yelp JSONL Dataset (>550K rows)"),
        ("2", "Run Out-of-Core JSONL -> Parquet ETL & Data Quality Audit"),
        ("3", "Run Engine Benchmarks (JSONL Baseline vs Pandas vs Dask vs Polars)"),
        ("4", "Build Normalized SQLite DB & Run 8 Analytical SQL Queries"),
        ("5", "Run Statistical Hypothesis Tests & Generate Visualizations"),
        ("6", "Train & Evaluate NLP Star-Rating Prediction Models"),
        ("7", "Predict Star Rating for Your Custom Review Text (Interactive NLP Inference)"),
        ("8", "Run Complete End-to-End Automated Pipeline (Stages 1 to 6)"),
        ("0", "Exit Program"),
    ]

    while True:
        print("\n" + "-" * 78)
        print("MAIN MENU — Select an operation:")
        for code, desc in menu_options:
            print(f"  [{code}] {desc}")
        print("-" * 78)

        try:
            choice = input("Enter choice (0-8): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting program. Goodbye!")
            break

        try:
            if choice == "1":
                stats = generate_dataset_files()
                print(json.dumps(stats, indent=2))
            elif choice == "2":
                report = run_out_of_core_reviews_etl()
                print(json.dumps(report["reviews_audit"], indent=2))
                print(json.dumps(report["reviews_memory_optimization"], indent=2))
            elif choice == "3":
                res = run_all_benchmarks()
                print(json.dumps(res["benchmarks"], indent=2))
            elif choice == "4":
                counts = build_sqlite_database()
                print("Database counts:", counts)
                queries = execute_all_sql_queries()
                for q_name, df_q in queries.items():
                    print(f"\n--- {q_name} ---")
                    print(df_q.head(5).to_string(index=False))
            elif choice == "5":
                hyps = run_statistical_hypotheses()
                print(json.dumps(hyps, indent=2))
                plots = generate_all_visualizations()
                print("Saved plots:", json.dumps(plots, indent=2))
            elif choice == "6":
                ml_res = train_and_evaluate_models()
                print(json.dumps(ml_res["models_comparison"], indent=2))
            elif choice == "7":
                user_txt = input("\nEnter a restaurant/business review text to classify: ").strip()
                pred = predict_custom_review(user_txt)
                print("\n--- NLP Star Rating Prediction ---")
                print(f"Predicted Rating : {'★' * pred['predicted_stars']} ({pred['predicted_stars']} / 5 Stars)")
                print(f"Sentiment Class  : {pred['sentiment_category']}")
                print(f"Class Probabilities (%): {pred['class_probabilities_pct']}")
            elif choice == "8":
                run_full_automated_pipeline()
            elif choice == "0":
                print("Exiting Yelp Analytics CLI. Goodbye!")
                break
            else:
                print(f"[ERROR] Unrecognized option '{choice}'. Please enter a number from 0 to 8.")
        except Exception as exc:
            print(f"[RUNTIME ERROR CAUGHT]: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Yelp Reviews at Scale — Final Project CLI")
    parser.add_argument("--auto", action="store_true", help="Run full automated pipeline non-interactively")
    args = parser.parse_args()

    if args.auto:
        print_banner()
        run_full_automated_pipeline()
    else:
        interactive_menu()
