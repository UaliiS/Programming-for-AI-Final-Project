# ⭐ Yelp Reviews at Scale: Ratings, Categories and Cities

**Final Project — Topic #44 | Cluster 8: Large-Scale Data Processing (Pandas, Polars, Dask, Parquet)**  
**Course:** Programming for Data Analysis and Database (MSc in Applied Artificial Intelligence)  
**Authors:** **Uali Sembay** & **Arystan Aldiyev** (`AAI-2605`)  
**Dataset Source:** [Official Yelp Open Dataset (`Yelp-JSON.zip`, 4.35 GB)](https://business.yelp.com/data/resources/open-dataset/)

---

## 📌 Project Overview

This repository contains the complete, reproducible implementation for **Topic #44: "Yelp Reviews at Scale: Ratings, Categories and Cities"**. It processes **1,345,026 real records** (**747.30 MB** of raw JSONL data extracted directly from the official Yelp Open Dataset: **750,000 real reviews**, **150,346 real businesses**, and **444,680 user profiles**) using out-of-core chunked streaming, partitioned **Apache Parquet**, **Polars LazyFrame**, **Dask**, **Pandas**, **SQLite3** (5-table normalized 3NF schema with 8 advanced SQL queries), parametric/non-parametric statistical hypothesis testing, and leakage-free **TF-IDF NLP star-rating prediction**.

---

## 🗂️ Repository Architecture

```text
yelp_analytics_project/
├── main.py                                   # Interactive Console CLI Menu & Automated Pipeline (--auto)
├── app.py                                    # 5-Tab Interactive Streamlit Analytics Dashboard
├── Yelp_Reviews_At_Scale_Analysis.ipynb      # Complete Jupyter Notebook with Code, Outputs & Plots
├── requirements.txt                          # Exact pinned Python 3.12 package versions
├── README.md                                 # Project Documentation & Execution Guide
├── src/
│   ├── __init__.py
│   ├── config.py                             # Paths, constants, and directory initialization
│   ├── data_generator.py                     # HTTP Range streaming extractor for official Yelp-JSON.zip
│   ├── etl_pipeline.py                       # Out-of-core JSONL -> Partitioned Parquet ETL & Data Quality Audit
│   ├── benchmarks.py                         # Comparative Engine Benchmarks (JSONL Baseline vs Pandas vs Dask vs Polars)
│   ├── database.py                           # Normalized 5-table SQLite builder & 8 Analytical SQL Queries
│   ├── eda_analytics.py                      # Statistical Hypothesis Tests (Welch t, ANOVA, Mann-Whitney) & Plots
│   ├── ml_models.py                          # Leakage-Free Chronological NLP Star-Rating Predictor & Error Analysis
│   └── generate_deliverables.py              # Automated generator for .ipynb and .pptx deliverables
├── docs/
│   ├── FINAL_PROJECT_REPORT.md               # Comprehensive Written Project Report (Methods, Tables, ER Diagram)
│   └── Presentation_Yelp_Reviews_At_Scale.pptx # 10-Slide Project Defense Presentation (.pptx)
├── models/
│   └── yelp_star_predictor.joblib            # Trained TF-IDF + LogisticRegression Pipeline (Saved via joblib)
├── outputs/
│   ├── data_quality_audit.json               # Detailed Data Quality Audit & Memory Optimization Metrics
│   ├── benchmark_results.json                # Wall-clock Time, Peak RAM & Equivalence Check across 4 Engines
│   ├── hypothesis_tests.json                 # Test statistics and p-values for H1, H2, and H3
│   ├── ml_model_metrics.json                 # Accuracy, F1-Macro, MAE, RMSE, Confusion Matrix & Error Samples
│   └── plots/                                # 6 High-Resolution Static PNG Plots + 1 Interactive Plotly HTML
└── data/
    ├── raw/                                  # Extracted official Yelp JSONL files (747.30 MB)
    ├── parquet/                              # Cleaned Snappy Parquet & Year-Partitioned Parquet datasets
    └── db/
        └── yelp_analytics.db                 # Normalized 5-table SQLite Relational Database (1.74M total rows)
```

---

## 🚀 Quick Start & Run Instructions

### 1. Environment Setup
```bash
cd /Users/ualisembay/Desktop/yelp_analytics_project
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the Full Automated Pipeline (Stages 1 to 6)
```bash
python main.py --auto
```
*(If `data/raw/*.json` is already present, Stage 1 verifies the cached 747.30 MB official Yelp JSONL files in `<1s`; otherwise it streams them directly from `https://business.yelp.com/external-assets/files/Yelp-JSON.zip`.)*

### 3. Launch the Interactive Console Menu (CLI)
```bash
python main.py
```
Allows you to interactively run any stage (`1`–`8`) or type custom restaurant/business reviews in Option `[7]` to receive live NLP star-rating predictions and class probabilities.

### 4. Launch the Interactive Streamlit Dashboard
```bash
streamlit run app.py
```
Opens the 5-tab interactive web dashboard powered by **Polars LazyFrame** with sidebar filters for **Year Range (`2005–2022`)**, **Metropolitan City**, **Primary Category**, and **Elite vs. Regular User Tier**, plus live SQL query execution and an interactive review classifier.

---

## 📊 Summary of Key Results

### 1. Multi-Engine Benchmark (748,714 Cleaned Reviews)
| Engine / Approach | Time (s) | Peak RAM (MB) | Global Mean Stars | Speedup vs. Baseline |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline: JSONL Chunked (Pandas)** | `68.559 s` | `405.53 MB` | `3.769781` | `1.00x` |
| **Approach 1: Pandas (Parquet Eager)** | `1.839 s` | `85.12 MB` | `3.769781` | `37.28x` |
| **Approach 2: Dask (Lazy Parquet)** | `2.683 s` | `85.76 MB` | `3.769781` | `25.55x` |
| **Approach 3: Polars LazyFrame (`scan_parquet`)** | **`0.190 s`** | **`0.13 MB`** | `3.769781` | **`361.79x`** |

### 2. NLP Star-Rating Prediction (`Train: 2005–2019`, `Test: 2020–2022`)
| Model Pipeline | Accuracy | Macro-F1 | Weighted-F1 | MAE (Stars) | RMSE (Stars) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline: `DummyClassifier` (Majority 5★)** | `0.5378` | `0.1399` | `0.3761` | `1.2432` | `2.0260` |
| **Model 1: `TF-IDF (1-2 ngrams)` + `MultinomialNB`** | `0.7198` | `0.5043` | `0.6991` | `0.4357` | `0.9676` |
| **Model 2: `TF-IDF (1-2 ngrams)` + `SGDClassifier`** | **`0.7465`** | `0.5444` | `0.7265` | `0.3658` | `0.8449` |
| **Model 3: `TF-IDF (1-2 ngrams)` + `LogisticRegression`** | `0.7381` | **`0.5582`** | **`0.7295`** | **`0.3586`** | **`0.8103`** |

---

## 👥 Team Contributions
- **Uali Sembay (`AAI-2605`, 50%):** Out-of-core HTTP Range streaming extractor, chunked JSONL $\rightarrow$ Partitioned Parquet ETL, memory optimization & data quality audit, multi-engine benchmarking (Pandas vs. Dask vs. Polars), 5-table normalized SQLite database & 8 analytical SQL queries.
- **Arystan Aldiyev (`AAI-2605`, 50%):** Exploratory data analysis (EDA), statistical hypothesis testing (`H1`–`H3`), static & interactive Plotly visualizations, leakage-free chronological NLP modeling & error analysis, interactive CLI menu (`main.py`), and Streamlit dashboard (`app.py`).
