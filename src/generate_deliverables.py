"""
Generate the Jupyter Notebook (Yelp_Reviews_At_Scale_Analysis.ipynb) and
PowerPoint Presentation (docs/Presentation_Yelp_Reviews_At_Scale.pptx)
using the exact empirical results from the official Yelp Open Dataset run.

Authors: Uali Sembay & Arystan Aldiyev (AAI-2605)
"""

import json
from pathlib import Path
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

from src.config import (
    AUDIT_REPORT_JSON,
    BASE_DIR,
    BENCHMARK_RESULTS_JSON,
    HYPOTHESIS_RESULTS_JSON,
    MODEL_METRICS_JSON,
    PLOTS_DIR,
    REPORTS_DIR,
)


def build_jupyter_notebook() -> Path:
    """Create a complete, runnable Jupyter Notebook (.ipynb) covering all project tasks."""
    nb_path = BASE_DIR / "Yelp_Reviews_At_Scale_Analysis.ipynb"

    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Final Project — Topic #44: Yelp Reviews at Scale: Ratings, Categories and Cities\n",
                "**Course:** Programming for Data Analysis and Database  \n",
                "**Program:** MSc in Applied Artificial Intelligence  \n",
                "**Cluster:** Cluster 8 — Large-Scale Data Processing (Pandas, Polars, Dask, Parquet)  \n",
                "**Team Members:** **Uali Sembay** & **Arystan Aldiyev** (`AAI-2605`)  \n",
                "**Dataset Source:** [Official Yelp Open Dataset (`Yelp-JSON.zip`, 4.35 GB)](https://business.yelp.com/data/resources/open-dataset/)\n",
                "\n",
                "---\n",
                "### Executive Summary of Notebook Sections\n",
                "1. **Out-of-Core Streaming Extraction & Parquet ETL:** Line-by-line JSONL processing of **750,000 real Yelp reviews** (`547.06 MB`), **150,346 real businesses** (`113.36 MB`), and **444,680 user profiles** (`86.88 MB`), totaling **747.30 MB** (`1,345,026` total records).\n",
                "2. **Data Quality Audit & Memory Optimization:** Auditing duplicate copy-paste spam (`1,273` rows), ultra-short reviews (`13` rows), missing business categories (`103` rows), and extreme vote outliers (`55` rows), achieving **43.3% RAM reduction** and **1.8x Snappy Parquet compression**.\n",
                "3. **Multi-Engine Benchmarking:** Comparing **JSONL Chunked Baseline** (`68.56s`) vs. **Pandas Parquet** (`1.84s`) vs. **Dask Lazy DataFrame** (`2.68s`) vs. **Polars LazyFrame** (`0.19s`, **361.8x speedup**).\n",
                "4. **Relational Database (5-Table Normalized SQLite Schema) & 8 Analytical SQL Queries:** Executing `JOIN`, `GROUP BY / HAVING`, `CTE`, subqueries, and window functions (`RANK`, `DENSE_RANK`, `LAG`, `NTILE`, `AVG OVER`).\n",
                "5. **Exploratory Data Analysis & Statistical Hypothesis Testing:** Testing 3 hypotheses (`Welch's t-test`, `Levene's test`, `One-Way ANOVA`, `Mann-Whitney U`, `Spearman rho`) and rendering static + interactive Plotly charts.\n",
                "6. **Leakage-Free NLP Star-Rating Prediction:** Training Baseline (`DummyClassifier`) vs. `MultinomialNB`, `SGDClassifier`, and `LogisticRegression` on `TF-IDF (1-2 ngrams)` with a strict chronological split (`Train: 2005–2019`, `Test: 2020–2022`)."
            ],
        },
        {
            "cell_type": "code",
            "execution_count": 1,
            "metadata": {},
            "outputs": [],
            "source": [
                "import json\n",
                "import pandas as pd\n",
                "import polars as pl\n",
                "from IPython.display import Image, display\n",
                "\n",
                "from src.config import (\n",
                "    AUDIT_REPORT_JSON,\n",
                "    BENCHMARK_RESULTS_JSON,\n",
                "    HYPOTHESIS_RESULTS_JSON,\n",
                "    MODEL_METRICS_JSON,\n",
                "    PLOTS_DIR,\n",
                ")\n",
                "from src.database import execute_all_sql_queries\n",
                "from src.ml_models import predict_custom_review\n",
                "\n",
                "print('Environment and project modules loaded successfully!')"
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Data Quality Audit & Memory Optimization Results"
            ],
        },
        {
            "cell_type": "code",
            "execution_count": 2,
            "metadata": {},
            "outputs": [],
            "source": [
                "with open(AUDIT_REPORT_JSON, 'r') as f:\n",
                "    audit_report = json.load(f)\n",
                "\n",
                "print('=== REVIEWS DATA QUALITY AUDIT ===')\n",
                "display(pd.DataFrame([audit_report['reviews_audit']]))\n",
                "\n",
                "print('\\n=== MEMORY & DISK OPTIMIZATION ===')\n",
                "display(pd.DataFrame([audit_report['reviews_memory_optimization']]))"
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Large-Scale Engine Benchmark: JSONL Chunked Baseline vs. Pandas vs. Dask vs. Polars"
            ],
        },
        {
            "cell_type": "code",
            "execution_count": 3,
            "metadata": {},
            "outputs": [],
            "source": [
                "with open(BENCHMARK_RESULTS_JSON, 'r') as f:\n",
                "    bench_data = json.load(f)\n",
                "\n",
                "df_bench = pd.DataFrame(bench_data['benchmarks'])\n",
                "display(df_bench)\n",
                "display(Image(filename=str(PLOTS_DIR / 'fig3_engine_benchmarks.png')))"
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Normalized 5-Table SQLite Database & 8 Analytical SQL Queries"
            ],
        },
        {
            "cell_type": "code",
            "execution_count": 4,
            "metadata": {},
            "outputs": [],
            "source": [
                "sql_results = execute_all_sql_queries()\n",
                "for q_name, df_q in sql_results.items():\n",
                "    print(f'\\n================ {q_name} ================')\n",
                "    display(df_q.head(6))"
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Exploratory Data Analysis & Statistical Hypothesis Testing"
            ],
        },
        {
            "cell_type": "code",
            "execution_count": 5,
            "metadata": {},
            "outputs": [],
            "source": [
                "with open(HYPOTHESIS_RESULTS_JSON, 'r') as f:\n",
                "    hyp_data = json.load(f)\n",
                "\n",
                "for key, val in hyp_data.items():\n",
                "    print(f\"*** {val['title']} -> {val['decision']} ***\")\n",
                "    print(f\"Conclusion: {val['conclusion']}\\n\")\n",
                "\n",
                "display(Image(filename=str(PLOTS_DIR / 'fig1_elite_vs_regular_ratings.png')))\n",
                "display(Image(filename=str(PLOTS_DIR / 'fig2_city_category_heatmap.png')))\n",
                "display(Image(filename=str(PLOTS_DIR / 'fig4_yearly_city_trends.png')))\n",
                "display(Image(filename=str(PLOTS_DIR / 'fig5_iqr_boxplots.png')))"
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. NLP Star-Rating Prediction from Review Text (Chronological Split) & Error Analysis"
            ],
        },
        {
            "cell_type": "code",
            "execution_count": 6,
            "metadata": {},
            "outputs": [],
            "source": [
                "with open(MODEL_METRICS_JSON, 'r') as f:\n",
                "    ml_data = json.load(f)\n",
                "\n",
                "print('Split Strategy:', ml_data['split_strategy'])\n",
                "display(pd.DataFrame(ml_data['models_comparison']))\n",
                "display(Image(filename=str(PLOTS_DIR / 'fig6_confusion_matrix.png')))\n",
                "\n",
                "print('\\n=== SAMPLE MISCLASSIFIED REVIEWS (ERROR ANALYSIS) ===')\n",
                "display(pd.DataFrame(ml_data['error_analysis_samples']).head(5))"
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 6. Interactive Custom Review Prediction Demo"
            ],
        },
        {
            "cell_type": "code",
            "execution_count": 7,
            "metadata": {},
            "outputs": [],
            "source": [
                "sample_reviews = [\n",
                "    'Phenomenal dinner in New Orleans! Fresh oysters, attentive waiter, and incredible jazz ambiance.',\n",
                "    'Waited over an hour for lukewarm pizza and the cashier was extremely rude. Never coming back.',\n",
                "    'Decent neighborhood coffee shop. Pastries are okay and WiFi works, though prices are a bit average.'\n",
                "]\n",
                "for txt in sample_reviews:\n",
                "    res = predict_custom_review(txt)\n",
                "    print(f\"Review: {txt[:75]}...\")\n",
                "    print(f\"  -> Predicted: {res['predicted_stars']} Stars ({res['sentiment_category']}) | Probs: {res['class_probabilities_pct']}\\n\")"
            ],
        },
    ]

    nb_dict = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.12.0",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb_dict, f, indent=2)

    return nb_path


def build_powerpoint_presentation() -> Path:
    """Create a 12-slide professional PowerPoint presentation (.pptx) with embedded charts & tables."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    pptx_path = REPORTS_DIR / "Presentation_Yelp_Reviews_At_Scale.pptx"

    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    def add_header(slide, title_text: str, subtitle_text: str = ""):
        title_box = slide.shapes.add_textbox(Inches(0.6), Inches(0.35), Inches(12.1), Inches(0.9))
        tf = title_box.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = title_text
        p.font.size = Pt(28)
        p.font.bold = True
        p.font.color.rgb = RGBColor(20, 45, 90)
        if subtitle_text:
            p2 = tf.add_paragraph()
            p2.text = subtitle_text
            p2.font.size = Pt(14)
            p2.font.color.rgb = RGBColor(90, 100, 115)

    def add_bullets(slide, items: list, left=0.6, top=1.4, width=12.0, height=5.5, font_size=16):
        box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
        tf = box.text_frame
        tf.word_wrap = True
        for i, item in enumerate(items):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.text = f"• {item}"
            p.font.size = Pt(font_size)
            p.space_after = Pt(10)
            p.font.color.rgb = RGBColor(35, 35, 35)

    blank_layout = prs.slide_layouts[6]

    # Slide 1: Title Slide
    s1 = prs.slides.add_slide(blank_layout)
    add_header(
        s1,
        "Yelp Reviews at Scale: Ratings, Categories and Cities",
        "Final Project — Topic #44 | Cluster 8: Large-Scale Data Processing (Pandas, Polars, Dask, Parquet)",
    )
    add_bullets(
        s1,
        [
            "Course: Programming for Data Analysis and Database (MSc in Applied Artificial Intelligence)",
            "Team Members: Uali Sembay & Arystan Aldiyev | Group: AAI-2605",
            "Official Data Source: Yelp Open Dataset (https://business.yelp.com/data/resources/open-dataset/)",
            "Scale Processed: 750,000 Real Reviews (547.06 MB) + 150,346 Real Businesses (113.36 MB) + 444,680 Users (86.88 MB) = 747.30 MB Raw JSONL (1,345,026 Records)",
            "Core Stack: Python 3.12, Polars LazyFrame, Dask, PyArrow Parquet, Pandas, SQLite3, Scikit-Learn, Plotly, Streamlit",
        ],
        top=1.8,
        font_size=18,
    )

    # Slide 2: Team Contributions & Project Architecture
    s2 = prs.slides.add_slide(blank_layout)
    add_header(s2, "1. Team Contributions & Modular Software Architecture", "Equal Partnership & Complete Technical Compliance (Requirements 1–11)")
    add_bullets(
        s2,
        [
            "Uali Sembay (50% Contribution):",
            "   - Out-of-Core HTTP Range Streaming Extractor for 4.35 GB Yelp-JSON.zip & Chunked JSONL -> Partitioned Parquet ETL (src/data_generator.py, src/etl_pipeline.py)",
            "   - Multi-Engine Benchmarking: JSONL Chunked Baseline vs. Pandas vs. Dask vs. Polars LazyFrame (src/benchmarks.py)",
            "   - Normalized 5-Table SQLite Database Schema, ER Diagram & 8 Advanced SQL Queries (src/database.py)",
            "Arystan Aldiyev (50% Contribution):",
            "   - Statistical Hypothesis Testing (Welch t-test, Levene test, ANOVA, Mann-Whitney U, Spearman rho) & Visualizations (src/eda_analytics.py)",
            "   - Leakage-Free NLP Star-Rating Prediction Pipeline (TF-IDF + NB / SGD / Logistic Regression) & Error Analysis (src/ml_models.py)",
            "   - Interactive Console CLI (main.py) & 5-Tab Streamlit Analytics Dashboard (app.py)",
        ],
        top=1.4,
        font_size=15,
    )

    # Slide 3: Out-of-Core ETL, Data Quality Audit & Memory Optimization
    s3 = prs.slides.add_slide(blank_layout)
    add_header(s3, "2. Out-of-Core JSONL → Parquet ETL & Data Quality Audit", "Streaming 747.30 MB Raw JSONL in Bounded Chunks with Explicit Cleaning Rules")
    add_bullets(
        s3,
        [
            "Real Dataset Volume: 750,000 reviews (547.06 MB), 150,346 businesses (113.36 MB), 444,680 users (86.88 MB).",
            "Data Quality Audit Findings on Real Yelp Data:",
            "   - 1,273 exact duplicate review texts (copy-pasted user spam across chains) -> Dropped to prevent train/test leakage.",
            "   - 13 ultra-short/empty reviews (<10 chars, e.g., '.', 'ok') -> Dropped due to zero lexical signal.",
            "   - 103 businesses with NULL categories -> Imputed as 'Uncategorized' to preserve relational integrity.",
            "   - City name normalization ('St. Louis' / 'St Louis' -> 'Saint Louis', whitespace stripping).",
            "   - 55 viral reviews with >100 useful votes (up to 539) -> Winsorized at 100 for robust aggregation.",
            "Memory & Disk Optimization:",
            "   - In-Memory Tracked Columns: 65.80 MB -> 37.31 MB (43.30% RAM reduction via category & int16/float32 downcasting).",
            "   - Disk Storage: 547.06 MB Raw JSONL -> 303.39 MB Snappy Parquet (1.80x compression + year-partitioned dataset).",
        ],
        top=1.35,
        font_size=14,
    )

    # Slide 4: Benchmark Comparison (Pandas vs Dask vs Polars)
    s4 = prs.slides.add_slide(blank_layout)
    add_header(s4, "3. Large-Scale Engine Benchmark: JSONL vs. Pandas vs. Dask vs. Polars", "Identical Aggregation & Join Task Across 748,714 Cleaned Reviews")
    add_bullets(
        s4,
        [
            "Baseline (JSONL Chunked Pandas): 68.559 s | Peak RAM: 405.53 MB | Global Mean Stars: 3.769781",
            "Approach 1 (Pandas Eager Parquet): 1.839 s | Peak RAM: 85.12 MB | Global Mean Stars: 3.769781 (37.3x faster)",
            "Approach 2 (Dask Lazy Parquet): 2.683 s | Peak RAM: 85.76 MB | Global Mean Stars: 3.769781 (25.6x faster)",
            "Approach 3 (Polars LazyFrame scan_parquet): 0.190 s | Peak RAM: 0.13 MB | Global Mean Stars: 3.769781 (361.8x faster!)",
            "Justification: Polars LazyFrame pushes predicates and column projections directly into the Rust Arrow reader.",
        ],
        left=0.6,
        top=1.35,
        width=12.0,
        height=2.2,
        font_size=14,
    )
    p3_img = PLOTS_DIR / "fig3_engine_benchmarks.png"
    if p3_img.exists():
        s4.shapes.add_picture(str(p3_img), Inches(1.2), Inches(3.6), width=Inches(10.8))

    # Slide 5: Relational Database Design & 8 Analytical SQL Queries
    s5 = prs.slides.add_slide(blank_layout)
    add_header(s5, "4. Normalized 5-Table SQLite Database & 8 SQL Queries", "1,743,065 Total Relational Rows Indexed in 10.54 Seconds")
    add_bullets(
        s5,
        [
            "Normalized 3NF Relational Schema (data/db/yelp_analytics.db):",
            "   1. businesses (150,346 rows, PK: business_id)   2. categories (60 top macro categories, PK: category_id)",
            "   3. business_categories (399,265 M:N links)      4. users (444,680 rows, PK: user_id)",
            "   5. reviews (748,714 cleaned rows, PK: review_id, FKs: user_id, business_id)",
            "8 Advanced SQL Queries Executed into Pandas DataFrames:",
            "   - Q1 (4-Table JOIN + GROUP BY + HAVING): Top city-category pairs with >=5,000 reviews (Philadelphia Restaurants leads with 112,265 reviews, 3.802 ★).",
            "   - Q2 (CTE + RANK() & AVG() OVER PARTITION BY city): Top-ranked businesses outperforming city average.",
            "   - Q3 (CTE + LAG() & Cumulative SUM() OVER): Year-over-Year city growth & rating drift (2005–2022).",
            "   - Q4 (Multi-Table JOIN + CASE): Elite (3.839 ★, 768 chars) vs. Regular (3.749 ★, 509 chars) rating polarization.",
            "   - Q5 (Correlated Subquery): Categories with extreme 1★/5★ polarization (Automotive 76.2%, Home Services 82.1%).",
            "   - Q6–Q8: Price tier survival (Open vs Closed), NTILE(4) user cohorts, and 3-month rolling window seasonality.",
        ],
        top=1.35,
        font_size=14,
    )

    # Slide 6: EDA & Hypothesis 1 (Elite vs Regular Rating Bias)
    s6 = prs.slides.add_slide(blank_layout)
    add_header(s6, "5. Hypothesis 1: Elite Reviewer Rating Bias & Polarization", "Welch's t-test (t = 25.56, p < 0.0001) & Levene's Variance Test (F = 20,193.1, p < 0.0001)")
    add_bullets(
        s6,
        [
            "Empirical Finding: Elite reviewers assign higher mean ratings (3.8392 ★ vs. 3.7491 ★) with much lower variance (SD 1.197 vs. 1.536).",
            "Regular users exhibit a stark J-shaped polarization (51.7% 5-star and 16.9% 1-star rants), whereas Elite users write balanced 4-star evaluations (36.8%).",
        ],
        left=0.6,
        top=1.3,
        width=12.0,
        height=1.2,
        font_size=14,
    )
    p1_img = PLOTS_DIR / "fig1_elite_vs_regular_ratings.png"
    if p1_img.exists():
        s6.shapes.add_picture(str(p1_img), Inches(2.2), Inches(2.55), width=Inches(8.8))

    # Slide 7: Hypothesis 2 (City & Category Heatmap)
    s7 = prs.slides.add_slide(blank_layout)
    add_header(s7, "6. Hypothesis 2: Geographic & Category Rating Heterogeneity", "One-Way ANOVA Across Cities (F = 179.54, p < 0.0001) & Categories (F = 447.25, p < 0.0001)")
    add_bullets(
        s7,
        [
            "Tourism & leisure cities (Saint Petersburg 4.01 ★, New Orleans 3.96 ★, Santa Barbara 3.94 ★) rate significantly higher than commuter hubs.",
            "Experiential categories (Active Life 4.14 ★, Arts & Entertainment 4.05 ★) outperform high-friction services (Automotive 3.59 ★, Home Services 3.63 ★).",
        ],
        left=0.6,
        top=1.3,
        width=12.0,
        height=1.2,
        font_size=14,
    )
    p2_img = PLOTS_DIR / "fig2_city_category_heatmap.png"
    if p2_img.exists():
        s7.shapes.add_picture(str(p2_img), Inches(2.0), Inches(2.5), width=Inches(9.2))

    # Slide 8: Temporal Trends & Hypothesis 3 (IQR Boxplots)
    s8 = prs.slides.add_slide(blank_layout)
    add_header(s8, "7. Temporal Trends & Hypothesis 3: Text Length vs. Useful Votes", "Spearman Correlation (rho = 0.3361, p < 0.0001) & Mann-Whitney U Test (p < 0.0001)")
    p4_img = PLOTS_DIR / "fig4_yearly_city_trends.png"
    p5_img = PLOTS_DIR / "fig5_iqr_boxplots.png"
    if p4_img.exists():
        s8.shapes.add_picture(str(p4_img), Inches(0.5), Inches(1.5), width=Inches(5.9))
    if p5_img.exists():
        s8.shapes.add_picture(str(p5_img), Inches(6.5), Inches(1.5), width=Inches(6.3))
    add_bullets(
        s8,
        [
            "Left: Review volume expanded steadily from 2010 to 2019 across all top 5 metropolitan areas before dipping in 2020 due to COVID-19 lockdowns.",
            "Right: Elite reviews are 50.8% longer (768 vs. 509 chars) and attract substantially more 'useful' votes across every star tier.",
        ],
        left=0.6,
        top=5.6,
        width=12.0,
        height=1.5,
        font_size=14,
    )

    # Slide 9: NLP Star-Rating Prediction & Confusion Matrix
    s9 = prs.slides.add_slide(blank_layout)
    add_header(s9, "8. Star-Rating Prediction from Review Text (NLP Modeling)", "Leakage-Free Chronological Split (Train: 2005–2019 [120K], Test: 2020–2022 [30K])")
    add_bullets(
        s9,
        [
            "Baseline (DummyClassifier Majority): Accuracy = 53.78% | Macro-F1 = 0.1399 | MAE = 1.2432 stars",
            "Model 1 (TF-IDF + MultinomialNB): Accuracy = 71.98% | Macro-F1 = 0.5043 | MAE = 0.4357 stars | Train: 13.08s",
            "Model 2 (TF-IDF + SGD Log-Loss): Accuracy = 74.65% | Macro-F1 = 0.5444 | MAE = 0.3658 stars | Train: 14.00s",
            "Model 3 (TF-IDF + LogisticRegression): Accuracy = 73.81% | Macro-F1 = 0.5582 (Best) | MAE = 0.3586 stars | Latency: 40.8 ms / 1K",
            "Error Analysis: 89% of errors occur between adjacent classes (e.g. 4★ vs 5★ or 1★ vs 2★) due to mixed sentiment and hedging.",
        ],
        left=0.6,
        top=1.35,
        width=6.6,
        height=5.5,
        font_size=14,
    )
    p6_img = PLOTS_DIR / "fig6_confusion_matrix.png"
    if p6_img.exists():
        s9.shapes.add_picture(str(p6_img), Inches(7.4), Inches(1.4), width=Inches(5.3))

    # Slide 10: Interactive Streamlit Dashboard & Conclusion
    s10 = prs.slides.add_slide(blank_layout)
    add_header(s10, "9. Interactive Streamlit Dashboard, CLI & Conclusions", "Production-Ready Reproducible Codebase")
    add_bullets(
        s10,
        [
            "Interactive Dual User Interfaces:",
            "   1. Console CLI Menu (`python main.py`): 8 interactive operations with `try...except` error handling + `--auto` batch mode.",
            "   2. Streamlit Web Dashboard (`streamlit run app.py`): Real-time Polars LazyFrame filtering by Year, City, Category, and Elite Tier, live SQL query runner, and live NLP review star predictor.",
            "Key Takeaways & Limitations:",
            "   - Migrating raw JSONL to partitioned Snappy Parquet + Polars LazyFrame cuts query time from 68.56s to 0.19s (361.8x speedup) and reduces peak RAM from 405.5 MB to 0.13 MB.",
            "   - Unigram/Bigram TF-IDF + Logistic Regression achieves 0.359 star MAE on future post-2020 reviews without GPU overhead, though sarcasm and historical narratives remain challenging without contextual embeddings.",
        ],
        top=1.4,
        font_size=16,
    )

    prs.save(str(pptx_path))
    return pptx_path


if __name__ == "__main__":
    nb = build_jupyter_notebook()
    ppt = build_powerpoint_presentation()
    print(f"Created Notebook: {nb}")
    print(f"Created Presentation: {ppt}")
