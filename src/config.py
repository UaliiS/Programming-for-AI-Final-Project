"""
Configuration: directory layout, file paths and project-wide constants
for "Yelp Reviews at Scale" (Topic #44, Cluster 8).

Authors: Uali Sembay & Arystan Aldiyev (AAI-2605)
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------
BASE_DIR: Path = Path(__file__).resolve().parent.parent

DATA_DIR: Path = BASE_DIR / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PARQUET_DIR: Path = DATA_DIR / "parquet"
DB_DIR: Path = DATA_DIR / "db"
MODELS_DIR: Path = BASE_DIR / "models"
OUTPUTS_DIR: Path = BASE_DIR / "outputs"
PLOTS_DIR: Path = OUTPUTS_DIR / "plots"
SQL_RESULTS_DIR: Path = OUTPUTS_DIR / "sql_results"
AGGREGATES_DIR: Path = OUTPUTS_DIR / "aggregates"
REPORTS_DIR: Path = BASE_DIR / "docs"

# ---------------------------------------------------------------------------
# Raw files (official Yelp Open Dataset, JSON Lines)
# ---------------------------------------------------------------------------
RAW_BUSINESS_JSON: Path = RAW_DIR / "yelp_academic_dataset_business.json"
RAW_REVIEW_JSON: Path = RAW_DIR / "yelp_academic_dataset_review.json"
RAW_USER_JSON: Path = RAW_DIR / "yelp_academic_dataset_user.json"
RAW_MANIFEST_JSON: Path = RAW_DIR / "_extraction_manifest.json"

# ---------------------------------------------------------------------------
# Processed files
# ---------------------------------------------------------------------------
PARQUET_BUSINESS: Path = PARQUET_DIR / "business.parquet"
PARQUET_USER: Path = PARQUET_DIR / "user.parquet"
PARQUET_REVIEWS_DIR: Path = PARQUET_DIR / "reviews_partitioned"   # Hive-style, partitioned by year
PARQUET_REVIEWS_SINGLE: Path = PARQUET_DIR / "reviews_cleaned.parquet"

SQLITE_DB_PATH: Path = DB_DIR / "yelp_analytics.db"

AUDIT_REPORT_JSON: Path = OUTPUTS_DIR / "data_quality_audit.json"
BENCHMARK_RESULTS_JSON: Path = OUTPUTS_DIR / "benchmark_results.json"
HYPOTHESIS_RESULTS_JSON: Path = OUTPUTS_DIR / "hypothesis_tests.json"
MODEL_METRICS_JSON: Path = OUTPUTS_DIR / "ml_model_metrics.json"
EDA_SUMMARY_JSON: Path = OUTPUTS_DIR / "eda_summary.json"
DB_SUMMARY_JSON: Path = OUTPUTS_DIR / "database_summary.json"
ER_DIAGRAM_PNG: Path = PLOTS_DIR / "er_diagram.png"
BEST_MODEL_PATH: Path = MODELS_DIR / "yelp_star_predictor.joblib"

# ---------------------------------------------------------------------------
# Extraction settings
# ---------------------------------------------------------------------------
YELP_ZIP_URL = "https://business.yelp.com/external-assets/files/Yelp-JSON.zip"
# The full review file has ~7M lines and is grouped by business, so taking the
# first N lines gives a biased sample (only ~15% of businesses). Instead we keep
# every k-th review while streaming the whole file (systematic sampling).
REVIEW_SAMPLE_STEP: int = 9          # ~6.99M / 9 ≈ 777K reviews
MAX_REVIEWS: int = 800_000           # hard cap on kept reviews

ALL_DIRS = [
    DATA_DIR, RAW_DIR, PARQUET_DIR, DB_DIR, MODELS_DIR,
    OUTPUTS_DIR, PLOTS_DIR, SQL_RESULTS_DIR, AGGREGATES_DIR, REPORTS_DIR,
]


def ensure_directories() -> dict:
    """Create all project directories if they do not exist and return their status."""
    status = {}
    for d in ALL_DIRS:
        existed = d.exists()
        d.mkdir(parents=True, exist_ok=True)
        status[d.name] = "existed" if existed else "created"
    return status


PROJECT_METADATA = {
    "topic_number": 44,
    "cluster": "Cluster 8: Large-Scale Data Processing (Pandas, Polars, Dask, Parquet)",
    "title": "Yelp Reviews at Scale: Ratings, Categories and Cities",
    "authors": ["Uali Sembay", "Arystan Aldiyev"],
    "group": "AAI-2605",
    "course": "Programming for Data Analysis and Database",
    "program": "Applied Artificial Intelligence (MSc)",
}
