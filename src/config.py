"""
Configuration and directory paths for Yelp Reviews at Scale Project (Topic #44).
Authors: Uali Sembay & Arystan Aldiyev (AAI-2605)
"""

import os
from pathlib import Path

# Base project directory
BASE_DIR: Path = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR: Path = BASE_DIR / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PARQUET_DIR: Path = DATA_DIR / "parquet"
DB_DIR: Path = DATA_DIR / "db"
MODELS_DIR: Path = BASE_DIR / "models"
OUTPUTS_DIR: Path = BASE_DIR / "outputs"
PLOTS_DIR: Path = OUTPUTS_DIR / "plots"
REPORTS_DIR: Path = BASE_DIR / "docs"

# File paths
RAW_BUSINESS_JSON: Path = RAW_DIR / "yelp_academic_dataset_business.json"
RAW_USER_JSON: Path = RAW_DIR / "yelp_academic_dataset_user.json"
RAW_REVIEW_JSON: Path = RAW_DIR / "yelp_academic_dataset_review.json"

PARQUET_BUSINESS: Path = PARQUET_DIR / "business.parquet"
PARQUET_USER: Path = PARQUET_DIR / "user.parquet"
PARQUET_REVIEWS_DIR: Path = PARQUET_DIR / "reviews_partitioned"
PARQUET_REVIEWS_SINGLE: Path = PARQUET_DIR / "reviews_cleaned.parquet"

SQLITE_DB_PATH: Path = DB_DIR / "yelp_analytics.db"
BENCHMARK_RESULTS_JSON: Path = OUTPUTS_DIR / "benchmark_results.json"
AUDIT_REPORT_JSON: Path = OUTPUTS_DIR / "data_quality_audit.json"
MODEL_METRICS_JSON: Path = OUTPUTS_DIR / "ml_model_metrics.json"
HYPOTHESIS_RESULTS_JSON: Path = OUTPUTS_DIR / "hypothesis_tests.json"

# Ensure all directories exist
ALL_DIRS = [
    DATA_DIR,
    RAW_DIR,
    PARQUET_DIR,
    DB_DIR,
    MODELS_DIR,
    OUTPUTS_DIR,
    PLOTS_DIR,
    REPORTS_DIR,
]

def ensure_directories() -> dict:
    """Create all project directories if they do not exist and return their status."""
    status = {}
    for d in ALL_DIRS:
        existed = d.exists()
        d.mkdir(parents=True, exist_ok=True)
        status[d.name] = "existed" if existed else "created"
    return status

# Constants and metadata
PROJECT_METADATA = {
    "topic_number": 44,
    "cluster": "Cluster 8: Large-Scale Data Processing (Pandas, Polars, Dask, Parquet)",
    "title": "Yelp Reviews at Scale: Ratings, Categories and Cities",
    "authors": ["Uali Sembay", "Arystan Aldiyev"],
    "group": "AAI-2605",
    "course": "Programming for Data Analysis and Database",
    "program": "Applied Artificial Intelligence (MSc)",
}

CITIES_METADATA = [
    {"city": "Philadelphia", "state": "PA", "lat": 39.9526, "lon": -75.1652, "base_rating": 3.82, "weight": 0.18},
    {"city": "Tucson", "state": "AZ", "lat": 32.2226, "lon": -110.9747, "base_rating": 3.76, "weight": 0.12},
    {"city": "Tampa", "state": "FL", "lat": 27.9506, "lon": -82.4572, "base_rating": 3.85, "weight": 0.13},
    {"city": "Indianapolis", "state": "IN", "lat": 39.7684, "lon": -86.1581, "base_rating": 3.79, "weight": 0.11},
    {"city": "Nashville", "state": "TN", "lat": 36.1627, "lon": -86.7816, "base_rating": 3.91, "weight": 0.12},
    {"city": "New Orleans", "state": "LA", "lat": 29.9511, "lon": -90.0715, "base_rating": 4.02, "weight": 0.11},
    {"city": "Reno", "state": "NV", "lat": 39.5296, "lon": -119.8138, "base_rating": 3.74, "weight": 0.08},
    {"city": "Edmonton", "state": "AB", "lat": 53.5461, "lon": -113.4938, "base_rating": 3.68, "weight": 0.05},
    {"city": "Saint Louis", "state": "MO", "lat": 38.6270, "lon": -90.1994, "base_rating": 3.80, "weight": 0.05},
    {"city": "Santa Barbara", "state": "CA", "lat": 34.4208, "lon": -119.6982, "base_rating": 4.08, "weight": 0.05},
]

CATEGORY_BUNDLES = [
    "Restaurants, Nightlife, Bars, American (New)",
    "Restaurants, Mexican, Tacos, Fast Food",
    "Coffee & Tea, Food, Cafes, Bakeries",
    "Restaurants, Italian, Pizza, Wine Bars",
    "Beauty & Spas, Hair Salons, Nail Salons, Skin Care",
    "Auto Repair, Automotive, Oil Change Stations, Tires",
    "Home Services, Plumbing, Heating & Air Conditioning/HVAC",
    "Health & Medical, Dentists, General Dentistry, Doctors",
    "Shopping, Fashion, Department Stores, Accessories",
    "Hotels & Travel, Event Planning & Services, Hotels",
    "Active Life, Fitness & Instruction, Gyms, Trainers",
    "Restaurants, Japanese, Sushi Bars, Asian Fusion",
    "Restaurants, Burgers, Fast Food, American (Traditional)",
    "Local Services, Dry Cleaning & Laundry, Sewing & Alterations",
    "Pets, Veterinarians, Pet Services, Pet Groomers",
]
