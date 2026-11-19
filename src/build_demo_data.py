"""
Builds the small demo dataset in data/demo/ that ships with the repository, so the
Streamlit dashboard runs straight after `pip install` (locally or on Streamlit Community
Cloud) without the 4.3 GB download.

It takes a reproducible random sample of the cleaned reviews produced by the full pipeline,
keeps the businesses and users those reviews refer to, builds a matching SQLite database
and copies the trained model. Run it after `python main.py --auto`:

    python -m src.build_demo_data            # 50,000 reviews by default

Authors: Uali Sembay & Arystan Aldiyev (AAI-2605)
"""

import json
import shutil
import sys
from datetime import datetime

import pandas as pd

import src.database as database
from src.config import (
    BEST_MODEL_PATH,
    DEMO_BUSINESS,
    DEMO_DB_PATH,
    DEMO_DIR,
    DEMO_INFO_JSON,
    DEMO_MODEL_PATH,
    DEMO_REVIEWS,
    DEMO_USER,
    PARQUET_BUSINESS,
    PARQUET_REVIEWS_SINGLE,
    PARQUET_USER,
)


def build_demo_database() -> dict:
    """
    Build data/demo/yelp_demo.db from the demo Parquet files, with the same schema and loader
    as the full database. The .db file is not committed (it is rebuilt in a few seconds);
    the dashboard calls this on first start when the file is missing.
    """
    saved = (database.PARQUET_BUSINESS, database.PARQUET_USER, database.PARQUET_REVIEWS_SINGLE)
    database.PARQUET_BUSINESS, database.PARQUET_USER, database.PARQUET_REVIEWS_SINGLE = DEMO_BUSINESS, DEMO_USER, DEMO_REVIEWS
    try:
        return database.build_sqlite_database(DEMO_DB_PATH, write_summary=False)
    finally:
        database.PARQUET_BUSINESS, database.PARQUET_USER, database.PARQUET_REVIEWS_SINGLE = saved


def build_demo_data(n_reviews: int = 50_000, seed: int = 42) -> dict:
    """Sample the cleaned data into data/demo/ and return a short summary."""
    for p in (PARQUET_REVIEWS_SINGLE, PARQUET_BUSINESS, PARQUET_USER, BEST_MODEL_PATH):
        if not p.exists():
            raise FileNotFoundError(f"{p} is missing. Run `python main.py --auto` first.")
    DEMO_DIR.mkdir(parents=True, exist_ok=True)

    reviews = pd.read_parquet(PARQUET_REVIEWS_SINGLE)
    total = len(reviews)
    reviews = reviews.sample(n=min(n_reviews, total), random_state=seed).sort_values("date").reset_index(drop=True)
    businesses = pd.read_parquet(PARQUET_BUSINESS)
    businesses = businesses[businesses["business_id"].isin(set(reviews["business_id"]))]
    users = pd.read_parquet(PARQUET_USER)
    users = users[users["user_id"].isin(set(reviews["user_id"]))]

    reviews.to_parquet(DEMO_REVIEWS, index=False, compression="zstd")
    businesses.to_parquet(DEMO_BUSINESS, index=False, compression="zstd")
    users.to_parquet(DEMO_USER, index=False, compression="zstd")
    shutil.copyfile(BEST_MODEL_PATH, DEMO_MODEL_PATH)

    db_summary = build_demo_database()

    info = {
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "sample": f"random sample of {len(reviews):,} of {total:,} cleaned reviews (seed {seed})",
        "reviews": len(reviews),
        "businesses": len(businesses),
        "users": len(users),
        "full_reviews": total,
        "db_row_counts": db_summary["row_counts"],
        "sizes_mb": {p.name: round(p.stat().st_size / 2**20, 1) for p in sorted(DEMO_DIR.iterdir()) if p.is_file() and p.suffix != ".db"},
    }
    DEMO_INFO_JSON.write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50_000
    print(json.dumps(build_demo_data(n), indent=2))
