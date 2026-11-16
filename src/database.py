"""
Relational Database & SQL Analytics Module (Requirement 10).
Normalizes the cleaned Yelp dataset into a 5-table relational SQLite schema:
1. businesses (business_id PK, name, city, state, latitude, longitude, stars, is_open, price_range, primary_category)
2. categories (category_id PK, category_name UNIQUE)
3. business_categories (business_id FK, category_id FK, PRIMARY KEY (business_id, category_id))
4. users (user_id PK, name, review_count, yelping_since, useful, funny, cool, is_elite, fans, average_stars)
5. reviews (review_id PK, user_id FK, business_id FK, stars, useful, funny, cool, text_length, date, year, month)

Executes 8 advanced analytical SQL queries using JOINs, GROUP BY / HAVING,
Common Table Expressions (CTEs), subqueries, and Window Functions (RANK, DENSE_RANK, LAG, AVG OVER).

Authors: Uali Sembay & Arystan Aldiyev (AAI-2605)
"""

import sqlite3
import time
from pathlib import Path
from typing import Dict

import pandas as pd

from src.config import (
    PARQUET_BUSINESS,
    PARQUET_REVIEWS_SINGLE,
    PARQUET_USER,
    SQLITE_DB_PATH,
)

SCHEMA_DDL = """
PRAGMA foreign_keys = ON;

DROP TABLE IF EXISTS reviews;
DROP TABLE IF EXISTS business_categories;
DROP TABLE IF EXISTS categories;
DROP TABLE IF EXISTS users;
DROP TABLE IF EXISTS businesses;

CREATE TABLE businesses (
    business_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    city TEXT NOT NULL,
    state TEXT NOT NULL,
    latitude REAL,
    longitude REAL,
    stars REAL NOT NULL,
    is_open INTEGER NOT NULL,
    price_range INTEGER,
    primary_category TEXT
);

CREATE TABLE categories (
    category_id INTEGER PRIMARY KEY AUTOINCREMENT,
    category_name TEXT UNIQUE NOT NULL
);

CREATE TABLE business_categories (
    business_id TEXT NOT NULL,
    category_id INTEGER NOT NULL,
    PRIMARY KEY (business_id, category_id),
    FOREIGN KEY (business_id) REFERENCES businesses(business_id),
    FOREIGN KEY (category_id) REFERENCES categories(category_id)
);

CREATE TABLE users (
    user_id TEXT PRIMARY KEY,
    name TEXT,
    review_count INTEGER,
    yelping_since TEXT,
    useful INTEGER,
    funny INTEGER,
    cool INTEGER,
    is_elite INTEGER NOT NULL,
    fans INTEGER,
    average_stars REAL
);

CREATE TABLE reviews (
    review_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    business_id TEXT NOT NULL,
    stars REAL NOT NULL,
    useful INTEGER NOT NULL,
    funny INTEGER NOT NULL,
    cool INTEGER NOT NULL,
    text_length INTEGER NOT NULL,
    date TEXT NOT NULL,
    year INTEGER NOT NULL,
    month INTEGER NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(user_id),
    FOREIGN KEY (business_id) REFERENCES businesses(business_id)
);
"""

INDEX_DDL = """
CREATE INDEX IF NOT EXISTS idx_biz_city ON businesses(city);
CREATE INDEX IF NOT EXISTS idx_biz_cat ON businesses(primary_category);
CREATE INDEX IF NOT EXISTS idx_rev_biz ON reviews(business_id);
CREATE INDEX IF NOT EXISTS idx_rev_usr ON reviews(user_id);
CREATE INDEX IF NOT EXISTS idx_rev_year ON reviews(year);
CREATE INDEX IF NOT EXISTS idx_usr_elite ON users(is_elite);
"""


def build_sqlite_database(db_path: Path = SQLITE_DB_PATH) -> Dict[str, int]:
    """
    Populate the normalized SQLite database from the cleaned Parquet files.
    Returns row counts for each relational table.
    """
    t0 = time.perf_counter()
    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("PRAGMA temp_store = MEMORY;")
        conn.execute("PRAGMA cache_size = -64000;")
        conn.executescript(SCHEMA_DDL)

        # 1. Load businesses
        df_biz = pd.read_parquet(PARQUET_BUSINESS)
        biz_cols = [
            "business_id", "name", "city", "state", "latitude",
            "longitude", "stars", "is_open", "price_range", "primary_category"
        ]
        df_biz[biz_cols].to_sql("businesses", conn, if_exists="append", index=False, chunksize=10000)

        # 2. Normalize M:N categories and business_categories (top 50 most frequent categories + primary macro categories)
        from collections import Counter
        cat_counter: Counter = Counter()
        for cats_raw in df_biz["categories"].fillna("Uncategorized"):
            for c in str(cats_raw).split(","):
                c_clean = c.strip()
                if c_clean:
                    cat_counter[c_clean] += 1

        top_cats = set([c for c, _ in cat_counter.most_common(60)])
        biz_cat_pairs = []
        for row in df_biz[["business_id", "categories"]].itertuples(index=False):
            b_id = row.business_id
            cats_raw = str(row.categories) if row.categories else "Uncategorized"
            for c in cats_raw.split(","):
                c_clean = c.strip()
                if c_clean in top_cats:
                    biz_cat_pairs.append((b_id, c_clean))

        sorted_cats = sorted(top_cats)
        cat_to_id = {name: idx + 1 for idx, name in enumerate(sorted_cats)}

        df_categories = pd.DataFrame(
            [{"category_id": idx, "category_name": name} for name, idx in cat_to_id.items()]
        )
        df_categories.to_sql("categories", conn, if_exists="append", index=False)

        df_biz_cats = pd.DataFrame(
            [{"business_id": b_id, "category_id": cat_to_id[c_name]} for b_id, c_name in biz_cat_pairs]
        ).drop_duplicates()
        df_biz_cats.to_sql("business_categories", conn, if_exists="append", index=False, chunksize=25000)

        # 3. Load users
        df_usr = pd.read_parquet(PARQUET_USER)
        usr_cols = [
            "user_id", "name", "review_count", "yelping_since",
            "useful", "funny", "cool", "is_elite", "fans", "average_stars"
        ]
        df_usr[usr_cols].to_sql("users", conn, if_exists="append", index=False, chunksize=15000)

        # 4. Load reviews (excluding raw text column in relational fact table to keep DB fast, while storing text_length)
        df_rev = pd.read_parquet(
            PARQUET_REVIEWS_SINGLE,
            columns=[
                "review_id", "user_id", "business_id", "stars",
                "useful", "funny", "cool", "text_length", "date", "year", "month"
            ],
        )
        # Filter to valid foreign keys (businesses that passed city cleaning)
        valid_biz_ids = set(df_biz["business_id"].unique())
        df_rev = df_rev[df_rev["business_id"].isin(valid_biz_ids)]
        df_rev.to_sql("reviews", conn, if_exists="append", index=False, chunksize=50000)

        conn.executescript(INDEX_DDL)
        conn.commit()

        counts = {}
        for tbl in ["businesses", "categories", "business_categories", "users", "reviews"]:
            cur = conn.execute(f"SELECT COUNT(*) FROM {tbl}")
            counts[tbl] = int(cur.fetchone()[0])
        counts["build_seconds"] = round(time.perf_counter() - t0, 2)
        return counts
    finally:
        conn.close()


# 8 Analytical SQL Queries covering JOIN, GROUP BY/HAVING, CTEs, Subqueries, and Window Functions
SQL_QUERIES: Dict[str, str] = {
    "Q1_City_Category_Volume_HAVING": """
        -- Q1: JOIN + GROUP BY + HAVING: City & Category combinations with high review volume
        SELECT
            b.city,
            b.state,
            c.category_name,
            COUNT(r.review_id) AS total_reviews,
             COUNT(DISTINCT b.business_id) AS unique_businesses,
            ROUND(AVG(r.stars), 3) AS avg_review_stars,
            ROUND(AVG(r.useful), 2) AS avg_useful_votes
        FROM reviews r
        JOIN businesses b ON r.business_id = b.business_id
        JOIN business_categories bc ON b.business_id = bc.business_id
        JOIN categories c ON bc.category_id = c.category_id
        GROUP BY b.city, b.state, c.category_name
        HAVING COUNT(r.review_id) >= 5000
        ORDER BY total_reviews DESC
        LIMIT 15;
    """,
    "Q2_Window_Rank_Top_Businesses_Per_City": """
        -- Q2: CTE + Window Functions (RANK, DENSE_RANK, AVG OVER PARTITION): Top businesses per major city
        WITH biz_stats AS (
            SELECT
                b.business_id,
                b.name,
                b.city,
                b.primary_category,
                COUNT(r.review_id) AS rev_count,
                AVG(r.stars) AS mean_stars
            FROM businesses b
            JOIN reviews r ON b.business_id = r.business_id
            WHERE b.city IN (
                'Philadelphia', 'New Orleans', 'Nashville', 'Tampa', 'Tucson',
                'Indianapolis', 'Reno', 'Santa Barbara', 'Saint Louis', 'Boise'
            )
            GROUP BY b.business_id, b.name, b.city, b.primary_category
            HAVING COUNT(r.review_id) >= 50
        ),
        ranked_biz AS (
            SELECT
                city,
                name,
                primary_category,
                rev_count,
                ROUND(mean_stars, 3) AS biz_stars,
                ROUND(AVG(mean_stars) OVER (PARTITION BY city), 3) AS city_avg_stars,
                ROUND(mean_stars - AVG(mean_stars) OVER (PARTITION BY city), 3) AS stars_above_city_avg,
                RANK() OVER (PARTITION BY city ORDER BY mean_stars DESC, rev_count DESC) AS city_rank
            FROM biz_stats
        )
        SELECT *
        FROM ranked_biz
        WHERE city_rank <= 2
        ORDER BY city, city_rank;
    """,
    "Q3_Window_LAG_YoY_City_Growth": """
        -- Q3: CTE + Window Function (LAG, SUM OVER): Year-over-Year Review Activity & Rating Drift by City
        WITH yearly_city AS (
            SELECT
                b.city,
                r.year,
                COUNT(r.review_id) AS yearly_reviews,
                AVG(r.stars) AS yearly_stars
            FROM reviews r
            JOIN businesses b ON r.business_id = b.business_id
            WHERE b.city IN ('Philadelphia', 'New Orleans', 'Nashville', 'Tampa', 'Tucson')
            GROUP BY b.city, r.year
        )
        SELECT
            city,
            year,
            yearly_reviews,
            LAG(yearly_reviews, 1) OVER (PARTITION BY city ORDER BY year) AS prev_year_reviews,
            ROUND(
                100.0 * (yearly_reviews - LAG(yearly_reviews, 1) OVER (PARTITION BY city ORDER BY year))
                / NULLIF(LAG(yearly_reviews, 1) OVER (PARTITION BY city ORDER BY year), 0),
                2
            ) AS yoy_growth_pct,
            ROUND(yearly_stars, 3) AS avg_stars,
            SUM(yearly_reviews) OVER (
                PARTITION BY city ORDER BY year ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) AS cumulative_reviews
        FROM yearly_city
        ORDER BY city, year;
    """,
    "Q4_Elite_vs_Regular_User_Behavior": """
        -- Q4: Multi-table JOIN + CASE Aggregation: Elite vs Regular User Rating Bias & Text Length
        SELECT
            CASE WHEN u.is_elite = 1 THEN 'Elite Reviewers' ELSE 'Regular Users' END AS user_tier,
            COUNT(DISTINCT u.user_id) AS active_users,
            COUNT(r.review_id) AS total_reviews,
            ROUND(AVG(r.stars), 4) AS mean_stars,
            ROUND(AVG(r.text_length), 1) AS mean_text_length_chars,
            ROUND(AVG(r.useful), 3) AS mean_useful_votes,
            ROUND(100.0 * SUM(CASE WHEN r.stars = 1.0 THEN 1 ELSE 0 END) / COUNT(r.review_id), 2) AS pct_1_star,
            ROUND(100.0 * SUM(CASE WHEN r.stars = 5.0 THEN 1 ELSE 0 END) / COUNT(r.review_id), 2) AS pct_5_star
        FROM reviews r
        JOIN users u ON r.user_id = u.user_id
        GROUP BY u.is_elite
        ORDER BY u.is_elite DESC;
    """,
    "Q5_Subquery_Category_Polarization": """
        -- Q5: Correlated/Scalar Subquery + Polarization Ratio: Categories exceeding global average useful engagement
        SELECT
            c.category_name,
            COUNT(r.review_id) AS review_volume,
            ROUND(AVG(r.stars), 3) AS category_mean_stars,
            ROUND(
                AVG(r.stars) - (SELECT AVG(stars) FROM reviews),
                3
            ) AS diff_from_global_mean,
            ROUND(
                100.0 * SUM(CASE WHEN r.stars IN (1.0, 5.0) THEN 1 ELSE 0 END) / COUNT(r.review_id),
                2
            ) AS extreme_rating_polarization_pct,
            ROUND(AVG(r.useful), 3) AS avg_useful
        FROM reviews r
        JOIN business_categories bc ON r.business_id = bc.business_id
        JOIN categories c ON bc.category_id = c.category_id
        GROUP BY c.category_name
        HAVING AVG(r.useful) >= (SELECT AVG(useful) * 0.90 FROM reviews)
           AND COUNT(r.review_id) >= 8000
        ORDER BY extreme_rating_polarization_pct DESC
        LIMIT 12;
    """,
    "Q6_Price_Range_Open_Status_Survival": """
        -- Q6: Price Tier & Operating Status (is_open) Impact on Ratings Across Cities
        SELECT
            b.price_range,
            CASE WHEN b.is_open = 1 THEN 'Open' ELSE 'Closed' END AS operating_status,
            COUNT(DISTINCT b.business_id) AS business_count,
            COUNT(r.review_id) AS total_reviews,
            ROUND(AVG(r.stars), 3) AS avg_review_stars,
            ROUND(AVG(b.stars), 3) AS avg_business_stars,
            ROUND(AVG(r.text_length), 1) AS avg_review_length
        FROM businesses b
        JOIN reviews r ON b.business_id = r.business_id
        GROUP BY b.price_range, b.is_open
        ORDER BY b.price_range ASC, b.is_open DESC;
    """,
    "Q7_Window_NTILE_User_Tenure_Cohorts": """
        -- Q7: CTE + Window Function (NTILE & DENSE_RANK): User Join-Year Cohort Activity Analysis
        WITH user_cohorts AS (
            SELECT
                CAST(SUBSTR(u.yelping_since, 1, 4) AS INTEGER) AS cohort_year,
                u.is_elite,
                COUNT(r.review_id) AS reviews_written,
                AVG(r.stars) AS cohort_avg_stars,
                SUM(r.useful) AS total_useful_earned
            FROM users u
            JOIN reviews r ON u.user_id = r.user_id
            GROUP BY CAST(SUBSTR(u.yelping_since, 1, 4) AS INTEGER), u.is_elite
        )
        SELECT
            cohort_year,
            CASE WHEN is_elite = 1 THEN 'Elite' ELSE 'Regular' END AS tier,
            reviews_written,
            ROUND(cohort_avg_stars, 3) AS cohort_avg_stars,
            total_useful_earned,
            DENSE_RANK() OVER (PARTITION BY is_elite ORDER BY total_useful_earned DESC) AS impact_rank,
            NTILE(4) OVER (ORDER BY reviews_written DESC) AS activity_quartile
        FROM user_cohorts
        ORDER BY cohort_year DESC, is_elite DESC;
    """,
    "Q8_Monthly_Seasonality_Moving_Avg": """
        -- Q8: CTE + Window Moving Average (ROWS BETWEEN 2 PRECEDING AND CURRENT ROW): Monthly Seasonality
        WITH monthly_agg AS (
            SELECT
                year,
                month,
                COUNT(review_id) AS monthly_reviews,
                AVG(stars) AS monthly_mean_stars
            FROM reviews
            WHERE year BETWEEN 2018 AND 2021
            GROUP BY year, month
        )
        SELECT
            year,
            month,
            monthly_reviews,
            ROUND(
                AVG(monthly_reviews) OVER (
                    ORDER BY year, month ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
                ),
                1
            ) AS rolling_3m_reviews,
            ROUND(monthly_mean_stars, 3) AS monthly_mean_stars,
            ROUND(
                AVG(monthly_mean_stars) OVER (
                    ORDER BY year, month ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
                ),
                3
            ) AS rolling_3m_stars
        FROM monthly_agg
        ORDER BY year, month;
    """,
}


def execute_all_sql_queries(db_path: Path = SQLITE_DB_PATH) -> Dict[str, pd.DataFrame]:
    """
    Execute all 8 analytical SQL queries and load results into Pandas DataFrames.
    """
    if not db_path.exists():
        raise FileNotFoundError(f"SQLite database not found at {db_path}. Run build_sqlite_database() first.")

    results: Dict[str, pd.DataFrame] = {}
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("PRAGMA temp_store = MEMORY;")
        conn.execute("PRAGMA cache_size = -64000;")
        for q_name, q_sql in SQL_QUERIES.items():
            df_res = pd.read_sql_query(q_sql, conn)
            results[q_name] = df_res
    finally:
        conn.close()

    return results
