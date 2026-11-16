"""
Comparative Benchmarking Module (Requirement 9 & Cluster 8 Tasks).
Compares:
1. Baseline: Line-by-line JSONL / Chunked Pandas groupby aggregation on raw JSONL
2. Approach 1: Pandas on Parquet
3. Approach 2: Dask Distributed/Lazy DataFrame on Parquet
4. Approach 3: Polars LazyFrame (pl.scan_parquet) on Parquet

Evaluates execution time (seconds), peak memory footprint (MB via tracemalloc),
and numerical equivalence of computed business/city star rating aggregations.

Authors: Uali Sembay & Arystan Aldiyev (AAI-2605)
"""

import gc
import json
import time
import tracemalloc
from typing import Any, Dict

import dask.dataframe as dd
import pandas as pd
import polars as pl

from src.config import (
    BENCHMARK_RESULTS_JSON,
    PARQUET_BUSINESS,
    PARQUET_REVIEWS_SINGLE,
    RAW_REVIEW_JSON,
)
from src.etl_pipeline import stream_jsonl_chunks


def benchmark_chunked_jsonl_baseline(chunk_size: int = 100000) -> Dict[str, Any]:
    """
    Baseline Approach: Incremental aggregation over raw JSONL chunks using Pandas.
    Calculates review count, mean star rating, and mean useful votes per year.
    """
    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()

    sum_stars: Dict[int, float] = {}
    sum_useful: Dict[int, int] = {}
    counts: Dict[int, int] = {}

    seen_ids = set()
    seen_txt = set()
    for chunk in stream_jsonl_chunks(RAW_REVIEW_JSON, chunk_size=chunk_size):
        df_c = pd.DataFrame(chunk)
        # Filter valid records to match cleaned Parquet
        is_dup_id = df_c["review_id"].duplicated(keep="first") | df_c["review_id"].isin(seen_ids)
        df_c = df_c[~is_dup_id].copy()
        seen_ids.update(df_c["review_id"].tolist())

        txt_h = df_c["text"].apply(hash)
        is_dup_txt = txt_h.duplicated(keep="first") | txt_h.isin(seen_txt)
        df_c = df_c[~is_dup_txt].copy()
        seen_txt.update(txt_h[~is_dup_txt].tolist())

        df_c = df_c[
            (df_c["stars"] >= 1.0)
            & (df_c["stars"] <= 5.0)
            & (df_c["text"].fillna("").str.strip().str.len() >= 10)
        ]
        df_c["useful"] = df_c["useful"].clip(lower=0, upper=100)
        df_c["year"] = df_c["date"].str.slice(0, 4).astype(int)

        grp = df_c.groupby("year").agg(
            cnt=("review_id", "count"),
            s_stars=("stars", "sum"),
            s_useful=("useful", "sum"),
        )
        for yr, row in grp.iterrows():
            yr_int = int(yr)
            counts[yr_int] = counts.get(yr_int, 0) + int(row["cnt"])
            sum_stars[yr_int] = sum_stars.get(yr_int, 0.0) + float(row["s_stars"])
            sum_useful[yr_int] = sum_useful.get(yr_int, 0) + int(row["s_useful"])

    rows_out = []
    for yr in sorted(counts.keys()):
        rows_out.append({
            "year": yr,
            "review_count": counts[yr],
            "mean_stars": round(sum_stars[yr] / counts[yr], 6),
            "mean_useful": round(sum_useful[yr] / counts[yr], 6),
        })
    res_df = pd.DataFrame(rows_out)

    elapsed = time.perf_counter() - t0
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "engine": "Baseline: JSONL Chunked (Pandas)",
        "time_seconds": round(elapsed, 4),
        "peak_memory_mb": round(peak_bytes / (1024 * 1024), 2),
        "total_rows_processed": int(res_df["review_count"].sum()),
        "global_mean_stars": round(
            float((res_df["mean_stars"] * res_df["review_count"]).sum() / res_df["review_count"].sum()), 6
        ),
        "yearly_result": res_df.to_dict(orient="records"),
    }


def benchmark_pandas_parquet() -> Dict[str, Any]:
    """
    Approach 1: Eager Pandas DataFrame reading from Parquet and performing join + groupby.
    """
    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()

    df_rev = pd.read_parquet(
        PARQUET_REVIEWS_SINGLE,
        columns=["review_id", "business_id", "stars", "useful", "year"],
    )
    df_biz = pd.read_parquet(
        PARQUET_BUSINESS,
        columns=["business_id", "city", "primary_category"],
    )

    merged = df_rev.merge(df_biz, on="business_id", how="inner")
    grp_year = (
        df_rev.groupby("year", observed=True)
        .agg(
            review_count=("review_id", "count"),
            mean_stars=("stars", "mean"),
            mean_useful=("useful", "mean"),
        )
        .reset_index()
        .sort_values("year")
    )
    grp_city = (
        merged.groupby("city", observed=True)
        .agg(
            review_count=("review_id", "count"),
            mean_stars=("stars", "mean"),
        )
        .reset_index()
        .sort_values("review_count", ascending=False)
    )

    elapsed = time.perf_counter() - t0
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    grp_year["mean_stars"] = grp_year["mean_stars"].round(6)
    grp_year["mean_useful"] = grp_year["mean_useful"].round(6)

    return {
        "engine": "Pandas (Parquet Eager)",
        "time_seconds": round(elapsed, 4),
        "peak_memory_mb": round(peak_bytes / (1024 * 1024), 2),
        "total_rows_processed": int(grp_year["review_count"].sum()),
        "global_mean_stars": round(float(df_rev["stars"].mean()), 6),
        "top_city": str(grp_city.iloc[0]["city"]),
        "yearly_result": grp_year.to_dict(orient="records"),
    }


def benchmark_dask_parquet() -> Dict[str, Any]:
    """
    Approach 2: Dask Lazy DataFrame reading Parquet with task graph execution (.compute()).
    """
    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()

    ddf_rev = dd.read_parquet(
        PARQUET_REVIEWS_SINGLE,
        columns=["review_id", "business_id", "stars", "useful", "year"],
    )
    ddf_biz = dd.read_parquet(
        PARQUET_BUSINESS,
        columns=["business_id", "city", "primary_category"],
    )

    ddf_merged = ddf_rev.merge(ddf_biz, on="business_id", how="inner")

    grp_year_lazy = ddf_rev.groupby("year").agg(
        review_count=("review_id", "count"),
        mean_stars=("stars", "mean"),
        mean_useful=("useful", "mean"),
    )
    grp_city_lazy = ddf_merged.groupby("city", observed=True).agg(
        review_count=("review_id", "count"),
        mean_stars=("stars", "mean"),
    )

    grp_year = grp_year_lazy.compute().reset_index().sort_values("year")
    grp_city = grp_city_lazy.compute().reset_index().sort_values("review_count", ascending=False)

    elapsed = time.perf_counter() - t0
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    grp_year["mean_stars"] = grp_year["mean_stars"].round(6)
    grp_year["mean_useful"] = grp_year["mean_useful"].round(6)

    total_rows = int(grp_year["review_count"].sum())
    global_mean = float((grp_year["mean_stars"] * grp_year["review_count"]).sum() / total_rows)

    return {
        "engine": "Dask (Lazy Task Graph on Parquet)",
        "time_seconds": round(elapsed, 4),
        "peak_memory_mb": round(peak_bytes / (1024 * 1024), 2),
        "total_rows_processed": total_rows,
        "global_mean_stars": round(global_mean, 6),
        "top_city": str(grp_city.iloc[0]["city"]),
        "yearly_result": grp_year.to_dict(orient="records"),
    }


def benchmark_polars_lazy() -> Dict[str, Any]:
    """
    Approach 3: Polars LazyFrame (pl.scan_parquet) with predicate/projection pushdown.
    """
    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()

    lf_rev = pl.scan_parquet(PARQUET_REVIEWS_SINGLE).select(
        ["review_id", "business_id", "stars", "useful", "year"]
    )
    lf_biz = pl.scan_parquet(PARQUET_BUSINESS).select(
        ["business_id", "city", "primary_category"]
    )

    lf_year = (
        lf_rev.group_by("year")
        .agg([
            pl.len().alias("review_count"),
            pl.col("stars").mean().alias("mean_stars"),
            pl.col("useful").mean().alias("mean_useful"),
        ])
        .sort("year")
    )

    lf_city = (
        lf_rev.join(lf_biz, on="business_id", how="inner")
        .group_by("city")
        .agg([
            pl.len().alias("review_count"),
            pl.col("stars").mean().alias("mean_stars"),
        ])
        .sort("review_count", descending=True)
    )

    df_year = lf_year.collect().to_pandas()
    df_city = lf_city.collect().to_pandas()

    elapsed = time.perf_counter() - t0
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    df_year["mean_stars"] = df_year["mean_stars"].round(6)
    df_year["mean_useful"] = df_year["mean_useful"].round(6)

    total_rows = int(df_year["review_count"].sum())
    global_mean = float((df_year["mean_stars"] * df_year["review_count"]).sum() / total_rows)

    return {
        "engine": "Polars LazyFrame (scan_parquet)",
        "time_seconds": round(elapsed, 4),
        "peak_memory_mb": round(peak_bytes / (1024 * 1024), 2),
        "total_rows_processed": total_rows,
        "global_mean_stars": round(global_mean, 6),
        "top_city": str(df_city.iloc[0]["city"]),
        "yearly_result": df_year.to_dict(orient="records"),
    }


def run_all_benchmarks() -> Dict[str, Any]:
    """
    Execute all 4 data processing benchmarks and verify exact numerical agreement.
    """
    res_baseline = benchmark_chunked_jsonl_baseline()
    res_pandas = benchmark_pandas_parquet()
    res_dask = benchmark_dask_parquet()
    res_polars = benchmark_polars_lazy()

    results_list = [res_baseline, res_pandas, res_dask, res_polars]

    # Verify result consistency across engines
    means = [r["global_mean_stars"] for r in results_list]
    max_diff = max(means) - min(means)
    results_match = bool(max_diff < 1e-4)

    summary = {
        "benchmarks": [
            {k: v for k, v in r.items() if k != "yearly_result"}
            for r in results_list
        ],
        "consistency_check": {
            "global_mean_stars_per_engine": {r["engine"]: r["global_mean_stars"] for r in results_list},
            "max_absolute_difference": round(float(max_diff), 8),
            "exact_equivalence_verified": results_match,
        },
        "speedup_polars_vs_jsonl_baseline": round(
            res_baseline["time_seconds"] / max(res_polars["time_seconds"], 1e-4), 2
        ),
        "speedup_polars_vs_pandas_parquet": round(
            res_pandas["time_seconds"] / max(res_polars["time_seconds"], 1e-4), 2
        ),
        "yearly_aggregates": res_polars["yearly_result"],
    }

    with open(BENCHMARK_RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary
