"""
Engine benchmark (Requirement 9 + Cluster 8): the SAME task solved four ways.

Task: (a) per-year review count, mean stars, mean useful votes;
      (b) join reviews with businesses and compute review count + mean stars per city.

    Baseline   : raw JSONL read line by line in chunks, incremental groupby with Pandas
    Approach 1 : Pandas, eager, on Parquet
    Approach 2 : Dask DataFrame, lazy task graph on the year-partitioned Parquet
    Approach 3 : Polars LazyFrame (scan_parquet, projection pushdown)

Measurement: every engine runs in a fresh child process (spawn), so imports and caches
of one engine cannot help another. Peak memory is the child's resident set size (RSS),
sampled every 5 ms by a background thread with psutil. Unlike tracemalloc, RSS also
counts the native (Rust / C++ / Arrow) memory that Polars, PyArrow and Dask allocate.
We report peak RSS and the increase over the post-import baseline. Each engine is run
`repeats` times; the median time is reported. Results of all engines are compared
cell-by-cell for equality.

Authors: Uali Sembay & Arystan Aldiyev (AAI-2605)
"""

import json
import multiprocessing as mp
import platform
import statistics
import time
from typing import Any, Callable, Dict, List

from src.config import (
    BENCHMARK_RESULTS_JSON,
    PARQUET_BUSINESS,
    PARQUET_REVIEWS_DIR,
    PARQUET_REVIEWS_SINGLE,
    RAW_REVIEW_JSON,
)


class PeakRSSSampler:
    """Background thread that records the peak RSS (MB) of the current process."""

    def __init__(self, interval: float = 0.005):
        import threading
        import psutil
        self._proc = psutil.Process()
        self._interval = interval
        self._stop = threading.Event()
        self.baseline_mb = self._proc.memory_info().rss / 2**20
        self.peak_mb = self.baseline_mb
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            rss = self._proc.memory_info().rss / 2**20
            if rss > self.peak_mb:
                self.peak_mb = rss
            self._stop.wait(self._interval)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        rss = self._proc.memory_info().rss / 2**20
        self.peak_mb = max(self.peak_mb, rss)
        return False


# ---------------------------------------------------------------------------
# The four implementations (each returns (yearly_df, city_df) as pandas)
# ---------------------------------------------------------------------------
def task_jsonl_chunked_baseline():
    import pandas as pd
    from src.etl_pipeline import stream_jsonl_chunks
    from src.etl_pipeline import MIN_TEXT_CHARS, USEFUL_CAP  # same cleaning rules as the ETL

    biz = pd.read_parquet(PARQUET_BUSINESS, columns=["business_id", "city"])
    city_of = dict(zip(biz["business_id"].astype(str), biz["city"].astype(str)))
    clean_ids = set(pd.read_parquet(PARQUET_REVIEWS_SINGLE, columns=["review_id"])["review_id"])

    year_acc: Dict[int, List[float]] = {}
    city_acc: Dict[str, List[float]] = {}
    for chunk in stream_jsonl_chunks(RAW_REVIEW_JSON, chunk_size=100_000):
        df = pd.DataFrame(chunk, columns=["review_id", "business_id", "stars", "useful", "date"])
        df = df[df["review_id"].isin(clean_ids)]          # identical row set to the cleaned Parquet
        df["useful"] = df["useful"].clip(0, USEFUL_CAP)
        df["year"] = df["date"].str.slice(0, 4).astype(int)
        g = df.groupby("year").agg(n=("stars", "size"), s=("stars", "sum"), u=("useful", "sum"))
        for yr, row in g.iterrows():
            acc = year_acc.setdefault(int(yr), [0, 0.0, 0.0])
            acc[0] += row["n"]; acc[1] += row["s"]; acc[2] += row["u"]
        df["city"] = df["business_id"].map(city_of)
        gc = df.dropna(subset=["city"]).groupby("city").agg(n=("stars", "size"), s=("stars", "sum"))
        for city, row in gc.iterrows():
            acc = city_acc.setdefault(city, [0, 0.0])
            acc[0] += row["n"]; acc[1] += row["s"]
    yearly = pd.DataFrame([(y, v[0], v[1] / v[0], v[2] / v[0]) for y, v in year_acc.items()],
                          columns=["year", "review_count", "mean_stars", "mean_useful"])
    city = pd.DataFrame([(c, v[0], v[1] / v[0]) for c, v in city_acc.items()],
                        columns=["city", "review_count", "mean_stars"])
    return yearly, city


def task_pandas_parquet():
    import pandas as pd
    rev = pd.read_parquet(PARQUET_REVIEWS_SINGLE, columns=["business_id", "stars", "useful", "year"])
    biz = pd.read_parquet(PARQUET_BUSINESS, columns=["business_id", "city"])
    yearly = rev.groupby("year").agg(review_count=("stars", "size"), mean_stars=("stars", "mean"),
                                     mean_useful=("useful", "mean")).reset_index()
    merged = rev.merge(biz, on="business_id", how="inner")
    city = merged.groupby("city", observed=True).agg(review_count=("stars", "size"),
                                                     mean_stars=("stars", "mean")).reset_index()
    return yearly, city


def task_dask_parquet():
    import dask
    import dask.dataframe as dd
    dask.config.set(scheduler="threads")
    rev = dd.read_parquet(str(PARQUET_REVIEWS_DIR), columns=["business_id", "stars", "useful", "year"])
    biz = dd.read_parquet(str(PARQUET_BUSINESS), columns=["business_id", "city"])
    rev = rev.assign(stars=rev["stars"].astype("float64"), useful=rev["useful"].astype("float64"),
                     year=rev["year"].astype("int64"))
    yearly_lazy = rev.groupby("year").agg(review_count=("stars", "count"), mean_stars=("stars", "mean"),
                                          mean_useful=("useful", "mean"))
    city_lazy = rev.merge(biz, on="business_id", how="inner").groupby("city", observed=True).agg(
        review_count=("stars", "count"), mean_stars=("stars", "mean"))
    yearly, city = dask.compute(yearly_lazy, city_lazy)
    return yearly.reset_index(), city.reset_index()


def task_polars_lazy():
    import polars as pl
    rev = pl.scan_parquet(PARQUET_REVIEWS_SINGLE).select(["business_id", "stars", "useful", "year"])
    biz = pl.scan_parquet(PARQUET_BUSINESS).select(["business_id", pl.col("city").cast(pl.Utf8)])
    yearly = rev.group_by("year").agg(pl.len().alias("review_count"), pl.col("stars").mean().alias("mean_stars"),
                                      pl.col("useful").mean().alias("mean_useful"))
    city = rev.join(biz, on="business_id", how="inner").group_by("city").agg(
        pl.len().alias("review_count"), pl.col("stars").mean().alias("mean_stars"))
    y, c = pl.collect_all([yearly, city])
    return y.to_pandas(), c.to_pandas()


ENGINES: Dict[str, Callable] = {
    "Baseline: JSONL chunked + Pandas": task_jsonl_chunked_baseline,
    "Approach 1: Pandas (Parquet, eager)": task_pandas_parquet,
    "Approach 2: Dask (partitioned Parquet, lazy)": task_dask_parquet,
    "Approach 3: Polars LazyFrame (scan_parquet)": task_polars_lazy,
}


def _normalise(yearly, city):
    yearly = yearly[["year", "review_count", "mean_stars", "mean_useful"]].copy()
    yearly["year"] = yearly["year"].astype(int)
    yearly["review_count"] = yearly["review_count"].astype(int)
    yearly = yearly.sort_values("year").reset_index(drop=True)
    city = city[["city", "review_count", "mean_stars"]].copy()
    city["city"] = city["city"].astype(str)
    city["review_count"] = city["review_count"].astype(int)
    city = city.sort_values(["review_count", "city"], ascending=[False, True]).reset_index(drop=True)
    return yearly, city


def _child(engine_name: str, queue) -> None:
    """Runs inside a fresh process: import engine libs, measure baseline RSS, run task, measure peak."""
    try:
        import pandas  # noqa: F401  (common import cost is part of the baseline, not the task)
        func = ENGINES[engine_name]
        if "Dask" in engine_name:
            import dask.dataframe  # noqa: F401
        if "Polars" in engine_name:
            import polars  # noqa: F401
        with PeakRSSSampler() as sampler:
            t0 = time.perf_counter()
            yearly, city = func()
            elapsed = time.perf_counter() - t0
        base, peak = sampler.baseline_mb, sampler.peak_mb
        yearly, city = _normalise(yearly, city)
        queue.put({"ok": True, "time": elapsed, "peak_rss_mb": peak, "baseline_rss_mb": base,
                   "yearly": yearly.to_dict(orient="list"), "city": city.head(50).to_dict(orient="list")})
    except Exception as exc:  # report the error to the parent instead of hanging
        queue.put({"ok": False, "error": f"{type(exc).__name__}: {exc}"})


def _run_isolated(engine_name: str) -> Dict[str, Any]:
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_child, args=(engine_name, q))
    p.start()
    res = q.get()
    p.join()
    if not res["ok"]:
        raise RuntimeError(f"{engine_name} failed: {res['error']}")
    return res


def _tables_equal(a: Dict[str, list], b: Dict[str, list], tol: float = 1e-6) -> bool:
    if a.keys() != b.keys():
        return False
    for k in a:
        if len(a[k]) != len(b[k]):
            return False
        for x, y in zip(a[k], b[k]):
            if isinstance(x, float) or isinstance(y, float):
                if abs(float(x) - float(y)) > tol:
                    return False
            elif x != y:
                return False
    return True


def run_all_benchmarks(repeats: int = 3) -> Dict[str, Any]:
    """Run every engine `repeats` times in isolated processes and save a comparison report."""
    import pandas as pd

    rows: List[Dict[str, Any]] = []
    outputs: Dict[str, Dict[str, Any]] = {}
    for name in ENGINES:
        runs = []
        n = 1 if name.startswith("Baseline") else repeats   # baseline is slow and very stable
        for _ in range(n):
            runs.append(_run_isolated(name))
        times = [r["time"] for r in runs]
        peaks = [r["peak_rss_mb"] for r in runs]
        deltas = [r["peak_rss_mb"] - r["baseline_rss_mb"] for r in runs]
        rows.append({
            "engine": name,
            "runs": n,
            "time_seconds_median": round(statistics.median(times), 3),
            "time_seconds_min": round(min(times), 3),
            "peak_rss_mb": round(statistics.median(peaks), 1),
            "task_memory_mb": round(statistics.median(deltas), 1),
            "total_rows": int(sum(runs[0]["yearly"]["review_count"])),
            "top_city": runs[0]["city"]["city"][0],
        })
        outputs[name] = runs[0]
        print(f"  -> {name:<46} median {rows[-1]['time_seconds_median']:>8.3f}s | "
              f"peak RSS {rows[-1]['peak_rss_mb']:>7.1f} MB | task +{rows[-1]['task_memory_mb']:>7.1f} MB")

    ref_name = list(ENGINES)[1]
    ref = outputs[ref_name]
    agreement = {}
    for name, out in outputs.items():
        agreement[name] = {
            "yearly_table_identical": _tables_equal(out["yearly"], ref["yearly"]),
            "top50_city_table_identical": _tables_equal(out["city"], ref["city"]),
        }
    base_t = rows[0]["time_seconds_median"]
    for r in rows:
        r["speedup_vs_baseline"] = round(base_t / max(r["time_seconds_median"], 1e-9), 1)

    best_time = min(rows[1:], key=lambda r: r["time_seconds_median"])
    best_mem = min(rows[1:], key=lambda r: r["task_memory_mb"])
    summary = {
        "task": "per-year count/mean stars/mean useful + reviews⋈businesses per-city count/mean stars",
        "method": "each engine in a fresh spawned process; time = median of runs; memory = peak RSS sampled every 5 ms (psutil); task memory = peak minus RSS after imports",
        "machine": {"python": platform.python_version(), "platform": platform.platform(), "cpu_count": mp.cpu_count()},
        "benchmarks": rows,
        "result_agreement_vs_pandas": agreement,
        "all_results_identical": all(v["yearly_table_identical"] and v["top50_city_table_identical"] for v in agreement.values()),
        "fastest_engine": best_time["engine"],
        "lowest_memory_engine": best_mem["engine"],
        "yearly_aggregates": pd.DataFrame(ref["yearly"]).to_dict(orient="records"),
        "top_cities": pd.DataFrame(ref["city"]).head(15).to_dict(orient="records"),
    }
    with open(BENCHMARK_RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary
