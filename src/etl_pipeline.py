"""
Out-of-core ETL, dtype optimisation and data-quality audit.

    raw JSONL --(line-by-line, bounded chunks)--> audit + clean + optimise dtypes
              --> year-partitioned Parquet  (data/parquet/reviews_partitioned/year=YYYY/)
              --> single cleaned Parquet    (data/parquet/reviews_cleaned.parquet, written
                                             incrementally with a ParquetWriter, never held
                                             in memory as one DataFrame)

Every number written to outputs/data_quality_audit.json is computed from the data in
this run; nothing is hard-coded.

Authors: Uali Sembay & Arystan Aldiyev (AAI-2605)
"""

import hashlib
import json
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Set, Tuple

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from src.config import (
    AUDIT_REPORT_JSON,
    PARQUET_BUSINESS,
    PARQUET_REVIEWS_DIR,
    PARQUET_REVIEWS_SINGLE,
    PARQUET_USER,
    RAW_BUSINESS_JSON,
    RAW_MANIFEST_JSON,
    RAW_REVIEW_JSON,
    RAW_USER_JSON,
)

# Yelp was founded in October 2004; the dataset snapshot ends in January 2022.
MIN_VALID_DATE = pd.Timestamp("2004-10-01")
MAX_VALID_DATE = pd.Timestamp("2022-12-31 23:59:59")
USEFUL_CAP = 100              # winsorisation cap for vote counts (justified in the audit)
MIN_TEXT_CHARS = 10

MAJOR_MACRO_CATEGORIES = [
    "Restaurants", "Food", "Shopping", "Beauty & Spas", "Home Services", "Nightlife",
    "Health & Medical", "Local Services", "Automotive", "Active Life",
    "Event Planning & Services", "Hotels & Travel", "Arts & Entertainment", "Pets",
    "Financial Services",
]

REVIEW_SCHEMA = pa.schema([
    ("review_id", pa.string()),
    ("user_id", pa.string()),
    ("business_id", pa.string()),
    ("stars", pa.int8()),
    ("useful", pa.int16()),
    ("funny", pa.int16()),
    ("cool", pa.int16()),
    ("text", pa.string()),
    ("date", pa.string()),
    ("year", pa.int16()),
    ("month", pa.int8()),
    ("text_length", pa.int32()),
])


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------
def stream_jsonl_chunks(filepath: Path, chunk_size: int = 100_000) -> Generator[List[Dict[str, Any]], None, None]:
    """
    Read a JSON Lines file line by line and yield lists of at most `chunk_size` records,
    so memory use is bounded by the chunk size, not the file size.
    Malformed lines are reported and skipped (try...except).
    """
    if not filepath.exists():
        raise FileNotFoundError(f"JSONL source file not found: {filepath}. Run stage 1 (data download) first.")

    chunk: List[Dict[str, Any]] = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                chunk.append(json.loads(stripped))
            except json.JSONDecodeError as err:
                print(f"[WARN] Malformed JSON at {filepath.name}:{line_num} -> {err}")
                continue
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
    if chunk:
        yield chunk


def memory_mb(df: pd.DataFrame) -> float:
    """Deep in-memory size of a DataFrame in MB (includes Python string objects)."""
    return float(df.memory_usage(deep=True).sum()) / 2**20


def optimize_dataframe_dtypes(df: pd.DataFrame, categorical_cols: List[str] = None) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Downcast numeric columns and convert low-cardinality strings to `category`.
    Returns the optimised frame and before/after memory measurements (MB).
    """
    before = memory_mb(df)
    out = df.copy()
    for col in categorical_cols or []:
        if col in out.columns:
            out[col] = out[col].astype("category")
    for col in out.columns:
        if pd.api.types.is_integer_dtype(out[col]) and not isinstance(out[col].dtype, pd.CategoricalDtype):
            out[col] = pd.to_numeric(out[col], downcast="integer")
        elif pd.api.types.is_float_dtype(out[col]):
            out[col] = pd.to_numeric(out[col], downcast="float")
    after = memory_mb(out)
    return out, {
        "memory_before_mb": round(before, 2),
        "memory_after_mb": round(after, 2),
        "reduction_percent": round((1 - after / max(before, 1e-9)) * 100, 2),
    }


def extract_macro_category(cat_str: Any) -> str:
    """Map Yelp's comma-separated category list to one primary macro-category."""
    if not isinstance(cat_str, str) or not cat_str.strip():
        return "Uncategorized"
    tokens = [t.strip() for t in cat_str.split(",") if t.strip()]
    token_set = set(tokens)
    for macro in MAJOR_MACRO_CATEGORIES:
        if macro in token_set:
            return macro
    return "Other"


def text_fingerprint(text: str) -> bytes:
    """Deterministic 64-bit fingerprint of a review text (stable across processes, unlike hash())."""
    return hashlib.blake2b(text.encode("utf-8", "ignore"), digest_size=8).digest()


def parse_elite_years(elite: Any) -> List[int]:
    """
    Parse Yelp's `elite` string (e.g. "2017,2018,2019") into a list of years.
    The official file encodes 2020 as "20,20"; we repair that known artefact.
    """
    if not isinstance(elite, str) or not elite.strip():
        return []
    s = elite.replace("20,20", "2020")
    years = []
    for tok in s.split(","):
        tok = tok.strip()
        if tok.isdigit() and len(tok) == 4:
            years.append(int(tok))
    return sorted(set(years))


# ---------------------------------------------------------------------------
# Businesses
# ---------------------------------------------------------------------------
def process_businesses() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Audit, clean and optimise business.json, then save it to Parquet."""
    records: List[Dict[str, Any]] = []
    for chunk in stream_jsonl_chunks(RAW_BUSINESS_JSON, chunk_size=50_000):
        records.extend(chunk)
    df = pd.DataFrame(records)

    audit: Dict[str, Any] = {"raw_rows": int(len(df))}
    audit["missing_values"] = {c: int(v) for c, v in df.isna().sum().items() if v > 0}
    audit["empty_postal_code"] = int((df["postal_code"].fillna("").astype(str).str.strip() == "").sum())
    audit["duplicate_business_id"] = int(df["business_id"].duplicated().sum())
    audit["invalid_coordinates"] = int((~df["latitude"].between(-90, 90) | ~df["longitude"].between(-180, 180)).sum())
    audit["invalid_business_stars"] = int((~df["stars"].between(1, 5)).sum())
    audit["negative_review_count"] = int((df["review_count"] < 0).sum())

    raw_cities = df["city"].fillna("").astype(str)
    clean_city = (
        raw_cities.str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .str.replace(r"^(St\.?|Saint)\s+", "Saint ", regex=True)
        .str.title()
    )
    clean_city = clean_city.replace("", "Unknown")
    audit["city_spellings_before"] = int(raw_cities.nunique())
    audit["city_spellings_after"] = int(clean_city.nunique())
    audit["city_rows_changed"] = int((raw_cities != clean_city).sum())
    df["city"] = clean_city
    df["state"] = df["state"].fillna("NA").astype(str).str.strip().str.upper()

    df = df.drop_duplicates(subset="business_id", keep="first")
    df["categories"] = df["categories"].fillna("Uncategorized")

    def parse_price(attr: Any) -> float:
        if isinstance(attr, dict):
            val = str(attr.get("RestaurantsPriceRange2", "")).strip()
            if val in {"1", "2", "3", "4"}:
                return float(val)
        return np.nan

    df["price_range"] = df["attributes"].apply(parse_price)
    audit["price_range_missing"] = int(df["price_range"].isna().sum())
    df["primary_category"] = df["categories"].apply(extract_macro_category)
    df = df.drop(columns=["attributes", "hours"], errors="ignore")

    df_opt, mem = optimize_dataframe_dtypes(df, categorical_cols=["city", "state", "primary_category"])
    df_opt["price_range"] = df_opt["price_range"].astype("Int8")
    df_opt.to_parquet(PARQUET_BUSINESS, index=False, compression="snappy")
    audit["cleaned_rows"] = int(len(df_opt))
    audit["memory"] = mem
    return df_opt, audit


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------
def process_users() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Audit, clean and optimise the real user.json (friends list already removed at download)."""
    records: List[Dict[str, Any]] = []
    for chunk in stream_jsonl_chunks(RAW_USER_JSON, chunk_size=100_000):
        records.extend(chunk)
    df = pd.DataFrame(records)
    df = df.drop(columns=["friends"], errors="ignore")

    audit: Dict[str, Any] = {"raw_rows": int(len(df))}
    audit["missing_values"] = {c: int(v) for c, v in df.isna().sum().items() if v > 0}
    audit["duplicate_user_id"] = int(df["user_id"].duplicated().sum())
    since = pd.to_datetime(df["yelping_since"], errors="coerce")
    audit["invalid_yelping_since"] = int((since.isna() | (since < MIN_VALID_DATE) | (since > MAX_VALID_DATE)).sum())
    audit["invalid_average_stars"] = int((~df["average_stars"].between(0, 5)).sum())

    elite_years = df["elite"].apply(parse_elite_years)
    df["elite_years"] = elite_years.apply(lambda ys: ",".join(str(y) for y in ys))
    df["n_elite_years"] = elite_years.apply(len).astype("int16")
    df["is_elite"] = (df["n_elite_years"] > 0).astype("int8")
    audit["elite_users"] = int(df["is_elite"].sum())
    audit["elite_share_percent"] = round(100 * df["is_elite"].mean(), 2)
    audit["elite_2020_artifact_repaired"] = int(df["elite"].fillna("").str.contains("20,20", regex=False).sum())

    df = df.drop_duplicates(subset="user_id", keep="first").drop(columns=["elite"])
    compliment_cols = [c for c in df.columns if c.startswith("compliment_")]
    df_opt, mem = optimize_dataframe_dtypes(df)
    df_opt.to_parquet(PARQUET_USER, index=False, compression="snappy")
    audit["cleaned_rows"] = int(len(df_opt))
    audit["compliment_columns"] = len(compliment_cols)
    audit["memory"] = mem
    return df_opt, audit


# ---------------------------------------------------------------------------
# Reviews (the large table) — fully chunked
# ---------------------------------------------------------------------------
def run_out_of_core_reviews_etl(chunk_size: int = 100_000) -> Dict[str, Any]:
    """
    Stream review.json in chunks, audit every chunk, clean it, optimise dtypes,
    and append it to both Parquet outputs. Businesses and users are processed first
    so that referential integrity (foreign keys) can be audited per chunk.
    """
    t0 = time.perf_counter()
    df_biz, biz_audit = process_businesses()
    df_usr, usr_audit = process_users()
    valid_biz: Set[str] = set(df_biz["business_id"].astype(str))
    valid_usr: Set[str] = set(df_usr["user_id"].astype(str))
    del df_biz, df_usr

    if PARQUET_REVIEWS_DIR.exists():
        shutil.rmtree(PARQUET_REVIEWS_DIR)
    PARQUET_REVIEWS_DIR.mkdir(parents=True, exist_ok=True)

    a = {
        "raw_rows": 0,
        "missing_values": {},
        "duplicate_review_id": 0,
        "duplicate_text": 0,
        "short_or_empty_text": 0,
        "invalid_stars": 0,
        "invalid_or_out_of_range_date": 0,
        "orphan_business_id": 0,
        "orphan_user_id": 0,
        "negative_votes": 0,
        "votes_above_cap": 0,
        "max_useful_before_cap": 0,
        "cleaned_rows": 0,
    }
    seen_ids: Set[str] = set()
    seen_text: Set[bytes] = set()
    mem_raw_total = 0.0
    mem_opt_total = 0.0
    mem_raw_notext = 0.0
    mem_opt_notext = 0.0
    writer = pq.ParquetWriter(str(PARQUET_REVIEWS_SINGLE), REVIEW_SCHEMA, compression="snappy")
    try:
        for records in stream_jsonl_chunks(RAW_REVIEW_JSON, chunk_size=chunk_size):
            df = pd.DataFrame(records)
            a["raw_rows"] += len(df)
            for c, v in df.isna().sum().items():
                if v:
                    a["missing_values"][c] = a["missing_values"].get(c, 0) + int(v)
            mem_raw_total += memory_mb(df)
            mem_raw_notext += memory_mb(df.drop(columns=["text"], errors="ignore"))

            # 1. duplicate review_id (within chunk and against earlier chunks)
            dup = df["review_id"].duplicated() | df["review_id"].isin(seen_ids)
            a["duplicate_review_id"] += int(dup.sum())
            df = df[~dup]
            seen_ids.update(df["review_id"])

            # 2. empty / very short texts (no signal for text models)
            df = df.assign(text=df["text"].fillna("").astype(str))
            short = df["text"].str.strip().str.len() < MIN_TEXT_CHARS
            a["short_or_empty_text"] += int(short.sum())
            df = df[~short]

            # 3. exact duplicate texts (copy-pasted reviews -> would leak between train/test)
            fp = df["text"].map(text_fingerprint)
            dup_t = fp.duplicated() | fp.isin(seen_text)
            a["duplicate_text"] += int(dup_t.sum())
            df = df[~dup_t]
            seen_text.update(fp[~dup_t])

            # 4. star rating must be an integer 1..5
            stars = pd.to_numeric(df["stars"], errors="coerce")
            bad_star = stars.isna() | ~stars.isin([1, 2, 3, 4, 5])
            a["invalid_stars"] += int(bad_star.sum())
            df = df[~bad_star]

            # 5. dates must parse and fall within Yelp's lifetime / dataset window
            dt = pd.to_datetime(df["date"], errors="coerce")
            bad_date = dt.isna() | (dt < MIN_VALID_DATE) | (dt > MAX_VALID_DATE)
            a["invalid_or_out_of_range_date"] += int(bad_date.sum())
            df, dt = df[~bad_date], dt[~bad_date]

            # 6. referential integrity: business and user must exist
            orphan_b = ~df["business_id"].isin(valid_biz)
            orphan_u = ~df["user_id"].isin(valid_usr)
            a["orphan_business_id"] += int(orphan_b.sum())
            a["orphan_user_id"] += int((orphan_u & ~orphan_b).sum())
            keep = ~(orphan_b | orphan_u)
            df, dt = df[keep], dt[keep]

            # 7. vote counts: negative values invalid, extreme values winsorised
            votes = df[["useful", "funny", "cool"]]
            a["negative_votes"] += int((votes < 0).any(axis=1).sum())
            a["votes_above_cap"] += int((votes > USEFUL_CAP).any(axis=1).sum())
            a["max_useful_before_cap"] = max(a["max_useful_before_cap"], int(df["useful"].max()) if len(df) else 0)

            out = pd.DataFrame({
                "review_id": df["review_id"].astype(str),
                "user_id": df["user_id"].astype(str),
                "business_id": df["business_id"].astype(str),
                "stars": df["stars"].astype("int8"),
                "useful": df["useful"].clip(0, USEFUL_CAP).astype("int16"),
                "funny": df["funny"].clip(0, USEFUL_CAP).astype("int16"),
                "cool": df["cool"].clip(0, USEFUL_CAP).astype("int16"),
                "text": df["text"],
                "date": dt.dt.strftime("%Y-%m-%d %H:%M:%S"),
                "year": dt.dt.year.astype("int16"),
                "month": dt.dt.month.astype("int8"),
                "text_length": df["text"].str.len().astype("int32"),
            })
            # memory of the optimised chunk (ids as category, as they would be held in an analysis frame)
            opt_view = out.assign(user_id=out["user_id"].astype("category"),
                                  business_id=out["business_id"].astype("category"))
            mem_opt_total += memory_mb(opt_view)
            mem_opt_notext += memory_mb(opt_view.drop(columns=["text"]))

            table = pa.Table.from_pandas(out, schema=REVIEW_SCHEMA, preserve_index=False)
            writer.write_table(table)
            pq.write_to_dataset(table, root_path=str(PARQUET_REVIEWS_DIR), partition_cols=["year"],
                                compression="snappy", existing_data_behavior="overwrite_or_ignore")
            a["cleaned_rows"] += len(out)
    finally:
        writer.close()

    # IQR outlier audit on the cleaned data (computed lazily with Polars, not by loading everything)
    lf = pl.scan_parquet(PARQUET_REVIEWS_SINGLE)
    iqr: Dict[str, Any] = {}
    for col in ["text_length", "useful"]:
        q = lf.select([pl.col(col).quantile(0.25).alias("q1"), pl.col(col).quantile(0.75).alias("q3"),
                       pl.col(col).quantile(0.999).alias("p999")]).collect().row(0, named=True)
        upper = q["q3"] + 1.5 * (q["q3"] - q["q1"])
        n_out = lf.filter(pl.col(col) > upper).select(pl.len()).collect().item()
        iqr[col] = {"q1": q["q1"], "q3": q["q3"], "iqr_upper_fence": upper, "p99_9": q["p999"],
                    "rows_above_fence": int(n_out),
                    "share_above_fence_percent": round(100 * n_out / max(a["cleaned_rows"], 1), 2)}

    raw_mb = RAW_REVIEW_JSON.stat().st_size / 2**20
    parquet_mb = PARQUET_REVIEWS_SINGLE.stat().st_size / 2**20
    removed = a["raw_rows"] - a["cleaned_rows"]
    manifest = json.loads(RAW_MANIFEST_JSON.read_text()) if RAW_MANIFEST_JSON.exists() else {}

    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "dataset_source": "Official Yelp Open Dataset (https://business.yelp.com/data/resources/open-dataset/)",
        "sampling": manifest.get("review_sampling", "n/a"),
        "etl_elapsed_seconds": round(time.perf_counter() - t0, 2),
        "chunk_size": chunk_size,
        "reviews_audit": a,
        "reviews_removed_total": removed,
        "reviews_removed_percent": round(100 * removed / max(a["raw_rows"], 1), 3),
        "iqr_outliers": iqr,
        "businesses_audit": biz_audit,
        "users_audit": usr_audit,
        "cleaning_rules_justification": {
            "duplicate_review_id": f"{a['duplicate_review_id']:,} repeated review_id values; review_id is the primary key, so only the first occurrence is kept.",
            "short_or_empty_text": f"{a['short_or_empty_text']:,} reviews with fewer than {MIN_TEXT_CHARS} characters carry no usable text for the rating model; dropped.",
            "duplicate_text": f"{a['duplicate_text']:,} reviews repeat an earlier text exactly (copy-pasted reviews). Keeping them would put identical texts in train and test (leakage); dropped.",
            "invalid_stars": f"{a['invalid_stars']:,} reviews had a star value outside the integers 1–5; dropped.",
            "invalid_or_out_of_range_date": f"{a['invalid_or_out_of_range_date']:,} reviews had an unparseable date or one outside {MIN_VALID_DATE.date()}–{MAX_VALID_DATE.date()}; dropped.",
            "orphan_keys": f"{a['orphan_business_id']:,} reviews point to an unknown business and {a['orphan_user_id']:,} to an unknown user; dropped so that SQL foreign keys hold.",
            "vote_winsorisation": (f"{a['votes_above_cap']:,} reviews have more than {USEFUL_CAP} useful/funny/cool votes (max {a['max_useful_before_cap']}). "
                                   f"The 99.9th percentile of useful votes is {iqr['useful']['p99_9']}, so a cap of {USEFUL_CAP} only touches extreme viral reviews; they are kept but capped so means are not dominated by a handful of rows."),
            "text_length_outliers": (f"{iqr['text_length']['rows_above_fence']:,} reviews ({iqr['text_length']['share_above_fence_percent']}%) are above the IQR fence for length "
                                     f"({iqr['text_length']['iqr_upper_fence']:.0f} chars). Long reviews are genuine (Yelp allows 5,000 chars), so they are kept."),
            "business_city_normalisation": f"City names were trimmed, title-cased and 'St./St' unified to 'Saint': {biz_audit['city_spellings_before']:,} spellings -> {biz_audit['city_spellings_after']:,}.",
            "business_missing_categories": f"{biz_audit['missing_values'].get('categories', 0):,} businesses have no categories; labelled 'Uncategorized' so they still join.",
            "price_range": f"{biz_audit['price_range_missing']:,} businesses have no RestaurantsPriceRange2 attribute; left as NULL instead of inventing a value.",
        },
        "reviews_memory_optimization": {
            "raw_chunks_in_memory_mb": round(mem_raw_total, 2),
            "optimized_chunks_in_memory_mb": round(mem_opt_total, 2),
            "memory_reduction_percent": round((1 - mem_opt_total / max(mem_raw_total, 1e-9)) * 100, 2),
            "raw_without_text_mb": round(mem_raw_notext, 2),
            "optimized_without_text_mb": round(mem_opt_notext, 2),
            "reduction_without_text_percent": round((1 - mem_opt_notext / max(mem_raw_notext, 1e-9)) * 100, 2),
            "note": "Review text dominates memory and cannot be downcast; the dtype gain is visible on the other columns "
                    "(int64->int8/int16, date string->int16 year/int8 month, id strings->category).",
            "raw_jsonl_disk_mb": round(raw_mb, 2),
            "cleaned_parquet_disk_mb": round(parquet_mb, 2),
            "disk_compression_ratio": round(raw_mb / max(parquet_mb, 1e-9), 2),
            "partition_folders": sorted(p.name for p in PARQUET_REVIEWS_DIR.iterdir() if p.is_dir()),
        },
    }
    with open(AUDIT_REPORT_JSON, "w", encoding="utf-8") as f_out:
        json.dump(report, f_out, indent=2, default=str)
    return report
