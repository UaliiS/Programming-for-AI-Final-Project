"""
Streamlit Interactive Analytics Dashboard for Yelp Reviews at Scale (Topic #44).
Provides:
- Sidebar filters by Year range, Metropolitan City, Primary Business Category, and Elite Reviewer Status
- Tab 1: Executive Overview & Interactive Plotly Charts (Treemap, City-Category Heatmap, Yearly Trends)
- Tab 2: Out-of-Core ETL, Memory Optimization & Engine Benchmarks (Pandas vs Dask vs Polars)
- Tab 3: Relational Database (ER Schema) & Live Execution of the 8 Analytical SQL Queries
- Tab 4: Statistical Hypothesis Testing (H1: Elite Bias, H2: City/Category ANOVA, H3: Review Length & Useful Votes)
- Tab 5: NLP Star-Rating Predictor & Live Interactive Review Classifier

Run with:
  streamlit run app.py

Authors: Uali Sembay & Arystan Aldiyev (AAI-2605)
"""

import json
import sqlite3
import pandas as pd
import plotly.express as px
import polars as pl
import streamlit as st

from src.config import (
    AUDIT_REPORT_JSON,
    BENCHMARK_RESULTS_JSON,
    HYPOTHESIS_RESULTS_JSON,
    MODEL_METRICS_JSON,
    PARQUET_BUSINESS,
    PARQUET_REVIEWS_SINGLE,
    PARQUET_USER,
    ER_DIAGRAM_PNG,
    PROJECT_METADATA,
    SQLITE_DB_PATH,
)
from src.database import SQL_QUERIES
from src.ml_models import predict_custom_review

st.set_page_config(
    page_title="Yelp Reviews at Scale — Topic #44 Dashboard",
    page_icon="⭐",
    layout="wide",
)


@st.cache_data
def load_json_artifact(path_str: str) -> dict:
    with open(path_str, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_filtered_aggregates(year_min: int, year_max: int, selected_cities: list, selected_cats: list, elite_filter: str):
    """Filter the cleaned reviews with a Polars LazyFrame (predicate + projection pushdown)."""
    lf_rev = pl.scan_parquet(PARQUET_REVIEWS_SINGLE).filter(
        (pl.col("year") >= year_min) & (pl.col("year") <= year_max)
    )
    lf_biz = pl.scan_parquet(PARQUET_BUSINESS)
    lf_usr = pl.scan_parquet(PARQUET_USER).select(["user_id", "is_elite"])

    if selected_cities:
        lf_biz = lf_biz.filter(pl.col("city").cast(pl.Utf8).is_in(selected_cities))
    if selected_cats:
        lf_biz = lf_biz.filter(pl.col("primary_category").cast(pl.Utf8).is_in(selected_cats))

    if elite_filter == "Elite Only":
        lf_usr = lf_usr.filter(pl.col("is_elite") == 1)
    elif elite_filter == "Regular Only":
        lf_usr = lf_usr.filter(pl.col("is_elite") == 0)

    lf_joined = lf_rev.join(lf_biz, on="business_id", how="inner").join(lf_usr, on="user_id", how="inner")

    kpi_df = lf_joined.select([
        pl.len().alias("total_reviews"),
        pl.col("business_id").n_unique().alias("unique_businesses"),
        pl.col("user_id").n_unique().alias("unique_users"),
        pl.col("stars").mean().alias("mean_stars"),
        pl.col("useful").mean().alias("mean_useful"),
    ]).collect().to_pandas()

    by_city_cat = (
        lf_joined.group_by(["city", "primary_category"])
        .agg([
            pl.len().alias("review_count"),
            pl.col("stars").mean().alias("mean_stars"),
        ])
        .collect()
        .to_pandas()
    )

    by_year_tier = (
        lf_joined.group_by(["year", "is_elite"])
        .agg([
            pl.len().alias("review_count"),
            pl.col("stars").mean().alias("mean_stars"),
        ])
        .sort("year")
        .collect()
        .to_pandas()
    )

    star_dist = (
        lf_joined.group_by(["stars", "is_elite"])
        .agg(pl.len().alias("count"))
        .sort("stars")
        .collect()
        .to_pandas()
    )

    return kpi_df, by_city_cat, by_year_tier, star_dist


def main():
    st.title(f"⭐ {PROJECT_METADATA['title']}")
    st.caption(
        f"**Topic #{PROJECT_METADATA['topic_number']}** | **{PROJECT_METADATA['cluster']}** | "
        f"**Authors:** {', '.join(PROJECT_METADATA['authors'])} ({PROJECT_METADATA['group']})"
    )

    if not PARQUET_REVIEWS_SINGLE.exists():
        st.error("Cleaned Parquet files not found! Please run `python main.py --auto` first.")
        return

    df_biz_meta = pd.read_parquet(PARQUET_BUSINESS, columns=["city", "primary_category"])
    top_cities = df_biz_meta["city"].astype(str).value_counts().head(20).index.tolist()
    top_cats = df_biz_meta["primary_category"].astype(str).value_counts().head(15).index.tolist()

    # Sidebar Filters
    st.sidebar.header("🔍 Interactive Filters")
    years = pl.scan_parquet(PARQUET_REVIEWS_SINGLE).select(pl.col("year").min().alias("lo"), pl.col("year").max().alias("hi")).collect().row(0)
    y_lo, y_hi = int(years[0]), int(years[1])
    year_range = st.sidebar.slider("Select Review Year Range", min_value=y_lo, max_value=y_hi, value=(max(y_lo, 2010), y_hi))
    selected_cities = st.sidebar.multiselect("Filter by City (Top 20)", options=top_cities, default=top_cities[:10])
    selected_cats = st.sidebar.multiselect("Filter by Primary Category", options=top_cats, default=top_cats[:10])
    elite_filter = st.sidebar.radio("User Tier Filter", options=["All Users", "Elite Only", "Regular Only"], help="Elite = user had Yelp Elite status in at least one year", index=0)

    kpi_df, by_city_cat, by_year_tier, star_dist = load_filtered_aggregates(
        year_range[0], year_range[1], selected_cities, selected_cats, elite_filter
    )

    # Top KPI Metrics Row
    c1, c2, c3, c4, c5 = st.columns(5)
    total_rev = int(kpi_df["total_reviews"].iloc[0]) if len(kpi_df) else 0
    uniq_biz = int(kpi_df["unique_businesses"].iloc[0]) if len(kpi_df) else 0
    uniq_usr = int(kpi_df["unique_users"].iloc[0]) if len(kpi_df) else 0
    mean_st = float(kpi_df["mean_stars"].iloc[0]) if total_rev > 0 else 0.0
    mean_uf = float(kpi_df["mean_useful"].iloc[0]) if total_rev > 0 else 0.0

    c1.metric("Filtered Reviews", f"{total_rev:,}")
    c2.metric("Active Businesses", f"{uniq_biz:,}")
    c3.metric("Active Reviewers", f"{uniq_usr:,}")
    c4.metric("Mean Star Rating", f"{mean_st:.3f} ★")
    c5.metric("Mean Useful Votes", f"{mean_uf:.2f}")

    tabs = st.tabs([
        "📊 1. Interactive EDA & Geo/Category Trends",
        "⚡ 2. ETL Audit & Engine Benchmarks (Pandas/Dask/Polars)",
        "🗄️ 3. SQLite Database & 8 SQL Queries",
        "🧪 4. Statistical Hypotheses",
        "🤖 5. NLP Star Predictor & Live Demo",
    ])

    # TAB 1: Interactive EDA
    with tabs[0]:
        st.subheader("Interactive City & Category Analytics (Powered by Polars LazyFrame)")
        if total_rev > 0:
            col_a, col_b = st.columns(2)
            with col_a:
                star_dist["User Tier"] = star_dist["is_elite"].map({1: "Ever-Elite users", 0: "Never-Elite users"})
                fig_bar = px.bar(
                    star_dist,
                    x="stars",
                    y="count",
                    color="User Tier",
                    barmode="group",
                    title="Star Rating Distribution by Reviewer Tier",
                )
                st.plotly_chart(fig_bar, width="stretch")

            with col_b:
                by_year_tier["User Tier"] = by_year_tier["is_elite"].map({1: "Ever-Elite users", 0: "Never-Elite users"})
                fig_line = px.line(
                    by_year_tier,
                    x="year",
                    y="review_count",
                    color="User Tier",
                    markers=True,
                    title="Annual Review Volume Trajectory by Tier",
                )
                st.plotly_chart(fig_line, width="stretch")

            fig_tree = px.treemap(
                by_city_cat,
                path=[px.Constant("Selected Yelp Scope"), "city", "primary_category"],
                values="review_count",
                color="mean_stars",
                color_continuous_scale="RdYlGn",
                title="Interactive Hierarchy: City -> Primary Category Volume & Mean Stars",
            )
            st.plotly_chart(fig_tree, width="stretch")

    # TAB 2: Benchmarks & ETL Audit
    with tabs[1]:
        st.subheader("Out-of-Core JSONL → Partitioned Parquet ETL & Memory Optimization")
        if AUDIT_REPORT_JSON.exists():
            audit = load_json_artifact(str(AUDIT_REPORT_JSON))
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Data Quality Audit Summary")
                st.json(audit["reviews_audit"])
            with col2:
                st.markdown("#### Memory & Storage Optimization")
                st.json(audit["reviews_memory_optimization"])

        st.subheader("Engine Benchmark Comparison (Requirement 9 & Cluster 8)")
        if BENCHMARK_RESULTS_JSON.exists():
            bench = load_json_artifact(str(BENCHMARK_RESULTS_JSON))
            df_b = pd.DataFrame(bench["benchmarks"])
            st.dataframe(df_b, width="stretch")
            st.caption(f"Identical results across engines: {bench['all_results_identical']} — {bench['method']}")
            col_b1, col_b2 = st.columns(2)
            with col_b1:
                fig_t = px.bar(
                    df_b, x="time_seconds_median", y="engine", orientation="h", color="engine", log_x=True,
                    title="Median run time (seconds, log scale)"
                )
                st.plotly_chart(fig_t, width="stretch")
            with col_b2:
                fig_m = px.bar(
                    df_b, x="peak_rss_mb", y="engine", orientation="h", color="engine",
                    title="Peak process memory, RSS (MB)"
                )
                st.plotly_chart(fig_m, width="stretch")

    # TAB 3: SQL Queries
    with tabs[2]:
        st.subheader("Normalized 5-Table SQLite Database & 8 Analytical SQL Queries")
        if ER_DIAGRAM_PNG.exists():
            with st.expander("ER diagram", expanded=False):
                st.image(str(ER_DIAGRAM_PNG))
        selected_q = st.selectbox("Select an Analytical SQL Query to Execute Live:", list(SQL_QUERIES.keys()))
        st.code(SQL_QUERIES[selected_q], language="sql")
        if SQLITE_DB_PATH.exists():
            conn = sqlite3.connect(str(SQLITE_DB_PATH))
            conn.execute("PRAGMA temp_store = MEMORY;")
            conn.execute("PRAGMA cache_size = -64000;")
            df_q = pd.read_sql_query(SQL_QUERIES[selected_q], conn)
            conn.close()
            st.dataframe(df_q, width="stretch")

    # TAB 4: Hypotheses
    with tabs[3]:
        st.subheader("Statistical Hypothesis Testing on Yelp Reviews")
        if HYPOTHESIS_RESULTS_JSON.exists():
            hyps = load_json_artifact(str(HYPOTHESIS_RESULTS_JSON))
            for k, val in hyps.items():
                with st.expander(f"{val['title']} — {val['decision']}", expanded=True):
                    st.write(f"**Null Hypothesis ($H_0$):** {val['null_hypothesis']}")
                    st.write(f"**Empirical Conclusion:** {val['conclusion']}")
                    st.json(val)

    # TAB 5: NLP Models & Live Predictor
    with tabs[4]:
        st.subheader("Star-Rating Prediction from Review Text (Leakage-Free Chronological Split)")
        if MODEL_METRICS_JSON.exists():
            ml_data = load_json_artifact(str(MODEL_METRICS_JSON))
            st.markdown(f"**Split Strategy:** `{ml_data['split_strategy']}` | **Best Model:** `{ml_data['best_model']}`")
            st.markdown(f"**Model selection:** {ml_data['selection_rule']}")
            st.dataframe(pd.json_normalize(ml_data["models_comparison"]), width="stretch")

        st.markdown("---")
        st.subheader("🎯 Live Interactive Review Star-Rating Predictor")
        sample_input = st.text_area(
            "Enter any business or restaurant review below:",
            value="Absolutely amazing dinner! The chef came out to greet us, the tacos were fresh and delicious, and service was super friendly.",
            height=100,
        )
        if st.button("Predict Star Rating"):
            try:
                pred_res = predict_custom_review(sample_input)
                st.success(
                    f"**Predicted Star Rating:** {'⭐' * pred_res['predicted_stars']} "
                    f"({pred_res['predicted_stars']} / 5 Stars) — **{pred_res['sentiment_category']}**"
                )
                if pred_res["class_probabilities_pct"]:
                    prob_df = pd.DataFrame(
                        list(pred_res["class_probabilities_pct"].items()),
                        columns=["Star Class", "Probability (%)"],
                    )
                    st.bar_chart(prob_df.set_index("Star Class"))
            except Exception as err:
                st.error(f"Error: {err}")


if __name__ == "__main__":
    main()
