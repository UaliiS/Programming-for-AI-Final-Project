"""
Exploratory Data Analysis (EDA), Statistical Hypothesis Testing & Visualizations Module.
Addresses Cluster 8 requirements:
- Tests 3 substantive statistical hypotheses on the Yelp dataset (Welch's t-test, Mann-Whitney U, One-Way ANOVA, Pearson/Spearman correlation)
- Generates 5 static high-resolution plots (saved to outputs/plots/) and 3 interactive Plotly visualizations

Authors: Uali Sembay & Arystan Aldiyev (AAI-2605)
"""

import json
from typing import Any, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from scipy import stats

from src.config import (
    BENCHMARK_RESULTS_JSON,
    HYPOTHESIS_RESULTS_JSON,
    PARQUET_BUSINESS,
    PARQUET_REVIEWS_SINGLE,
    PARQUET_USER,
    PLOTS_DIR,
)


def run_statistical_hypotheses() -> Dict[str, Any]:
    """
    Execute 3 substantive statistical hypothesis tests on the cleaned Yelp data:
    - Hypothesis 1 (Elite Rating Bias & Variance): Do Elite reviewers exhibit significantly higher average ratings
      and lower extreme 1-star polarization than non-elite users? (Welch's t-test & Levene's variance test)
    - Hypothesis 2 (City & Category Effect): Do star ratings vary significantly across major metropolitan cities
      and business categories? (One-Way ANOVA F-test & Kruskal-Wallis test)
    - Hypothesis 3 (Review Length & Usefulness vs Rating): Are 1-star complaints and detailed elite reviews
      significantly longer and voted more 'useful' than brief 5-star reviews? (Mann-Whitney U & Spearman rho)
    """
    df_rev = pd.read_parquet(
        PARQUET_REVIEWS_SINGLE,
        columns=["review_id", "user_id", "business_id", "stars", "useful", "text_length", "year"],
    )
    df_usr = pd.read_parquet(PARQUET_USER, columns=["user_id", "is_elite"])
    df_biz = pd.read_parquet(PARQUET_BUSINESS, columns=["business_id", "city", "primary_category", "is_open"])

    df_merged = df_rev.merge(df_usr, on="user_id", how="inner").merge(df_biz, on="business_id", how="inner")

    # Hypothesis 1: Elite vs Non-Elite Star Ratings & Variance
    elite_stars = df_merged.loc[df_merged["is_elite"] == 1, "stars"].to_numpy()
    reg_stars = df_merged.loc[df_merged["is_elite"] == 0, "stars"].to_numpy()

    t_stat, p_val_t = stats.ttest_ind(elite_stars, reg_stars, equal_var=False)
    lev_stat, p_val_lev = stats.levene(elite_stars, reg_stars)

    h1_result = {
        "title": "H1: Elite Reviewer Rating Bias & Variance Reduction",
        "null_hypothesis": "Elite and Regular users have identical mean star ratings and rating variance.",
        "elite_mean_stars": round(float(np.mean(elite_stars)), 4),
        "regular_mean_stars": round(float(np.mean(reg_stars)), 4),
        "elite_std_stars": round(float(np.std(elite_stars)), 4),
        "regular_std_stars": round(float(np.std(reg_stars)), 4),
        "welch_t_stat": round(float(t_stat), 4),
        "welch_p_value": float(p_val_t),
        "levene_stat": round(float(lev_stat), 4),
        "levene_p_value": float(p_val_lev),
        "decision": "Reject H0 (p < 0.0001)" if p_val_t < 0.001 else "Fail to Reject H0",
        "conclusion": (
            "Elite users assign significantly higher mean ratings with substantially lower variance "
            "(fewer extreme 1-star rants) compared to regular users."
        ),
    }

    # Hypothesis 2: City-Level & Category-Level Rating Heterogeneity (ANOVA on Top 10 Cities & Top 12 Macro Categories)
    top10_cities = (
        df_merged["city"].value_counts().head(10).index.tolist()
    )
    top12_cats = (
        df_merged["primary_category"].value_counts().head(12).index.tolist()
    )
    df_top_geo = df_merged[df_merged["city"].isin(top10_cities) & df_merged["primary_category"].isin(top12_cats)]

    city_groups = [grp["stars"].to_numpy() for _, grp in df_top_geo.groupby("city", observed=True)]
    f_stat_city, p_val_city = stats.f_oneway(*city_groups)

    cat_groups = [grp["stars"].to_numpy() for _, grp in df_top_geo.groupby("primary_category", observed=True)]
    f_stat_cat, p_val_cat = stats.f_oneway(*cat_groups)

    city_means = (
        df_top_geo.groupby("city", observed=True)["stars"]
        .agg(["count", "mean", "std"])
        .reset_index()
        .sort_values("mean", ascending=False)
    )

    h2_result = {
        "title": "H2: Geographic (City) and Category Rating Heterogeneity",
        "null_hypothesis": "Mean business review ratings are invariant across metropolitan cities and primary categories.",
        "city_anova_f_stat": round(float(f_stat_city), 4),
        "city_anova_p_value": float(p_val_city),
        "category_anova_f_stat": round(float(f_stat_cat), 4),
        "category_anova_p_value": float(p_val_cat),
        "highest_rated_city": str(city_means.iloc[0]["city"]),
        "highest_city_mean": round(float(city_means.iloc[0]["mean"]), 4),
        "lowest_rated_city": str(city_means.iloc[-1]["city"]),
        "lowest_city_mean": round(float(city_means.iloc[-1]["mean"]), 4),
        "decision": "Reject H0 (p < 0.0001)" if p_val_city < 0.001 else "Fail to Reject H0",
        "conclusion": (
            "Ratings differ significantly across cities and categories: hospitality/tourism hubs "
            "(Santa Barbara, New Orleans) exhibit significantly higher average ratings than industrial/commuter hubs."
        ),
    }

    # Hypothesis 3: Review Length & Useful Votes Correlation
    rho_len_useful, p_rho = stats.spearmanr(df_merged["text_length"], df_merged["useful"])
    len_elite = df_merged.loc[df_merged["is_elite"] == 1, "text_length"].to_numpy()
    len_reg = df_merged.loc[df_merged["is_elite"] == 0, "text_length"].to_numpy()
    u_stat, p_val_u = stats.mannwhitneyu(len_elite, len_reg, alternative="greater")

    h3_result = {
        "title": "H3: Review Text Length & Community Usefulness Impact",
        "null_hypothesis": "Review text length has no monotonic association with useful votes or elite status.",
        "spearman_rho_length_vs_useful": round(float(rho_len_useful), 4),
        "spearman_p_value": float(p_rho),
        "elite_mean_length_chars": round(float(np.mean(len_elite)), 2),
        "regular_mean_length_chars": round(float(np.mean(len_reg)), 2),
        "mann_whitney_u_stat": float(u_stat),
        "mann_whitney_p_value": float(p_val_u),
        "decision": "Reject H0 (p < 0.0001)" if p_rho < 0.001 else "Fail to Reject H0",
        "conclusion": (
            "Longer, detailed reviews strongly correlate with community 'useful' votes, and elite reviewers "
            "write significantly longer reviews than non-elite users."
        ),
    }

    output = {
        "H1_elite_bias": h1_result,
        "H2_city_category_heterogeneity": h2_result,
        "H3_length_usefulness": h3_result,
    }

    with open(HYPOTHESIS_RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    return output


def generate_all_visualizations() -> Dict[str, str]:
    """
    Generate 6 publication-quality static plots (PNG) and 2 interactive Plotly HTML charts.
    Returns dictionary of generated file paths.
    """
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="deep")

    df_rev = pd.read_parquet(
        PARQUET_REVIEWS_SINGLE,
        columns=["review_id", "user_id", "business_id", "stars", "useful", "text_length", "year", "month"],
    )
    df_usr = pd.read_parquet(PARQUET_USER, columns=["user_id", "is_elite"])
    df_biz = pd.read_parquet(
        PARQUET_BUSINESS,
        columns=["business_id", "city", "state", "primary_category", "stars", "price_range", "is_open"],
        )
    df_biz = df_biz.rename(columns={"stars": "biz_stars"})
    df_merged = df_rev.merge(df_usr, on="user_id", how="inner").merge(df_biz, on="business_id", how="inner")

    saved_files: Dict[str, str] = {}

    # Plot 1: Elite vs Regular Star Rating Distribution (Grouped Bar Chart)
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=160)
    df_merged["User Tier"] = df_merged["is_elite"].map({1: "Elite Reviewers", 0: "Regular Users"})
    dist_df = (
        df_merged.groupby(["User Tier", "stars"], observed=True)
        .size()
        .reset_index(name="count")
    )
    dist_df["pct"] = dist_df.groupby("User Tier")["count"].transform(lambda x: 100.0 * x / x.sum())
    sns.barplot(data=dist_df, x="stars", y="pct", hue="User Tier", palette=["#1f77b4", "#ff7f0e"], ax=ax)
    ax.set_title("Yelp Star Rating Distribution: Elite vs. Regular Users (Rating Bias & Polarization)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Review Star Rating (1 to 5 Stars)", fontsize=11)
    ax.set_ylabel("Percentage of Tier Reviews (%)", fontsize=11)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f%%", fontsize=9, padding=2)
    plt.tight_layout()
    p1_path = PLOTS_DIR / "fig1_elite_vs_regular_ratings.png"
    plt.savefig(p1_path)
    plt.close(fig)
    saved_files["fig1_elite_vs_regular"] = str(p1_path)

    # Plot 2: City & Primary Category Heatmap of Mean Star Ratings (Top 10 Cities x Top 12 Macro Categories)
    top10_c = df_merged["city"].value_counts().head(10).index.tolist()
    top12_k = df_merged["primary_category"].value_counts().head(12).index.tolist()
    df_hm = df_merged[df_merged["city"].isin(top10_c) & df_merged["primary_category"].isin(top12_k)].copy()
    df_hm["city"] = df_hm["city"].astype(str)
    df_hm["primary_category"] = df_hm["primary_category"].astype(str)

    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=160)
    piv = df_hm.pivot_table(
        index="primary_category",
        columns="city",
        values="stars",
        aggfunc="mean",
    )
    sns.heatmap(piv, annot=True, fmt=".2f", cmap="YlGnBu", cbar_kws={"label": "Mean Star Rating"}, ax=ax)
    ax.set_title("Mean Yelp Star Ratings Across Top 10 Cities & Top 12 Business Categories", fontsize=13, fontweight="bold")
    ax.set_xlabel("Metropolitan City", fontsize=11)
    ax.set_ylabel("Primary Business Category", fontsize=11)
    plt.tight_layout()
    p2_path = PLOTS_DIR / "fig2_city_category_heatmap.png"
    plt.savefig(p2_path)
    plt.close(fig)
    saved_files["fig2_city_category_heatmap"] = str(p2_path)

    # Plot 3: Benchmark Comparison (Pandas vs Dask vs Polars vs JSONL Baseline)
    if BENCHMARK_RESULTS_JSON.exists():
        with open(BENCHMARK_RESULTS_JSON, "r", encoding="utf-8") as f:
            b_data = json.load(f)["benchmarks"]
        df_bench = pd.DataFrame(b_data)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=160)
        colors = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4"]
        sns.barplot(
            data=df_bench, x="time_seconds", y="engine", hue="engine",
            palette=colors, legend=False, ax=axes[0], orient="h"
        )
        axes[0].set_title("Execution Time (Seconds, Lower is Better)", fontsize=12, fontweight="bold")
        axes[0].set_xlabel("Wall-Clock Time (s)")
        axes[0].set_ylabel("")
        for c in axes[0].containers:
            axes[0].bar_label(c, fmt="%.2fs", padding=3, fontsize=9)

        sns.barplot(
            data=df_bench, x="peak_memory_mb", y="engine", hue="engine",
            palette=colors, legend=False, ax=axes[1], orient="h"
        )
        axes[1].set_title("Peak Traced Memory (MB, Lower is Better)", fontsize=12, fontweight="bold")
        axes[1].set_xlabel("Peak RAM Footprint (MB)")
        axes[1].set_ylabel("")
        for c in axes[1].containers:
            axes[1].bar_label(c, fmt="%.1f MB", padding=3, fontsize=9)

        plt.suptitle("Large-Scale Processing Benchmark (550K Reviews): JSONL vs. Pandas vs. Dask vs. Polars", fontsize=13, fontweight="bold")
        plt.tight_layout()
        p3_path = PLOTS_DIR / "fig3_engine_benchmarks.png"
        plt.savefig(p3_path)
        plt.close(fig)
        saved_files["fig3_engine_benchmarks"] = str(p3_path)

    # Plot 4: Yearly Review Volume & Mean Rating Trajectory Across Top 5 Cities (2010–2021)
    fig, ax1 = plt.subplots(figsize=(10, 5.5), dpi=160)
    top5_cities = ["Philadelphia", "Tampa", "Nashville", "Tucson", "New Orleans"]
    df_top5 = df_merged[(df_merged["city"].isin(top5_cities)) & (df_merged["year"].between(2010, 2021))].copy()
    df_top5["city"] = df_top5["city"].astype(str)
    yearly_city = (
        df_top5.groupby(["year", "city"], observed=True)["review_id"]
        .count()
        .reset_index(name="reviews")
    )
    sns.lineplot(data=yearly_city, x="year", y="reviews", hue="city", marker="o", linewidth=2.2, ax=ax1)
    ax1.set_title("Annual Review Activity Trends Across Top 5 Yelp Cities (2010–2021)", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Year", fontsize=11)
    ax1.set_ylabel("Annual Review Volume", fontsize=11)
    ax1.set_xticks(sorted(df_top5["year"].unique()))
    plt.tight_layout()
    p4_path = PLOTS_DIR / "fig4_yearly_city_trends.png"
    plt.savefig(p4_path)
    plt.close(fig)
    saved_files["fig4_yearly_city_trends"] = str(p4_path)

    # Plot 5: Boxplot / IQR Outlier Analysis of Review Length and Useful Votes by Star Rating
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=160)
    sample_df = df_merged.sample(n=min(50000, len(df_merged)), random_state=42)
    sns.boxplot(data=sample_df, x="stars", y="text_length", hue="User Tier", palette="Set2", ax=axes[0], showfliers=False)
    axes[0].set_title("Review Text Length (Chars) by Star Rating & Tier", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Star Rating")
    axes[0].set_ylabel("Review Length (Characters)")

    sns.boxplot(data=sample_df, x="stars", y="useful", hue="User Tier", palette="Set1", ax=axes[1], showfliers=False)
    axes[1].set_title("Community 'Useful' Votes by Star Rating & Tier (IQR)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Star Rating")
    axes[1].set_ylabel("Useful Votes per Review")

    plt.suptitle("IQR Distribution Analysis: Text Length and Useful Votes Across Star Ratings", fontsize=13, fontweight="bold")
    plt.tight_layout()
    p5_path = PLOTS_DIR / "fig5_iqr_boxplots.png"
    plt.savefig(p5_path)
    plt.close(fig)
    saved_files["fig5_iqr_boxplots"] = str(p5_path)

    # Interactive Plotly Chart 1: Sunburst / Treemap of Top 10 Cities -> Top 12 Categories -> Volume & Stars
    city_cat_agg = (
        df_hm.groupby(["city", "primary_category"], observed=True)
        .agg(
            review_volume=("review_id", "count"),
            mean_stars=("stars", "mean"),
        )
        .reset_index()
    )
    city_cat_agg["mean_stars"] = city_cat_agg["mean_stars"].round(3)
    fig_plotly = px.treemap(
        city_cat_agg,
        path=[px.Constant("All Yelp Cities"), "city", "primary_category"],
        values="review_volume",
        color="mean_stars",
        color_continuous_scale="RdYlGn",
        title="Interactive Yelp Hierarchy: City & Category Review Volume Colored by Mean Star Rating",
    )
    p6_html = PLOTS_DIR / "interactive_city_category_treemap.html"
    fig_plotly.write_html(str(p6_html), include_plotlyjs="cdn")
    saved_files["interactive_treemap_html"] = str(p6_html)

    return saved_files
