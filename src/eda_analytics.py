"""
EDA, hypothesis tests and visualisations (Cluster 8).

All conclusions are generated from the computed statistics (no hard-coded text about
which city / group is higher). Effect sizes are reported next to p-values because with
~750K reviews almost any difference is "significant".

Elite definition: Yelp's `elite` field lists the years in which a user had Elite status.
A review is an "elite review" when it was written in one of those years; "never elite"
users never had the badge. This avoids labelling a user's 2008 review as elite because
they became Elite in 2019.

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
import seaborn as sns
from scipy import stats

from src.config import (
    BENCHMARK_RESULTS_JSON,
    EDA_SUMMARY_JSON,
    HYPOTHESIS_RESULTS_JSON,
    PARQUET_BUSINESS,
    PARQUET_REVIEWS_SINGLE,
    PARQUET_USER,
    PLOTS_DIR,
)

ALPHA = 0.05
TIER_ELITE = "Elite (review in elite year)"
TIER_ELITE_OTHER = "Elite user, other year"
TIER_NEVER = "Never Elite"


def load_analysis_frame() -> pd.DataFrame:
    """Join cleaned reviews with users and businesses (only the columns needed for EDA)."""
    rev = pd.read_parquet(PARQUET_REVIEWS_SINGLE, columns=[
        "review_id", "user_id", "business_id", "stars", "useful", "funny", "cool", "text_length", "year", "month"])
    usr = pd.read_parquet(PARQUET_USER, columns=["user_id", "elite_years", "is_elite", "review_count", "fans"])
    usr = usr.rename(columns={"review_count": "user_review_count"})
    biz = pd.read_parquet(PARQUET_BUSINESS, columns=["business_id", "city", "state", "primary_category", "price_range", "is_open"])
    df = rev.merge(usr, on="user_id", how="inner").merge(biz, on="business_id", how="inner")
    df["city"] = df["city"].astype(str)
    df["primary_category"] = df["primary_category"].astype(str)

    in_elite_year = [
        (str(y) in ys.split(",")) if isinstance(ys, str) and ys else False
        for y, ys in zip(df["year"].to_numpy(), df["elite_years"].to_numpy())
    ]
    df["tier"] = np.where(df["is_elite"] == 0, TIER_NEVER, np.where(in_elite_year, TIER_ELITE, TIER_ELITE_OTHER))
    return df


def signed(x: float, digits: int = 2) -> str:
    """Format a number with a typographic minus sign (−0.22)."""
    return f"{x:.{digits}f}".replace("-", "−")


def fmt_p(p: float) -> str:
    """Readable p-value (scipy returns 0.0 when p is below double precision)."""
    return "< 1e-300" if p == 0 else (f"= {p:.3g}" if p >= 1e-4 else f"= {p:.1e}")


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float((a.mean() - b.mean()) / pooled)


def effect_label(d: float) -> str:
    d = abs(d)
    if d < 0.2:
        return "negligible"
    if d < 0.5:
        return "small"
    if d < 0.8:
        return "medium"
    return "large"


def _json_safe(obj: Any) -> Any:
    """Replace NaN/inf (not valid JSON) with None, recursively."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def describe_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """Descriptive statistics used in the report."""
    num = df[["stars", "useful", "funny", "cool", "text_length", "user_review_count", "fans"]]
    desc = num.describe(percentiles=[0.25, 0.5, 0.75, 0.99]).T.round(3)
    star_dist = (df["stars"].value_counts(normalize=True).sort_index() * 100).round(2)
    top_cities = df["city"].value_counts().head(10)
    summary = {
        "rows": int(len(df)),
        "businesses": int(df["business_id"].nunique()),
        "users": int(df["user_id"].nunique()),
        "cities": int(df["city"].nunique()),
        "states": int(df["state"].nunique()),
        "years": [int(df["year"].min()), int(df["year"].max())],
        "feature_types": {
            "numeric": ["stars", "useful", "funny", "cool", "text_length", "user_review_count", "fans", "price_range"],
            "categorical": ["city", "state", "primary_category", "tier", "is_open"],
            "temporal": ["year", "month"],
            "text": ["text (Parquet only)"],
        },
        "describe": desc.to_dict(orient="index"),
        "star_distribution_percent": {int(k): float(v) for k, v in star_dist.items()},
        "top_cities_by_reviews": {k: int(v) for k, v in top_cities.items()},
        "tier_counts": {k: int(v) for k, v in df["tier"].value_counts().items()},
        "spearman_correlation": num.corr(method="spearman").round(3).to_dict(),
    }
    summary = _json_safe(summary)
    with open(EDA_SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def run_statistical_hypotheses(df: pd.DataFrame = None) -> Dict[str, Any]:
    """Test three hypotheses; return and save results with effect sizes and generated conclusions."""
    if df is None:
        df = load_analysis_frame()
    describe_dataset(df)

    # ---------------- H2: Elite reviews vs never-Elite reviews ----------------
    e = df.loc[df["tier"] == TIER_ELITE, "stars"].to_numpy(dtype=float)
    n = df.loc[df["tier"] == TIER_NEVER, "stars"].to_numpy(dtype=float)
    t_stat, p_t = stats.ttest_ind(e, n, equal_var=False)
    lev, p_lev = stats.levene(e, n)
    d = cohens_d(e, n)
    pct = lambda arr, v: round(float(100 * np.mean(arr == v)), 2)
    # robustness: one value per user (removes dependence between reviews of the same user)
    per_user = df[df["tier"].isin([TIER_ELITE, TIER_NEVER])].groupby(["user_id", "tier"])["stars"].mean().reset_index()
    ue = per_user.loc[per_user["tier"] == TIER_ELITE, "stars"].to_numpy()
    un = per_user.loc[per_user["tier"] == TIER_NEVER, "stars"].to_numpy()
    u_stat, p_u = stats.mannwhitneyu(ue, un, alternative="two-sided")
    higher = "higher" if e.mean() > n.mean() else "lower"
    h1 = {
        "title": "H2: Elite reviewers rate differently from never-Elite users",
        "null_hypothesis": "Reviews written by Elite users (in their Elite years) and by never-Elite users have the same mean and variance of stars.",
        "test": "Welch t-test (means), Levene (variances), Mann–Whitney U on per-user means (robustness)",
        "n_elite_reviews": int(len(e)), "n_never_elite_reviews": int(len(n)),
        "elite_mean_stars": round(float(e.mean()), 4), "never_elite_mean_stars": round(float(n.mean()), 4),
        "elite_std_stars": round(float(e.std(ddof=1)), 4), "never_elite_std_stars": round(float(n.std(ddof=1)), 4),
        "elite_pct_1_star": pct(e, 1), "never_elite_pct_1_star": pct(n, 1),
        "elite_pct_5_star": pct(e, 5), "never_elite_pct_5_star": pct(n, 5),
        "welch_t": round(float(t_stat), 3), "welch_p": float(p_t),
        "levene_W": round(float(lev), 3), "levene_p": float(p_lev),
        "cohens_d": round(d, 4), "effect_size": effect_label(d),
        "per_user_mann_whitney_p": float(p_u), "n_users_elite": int(len(ue)), "n_users_never": int(len(un)),
        "decision": "Reject H0" if p_t < ALPHA else "Fail to reject H0",
    }
    spread = "less" if h1["elite_std_stars"] < h1["never_elite_std_stars"] else "more"
    if (p_u < ALPHA) and (p_t < ALPHA):
        robust = "agrees, so the effect is not driven by a few prolific users"
    elif (p_u >= ALPHA) and (p_t >= ALPHA):
        robust = "also finds no difference"
    else:
        robust = "disagrees, so the review-level result should be read with care"
    h1["conclusion"] = (
        f"Elite reviews average {h1['elite_mean_stars']:.2f} stars vs {h1['never_elite_mean_stars']:.2f} for never-Elite users "
        f"({higher}; Welch p {fmt_p(p_t)}; Cohen’s d = {d:.2f}, a {effect_label(d)} effect). Elite ratings are {spread} spread out "
        f"(SD {h1['elite_std_stars']:.2f} vs {h1['never_elite_std_stars']:.2f}; Levene p {fmt_p(p_lev)}): {h1['elite_pct_1_star']}% are 1-star vs "
        f"{h1['never_elite_pct_1_star']}%, and {h1['elite_pct_5_star']}% are 5-star vs {h1['never_elite_pct_5_star']}%. "
        f"The per-user test (one mean per user, p {fmt_p(p_u)}) {robust}."
    )

    # ---------------- H1: city and category heterogeneity ----------------
    top_cities = df["city"].value_counts().head(10).index.tolist()
    sub = df[df["city"].isin(top_cities)]
    groups = [g["stars"].to_numpy(dtype=float) for _, g in sub.groupby("city")]
    f_city, p_city = stats.f_oneway(*groups)
    h_city, p_kw = stats.kruskal(*groups)
    grand = sub["stars"].mean()
    ss_between = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    ss_total = float(((sub["stars"] - grand) ** 2).sum())
    eta2_city = ss_between / ss_total
    city_means = sub.groupby("city")["stars"].agg(["count", "mean"]).sort_values("mean", ascending=False)

    top_cats = df["primary_category"].value_counts().head(12).index.tolist()
    sub_c = df[df["primary_category"].isin(top_cats)]
    cgroups = [g["stars"].to_numpy(dtype=float) for _, g in sub_c.groupby("primary_category")]
    f_cat, p_cat = stats.f_oneway(*cgroups)
    grand_c = sub_c["stars"].mean()
    eta2_cat = sum(len(g) * (g.mean() - grand_c) ** 2 for g in cgroups) / float(((sub_c["stars"] - grand_c) ** 2).sum())
    cat_means = sub_c.groupby("primary_category")["stars"].mean().sort_values(ascending=False)
    h2 = {
        "title": "H1: Average ratings differ between cities and between categories",
        "null_hypothesis": "Mean review stars are equal across the 10 largest cities (and across the 12 largest categories).",
        "test": "One-way ANOVA + Kruskal–Wallis; effect size η²",
        "cities_tested": top_cities,
        "city_anova_F": round(float(f_city), 3), "city_anova_p": float(p_city),
        "city_kruskal_H": round(float(h_city), 3), "city_kruskal_p": float(p_kw),
        "city_eta_squared": round(float(eta2_city), 5),
        "city_means": {c: round(float(m), 4) for c, m in city_means["mean"].items()},
        "category_anova_F": round(float(f_cat), 3), "category_anova_p": float(p_cat),
        "category_eta_squared": round(float(eta2_cat), 5),
        "category_means": {c: round(float(m), 4) for c, m in cat_means.items()},
        "decision": "Reject H0" if p_city < ALPHA else "Fail to reject H0",
    }
    h2["conclusion"] = (
        f"Among the 10 largest cities, {city_means.index[0]} has the highest mean ({city_means['mean'].iloc[0]:.2f}) and "
        f"{city_means.index[-1]} the lowest ({city_means['mean'].iloc[-1]:.2f}). The difference is "
        f"{'statistically significant' if p_city < ALPHA else 'not statistically significant'} "
        f"(ANOVA p {fmt_p(p_city)}, Kruskal–Wallis p {fmt_p(p_kw)}); city explains {100 * eta2_city:.2f}% of rating variance. "
        f"Category {'explains more' if eta2_cat > eta2_city else 'explains less'} ({100 * eta2_cat:.2f}% of variance): {cat_means.index[0]} is rated highest "
        f"({cat_means.iloc[0]:.2f}) and {cat_means.index[-1]} lowest ({cat_means.iloc[-1]:.2f})."
    )

    # ---------------- H3: length vs usefulness ----------------
    rho, p_rho = stats.spearmanr(df["text_length"], df["useful"])
    le = df.loc[df["tier"] == TIER_ELITE, "text_length"].to_numpy()
    ln = df.loc[df["tier"] == TIER_NEVER, "text_length"].to_numpy()
    u_len, p_len = stats.mannwhitneyu(le, ln, alternative="greater")
    rho_star_len, p_sl = stats.spearmanr(df["stars"], df["text_length"])
    h3 = {
        "title": "H3: Longer reviews receive more “useful” votes",
        "null_hypothesis": "There is no monotonic association between review length and useful votes.",
        "test": "Spearman rank correlation; Mann–Whitney U (Elite vs never-Elite length)",
        "spearman_rho_length_useful": round(float(rho), 4), "spearman_p": float(p_rho),
        "spearman_rho_stars_length": round(float(rho_star_len), 4), "spearman_p_stars_length": float(p_sl),
        "elite_median_length": float(np.median(le)), "never_elite_median_length": float(np.median(ln)),
        "mann_whitney_p_length": float(p_len),
        "decision": "Reject H0" if p_rho < ALPHA else "Fail to reject H0",
    }
    strength = "weak" if abs(rho) < 0.3 else ("moderate" if abs(rho) < 0.5 else "strong")
    direction = "positive" if rho > 0 else "negative"
    longer = "longer" if np.median(le) > np.median(ln) else "not longer"
    star_note = ("Lower-star reviews tend to be longer: complaints are written in more detail"
                 if rho_star_len < 0 else "Higher-star reviews tend to be longer")
    h3["conclusion"] = (
        f"Review length and useful votes have a {strength} {direction} rank correlation (Spearman ρ = {rho:.2f}, p {fmt_p(p_rho)}). "
        f"Elite reviews are {longer} than never-Elite ones (median {np.median(le):.0f} vs {np.median(ln):.0f} characters; "
        f"one-sided Mann–Whitney p {fmt_p(p_len)}). {star_note} (Spearman ρ between stars and length = {signed(rho_star_len)})."
    )

    # Numbered in the order of the research questions: RQ1 cities/categories, RQ2 Elite, RQ3 length.
    out = {"H1_city_category": h2, "H2_elite_bias": h1, "H3_length_usefulness": h3}
    out = _json_safe(out)
    with open(HYPOTHESIS_RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return out


def generate_all_visualizations(df: pd.DataFrame = None) -> Dict[str, str]:
    """Create the static figures (PNG) and the interactive Plotly figures (HTML)."""
    if df is None:
        df = load_analysis_frame()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    saved: Dict[str, str] = {}
    tier_order = [TIER_ELITE, TIER_ELITE_OTHER, TIER_NEVER]
    tier_colors = {TIER_ELITE: "#c0392b", TIER_ELITE_OTHER: "#e59866", TIER_NEVER: "#2e86c1"}

    # Fig 1 — star distribution by reviewer tier
    dist = df.groupby(["tier", "stars"]).size().reset_index(name="n")
    dist["pct"] = 100 * dist["n"] / dist.groupby("tier")["n"].transform("sum")
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=150)
    sns.barplot(data=dist, x="stars", y="pct", hue="tier", hue_order=tier_order, palette=tier_colors, ax=ax)
    ax.set_title("Star rating distribution by reviewer tier", fontsize=13, fontweight="bold")
    ax.set_xlabel("Review stars")
    ax.set_ylabel("Share of the tier's reviews (%)")
    ax.legend(title="Reviewer tier")
    for c in ax.containers:
        ax.bar_label(c, fmt="%.0f%%", fontsize=8, padding=1)
    fig.tight_layout()
    p = PLOTS_DIR / "fig1_elite_vs_regular_ratings.png"
    fig.savefig(p); plt.close(fig); saved["fig1"] = str(p)

    # Fig 2 — city x category heatmap
    top10 = df["city"].value_counts().head(10).index
    top12 = df["primary_category"].value_counts().head(12).index
    hm = df[df["city"].isin(top10) & df["primary_category"].isin(top12)]
    piv = hm.pivot_table(index="primary_category", columns="city", values="stars", aggfunc="mean")
    cnt = hm.pivot_table(index="primary_category", columns="city", values="stars", aggfunc="size")
    piv = piv.where(cnt >= 30)  # hide cells with too few reviews to be reliable
    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=150)
    sns.heatmap(piv, annot=True, fmt=".2f", cmap="RdYlGn", vmin=2.8, vmax=4.6,
                cbar_kws={"label": "Mean review stars"}, ax=ax)
    ax.set_title("Mean review stars: 10 largest cities × 12 largest categories (cells with <30 reviews hidden)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("City"); ax.set_ylabel("Primary category")
    fig.tight_layout()
    p = PLOTS_DIR / "fig2_city_category_heatmap.png"
    fig.savefig(p); plt.close(fig); saved["fig2"] = str(p)

    # Fig 3 — engine benchmark
    if BENCHMARK_RESULTS_JSON.exists():
        bench = pd.DataFrame(json.loads(BENCHMARK_RESULTS_JSON.read_text())["benchmarks"])
        bench["label"] = bench["engine"].str.replace(r"^(Baseline|Approach \d): ", "", regex=True)
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), dpi=150)
        colors = ["#7f8c8d", "#2e86c1", "#8e44ad", "#27ae60"]
        axes[0].barh(bench["label"], bench["time_seconds_median"], color=colors)
        axes[0].set_xscale("log")
        axes[0].set_title("Run time (s, log scale; lower is better)", fontweight="bold")
        for i, v in enumerate(bench["time_seconds_median"]):
            axes[0].text(v, i, f" {v:.2f}s", va="center", fontsize=9)
        axes[1].barh(bench["label"], bench["peak_rss_mb"], color=colors)
        axes[1].set_title("Peak process memory, RSS (MB; lower is better)", fontweight="bold")
        for i, v in enumerate(bench["peak_rss_mb"]):
            axes[1].text(v, i, f" {v:.0f}", va="center", fontsize=9)
        for a in axes:
            a.invert_yaxis()
        n_rows = int(bench["total_rows"].iloc[0])
        fig.suptitle(f"Same aggregation + join task on {n_rows:,} cleaned reviews", fontsize=13, fontweight="bold")
        fig.tight_layout()
        p = PLOTS_DIR / "fig3_engine_benchmarks.png"
        fig.savefig(p); plt.close(fig); saved["fig3"] = str(p)

    # Fig 4 — yearly volume and mean stars for the 5 largest cities
    top5 = df["city"].value_counts().head(5).index.tolist()
    yc = (df[df["city"].isin(top5) & (df["year"] <= 2021)]
          .groupby(["year", "city"]).agg(reviews=("stars", "size"), mean_stars=("stars", "mean")).reset_index())
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
    city_colors = dict(zip(top5, sns.color_palette("deep", len(top5))))  # same colour per city in both panels
    sns.lineplot(data=yc, x="year", y="reviews", hue="city", hue_order=top5, palette=city_colors, marker="o", ax=axes[0])
    axes[0].set_title("Reviews per year (sample), 5 largest cities", fontweight="bold")
    axes[0].set_xlabel("Year"); axes[0].set_ylabel("Reviews in sample")
    sns.lineplot(data=yc[yc["reviews"] >= 100], x="year", y="mean_stars", hue="city", hue_order=top5,
                 palette=city_colors, marker="o", ax=axes[1], legend=False)
    axes[1].set_title("Mean stars per year (years with ≥100 reviews)", fontweight="bold")
    axes[1].set_xlabel("Year"); axes[1].set_ylabel("Mean stars")
    fig.tight_layout()
    p = PLOTS_DIR / "fig4_yearly_city_trends.png"
    fig.savefig(p); plt.close(fig); saved["fig4"] = str(p)

    # Fig 5 — IQR / boxplots
    sample = df.sample(n=min(60_000, len(df)), random_state=42)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
    sns.boxplot(data=sample, x="stars", y="text_length", hue="tier", hue_order=tier_order,
                palette=tier_colors, ax=axes[0], showfliers=False)
    axes[0].set_title("Review length by stars (whiskers = 1.5×IQR, outliers hidden)", fontweight="bold")
    axes[0].set_xlabel("Review stars"); axes[0].set_ylabel("Length (characters)")
    sns.boxplot(data=sample, x="stars", y="useful", hue="tier", hue_order=tier_order,
                palette=tier_colors, ax=axes[1], showfliers=False)
    axes[1].set_title("Useful votes by stars (whiskers = 1.5×IQR)", fontweight="bold")
    axes[1].set_xlabel("Review stars"); axes[1].set_ylabel("Useful votes")
    axes[1].get_legend().remove()
    fig.tight_layout()
    p = PLOTS_DIR / "fig5_iqr_boxplots.png"
    fig.savefig(p); plt.close(fig); saved["fig5"] = str(p)

    # Fig 7 — Spearman correlation heatmap
    cols = ["stars", "useful", "funny", "cool", "text_length", "user_review_count", "fans", "price_range"]
    corr = df[cols].astype(float).corr(method="spearman")
    fig, ax = plt.subplots(figsize=(8, 6.5), dpi=150)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax, square=True)
    ax.set_title("Spearman correlation of numeric features", fontweight="bold")
    fig.tight_layout()
    p = PLOTS_DIR / "fig7_correlation_heatmap.png"
    fig.savefig(p); plt.close(fig); saved["fig7"] = str(p)

    # Interactive 1 — treemap city -> category
    agg = (hm.groupby(["city", "primary_category"]).agg(reviews=("stars", "size"), mean_stars=("stars", "mean"))
           .reset_index())
    agg["mean_stars"] = agg["mean_stars"].round(3)
    fig_t = px.treemap(agg, path=[px.Constant("10 largest cities"), "city", "primary_category"],
                       values="reviews", color="mean_stars", color_continuous_scale="RdYlGn",
                       range_color=(3.0, 4.5),
                       title="Review volume (size) and mean stars (colour) by city and category")
    p = PLOTS_DIR / "interactive_city_category_treemap.html"
    fig_t.write_html(str(p), include_plotlyjs="cdn"); saved["interactive_treemap"] = str(p)

    # Interactive 2 — monthly volume with 12-month rolling mean
    monthly = df.groupby(["year", "month"]).agg(reviews=("stars", "size"), mean_stars=("stars", "mean")).reset_index()
    monthly["date"] = pd.to_datetime(dict(year=monthly["year"], month=monthly["month"], day=1))
    monthly = monthly.sort_values("date")
    monthly["reviews_12m_avg"] = monthly["reviews"].rolling(12, min_periods=1).mean()
    long = monthly.melt(id_vars="date", value_vars=["reviews", "reviews_12m_avg"], var_name="series", value_name="value")
    fig_m = px.line(long, x="date", y="value", color="series",
                    title="Reviews per month in the sample with 12-month rolling mean",
                    labels={"value": "Reviews", "date": "Month"})
    p = PLOTS_DIR / "interactive_monthly_reviews.html"
    fig_m.write_html(str(p), include_plotlyjs="cdn"); saved["interactive_monthly"] = str(p)

    return saved
