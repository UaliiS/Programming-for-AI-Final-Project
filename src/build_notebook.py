"""
Builds Yelp_Reviews_At_Scale_Analysis.ipynb and (optionally) executes it, so every
table and chart in the notebook is produced by running code, not pasted in.

Usage:  python -m src.build_notebook            # build + execute (needs the pipeline outputs)
        python -m src.build_notebook --no-run   # build only

Authors: Uali Sembay & Arystan Aldiyev (AAI-2605)
"""

import sys

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

from src.config import BASE_DIR

NB_PATH = BASE_DIR / "Yelp_Reviews_At_Scale_Analysis.ipynb"


def build_cells():
    md, code = new_markdown_cell, new_code_cell
    return [
        md("# Yelp Reviews at Scale: Ratings, Categories and Cities\n"
           "**Final project — Topic #44, Cluster 8 (Large-Scale Data Processing)**  \n"
           "Course: Programming for Data Analysis and Database · MSc Applied AI  \n"
           "Authors: **Uali Sembay**, **Arystan Aldiyev** (AAI-2605)\n\n"
           "This notebook is a guided tour of the project. The heavy work is done by the modules in `src/`; "
           "run `python main.py --auto` once (≈30–60 min, mostly the streamed download) and then run this notebook top to bottom. "
           "Every table below is computed from the files the pipeline produced."),
        md("### Google Colab\nIn Colab, set `REPO_URL` to the GitHub repository and run the next cell; "
           "it clones the project, installs the requirements and runs the full pipeline (the download needs ~20–40 min)."),
        code("import sys, os, subprocess\n"
             "REPO_URL = ''  # e.g. 'https://github.com/<user>/yelp-reviews-at-scale.git'\n"
             "if 'google.colab' in sys.modules and REPO_URL:\n"
             "    if not os.path.exists('yelp-reviews-at-scale'):\n"
             "        subprocess.run(['git', 'clone', REPO_URL, 'yelp-reviews-at-scale'], check=True)\n"
             "    os.chdir('yelp-reviews-at-scale')\n"
             "    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements.txt'], check=True)\n"
             "    subprocess.run([sys.executable, 'main.py', '--auto'], check=True)"),
        code("import json\nfrom pathlib import Path\nimport pandas as pd\n"
             "from IPython.display import Image, display, Markdown\n"
             "pd.set_option('display.max_colwidth', 120)\n"
             "from src.config import *\n"
             "load = lambda p: json.loads(Path(p).read_text(encoding='utf-8'))"),

        md("## 1. Data: streamed download, sampling and data-quality audit"),
        code("manifest = load(RAW_MANIFEST_JSON)\n"
             "pd.Series({k: manifest[k] for k in ['source', 'review_sampling', 'business_rows', 'review_lines_seen',\n"
             "           'review_rows_kept', 'user_lines_seen', 'user_rows_kept', 'total_raw_mb']}).to_frame('value')"),
        code("audit = load(AUDIT_REPORT_JSON)\n"
             "display(pd.Series(audit['reviews_audit']).drop('missing_values').to_frame('reviews'))\n"
             "display(pd.DataFrame(audit['iqr_outliers']).T)\n"
             "for rule, why in audit['cleaning_rules_justification'].items():\n"
             "    print(f'- {rule}: {why}')"),
        code("mem = audit['reviews_memory_optimization']\n"
             "pd.DataFrame({\n"
             "  'before (MB)': [mem['raw_chunks_in_memory_mb'], mem['raw_without_text_mb'], mem['raw_jsonl_disk_mb']],\n"
             "  'after (MB)': [mem['optimized_chunks_in_memory_mb'], mem['optimized_without_text_mb'], mem['cleaned_parquet_disk_mb']]},\n"
             "  index=['in memory, all columns', 'in memory, without text', 'on disk (JSONL -> Parquet)'])"),
        code("pd.DataFrame({'businesses': {k: v for k, v in audit['businesses_audit'].items() if not isinstance(v, dict)},\n"
             "              'users': {k: v for k, v in audit['users_audit'].items() if not isinstance(v, dict)}})"),

        md("## 2. Same task, four engines (Requirement 9)"),
        code("bench = load(BENCHMARK_RESULTS_JSON)\n"
             "print(bench['method'])\nprint('Identical results across engines:', bench['all_results_identical'])\n"
             "pd.DataFrame(bench['benchmarks']).set_index('engine')"),
        code("display(Image(str(PLOTS_DIR / 'fig3_engine_benchmarks.png'), width=900))"),
        code("import polars as pl\n"
             "# Lazy query example: nothing is read until .collect(); only 3 columns are scanned\n"
             "(pl.scan_parquet(PARQUET_REVIEWS_SINGLE)\n"
             "   .group_by('year').agg(pl.len().alias('reviews'), pl.col('stars').mean().round(3).alias('mean_stars'))\n"
             "   .sort('year').collect())"),

        md("## 3. SQLite database: ER diagram and 8 SQL queries"),
        code("display(Image(str(ER_DIAGRAM_PNG), width=900))\nload(DB_SUMMARY_JSON)"),
        code("from src.database import SQL_QUERIES, execute_all_sql_queries\n"
             "sql = execute_all_sql_queries()\n"
             "for name, df in sql.items():\n"
             "    display(Markdown(f'**{name}** — {len(df)} rows'))\n"
             "    display(df.head(8))"),

        md("## 4. EDA and hypothesis tests"),
        code("from src.eda_analytics import load_analysis_frame\n"
             "frame = load_analysis_frame()\n"
             "eda = load(EDA_SUMMARY_JSON)\n"
             "print(f\"{eda['rows']:,} reviews · {eda['businesses']:,} businesses · {eda['users']:,} users · years {eda['years']}\")\n"
             "display(pd.DataFrame(eda['describe']).T)\n"
             "display(frame.isna().sum().to_frame('missing values'))\n"
             "pd.Series(eda['star_distribution_percent'], name='% of reviews').to_frame()"),
        code("for f in ['fig1_elite_vs_regular_ratings.png', 'fig2_city_category_heatmap.png', 'fig4_yearly_city_trends.png',\n"
             "          'fig5_iqr_boxplots.png', 'fig7_correlation_heatmap.png']:\n"
             "    display(Image(str(PLOTS_DIR / f), width=900))"),
        code("import plotly.express as px\n"
             "monthly = frame.groupby(['year', 'month']).size().reset_index(name='reviews')\n"
             "monthly['date'] = pd.to_datetime(dict(year=monthly.year, month=monthly.month, day=1))\n"
             "px.line(monthly.sort_values('date'), x='date', y='reviews', title='Interactive: reviews per month (sample)')"),
        code("hyp = load(HYPOTHESIS_RESULTS_JSON)\n"
             "for h in hyp.values():\n"
             "    display(Markdown(f\"**{h['title']}** — {h['decision']}  \\nH0: {h['null_hypothesis']}  \\nTest: {h['test']}  \\n{h['conclusion']}\"))"),

        md("## 5. Star-rating prediction from review text"),
        code("ml = load(MODEL_METRICS_JSON)\n"
             "print(ml['split_strategy']); print('Selection:', ml['selection_rule']); print('Best:', ml['best_model'])\n"
             "pd.json_normalize(ml['models_comparison']).set_index('model_name')"),
        code("display(Image(str(PLOTS_DIR / 'fig6_confusion_matrix.png'), width=550))\n"
             "print('Share of errors that are only one star off:', ml['errors_adjacent_share'])\n"
             "pd.DataFrame(ml['error_analysis_samples'])"),
        code("from src.ml_models import predict_custom_review\n"
             "for t in ['The tacos were fresh, the staff friendly, and the salsa bar is amazing.',\n"
             "          'Waited 50 minutes, food was cold and the manager did not care.',\n"
             "          'It was fine. Nothing special, prices a bit high for what you get.']:\n"
             "    r = predict_custom_review(t)\n"
             "    print(r['predicted_stars'], '★ |', t)"),
        md("## 6. Conclusions\nSee the written report (`docs/FINAL_PROJECT_REPORT.docx`) for the discussion, limitations and references."),
    ]


def build(run: bool = True) -> str:
    nb = new_notebook(cells=build_cells())
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    if run:
        from nbclient import NotebookClient
        NotebookClient(nb, timeout=1800, kernel_name="python3", resources={"metadata": {"path": str(BASE_DIR)}}).execute()
    nbformat.write(nb, str(NB_PATH))
    return str(NB_PATH)


if __name__ == "__main__":
    print(build(run="--no-run" not in sys.argv))
