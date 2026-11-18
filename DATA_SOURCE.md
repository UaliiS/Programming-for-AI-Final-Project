# Data source

The raw data is too large to upload (the official archive is about 4.3 GB; the extracted
subset used here is 878 MB), so this submission includes a link instead of the files.

- **Dataset:** Yelp Open Dataset, Yelp Inc. — https://business.yelp.com/data/resources/open-dataset/
- **Direct archive:** https://business.yelp.com/external-assets/files/Yelp-JSON.zip

**Getting the data.** `python main.py --auto` streams the archive and writes the extracted
files to `data/raw/`. It keeps:

| File | Rows kept |
|---|---|
| business.json | all 150,346 rows |
| review.json | every 9th line: 776,698 of 6,990,280 reviews |
| user.json | the real profiles of the 462,196 users who wrote those reviews |

`data/raw/_extraction_manifest.json` records exactly what was kept.

**Offline use.** If you download the archive yourself:
`YELP_ARCHIVE=/path/to/Yelp-JSON.zip python main.py --auto` (macOS / Linux) or
`$env:YELP_ARCHIVE="C:\path\to\Yelp-JSON.zip"; python main.py --auto` (Windows PowerShell)

**Included in this submission.** The processed results are small enough to include:

- `outputs/*.json`: audit, benchmark, hypothesis tests, model metrics
- `outputs/sql_results/*.csv`: results of the 8 SQL queries
- `outputs/aggregates/city_category_year.parquet`: final aggregates
- `outputs/plots/`: all figures, including the two interactive HTML figures

The Parquet files and the SQLite database (about 600 MB) are rebuilt by the pipeline.
