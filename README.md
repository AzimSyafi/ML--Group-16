# Household Electricity Segmentation Dashboard

TEB3123 Machine Learning, Group 16. A Streamlit dashboard for clustering London household smart meter data to find efficiency and carbon-saving opportunities (UTP Energy Security, SDG 7.3).

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

The `data/` folder ships with **synthetic demo data**, so the dashboard opens straight away. A yellow banner says so on every page until you replace it with the real outputs.

## Use the real data

1. Download **Smart meters in London** from Kaggle: https://www.kaggle.com/datasets/jeanmidev/smart-meters-in-london and unzip it. You need `hhblock_dataset/` and `informations_households.csv`.
2. Build the dashboard data:

```bash
python pipeline/prepare_data.py kaggle --root path/to/smart-meters-in-london
```

If you use the original London Datastore CSVs instead (https://data.london.gov.uk/dataset/smartmeter-energy-use-data-in-london-households):

```bash
python pipeline/prepare_data.py datastore --csv "path/to/LCL-June2015v2_*.csv"
```

Useful options: `--year 2013`, `--coverage 0.9`, `--reps 10` (subsample repeats for ARI stability), and `--model "K-Means" --k 5` to override the automatic choice.

The full run on about 5,500 households takes a while because each of the 27 model/k combinations is refitted 10 times for stability. Use `--reps 3` for a quick first pass.

## What the pipeline does

| Step | Implementation |
|---|---|
| Cleaning | Readings coerced to numeric ("Null" becomes missing), negatives removed, duplicate household-days dropped, year 2013 only |
| Coverage | Households under 90% of half hours recorded are excluded |
| Gaps | Gaps of up to 2 hours (4 half hours) are linearly interpolated; days still incomplete are left out of averages |
| Features | Mean daily kWh, five time-of-day shares, baseload (01:00 to 05:00), peak-to-average (17:00 to 21:00), weekend-to-weekday ratio, winter-to-summer ratio |
| Outliers | Near-zero homes (under 0.5 kWh/day) and 3×IQR extremes on daily use and baseload are set aside in `data/outliers.csv` |
| Scaling | log1p on the two kWh features, then standardised; PCA keeps 95% of variance |
| Models | K-Means (k-means++), Agglomerative Ward, GMM (full or diag covariance by BIC), k = 2 to 10 |
| Evaluation | Silhouette, Davies-Bouldin, Calinski-Harabasz, elbow, BIC, and ARI across 80% subsamples |
| Selection | Best combined rank for k = 3 to 8 with every cluster at least 3% of homes |
| Profiling | Tariff type and Acorn group used only after clustering |

## Editing segment names and actions

Segments are named automatically from their feature profile. To rename a segment, change its recommended action or default reduction %, edit `data/segment_labels.csv` and refresh the dashboard. You do not need to rerun the pipeline.

## Emission factors

The sidebar switches between the DESNZ 2025 UK grid factor (0.177 kg CO2e/kWh, the edition cited in the proposal) and DESNZ 2026 (0.131), or a custom value.

## Project layout

```
app.py                     navigation, filters, data loading
dashboard/views.py         the six pages
dashboard/style.py         colours, chart template, UI helpers
pipeline/features.py       per-household feature engineering
pipeline/build.py          clustering, evaluation, segment naming, outputs
pipeline/prepare_data.py   real data entry point (Kaggle or Datastore)
pipeline/make_demo_data.py synthetic demo data through the same pipeline
data/                      dashboard inputs (replace with real outputs)
```

## Deploy

Push the folder to GitHub, including `data/`, and deploy it free on Streamlit Community Cloud (https://streamlit.io/cloud) with `app.py` as the entry point.
