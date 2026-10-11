# Day-type structure and forecasting of daily demand (June 2026)

A preregistered, fully reproducible data-science project on `data/raw/daily-demand-series.csv`
(30 consecutive days: `demand`, `marketing_event`, `holiday`).

| Task | Proof artifact in this repo |
|---|---|
| 1. Reproducible research question & preregistration | `prereg/PREREGISTRATION.md` (git tag **`prereg-v1`**), `requirements.in` + `requirements.lock`, raw/interim/processed/notebooks/reports layout, `docs/OUTPUT_CONTRACTS.md`, synthetic tests `tests/`, `notebooks/01_preregistration_pipeline_check.ipynb` |
| 2. Statistical analysis & hypothesis testing | `notebooks/02_statistical_tests_anova.ipynb` (Shapiro-Wilk, KS/Lilliefors, Welch t, Mann-Whitney U, one- and two-way ANOVA, Tukey HSD, CIs, robustness) |
| 3. PCA & clustering | `notebooks/03_pca_clustering.ipynb` (scree, elbow, silhouette, K-Means, DBSCAN, Ward, Seaborn/Plotly/3D) |
| 4. Time-series forecasting | `notebooks/04_timeseries_forecasting.ipynb` (decomposition, ADF, SARIMA, MAPE/RMSE, 30-day forecast) |
| 5. Capstone & whitepaper | `reports/whitepaper.pdf` (9 pages), `notebooks/05_capstone_predictive_modelling.ipynb`, this repository |

## Proof of order (plan before analysis)
```
git log --reverse --oneline --decorate     # prereg-v1 precedes every analysis commit
git show prereg-v1 --stat
```
Post-lock deviations are logged in the preregistration's "Deviations" table.

## Reproduce
```
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.lock          # Python 3.12
make test         # 18 data-blind tests on synthetic fixtures
make notebooks    # executes notebooks 01-05 on the real data -> reports/figures, reports/tables
make report       # builds reports/whitepaper.pdf
```
All stochastic steps use seed `20260601`. The raw file's SHA-256 is recorded in the preregistration and re-checked in notebook 01.

## Headline results (see whitepaper for detail and caveats)
* Event days average 156.6 vs 133.9 on regular days (difference 22.7, Welch 95% CI 12.4 to 32.9, Hedges g 2.24, p < 0.001); robust to permutation, trend-adjusted OLS, log scale, leave-one-week-out.
* **Every event day is a Fri/Sat and every holiday a Sunday**, so this is an association bundled with day-of-week, not a causal marketing effect.
* K-Means (k=4, silhouette 0.45), DBSCAN and Ward agree with each other and follow the weekly rhythm.
* SARIMA(1,0,0)x(0,1,0,7): 7-day holdout RMSE 1.44 / MAPE 0.88% vs seasonal-naive RMSE 6.02. Rolling-origin RMSE: SARIMA 1.24, Ridge 1.75, seasonal naive 6.11.
* Limits: n = 30, single and unusually regular series.

## Layout
```
prereg/            locked plan          data/raw|interim|processed
src/rq/            pipeline package     notebooks/         executed notebooks 01-05
tests/             synthetic tests      reports/           whitepaper.pdf, figures/, tables/
docs/              output contracts     scripts/           notebook + report builders, lock generator
```
"# minakshi-demand-research-capstone" 
"# minakshi-demand-research-capstone" 
