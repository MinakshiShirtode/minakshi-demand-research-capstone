# Output contracts (defined before the real outcome was analysed)

`rq.pipeline.run_pipeline(df)` accepts any dataframe with columns
`date, demand, marketing_event, holiday` (consecutive daily dates, binary flags) and must return
the following. `rq.pipeline.check_contract` and `tests/test_pipeline_synthetic.py` enforce them.

| Artifact | Contract |
|---|---|
| `calendar` | input + `dow` (0-6), `t` (0..n-1), `week` (>=1), `day_type` in {Event, Holiday, Regular, Both} |
| `normality` | one row per Event/Holiday/Regular; columns `group, n, shapiro_W, shapiro_p, ks_D, ks_p, lilliefors_D, lilliefors_p, normal_by_shapiro` |
| `primary` | dict with `n_a, n_b, mean_a, mean_b, diff, welch_t/df/p, welch_ci_low/high, hedges_g (+CI), mwu_U/p, rank_biserial, hl_shift (+CI), perm_p, primary_test, primary_p, reject_null`; all p in [0,1]; Welch CI contains `diff` |
| `oneway`, `tukey` | ANOVA dict (F, p, eta2, omega2, Brown-Forsythe p, Kruskal p); Tukey table with exactly 3 pairs, adjusted p in [0,1] |
| `twoway` | table with rows `C(week)`, `C(day_type)`, `C(week):C(day_type)`, `Residual` |
| `robustness` | permutation p, OLS(+trend) coef/CI/p, log-scale Welch p; `leave_one_week_out` table |
| `pca` | explained-variance ratios sum to 1; cumulative non-decreasing; `n_keep` = smallest k with cumulative >= 0.80 (min 2) |
| `kmeans_table` | k = 2..6, silhouette in [-1, 1], `k_best` = argmax silhouette |
| `dbscan`, `hier` | labels length = rows used; hierarchical uses `k_best` clusters |
| `adf`, `d` | ADF table; integer d in {0,1,2} |
| `sarima_grid` | candidates sorted by AICc ascending, training window only |
| `holdout` | models {SARIMA, SeasonalNaive, Naive}; MAPE, RMSE, MAE >= 0; last 7 days |
| `forecast` | exactly 30 rows, first date = last observed date + 1 day, no NaN, `lo95 < lo80 < forecast < hi80 < hi95` |
| Determinism | two runs with the same data and seed give identical numbers |
| Data-blindness | pipeline never reads `data/raw`; the file is only loaded by notebooks/scripts |
