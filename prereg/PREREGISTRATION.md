# Analysis Preregistration

**Project:** Day-type structure of daily demand, June 2026
**Version:** v1 (locked by git tag `prereg-v1`, committed *before* any outcome analysis)
**Locked (UTC):** 2026-10-11T05:45:00Z
**Raw data SHA-256:** `171eb7a471f10bfdb4674d34bfa78428c6118823de728245e24239409a053f89` (`data/raw/daily-demand-series.csv`)

---

## Data access log (disclosure)

Before this plan was written, the following was looked at, and nothing else:

1. Column names, dtypes, row count (30), date range, missing-value counts, duplicate dates.
2. Counts and calendar positions of the two exposure/calendar flags (`marketing_event`, `holiday`) and their cross-tabulation, plus weekday counts.
3. The first four raw lines of the file were printed by a `head -5` command during schema inspection, which incidentally showed the first four `demand` values (118, 121, 125, 123, all on non-event, non-holiday Mon-Thu days). No summary, plot, test, or group comparison of `demand` was run.

**Structural finding from (2) that shapes this plan:** every `marketing_event = 1` day is a Friday or Saturday (8 days: 2 per week x 4 weeks), and every `holiday = 1` day is a Sunday (4 days). Event days and holiday days therefore cannot be separated from weekday effects: there are no non-event Fridays/Saturdays and no non-holiday Sundays in the window. All inference below is framed accordingly (see "Estimand and its limits").

---

## Research question

**Primary (falsifiable):** In the 30-day observation window 2026-06-01 to 2026-06-30, is mean daily demand on *marketing-event days* different from mean daily demand on *regular days* (no event, no holiday)?

| Element | Definition |
|---|---|
| Population | Daily demand of the single series in `daily-demand-series.csv` (one product/business unit, unit not otherwise identified). Inference is to this series in this window only. |
| Unit of analysis | One calendar day (n = 30). |
| Exposure / intervention | `marketing_event = 1` (n = 8). |
| Comparator | Regular days: `marketing_event = 0` and `holiday = 0` (n = 18). |
| Outcome | `demand` (daily count, raw scale). |
| Observation window | 2026-06-01 (Mon) through 2026-06-30 (Tue), inclusive, daily frequency. |

**Estimand and its limits.** The estimand is the *difference in mean daily demand between event days and regular days*. Because event days are always Fri/Sat and regular days are always Mon-Thu, this contrast is the sum of (a) any causal marketing effect and (b) any Fri/Sat-vs-Mon-Thu weekday effect. The design cannot separate them. Results will be reported as an **association bundled with day-of-week**, never as a causal marketing lift. Observational, single series, no randomisation.

**Secondary questions** (labelled as secondary in all reports):

- S1. Do mean demands differ across the three day types Event / Holiday / Regular (one-way ANOVA, Tukey HSD)?
- S2. Do day type and calendar week interact (two-way ANOVA)?
- S3. Is the series stationary, seasonal at period 7, and forecastable better than a seasonal-naive baseline?
- S4. Is there unsupervised cluster structure in engineered features of the series?

---

## Hypotheses

**Primary (H1 family, Event vs Regular).**
H0: mu_event = mu_regular.
HA: mu_event != mu_regular (two-sided). Expected direction, justified by the purpose of marketing events: mu_event > mu_regular. The test is two-sided; direction is recorded as an expectation only.

**S1 (one-way ANOVA across Event, Holiday, Regular).**
H0: mu_event = mu_holiday = mu_regular. HA: at least one mean differs. Post-hoc: Tukey HSD on all 3 pairs (family-wise alpha = 0.05).

**S2 (two-way ANOVA, week x day type, complete weeks only).**
H0 (week main effect): no week differences. H0 (day-type main effect): no day-type differences. H0 (interaction): the day-type pattern is the same in every week.

**S3.** H0: the selected SARIMA model has RMSE on the holdout no lower than seasonal-naive. HA: SARIMA RMSE is lower.

**S4.** Exploratory; no formal hypothesis. A clustering is called "meaningful" only if the best mean silhouette >= 0.25.

---

## Population, sample, and exclusions

- **Sample:** all 30 rows. **Stopping rule:** fixed; no further data will be collected or appended to this analysis. No optional stopping.
- **Primary comparison:** Event (n = 8) vs Regular (n = 18). Holiday days (n = 4) are **excluded from the primary contrast** and analysed only in S1/S2, because they are a distinct category (all Sundays).
- **Rows with both `marketing_event = 1` and `holiday = 1`:** labelled `Both`, excluded from primary and S1/S2 (none are expected).
- **Missing data:** none expected. If any `demand` is missing, use complete cases for tests and record a deviation; time-series models would then require a documented interpolation, also recorded as a deviation.
- **Outliers:** no outlier removal. Robustness is handled by rank-based tests, permutation tests and log-scale sensitivity analysis, not by deleting points.
- **Two-way ANOVA exclusion:** days 2026-06-29 and 2026-06-30 form a partial 5th week (Mon, Tue; Regular only) and are excluded from S2 only, so every week x day-type cell is populated (weeks 1-4, n = 28).
- **Unsupervised analysis (S4):** the first 2 rows are dropped because the lag-2 and 3-day rolling features are undefined there (n = 28 rows).

---

## Variables and measures

**Outcome:** `demand` (raw scale, no transformation in primary analyses).

**Derived variables (deterministic functions of `date`, `marketing_event`, `holiday`):**

| Name | Definition |
|---|---|
| `day_type` | `Holiday` if holiday = 1 (and event = 0); `Event` if event = 1 (and holiday = 0); `Both` if both; else `Regular` |
| `dow` | day of week, Mon = 0 ... Sun = 6 |
| `week` | `floor((date - 2026-06-01) / 7) + 1` (weeks 1-5; weeks begin Monday) |
| `t` | integer day index, 0 ... 29 |

**Transformations:** none for primary. `log(demand)` is used **only** as a pre-specified robustness check.

**Features for S4 (PCA / clustering), n = 28 x 11:** `demand`, `lag1`, `lag2`, `diff1`, `roll3_mean`, `roll3_std` (3-day trailing window, sample SD), `dow_sin`, `dow_cos` (period 7), `marketing_event`, `holiday`, `t`. All standardised (z-score) before PCA.

---

## Analysis plan

**Global settings.** alpha = 0.05 two-sided. Random seed = **20260601** for every stochastic step. Confidence intervals are 95%. Python 3.12; exact package versions in `requirements.lock`.

### A. Normality and assumptions (Task 2, step 1)
- Shapiro-Wilk and Kolmogorov-Smirnov (vs N(mean, SD) estimated from the sample) per day-type group. Because KS with estimated parameters is anti-conservative, the **Lilliefors-corrected** p-value is also reported and is the one used if the two disagree.
- Variance homogeneity: Brown-Forsythe (Levene, median-centred).
- Decision rule: if Shapiro-Wilk p < 0.05 in **either** group of the primary contrast, the Mann-Whitney U test is the primary inference; otherwise Welch's t-test is primary. **Both tests are always reported.**

### B. Primary test (Event vs Regular)
- Welch two-sample t-test (no equal-variance assumption), mean difference with Welch 95% CI.
- Mann-Whitney U (two-sided, exact/asymptotic as chosen by scipy `method="auto"`), rank-biserial correlation, Hodges-Lehmann location shift with 95% percentile bootstrap CI (10,000 resamples, seed above).
- Effect size: Hedges' g with 95% CI (large-sample SE).
- **Primary decision:** reject H0 iff the primary test (per rule A) has p < 0.05. This is the **only confirmatory test**; everything else is secondary/exploratory and no multiplicity correction is applied across them (they are reported as such, with exact p-values).

### C. One-way ANOVA with Tukey HSD (S1)
- `demand ~ day_type` over Event, Holiday, Regular. F-test; effect sizes eta-squared and omega-squared; Tukey HSD with simultaneous 95% CIs for all 3 pairwise differences.
- If Brown-Forsythe p < 0.05, also report Kruskal-Wallis as the robust counterpart.

### D. Two-way ANOVA (S2)
- `demand ~ C(week) * C(day_type)`, weeks 1-4, Type II sums of squares. Report F, p, partial eta-squared for week, day type, interaction. Interpret the interaction cautiously: the Holiday cell has n = 1 per week.

### E. Robustness checks for the primary contrast
1. Permutation test of the mean difference (10,000 permutations).
2. OLS with `demand ~ event + t` (linear trend adjustment), HC3 robust SEs.
3. Leave-one-week-out: recompute the primary difference dropping each week in turn; report range.
4. Log scale: Welch t-test on `log(demand)`.
5. Rank-based (A/B) vs parametric agreement reported explicitly.
A conclusion is called **robust** only if checks 1-4 all agree in sign with the primary estimate and the primary CI excludes 0.

### F. Time-series pipeline (S3, Task 4)
- **Decomposition:** additive classical decomposition, period 7; robust STL (period 7) as secondary.
- **Stationarity:** Augmented Dickey-Fuller (constant, `autolag="AIC"`, `maxlag=3`). Rule: if p >= 0.05, difference once and retest; at most d = 2. The number of differences needed sets `d`. Note ADF has low power at n = 30; this is stated, and `D` (seasonal differencing) is chosen by AICc, not ADF.
- **Split:** train 2026-06-01..2026-06-23 (23 days), test 2026-06-24..2026-06-30 (7 days). The test set is untouched by model selection.
- **Model family:** SARIMA(p,d,q)(P,D,Q)[7], p,q in {0,1,2}, P,Q in {0,1}, D in {0,1}, d fixed by ADF rule; selection by AICc on the training set only. Fits use enforced stationarity and invertibility (unconstrained fits on ~23 points were found, in the synthetic pipeline check, to land on boundary solutions with degenerate prediction intervals). Candidates are discarded if non-finite or if the 95% prediction-interval half-width at horizon 7 exceeds 20 x the SD of the training series (numerical-sanity filter, fixed before real data analysis). AICc is compared across D in {0,1}; since likelihoods under differencing use different effective samples this is an approximation and is flagged as such. No exogenous regressors (event/holiday are perfectly tied to weekday in-sample, so the seasonal term already carries them).
- **Baselines:** seasonal naive (lag 7) and naive (last value).
- **Accuracy:** MAPE and RMSE (and MAE) on the 7-day test set.
- **Decision rule:** SARIMA is recommended over seasonal naive only if its test RMSE is strictly lower; otherwise seasonal naive is recommended and this is reported.
- **Diagnostics:** Ljung-Box (lag 7) on residuals; residual plots.
- **Final forecast:** refit the selected specification on all 30 days; forecast 2026-07-01..2026-07-30 (30 days) with 80% and 95% prediction intervals. Facebook Prophet is not used (not required; SARIMA is allowed by the task and Prophet adds a heavy dependency for 30 observations).

### G. Dimensionality reduction and clustering (S4, Task 3)
- Standardise features (z-score); PCA on all components.
- **Components retained:** smallest number reaching >= 80% cumulative explained variance (minimum 2). Scree plot, Kaiser (eigenvalue > 1) and elbow reported as cross-checks, not decision rules.
- **K-Means** on retained PCs, k = 2..6, `n_init = 50`: elbow (inertia) and mean silhouette; selected k = argmax silhouette. "Meaningful structure" requires best silhouette >= 0.25; otherwise reported as "no clear cluster structure".
- **DBSCAN** on retained PCs: `min_samples = 3`; `eps` = knee of the sorted 3-NN distance curve (point of maximum distance from the chord joining the curve endpoints).
- **Hierarchical:** Ward linkage on retained PCs, dendrogram, cut at the K-Means k. Cophenetic correlation reported.
- **Agreement:** adjusted Rand index between methods, and between each method and `day_type` (exploratory).
- Visualisation: Seaborn 2D scatter (PC1/PC2), Plotly 3D (PC1-3), static Matplotlib 3D fallback.

### H. Capstone predictive modelling (Task 5)
- **Question:** Do models using calendar information and recent history predict next-day demand more accurately than a seasonal-naive baseline?
- **Design:** expanding-window, one-step-ahead rolling-origin evaluation, first origin after 14 days of training, 16 forecast origins (target days 15-30).
- **Models:** M0 naive (lag 1); M1 seasonal naive (lag 7); M2 SARIMA (small grid selected by AICc inside each origin): non-seasonal (p,q) in {0,1}^2 with seasonal (0,1,0)[7] or (1,0,0)[7], d = 0; M3 Ridge regression on day-of-week one-hot, event, holiday, t, lag1, with alpha in {0.01, 0.1, 1, 10} chosen by `TimeSeriesSplit(3)` inside each origin.
- **Metrics:** MAE, RMSE, MAPE over the 16 origins. Paired comparison vs M1 using Wilcoxon signed-rank on absolute errors and a paired bootstrap 95% CI for the MAE difference (10,000 resamples). Descriptive only given n = 16.
- **Final recommendation rule:** the model with the lowest RMSE over the 16 origins; ties (within 1%) go to the simpler model.

---

## Output contracts (synthetic-fixture pipeline check)

Defined in `docs/OUTPUT_CONTRACTS.md` and enforced in `tests/test_pipeline_synthetic.py`, run on **synthetic data only** (generated by `src/rq/synthetic.py`) *before* the real outcome is analysed. The contract covers: schema validation; keys, types, finite values, and ranges for each test result; p-values in [0, 1]; CI contains its point estimate; effect-sign recovery on a planted effect; approximately nominal false-positive rate on null fixtures; PCA variance sums to 1; silhouette in [-1, 1]; forecast has exactly 30 rows starting the day after the series ends with lower < mean < upper; determinism under the fixed seed.

---

## Deviations

*Every later deviation from this plan is recorded below with timestamp (UTC) and reason. Empty at lock time.*

| Timestamp (UTC) | Section | Deviation | Reason |
|---|---|---|---|
| 2026-10-11T05:44:34Z | Header | Header states lock time 05:45:00Z; the actual commit/tag time is 05:43:07Z (git tag `prereg-v1` is authoritative). | Typing error in header; no analytic content changed. |
| 2026-10-11T05:44:34Z | F (diagnostics) | Ljung-Box residual test now drops differencing burn-in residuals (`d + s*D` observations) and uses `model_df = p+q+P+Q`. The first real-data run (full pipeline) returned p = 8.6e-6, an artefact of including the diffuse-initialisation residuals. After the correction p = 0.58. | Implementation bug in a diagnostic, found on first real run. No model-selection or test decision depended on it. Code fix in commit following `prereg-v1`. |
| 2026-10-11T05:52Z | Deviations log | The two timestamps first written in this table (05:50Z and 05:55Z) were estimates typed from memory and were later than the true commit time of that edit (05:44:34Z, commit `2cc7cfb`). Corrected above; the original wording is preserved in git history. | Accuracy of the audit trail. No analytic content changed. |
