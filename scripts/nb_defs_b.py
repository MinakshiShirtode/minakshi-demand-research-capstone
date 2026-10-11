"""Notebook definitions 03, 04 and 05."""
from nb_defs_a import SETUP

SETUP_ML = SETUP + r'''
from rq import clustering as cl
'''

NB03 = [
("md", r'''# 03 · Dimensionality Reduction & Unsupervised Clustering

Executes preregistered plan **section G**. Features are engineered from the series (11 numeric features, 28 usable days after lag/rolling windows). This is **exploratory** (S4): with n = 28 rows the clusters describe this month's structure and are not claimed to generalise.

Features: `demand, lag1, lag2, diff1, roll3_mean, roll3_std, dow_sin, dow_cos, marketing_event, holiday, t`.'''),
("code", SETUP_ML),
("code", r'''X = cl.build_features(df)
print("feature matrix:", X.shape)
display(X.describe().T[["mean", "std", "min", "max"]].round(2))
res = cl.run_pca(X)                      # StandardScaler (z-score) + PCA, all components
Z = res["Z"]; print("standardised: column means ≈", np.abs(Z.mean(0)).max().round(12), "| column SD =", Z.std(0, ddof=1).round(3)[:3], "...")'''),
("md", "## 1. PCA: scree plot, explained variance and component choice"),
("code", r'''evr, cum, k = res["evr"], res["cum"], res["n_keep"]
tab = pd.DataFrame({"PC": [f"PC{i+1}" for i in range(len(evr))], "eigenvalue": res["eigenvalues"], "explained_var_ratio": evr, "cumulative": cum})
display(tab.round(4)); tab.round(4).to_csv(TABLES / "03_pca_variance.csv", index=False)
fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
ax[0].bar(range(1, len(evr)+1), evr, color="#7570b3"); ax[0].plot(range(1, len(evr)+1), evr, "ko-")
ax[0].set_xlabel("Principal component"); ax[0].set_ylabel("Explained variance ratio"); ax[0].set_title("Scree plot")
ax[1].plot(range(1, len(cum)+1), cum, "o-", color="#d95f02"); ax[1].axhline(.8, ls="--", color="k"); ax[1].axvline(k, ls=":", color="grey")
ax[1].set_xlabel("Number of components"); ax[1].set_ylabel("Cumulative explained variance"); ax[1].set_title(f"Cumulative variance: {k} PCs reach {cum[k-1]:.1%} (≥80% rule)")
plt.tight_layout(); plt.savefig(FIGS / "03_scree.png", dpi=150); plt.show()
print(f"Retained (≥80% rule): {k} PCs  |  Kaiser (eigenvalue>1): {res['n_kaiser']} PCs  |  largest drop in explained variance: PC{int(np.argmax(-np.diff(evr)))+1} → PC{int(np.argmax(-np.diff(evr)))+2}")'''),
("code", r'''load = res["loadings"].iloc[:, :3]
fig, ax = plt.subplots(figsize=(5.5, 5))
sns.heatmap(load, annot=True, fmt=".2f", cmap="vlag", center=0, ax=ax, cbar_kws={"shrink": .7}); ax.set_title("PCA loadings (PC1–PC3)")
plt.tight_layout(); plt.savefig(FIGS / "03_loadings.png", dpi=150); plt.show()'''),
("md", "## 2. K-Means: Elbow method and Silhouette analysis (k = 2…6) on the retained PCs"),
("code", r'''Pk = res["scores"][:, :k]
ktab, kbest, klab = cl.kmeans_sweep(Pk)
display(ktab.round(4)); ktab.round(4).to_csv(TABLES / "03_kmeans_sweep.csv", index=False)
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(ktab.k, ktab.inertia, "o-", color="#1b9e77"); ax[0].set_title("Elbow method (inertia)"); ax[0].set_xlabel("k"); ax[0].set_ylabel("Within-cluster SS")
ax[1].plot(ktab.k, ktab.silhouette, "o-", color="#d95f02"); ax[1].axvline(kbest, ls="--", color="k"); ax[1].axhline(.25, ls=":", color="grey")
ax[1].set_title(f"Mean silhouette (best k = {kbest}, s = {ktab.silhouette.max():.3f})"); ax[1].set_xlabel("k")
plt.tight_layout(); plt.savefig(FIGS / "03_elbow_silhouette.png", dpi=150); plt.show()
print("Meaningful structure (best silhouette ≥ 0.25):", ktab.silhouette.max() >= .25)'''),
("code", r'''from sklearn.metrics import silhouette_samples
sv = silhouette_samples(Pk, klab)
fig, ax = plt.subplots(figsize=(6.5, 4.5)); y0 = 0
for c in sorted(set(klab)):
    v = np.sort(sv[klab == c]); ax.fill_betweenx(np.arange(y0, y0+len(v)), 0, v, alpha=.8, label=f"cluster {c} (n={len(v)})"); y0 += len(v) + 1
ax.axvline(sv.mean(), color="k", ls="--"); ax.set_xlabel("Silhouette coefficient"); ax.set_yticks([]); ax.legend(fontsize=8); ax.set_title(f"Per-sample silhouette, k={kbest}")
plt.tight_layout(); plt.savefig(FIGS / "03_silhouette_samples.png", dpi=150); plt.show()'''),
("md", "## 3. DBSCAN and Hierarchical (Ward) clustering"),
("code", r'''db = cl.run_dbscan(Pk); hi = cl.run_hier(Pk, kbest)
print(f"DBSCAN: eps (3-NN knee) = {db['eps']:.3f}, clusters = {db['n_clusters']}, noise points = {db['n_noise']}, silhouette (non-noise) = {db['silhouette']:.3f}")
print(f"Hierarchical (Ward, k={kbest}): silhouette = {hi['silhouette']:.3f}, cophenetic correlation = {hi['cophenetic']:.3f}")
from scipy.cluster.hierarchy import dendrogram
fig, ax = plt.subplots(1, 2, figsize=(14, 4.2))
ax[0].plot(db["kdist"], "o-", color="#7570b3"); ax[0].axhline(db["eps"], ls="--", color="k"); ax[0].set_title("DBSCAN sorted 3-NN distance (knee ⇒ eps)")
lbl = [d.strftime("%m-%d") for d in df.loc[X.index, "date"]]
dendrogram(hi["linkage"], labels=lbl, ax=ax[1], leaf_rotation=90, leaf_font_size=8, color_threshold=hi["linkage"][-(kbest-1), 2] - 1e-9); ax[1].set_title("Ward dendrogram (labels = date)")
plt.tight_layout(); plt.savefig(FIGS / "03_dbscan_dendrogram.png", dpi=150); plt.show()'''),
("md", "## 4. 2D and 3D cluster projections (Seaborn / Plotly)"),
("code", r'''S = res["scores"]; dt = df.loc[X.index, "day_type"].to_numpy()
plot = pd.DataFrame({"PC1": S[:, 0], "PC2": S[:, 1], "PC3": S[:, 2], "KMeans": klab.astype(str), "DBSCAN": np.where(db["labels"] == -1, "noise", db["labels"].astype(str)),
                     "Hierarchical": hi["labels"].astype(str), "day_type": dt, "date": lbl})
fig, ax = plt.subplots(1, 3, figsize=(17, 5))
for a, col in zip(ax, ["KMeans", "DBSCAN", "Hierarchical"]):
    sns.scatterplot(data=plot, x="PC1", y="PC2", hue=col, style="day_type", s=110, ax=a, palette="Set2")
    a.set_title(f"{col}: PC1 vs PC2 ({evr[0]:.0%} + {evr[1]:.0%} var.)"); a.legend(fontsize=7, loc="best")
plt.tight_layout(); plt.savefig(FIGS / "03_clusters_2d.png", dpi=150); plt.show()'''),
("code", r'''import plotly.express as px
fig3 = px.scatter_3d(plot, x="PC1", y="PC2", z="PC3", color="KMeans", symbol="day_type", hover_name="date",
                     title=f"K-Means clusters in PC space (3 PCs = {cum[2]:.0%} variance)")
fig3.update_traces(marker_size=6); fig3.update_layout(height=620)
fig3.write_html(str(FIGS / "03_clusters_3d_plotly.html"), include_plotlyjs="cdn")   # interactive file (open in a browser)
try:
    fig3.write_image(str(FIGS / "03_clusters_3d_plotly.png"), width=900, height=650, scale=2)
    display(Image(str(FIGS / "03_clusters_3d_plotly.png")))
except Exception as e:
    print("Plotly static export unavailable here (no Chrome for kaleido); interactive HTML saved to reports/figures/03_clusters_3d_plotly.html")
# static 3D fallback that renders everywhere (incl. GitHub)
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(8, 6)); a = fig.add_subplot(111, projection="3d")
for c, col in zip(sorted(set(klab)), sns.color_palette("Set2", kbest)):
    m = klab == c; a.scatter(S[m, 0], S[m, 1], S[m, 2], s=60, color=col, label=f"cluster {c}", depthshade=False)
a.set_xlabel("PC1"); a.set_ylabel("PC2"); a.set_zlabel("PC3"); a.legend(); a.set_title("K-Means clusters, 3D PCA projection"); a.view_init(22, 50)
plt.tight_layout(); plt.savefig(FIGS / "03_clusters_3d_static.png", dpi=150); plt.show()'''),
("md", "## 5. Agreement between methods and with the calendar day types (exploratory)"),
("code", r'''A = cl.agreement({"KMeans": klab, "DBSCAN": db["labels"], "Hierarchical": hi["labels"]}, dt)
fig, ax = plt.subplots(figsize=(5.5, 4.2)); sns.heatmap(A.astype(float), annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1, ax=ax); ax.set_title("Adjusted Rand index")
plt.tight_layout(); plt.savefig(FIGS / "03_ari.png", dpi=150); plt.show(); A.round(3).to_csv(TABLES / "03_ari.csv")
ct = pd.crosstab(pd.Series(klab, name="K-Means cluster"), pd.Series(dt, name="day type")); display(ct); ct.to_csv(TABLES / "03_kmeans_vs_daytype.csv")'''),
("code", r'''display(Markdown(f"""## Findings

* **PCA.** {k} components capture {cum[k-1]:.1%} of the variance (PC1 {evr[0]:.1%}, PC2 {evr[1]:.1%}, PC3 {evr[2]:.1%}); the Kaiser rule would keep {res['n_kaiser']}. PC1 loads on the level/lag/rolling-mean features (the *level & trend* axis); PC2 on `marketing_event`, `demand` and `diff1` (the *event-day jump* axis); PC3 on `holiday`, `t` and the weekday phase.
* **K-Means.** The silhouette is maximal at **k = {kbest}** (s = {ktab.silhouette.max():.3f}), above the preregistered 0.25 threshold, so structure is called meaningful. The elbow is gradual (no sharp kink), so silhouette is the deciding criterion, as preregistered.
* **DBSCAN** (eps = {db['eps']:.2f}) finds {db['n_clusters']} clusters plus {db['n_noise']} noise points; **Ward** at k = {kbest} has cophenetic correlation {hi['cophenetic']:.2f}. The three methods agree strongly (ARI K-Means–Ward = {A.loc['KMeans','Hierarchical']:.2f}).
* **Relation to calendar.** Clusters recover the weekly rhythm: one cluster is almost purely event days (7 of 8), one is exactly the four holidays (Sundays), and Regular days split into two groups (ARI vs day type = {A.loc['KMeans','day_type']:.2f}). Because the features include the event/holiday flags and the weekday phase, this is partly built-in; it is a description of the month, not an independent discovery.
* **Caveat.** n = 28, 11 correlated features; silhouette/ARI values are unstable at this size and no inferential claims are made.
"""))'''),
]

NB04 = [
("md", r'''# 04 · Time-Series Forecasting with SARIMA

Executes preregistered plan **section F**: decomposition → ADF stationarity test → SARIMA selection on a training window → accuracy on a 7-day test split (MAPE, RMSE) → 30-day forecast. Prophet is not used (see preregistration); SARIMA is an allowed option.'''),
("code", SETUP + r'''
from rq import forecasting as fc
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
y = fc.as_series(df); print(y.index.freq, len(y))'''),
("md", "## 1. Decomposition into Trend, Seasonality (period 7) and Residual"),
("code", r'''classical, stl = fc.decompose(y)
fig, ax = plt.subplots(4, 2, figsize=(14, 9), sharex=True)
for j, (name, dec) in enumerate([("Classical additive (period 7)", classical), ("STL robust (period 7)", stl)]):
    for i, (lab, comp) in enumerate([("Observed", dec.observed), ("Trend", dec.trend), ("Seasonal", dec.seasonal), ("Residual", dec.resid)]):
        ax[i, j].plot(comp, color=["k", "#d95f02", "#7570b3", "#1b9e77"][i], marker="o" if i == 3 else None, ms=3, lw=1.2)
        ax[i, j].set_ylabel(lab)
        if i == 3: ax[i, j].axhline(0, color="grey", lw=.6)
    ax[0, j].set_title(name)
for a in ax[-1]: a.tick_params(axis="x", rotation=45)
plt.tight_layout(); plt.savefig(FIGS / "04_decomposition.png", dpi=150); plt.show()
seas = pd.Series(classical.seasonal.iloc[:7].to_numpy(), index=[d.day_name()[:3] for d in y.index[:7]]).round(2)
print("Weekly seasonal profile (classical):\n", seas.to_string())
strength = lambda dec: max(0, 1 - np.var(dec.resid.dropna()) / np.var((dec.resid + dec.seasonal).dropna()))
print(f"Seasonal strength (STL) = {max(0, 1 - np.var(stl.resid)/np.var(stl.resid + stl.seasonal)):.2f};  trend strength = {max(0, 1 - np.var(stl.resid)/np.var(stl.resid + stl.trend)):.2f}")
pd.DataFrame({"observed": y, "trend_classical": classical.trend, "seasonal_classical": classical.seasonal, "resid_classical": classical.resid,
              "trend_stl": stl.trend, "seasonal_stl": stl.seasonal, "resid_stl": stl.resid}).round(3).to_csv(TABLES / "04_decomposition.csv")'''),
("md", "## 2. Stationarity: Augmented Dickey-Fuller and differencing\nNull hypothesis of ADF: a unit root is present (non-stationary). Preregistered rule: difference only while p ≥ 0.05 (max 2). ADF has low power at n = 30 and a constant-only specification does not model a deterministic trend/season, so ADF is complemented by ACF/PACF and by choosing the seasonal difference D with AICc."),
("code", r'''adf, d_rule = fc.adf_table(y); display(adf.round(4)); print("d from preregistered ADF rule:", d_rule); adf.round(4).to_csv(TABLES / "04_adf.csv", index=False)
# illustration only (not used to choose d): ADF after first and seasonal differencing
ill = []
for name, s in {"level": y, "diff(1)": y.diff().dropna(), "seasonal diff(7)": y.diff(7).dropna()}.items():
    r = fc.adfuller(s, regression="c", autolag="AIC", maxlag=3, regresults=False); ill.append(dict(series=name, n=len(s), adf=r[0], p=r[1], stationary_at_5pct=r[1] < .05))
display(pd.DataFrame(ill).round(4))
fig, ax = plt.subplots(2, 3, figsize=(15, 6))
y.plot(ax=ax[0, 0], marker="o", ms=3); ax[0, 0].set_title("Level"); y.diff().plot(ax=ax[0, 1], marker="o", ms=3); ax[0, 1].set_title("First difference")
y.diff(7).plot(ax=ax[0, 2], marker="o", ms=3); ax[0, 2].set_title("Seasonal difference (lag 7)")
plot_acf(y, lags=12, ax=ax[1, 0]); plot_pacf(y, lags=12, ax=ax[1, 1], method="ywm"); plot_acf(y.diff(7).dropna(), lags=10, ax=ax[1, 2]); ax[1, 2].set_title("ACF of seasonal difference")
plt.tight_layout(); plt.savefig(FIGS / "04_stationarity.png", dpi=150); plt.show()'''),
("md", "## 3. Hyperparameter selection (train = 2026-06-01…06-23, 23 days; AICc)\nGrid: p,q ∈ {0,1,2}; P,Q ∈ {0,1}; D ∈ {0,1}; s = 7; d from the ADF rule. Selection uses the **training window only**."),
("code", r'''train, test = y.iloc[:-TEST_DAYS], y.iloc[-TEST_DAYS:]
print("train:", train.index[0].date(), "→", train.index[-1].date(), "| test:", test.index[0].date(), "→", test.index[-1].date())
grid = fc.select_sarima(train, d_rule); print(len(grid), "candidate models converged and passed the sanity filter")
display(grid.head(10).round(3)); grid.round(3).to_csv(TABLES / "04_sarima_grid.csv", index=False)
best = grid.iloc[0]; order, sorder = best["order"], best["seasonal"]; print("Selected:", order, sorder, f"AICc={best.aicc:.2f}")'''),
("md", "## 4. Accuracy on the 7-day test split (MAPE, RMSE) vs baselines"),
("code", r'''hold, preds, actual, res_tr, lb_p = fc.evaluate_split(y, order, sorder)
display(hold.round(3)); hold.round(4).to_csv(TABLES / "04_holdout_metrics.csv", index=False)
rule = hold.set_index("model").RMSE
print("Preregistered rule: SARIMA recommended only if its RMSE < seasonal-naive RMSE →", rule["SARIMA"] < rule["SeasonalNaive"], f"({rule['SARIMA']:.2f} vs {rule['SeasonalNaive']:.2f})")
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(train.index, train, "k-o", ms=3, label="train"); ax.plot(test.index, test, "-o", color="#1b9e77", label="test (actual)")
for k, c in [("SARIMA", "#d95f02"), ("SeasonalNaive", "#7570b3"), ("Naive", "grey")]: ax.plot(test.index, preds[k], "--s", color=c, ms=4, label=f"{k}")
ax.axvline(test.index[0], color="grey", ls=":"); ax.set_title("Holdout forecasts (7 days)"); ax.legend(ncol=5, fontsize=8); ax.tick_params(axis="x", rotation=30)
plt.tight_layout(); plt.savefig(FIGS / "04_holdout.png", dpi=150); plt.show()
print(f"Ljung-Box (lag 7, residuals after differencing burn-in) p = {lb_p:.3f}  → {'no evidence of residual autocorrelation' if lb_p >= .05 else 'residual autocorrelation present'}")
burn = order[1] + sorder[3] * sorder[1]
fig, ax = plt.subplots(1, 3, figsize=(15, 3.6)); r = res_tr.resid.iloc[max(burn, 1):]
ax[0].plot(r, "o-", ms=3); ax[0].axhline(0, color="k", lw=.6); ax[0].set_title("Residuals (after burn-in)"); plot_acf(r, lags=7, ax=ax[1]); stats.probplot(r, plot=ax[2])
plt.tight_layout(); plt.savefig(FIGS / "04_residuals.png", dpi=150); plt.show()'''),
("md", "## 5. Final model refit on all 30 days → 30-day forecast (2026-07-01 … 2026-07-30)"),
("code", r'''fcst, res_all = fc.forecast_future(y, order, sorder, HORIZON)
fcst.round(2).to_csv(TABLES / "04_forecast_30d.csv"); display(fcst.round(1).head(10)); print("…"); display(fcst.round(1).tail(3))
fig, ax = plt.subplots(figsize=(12, 4.5))
ax.plot(y.index, y, "k-o", ms=3, label="observed"); ax.plot(fcst.index, fcst.forecast, "-", color="#d95f02", label="30-day forecast")
ax.fill_between(fcst.index, fcst.lo95, fcst.hi95, color="#d95f02", alpha=.15, label="95% PI"); ax.fill_between(fcst.index, fcst.lo80, fcst.hi80, color="#d95f02", alpha=.3, label="80% PI")
ax.axvline(y.index[-1], color="grey", ls=":"); ax.set_title(f"SARIMA{order}x{sorder}: 30-day demand forecast"); ax.legend(); ax.tick_params(axis="x", rotation=30)
plt.tight_layout(); plt.savefig(FIGS / "04_forecast_30d.png", dpi=150); plt.show()'''),
("code", r'''wk = fcst.forecast.groupby(fcst.index.to_period("W")).mean().round(1)
wk_txt = "; ".join(f"week of {p.start_time:%b %d}: {v}" for p, v in wk.items()) + " (last week is partial, 4 days)"
display(Markdown(f"""## Findings

* **Decomposition.** A strong weekly cycle (peak Saturday, trough Monday) on top of a clear upward trend through June (weekly means rise from ≈131 to ≈152 over weeks 1–4).
* **Stationarity.** ADF on the level gives p = {adf.p_value.iloc[0]:.3f} (lag {int(adf.used_lag.iloc[0])}), so the preregistered rule sets **d = {d_rule}** (no non-seasonal differencing). Seasonal differencing (D) was left to AICc and was selected (D = {sorder[1]}). ADF has low power at n = 30, so this is a modest result.
* **Model.** Selected on the training window: SARIMA{order}x{sorder} (AICc = {best.aicc:.1f}). The AICc comparison across D = 0 and D = 1 is approximate (see preregistration).
* **Accuracy (7-day holdout).** SARIMA MAPE = {hold.set_index('model').loc['SARIMA','MAPE']:.2f}%, RMSE = {rule['SARIMA']:.2f}; seasonal-naive MAPE = {hold.set_index('model').loc['SeasonalNaive','MAPE']:.2f}%, RMSE = {rule['SeasonalNaive']:.2f}; naive RMSE = {rule['Naive']:.2f}. The preregistered rule is met, so SARIMA is recommended. Seasonal-naive loses mainly because it ignores the weekly upward drift.
* **Diagnostics.** Ljung-Box p = {lb_p:.2f}: no evidence of residual autocorrelation.
* **Forecast.** 30-day mean forecast by week: {wk_txt}. The 95% interval widens from ±{(fcst.hi95.iloc[0]-fcst.lo95.iloc[0])/2:.1f} on day 1 to ±{(fcst.hi95.iloc[-1]-fcst.lo95.iloc[-1])/2:.1f} on day 30.
* **Caveats.** Only 4 weekly cycles and a 7-point test set: the accuracy estimate is itself very uncertain, and the extremely low error suggests a highly regular series. The forecast assumes the June weekly pattern and the recent upward drift persist; it cannot anticipate regime changes (e.g. calendar events outside June).
"""))'''),
]

NB05 = [
("md", r'''# 05 · Capstone: Predictive Modelling Workflow

Executes preregistered plan **section H**. *Question:* do models using calendar information and recent history predict **next-day demand** more accurately than a seasonal-naive baseline?

Design: expanding-window, one-step-ahead rolling-origin evaluation; first origin after 14 training days → **16 forecast origins** (2026-06-15 … 06-30). Models: M0 Naive, M1 Seasonal naive, M2 SARIMA (small grid chosen by AICc *inside each origin*), M3 Ridge (calendar + lag1, alpha by `TimeSeriesSplit(3)` *inside each origin*). No test information is used for any tuning.'''),
("code", SETUP + r'''
from rq import capstone as cs, forecasting as fc'''),
("code", r'''pred = cs.rolling_origin(df); pred.to_csv(TABLES / "05_rolling_origin_predictions.csv", index=False)
print(len(pred), "origins:", pred.date.min().date(), "→", pred.date.max().date())
display(pred.drop(columns=["sarima_spec"]).round(1).head(6))
print("SARIMA spec chosen at origins:", pred.sarima_spec.value_counts().to_dict()); print("Ridge alpha chosen:", pred.ridge_alpha.value_counts().to_dict())'''),
("code", r'''score = cs.score(pred); cmp = cs.paired_vs_baseline(pred)
display(score.round(3)); display(cmp.round(3)); score.round(4).to_csv(TABLES / "05_model_scores.csv", index=False); cmp.round(4).to_csv(TABLES / "05_paired_vs_seasonal_naive.csv", index=False)
fig, ax = plt.subplots(1, 2, figsize=(14, 4.3))
ax[0].plot(pred.date, pred.actual, "k-o", label="actual")
for m, c in [("SARIMA", "#d95f02"), ("Ridge", "#7570b3"), ("SeasonalNaive", "#1b9e77"), ("Naive", "grey")]: ax[0].plot(pred.date, pred[m], "--", color=c, label=m)
ax[0].legend(fontsize=8); ax[0].set_title("One-step-ahead predictions (16 origins)"); ax[0].tick_params(axis="x", rotation=30)
err = pd.melt(pd.DataFrame({m: (pred.actual - pred[m]).abs() for m in ["Naive", "SeasonalNaive", "SARIMA", "Ridge"]}), var_name="model", value_name="abs_error")
sns.boxplot(data=err, x="model", y="abs_error", ax=ax[1], palette="Set2"); sns.stripplot(data=err, x="model", y="abs_error", color="k", size=3, ax=ax[1]); ax[1].set_title("Absolute error by model")
plt.tight_layout(); plt.savefig(FIGS / "05_model_comparison.png", dpi=150); plt.show()'''),
("md", "## Interpretable view: Ridge coefficients on all 30 days (standardised features)"),
("code", r'''from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
Xd = cs.design(df).iloc[1:]; yd = df["demand"].iloc[1:].to_numpy(float)
sc = StandardScaler().fit(Xd); rg = Ridge(alpha=float(pred.ridge_alpha.mode().iloc[0])).fit(sc.transform(Xd), yd)
coef = pd.Series(rg.coef_, index=Xd.columns).sort_values(); display(coef.round(2).to_frame("standardised coefficient"))
fig, ax = plt.subplots(figsize=(6, 4)); coef.plot.barh(ax=ax, color="#7570b3"); ax.set_title("Ridge coefficients (full data)"); plt.tight_layout(); plt.savefig(FIGS / "05_ridge_coef.png", dpi=150); plt.show()
coef.round(4).to_csv(TABLES / "05_ridge_coef.csv")'''),
("code", r'''best = score.iloc[0]; s = score.set_index("model")
simpler_tie = (score.RMSE.iloc[1] - score.RMSE.iloc[0]) / score.RMSE.iloc[0] <= .01
display(Markdown(f"""## Findings

* **Winner by preregistered rule (lowest RMSE over 16 origins): {best.model}** (MAE {best.MAE:.2f}, RMSE {best.RMSE:.2f}, MAPE {best.MAPE:.2f}%). Ridge: RMSE {s.loc['Ridge','RMSE']:.2f}; seasonal-naive: RMSE {s.loc['SeasonalNaive','RMSE']:.2f}; naive: RMSE {s.loc['Naive','RMSE']:.2f}. (Runner-up within 1% RMSE, i.e. tie rule applies: {simpler_tie}.)
* **Versus seasonal-naive baseline (paired, MAE):** SARIMA {cmp.set_index('model').loc['SARIMA','mae_diff_vs_baseline']:.2f} (95% bootstrap CI {cmp.set_index('model').loc['SARIMA','ci_low']:.2f} to {cmp.set_index('model').loc['SARIMA','ci_high']:.2f}), Ridge {cmp.set_index('model').loc['Ridge','mae_diff_vs_baseline']:.2f} (CI {cmp.set_index('model').loc['Ridge','ci_low']:.2f} to {cmp.set_index('model').loc['Ridge','ci_high']:.2f}). Negative = better than baseline. Both clear the baseline decisively; plain naive is *worse* than seasonal-naive.
* **Why:** the series has a strong weekly pattern plus an upward drift of roughly 6–8 units per week. Seasonal-naive repeats last week's value and so is systematically about 6 units too low; SARIMA's seasonal differencing with an AR(1) term, and Ridge's lag-1 + time terms, absorb that drift.
* **Caveats.** 16 origins, strongly autocorrelated errors, and one 30-day series: treat intervals as descriptive. The near-perfect accuracy indicates an unusually regular series; this must be re-checked on longer, noisier data before any operational use.
"""))'''),
]
