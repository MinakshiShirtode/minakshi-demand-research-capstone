"""Capstone predictive modelling (Task 5): rolling-origin one-step-ahead evaluation."""
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import SEED, N_BOOT
from .forecasting import fit_sarima, sane, mape, rmse, mae

warnings.filterwarnings("ignore")
ALPHAS = [0.01, 0.1, 1, 10]
SARIMA_GRID = [((p, 0, q), so) for p in (0, 1) for q in (0, 1) for so in ((0, 1, 0, 7), (1, 0, 0, 7))]


def design(df):
    X = pd.get_dummies(df["dow"].astype(int), prefix="dow").reindex(columns=[f"dow_{i}" for i in range(7)], fill_value=0).astype(float)
    X["event"] = df["marketing_event"].astype(float)
    X["holiday"] = df["holiday"].astype(float)
    X["t"] = df["t"].astype(float)
    X["lag1"] = df["demand"].shift(1).astype(float)
    X.index = df.index
    return X


def _ridge(Xtr, ytr, Xte):
    gs = GridSearchCV(make_pipeline(StandardScaler(), Ridge()), {"ridge__alpha": ALPHAS},
                      cv=TimeSeriesSplit(n_splits=3), scoring="neg_mean_squared_error")
    gs.fit(Xtr, ytr)
    return float(gs.predict(Xte)[0]), gs.best_params_["ridge__alpha"]


def _sarima_best(ytr):
    best, best_aicc = None, np.inf
    for order, so in SARIMA_GRID:
        try:
            r = fit_sarima(ytr, order, so)
            if np.isfinite(r.aicc) and r.aicc < best_aicc and sane(r, ytr):
                best, best_aicc = (r, order, so), r.aicc
        except Exception:
            continue
    if best is None:
        return float(ytr.iloc[-7]), None
    return float(best[0].get_forecast(1).predicted_mean.iloc[0]), (best[1], best[2])


def rolling_origin(df, first_train=14):
    d = df.reset_index(drop=True)
    y = pd.Series(d["demand"].to_numpy(float), index=pd.DatetimeIndex(d["date"], freq="D"))
    X = design(d)
    rows = []
    for i in range(first_train, len(d)):
        ytr = y.iloc[:i]
        Xtr, Xte = X.iloc[1:i], X.iloc[[i]]
        ytr_r = y.iloc[1:i].to_numpy()
        ridge_pred, alpha = _ridge(Xtr, ytr_r, Xte)
        sar_pred, spec = _sarima_best(ytr)
        rows.append(dict(date=d.loc[i, "date"], actual=y.iloc[i],
                         Naive=y.iloc[i - 1], SeasonalNaive=y.iloc[i - 7],
                         SARIMA=sar_pred, Ridge=ridge_pred, ridge_alpha=alpha, sarima_spec=str(spec)))
    return pd.DataFrame(rows)


def score(preds: pd.DataFrame, models=("Naive", "SeasonalNaive", "SARIMA", "Ridge")):
    tab = pd.DataFrame([dict(model=m, MAE=mae(preds.actual, preds[m]),
                             RMSE=rmse(preds.actual, preds[m]), MAPE=mape(preds.actual, preds[m]))
                        for m in models]).sort_values("RMSE").reset_index(drop=True)
    return tab


def paired_vs_baseline(preds, baseline="SeasonalNaive", models=("Naive", "SARIMA", "Ridge"), n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    base_err = (preds.actual - preds[baseline]).abs().to_numpy()
    rows = []
    for m in models:
        err = (preds.actual - preds[m]).abs().to_numpy()
        diff = err - base_err
        try:
            w = stats.wilcoxon(err, base_err)
            wp = w.pvalue
        except ValueError:
            wp = np.nan
        idx = rng.integers(0, len(diff), (n_boot, len(diff)))
        bs = diff[idx].mean(axis=1)
        rows.append(dict(model=m, mae_diff_vs_baseline=diff.mean(),
                         ci_low=np.percentile(bs, 2.5), ci_high=np.percentile(bs, 97.5), wilcoxon_p=wp))
    return pd.DataFrame(rows)
