"""Time-series pipeline (Task 4): decomposition, ADF, SARIMA selection, evaluation, forecast."""
import itertools
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose, STL
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox

from .config import ALPHA, SEASON, HORIZON, TEST_DAYS

warnings.filterwarnings("ignore")


def as_series(df):
    return pd.Series(df["demand"].to_numpy(float), index=pd.DatetimeIndex(df["date"], freq="D"), name="demand")


def decompose(y, period=SEASON):
    classical = seasonal_decompose(y, model="additive", period=period)
    stl = STL(y, period=period, robust=True).fit()
    return classical, stl


def adf_table(y, max_diff=2):
    rows, cur, d_needed = [], y.copy(), None
    for d in range(max_diff + 1):
        r = adfuller(cur.dropna(), regression="c", autolag="AIC", maxlag=3, regresults=False)
        rows.append(dict(differences=d, adf_stat=r[0], p_value=r[1], used_lag=r[2],
                         nobs=r[3], crit_5pct=r[4]["5%"], stationary=bool(r[1] < ALPHA)))
        if r[1] < ALPHA and d_needed is None:
            d_needed = d
            break
        cur = cur.diff().dropna()
    if d_needed is None:
        d_needed = max_diff
    return pd.DataFrame(rows), d_needed


def mape(y, f):
    y, f = np.asarray(y, float), np.asarray(f, float)
    return float(np.mean(np.abs((y - f) / y)) * 100)


def rmse(y, f):
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(f, float)) ** 2)))


def mae(y, f):
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(f, float))))


def fit_sarima(y, order, seasonal_order):
    # stationarity/invertibility enforced: unconstrained fits on ~23 points hit boundary solutions
    # with degenerate (astronomically wide) prediction intervals.
    return SARIMAX(y, order=order, seasonal_order=seasonal_order,
                   enforce_stationarity=True, enforce_invertibility=True).fit(disp=False, maxiter=200)


def sane(res, y, h=SEASON, max_ratio=20.0):
    """Numerical-sanity filter: 95% PI half-width at horizon h must be <= max_ratio * SD(y)."""
    ci = res.get_forecast(h).conf_int(alpha=0.05)
    half = float((ci.iloc[-1, 1] - ci.iloc[-1, 0]) / 2)
    return np.isfinite(half) and half <= max_ratio * float(np.std(y, ddof=1))


def select_sarima(train, d, pq=(0, 1, 2), PQ=(0, 1), Dset=(0, 1), s=SEASON):
    rows = []
    for p, q, P, Q, D in itertools.product(pq, pq, PQ, PQ, Dset):
        try:
            r = fit_sarima(train, (p, d, q), (P, D, Q, s))
            aicc = float(r.aicc)
            if np.isfinite(aicc) and np.isfinite(r.llf) and sane(r, train):
                rows.append(dict(order=(p, d, q), seasonal=(P, D, Q, s), aicc=aicc, aic=float(r.aic), bic=float(r.bic)))
        except Exception:
            continue
    tab = pd.DataFrame(rows).sort_values("aicc").reset_index(drop=True)
    if tab.empty:
        raise RuntimeError("no SARIMA candidate converged")
    return tab


def seasonal_naive(train, h, s=SEASON):
    last = np.asarray(train, float)[-s:]
    return np.resize(last, h)


def naive(train, h):
    return np.repeat(float(np.asarray(train)[-1]), h)


def evaluate_split(y, order, seasonal_order, test_days=TEST_DAYS):
    train, test = y.iloc[:-test_days], y.iloc[-test_days:]
    res = fit_sarima(train, order, seasonal_order)
    fc = res.get_forecast(len(test)).predicted_mean
    preds = {"SARIMA": fc.to_numpy(), "SeasonalNaive": seasonal_naive(train, len(test)),
             "Naive": naive(train, len(test))}
    tab = pd.DataFrame([dict(model=k, MAPE=mape(test, v), RMSE=rmse(test, v), MAE=mae(test, v))
                        for k, v in preds.items()]).sort_values("RMSE").reset_index(drop=True)
    # drop the burn-in residuals created by differencing (diffuse initialisation: resid == raw value)
    burn = order[1] + seasonal_order[3] * seasonal_order[1]
    pq = order[0] + order[2] + seasonal_order[0] + seasonal_order[2]
    lb = acorr_ljungbox(res.resid.iloc[max(burn, 1):], lags=[SEASON], model_df=pq, return_df=True)
    return tab, preds, test, res, float(lb["lb_pvalue"].iloc[0])


def forecast_future(y, order, seasonal_order, h=HORIZON):
    res = fit_sarima(y, order, seasonal_order)
    fc = res.get_forecast(h)
    out = pd.DataFrame({"forecast": fc.predicted_mean})
    ci95, ci80 = fc.conf_int(alpha=0.05), fc.conf_int(alpha=0.20)
    out["lo80"], out["hi80"] = ci80.iloc[:, 0], ci80.iloc[:, 1]
    out["lo95"], out["hi95"] = ci95.iloc[:, 0], ci95.iloc[:, 1]
    out.index.name = "date"
    return out, res
