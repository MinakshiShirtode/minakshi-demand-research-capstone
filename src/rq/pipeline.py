"""End-to-end data-blind pipeline: takes ANY dataframe with the raw schema and returns a dict
of named artifacts. Contract is checked by `check_contract` (see docs/OUTPUT_CONTRACTS.md)."""
import numpy as np
import pandas as pd

from . import stats_tests as st, clustering as cl, forecasting as fc
from .data import add_calendar
from .config import HORIZON, TEST_DAYS

CONTRACT = {
    "primary": ["n_a", "n_b", "mean_a", "mean_b", "diff", "welch_t", "welch_df", "welch_p",
                "welch_ci_low", "welch_ci_high", "hedges_g", "g_ci_low", "g_ci_high",
                "mwu_U", "mwu_p", "rank_biserial", "hl_shift", "hl_ci_low", "hl_ci_high",
                "perm_p", "primary_test", "primary_p", "reject_null"],
    "normality_cols": ["group", "n", "mean", "sd", "shapiro_W", "shapiro_p", "ks_D", "ks_p",
                       "lilliefors_D", "lilliefors_p", "normal_by_shapiro"],
    "oneway": ["F", "p", "df_between", "df_within", "eta2", "omega2", "brown_forsythe_p", "kruskal_p"],
    "tukey_cols": ["group1", "group2", "mean_diff", "ci_low", "ci_high", "p_adj", "reject"],
    "twoway_rows": ["C(week)", "C(day_type)", "C(week):C(day_type)", "Residual"],
}


def run_pipeline(df: pd.DataFrame, fast: bool = True) -> dict:
    d = add_calendar(df)
    out = {"calendar": d}
    ev = d.loc[d.day_type == "Event", "demand"]
    rg = d.loc[d.day_type == "Regular", "demand"]
    out["normality"] = st.normality_table({k: d.loc[d.day_type == k, "demand"]
                                           for k in ("Event", "Holiday", "Regular")})
    out["primary"] = st.two_sample(ev, rg)
    out["oneway"], out["tukey"] = st.anova_oneway(d)
    out["twoway"], out["twoway_n"] = st.anova_twoway(d)
    out["robustness"], out["leave_one_week_out"] = st.robustness(d)

    X = cl.build_features(d)
    pca = cl.run_pca(X)
    P = pca["scores"][:, :pca["n_keep"]]
    ktab, kbest, klab = cl.kmeans_sweep(P)
    out.update(features=X, pca=pca, kmeans_table=ktab, k_best=kbest, kmeans_labels=klab,
               dbscan=cl.run_dbscan(P), hier=cl.run_hier(P, kbest))

    y = fc.as_series(d)
    out["decomp"] = fc.decompose(y)
    out["adf"], out["d"] = fc.adf_table(y)
    grid = dict(pq=(0, 1), PQ=(0, 1), Dset=(0, 1)) if fast else {}
    train = y.iloc[:-TEST_DAYS]
    out["sarima_grid"] = fc.select_sarima(train, out["d"], **grid)
    best = out["sarima_grid"].iloc[0]
    out["holdout"], out["holdout_preds"], out["holdout_actual"], _, out["ljung_box_p"] = \
        fc.evaluate_split(y, best["order"], best["seasonal"])
    out["forecast"], _ = fc.forecast_future(y, best["order"], best["seasonal"], HORIZON)
    return out


def check_contract(res: dict) -> None:
    p = res["primary"]
    for k in CONTRACT["primary"]:
        assert k in p, f"primary missing {k}"
    for k in CONTRACT["oneway"]:
        assert k in res["oneway"], f"oneway missing {k}"
    assert list(res["normality"].columns) == CONTRACT["normality_cols"]
    assert list(res["tukey"].columns) == CONTRACT["tukey_cols"]
    assert len(res["tukey"]) == 3
    for r in CONTRACT["twoway_rows"]:
        assert r in res["twoway"].index, f"twoway missing {r}"
    for k in ("welch_p", "mwu_p", "perm_p", "primary_p"):
        assert 0.0 <= p[k] <= 1.0, f"{k} out of range"
    assert p["welch_ci_low"] <= p["diff"] <= p["welch_ci_high"]
    assert res["tukey"]["p_adj"].between(0, 1).all()
    assert np.isclose(res["pca"]["evr"].sum(), 1.0)
    assert (np.diff(res["pca"]["cum"]) >= -1e-12).all()
    assert res["kmeans_table"]["silhouette"].between(-1, 1).all()
    f = res["forecast"]
    assert len(f) == HORIZON
    assert f.index[0] == res["calendar"]["date"].max() + pd.Timedelta(days=1)
    assert f.notna().all().all()
    assert ((f["lo95"] < f["lo80"]) & (f["lo80"] < f["forecast"]) &
            (f["forecast"] < f["hi80"]) & (f["hi80"] < f["hi95"])).all()
    h = res["holdout"]
    assert set(h["model"]) == {"SARIMA", "SeasonalNaive", "Naive"}
    assert (h[["MAPE", "RMSE", "MAE"]] >= 0).all().all()
