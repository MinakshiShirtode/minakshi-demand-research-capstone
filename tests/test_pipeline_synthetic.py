"""Data-blind pipeline check on SYNTHETIC fixtures only (run before touching real outcomes)."""
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from rq.data import validate, add_calendar
from rq.synthetic import make_synthetic
from rq.pipeline import run_pipeline, check_contract
from rq import stats_tests as st, clustering as cl, forecasting as fc


@pytest.fixture(scope="module")
def planted():
    return make_synthetic(seed=1, event_effect=30.0, weekly_amp=0.0, sd=4.0)


@pytest.fixture(scope="module")
def result(planted):
    return run_pipeline(planted, fast=True)


# ---- schema / calendar -------------------------------------------------------
def test_schema_rejects_bad_inputs():
    good = make_synthetic()
    validate(good)
    with pytest.raises(ValueError):
        validate(good.drop(columns=["holiday"]))
    with pytest.raises(ValueError):
        validate(pd.concat([good, good.iloc[[0]]]))
    bad = good.copy(); bad.loc[0, "marketing_event"] = 2
    with pytest.raises(ValueError):
        validate(bad)
    gap = good.drop(index=5)
    with pytest.raises(ValueError):
        validate(gap)


def test_calendar_derivation():
    d = add_calendar(make_synthetic())
    assert set(d.day_type) == {"Event", "Holiday", "Regular"}
    assert d["week"].min() == 1 and d["week"].max() == 5
    assert (d.loc[d.day_type == "Event", "dow"].isin([4, 5])).all()
    assert (d.loc[d.day_type == "Holiday", "dow"] == 6).all()
    assert d["t"].tolist() == list(range(30))


# ---- output contract ---------------------------------------------------------
def test_full_output_contract(result):
    check_contract(result)


def test_group_sizes_match_prereg(result):
    p = result["primary"]
    assert (p["n_a"], p["n_b"]) == (8, 18)
    assert result["twoway_n"] == 28
    assert len(result["features"]) == 28


def test_planted_effect_recovered(result):
    p = result["primary"]
    assert p["diff"] > 20 and p["diff"] < 40
    assert p["reject_null"] and p["primary_p"] < 0.05
    assert p["hl_shift"] > 0 and p["hedges_g"] > 1
    assert p["welch_ci_low"] > 0
    assert result["oneway"]["p"] < 0.05


def test_null_false_positive_rate_nominal():
    rej = 0
    n = 400
    for s in range(n):
        d = add_calendar(make_synthetic(seed=1000 + s, event_effect=0.0))
        a = d.loc[d.day_type == "Event", "demand"]; b = d.loc[d.day_type == "Regular", "demand"]
        rej += stats.ttest_ind(a, b, equal_var=False).pvalue < 0.05
    assert 0.02 <= rej / n <= 0.09


def test_normality_and_rule_selection():
    d = add_calendar(make_synthetic(seed=3))
    a = d.loc[d.day_type == "Event", "demand"]; b = d.loc[d.day_type == "Regular", "demand"]
    r = st.two_sample(a, b, n_boot=500, n_perm=500)
    assert r["primary_test"] in ("welch_t", "mann_whitney_u")
    # rule: primary is Welch iff both Shapiro p >= .05
    assert (r["primary_test"] == "welch_t") == (r["shapiro_p_a"] >= .05 and r["shapiro_p_b"] >= .05)


def test_twoway_has_interaction_row(result):
    assert "C(week):C(day_type)" in result["twoway"].index
    assert result["twoway"]["PR(>F)"].dropna().between(0, 1).all()


# ---- PCA / clustering --------------------------------------------------------
def test_pca_contract(result):
    pca = result["pca"]
    assert pca["scores"].shape == (28, 11)
    k = pca["n_keep"]
    assert pca["cum"][k - 1] >= 0.80 - 1e-9
    assert k == 2 or pca["cum"][k - 2] < 0.80


def test_kmeans_picks_argmax_silhouette(result):
    t = result["kmeans_table"]
    assert t["k"].tolist() == [2, 3, 4, 5, 6]
    assert result["k_best"] == int(t.loc[t.silhouette.idxmax(), "k"])
    assert len(result["kmeans_labels"]) == 28
    assert len(result["hier"]["labels"]) == 28 and len(set(result["hier"]["labels"])) == result["k_best"]


def test_clustering_recovers_two_obvious_blobs():
    rng = np.random.default_rng(0)
    P = np.r_[rng.normal(0, .3, (20, 2)), rng.normal(6, .3, (20, 2))]
    t, k, lab = cl.kmeans_sweep(P)
    assert k == 2 and t.silhouette.max() > 0.8
    assert len(set(lab[:20])) == 1 and len(set(lab[20:])) == 1 and lab[0] != lab[-1]
    db = cl.run_dbscan(P)
    assert db["n_clusters"] >= 2


# ---- time series -------------------------------------------------------------
def test_decomposition_periodic():
    y = fc.as_series(add_calendar(make_synthetic(seed=2, weekly_amp=15)))
    classical, stl = fc.decompose(y)
    s = classical.seasonal.to_numpy()
    assert np.allclose(s[:-7], s[7:])
    assert np.allclose(classical.trend.dropna() + classical.seasonal[classical.trend.notna()] + classical.resid.dropna(),
                       y[classical.trend.notna()])


def test_adf_detects_random_walk_vs_noise():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2026-01-01", periods=120, freq="D")
    noise = pd.Series(rng.normal(0, 1, 120), index=idx)
    walk = noise.cumsum()
    assert fc.adf_table(noise)[1] == 0
    assert fc.adf_table(walk)[1] >= 1


def test_forecast_contract_and_beats_naive_on_seasonal_fixture():
    d = make_synthetic(seed=5, n=30, weekly_amp=20, sd=2.0)
    r = run_pipeline(d, fast=True)
    f = r["forecast"]
    assert len(f) == 30 and f.index[0] == pd.Timestamp("2026-07-01")
    h = r["holdout"].set_index("model")
    assert h.loc["SeasonalNaive", "RMSE"] < h.loc["Naive", "RMSE"]
    assert (r["sarima_grid"]["aicc"].diff().dropna() >= 0).all()


def test_metrics_known_values():
    assert fc.rmse([1, 2, 3], [1, 2, 3]) == 0
    assert fc.mape([100, 200], [110, 180]) == pytest.approx(10.0)
    assert fc.mae([1, 2], [2, 4]) == pytest.approx(1.5)
    np.testing.assert_array_equal(fc.seasonal_naive(np.arange(14), 9), np.r_[7:14, 7, 8])


# ---- determinism -------------------------------------------------------------
def test_determinism(planted):
    a = run_pipeline(planted, fast=True)
    b = run_pipeline(planted, fast=True)
    for k in ("welch_p", "mwu_p", "perm_p", "hl_ci_low", "hl_ci_high", "hedges_g"):
        assert a["primary"][k] == b["primary"][k]
    np.testing.assert_allclose(a["forecast"].to_numpy(), b["forecast"].to_numpy())
    np.testing.assert_array_equal(a["kmeans_labels"], b["kmeans_labels"])


def test_pipeline_does_not_touch_raw_file(monkeypatch, planted):
    import builtins
    real_open = builtins.open
    def guard(f, *a, **k):
        assert "data/raw" not in str(f), "pipeline must not read real data"
        return real_open(f, *a, **k)
    monkeypatch.setattr(builtins, "open", guard)
    run_pipeline(planted, fast=True)


# ---- capstone rolling-origin -------------------------------------------------
def test_capstone_contract():
    from rq import capstone as cs
    d = add_calendar(make_synthetic(seed=7, weekly_amp=15, sd=3.0))
    p = cs.rolling_origin(d)
    assert len(p) == 16 and p["date"].iloc[0] == d["date"].iloc[14]
    assert p[["actual", "Naive", "SeasonalNaive", "SARIMA", "Ridge"]].notna().all().all()
    tab = cs.score(p)
    assert set(tab.model) == {"Naive", "SeasonalNaive", "SARIMA", "Ridge"}
    assert (tab[["MAE", "RMSE", "MAPE"]] >= 0).all().all()
    cmp = cs.paired_vs_baseline(p, n_boot=500)
    assert (cmp.ci_low <= cmp.ci_high).all()
    # strong weekly pattern + calendar features: Ridge/SeasonalNaive must beat plain naive
    r = tab.set_index("model")["RMSE"]
    assert min(r["Ridge"], r["SeasonalNaive"]) < r["Naive"]
