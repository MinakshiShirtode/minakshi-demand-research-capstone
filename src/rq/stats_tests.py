"""Inferential statistics (Task 2). Pure functions; no file I/O."""
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.diagnostic import lilliefors

from .config import ALPHA, N_BOOT, N_PERM, SEED


def normality_table(groups: dict) -> pd.DataFrame:
    rows = []
    for name, x in groups.items():
        x = np.asarray(x, float)
        sw = stats.shapiro(x) if len(x) >= 3 else (np.nan, np.nan)
        sd = x.std(ddof=1) if len(x) > 1 else np.nan
        ks = stats.kstest(x, "norm", args=(x.mean(), sd)) if sd > 0 else (np.nan, np.nan)
        lf = lilliefors(x, dist="norm", pvalmethod="table") if len(x) >= 4 else (np.nan, np.nan)
        rows.append(dict(group=name, n=len(x), mean=x.mean(), sd=sd,
                         shapiro_W=sw[0], shapiro_p=sw[1],
                         ks_D=ks[0], ks_p=ks[1],
                         lilliefors_D=lf[0], lilliefors_p=lf[1],
                         normal_by_shapiro=bool(sw[1] >= ALPHA)))
    return pd.DataFrame(rows)


def hedges_g(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    n1, n2 = len(a), len(b)
    sp = np.sqrt(((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2))
    d = (a.mean() - b.mean()) / sp
    j = 1 - 3 / (4 * (n1 + n2) - 9)
    g = d * j
    se = np.sqrt((n1 + n2) / (n1 * n2) + g ** 2 / (2 * (n1 + n2)))
    z = stats.norm.ppf(1 - ALPHA / 2)
    return g, g - z * se, g + z * se


def hodges_lehmann(a, b, n_boot=N_BOOT, seed=SEED):
    a, b = np.asarray(a, float), np.asarray(b, float)
    est = np.median(np.subtract.outer(a, b))
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        aa = rng.choice(a, len(a)); bb = rng.choice(b, len(b))
        boots[i] = np.median(np.subtract.outer(aa, bb))
    lo, hi = np.percentile(boots, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
    return est, lo, hi


def two_sample(a, b, n_boot=N_BOOT, n_perm=N_PERM, seed=SEED) -> dict:
    """a = exposed (event), b = comparator (regular). Welch t, Mann-Whitney, effect sizes."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    t = stats.ttest_ind(a, b, equal_var=False)
    ci = t.confidence_interval(1 - ALPHA)
    mw = stats.mannwhitneyu(a, b, alternative="two-sided", method="auto")
    rbc = 2 * mw.statistic / (len(a) * len(b)) - 1
    g, g_lo, g_hi = hedges_g(a, b)
    hl, hl_lo, hl_hi = hodges_lehmann(a, b, n_boot, seed)
    perm = stats.permutation_test((a, b), lambda x, y: np.mean(x) - np.mean(y),
                                  n_resamples=n_perm, alternative="two-sided",
                                  random_state=seed)
    sw_a, sw_b = (stats.shapiro(a).pvalue, stats.shapiro(b).pvalue)
    normal_ok = bool(sw_a >= ALPHA and sw_b >= ALPHA)
    primary = "welch_t" if normal_ok else "mann_whitney_u"
    primary_p = t.pvalue if normal_ok else mw.pvalue
    return dict(
        n_a=len(a), n_b=len(b), mean_a=a.mean(), mean_b=b.mean(),
        sd_a=a.std(ddof=1), sd_b=b.std(ddof=1), diff=a.mean() - b.mean(),
        welch_t=t.statistic, welch_df=t.df, welch_p=t.pvalue,
        welch_ci_low=ci.low, welch_ci_high=ci.high,
        hedges_g=g, g_ci_low=g_lo, g_ci_high=g_hi,
        mwu_U=mw.statistic, mwu_p=mw.pvalue, rank_biserial=rbc,
        hl_shift=hl, hl_ci_low=hl_lo, hl_ci_high=hl_hi,
        perm_p=perm.pvalue, shapiro_p_a=sw_a, shapiro_p_b=sw_b,
        primary_test=primary, primary_p=primary_p,
        reject_null=bool(primary_p < ALPHA),
    )


def anova_oneway(df, y="demand", g="day_type", order=("Event", "Holiday", "Regular")):
    groups = [df.loc[df[g] == k, y].to_numpy(float) for k in order]
    F, p = stats.f_oneway(*groups)
    allv = np.concatenate(groups)
    grand = allv.mean()
    ss_b = sum(len(x) * (x.mean() - grand) ** 2 for x in groups)
    ss_w = sum(((x - x.mean()) ** 2).sum() for x in groups)
    k, n = len(groups), len(allv)
    ms_w = ss_w / (n - k)
    eta2 = ss_b / (ss_b + ss_w)
    omega2 = (ss_b - (k - 1) * ms_w) / (ss_b + ss_w + ms_w)
    bf = stats.levene(*groups, center="median")
    kw = stats.kruskal(*groups)
    tk = stats.tukey_hsd(*groups)
    ci = tk.confidence_interval(1 - ALPHA)
    rows = []
    for i in range(k):
        for j in range(i + 1, k):
            rows.append(dict(group1=order[i], group2=order[j],
                             mean_diff=groups[i].mean() - groups[j].mean(),
                             ci_low=ci.low[i, j], ci_high=ci.high[i, j],
                             p_adj=tk.pvalue[i, j],
                             reject=bool(tk.pvalue[i, j] < ALPHA)))
    summary = dict(F=F, p=p, df_between=k - 1, df_within=n - k, eta2=eta2,
                   omega2=max(omega2, 0.0), brown_forsythe_stat=bf.statistic,
                   brown_forsythe_p=bf.pvalue, kruskal_H=kw.statistic,
                   kruskal_p=kw.pvalue, n=n)
    return summary, pd.DataFrame(rows)


def anova_twoway(df, y="demand", max_week=4):
    d = df[(df["week"] <= max_week) & (df["day_type"].isin(["Event", "Holiday", "Regular"]))]
    model = smf.ols(f"{y} ~ C(week) * C(day_type)", data=d).fit()
    tab = sm.stats.anova_lm(model, typ=2)
    ss_res = tab.loc["Residual", "sum_sq"]
    tab["partial_eta2"] = tab["sum_sq"] / (tab["sum_sq"] + ss_res)
    tab.loc["Residual", "partial_eta2"] = np.nan
    return tab, int(len(d))


def robustness(df, y="demand"):
    d = df[df["day_type"].isin(["Event", "Regular"])].copy()
    d["event"] = (d["day_type"] == "Event").astype(int)
    a = d.loc[d.event == 1, y].to_numpy(float); b = d.loc[d.event == 0, y].to_numpy(float)
    out = {}
    out["permutation_p"] = stats.permutation_test(
        (a, b), lambda x, z: np.mean(x) - np.mean(z), n_resamples=N_PERM,
        alternative="two-sided", random_state=SEED).pvalue
    ols = smf.ols(f"{y} ~ event + t", data=d).fit(cov_type="HC3")
    out["ols_trend_coef"] = ols.params["event"]
    out["ols_trend_ci_low"], out["ols_trend_ci_high"] = ols.conf_int().loc["event"]
    out["ols_trend_p"] = ols.pvalues["event"]
    la, lb = np.log(a), np.log(b)
    lt = stats.ttest_ind(la, lb, equal_var=False)
    out["log_welch_p"] = lt.pvalue
    out["log_diff"] = la.mean() - lb.mean()
    loo = []
    for w in sorted(d["week"].unique()):
        s = d[d.week != w]
        aa = s.loc[s.event == 1, y]; bb = s.loc[s.event == 0, y]
        loo.append(dict(dropped_week=w, diff=aa.mean() - bb.mean(),
                        p=stats.ttest_ind(aa, bb, equal_var=False).pvalue))
    return out, pd.DataFrame(loo)
