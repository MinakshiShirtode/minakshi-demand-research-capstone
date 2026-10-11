"""PCA + unsupervised clustering (Task 3)."""
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, cophenet
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from .config import SEED

FEATURES = ["demand", "lag1", "lag2", "diff1", "roll3_mean", "roll3_std",
            "dow_sin", "dow_cos", "marketing_event", "holiday", "t"]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)
    y = df["demand"].astype(float)
    f["demand"] = y
    f["lag1"] = y.shift(1)
    f["lag2"] = y.shift(2)
    f["diff1"] = y.diff()
    f["roll3_mean"] = y.rolling(3).mean()
    f["roll3_std"] = y.rolling(3).std(ddof=1)
    f["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7)
    f["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7)
    f["marketing_event"] = df["marketing_event"]
    f["holiday"] = df["holiday"]
    f["t"] = df["t"]
    return f.dropna()[FEATURES]


def run_pca(X: pd.DataFrame, var_target=0.80):
    Z = StandardScaler().fit_transform(X.to_numpy(float))
    pca = PCA(random_state=SEED).fit(Z)
    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)
    n_keep = int(max(2, np.searchsorted(cum, var_target - 1e-12) + 1))
    n_kaiser = int((pca.explained_variance_ > 1).sum())
    scores = pca.transform(Z)
    return dict(Z=Z, pca=pca, evr=evr, cum=cum, n_keep=n_keep, n_kaiser=n_kaiser,
                eigenvalues=pca.explained_variance_, scores=scores,
                loadings=pd.DataFrame(pca.components_.T, index=X.columns,
                                      columns=[f"PC{i+1}" for i in range(len(evr))]))


def kmeans_sweep(P, k_range=range(2, 7)):
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=50, random_state=SEED).fit(P)
        rows.append(dict(k=k, inertia=km.inertia_,
                         silhouette=silhouette_score(P, km.labels_)))
    tab = pd.DataFrame(rows)
    best_k = int(tab.loc[tab["silhouette"].idxmax(), "k"])
    km = KMeans(n_clusters=best_k, n_init=50, random_state=SEED).fit(P)
    return tab, best_k, km.labels_


def knee_eps(P, min_samples=3):
    nn = NearestNeighbors(n_neighbors=min_samples).fit(P)
    d = np.sort(nn.kneighbors(P)[0][:, -1])
    x = np.arange(len(d), dtype=float)
    p1, p2 = np.array([x[0], d[0]]), np.array([x[-1], d[-1]])
    v = (p2 - p1) / np.linalg.norm(p2 - p1)
    pts = np.c_[x, d] - p1
    dist = np.abs(pts[:, 0] * v[1] - pts[:, 1] * v[0])
    return float(d[int(dist.argmax())]), d


def run_dbscan(P, min_samples=3):
    eps, kd = knee_eps(P, min_samples)
    lab = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(P)
    n_cl = len(set(lab) - {-1})
    sil = silhouette_score(P[lab != -1], lab[lab != -1]) if n_cl >= 2 and (lab != -1).sum() > n_cl else np.nan
    return dict(eps=eps, labels=lab, n_clusters=n_cl, n_noise=int((lab == -1).sum()),
                silhouette=sil, kdist=kd)


def run_hier(P, k):
    lab = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(P)
    Zl = linkage(P, method="ward")
    coph = cophenet(Zl, pdist(P))[0]
    return dict(labels=lab, linkage=Zl, cophenetic=coph,
                silhouette=silhouette_score(P, lab) if len(set(lab)) > 1 else np.nan)


def agreement(labels: dict, day_type) -> pd.DataFrame:
    names = list(labels) + ["day_type"]
    all_l = {**labels, "day_type": np.asarray(day_type)}
    m = pd.DataFrame(index=names, columns=names, dtype=float)
    for a in names:
        for b in names:
            m.loc[a, b] = adjusted_rand_score(all_l[a], all_l[b])
    return m
