"""
loglift_bench.py — the logLift estimator and the alternatives it is compared with.

Every estimator answers the same question for one focal species and one site
property V (here: number of red-listed species within the H3 k=1 disk, or the
species richness of the cell): are the focal species' sites enriched in
{V >= t} relative to background sites, averaged over thresholds t?

    lift_t  = log2( P(V >= t | focal) / P(V >= t | background) )
    logLift = mean_t lift_t

logLift (effort-stratified)
    Sites are divided into K rank-quantile strata of recording effort. The
    background proportion is directly standardised to the focal species'
    stratum composition,  q~_t = sum_s pi_s q_{s,t},  and the log2 ratio of
    Jeffreys-smoothed proportions is averaged over thresholds. Closed form.

Alternatives
    raw          proportions without stratification
    tgb          target-group background (Phillips et al. 2009): background
                 restricted to cells where the focal species' taxonomic group
                 has been recorded at all; no stratification
    glm_factor   logistic GLM  E_t ~ focal + C(effort stratum), g-computation
                 to the focal stratum composition (parametric cousin of logLift)
    glm_linear   logistic GLM  E_t ~ focal + log1p(effort_index), g-computation
    glm_nb       negative-binomial GLM  V ~ focal + log1p(effort_index),
                 log2 rate ratio (one number, not threshold-averaged)
    pmi / npmi   pointwise mutual information between "focal present" and
                 {V >= t} over all cells, and its normalised form
    fisher       log2 odds ratio of the 2x2 table (0.5 continuity), Fisher p
    jaccard      Jaccard index between the focal cell set and the {V >= t} set

All proportion-type estimators use the same Jeffreys (1/2, 1/2) smoothing, so
that differences between them are due to the effort handling and nothing else.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import warnings

ALPHA = 0.5   # Jeffreys prior


# ---------------------------------------------------------------- helpers ----
def event_matrix(V: np.ndarray, T: np.ndarray) -> np.ndarray:
    """E[j, i] = 1 if V_j >= T_i."""
    return (V[:, None] >= T[None, :])


def rank_quantile_strata(effort_index: np.ndarray, K: int, collapse_frac: float = 0.4) -> np.ndarray:
    """K rank-quantile strata of log(1 + effort), ties broken by a stable sort.
    The bottom collapse_frac*K levels, which consist of tied single-record
    cells in this data, are merged into one stratum."""
    ei = np.log1p(np.asarray(effort_index, float))
    order = np.argsort(ei, kind="mergesort")
    ranks = np.empty_like(order, dtype=float); ranks[order] = np.arange(1, len(ei) + 1)
    edges = np.linspace(0, len(ei), K + 1)[1:-1]
    s = np.clip(np.digitize(ranks, edges, right=True), 0, K - 1)
    return np.maximum(s, int(collapse_frac * K))


def smoothed_log2_ratio(cF, nF, cB, nB):
    """log2 of Jeffreys-smoothed proportions: (c + 1/2) / (n + 1)."""
    return np.log2((cF + ALPHA) / (nF + 2 * ALPHA)) - np.log2((cB + ALPHA) / (nB + 2 * ALPHA))


# ----------------------------------------------------------------- logLift ---
def loglift(Vf, sf, Vb, sb, T):
    """Effort-stratified log2 lift per threshold.

    Vf, Vb : property values of focal and background sites
    sf, sb : stratum codes (any integers) of focal and background sites
    T      : thresholds
    Returns an array lift_t, one value per threshold; logLift = its mean.
    """
    Ef, Eb = event_matrix(Vf, T), event_matrix(Vb, T)
    levels = np.unique(np.concatenate([sf, sb]))
    cf, cb = np.searchsorted(levels, sf), np.searchsorted(levels, sb)
    K = len(levels)
    nF, nB = len(Vf), len(Vb)
    pi = np.bincount(cf, minlength=K) / nF                       # focal composition
    nB_s = np.bincount(cb, minlength=K).astype(float)             # background sites per stratum
    cB_s = np.column_stack([np.bincount(cb, weights=Eb[:, i], minlength=K) for i in range(len(T))])
    ok = nB_s > 0
    q_s = np.zeros_like(cB_s); q_s[ok] = cB_s[ok] / nB_s[ok, None]
    q_std = (pi[:, None] * q_s).sum(0) / pi[ok].sum()            # directly standardised proportion
    cF = Ef.sum(0)
    return smoothed_log2_ratio(cF, nF, q_std * nB, nB)


def raw_lift(Vf, Vb, T):
    Ef, Eb = event_matrix(Vf, T), event_matrix(Vb, T)
    return smoothed_log2_ratio(Ef.sum(0), len(Vf), Eb.sum(0), len(Vb))


# -------------------------------------------------------------------- GLMs ---
def _binomial_glm_gcomp(E, focal, X_cov, X_cov_focal):
    """Fit logit(E) ~ focal + covariates on aggregated binomial rows and
    g-compute log2( P(E|focal=1) / P(E|focal=0) ) at the focal covariate
    distribution."""
    df = pd.DataFrame(X_cov, columns=[f"x{i}" for i in range(X_cov.shape[1])])
    df["focal"] = focal.astype(int); df["E"] = E.astype(int)
    keys = ["focal"] + [f"x{i}" for i in range(X_cov.shape[1])]
    agg = df.groupby(keys, as_index=False).agg(succ=("E", "sum"), n=("E", "size"))
    X = sm.add_constant(agg[keys].to_numpy(float), has_constant="add")
    y = np.column_stack([agg["succ"], agg["n"] - agg["succ"]])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = sm.GLM(y, X, family=sm.families.Binomial()).fit()
    Xf = np.column_stack([np.ones(len(X_cov_focal)), np.ones(len(X_cov_focal)), X_cov_focal])
    X0 = Xf.copy(); X0[:, 1] = 0
    return np.log2(fit.predict(Xf).mean() / fit.predict(X0).mean())


def est_glm(Vf, sf, ef, Vb, sb, eb, T, mode="factor"):
    """mode='factor': covariate = effort stratum as factor (one-hot);
       mode='linear': covariate = log1p(effort_index)."""
    if mode == "factor":
        # aggregate to (focal, stratum) binomial rows first; the one-hot design is
        # then 2K x K instead of n x K
        levels = np.unique(np.concatenate([sf, sb])); K = len(levels)
        cf, cb = np.searchsorted(levels, sf), np.searchsorted(levels, sb)
        nF_s = np.bincount(cf, minlength=K).astype(float); nB_s = np.bincount(cb, minlength=K).astype(float)
        Ef, Eb = event_matrix(Vf, T), event_matrix(Vb, T)
        pi = nF_s / nF_s.sum()
        onehot = np.eye(K)[:, 1:]
        X = np.vstack([np.column_stack([np.ones(K), np.ones(K), onehot]),
                       np.column_stack([np.ones(K), np.zeros(K), onehot])])
        X1 = np.column_stack([np.ones(K), np.ones(K), onehot]); X0 = np.column_stack([np.ones(K), np.zeros(K), onehot])
        out = []
        for i in range(len(T)):
            sF = np.bincount(cf, weights=Ef[:, i], minlength=K); sB = np.bincount(cb, weights=Eb[:, i], minlength=K)
            y = np.vstack([np.column_stack([sF, nF_s - sF]), np.column_stack([sB, nB_s - sB])])
            keep = y.sum(1) > 0
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = sm.GLM(y[keep], X[keep], family=sm.families.Binomial()).fit()
            p1 = (pi * fit.predict(X1)).sum(); p0 = (pi * fit.predict(X0)).sum()
            out.append(np.log2(p1 / p0))
        return np.array(out)
    V = np.concatenate([Vf, Vb]); focal = np.r_[np.ones(len(Vf)), np.zeros(len(Vb))]
    x = np.round(np.log1p(np.concatenate([ef, eb])), 2)
    X = x[:, None]; Xf = np.round(np.log1p(ef), 2)[:, None]
    return np.array([_binomial_glm_gcomp((V >= t), focal, X, Xf) for t in T])


def est_glm_nb(Vf, ef, Vb, eb):
    V = np.concatenate([Vf, Vb]).astype(float)
    focal = np.r_[np.ones(len(Vf)), np.zeros(len(Vb))]
    x = np.log1p(np.concatenate([ef, eb]))
    X = sm.add_constant(np.column_stack([focal, x]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            b = sm.NegativeBinomial(V, X).fit(disp=0, maxiter=200).params[1]
        except Exception:
            b = sm.GLM(V, X, family=sm.families.Poisson()).fit().params[1]
    return float(b / np.log(2))


# ---------------------------------------------------- co-occurrence measures --
def est_cooccurrence(is_focal: np.ndarray, V: np.ndarray, T):
    N = len(V); nF = is_focal.sum()
    pmi, npmi, lor, pfish, jac = [], [], [], [], []
    for t in T:
        E = V >= t
        a = int((is_focal & E).sum()); b = int(nF - a)
        c = int(E.sum() - a); d = int(N - a - b - c)
        pF, pE, pFE = nF / N, (a + c) / N, max(a, 0.5) / N
        pmi.append(np.log2(pFE / (pF * pE)))
        npmi.append(np.log2(pFE / (pF * pE)) / (-np.log2(pFE)))
        lor.append(np.log2(((a + .5) * (d + .5)) / ((b + .5) * (c + .5))))
        pfish.append(stats.fisher_exact([[a, b], [c, d]], alternative="greater")[1])
        jac.append(a / (a + b + c) if (a + b + c) > 0 else np.nan)
    return dict(pmi=np.array(pmi), npmi=np.array(npmi), fisher_log2OR=np.array(lor),
                fisher_p=np.array(pfish), jaccard=np.array(jac))


# --------------------------------------------------------------- wrapper -----
class Problem:
    """One focal species x one property. Holds arrays for focal and background:
    property V, effort index e, decile strata s10, 100-quantile strata s100,
    and the target-group mask of the background."""

    def __init__(self, Vf, ef, s10f, s100f, Vb, eb, s10b, s100b, T, tg_mask_b=None):
        self.Vf, self.ef, self.s10f, self.s100f = [np.asarray(x) for x in (Vf, ef, s10f, s100f)]
        self.Vb, self.eb, self.s10b, self.s100b = [np.asarray(x) for x in (Vb, eb, s10b, s100b)]
        self.T = np.asarray(T)
        self.tg_mask_b = None if tg_mask_b is None else np.asarray(tg_mask_b, bool)

    def subsample(self, rng, nb=None, bootstrap=True):
        nF, nB = len(self.Vf), len(self.Vb)
        i = rng.integers(0, nF, nF) if bootstrap else np.arange(nF)
        j = rng.integers(0, nB, nb if nb else nB)
        return Problem(self.Vf[i], self.ef[i], self.s10f[i], self.s100f[i],
                       self.Vb[j], self.eb[j], self.s10b[j], self.s100b[j], self.T,
                       None if self.tg_mask_b is None else self.tg_mask_b[j])

    def estimates(self, include_glm=True, include_cooc=True) -> dict:
        P = self
        r = {"raw": raw_lift(P.Vf, P.Vb, P.T),
             "loglift10": loglift(P.Vf, P.s10f, P.Vb, P.s10b, P.T),
             "loglift100": loglift(P.Vf, P.s100f, P.Vb, P.s100b, P.T)}
        if P.tg_mask_b is not None and P.tg_mask_b.sum() > 0:
            r["tgb"] = raw_lift(P.Vf, P.Vb[P.tg_mask_b], P.T)
        if include_glm:
            r["glm_factor"] = est_glm(P.Vf, P.s100f, P.ef, P.Vb, P.s100b, P.eb, P.T, "factor")
            r["glm_linear"] = est_glm(P.Vf, P.s100f, P.ef, P.Vb, P.s100b, P.eb, P.T, "linear")
            r["glm_nb"] = np.array([est_glm_nb(P.Vf, P.ef, P.Vb, P.eb)])
        if include_cooc:
            is_f = np.r_[np.ones(len(P.Vf), bool), np.zeros(len(P.Vb), bool)]
            r.update(est_cooccurrence(is_f, np.concatenate([P.Vf, P.Vb]), P.T))
        return r


def mean_over_t(r: dict) -> dict:
    return {k: float(np.nanmean(v)) for k, v in r.items()}
