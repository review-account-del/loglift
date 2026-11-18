"""
loglift_core.py - logLift and the estimators it is compared with (revision v0.5).

Changes from v0.4 (loglift_bench.py), made in response to review:
  * Effort strata respect ties: cells with the same effort value always fall in
    the same stratum. The K nominal rank-quantile strata therefore collapse to
    K_eff <= K effective strata (reported). v0.4 broke ties by file order and
    merged the bottom 40 % of strata by hand.
  * Everything is computed from count tables. A focal species and its
    background are summarised as counts over (effort level x property class),
    where the property class of a cell is the number of thresholds it reaches.
    All estimators are exact functions of these tables, so point estimates use
    the full background (no subsampling) and cost almost nothing.
  * Bootstrap: focal sites are resampled with replacement (site bootstrap) or
    in spatial blocks (block bootstrap, H3 parent cells); the full background
    is resampled with replacement as a multinomial draw on its count table.
    The interval therefore refers to the same quantity as the point estimate.
  * New comparators: binomial GLM with a restricted cubic spline of
    log(1 + E); binomial GLM with log-linear effort fitted by Firth's
    penalised likelihood. All GLM fits return diagnostics (convergence,
    iterations, focal coefficient and its SE, separation flag).

Notation: F = focal sites, B = background sites, T = thresholds,
class c(j) = #{t in T: V_j >= t}, so that V_j >= T[i]  <=>  c(j) > i.
"""
from __future__ import annotations
import numpy as np
import warnings
from scipy import stats
from scipy.special import expit
import statsmodels.api as sm

ALPHA = 0.5   # Jeffreys prior: (c + 1/2) / (n + 1)


# --------------------------------------------------------------- strata -----
def effort_strata(effort, K):
    """Tie-respecting rank-quantile strata of log(1 + effort).
    Breakpoints are the empirical k/K quantiles (k = 1..K-1); duplicated
    breakpoints are dropped, and a cell goes to stratum s if
    b[s-1] < x <= b[s]. Cells with equal effort therefore share a stratum.
    Returns (codes 0..K_eff-1, K_eff)."""
    x = np.log1p(np.asarray(effort, float))
    if K == "exact" or K is None:
        u, s = np.unique(x, return_inverse=True)
        return s.astype(np.int32), len(u)
    if K <= 1:
        return np.zeros(len(x), np.int32), 1
    br = np.unique(np.quantile(x, np.arange(1, K) / K, method="inverted_cdf"))
    s = np.searchsorted(br, x, side="left")
    u, s = np.unique(s, return_inverse=True)
    return s.astype(np.int32), len(u)


def exact_strata(effort):
    u, s = np.unique(np.asarray(effort, float), return_inverse=True)
    return s.astype(np.int32), len(u)


def classes(V, T):
    """c(j) = number of thresholds reached by V_j."""
    return np.searchsorted(np.asarray(T, float), np.asarray(V, float), side="right").astype(np.int16)


def exceed(tab):
    """tab[..., c] counts of class c -> counts with class > i, i = 0..C-2 (i.e. V >= T[i])."""
    rc = np.cumsum(tab[..., ::-1], axis=-1)[..., ::-1]
    return rc[..., 1:]


def lj(c, n):
    return np.log2((c + ALPHA) / (n + 2 * ALPHA))


# ------------------------------------------------------------- logLift ------
def loglift_tab(F, B, common_support=True):
    """F, B: tables (strata x classes) for focal and background sites.
    Returns lift per threshold. Focal sites in strata without background sites
    have no comparison and are dropped (common support); with K <= 10,000
    nominal strata on these data no stratum is empty."""
    F = np.asarray(F, float); B = np.asarray(B, float)
    nB = B.sum(); nB_s = B.sum(1)
    ok = nB_s > 0
    if common_support:
        F = F[ok]; B = B[ok]; nB_s = nB_s[ok]; ok = np.ones(len(nB_s), bool)
    nF_s = F.sum(1); nF = nF_s.sum()
    pi = nF_s / nF
    q_s = np.zeros((F.shape[0], F.shape[1] - 1))
    q_s[ok] = exceed(B[ok]) / nB_s[ok, None]
    q = (pi[:, None] * q_s).sum(0) / pi[ok].sum()
    cF = exceed(F.sum(0))
    return lj(cF, nF) - lj(q * nB, nB)


def raw_tab(F1, B1):
    F1 = np.asarray(F1, float); B1 = np.asarray(B1, float)
    return lj(exceed(F1), F1.sum()) - lj(exceed(B1), B1.sum())


def cooc_tab(F1, B1):
    """PMI, NPMI, Fisher log2 OR (0.5 added to every cell), one-sided Fisher p and
    Jaccard, per threshold, from the 2 x 2 table of focal presence x exceedance over
    all cells (focal + background)."""
    F1 = np.asarray(F1, float); B1 = np.asarray(B1, float)
    nF, nB = F1.sum(), B1.sum(); N = nF + nB
    a = exceed(F1); c = exceed(B1); b = nF - a; d = nB - c
    pF, pE, pFE = nF / N, (a + c) / N, np.maximum(a, 0.5) / N
    pmi = np.log2(pFE / (pF * pE))
    npmi = pmi / (-np.log2(pFE))
    lor = np.log2(((a + .5) * (d + .5)) / ((b + .5) * (c + .5)))
    jac = np.where(a + b + c > 0, a / np.maximum(a + b + c, 1), np.nan)
    p = np.array([stats.fisher_exact([[int(a[i]), int(b[i])], [int(c[i]), int(d[i])]], alternative="greater")[1]
                  for i in range(len(a))]) if nF < 1e6 else np.full(len(a), np.nan)
    return dict(pmi=pmi, npmi=npmi, fisher_log2OR=lor, fisher_p=p, jaccard=jac)


# ----------------------------------------------------------------- GLMs -----
def rcs_basis(x, knots):
    """Restricted (natural) cubic spline basis (Harrell): columns x and k-2
    non-linear terms; linear beyond the outer knots."""
    x = np.asarray(x, float); t = np.asarray(knots, float); k = len(t)
    cols = [x]
    sc = (t[-1] - t[0]) ** 2
    for j in range(k - 2):
        cols.append((np.maximum(x - t[j], 0) ** 3
                     - np.maximum(x - t[k - 2], 0) ** 3 * (t[k - 1] - t[j]) / (t[k - 1] - t[k - 2])
                     + np.maximum(x - t[k - 1], 0) ** 3 * (t[k - 2] - t[j]) / (t[k - 1] - t[k - 2])) / sc)
    return np.column_stack(cols)


def _fit_ml(X, y, n):
    """Binomial GLM (logit) on aggregated rows, maximum likelihood (statsmodels IRLS)."""
    Y = np.column_stack([y, n - y])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = sm.GLM(Y, X, family=sm.families.Binomial()).fit(maxiter=100)
    try:
        bse = np.asarray(fit.bse)
    except Exception:
        bse = np.full(X.shape[1], np.nan)
    return dict(params=np.asarray(fit.params), bse=bse, converged=bool(fit.converged),
                iterations=int(fit.fit_history.get("iteration", -1)), deviance=float(fit.deviance),
                pearson_chi2=float(fit.pearson_chi2), df_resid=float(fit.df_resid))


def _fit_firth(X, y, n, maxit=200, tol=1e-7):
    """Firth (1993) bias-reduced logistic regression on aggregated binomial rows.
    Modified score U*(b) = X'(y - n p + h (1/2 - p)), h = leverages of
    W^(1/2) X (X'WX)^-1 X' W^(1/2), W = n p (1 - p) (Heinze and Schemper 2002).
    Newton steps on the penalised log-likelihood l(b) + 1/2 log|X'WX|, steps
    capped at 5 and halved until the penalised likelihood increases.
    Converged = full Newton step below tol."""
    X = np.asarray(X, float); y = np.asarray(y, float); n = np.asarray(n, float)
    b = np.zeros(X.shape[1])
    p0 = np.clip((y.sum() + .5) / (n.sum() + 1), 1e-9, 1 - 1e-9); b[0] = np.log(p0 / (1 - p0))

    def pen_ll(b):
        eta = X @ b
        p = expit(eta)
        W = n * p * (1 - p)
        sign, logdet = np.linalg.slogdet((X * W[:, None]).T @ X)
        if sign <= 0 or not np.isfinite(logdet):
            return -np.inf, p, W
        ll = -np.sum(y * np.logaddexp(0, -eta) + (n - y) * np.logaddexp(0, eta))
        return ll + 0.5 * logdet, p, W
    ll, p, W = pen_ll(b)
    conv, it = False, 0
    for it in range(1, maxit + 1):
        XtWX = (X * W[:, None]).T @ X
        inv = np.linalg.pinv(XtWX)
        h = W * np.einsum("ij,jk,ik->i", X, inv, X)
        U = X.T @ (y - n * p + h * (0.5 - p))
        step = inv @ U
        mx = np.max(np.abs(step))
        if mx < tol:
            conv = True
            break
        if mx > 5:
            step = step * (5 / mx)
        lam, ok = 1.0, False
        for _ in range(40):
            nb = b + lam * step
            nll, npp, nW = pen_ll(nb)
            if np.isfinite(nll) and nll >= ll - 1e-12 * (abs(ll) + 1):
                ok = True
                break
            lam /= 2
        if not ok:
            break
        b, ll, p, W = nb, nll, npp, nW
    XtWX = (X * W[:, None]).T @ X
    cov = np.linalg.pinv(XtWX)
    return dict(params=b, bse=np.sqrt(np.clip(np.diag(cov), 0, None)), converged=conv, iterations=it,
                deviance=np.nan, pearson_chi2=np.nan, df_resid=float(len(y) - X.shape[1]))


def glm_gcomp_tab(Fx, Bx, xval, kind="linear", knots=None, firth=False, diag=False):
    """Binomial GLM  logit P(V >= t) = b0 + bF * focal + f(x),  x = log(1 + E) on a
    grid of effort bins. Fx, Bx: tables (x-bins x classes) for focal and
    background; xval: x value of each bin. f(x) is linear (kind='linear'), a
    restricted cubic spline (kind='spline', knots) or a stratum factor
    (kind='factor', bins = strata). The lift per threshold is g-computed:
    log2( sum_x pi_x P(t | focal, x) / sum_x pi_x P(t | background, x) ),
    pi_x = focal composition over bins."""
    Fx = np.asarray(Fx, float); Bx = np.asarray(Bx, float)
    nFx, nBx = Fx.sum(1), Bx.sum(1)
    pi = nFx / nFx.sum()
    if kind == "linear":
        Z = np.asarray(xval, float)[:, None]
    elif kind == "spline":
        Z = rcs_basis(xval, knots)
    elif kind == "factor":
        Z = np.eye(len(xval))[:, 1:]
    else:
        raise ValueError(kind)
    mF, mB = nFx > 0, nBx > 0
    X1 = np.column_stack([np.ones(len(xval)), np.ones(len(xval)), Z])
    X0 = np.column_stack([np.ones(len(xval)), np.zeros(len(xval)), Z])
    X = np.vstack([X1[mF], X0[mB]])
    if kind == "factor":                       # drop dummies of empty strata
        keep = np.r_[True, True, (np.abs(X[:, 2:]).sum(0) > 0)]
        X, X1, X0 = X[:, keep], X1[:, keep], X0[:, keep]
    cF, cB = exceed(Fx), exceed(Bx)
    out, dg = [], []
    for i in range(cF.shape[1]):
        y = np.r_[cF[mF, i], cB[mB, i]]; n = np.r_[nFx[mF], nBx[mB]]
        fit = _fit_firth(X, y, n) if firth else _fit_ml(X, y, n)
        b = fit["params"]
        p1 = (pi * expit(X1 @ b)).sum(); p0 = (pi * expit(X0 @ b)).sum()
        out.append(np.log2(p1 / p0))
        if diag:
            sepF = (cF[mF, i].sum() == 0) or (cF[mF, i].sum() == nFx.sum())
            dg.append(dict(t_index=i, converged=fit["converged"], iterations=fit["iterations"],
                           b_focal=b[1], se_focal=fit["bse"][1],
                           b_effort=(b[2] if kind == "linear" else np.nan),
                           se_effort=(fit["bse"][2] if kind == "linear" else np.nan),
                           deviance=fit["deviance"], pearson_chi2=fit["pearson_chi2"], df_resid=fit["df_resid"],
                           focal_exceed=float(cF[mF, i].sum()), n_focal=float(nFx.sum()), separation=bool(sepF)))
    return (np.array(out), dg) if diag else np.array(out)


def glm_nb(Vf, xf, Vb, xb):
    """Negative-binomial regression of V on focal + log(1 + E); log2 rate ratio."""
    V = np.r_[Vf, Vb].astype(float)
    X = sm.add_constant(np.column_stack([np.r_[np.ones(len(Vf)), np.zeros(len(Vb))], np.r_[xf, xb]]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            b = sm.NegativeBinomial(V, X).fit(disp=0, maxiter=200).params[1]
        except Exception:
            b = sm.GLM(V, X, family=sm.families.Poisson()).fit().params[1]
    return float(b / np.log(2))


# --------------------------------------------------------------- universe ---
class Universe:
    """All recorded cells with an effort array. Effort levels are the distinct
    effort values; strata, x-bins and spline knots are functions of the level."""

    def __init__(self, effort, Ks=(10, 100), xround=2):
        self.E = np.asarray(effort, float)
        self.lev, self.L = exact_strata(self.E)                      # level code per cell
        self.Elev = np.unique(self.E)                                 # effort value per level
        self.S, self.Keff = {}, {}
        for K in Ks:
            self.add_strata(K)
        xr = np.round(np.log1p(self.Elev), xround)
        self.xval, self.lev2x = np.unique(xr, return_inverse=True)   # x-bin per level
        x = np.log1p(self.E)
        self.knots = np.unique(np.quantile(x, [0.5, 0.65, 0.8, 0.9, 0.95, 0.99, 0.999]))

    def add_strata(self, K):
        s, k = effort_strata(self.E, K) if K != "exact" else exact_strata(self.E)
        # strata are functions of the level
        m = np.zeros(self.L, np.int32); m[self.lev] = s
        self.S[K], self.Keff[K] = m, k

    # joint table over (level, class[, tg])
    def joint(self, idx, cls, C, tg=None):
        if tg is None:
            return np.bincount(self.lev[idx] * C + cls[idx], minlength=self.L * C).reshape(self.L, C)
        return np.bincount((self.lev[idx] * C + cls[idx]) * 2 + tg[idx],
                           minlength=self.L * C * 2).reshape(self.L, C, 2)

    def collapse(self, Jlc, mapping, n):
        out = np.zeros((n, Jlc.shape[1])); np.add.at(out, mapping, Jlc)
        return out


def estimates_from_joint(U, JF, JB, which=("raw", "tgb", "ll10", "ll100", "glm_factor", "glm_linear",
                                            "glm_spline", "glm_firth", "cooc"), diag=False, extraK=()):
    """JF, JB: joint tables (level x class x tg). Returns dict of per-threshold arrays."""
    F = JF.sum(2); B = JB.sum(2)
    r, D = {}, {}
    if "raw" in which:
        r["raw"] = raw_tab(F.sum(0), B.sum(0))
    if "tgb" in which:
        r["tgb"] = raw_tab(F.sum(0), JB[:, :, 1].sum(0))
    for K, name in [(10, "ll10"), (100, "ll100")] + [(k, f"ll{k}") for k in extraK]:
        if name in which or K in extraK:
            m = U.S[K]; n = U.Keff[K]
            r[name] = loglift_tab(U.collapse(F, m, n), U.collapse(B, m, n))
    if "glm_factor" in which:
        m = U.S[100]; n = U.Keff[100]
        r["glm_factor"] = glm_gcomp_tab(U.collapse(F, m, n), U.collapse(B, m, n), np.arange(n), "factor")
    Fx = Bx = None
    if any(k in which for k in ("glm_linear", "glm_spline", "glm_firth", "glm_spline_firth")):
        nx = len(U.xval)
        Fx, Bx = U.collapse(F, U.lev2x, nx), U.collapse(B, U.lev2x, nx)
    for name, kw in [("glm_linear", dict(kind="linear")), ("glm_spline", dict(kind="spline", knots=U.knots)),
                     ("glm_firth", dict(kind="linear", firth=True)),
                     ("glm_spline_firth", dict(kind="spline", knots=U.knots, firth=True))]:
        if name in which:
            if diag:
                r[name], D[name] = glm_gcomp_tab(Fx, Bx, U.xval, diag=True, **kw)
            else:
                r[name] = glm_gcomp_tab(Fx, Bx, U.xval, **kw)
    if "cooc" in which:
        r.update(cooc_tab(F.sum(0), B.sum(0)))
    return (r, D) if diag else r


def mean_over_t(r):
    return {k: float(np.nanmean(v)) for k, v in r.items() if k != "fisher_p"}


def multinomial_table(rng, J):
    """Resample a table of counts with replacement (same total): multinomial draw."""
    flat = J.ravel().astype(float); n = int(flat.sum())
    return rng.multinomial(n, flat / flat.sum()).reshape(J.shape)
