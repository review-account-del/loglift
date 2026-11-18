"""
run_gamma_v05.py - empirical preferential-sampling exponent of the real
species (review R2, lines 155-157).

(1) Binned estimate, as in v0.4 but documented: 40 quantile breakpoints of
    log(1 + E); because E takes few distinct values in the lower half of the
    distribution, duplicated breakpoints collapse and 16 distinct bins remain.
    Bins with < 3 focal cells or < 100 cells are dropped; slope of
    log(share of cells occupied) on log(mean E of bin).
(2) Maximum-likelihood estimate on all cells, no binning: Poisson regression of
    the number of occupied cells per distinct effort value on log E with
    offset log(number of cells), i.e. P(occupied | E) = a E^gamma.
"""
import numpy as np, pandas as pd, warnings
import statsmodels.api as sm
from bench_v05 import *

x = np.log1p(E)
q = np.unique(np.quantile(x, np.linspace(0, 1, 41)))
b = np.clip(np.digitize(x, q[1:-1], right=True), 0, len(q) - 2)
n_bin = np.bincount(b, minlength=len(q) - 1)
E_bin = np.array([E[b == k].mean() for k in range(len(q) - 1)])
lev, inv, n_lev = np.unique(E, return_inverse=True, return_counts=True)
rows = []
for sp, grp in SPECIES:
    iF = focal_idx(sp)
    f_bin = np.bincount(b[iF], minlength=len(q) - 1)
    ok = (f_bin >= 3) & (n_bin >= 100)
    g_bin = r_bin = np.nan
    if ok.sum() >= 3:
        xx, yy = np.log(E_bin[ok]), np.log(f_bin[ok] / n_bin[ok])
        g_bin = np.polyfit(xx, yy, 1)[0]; r_bin = np.corrcoef(xx, yy)[0, 1]
    y = np.bincount(inv[iF], minlength=len(lev))
    X = sm.add_constant(np.log(lev))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = sm.GLM(y, X, family=sm.families.Poisson(), offset=np.log(n_lev)).fit()
    rows.append(dict(species=sp, n_cells=len(iF), gamma_binned=g_bin, r_binned=r_bin, n_bins_used=int(ok.sum()),
                     n_bins_total=len(q) - 1, gamma_ml=fit.params[1], gamma_ml_se=fit.bse[1],
                     top10_share=float((E[iF] > np.quantile(E, 0.9)).mean())))
df = pd.DataFrame(rows).sort_values("gamma_ml")
df.to_csv(f"{RES}/gamma_empirical.csv", index=False)
print("distinct bins:", len(q) - 1); print(df.round(3).to_string(index=False))

# (3) calibration by effort profile: the gamma at which a pseudo-species drawn
# with P ~ E^gamma has the same share of its cells in the top effort decile as
# the real species (and the same mean log E)
from scipy.optimize import brentq
top = E > np.quantile(E, 0.9); lx = np.log(E)
def top_share(g):
    w = E ** g; return w[top].sum() / w.sum()
def mean_logE(g):
    w = E ** g; return (w * lx).sum() / w.sum()
cal = pd.DataFrame([dict(gamma=g, top10_share=top_share(g), mean_logE=mean_logE(g)) for g in (0, 0.5, 1, 1.5, 2, 3)])
print(cal.round(3).to_string(index=False))
df["gamma_match_top10"] = [brentq(lambda g: top_share(g) - s, -2, 6) for s in df.top10_share]
df["gamma_match_meanlogE"] = [brentq(lambda g: mean_logE(g) - lx[focal_idx(sp)].mean(), -2, 6) for sp in df.species]
df.to_csv(f"{RES}/gamma_empirical.csv", index=False); cal.to_csv(f"{RES}/gamma_calibration.csv", index=False)
print(df[["species", "n_cells", "gamma_binned", "gamma_ml", "top10_share", "gamma_match_top10", "gamma_match_meanlogE"]].round(2).to_string(index=False))
