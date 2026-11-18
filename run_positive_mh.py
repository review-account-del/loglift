"""
run_positive_mh.py - the scale of the score. Same positive-control draws as
run_positive_v05.py (S = E), scored with logLift (risk ratio, K = 10,000) and
with the Mantel-Haenszel common odds ratio over the same strata, at the habitat
threshold t* and averaged over thresholds. Under the planted mechanism the
within-stratum odds ratio of the habitat is rho whatever the effort profile,
whereas the standardised risk ratio shrinks where the habitat is common.
"""
import numpy as np, pandas as pd
import loglift_core as lc
from bench_v05 import *

U = universe(); U.add_strata(10000)
TSTAR = {"red": 2, "richness": 20}
m, n = U.S[10000], U.Keff[10000]


def mh_log2or(F, B):
    """Mantel-Haenszel odds ratio per threshold from stratum x class tables."""
    a = lc.exceed(F); b = F.sum(1)[:, None] - a
    c = lc.exceed(B); d = B.sum(1)[:, None] - c
    Ns = (F.sum(1) + B.sum(1))[:, None]
    num = (a * d / Ns).sum(0); den = (b * c / Ns).sum(0)
    return np.log2((num + 1e-12) / (den + 1e-12))


rows = []
for prop in PROPS:
    V, T = prop_values(prop); C = len(T) + 1; cls = lc.classes(V, T)
    H = (V >= TSTAR[prop]).astype(float); it = list(T).index(TSTAR[prop])
    for rho in (1.0, 2.0, 4.0):
        for g in (0.0, 1.0, 2.0):
            w = (E ** g) * rho ** H; p = w / w.sum()
            for rep in range(10):
                rr = np.random.default_rng(SEED + 101 * rep + int(10 * g) + int(100 * rho))
                iF = np.sort(rr.choice(N, 2000, replace=False, p=p))
                JF, JB = tables(U, iF, cls, C, TG["Tracheophyta"])
                F, B = U.collapse(JF.sum(2), m, n), U.collapse(JB.sum(2), m, n)
                ll = lc.loglift_tab(F, B); mh = mh_log2or(F, B)
                rows.append(dict(prop=prop, rho=rho, gamma=g, rep=rep, ll10000_mean=ll.mean(), ll10000_tstar=ll[it],
                                 mh_mean=mh.mean(), mh_tstar=mh[it]))
    print(prop, flush=True)
d = pd.DataFrame(rows)
s = d.groupby(["prop", "rho", "gamma"])[["ll10000_tstar", "mh_tstar", "ll10000_mean", "mh_mean"]].mean().reset_index()
s["log2_rho"] = np.log2(s.rho)
s.round(3).to_csv(f"{RES}/positive_control_mh.csv", index=False)
print(s.round(2).to_string(index=False))
