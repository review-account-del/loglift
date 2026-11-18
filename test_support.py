import numpy as np, pandas as pd
import loglift_core as lc
from bench_v05 import *
U = universe(); 
for K in (1000, 3000, 10000, "exact"): U.add_strata(K)
def ll_cs(F, B):
    F = np.asarray(F, float); B = np.asarray(B, float)
    nB = B.sum(); nB_s = B.sum(1); ok = nB_s > 0
    Fk = F[ok]; nF = Fk.sum(); pi = Fk.sum(1) / nF
    q = (pi[:, None] * (lc.exceed(B[ok]) / nB_s[ok, None])).sum(0)
    return lc.lj(lc.exceed(Fk.sum(0)), nF) - lc.lj(q * nB, nB)
rows = []
for prop in PROPS:
    V, T = prop_values(prop); C = len(T) + 1; cls = lc.classes(V, T)
    for g in (0.0, 1.0, 2.0):
        p = E ** g; p /= p.sum()
        for rep in range(8):
            rr = np.random.default_rng(SEED + 31 * rep + int(g * 10))
            iF = np.sort(rr.choice(N, 2000, replace=False, p=p))
            JF, JB = tables(U, iF, cls, C, TG["Tracheophyta"]); F, B = JF.sum(2), JB.sum(2)
            for K in (100, 1000, 3000, 10000, "exact"):
                Fs, Bs = U.collapse(F, U.S[K], U.Keff[K]), U.collapse(B, U.S[K], U.Keff[K])
                rows.append(dict(prop=prop, gamma=g, K=str(K), Keff=U.Keff[K], v04=np.mean(lc.loglift_tab(Fs, Bs)), common=np.mean(ll_cs(Fs, Bs)),
                                 dropped=float(Fs[Bs.sum(1) == 0].sum() / Fs.sum())))
d = pd.DataFrame(rows)
print(d.groupby(["prop","gamma","K","Keff"])[["v04","common","dropped"]].mean().round(3).to_string())
