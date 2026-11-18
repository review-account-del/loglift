"""
run_mh_real.py - the Mantel-Haenszel common odds ratio over the same effort
strata (K = 10,000), for the eight species and the 200 species: point
estimates, by effort band, and in the effort-perturbation benchmark, next to
logLift. Both are closed-form functions of the same stratum x class tables.
"""
import numpy as np, pandas as pd, polars as pl, h3
from scipy.stats import spearmanr
import loglift_core as lc
from bench_v05 import *

U = universe(); U.add_strata(10000)
m, n = U.S[10000], U.Keff[10000]
tg0 = np.zeros(N, np.int64)


def mh(F, B):
    a = lc.exceed(F); b = F.sum(1)[:, None] - a
    c = lc.exceed(B); d = B.sum(1)[:, None] - c
    Ns = (F.sum(1) + B.sum(1))[:, None]
    num = (a * d / Ns).sum(0); den = (b * c / Ns).sum(0)
    return np.log2((num + 0.5 / Ns.sum()) / (den + 0.5 / Ns.sum()))


rows, bands, pert = [], [], []
for sp, grp in SPECIES:
    iF = focal_idx(sp)
    for prop in PROPS:
        V, T = prop_values(prop, sp); C = len(T) + 1; cls = lc.classes(V, T)
        JF, JB = tables(U, iF, cls, C, tg0); F, B = U.collapse(JF.sum(2), m, n), U.collapse(JB.sum(2), m, n)
        rows.append(dict(species=sp, prop=prop, n=len(iF), ll10000=lc.loglift_tab(F, B).mean(), mh10000=mh(F, B).mean()))
        if len(iF) >= 60:
            cum = np.cumsum(F.sum(1)) / F.sum(); edges = [0] + list(np.searchsorted(cum, [0.2, 0.4, 0.6, 0.8]) + 1) + [n]
            for k, (a_, b_) in enumerate(zip(edges[:-1], edges[1:])):
                if F[a_:b_].sum() > 0:
                    bands.append(dict(species=sp, prop=prop, band=k + 1, ll=lc.loglift_tab(F[a_:b_], B[a_:b_]).mean(), mh=mh(F[a_:b_], B[a_:b_]).mean()))
            iB = np.setdiff1d(np.arange(N), iF)
            for g in (-1.0, 0.0, 1.0, 2.0):
                p = E[iF] ** g; p = p / p.sum()
                for rep in range(10):
                    rr = np.random.default_rng(SEED + 7 * rep + int((g + 5) * 100))
                    sub = np.sort(rr.choice(iF, len(iF) // 2, replace=False, p=p))
                    JFp, JBp = tables(U, sub, cls, C, tg0, idxB=iB)
                    Fp, Bp = U.collapse(JFp.sum(2), m, n), U.collapse(JBp.sum(2), m, n)
                    pert.append(dict(species=sp, prop=prop, gamma=g, rep=rep, ll=lc.loglift_tab(Fp, Bp).mean(), mh=mh(Fp, Bp).mean()))
    print(sp, flush=True)
r = pd.DataFrame(rows); b = pd.DataFrame(bands); p = pd.DataFrame(pert)
r.to_csv(f"{RES}/mh_real.csv", index=False); b.to_csv(f"{RES}/mh_bands.csv", index=False); p.to_csv(f"{RES}/mh_perturbation.csv", index=False)
print(r.round(2).to_string(index=False))
print(b.pivot_table(index=["species", "prop"], columns="band", values=["ll", "mh"]).round(2).to_string())
g = p.groupby(["species", "prop", "gamma"])[["ll", "mh"]].mean()
rng = g.groupby(level=[0, 1]).agg(lambda x: x.max() - x.min())
print(rng.round(2).to_string())
# 200 species
occ = pl.read_parquet("species200_cells.parquet").to_pandas(); occ = occ[occ.LocalityID.isin(pos.index)]
red_names = set(pl.read_parquet("red_cell_species.parquet")["species"].unique().to_list())
out = []
for sp, d in occ.groupby("species"):
    iF = np.sort(pos[d.LocalityID.unique()].to_numpy())
    if len(iF) < 30:
        continue
    for prop in PROPS:
        V, T = prop_values(prop)
        if prop == "red" and sp in red_names:
            disk = {"h3_" + c for cid in ids[iF] for c in h3.grid_disk(cid[3:], 1)}
            V = V.copy(); V[pos.reindex(list(disk)).dropna().to_numpy(int)] -= 1.0
        C = len(T) + 1; cls = lc.classes(V, T)
        JF, JB = tables(U, iF, cls, C, tg0); F, B = U.collapse(JF.sum(2), m, n), U.collapse(JB.sum(2), m, n)
        out.append(dict(species=sp, prop=prop, n=len(iF), ll10000=lc.loglift_tab(F, B).mean(), mh10000=mh(F, B).mean()))
o = pd.DataFrame(out); o.to_csv(f"{RES}/mh_species200.csv", index=False)
for prop, d in o.groupby("prop"):
    print(prop, "Spearman ll10000 vs MH:", round(spearmanr(d.ll10000, d.mh10000)[0], 3), " opposite sign:", round(float(((d.ll10000 > 0) != (d.mh10000 > 0)).mean()), 3))
