"""
run_ring_v05.py - the list-slot effect. Within an effort stratum, a cell with a
record of a (non-red-listed) focal species has one species slot less for other
species, including red-listed ones, than a background cell with the same
effort index. This mainly matters at low effort (one or two species per cell).
We compare red-listed neighbours counted over the 7-cell disk (red_disk, the
property used throughout) with the count over the 6 neighbouring cells only
(red_ring, which does not include the focal cell's own species list):
(1) logLift within five bands of the focal species' effort distribution,
(2) logLift (K = 10,000) for the eight species, and
(3) the effort-perturbation range of logLift (K = 10,000).
"""
import numpy as np, pandas as pd, polars as pl
import loglift_core as lc
from bench_v05 import *

U = universe(); U.add_strata(10000)
ring = pl.read_parquet("red_ring.parquet").to_pandas()
assert (ring.LocalityID.to_numpy() == ids).all()
T = np.arange(1, 5); C = len(T) + 1; tg0 = np.zeros(N, np.int64)
red_names = set(pl.read_parquet("red_cell_species.parquet")["species"].unique().to_list())


def V_of(prop, sp):
    if prop == "red_disk":
        return prop_values("red", sp)[0]
    v = ring.red_ring.to_numpy(float).copy()
    sc = "selfring_" + sp.replace(" ", "_")
    if sc in ring.columns:
        v = v - ring[sc].to_numpy(float)
    return v


rows, bands, pert = [], [], []
for sp, grp in SPECIES:
    iF = focal_idx(sp)
    for prop in ("red_disk", "red_ring"):
        V = V_of(prop, sp); cls = lc.classes(V, T)
        JF, JB = tables(U, iF, cls, C, tg0); F, B = JF.sum(2), JB.sum(2)
        m, n = U.S[10000], U.Keff[10000]
        Fs, Bs = U.collapse(F, m, n), U.collapse(B, m, n)
        rows.append(dict(species=sp, prop=prop, n=len(iF), raw=float(np.mean(lc.raw_tab(F.sum(0), B.sum(0)))),
                         ll10000=float(np.mean(lc.loglift_tab(Fs, Bs)))))
        if len(iF) >= 60:
            cum = np.cumsum(Fs.sum(1)) / Fs.sum()
            cut = np.searchsorted(cum, [0.2, 0.4, 0.6, 0.8]) + 1
            edges = [0] + list(cut) + [n]
            for k, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
                if Fs[a:b].sum() > 0:
                    bands.append(dict(species=sp, prop=prop, band=k + 1, E_low=float(U.Elev[U.S[10000] == a].min()) if (U.S[10000] == a).any() else np.nan,
                                      ll=float(np.mean(lc.loglift_tab(Fs[a:b], Bs[a:b])))))
            occ = iF; iB = np.setdiff1d(np.arange(N), occ)
            for g in (-1.0, 0.0, 1.0, 2.0):
                p = E[occ] ** g; p = p / p.sum()
                for rep in range(10):
                    rr = np.random.default_rng(SEED + 7 * rep + int((g + 5) * 100))
                    sub = np.sort(rr.choice(occ, len(occ) // 2, replace=False, p=p))
                    JFp, JBp = tables(U, sub, cls, C, tg0, idxB=iB)
                    est = lc.loglift_tab(U.collapse(JFp.sum(2), m, n), U.collapse(JBp.sum(2), m, n))
                    pert.append(dict(species=sp, prop=prop, gamma=g, rep=rep, ll10000=float(np.mean(est))))
    print(sp, flush=True)
r = pd.DataFrame(rows); b = pd.DataFrame(bands); p = pd.DataFrame(pert)
r.to_csv(f"{RES}/ring_real.csv", index=False); b.to_csv(f"{RES}/ring_bands.csv", index=False); p.to_csv(f"{RES}/ring_perturbation.csv", index=False)
print(r.pivot_table(index="species", columns="prop", values=["raw", "ll10000"]).round(2).to_string())
print(b.pivot_table(index=["species", "prop"], columns="band", values="ll").round(2).to_string())
g = p.groupby(["species", "prop", "gamma"]).ll10000.mean().unstack("gamma")
print((g.max(axis=1) - g.min(axis=1)).unstack("prop").round(2).to_string())
# how often is the single species of a single-record background cell red-listed?
c1 = (cells.n_species.to_numpy() == 1) & (cells.n_events.to_numpy() == 1)
redcells = set(pl.read_parquet("red_cell_species.parquet")["LocalityID"].to_list())
print("share of one-species, one-day cells whose species is red-listed:", round(float(np.isin(ids[c1], list(redcells)).mean()), 4))
