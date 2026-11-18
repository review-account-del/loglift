"""
run_species200_v05.py - 200 species from the national focal list: all
estimators per threshold on an extended threshold grid (so that threshold sets
can be compared), GLM diagnostics, and rank comparisons between estimators.
Full background; red-listed focal species are removed from their own count.
"""
import numpy as np, pandas as pd, polars as pl, time, os
import h3
from scipy.stats import spearmanr
import loglift_core as lc
from bench_v05 import *

t0 = time.time()
U = universe(); U.add_strata(1000); U.add_strata(10000)
occ = pl.read_parquet("species200_cells.parquet").to_pandas()
occ = occ[occ.LocalityID.isin(pos.index)]
info = pd.read_csv("species200.tsv", sep="\t")
red_names = set(pl.read_parquet("red_cell_species.parquet")["species"].unique().to_list())
TEXT = {"red": np.arange(1, 9), "richness": np.array([2, 5, 10, 20, 30, 50, 100])}
MAIN = {"red": [1, 2, 3, 4], "richness": [5, 10, 20, 30, 50]}
COLS = {"red": "red_disk", "richness": "n_species"}
METH = ("raw", "ll10", "ll100", "glm_factor", "glm_linear", "glm_spline", "glm_firth")
tg0 = np.zeros(N, np.int64)
per_t, diag = [], []
for i, (sp, d) in enumerate(occ.groupby("species")):
    iF = np.sort(pos[d.LocalityID.unique()].to_numpy())
    if len(iF) < 30:
        continue
    for prop in PROPS:
        V = cells[COLS[prop]].to_numpy(float).copy()
        if prop == "red" and sp in red_names:
            disk = {"h3_" + c for cid in ids[iF] for c in h3.grid_disk(cid[3:], 1)}
            V = V.copy(); V[pos.reindex(list(disk)).dropna().to_numpy(int)] -= 1.0
        T = TEXT[prop]; C = len(T) + 1; cls = lc.classes(V, T)
        JF, JB = tables(U, iF, cls, C, tg0)
        r, D = lc.estimates_from_joint(U, JF, JB, which=METH, diag=True, extraK=(1000, 10000))
        F1 = JF.sum((0, 2)); cF = lc.exceed(F1)
        for k, v in r.items():
            for j, t in enumerate(T):
                per_t.append(dict(species=sp, prop=prop, n_cells=len(iF), method=k, t=t, value=float(v[j]),
                                  focal_exceed=float(cF[j])))
        for k, dl in D.items():
            for dd in dl:
                diag.append(dict(species=sp, prop=prop, method=k, t=T[dd["t_index"]], **dd))
    if i % 25 == 0:
        print(i, sp, f"[{time.time()-t0:.0f}s]", flush=True)
pt = pd.DataFrame(per_t); pt.to_csv(f"{RES}/species200_per_threshold.csv", index=False)
pd.DataFrame(diag).to_csv(f"{RES}/species200_glm_diagnostics.csv", index=False)

# main threshold sets -> one value per species x method
main = pd.concat([pt[(pt.prop == p) & pt.t.isin(ts)] for p, ts in MAIN.items()])
wide = main.groupby(["species", "prop", "n_cells", "method"]).value.mean().unstack("method").reset_index()
wide = wide.merge(info[["species", "group", "occupancy_block"]], on="species", how="left")
wide.to_csv(f"{RES}/species200.csv", index=False)
out = []
for prop, d in wide.groupby("prop"):
    for a, b in [("ll10", "ll10000"), ("ll100", "ll10000"), ("ll1000", "ll10000"), ("raw", "ll10000"), ("ll10", "ll100"),
                 ("ll10000", "glm_linear"), ("ll10000", "glm_spline"), ("ll10000", "glm_firth"), ("ll10000", "glm_factor"),
                 ("glm_linear", "glm_firth"), ("glm_linear", "glm_spline"), ("ll100", "glm_linear")]:
        out.append(dict(prop=prop, a=a, b=b, spearman=spearmanr(d[a], d[b])[0],
                        share_opposite_sign=float(((d[a] > 0) != (d[b] > 0)).mean()),
                        mean_diff=float((d[a] - d[b]).mean()), mean_abs_diff=float((d[a] - d[b]).abs().mean()),
                        max_abs_diff=float((d[a] - d[b]).abs().max()), n=len(d)))
res = pd.DataFrame(out); res.to_csv(f"{RES}/species200_summary.csv", index=False)
print(res.round(3).to_string(index=False))

# threshold-set sensitivity (logLift K = 100 and 1000)
SETS = {"red": {"t1": [1], "t1-2": [1, 2], "t2-4": [2, 3, 4], "t1-4 (main)": [1, 2, 3, 4], "t1-8": list(range(1, 9))},
        "richness": {"t5": [5], "t10-30": [10, 20, 30], "t5-50 (main)": [5, 10, 20, 30, 50],
                     "t5-100": [5, 10, 20, 30, 50, 100], "t2-100": [2, 5, 10, 20, 30, 50, 100]}}
rows = []
for prop, sets in SETS.items():
    for meth in ("ll100", "ll10000", "raw"):
        x = pt[(pt.prop == prop) & (pt.method == meth)]
        ref = x[x.t.isin(MAIN[prop])].groupby("species").value.mean()
        for name, ts in sets.items():
            v = x[x.t.isin(ts)].groupby("species").value.mean()
            rows.append(dict(prop=prop, method=meth, set=name, spearman_vs_main=spearmanr(ref, v.loc[ref.index])[0],
                             mean=v.mean(), sd=v.std()))
ts = pd.DataFrame(rows); ts.to_csv(f"{RES}/threshold_sets.csv", index=False)
print(ts.round(3).to_string(index=False))
print("DONE", round(time.time() - t0))
