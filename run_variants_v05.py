"""
run_variants_v05.py - data-heterogeneity sensitivity (review R2-M2): rebuild the
per-cell tables from the raw records for subsets of the data and recompute the
estimates. Variants (gbif_cells.py): full (all records, rebuilt from raw),
precise (coordinate uncertainty known and <= 100 m), artportalen (Artportalen
records only), early (2000-2017), late (2018-2026). In each variant, effort,
both properties and the occupied cells are recomputed from that subset only.
"""
import numpy as np, pandas as pd, polars as pl, time, os
import h3
from scipy.stats import spearmanr
import loglift_core as lc

RES = os.environ.get("RESULTS", "results_v05")
PROPS = {"red": ("red_disk", np.arange(1, 5)), "richness": ("n_species", np.array([5, 10, 20, 30, 50]))}
FOCAL = ["Aglais urticae", "Colias hecla", "Achillea millefolium", "Bartsia alpina",
         "Cerceris rybyensis", "Mellinus crabroneus", "Primula veris", "Primula farinosa"]
red_names = set(pl.read_parquet("red_cell_species.parquet")["species"].unique().to_list())
t0 = time.time()
rows = []
for var in ["full", "precise", "artportalen", "early", "late"]:
    d = f"variants/{var}"
    if not os.path.exists(f"{d}/cells.parquet"):
        continue
    cells = pl.read_parquet(f"{d}/cells.parquet").to_pandas()
    ids = cells.LocalityID.to_numpy(); pos = pd.Series(np.arange(len(ids)), index=ids); N = len(ids)
    U = lc.Universe(cells.effort_index.to_numpy(float), Ks=(10, 100, 10000))
    tg0 = np.zeros(N, np.int64)
    occ = pd.concat([pl.read_parquet(f"{d}/focal_cells.parquet").to_pandas(),
                     pl.read_parquet(f"{d}/species200_cells.parquet").to_pandas()])
    for sp, g in occ.groupby("species"):
        iF = np.sort(pos[g.LocalityID.unique()].to_numpy())
        if len(iF) < (6 if sp in FOCAL else 30):
            continue
        for prop, (col, T) in PROPS.items():
            V = cells[col].to_numpy(float).copy()
            if prop == "red" and sp in red_names:
                sc = "self_" + sp.replace(" ", "_")
                if sc in cells.columns:
                    V = V - cells[sc].to_numpy(float)
                else:
                    disk = {"h3_" + c for cid in ids[iF] for c in h3.grid_disk(cid[3:], 1)}
                    V[pos.reindex(list(disk)).dropna().to_numpy(int)] -= 1.0
            C = len(T) + 1; cls = lc.classes(V, T)
            JF = U.joint(iF, cls, C, tg0); JB = U.joint(np.arange(N), cls, C, tg0) - JF
            which = ("raw", "ll100", "glm_spline") if sp in FOCAL else ("raw", "ll100")
            r = lc.mean_over_t(lc.estimates_from_joint(U, JF, JB, which=which, extraK=(10000,)))
            for k, v in r.items():
                rows.append(dict(variant=var, species=sp, focal8=sp in FOCAL, prop=prop, n_focal=len(iF),
                                 n_cells_variant=N, method=k, estimate=v))
    print(var, N, f"[{time.time()-t0:.0f}s]", flush=True)
df = pd.DataFrame(rows); df.to_csv(f"{RES}/variants.csv", index=False)
w = df.pivot_table(index=["species", "focal8", "prop", "method"], columns="variant", values="estimate").reset_index()
out = []
for (prop, meth), x in w[~w.focal8].groupby(["prop", "method"]):
    for v in ["precise", "artportalen", "early", "late"]:
        if v in x and "full" in x:
            y = x[["full", v]].dropna()
            out.append(dict(prop=prop, method=meth, variant=v, n_species=len(y), spearman_vs_full=spearmanr(y.full, y[v])[0],
                            mean_diff=float((y[v] - y.full).mean()), mean_abs_diff=float((y[v] - y.full).abs().mean())))
    if {"early", "late"} <= set(x.columns):
        y = x[["early", "late"]].dropna()
        out.append(dict(prop=prop, method=meth, variant="early_vs_late", n_species=len(y), spearman_vs_full=spearmanr(y.early, y.late)[0],
                        mean_diff=float((y.late - y.early).mean()), mean_abs_diff=float((y.late - y.early).abs().mean())))
pd.DataFrame(out).to_csv(f"{RES}/variants_summary.csv", index=False)
print(pd.DataFrame(out).round(3).to_string(index=False))
print(w[w.focal8].round(2).to_string(index=False))
