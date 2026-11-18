"""
run_proxies_v05.py - sensitivity to the effort proxy (review R2-M1, R2-M3,
R2 minor 331-333).

Correction proxies: E (0.5 days + 0.5 species/day), n_events (recording days),
n_records, n_obs (distinct recorders, from the raw records), disk_E (E summed
over the 7-cell disk) and two-dimensional strata E x disk_E.
(a) Mismatched null: pseudo-species sampled with P ~ S^gamma on one proxy,
    corrected with another (logLift, K = 100 and 10,000 nominal strata).
(b) The eight real species and (c) the 200 species, corrected with each proxy.
"""
import numpy as np, pandas as pd, polars as pl, time, os
import h3
from scipy.stats import spearmanr
import loglift_core as lc
from bench_v05 import *

t0 = time.time()
nbr = pl.read_parquet("neighbour_effort.parquet").to_pandas()
assert (nbr.LocalityID.to_numpy() == ids).all()
P = {"E": E, "n_events": cells.n_events.to_numpy(float), "n_records": cells.n_records.to_numpy(float),
     "disk_E": nbr.disk_E.to_numpy(float)}
if os.path.exists("variants/full/cells.parquet"):
    raw = pl.read_parquet("variants/full/cells.parquet", columns=["LocalityID", "n_obs"]).to_pandas().set_index("LocalityID")
    P["n_obs"] = raw.n_obs.reindex(ids).fillna(0).to_numpy(float) + 1.0     # +1: cells without recorder name
UP = {k: lc.Universe(v, Ks=(100, 10000)) for k, v in P.items()}
for kE, kD, name in [(100, 10, "E100xdisk10"), (10000, 10, "E10000xdisk10")]:
    sE, _ = lc.effort_strata(E, kE); sD, _ = lc.effort_strata(P["disk_E"], kD)
    code = sE.astype(float) * 1000 + sD
    u = lc.Universe(code, Ks=()); u.add_strata("exact"); UP[name] = u
# two-dimensional strata of the two components of E: recording days x species per day (list length)
ne = cells.n_events.to_numpy(float); spd = cells.n_species.to_numpy(float) / ne
s1, k1 = lc.effort_strata(ne, 100); s2, k2 = lc.effort_strata(spd, 100)
u = lc.Universe(s1.astype(float) * 1000 + s2, Ks=()); u.add_strata("exact"); UP["days_x_listlength"] = u
print("days x list-length strata:", k1, "x", k2, "->", u.Keff["exact"], flush=True)
print("K_eff:", {k: (u.Keff.get(100), u.Keff.get(1000), u.Keff.get("exact")) for k, u in UP.items()}, flush=True)


def ll(u, iF, cls, C, K):
    JF, JB = tables(u, iF, cls, C, np.zeros(N, np.int64))
    F, B = JF.sum(2), JB.sum(2)
    return float(np.mean(lc.loglift_tab(u.collapse(F, u.S[K], u.Keff[K]), u.collapse(B, u.S[K], u.Keff[K]))))


def all_corr(iF, cls, C):
    r = {}
    for k, u in UP.items():
        for K in ((100, 10000) if "x" not in k else ("exact",)):
            r[f"{k}|{K}"] = ll(u, iF, cls, C, K)
    F1 = np.bincount(cls[iF], minlength=C); B1 = np.bincount(cls, minlength=C) - F1
    r["raw|-"] = float(np.mean(lc.raw_tab(F1, B1)))
    return r


PARTS = os.environ.get('PARTS', 'abc')
# (a) mismatched null
rows = []
if 'a' not in PARTS:
    SAMP = []
SAMP = [k for k in P if k != "disk_E"] + ["disk_E"]
for prop in PROPS:
    V, T = prop_values(prop); C = len(T) + 1; cls = lc.classes(V, T)
    for sname in SAMP:
        S = P[sname]
        for g in (1.0, 2.0):
            p = S ** g; p = p / p.sum()
            for rep in range(10):
                rr = np.random.default_rng(SEED + 17 * rep + int(10 * g))
                iF = np.sort(rr.choice(N, 2000, replace=False, p=p))
                for k, v in all_corr(iF, cls, C).items():
                    corr, K = k.split("|")
                    rows.append(dict(prop=prop, sampling=sname, gamma=g, rep=rep, correction=corr, K=K, estimate=v))
        print("null", prop, sname, f"[{time.time()-t0:.0f}s]", flush=True)
nd = pd.DataFrame(rows) if rows else pd.read_csv(f"{RES}/proxy_null.csv")
if rows: nd.to_csv(f"{RES}/proxy_null.csv", index=False)
tab = nd.groupby(["prop", "gamma", "sampling", "correction", "K"]).estimate.mean().unstack(["correction", "K"]).round(3)
tab.to_csv(f"{RES}/proxy_null_table.csv"); print(tab.to_string())

# (b) real species
rows = []
for sp, grp in (SPECIES if 'b' in PARTS else []):
    iF = focal_idx(sp)
    for prop in PROPS:
        V, T = prop_values(prop, sp); C = len(T) + 1; cls = lc.classes(V, T)
        for k, v in all_corr(iF, cls, C).items():
            corr, K = k.split("|")
            rows.append(dict(species=sp, prop=prop, n_focal=len(iF), correction=corr, K=K, estimate=v))
rd = pd.DataFrame(rows) if rows else pd.read_csv(f"{RES}/proxy_real.csv")
if rows: rd.to_csv(f"{RES}/proxy_real.csv", index=False)
print(rd.pivot_table(index=["prop", "species"], columns=["correction", "K"], values="estimate").round(2).to_string())

# (c) 200 species
occ = pl.read_parquet("species200_cells.parquet").to_pandas(); occ = occ[occ.LocalityID.isin(pos.index)]
red_names = set(pl.read_parquet("red_cell_species.parquet")["species"].unique().to_list())
rows = []
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
        for k, v in all_corr(iF, cls, C).items():
            corr, K = k.split("|")
            rows.append(dict(species=sp, prop=prop, n_focal=len(iF), correction=corr, K=K, estimate=v))
sd = pd.DataFrame(rows); sd.to_csv(f"{RES}/proxy_species200.csv", index=False)
w = sd.assign(key=sd.correction + "|" + sd.K).pivot_table(index=["prop", "species"], columns="key", values="estimate")
out = []
for prop in PROPS:
    x = w.loc[prop]
    for c in x.columns:
        out.append(dict(prop=prop, proxy=c, spearman_vs_E100=spearmanr(x["E|100"], x[c])[0],
                        spearman_vs_E10000=spearmanr(x["E|10000"], x[c])[0], mean=x[c].mean(), sd=x[c].std()))
pd.DataFrame(out).to_csv(f"{RES}/proxy_species200_summary.csv", index=False)
print(pd.DataFrame(out).round(3).to_string(index=False))
print("DONE", round(time.time() - t0))
