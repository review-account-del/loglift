"""
run_positive_v05.py - positive controls and imperfect effort proxies.

Artificial species of 2,000 distinct cells drawn without replacement with
    P(cell j) proportional to  S_j^gamma * rho^H_j
  S   = the 'true' observation process: E (the index used for correction),
        n_records, or disk_events (recording days summed over the 7-cell disk);
  H_j = 1[V_j >= t*]: a habitat defined by the property (t* = 2 red-listed
        neighbours, t* = 20 species), rho = 1 (null), 2 or 4 (planted
        preference for that habitat).
Reference values, computed exactly from the sampling probabilities over all
cells (no sampling noise):
  truth_marginal  the lift the planted preference produces when gamma = 0
                  (no preferential sampling), averaged over thresholds;
  truth_cond      the same preference measured within strata of the true
                  process S (K = 10,000 nominal quantile strata on S): the
                  quantity an estimator that knew S would target.
Estimators correct with E unless stated; 'll1000_S' corrects with S itself.
"""
import numpy as np, pandas as pd, polars as pl, time
import loglift_core as lc
from bench_v05 import *

t0 = time.time()
U = universe(); U.add_strata(1000); U.add_strata(10000)
nbr = pl.read_parquet("neighbour_effort.parquet").to_pandas()
assert (nbr.LocalityID.to_numpy() == ids).all()
PROXY = {"E": E, "n_records": cells.n_records.to_numpy(float), "disk_events": nbr.disk_events.to_numpy(float)}
US = {k: (U if k == "E" else lc.Universe(v, Ks=(10000,))) for k, v in PROXY.items()}
TSTAR = {"red": 2, "richness": 20}
R = 10
rows, truths = [], []
for prop in PROPS:
    V, T = prop_values(prop); C = len(T) + 1; cls = lc.classes(V, T)
    Ex = (V[:, None] >= T[None, :]).astype(float)              # N x |T| exceedance
    H = (V >= TSTAR[prop]).astype(float)
    base = Ex.mean(0)
    for sname, S in PROXY.items():
        Us = US[sname]
        s_strata = Us.S[10000][Us.lev]                             # cell -> stratum on S
        ns = Us.Keff[10000]
        cnt = np.bincount(s_strata, minlength=ns).astype(float)
        qS = np.column_stack([np.bincount(s_strata, weights=Ex[:, i], minlength=ns) for i in range(len(T))]) / cnt[:, None]
        for rho in (1.0, 2.0, 4.0):
            w0 = rho ** H; p0 = w0 / w0.sum()
            tm = float(np.mean(np.log2((p0[:, None] * Ex).sum(0)) - np.log2(base)))
            for g in (0.0, 1.0, 2.0):
                w = (S ** g) * w0; p = w / w.sum()
                num = (p[:, None] * Ex).sum(0)
                tc = float(np.mean(np.log2(num) - np.log2((p[:, None] * qS[s_strata]).sum(0))))
                truths.append(dict(prop=prop, S=sname, rho=rho, gamma=g, truth_marginal=tm, truth_cond=tc))
                for rep in range(R):
                    rr = np.random.default_rng(SEED + 101 * rep + int(10 * g) + int(100 * rho))
                    iF = np.sort(rr.choice(N, 2000, replace=False, p=p))
                    JF, JB = tables(U, iF, cls, C, TG["Tracheophyta"])
                    r = lc.mean_over_t(lc.estimates_from_joint(
                        U, JF, JB, which=("raw", "ll10", "ll100", "glm_linear", "glm_spline"), extraK=(10000,)))
                    if sname != "E":
                        JFs, JBs = tables(Us, iF, cls, C, TG["Tracheophyta"])
                        F, B = JFs.sum(2), JBs.sum(2)
                        r["ll10000_S"] = float(np.mean(lc.loglift_tab(Us.collapse(F, Us.S[10000], ns), Us.collapse(B, Us.S[10000], ns))))
                    else:
                        r["ll10000_S"] = r["ll10000"]
                    for k, v in r.items():
                        rows.append(dict(prop=prop, S=sname, rho=rho, gamma=g, rep=rep, method=k, estimate=v))
            print(prop, sname, rho, f"[{time.time()-t0:.0f}s]", flush=True)
df = pd.DataFrame(rows); tr = pd.DataFrame(truths)
df.to_csv(f"{RES}/positive_control.csv", index=False); tr.to_csv(f"{RES}/positive_control_truth.csv", index=False)
summ = df.groupby(["prop", "S", "rho", "gamma", "method"]).estimate.mean().unstack("method").reset_index()
summ = summ.merge(tr, on=["prop", "S", "rho", "gamma"])
summ.to_csv(f"{RES}/positive_control_summary.csv", index=False)   # unrounded: tables round once
print(summ.round(2).to_string(index=False))
