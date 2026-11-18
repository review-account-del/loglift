"""
run_real_v05.py - eight focal species x two properties: point estimates for all
estimators (full background), per-threshold values, GLM diagnostics, and
three bootstraps: site (iid focal sites), and spatial blocks of H3 res-6 and
res-5 parent cells. The full background is resampled as a multinomial draw on
its count table in every replicate.
"""
import numpy as np, pandas as pd, time, os, sys
from concurrent.futures import ProcessPoolExecutor
import loglift_core as lc
from bench_v05 import *

B_BOOT = int(os.environ.get("B_BOOT", 200))
U = universe()
U.add_strata(1000); U.add_strata(10000)
GLM_BG = 500_000
METH_BOOT = ("raw", "tgb", "ll10", "ll100", "ll1000", "ll10000", "glm_factor", "glm_linear", "glm_spline", "glm_firth")
METH_BLOCK = ("raw", "ll10", "ll100", "ll1000", "ll10000", "glm_linear", "glm_spline", "glm_firth")


def job(args):
    sp, grp, prop = args
    t0 = time.time()
    V, T = prop_values(prop, sp); C = len(T) + 1
    cls = lc.classes(V, T); tg = TG[grp]
    iF = focal_idx(sp)
    JF, JB = tables(U, iF, cls, C, tg)
    r, D = lc.estimates_from_joint(U, JF, JB, diag=True, extraK=(1000, 10000))
    rng = np.random.default_rng(SEED)
    # negative binomial on a fixed 500k background subsample
    iB = np.setdiff1d(np.arange(N), iF)
    j = rng.choice(iB, GLM_BG, replace=False)
    x = np.log1p(E)
    r["glm_nb"] = np.array([lc.glm_nb(V[iF], x[iF], V[j], x[j])])
    per_t = [dict(species=sp, prop=prop, method=k, t=(T[i] if len(v) == len(T) else np.nan), value=float(v[i]))
             for k, v in r.items() for i in range(len(v))]
    diag = [dict(species=sp, prop=prop, method=k, t=T[d["t_index"]], **d) for k, dl in D.items() for d in dl]
    m = lc.mean_over_t(r)
    boots = {}
    designs = [("site", None)]
    if len(iF) >= 6:
        designs += [("block_res6", parents(iF, 6)), ("block_res5", parents(iF, 5))]
    for name, blk in designs:
        rows = []
        meth = METH_BOOT if name == "site" else METH_BLOCK
        if blk is not None:
            ub, inv = np.unique(blk, return_inverse=True)
            members = [iF[inv == k] for k in range(len(ub))]
        for b in range(B_BOOT):
            rr = np.random.default_rng(SEED + 1 + b)
            if blk is None:
                draw = iF[rr.integers(0, len(iF), len(iF))]
            else:
                pick = rr.integers(0, len(members), len(members))
                draw = np.concatenate([members[k] for k in pick])
            JFb = U.joint(draw, cls, C, tg)
            JBb = lc.multinomial_table(rr, JB)
            rows.append(lc.mean_over_t(lc.estimates_from_joint(U, JFb, JBb, which=meth, extraK=(1000, 10000))))
        boots[name] = pd.DataFrame(rows)
    out = []
    for k, v in m.items():
        row = dict(species=sp, group=grp, prop=prop, method=k, n_focal=len(iF), estimate=v)
        for name, df in boots.items():
            if k in df:
                row[f"{name}_lo"] = df[k].quantile(0.025); row[f"{name}_hi"] = df[k].quantile(0.975)
                row[f"{name}_sd"] = df[k].std()
        if len(iF) >= 6:
            row["n_blocks_res6"] = len(np.unique(parents(iF, 6))); row["n_blocks_res5"] = len(np.unique(parents(iF, 5)))
        out.append(row)
    print(f"{sp:22s} {prop:9s} n={len(iF):6d} ll100={m['ll100']:+.3f} raw={m['raw']:+.3f} "
          f"spline={m['glm_spline']:+.3f} [{time.time()-t0:.0f}s]", flush=True)
    return out, per_t, diag


if __name__ == "__main__":
    jobs = [(sp, g, p) for sp, g in SPECIES for p in PROPS]
    with ProcessPoolExecutor(int(os.environ.get("NPROC", 2))) as ex:
        res = list(ex.map(job, jobs))
    pd.DataFrame([r for o, _, _ in res for r in o]).to_csv(f"{RES}/real_species.csv", index=False)
    pd.DataFrame([r for _, p, _ in res for r in p]).to_csv(f"{RES}/real_species_per_threshold.csv", index=False)
    pd.DataFrame([r for _, _, d in res for r in d]).to_csv(f"{RES}/real_species_glm_diagnostics.csv", index=False)
    print("Keff", U.Keff, "DONE")
