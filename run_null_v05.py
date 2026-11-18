"""
run_null_v05.py - (ii) null pseudo-species, (iii) effort perturbation of real
species, (iv) number of strata. Full background everywhere except for the
negative-binomial GLM (fixed 200k background subsample, as in v0.4).

Null pseudo-species: 2,000 distinct cells drawn WITHOUT replacement from all
recorded cells, P(cell) proportional to E^gamma. Background = all other cells.
Effort perturbation: a subsample of HALF of the species' occupied cells, drawn
without replacement with P proportional to E^gamma (gamma = -1, 0, 1, 2), i.e.
n/2 distinct cells; no new cells are added. The background is all cells
outside the species' full set of occupied cells (v0.4 left the unselected half
in the background).
"""
import numpy as np, pandas as pd, time, os, sys
import loglift_core as lc
import loglift_bench as old
from bench_v05 import *

PART = sys.argv[1] if len(sys.argv) > 1 else "all"
U = universe()
U.add_strata(1000); U.add_strata(10000)
x = np.log1p(E)
NB_BG = 200_000
t0 = time.time()


def all_estimates(iF, cls, C, tg, iB=None, rng=None, nb=True, V=None):
    JF, JB = tables(U, iF, cls, C, tg, idxB=iB)
    r = lc.estimates_from_joint(U, JF, JB, extraK=(1000, 10000))
    if nb:
        pool = np.setdiff1d(np.arange(N), iF) if iB is None else iB
        j = rng.choice(pool, NB_BG, replace=False)
        r["glm_nb"] = np.array([lc.glm_nb(V[iF], x[iF], V[j], x[j])])
    return lc.mean_over_t(r)


if PART in ("all", "null"):
    rows = []
    for prop in PROPS:
        V, T = prop_values(prop); C = len(T) + 1; cls = lc.classes(V, T)
        for g in [0.0, 0.5, 1.0, 2.0]:
            p = E ** g; p = p / p.sum()
            for rep in range(20):
                rr = np.random.default_rng(SEED + 1000 * rep + int(g * 10))
                iF = np.sort(rr.choice(N, 2000, replace=False, p=p))
                for k, v in all_estimates(iF, cls, C, TG["Tracheophyta"], rng=rr, V=V).items():
                    rows.append(dict(prop=prop, gamma=g, rep=rep, method=k, estimate=v))
            print(f"null {prop} {g} [{time.time()-t0:.0f}s]", flush=True)
    pd.DataFrame(rows).to_csv(f"{RES}/null_test.csv", index=False)

if PART in ("all", "pert"):
    rows = []
    for sp, grp in SPECIES:
        occ = focal_idx(sp)
        if len(occ) < 2 * MIN_N:
            continue
        iB = np.setdiff1d(np.arange(N), occ)
        for prop in PROPS:
            V, T = prop_values(prop, sp); C = len(T) + 1; cls = lc.classes(V, T)
            for g in [-1.0, 0.0, 1.0, 2.0]:
                p = E[occ] ** g; p = p / p.sum()
                for rep in range(10):
                    rr = np.random.default_rng(SEED + 7 * rep + int((g + 5) * 100))
                    iF = np.sort(rr.choice(occ, len(occ) // 2, replace=False, p=p))
                    for k, v in all_estimates(iF, cls, C, TG[grp], iB=iB, rng=rr, V=V).items():
                        rows.append(dict(species=sp, prop=prop, gamma=g, rep=rep, method=k, estimate=v))
            print(f"pert {sp} {prop} [{time.time()-t0:.0f}s]", flush=True)
    pd.DataFrame(rows).to_csv(f"{RES}/effort_perturbation.csv", index=False)

if PART in ("all", "strata"):
    KS = [5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, "exact"]
    for K in KS:
        if K not in U.S:
            U.add_strata(K)
    OLD = {K: old.rank_quantile_strata(E, K) for K in (10, 100)}
    rows = []
    for prop in PROPS:
        V, T = prop_values(prop); C = len(T) + 1; cls = lc.classes(V, T)
        for g in [0.0, 1.0, 2.0]:
            p = E ** g; p = p / p.sum()
            for rep in range(15):
                rr = np.random.default_rng(SEED + 31 * rep + int(g * 10))
                iF = np.sort(rr.choice(N, 2000, replace=False, p=p))
                JF, JB = tables(U, iF, cls, C, TG["Tracheophyta"])
                F, B = JF.sum(2), JB.sum(2)
                for K in KS:
                    Fs, Bs = U.collapse(F, U.S[K], U.Keff[K]), U.collapse(B, U.S[K], U.Keff[K])
                    nb_min = Bs.sum(1)[Fs.sum(1) > 0].min()
                    dropped = float(Fs[Bs.sum(1) == 0].sum() / Fs.sum())
                    for cs, scheme in ((True, "v05"), (False, "v05_no_common_support")):
                        rows.append(dict(prop=prop, gamma=g, rep=rep, K=str(K), K_eff=U.Keff[K], scheme=scheme,
                                         estimate=float(np.mean(lc.loglift_tab(Fs, Bs, common_support=cs))),
                                         focal_dropped=dropped, min_bg_in_focal_strata=nb_min))
                isF = np.zeros(N, bool); isF[iF] = True
                for K, s in OLD.items():
                    Kc = s.max() + 1
                    Fo = np.bincount(s[isF] * C + cls[isF], minlength=Kc * C).reshape(Kc, C)
                    Bo = np.bincount(s[~isF] * C + cls[~isF], minlength=Kc * C).reshape(Kc, C)
                    rows.append(dict(prop=prop, gamma=g, rep=rep, K=str(K), K_eff=len(np.unique(s)), scheme="v04",
                                     estimate=float(np.mean(lc.loglift_tab(Fo, Bo)))))
            print(f"strata {prop} {g} [{time.time()-t0:.0f}s]", flush=True)
    df = pd.DataFrame(rows); df.to_csv(f"{RES}/strata_sensitivity.csv", index=False)
    tab = df.groupby(["scheme", "prop", "gamma", "K", "K_eff"]).agg(mean=("estimate", "mean"), std=("estimate", "std"), focal_dropped=("focal_dropped", "mean"), min_bg=("min_bg_in_focal_strata", "median")).round(4)
    tab.to_csv(f"{RES}/table_strata.csv")
    print(tab.to_string())
print("DONE", time.time() - t0)
