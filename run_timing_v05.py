"""
run_timing_v05.py - single-core cost per focal species, lean implementations.
Species: Primula farinosa; property: red-listed neighbours (4 thresholds).
Once per property (shared by every species): the class of every cell and the
count tables of all cells by effort stratum (logLift) and by effort x-bin (GLMs).
Per species: the focal table (bincount over the focal cells), the background
table (all-cells table minus focal table) and the estimate.
"""
import os
os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
import numpy as np, pandas as pd, time, platform
import loglift_core as lc
from bench_v05 import *

U = universe(); U.add_strata(10000)
sp = "Primula farinosa"
V, T = prop_values("red", sp); C = len(T) + 1
iF = focal_idx(sp)


def med(f, reps):
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); f(); ts.append(time.perf_counter() - t)
    return float(np.median(ts))


rows = []
t = time.perf_counter()
cls = lc.classes(V, T)
s_cell = {K: U.S[K][U.lev] for K in (10, 100, 10000)}
x_cell = U.lev2x[U.lev]
A = {K: np.bincount(s_cell[K] * C + cls, minlength=U.Keff[K] * C).reshape(U.Keff[K], C) for K in (10, 100, 10000)}
AX = np.bincount(x_cell * C + cls, minlength=len(U.xval) * C).reshape(len(U.xval), C)
t_once = time.perf_counter() - t
rows.append(dict(step="once per property: classes and all-cell count tables", seconds=t_once))


def ll(K):
    F = np.bincount(s_cell[K][iF] * C + cls[iF], minlength=U.Keff[K] * C).reshape(U.Keff[K], C)
    return lc.loglift_tab(F, A[K] - F)


def glm(kind, firth=False):
    if kind == "factor":
        Fs = np.bincount(s_cell[100][iF] * C + cls[iF], minlength=U.Keff[100] * C).reshape(U.Keff[100], C)
        return lc.glm_gcomp_tab(Fs, A[100] - Fs, np.arange(U.Keff[100]), "factor")
    F = np.bincount(x_cell[iF] * C + cls[iF], minlength=len(U.xval) * C).reshape(len(U.xval), C)
    return lc.glm_gcomp_tab(F, AX - F, U.xval, kind, knots=U.knots, firth=firth)


def raw():
    F1 = np.bincount(cls[iF], minlength=C); return lc.raw_tab(F1, A[10].sum(0) - F1)


for name, f in [("logLift K=10", lambda: ll(10)), ("logLift K=100", lambda: ll(100)), ("logLift K=10,000", lambda: ll(10000)),
                ("raw", raw), ("GLM stratum factor (K=100)", lambda: glm("factor")), ("GLM log-linear effort", lambda: glm("linear")),
                ("GLM spline effort", lambda: glm("spline")), ("GLM log-linear, Firth", lambda: glm("linear", True))]:
    rows.append(dict(step="per species: " + name, seconds=med(f, 50)))
F0 = np.bincount(s_cell[10000][iF] * C + cls[iF], minlength=U.Keff[10000] * C).reshape(U.Keff[10000], C); B0 = A[10000] - F0
FX0 = np.bincount(x_cell[iF] * C + cls[iF], minlength=len(U.xval) * C).reshape(len(U.xval), C); BX0 = AX - FX0


def boot_ll(Bn):
    for b in range(Bn):
        rr = np.random.default_rng(b); d = iF[rr.integers(0, len(iF), len(iF))]
        F = np.bincount(s_cell[10000][d] * C + cls[d], minlength=U.Keff[10000] * C).reshape(U.Keff[10000], C)
        lc.loglift_tab(F, lc.multinomial_table(rr, B0))


def boot_glm(Bn, kind):
    for b in range(Bn):
        rr = np.random.default_rng(b); d = iF[rr.integers(0, len(iF), len(iF))]
        F = np.bincount(x_cell[d] * C + cls[d], minlength=len(U.xval) * C).reshape(len(U.xval), C)
        lc.glm_gcomp_tab(F, lc.multinomial_table(rr, BX0), U.xval, kind, knots=U.knots)


rows.append(dict(step="bootstrap x200: logLift K=10,000", seconds=med(lambda: boot_ll(200), 3)))
rows.append(dict(step="bootstrap x200: GLM log-linear (extrapolated from 20)", seconds=10 * med(lambda: boot_glm(20, "linear"), 1)))
rows.append(dict(step="bootstrap x200: GLM spline (extrapolated from 20)", seconds=10 * med(lambda: boot_glm(20, "spline"), 1)))
df = pd.DataFrame(rows); df["cpu"] = platform.processor() or platform.machine()
df.to_csv(f"{RES}/timing.csv", index=False)
print(df.to_string(index=False))
per = {r.step: r.seconds for r in df.itertuples()}
for name, key, bkey in [("logLift K=10,000", "per species: logLift K=10,000", "bootstrap x200: logLift K=10,000"),
                        ("GLM log-linear", "per species: GLM log-linear effort", "bootstrap x200: GLM log-linear (extrapolated from 20)"),
                        ("GLM spline", "per species: GLM spline effort", "bootstrap x200: GLM spline (extrapolated from 20)")]:
    tot = 5000 * 2 * (per[key] + per[bkey])
    print(f"5,000 species x 2 properties, point + 200 bootstrap replicates, {name}: {tot / 3600:.2f} h")
