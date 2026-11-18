"""
bench_v05.py - shared set-up for the revision benchmarks (v0.5).
Loads cells.parquet / focal_cells.parquet / species200_cells.parquet and
builds the Universe (effort levels, tie-respecting strata, x-bins).
"""
import numpy as np, pandas as pd, polars as pl, os
import h3
import loglift_core as lc

SEED = 20260907
PAIRS = [("Aglais urticae", "Colias hecla", "Lepidoptera"),
         ("Achillea millefolium", "Bartsia alpina", "Tracheophyta"),
         ("Cerceris rybyensis", "Mellinus crabroneus", "Hymenoptera"),
         ("Primula veris", "Primula farinosa", "Tracheophyta")]
SPECIES = [(s, g) for a, b, g in PAIRS for s in (a, b)]
PROPS = {"red": ("red_disk", np.arange(1, 5)),
         "richness": ("n_species", np.array([5, 10, 20, 30, 50]))}
MIN_N = 30
RES = os.environ.get("RESULTS", "results_v05")
os.makedirs(RES, exist_ok=True)

cells = pl.read_parquet("cells.parquet").to_pandas()
ids = cells.LocalityID.to_numpy()
pos = pd.Series(np.arange(len(ids)), index=ids)
E = cells.effort_index.to_numpy(float)
N = len(E)
fc = pl.read_parquet("focal_cells.parquet").to_pandas()
TG = {g: (cells["grp_" + g].to_numpy() > 0).astype(np.int64) for g in ("Lepidoptera", "Hymenoptera", "Tracheophyta")}


def universe(effort=None, Ks=(10, 100)):
    return lc.Universe(E if effort is None else effort, Ks=Ks)


def focal_idx(species):
    return np.sort(pos[fc.loc[fc.species == species, "LocalityID"].to_numpy()].to_numpy())


def prop_values(prop, species=None):
    col, T = PROPS[prop]
    V = cells[col].to_numpy(float)
    if prop == "red" and species is not None:
        sc = "self_" + species.replace(" ", "_")
        if sc in cells.columns:
            V = V - cells[sc].to_numpy(float)
    return V, T


def tables(U, idxF, cls, C, tg, idxB=None):
    """Joint tables (level x class x tg) for focal cells idxF and background =
    all other cells (or idxB if given)."""
    JF = U.joint(idxF, cls, C, tg)
    if idxB is None:
        JB = U.joint(np.arange(N), cls, C, tg) - JF
    else:
        JB = U.joint(idxB, cls, C, tg)
    return JF, JB


def parents(idx, res):
    return np.array([h3.cell_to_parent(i[3:], res) for i in ids[idx]])
