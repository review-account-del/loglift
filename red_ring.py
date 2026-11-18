"""red_ring.py - red-listed species recorded in the six neighbouring cells only (the k = 1 ring,
excluding the cell itself); memory-light version with integer keys."""
import polars as pl, numpy as np, time
from h3.api import basic_int as h3i
t0 = time.time()
lm = pl.read_parquet("cells.parquet", columns=["LocalityID"])["LocalityID"].to_list()
cid = np.array([int(s[3:], 16) for s in lm], dtype=np.uint64); order = np.argsort(cid); sid = cid[order]
red = pl.read_parquet("red_cell_species.parquet")
spn, spc = np.unique(red["species"].to_numpy(), return_inverse=True)
rc = np.array([int(s[3:], 16) for s in red["LocalityID"].to_list()], dtype=np.uint64)
uc, inv = np.unique(rc, return_inverse=True)
KEYS = []
nb = [h3i.grid_ring(int(c), 1) for c in uc]
lens = np.fromiter((len(v) for v in nb), int, len(nb)); flat = np.fromiter((y for v in nb for y in v), np.uint64, lens.sum())
start = np.r_[0, np.cumsum(lens)[:-1]]
p = np.clip(np.searchsorted(sid, flat), 0, len(sid) - 1); hit = sid[p] == flat; posn = np.where(hit, order[p], -1)
del nb, flat
for k in range(6):                        # every red pair -> its k-th ring neighbour
    j = start[inv] + np.minimum(k, lens[inv] - 1)
    tgt = posn[j]; ok = tgt >= 0
    KEYS.append(tgt[ok].astype(np.int64) * 65536 + spc[ok])
key = np.unique(np.concatenate(KEYS)); del KEYS
pos_ = key // 65536; sp_ = key % 65536
red_ring = np.bincount(pos_, minlength=len(cid))
out = {"LocalityID": lm, "red_ring": red_ring}
for f in ["Colias hecla", "Mellinus crabroneus", "Primula farinosa"]:
    code = np.searchsorted(spn, f)
    s = np.zeros(len(cid), np.int64)
    if code < len(spn) and spn[code] == f:
        s[pos_[sp_ == code]] = 1
    out["selfring_" + f.replace(" ", "_")] = s
pl.DataFrame(out).write_parquet("red_ring.parquet"); print("done", round(time.time() - t0), "s, mean red_ring", red_ring.mean().round(3))
