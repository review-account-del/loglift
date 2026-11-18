"""
neighbour_effort.py - effort of the k = 1 H3 neighbourhood (the cell and its
six neighbours, the same 7-cell disk over which red-listed neighbours are
counted). Sums over recorded cells of the disk.
Output: neighbour_effort.parquet (LocalityID, disk_E, disk_events, disk_records, disk_cells)
"""
import numpy as np, polars as pl, time
from h3.api import basic_int as h3i
t0 = time.time()
c = pl.read_parquet("cells.parquet", columns=["LocalityID", "effort_index", "n_events", "n_records"])
cid = np.array([int(s[3:], 16) for s in c["LocalityID"].to_list()], dtype=np.uint64)
order = np.argsort(cid); sid = cid[order]
ROW, SRC = [], []
for a in range(0, len(cid), 400_000):
    nb = [h3i.grid_disk(int(x), 1) for x in cid[a:a + 400_000]]
    lens = np.fromiter((len(v) for v in nb), int, len(nb))
    flat = np.fromiter((y for v in nb for y in v), np.uint64, lens.sum())
    src = np.repeat(np.arange(a, a + len(nb)), lens)
    p = np.searchsorted(sid, flat); p = np.clip(p, 0, len(sid) - 1)
    hit = sid[p] == flat
    ROW.append(order[p[hit]]); SRC.append(src[hit])
row = np.concatenate(ROW); src = np.concatenate(SRC)
out = {"LocalityID": c["LocalityID"]}
for name, col in [("disk_E", "effort_index"), ("disk_events", "n_events"), ("disk_records", "n_records")]:
    v = c[col].to_numpy().astype(float)
    out[name] = np.bincount(src, weights=v[row], minlength=len(cid))
out["disk_cells"] = np.bincount(src, minlength=len(cid))
pl.DataFrame(out).write_parquet("neighbour_effort.parquet")
print("done", round(time.time() - t0, 1), "s; mean recorded cells per disk", out["disk_cells"].mean().round(2))
