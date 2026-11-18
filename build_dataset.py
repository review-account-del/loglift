"""
build_dataset.py — assemble the per-locality table used by the logLift benchmark.

Inputs (extracted on gis-dator16 from the GBIF Sweden SIMPLE_PARQUET download):
  LocalityMaster_SE.parquet   one row per H3 res-11 cell: effort_index, w_inv,
                              effort_stratum (rank deciles of log1p effort), n_species
  red_cell_species.parquet    distinct (cell, red-listed species) pairs
  focal_records.parquet       all records of the eight focal species
  cell_groups.parquet         records per (cell, taxonomic group) for the
                              target-group background

Output:
  cells.parquet     LocalityMaster + red_disk (distinct red-listed species within
                    the k=1 H3 disk, 7 cells) + per-focal self-corrections + group flags
  focal_cells.parquet  (species, LocalityID) occupancy
"""
import polars as pl, numpy as np, h3, time

t0 = time.time()
lm  = pl.read_parquet("LocalityMaster_SE.parquet")
red = pl.read_parquet("red_cell_species.parquet")
foc = pl.read_parquet("focal_records.parquet")
grp = pl.read_parquet("cell_groups.parquet")

FOCAL = sorted(foc["species"].unique().to_list())
red_focal = sorted(set(FOCAL) & set(red["species"].unique().to_list()))
print("focal on red list:", red_focal)

# --- expand red (cell, species) pairs to the k=1 disk -----------------------
cells = red["LocalityID"].to_numpy()
uniq = np.unique(cells)
disk = {c: ["h3_" + d for d in h3.grid_disk(c[3:], 1)] for c in uniq}
idx = {c: i for i, c in enumerate(uniq)}
sp = red["species"].to_numpy()
out_cell, out_sp = [], []
for c, s in zip(cells, sp):
    for d in disk[c]:
        out_cell.append(d); out_sp.append(s)
exp = pl.DataFrame({"LocalityID": out_cell, "species": out_sp}).unique()
print("expanded red pairs", exp.height, f"{time.time()-t0:.0f}s")

red_all = exp.group_by("LocalityID").agg(pl.len().alias("red_disk"))
self_cols = []
for f in red_focal:
    col = "self_" + f.replace(" ", "_")
    s = exp.filter(pl.col("species") == f).select("LocalityID").with_columns(pl.lit(1).alias(col))
    red_all = red_all.join(s, on="LocalityID", how="left").with_columns(pl.col(col).fill_null(0))
    self_cols.append(col)

# --- group flags (target-group background) ----------------------------------
gw = grp.pivot(on="grp", index="LocalityID", values="n", aggregate_function="sum").fill_null(0)
gw = gw.rename({c: "grp_" + c for c in gw.columns if c != "LocalityID"})

cells_df = (lm.join(red_all, on="LocalityID", how="left")
              .join(gw, on="LocalityID", how="left")
              .with_columns([pl.col("red_disk").fill_null(0)] +
                            [pl.col(c).fill_null(0) for c in self_cols] +
                            [pl.col(c).fill_null(0) for c in gw.columns if c != "LocalityID"]))
cells_df.write_parquet("cells.parquet")
print(cells_df.shape, cells_df.columns)

fc = foc.select(["species", "LocalityID"]).unique()
fc = fc.join(lm.select("LocalityID"), on="LocalityID", how="inner")   # keep cells present in the master
fc.write_parquet("focal_cells.parquet")
print(fc.group_by("species").agg(pl.len()).sort("species"))
print(f"done {time.time()-t0:.0f}s")
