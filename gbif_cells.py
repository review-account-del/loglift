"""
gbif_cells.py - per-cell tables from the raw GBIF records (rec/*.parquet made
by gbif_extract.py), for the full data and for filtered variants.

Usage: python gbif_cells.py <variant>
  full       all records (reproduces LocalityMaster: checked against it)
  precise    records with coordinateUncertaintyInMeters known and <= 100 m
  early      year 2000-2017      late   year 2018-2026 (median year of the records: 2018)
  artportalen  records from Artportalen only (71 % of all records)
Writes variants/<variant>/cells.parquet (LocalityID, n_records, n_events,
n_species, n_obs, effort_index, red_disk, self_<sp>), focal_cells.parquet
(8 focal species) and species200_cells.parquet (200-species set).

Definitions (identical to the per-cell table of the parent pipeline, LocalityMaster;
verified for 'full'):
  n_records  records in the cell
  n_events   distinct event dates, each taken at the precision given by GBIF's
             interpreted year/month/day (an incomplete date such as a year-month
             is its own value); matches the LocalityMaster value in 99.5 % of cells
  n_species  distinct GBIF 'species' values (records not identified to species
             are counted in n_records but not in n_species)
  E          0.5 n_events + 0.5 n_species / n_events
"""
import polars as pl, numpy as np, sys, os, time, glob
import h3
from h3.api import basic_int as h3i

VAR = sys.argv[1] if len(sys.argv) > 1 else "full"
REC = os.environ.get("REC", "rec")  # output folder of gbif_extract.py
OUT = f"variants/{VAR}"; os.makedirs(OUT, exist_ok=True)
t0 = time.time()
FOCAL = ["Aglais urticae", "Colias hecla", "Achillea millefolium", "Bartsia alpina",
         "Cerceris rybyensis", "Mellinus crabroneus", "Primula farinosa", "Primula veris"]
RED = set(pl.read_parquet("red_cell_species.parquet")["species"].unique().to_list())
SP200 = set(pl.read_csv("species200.tsv", separator="\t")["species"].to_list())

lf = pl.scan_parquet(sorted(glob.glob(f"{REC}/*.parquet")))
if VAR == "precise":
    lf = lf.filter(pl.col("unc").is_not_null() & (pl.col("unc") <= 100))
elif VAR == "early":
    lf = lf.filter(pl.col("year") <= 2017)
elif VAR == "late":
    lf = lf.filter(pl.col("year") >= 2018)
elif VAR == "artportalen":
    lf = lf.filter(pl.col("dataset") == "38b4c89f-584c-41bb-bd8f-cd1def33e92f")
lf = lf.with_columns(
    (pl.col("year").fill_null(0) * 10000 + pl.col("month").fill_null(0) * 100 + pl.col("day").fill_null(0)).alias("daykey"),
    pl.when(pl.col("sp").is_not_null()).then(pl.col("sp").hash()).otherwise(None).alias("sph"))

agg = (lf.select(["cell", "daykey", "sph", "obs"]).group_by("cell").agg(
            pl.len().alias("n_records"),
            pl.col("daykey").n_unique().alias("n_days"),
            pl.col("sph").drop_nulls().n_unique().alias("n_species"),
            pl.col("obs").drop_nulls().n_unique().alias("n_obs"))
       .collect(engine="streaming"))
# events: distinct complete dates; a cell with only undated records gets one event
agg = agg.with_columns(pl.col("n_days").alias("n_events"))
agg = agg.with_columns((0.5 * pl.col("n_events") + 0.5 * pl.col("n_species") / pl.col("n_events")).alias("effort_index"))
print(VAR, "cells", agg.height, f"[{time.time()-t0:.0f}s]", flush=True)

# red-listed (cell, species) pairs -> k = 1 disk
redp = (lf.filter(pl.col("sp").is_in(list(RED))).select(["cell", "sp"]).unique().collect(engine="streaming"))
uc = np.unique(redp["cell"].to_numpy())
nb = np.array([h3i.grid_disk(int(c), 1) for c in uc], dtype=object)
lens = np.array([len(x) for x in nb])
rep_cell = np.repeat(uc, lens); disk = np.concatenate([np.array(x, dtype=np.uint64) for x in nb])
m = pl.DataFrame({"cell": rep_cell, "dcell": disk})
exp = redp.join(m, on="cell").select([pl.col("dcell").alias("cell"), "sp"]).unique()
red_all = exp.group_by("cell").agg(pl.len().alias("red_disk"))
for f in sorted(set(FOCAL) & RED):
    col = "self_" + f.replace(" ", "_")
    s = exp.filter(pl.col("sp") == f).select("cell").unique().with_columns(pl.lit(1).alias(col))
    red_all = red_all.join(s, on="cell", how="left").with_columns(pl.col(col).fill_null(0))
cells = agg.join(red_all, on="cell", how="left").fill_null(0)
cells = cells.with_columns(pl.col("cell").map_elements(lambda c: "h3_" + format(c, "x"), return_dtype=pl.Utf8).alias("LocalityID"))
cells.drop("cell").write_parquet(f"{OUT}/cells.parquet")
print(VAR, "red pairs", exp.height, f"[{time.time()-t0:.0f}s]", flush=True)

occ = (lf.filter(pl.col("sp").is_in(FOCAL + list(SP200))).select(["sp", "cell"]).unique().collect(engine="streaming")
         .with_columns(pl.col("cell").map_elements(lambda c: "h3_" + format(c, "x"), return_dtype=pl.Utf8).alias("LocalityID"))
         .rename({"sp": "species"}).drop("cell"))
occ.filter(pl.col("species").is_in(FOCAL)).write_parquet(f"{OUT}/focal_cells.parquet")
occ.filter(pl.col("species").is_in(list(SP200))).write_parquet(f"{OUT}/species200_cells.parquet")
print(VAR, "done", f"[{time.time()-t0:.0f}s]", flush=True)
