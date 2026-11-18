"""
gbif_extract.py - compact per-record table from the raw GBIF download
(DOI 10.15468/dl.qr49ak, SIMPLE_PARQUET, 35,869,153 records), for the
sensitivity analyses requested in review (dates, duplicates, coordinate
uncertainty, datasets, observers, time periods).

Output: rec/part_XXXX.parquet with one row per record:
  cell   u64   H3 res-11 cell (int form)
  sp     str   GBIF 'species' (null if not identified to species)
  grp    str   Lepidoptera / Hymenoptera / Tracheophyta / OtherInsecta / Other
  year, month, day          GBIF-interpreted date parts
  interval bool             eventDate is a range ('/')
  unc    f32   coordinateUncertaintyInMeters
  basis  str   basisOfRecord
  dataset str  datasetKey
  obs    u64   hash of recordedBy (null if missing)
  rounded bool issue COORDINATE_ROUNDED
"""
import polars as pl, numpy as np, glob, os, sys, time
from h3.api import basic_int as h3i

SRC = "occurrence.parquet"
OUT = "rec"
os.makedirs(OUT, exist_ok=True)
parts = [p for p in sorted(glob.glob(f"{SRC}/*")) if os.path.getsize(p) > 0]
CH = 150
t0 = time.time()
for k in range(0, len(parts), CH):
    out = f"{OUT}/part_{k:05d}.parquet"
    if os.path.exists(out):
        continue
    df = (pl.scan_parquet(parts[k:k + CH])
          .select(["species", "phylum", "class", "order", "decimallatitude", "decimallongitude",
                   "coordinateuncertaintyinmeters", "eventdate", "year", "month", "day",
                   "basisofrecord", "datasetkey", "recordedby", "issue"])
          .filter(pl.col("decimallatitude").is_not_null() & pl.col("decimallongitude").is_not_null())
          .with_columns(
              pl.when(pl.col("order") == "Lepidoptera").then(pl.lit("Lepidoptera"))
                .when(pl.col("order") == "Hymenoptera").then(pl.lit("Hymenoptera"))
                .when(pl.col("phylum") == "Tracheophyta").then(pl.lit("Tracheophyta"))
                .when(pl.col("class") == "Insecta").then(pl.lit("OtherInsecta"))
                .otherwise(pl.lit("Other")).alias("grp"),
              pl.col("eventdate").str.contains("/").fill_null(False).alias("interval"),
              pl.col("coordinateuncertaintyinmeters").cast(pl.Float32).alias("unc"),
              pl.when(pl.col("recordedby").list.len() > 0)
                .then(pl.col("recordedby").list.join("|").str.to_lowercase().str.strip_chars().hash())
                .otherwise(None).alias("obs"),
              pl.col("issue").list.contains("COORDINATE_ROUNDED").fill_null(False).alias("rounded"),
          )
          .select(["species", "grp", "decimallatitude", "decimallongitude", "unc", "year", "month", "day",
                   "interval", "basisofrecord", "datasetkey", "obs", "rounded"])
          .collect())
    lat = df["decimallatitude"].to_list(); lon = df["decimallongitude"].to_list()
    cell = np.fromiter((h3i.latlng_to_cell(a, b, 11) for a, b in zip(lat, lon)), dtype=np.uint64, count=len(lat))
    df = (df.drop(["decimallatitude", "decimallongitude"])
            .with_columns(pl.Series("cell", cell))
            .rename({"species": "sp", "basisofrecord": "basis", "datasetkey": "dataset"}))
    df.write_parquet(out)
    print(k, df.height, f"{time.time() - t0:.0f}s", flush=True)
print("DONE", time.time() - t0, flush=True)
