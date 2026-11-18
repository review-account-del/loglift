"""
gbif_describe.py - description of the GBIF download for the revised Methods
(review R2, lines 65-77): contributing datasets and record types, dates,
duplicates, coordinate uncertainty, recorder names, H3 cell areas.
"""
import polars as pl, numpy as np, json, glob, os, time, urllib.request
import h3
REC = os.environ.get("REC", "rec")  # output folder of gbif_extract.py
lf = pl.scan_parquet(sorted(glob.glob(f"{REC}/*.parquet")))
out = {}
n = lf.select(pl.len()).collect().item(); out["n_records"] = n
ds = lf.group_by("dataset").agg(pl.len().alias("n")).sort("n", descending=True).collect()
out["n_datasets"] = ds.height
top = ds.head(10).to_dicts()
for d in top:
    try:
        with urllib.request.urlopen(f"https://api.gbif.org/v1/dataset/{d['dataset']}", timeout=20) as r:
            j = json.load(r); d["title"] = j.get("title"); d["type"] = j.get("type")
    except Exception as e:
        d["title"] = None
    d["share"] = d["n"] / n
out["top_datasets"] = top
out["top1_share"] = top[0]["share"]; out["top5_share"] = sum(d["share"] for d in top[:5])
out["basis"] = {r["basis"]: r["n"] / n for r in lf.group_by("basis").agg(pl.len().alias("n")).collect().to_dicts()}
u = lf.select("unc").collect()["unc"].to_numpy()
known = u[np.isfinite(u)]
out["unc_missing_share"] = float(np.mean(~np.isfinite(u)))
out["unc_quantiles_known"] = {str(q): float(np.quantile(known, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)}
for c in (10, 25, 50, 100, 250, 1000, 5000):
    out[f"unc_le_{c}_share_of_all"] = float(np.mean(known <= c) * len(known) / n)
out["rounded_share"] = lf.select(pl.col("rounded").mean()).collect().item()
d = lf.select(["year", "month", "day", "interval"]).collect()
out["complete_date_share"] = float((d["day"].is_not_null() & d["month"].is_not_null()).mean())
out["interval_share"] = float(d["interval"].mean())
out["interval_resolved_to_day_share"] = float((d["interval"] & d["day"].is_not_null()).sum() / max(d["interval"].sum(), 1))
yr = d["year"].to_numpy()
out["year_median"] = float(np.median(yr)); out["year_q25_q75"] = [float(np.quantile(yr, .25)), float(np.quantile(yr, .75))]
out["share_le_2015"] = float(np.mean(yr <= 2015))
del d
out["recordedby_missing_share"] = lf.select(pl.col("obs").is_null().mean()).collect().item()
out["species_missing_share"] = lf.select(pl.col("sp").is_null().mean()).collect().item()
dup = (lf.filter(pl.col("sp").is_not_null() & pl.col("day").is_not_null())
         .select(["cell", pl.col("sp").hash().alias("sph"), (pl.col("year") * 10000 + pl.col("month") * 100 + pl.col("day")).alias("dk")])
         .group_by(["cell", "sph", "dk"]).agg(pl.len().alias("k"))
         .select([(pl.col("k") - 1).sum().alias("extra"), pl.col("k").sum().alias("tot")]).collect())
out["same_species_cell_day_extra_records"] = int(dup["extra"][0]); out["same_species_cell_day_extra_share"] = float(dup["extra"][0] / dup["tot"][0])
# H3 cell areas of the recorded cells
ids = pl.read_parquet("cells.parquet", columns=["LocalityID", "lat_r"])
a = np.fromiter((h3.cell_area(s[3:], "m^2") for s in ids["LocalityID"].to_list()), float, ids.height)
lat = ids["lat_r"].to_numpy()
out["h3_area_m2"] = dict(mean=float(a.mean()), min=float(a.min()), max=float(a.max()), cv=float(a.std() / a.mean()),
                         p01=float(np.quantile(a, .01)), p99=float(np.quantile(a, .99)),
                         mean_lat_55_57=float(a[(lat >= 55) & (lat < 57)].mean()), mean_lat_67_69=float(a[(lat >= 67) & (lat < 69.5)].mean()))
out["h3_res11_global_mean_area_m2"] = h3.average_hexagon_area(11, "m^2")
out["h3_edge_mean_m_sweden"] = float(np.sqrt(2 * a.mean() / (3 * np.sqrt(3))))
json.dump(out, open("results_v05/gbif_description.json", "w"), indent=1, default=float)
print(json.dumps(out, indent=1, default=float)[:6000])
