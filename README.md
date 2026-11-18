# logLift: effort-stratified association scores from opportunistic occurrence records

Code for the manuscript *logLift: an effort-stratified estimator of species association strength from
opportunistic occurrence data, and how many effort strata it needs* (under review). The code reproduces
every number, figure and table in the manuscript and its supplement.

logLift compares how often the sites of a focal species have a site property (for example, at least two
red-listed species nearby) with how often other sites of the same recording effort have it. Sites are
grouped into quantile strata of an effort index; the background rate in each stratum is weighted by the
focal species' share of sites in that stratum, and the two rates are compared as a smoothed log2 risk
ratio, averaged over thresholds of the property.

## Scripts

| Script | What it does / writes |
|---|---|
| `loglift_core.py` | the estimators: effort strata (tied values kept together), logLift, raw ratio, PMI, NPMI, Fisher odds ratio, Jaccard, GLMs (stratum factor, log-linear, restricted cubic spline, Firth, negative binomial) with g-computation and diagnostics, bootstrap tables |
| `bench_v05.py` | shared set-up (seed 20260907, focal species, properties, thresholds) |
| `build_dataset.py` | per-cell table (`cells.parquet`, `focal_cells.parquet`) from the input tables below |
| `loglift_bench.py` | estimators of the first submitted version, used by `validate_core.py` to check the new core |
| `validate_core.py`, `test_support.py` | checks of the table-based core and of common support |
| `neighbour_effort.py` | effort summed over the seven-cell neighbourhood (`neighbour_effort.parquet`) |
| `red_ring.py` | red-listed species in the six neighbouring cells only (`red_ring.parquet`) |
| `gbif_extract.py`, `gbif_cells.py`, `gbif_describe.py` | compact record table from the raw GBIF download; per-cell tables for the data subsets (`variants/<subset>/`); description of the data |
| `run_null_v05.py [null\|pert\|strata]` | null benchmark, effort perturbation, strata sweep |
| `run_real_v05.py` | the eight focal species: estimates, per-threshold values, GLM diagnostics, site and spatial block bootstraps |
| `run_positive_v05.py`, `run_positive_mh.py` | positive controls (planted associations), Mantel–Haenszel comparison |
| `run_mh_real.py` | Mantel–Haenszel odds ratio for the real species |
| `run_proxies_v05.py` (`PARTS=abc`) | alternative effort indices and mismatched proxies |
| `run_species200_v05.py` | the 200-species comparison (`species200.tsv`) |
| `run_variants_v05.py` | data subsets (precise coordinates, Artportalen only, two periods) |
| `run_ring_v05.py` | neighbourhood effort and two-dimensional strata |
| `run_gamma_v05.py` | empirical preferential-sampling exponent |
| `run_timing_v05.py` | timing |
| `make_figures_v05.py`, `make_tables_v05.py` | figures and supplementary tables |

Results are written to `results_v05/` and figures to `figures_v05/`.

## Input data

- Occurrence records: GBIF.org (2026) GBIF Occurrence Download, https://doi.org/10.15468/dl.qr49ak
  (Sweden, SIMPLE_PARQUET).
- Red-listed species: the Swedish Red List (SLU Artdatabanken), matched to GBIF species names
  (`red_cell_species.parquet`: distinct cell × red-listed species pairs).
- `species200.tsv`: the 200-species set used in the comparison (included).

`build_dataset.py` reads per-cell tables of the parent pipeline (`LocalityMaster_SE.parquet`,
`red_cell_species.parquet`, `focal_records.parquet`, `cell_groups.parquet`). The same per-cell table can be
rebuilt from the raw download with `gbif_extract.py` followed by `gbif_cells.py full`, which reproduces
it in 99.5 % of the cells. The derived tables are too large for this repository and will be deposited
with a DOI on acceptance.

## Running

Python 3.13; package versions in `requirements.txt`. Run all scripts in one working folder that holds
the input tables.

    python build_dataset.py
    python neighbour_effort.py
    python red_ring.py
    python run_null_v05.py null
    python run_null_v05.py pert
    python run_null_v05.py strata
    python run_real_v05.py
    python run_positive_v05.py
    python run_positive_mh.py
    python run_mh_real.py
    python run_proxies_v05.py
    python run_species200_v05.py
    python run_variants_v05.py      # needs variants/ from gbif_cells.py
    python run_ring_v05.py
    python run_gamma_v05.py
    python run_timing_v05.py
    python make_figures_v05.py
    python make_tables_v05.py

To rebuild the data subsets, place the GBIF download in `occurrence.parquet/` and run `gbif_extract.py`,
then `gbif_cells.py <full|precise|artportalen|early|late>` and `gbif_describe.py`. Each run script takes
minutes on two cores; run at most two heavy scripts at a time on a 16 GB machine.
