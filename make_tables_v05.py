"""make_tables_v05.py - supplementary tables (markdown pipe tables) from results_v05/*.csv"""
import numpy as np, pandas as pd, json, os
RES = "results_v05"
SP = ["Aglais urticae", "Colias hecla", "Achillea millefolium", "Bartsia alpina",
      "Cerceris rybyensis", "Mellinus crabroneus", "Primula veris", "Primula farinosa"]
NAME = {"raw": "Raw", "tgb": "TGB", "ll10": "logLift K=10", "ll100": "logLift K=100", "ll1000": "logLift K=1,000",
        "ll10000": "logLift K=10,000", "glm_factor": "GLM factor", "glm_linear": "GLM log-lin.", "glm_spline": "GLM spline",
        "glm_firth": "GLM Firth", "glm_nb": "NB GLM", "pmi": "PMI", "npmi": "NPMI", "fisher_log2OR": "Fisher log2 OR", "jaccard": "Jaccard"}
PROP = {"red": "red-listed neighbours", "richness": "species richness"}
out = []


def md(df, digits=2):
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else f"{v:.{digits}f}")
    cols = [str(c) for c in d.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in d.itertuples(index=False):
        lines.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(lines)


def table(n, caption, df, digits=2, note=None):
    out.append(f"**Table S{n}.** {caption}\n\n{md(df, digits)}\n" + (f"\n{note}\n" if note else ""))


real = pd.read_csv(f"{RES}/real_species.csv")
m1 = ["raw", "tgb", "ll10", "ll100", "ll10000", "glm_factor", "glm_linear", "glm_spline", "glm_firth", "glm_nb", "pmi", "npmi", "fisher_log2OR", "jaccard"]
p = real.pivot_table(index=["prop", "species"], columns="method", values="estimate")[m1]
p = p.reindex([(a, s) for a in ["red", "richness"] for s in SP]).reset_index()
p["prop"] = p.prop.map(PROP); p = p.rename(columns=NAME | {"prop": "Property", "species": "Species"})
table(1, "Point estimates (mean over thresholds, log2 scale except NPMI and Jaccard) for all estimators, eight species and two properties. Full background; the negative-binomial GLM on a fixed 500,000-cell background subsample.", p, 2)

rows = []
for prop in ["red", "richness"]:
    for s in SP:
        for m in ["raw", "ll10", "ll100", "ll10000", "glm_linear", "glm_spline", "glm_firth"]:
            r = real[(real.prop == prop) & (real.species == s) & (real.method == m)]
            if r.empty: continue
            r = r.iloc[0]
            f = lambda a, b: "" if pd.isna(r.get(a)) else f"{r[a]:.2f} – {r[b]:.2f}"
            rows.append({"Property": PROP[prop], "Species": s, "Estimator": NAME[m], "Estimate": r.estimate,
                         "Site bootstrap": f("site_lo", "site_hi"), "Blocks, res. 6": f("block_res6_lo", "block_res6_hi"),
                         "Blocks, res. 5": f("block_res5_lo", "block_res5_hi")})
table(2, "95 % bootstrap intervals (200 replicates; full background resampled in every replicate). Blocks: all focal sites within the same H3 cell of resolution 6 (~36 km²) or 5 (~250 km²) resampled together. Mellinus crabroneus (six cells) is shown for illustration only.", pd.DataFrame(rows), 2)

null = pd.read_csv(f"{RES}/null_test.csv")
t3 = null.groupby(["prop", "gamma", "method"]).estimate.mean().unstack("method")[m1].reset_index()
t3["prop"] = t3.prop.map(PROP); t3 = t3.rename(columns=NAME | {"prop": "Property", "gamma": "γ"})
table(3, "Null benchmark: mean over 20 pseudo-species of 2,000 cells drawn with P ∝ E^γ (no association). Effort-adjusted risk-ratio estimators should be zero; the Jaccard index has no null value of zero.", t3, 3)

ss = pd.read_csv(f"{RES}/strata_sensitivity.csv")
v = ss[ss.scheme == "v05"]
t4 = v.groupby(["prop", "K", "K_eff", "gamma"]).agg(mean=("estimate", "mean"), sd=("estimate", "std"),
                                                    dropped=("focal_dropped", "mean"), min_bg=("min_bg_in_focal_strata", "median")).reset_index()
t4["Kn"] = t4.K.map(lambda k: 1e9 if k == "exact" else int(k)); t4 = t4.sort_values(["prop", "Kn", "gamma"])
t4w = t4.pivot_table(index=["prop", "Kn", "K", "K_eff"], columns="gamma", values="mean").reset_index()
t4w.columns = [str(c) for c in t4w.columns]
sdw = t4.pivot_table(index=["prop", "Kn"], columns="gamma", values="sd").reset_index()
t4w["SD (γ = 2)"] = sdw[2.0].to_numpy()
dr = t4.pivot_table(index=["prop", "Kn"], columns="gamma", values="dropped").reset_index()
t4w["focal sites dropped (γ = 2)"] = dr[2.0].to_numpy()
mb = t4.pivot_table(index=["prop", "Kn"], columns="gamma", values="min_bg").reset_index()
t4w["min. background cells in a focal stratum (γ = 2)"] = mb[2.0].to_numpy()
nocs = ss[(ss.scheme == "v05_no_common_support")].groupby(["prop", "K", "gamma"]).estimate.mean().unstack("gamma").reset_index()
t4w = t4w.merge(nocs.rename(columns={0.0: "no CS γ=0", 1.0: "no CS γ=1", 2.0: "no CS γ=2"}), on=["prop", "K"], how="left")
t4w = t4w.drop(columns=["Kn"]).rename(columns={"prop": "Property", "K": "nominal K", "K_eff": "effective strata", "0.0": "γ = 0", "1.0": "γ = 1", "2.0": "γ = 2"})
t4w["Property"] = t4w.Property.map(PROP)
t4w["focal sites dropped (γ = 2)"] = t4w["focal sites dropped (γ = 2)"].map(lambda x: f"{100 * x:.1f} %")
table(4, "Null bias of logLift by number of strata (15 pseudo-species per setting, common support). 'no CS': without dropping focal sites in strata without background cells.", t4w, 3)

if os.path.exists(f"{RES}/proxy_null_table.csv"):
    pn = pd.read_csv(f"{RES}/proxy_null.csv")
    pn["corr"] = pn.correction + " (" + pn.K.astype(str) + ")"
    t5 = pn[pn.K.astype(str).isin(["10000", "exact", "-"])].groupby(["prop", "gamma", "sampling", "corr"]).estimate.mean().unstack("corr").reset_index()
    t5["prop"] = t5.prop.map(PROP)
    table(5, "Effort index. (a) Mismatched null: pseudo-species sampled with P ∝ S^γ on one index (rows) and corrected by logLift with another (columns; K = 10,000 nominal quantile strata, or two-dimensional strata). Mean of 10 pseudo-species.", t5.rename(columns={"prop": "Property", "gamma": "γ", "sampling": "sampled on"}), 2)
    pr = pd.read_csv(f"{RES}/proxy_real.csv"); pr["corr"] = pr.correction + " (" + pr.K.astype(str) + ")"
    t5b = pr[pr.K.astype(str).isin(["10000", "exact", "-"])].pivot_table(index=["prop", "species"], columns="corr", values="estimate").reset_index()
    t5b["prop"] = t5b.prop.map(PROP)
    out.append("(b) The eight species, corrected with each index.\n\n" + md(t5b.rename(columns={"prop": "Property", "species": "Species"}), 2) + "\n")
    ps = pd.read_csv(f"{RES}/proxy_species200_summary.csv"); ps["prop"] = ps.prop.map(PROP)
    out.append("(c) 200 species: Spearman correlation with logLift corrected with E (K = 10,000).\n\n" + md(ps[["prop", "proxy", "spearman_vs_E10000", "mean", "sd"]].rename(columns={"prop": "Property", "proxy": "Index (K)", "spearman_vs_E10000": "ρ vs E", "mean": "mean", "sd": "SD"}), 2) + "\n")

g = pd.read_csv(f"{RES}/gamma_empirical.csv")
t6 = g[["species", "n_cells", "gamma_ml", "gamma_ml_se", "gamma_match_top10", "top10_share", "gamma_binned", "n_bins_used"]].copy()
t6 = t6.rename(columns={"species": "Species", "n_cells": "cells", "gamma_ml": "γ (max. likelihood)", "gamma_ml_se": "SE", "gamma_match_top10": "γ (top-decile match)",
                        "top10_share": "share of cells in top effort decile", "gamma_binned": "γ (binned, v1)", "n_bins_used": "bins used (v1)"})
cal = pd.read_csv(f"{RES}/gamma_calibration.csv")
table(6, "Empirical preferential-sampling exponent of the eight species. Pseudo-species drawn with γ = 0, 0.5, 1, 1.5 and 2 place " +
      ", ".join(f"{100 * s:.0f} %" for s in cal.top10_share[:5]) + " of their cells in the top effort decile.", t6, 2)

if os.path.exists(f"{RES}/effort_perturbation.csv"):
    pe = pd.read_csv(f"{RES}/effort_perturbation.csv")
    gm = pe.groupby(["prop", "species", "method", "gamma"]).estimate.mean().unstack("gamma")
    rng = (gm.max(axis=1) - gm.min(axis=1)).unstack("method")
    mm = [m for m in ["raw", "tgb", "ll10", "ll100", "ll10000", "glm_factor", "glm_linear", "glm_spline", "glm_firth", "glm_nb"] if m in rng.columns]
    t7 = rng[mm].reset_index(); t7["prop"] = t7.prop.map(PROP)
    table(7, "Effort perturbation: range of the mean estimate over γ = −1, 0, 1 and 2 (10 replicates each); a subsample of half of the species' cells drawn without replacement with P ∝ E^γ.", t7.rename(columns=NAME | {"prop": "Property", "species": "Species"}), 2)

if os.path.exists(f"{RES}/species200_glm_diagnostics.csv"):
    dg = pd.read_csv(f"{RES}/species200_glm_diagnostics.csv")
    main = dg[((dg.prop == "red") & dg.t.isin([1, 2, 3, 4])) | ((dg.prop == "richness") & dg.t.isin([5, 10, 20, 30, 50]))]
    s = main.groupby(["method", "prop"]).agg(fits=("converged", "size"), converged=("converged", "mean"), separated=("separation", "sum"),
                                             median_iter=("iterations", "median"), max_abs_bF=("b_focal", lambda x: np.nanmax(np.abs(x))),
                                             max_se=("se_focal", lambda x: np.nanmax(x))).reset_index()
    s["converged"] = s.converged.map(lambda x: f"{100 * x:.1f} %"); s["prop"] = s.prop.map(PROP)
    table(8, "GLM diagnostics for the 200 species (one fit per species, property and threshold). Separated: no focal site, or every focal site, exceeds the threshold.", s.rename(columns=NAME | {"method": "Model", "prop": "Property"}), 1)

if os.path.exists(f"{RES}/ring_bands.csv") and os.path.exists(f"{RES}/mh_bands.csv"):
    rb = pd.read_csv(f"{RES}/ring_bands.csv"); mb = pd.read_csv(f"{RES}/mh_bands.csv")
    t1 = rb.pivot_table(index=["species", "prop"], columns="band", values="ll").reset_index()
    t1["Score"] = t1.prop.map({"red_disk": "logLift, red-listed in 7-cell disk", "red_ring": "logLift, red-listed in 6 neighbours only"})
    t2 = mb[mb.prop == "red"].pivot_table(index=["species"], columns="band", values="mh").reset_index(); t2["Score"] = "MH odds ratio, red-listed in 7-cell disk"
    t9 = pd.concat([t1.drop(columns=["prop"]), t2]).sort_values(["species", "Score"])
    t9 = t9[["species", "Score", 1, 2, 3, 4, 5]]; t9.columns = ["Species", "Score (log2)", "band 1", "band 2", "band 3", "band 4", "band 5"]
    mp = pd.read_csv(f"{RES}/mh_perturbation.csv")
    g = mp.groupby(["species", "prop", "gamma"])[["ll", "mh"]].mean()
    rng = g.groupby(level=[0, 1]).agg(lambda x: x.max() - x.min()).reset_index()
    rng = rng[rng.prop == "red"][["species", "ll", "mh"]].rename(columns={"ll": "logLift", "mh": "MH odds ratio"})
    table(9, "Red-listed neighbours by effort band (K = 10,000 strata): scores within five bands of each species' own effort distribution (band 1: the fifth of its sites with the lowest effort). logLift with red-listed species counted in the 7-cell disk (as everywhere else) or in the six neighbouring cells only, and the Mantel–Haenszel log2 odds ratio over the same strata.", t9, 2)
    out.append("Effort perturbation, red-listed neighbours: range of the mean score over γ = −1 … 2 (as Table S7).\n\n" + md(rng.rename(columns={"species": "Species"}), 2) + "\n")

if os.path.exists(f"{RES}/variants.csv"):
    vv = pd.read_csv(f"{RES}/variants.csv")
    a = vv[vv.focal8 & (vv.method == "ll10000")].pivot_table(index=["prop", "species"], columns="variant", values="estimate")
    a = a[[c for c in ["full", "precise", "artportalen", "early", "late"] if c in a.columns]].reset_index(); a["prop"] = a.prop.map(PROP)
    nf = vv[vv.focal8 & (vv.method == "ll10000")].pivot_table(index=["species"], columns="variant", values="n_focal", aggfunc="first")
    table(10, "Data subsets: logLift (K = 10,000) recomputed after rebuilding effort, both properties and the occupied cells from subsets of the raw records: all records (rebuilt), coordinate uncertainty stated and ≤ 100 m, Artportalen only, 2000–2017 and 2018–2026.", a.rename(columns={"prop": "Property", "species": "Species"}), 2)
    vs = pd.read_csv(f"{RES}/variants_summary.csv"); vs["prop"] = vs.prop.map(PROP)
    out.append("200 species: rank correlation with the full data.\n\n" + md(vs.rename(columns=NAME | {"prop": "Property", "method": "Estimator"}), 2) + "\n")

if os.path.exists(f"{RES}/threshold_sets.csv"):
    ts = pd.read_csv(f"{RES}/threshold_sets.csv"); ts["prop"] = ts.prop.map(PROP); ts["method"] = ts.method.map(NAME)
    table(11, "Threshold sets, 200 species: Spearman correlation of the ranking with the main set (red-listed neighbours t = 1–4, richness t = 5–50).", ts.rename(columns={"prop": "Property", "method": "Estimator", "set": "Thresholds", "spearman_vs_main": "ρ vs main"}), 2)

if os.path.exists(f"{RES}/positive_control_summary.csv"):
    pc = pd.read_csv(f"{RES}/positive_control_summary.csv")
    cols = ["prop", "S", "rho", "gamma", "truth_marginal", "truth_cond", "raw", "ll10", "ll100", "ll10000", "glm_linear", "glm_spline", "ll10000_S"]
    pc = pc[[c for c in cols if c in pc.columns]].copy(); pc["prop"] = pc.prop.map(PROP)
    table(12, "Positive controls: artificial species drawn with P ∝ S^γ ρ^H (H: ≥ 2 red-listed neighbours or ≥ 20 species). truth_marginal: the association produced by ρ without preferential sampling; truth_cond: the same within fine strata of S. Estimators correct with E, except 'logLift, S': strata of S itself. Mean of 10 replicates.",
          pc.rename(columns=NAME | {"prop": "Property", "S": "sampling process S", "rho": "ρ", "gamma": "γ", "ll10000_S": "logLift, S"}), 2)

if os.path.exists(f"{RES}/timing.csv"):
    tm = pd.read_csv(f"{RES}/timing.csv")[["step", "seconds"]]
    tm["milliseconds"] = tm.seconds * 1000
    table(13, "Single-core time (median of repeated runs) for Primula farinosa, red-listed neighbours (4 thresholds), with the count tables of all cells computed once per property.", tm[["step", "milliseconds"]], 1)

open("supplementary_tables_v05.md", "w").write("\n\n".join(out))
print("written", len(out), "blocks")
