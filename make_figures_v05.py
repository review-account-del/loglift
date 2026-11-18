"""make_figures_v05.py - figures and tables for the revised manuscript (v0.5), from results_v05/*.csv"""
import numpy as np, pandas as pd, matplotlib, os, json, sys
matplotlib.use("Agg")
import matplotlib.pyplot as plt
RES = "results_v05"; FIG = "figures_v05"
os.makedirs(FIG, exist_ok=True)
ONLY = set(sys.argv[1:])

C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100",
     "magenta": "#e87ba4", "green": "#008300", "violet": "#4a3aa7", "red": "#e34948", "grey": "#7a7a7a", "navy": "#1d3f73"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.axisbelow": True, "legend.frameon": False})
METHODS = [("raw", "No correction", C["red"], "o"),
           ("tgb", "Target-group background", C["magenta"], "D"),
           ("ll10", "logLift, K = 10 (6 strata)", C["yellow"], "^"),
           ("ll100", "logLift, K = 100 (28 strata)", C["blue"], "s"),
           ("ll1000", "logLift, K = 1000 (108 strata)", C["navy"], "v"),
           ("glm_factor", "GLM, stratum factor (K = 100)", C["aqua"], "P"),
           ("glm_linear", "GLM, log-linear effort", C["green"], "X"),
           ("glm_spline", "GLM, spline effort", C["violet"], "<"),
           ("glm_firth", "GLM, log-linear effort, Firth", C["grey"], ">"),
           ("glm_nb", "Negative-binomial GLM", "#b07a2a", "h"),
           ("pmi", "PMI", "#7f3f98", "<"), ("npmi", "NPMI", "#c49bd0", "<"),
           ("fisher_log2OR", "Fisher log$_2$ OR", "#555555", ">"), ("jaccard", "Jaccard index", "#a0a0a0", "d")]
LAB = {m: (l, c, k) for m, l, c, k in METHODS}
SP_ORDER = ["Aglais urticae", "Colias hecla", "Achillea millefolium", "Bartsia alpina",
            "Cerceris rybyensis", "Mellinus crabroneus", "Primula veris", "Primula farinosa"]
SP_SHORT = {s: s.split()[0][0] + ". " + s.split()[1] for s in SP_ORDER}
PT = {"red": "red-listed neighbours", "richness": "species richness"}


def want(name):
    return not ONLY or name in ONLY


def save(fig, name):
    fig.savefig(f"{FIG}/{name}.png", dpi=300); fig.savefig(f"{FIG}/{name}.pdf"); plt.close(fig)


# --------------------------------------------------------- Fig S1: effort ----
if want("S1"):
    import polars as pl
    import loglift_core as lc
    E = pl.read_parquet("cells.parquet", columns=["effort_index"])["effort_index"].to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    ax = axes[0]
    bins = np.logspace(np.log10(0.45), np.log10(E.max() * 1.1), 70)
    ax.hist(E, bins=bins, color=C["blue"], alpha=0.85)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(top=1e9); ax.set_xlabel("effort index E"); ax.set_ylabel("number of cells")
    for K, col, y0 in [(10, C["yellow"], 0.93), (100, C["orange"], 0.83), (10000, C["navy"], 0.73)]:
        s, k = lc.effort_strata(E, K)
        ub = np.array([E[s == i].max() for i in range(k)])[:-1]
        ax.vlines(ub, y0, y0 + 0.06, transform=ax.get_xaxis_transform(), color=col, lw=0.6)
        ax.text(0.99, y0 + 0.03, f"K = {K:,}:\n{k} strata", transform=ax.transAxes, ha="right", va="center", color=col, fontsize=6, linespacing=1.0)
    ax.set_title("A  Distribution of E (log scales)", loc="left", fontsize=8.5)
    ax = axes[1]
    v, cnt = np.unique(E, return_counts=True); cs = np.cumsum(cnt) / cnt.sum()
    ax.step(v, cs, where="post", color=C["blue"]); ax.set_xscale("log")
    ax.set_xlabel("effort index E"); ax.set_ylabel("cumulative share of cells")
    for e, lab, dx, dy in [(0.5, "E = 0.5: no record identified to species (4 %)", 22, 12), (1.0, "E = 1: one species, one day (48 %)", 8, -12), (1.5, "E = 1.5 (14 %)", 8, -12)]:
        ax.annotate(lab, (e, cs[v == e][0]), xytext=(dx, dy), textcoords="offset points", fontsize=6.5, color=INK2)
    ax.set_title("B  Cumulative distribution", loc="left", fontsize=8.5)
    fig.tight_layout(); save(fig, "figS1_effort_distribution")
    print("S1 done")


def interval_cols(row, pref):
    lo, hi = row.get(f"{pref}_lo", np.nan), row.get(f"{pref}_hi", np.nan)
    return lo, hi


# ---------------------------------------------------------------- Fig 1 ------
if want("1") and os.path.exists(f"{RES}/real_species.csv"):
    real = pd.read_csv(f"{RES}/real_species.csv")
    main = ["raw", "tgb", "ll10", "ll100", "ll10000", "glm_linear", "glm_spline"]
    LAB["ll10000"] = ("logLift, K = 10,000 (417 strata)", C["navy"], "v")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.8), sharey=True)
    for ax, prop, title in zip(axes, ["red", "richness"], ["A  Red-listed neighbours (t = 1–4)", "B  Cell species richness (t = 5–50)"]):
        d = real[real.prop == prop]; y = np.arange(len(SP_ORDER))[::-1]; off = np.linspace(-0.36, 0.36, len(main))
        for m, o in zip(main, off):
            lab, col, mk = LAB[m]
            for i, sp in enumerate(SP_ORDER):
                r = d[(d.species == sp) & (d.method == m)]
                if r.empty: continue
                r = r.iloc[0]
                lo, hi = r.get("block_res6_lo", np.nan), r.get("block_res6_hi", np.nan)
                if sp != "Mellinus crabroneus" and np.isfinite(lo):
                    ax.plot([lo, hi], [y[i] + o] * 2, color=col, lw=1.2, solid_capstyle="round")
                ax.plot(r.estimate, y[i] + o, marker=mk, color=col, ms=4.2, mec="white", mew=0.5, label=lab if i == 0 else None, zorder=3)
        ax.axvline(0, color=INK2, lw=0.8); ax.set_yticks(y); ax.set_yticklabels([SP_SHORT[s] for s in SP_ORDER], fontstyle="italic")
        ax.set_xlabel("mean log$_2$ lift"); ax.set_title(title, loc="left", fontsize=8.5); ax.grid(axis="y", visible=False)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=6.5, handlelength=1.2, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.1, 1, 1)); save(fig, "fig1_estimates"); print("Fig 1 done")

# ---------------------------------------------------------------- Fig 2 ------
if want("2") and os.path.exists(f"{RES}/null_test.csv"):
    null = pd.read_csv(f"{RES}/null_test.csv")
    LAB["ll10000"] = ("logLift, K = 10,000 (417 strata)", C["navy"], "v")
    show = ["raw", "tgb", "ll10", "ll100", "ll10000", "glm_factor", "glm_linear", "glm_spline"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4))
    for ax, prop, title in zip(axes, ["red", "richness"], ["A  Red-listed neighbours", "B  Species richness"]):
        d = null[null.prop == prop].groupby(["method", "gamma"]).estimate.agg(["mean", "std"]).reset_index()
        for j, m in enumerate(show):
            lab, col, mk = LAB[m]; dd = d[d.method == m].sort_values("gamma")
            ax.errorbar(dd.gamma + (j - 3.5) * 0.012, dd["mean"], yerr=1.96 * dd["std"], color=col, marker=mk, ms=3.8, lw=1.2, capsize=0, label=lab, mec="white", mew=0.5)
        ax.axhline(0, color=INK2, lw=0.8); ax.set_xlabel("preferential-sampling exponent γ  (P ∝ E$^γ$)")
        ax.set_ylabel("mean log$_2$ lift (no association)"); ax.set_title(title, loc="left", fontsize=8.5); ax.set_xticks([0, 0.5, 1, 2])
    axes[1].legend(fontsize=6.2, loc="upper left", handlelength=1.5)
    fig.tight_layout(); save(fig, "fig2_null"); print("Fig 2 done")

# ---------------------------------------------------------------- Fig 4 ------
if want("4") and os.path.exists(f"{RES}/strata_sensitivity.csv"):
    ss = pd.read_csv(f"{RES}/strata_sensitivity.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    for ax, prop, title in zip(axes, ["red", "richness"], ["A  Red-listed neighbours", "B  Species richness"]):
        d = ss[(ss.prop == prop) & (ss.scheme == "v05")]
        for g, col, mk in [(0.0, C["grey"], "D"), (1.0, C["blue"], "o"), (2.0, C["orange"], "s")]:
            dd = d[(d.gamma == g) & (d.K != "exact")].groupby("K_eff").estimate.agg(["mean", "std"])
            ax.errorbar(dd.index, dd["mean"], yerr=1.96 * dd["std"], color=col, marker=mk, ms=3.5, lw=1.3, capsize=0, label=f"γ = {g:.0f}", mec="white", mew=0.5)
            ex = d[(d.gamma == g) & (d.K == "exact")].estimate.mean()
            exn = ss[(ss.prop == prop) & (ss.scheme == "v05_no_common_support") & (ss.gamma == g) & (ss.K == "exact")].estimate.mean()
            ax.plot(11969, ex, marker=mk, color=col, ms=5, mfc="white")
            ax.plot(11969, exn, marker="x", color=col, ms=5)
        ax.axhline(0, color=INK2, lw=0.8); ax.set_xscale("log")
        ks = d[d.K != "exact"].drop_duplicates("K_eff")[["K", "K_eff"]].sort_values("K_eff")
        sel = ks[ks.K.isin(["10", "100", "1000", "10000"])]
        for j, (_, r) in enumerate(sel.iterrows()):
            ax.annotate(f"K = {int(r.K):,}", (r.K_eff, ax.get_ylim()[1]), xytext=(0, -2 - 9 * (j % 2)), textcoords="offset points", ha="center", va="top", fontsize=6, color=INK2)
            ax.axvline(r.K_eff, color=GRID, lw=0.8, zorder=0)
        ax.annotate("one stratum per\nvalue of E:\n○ common support\n× all focal sites", (11969, ax.get_ylim()[0]), xytext=(0, 4), textcoords="offset points", ha="right", va="bottom", fontsize=5.8, color=INK2)
        ax.set_xlabel("effective number of effort strata"); ax.set_ylabel("null log$_2$ lift (truth = 0)"); ax.set_title(title, loc="left", fontsize=8.5)
    axes[0].legend(fontsize=6.5, loc="center left")
    fig.tight_layout(); save(fig, "fig4_strata"); print("Fig 4 done")

LAB["ll10000"] = ("logLift, K = 10,000 (417 strata)", C["navy"], "v")

# ---------------------------------------------------------------- Fig 3 ------
if want("3") and os.path.exists(f"{RES}/effort_perturbation.csv"):
    pert = pd.read_csv(f"{RES}/effort_perturbation.csv")
    show3 = ["raw", "tgb", "glm_linear", "glm_spline", "ll10", "ll100", "ll10000"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.4), sharey=True)
    spp = [s for s in SP_ORDER if s in pert.species.unique()]
    for ax, prop, title in zip(axes, ["red", "richness"], ["A  Red-listed neighbours", "B  Species richness"]):
        d = pert[pert.prop == prop].groupby(["species", "method", "gamma"]).estimate.mean().reset_index()
        y = np.arange(len(spp))[::-1]; off = np.linspace(-0.34, 0.34, len(show3))
        for m, o in zip(show3, off):
            lab, col, mk = LAB[m]
            for i, sp in enumerate(spp):
                dd = d[(d.species == sp) & (d.method == m)].sort_values("gamma")
                ax.plot([dd.estimate.min(), dd.estimate.max()], [y[i] + o] * 2, color=col, lw=1.2, alpha=0.9)
                ax.plot(dd.estimate.iloc[0], y[i] + o, marker="|", color=col, ms=6)
                ax.plot(dd.estimate.iloc[-1], y[i] + o, marker=mk, color=col, ms=4.2, mec="white", mew=0.5, label=lab if i == 0 else None, zorder=3)
        ax.axvline(0, color=INK2, lw=0.8); ax.set_yticks(y); ax.set_yticklabels([SP_SHORT[s] for s in spp], fontstyle="italic")
        ax.set_xlabel("mean log$_2$ lift  (bar: range over γ = −1 … +2;\n tick at γ = −1, symbol at γ = +2)"); ax.set_title(title, loc="left", fontsize=8.5); ax.grid(axis="y", visible=False)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=6.3, handlelength=1.2, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.08, 1, 1)); save(fig, "fig3_perturbation"); print("Fig 3 done")

# ---------------------------------------------------------------- Fig 5 ------
if want("5") and os.path.exists(f"{RES}/positive_control_summary.csv"):
    pc = pd.read_csv(f"{RES}/positive_control_summary.csv")
    show = ["raw", "ll10", "ll100", "ll10000", "glm_linear", "glm_spline"]
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.8), sharex=True)
    for row, prop in enumerate(["red", "richness"]):
        for col_i, rho in enumerate([1.0, 2.0, 4.0]):
            ax = axes[row, col_i]
            d = pc[(pc.prop == prop) & (pc.S == "E") & (pc.rho == rho)].sort_values("gamma")
            for m in show:
                lab, col, mk = LAB[m]
                ax.plot(d.gamma, d[m], color=col, marker=mk, ms=3.8, lw=1.2, mec="white", mew=0.5, label=lab)
            tm = d.truth_marginal.iloc[0]
            ax.axhline(tm, color=INK, lw=1.0, ls="--", label="planted association (γ = 0, no correction)")
            ax.axhline(0, color=INK2, lw=0.6)
            ax.set_title(f"{'AB'[row]}{col_i + 1}  {PT[prop]}, ρ = {rho:.0f}", loc="left", fontsize=7.5)
            if col_i == 0: ax.set_ylabel("mean log$_2$ lift")
            if row == 1: ax.set_xlabel("γ"); ax.set_xticks([0, 1, 2])
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=6.2, handlelength=1.4, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.09, 1, 1)); save(fig, "fig5_positive_controls"); print("Fig 5 done")

# ---------------------------------------------------------------- Fig 6 ------
if want("6") and os.path.exists(f"{RES}/species200.csv"):
    s2 = pd.read_csv(f"{RES}/species200.csv")
    dg = pd.read_csv(f"{RES}/species200_glm_diagnostics.csv")
    sep = dg[(dg.method == "glm_linear") & (((dg.prop == "red") & dg.t.isin([1, 2, 3, 4])) | ((dg.prop == "richness") & dg.t.isin([5, 10, 20, 30, 50])))]
    sepk = set(map(tuple, sep[sep.separation & (sep.focal_exceed == 0)][["species", "prop"]].drop_duplicates().to_numpy()))
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 6.4))
    pairs = [("ll10", "logLift, K = 10 (6 strata)"), ("raw", "no correction"),
             ("glm_linear", "GLM, log-linear effort (max. likelihood)"), ("glm_firth", "GLM, log-linear effort (Firth)")]
    lim = [-4.6, 2.4] 
    for ax, (a_, xl), t in zip(axes.ravel(), pairs, "ABCD"):
        for prop, col, mk, lab in [("red", C["blue"], "o", "red-listed neighbours"), ("richness", C["orange"], "^", "species richness")]:
            d = s2[s2.prop == prop]
            x = d[a_].to_numpy(); yv = d["ll10000"].to_numpy()
            isS = np.array([(sp, prop) in sepk for sp in d.species]) if a_ == "glm_linear" else np.zeros(len(d), bool)
            out = x < lim[0]
            ax.scatter(x[~out & ~isS], yv[~out & ~isS], s=10, color=col, marker=mk, alpha=0.75, edgecolor="white", linewidth=0.3, label=lab)
            if (isS | out).any():
                ax.scatter(np.clip(x[isS | out], lim[0] + 0.08, None), yv[isS | out], s=26, facecolor="none", edgecolor=col, marker="s", linewidth=0.9,
                           label="no focal site above a threshold (separation)" if a_ == "glm_linear" else "below the axis range")
        if a_ == "raw":
            ax.set_xlim(-3.6, 4.6)
            ax.plot([-3.6, 4.6], [-3.6, 4.6], color=INK2, lw=0.8, ls="--")
        else:
            ax.set_xlim(lim); ax.plot(lim, lim, color=INK2, lw=0.8, ls="--")
        ax.set_ylim(lim[0], 2.4)
        ax.set_xlabel(xl, fontsize=7); ax.set_ylabel("logLift, K = 10,000 (417 strata)", fontsize=7)
        ax.set_title(t, loc="left", fontsize=8.5)
    axes[0, 0].legend(fontsize=6, loc="upper left")
    h, l = axes[1, 0].get_legend_handles_labels()
    axes[1, 0].legend([h[1]], [l[1]], fontsize=5.8, loc="lower right")
    fig.tight_layout(); save(fig, "fig6_species200"); print("Fig 6 done")

# ------------------------------------------------- Fig S2: co-occurrence -----
if want("S2") and os.path.exists(f"{RES}/real_species.csv"):
    real = pd.read_csv(f"{RES}/real_species.csv"); null = pd.read_csv(f"{RES}/null_test.csv")
    cm = ["raw", "pmi", "fisher_log2OR", "npmi", "jaccard", "ll10000"]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4))
    for j, prop in enumerate(["red", "richness"]):
        ax = axes[0, j]; d = real[real.prop == prop]; y = np.arange(len(SP_ORDER))[::-1]; off = np.linspace(-0.3, 0.3, len(cm))
        for m, o in zip(cm, off):
            lab, col, mk = LAB[m]
            for i, sp in enumerate(SP_ORDER):
                r = d[(d.species == sp) & (d.method == m)]
                if not r.empty:
                    ax.plot(r.estimate.iloc[0], y[i] + o, marker=mk, color=col, ms=4, mec="white", mew=0.4, label=lab if i == 0 else None)
        ax.axvline(0, color=INK2, lw=0.8); ax.set_yticks(y); ax.set_yticklabels([SP_SHORT[s] for s in SP_ORDER], fontstyle="italic", fontsize=6.5)
        ax.set_title(f"{'AB'[j]}  Real species, {PT[prop]}", loc="left", fontsize=8); ax.grid(axis="y", visible=False)
        ax.set_xlabel("value (log$_2$ scale for raw, PMI, OR, logLift)")
        ax = axes[1, j]; d = null[null.prop == prop].groupby(["method", "gamma"]).estimate.mean().reset_index()
        for m in cm:
            lab, col, mk = LAB[m]; dd = d[d.method == m].sort_values("gamma")
            ax.plot(dd.gamma, dd.estimate, color=col, marker=mk, ms=3.8, lw=1.2, mec="white", mew=0.4)
        ax.axhline(0, color=INK2, lw=0.8); ax.set_xticks([0, 0.5, 1, 2]); ax.set_xlabel("γ"); ax.set_ylabel("mean value, null pseudo-species")
        ax.set_title(f"{'CD'[j]}  Null pseudo-species, {PT[prop]}", loc="left", fontsize=8)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=6, fontsize=6.2, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.05, 1, 1)); save(fig, "figS2_cooccurrence"); print("S2 done")

# ------------------------------------------------- Fig S3: per threshold -----
if want("S3") and os.path.exists(f"{RES}/real_species_per_threshold.csv"):
    pt = pd.read_csv(f"{RES}/real_species_per_threshold.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3))
    cols = plt.cm.tab10(np.arange(8))
    for ax, prop in zip(axes, ["red", "richness"]):
        for k, sp in enumerate(SP_ORDER):
            for m, ls in [("ll10000", "-"), ("raw", ":")]:
                d = pt[(pt.prop == prop) & (pt.species == sp) & (pt.method == m)].sort_values("t")
                ax.plot(d.t, d.value, ls=ls, color=cols[k], marker="o" if m == "ll10000" else None, ms=3, lw=1.1,
                        label=SP_SHORT[sp] if (m == "ll10000") else None)
        ax.axhline(0, color=INK2, lw=0.8); ax.set_xlabel("threshold t"); ax.set_ylabel("log$_2$ lift at threshold t")
        if prop == "richness": ax.set_xscale("log"); ax.set_xticks([5, 10, 20, 30, 50]); ax.set_xticklabels([5, 10, 20, 30, 50])
        ax.set_title(f"{'AB'[prop == 'richness']}  {PT[prop]} (solid: logLift K = 10,000; dotted: raw)", loc="left", fontsize=7.5)
    axes[0].legend(fontsize=6, ncol=2, loc="lower left")
    fig.tight_layout(); save(fig, "figS3_per_threshold"); print("S3 done")

# ------------------------------------------------- Fig S4: GLM fit ------------
if want("S4"):
    import polars as pl, warnings
    import statsmodels.api as sm
    import loglift_core as lc
    from scipy.special import expit
    cells = pl.read_parquet("cells.parquet", columns=["effort_index", "red_disk", "n_species"]).to_pandas()
    U = lc.Universe(cells.effort_index.to_numpy(float), Ks=(100,))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
    for ax, (col, t, title) in zip(axes, [("red_disk", 2, "A  red-listed neighbours ≥ 2"), ("n_species", 20, "B  species richness ≥ 20")]):
        ev = (cells[col].to_numpy() >= t).astype(int)
        J = U.joint(np.arange(len(ev)), ev.astype(np.int16), 2)
        Jx = U.collapse(J, U.lev2x, len(U.xval)); n = Jx.sum(1); y = Jx[:, 1]; ok = n > 0
        x = U.xval[ok]; n = n[ok]; y = y[ok]
        p_obs = y / n
        big = n >= 200
        ax.scatter(np.expm1(x[big]), p_obs[big], s=np.clip(n[big] / 2000, 2, 30), color=C["grey"], alpha=0.6, label="observed share (bins with ≥ 200 cells)")
        for kind, colr, lab in [("linear", C["green"], "logistic, linear in log(1+E)"), ("spline", C["violet"], "logistic, restricted cubic spline")]:
            Z = x[:, None] if kind == "linear" else lc.rcs_basis(x, U.knots)
            X = sm.add_constant(Z)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = sm.GLM(np.column_stack([y, n - y]), X, family=sm.families.Binomial()).fit(maxiter=100)
            ax.plot(np.expm1(x), fit.predict(X), color=colr, lw=1.4, label=lab)
        ax.set_xscale("log"); ax.set_xlabel("effort index E"); ax.set_ylabel("share of background cells exceeding t")
        ax.set_title(title, loc="left", fontsize=8)
    axes[0].legend(fontsize=6, loc="upper left")
    fig.tight_layout(); save(fig, "figS4_glm_effort_fit"); print("S4 done")

# ------------------------------------------------- Fig S5: interval widths ----
if want("S5") and os.path.exists(f"{RES}/real_species.csv"):
    real = pd.read_csv(f"{RES}/real_species.csv")
    d = real[(real.method == "ll10000") & (real.species != "Mellinus crabroneus")].copy()
    for p in ("site", "block_res6", "block_res5"):
        d[p] = d[f"{p}_hi"] - d[f"{p}_lo"]
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    xs = np.arange(len(d))
    for k, (p, col, lab) in enumerate([("site", C["blue"], "site bootstrap"), ("block_res6", C["orange"], "blocks, H3 res. 6 (~36 km²)"), ("block_res5", C["red"], "blocks, H3 res. 5 (~250 km²)")]):
        ax.bar(xs + (k - 1) * 0.27, d[p], width=0.26, color=col, label=lab)
    ax.set_xticks(xs); ax.set_xticklabels([f"{SP_SHORT[s]}\n{'red' if pr == 'red' else 'rich.'}" for s, pr in zip(d.species, d.prop)], fontsize=5.5, rotation=90)
    ax.set_ylabel("width of 95 % interval (log$_2$ units)"); ax.legend(fontsize=6.5)
    ax.set_title("logLift, K = 10,000: width of bootstrap intervals", loc="left", fontsize=8)
    fig.tight_layout(); save(fig, "figS5_interval_widths"); print("S5 done")

# ------------------------------------------------- Fig S6: effort index -------
if want("S6") and os.path.exists(f"{RES}/proxy_null.csv"):
    pn = pd.read_csv(f"{RES}/proxy_null.csv"); pr = pd.read_csv(f"{RES}/proxy_real.csv")
    corr_order = ["E|10000", "n_events|10000", "n_records|10000", "n_obs|10000", "disk_E|10000", "E10000xdisk10|exact", "days_x_listlength|exact", "raw|-"]
    corr_lab = {"E|10000": "E", "n_events|10000": "days", "n_records|10000": "records", "n_obs|10000": "recorders",
                "disk_E|10000": "E, 7-cell disk", "E10000xdisk10|exact": "E × disk E", "days_x_listlength|exact": "days × list length", "raw|-": "none (raw)"}
    samp_order = ["E", "n_events", "n_records", "n_obs", "disk_E"]
    samp_lab = {"E": "E", "n_events": "days", "n_records": "records", "n_obs": "recorders", "disk_E": "E, 7-cell disk"}
    pn["key"] = pn.correction + "|" + pn.K.astype(str)
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 4.0), constrained_layout=True)
    for ax, prop in zip(axes, ["red", "richness"]):
        m = pn[(pn.prop == prop) & (pn.gamma == 2.0)].groupby(["sampling", "key"]).estimate.mean().unstack("key")
        m = m.reindex(index=[s for s in samp_order if s in m.index], columns=[c for c in corr_order if c in m.columns])
        im = ax.imshow(m.to_numpy(), cmap="RdBu_r", vmin=-1.5, vmax=1.5, aspect="auto")
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                v = m.iat[i, j]; ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5.6, color="white" if abs(v) > 0.9 else INK)
        ax.set_xticks(range(m.shape[1])); ax.set_xticklabels([corr_lab[c] for c in m.columns], rotation=45, ha="right", fontsize=6)
        ax.set_yticks(range(m.shape[0])); ax.set_yticklabels([samp_lab[s] for s in m.index] if prop == "red" else [], fontsize=6.5)
        ax.set_xlabel("index used for correction (logLift)", fontsize=7)
        if prop == "red": ax.set_ylabel("index that drives sampling (γ = 2)", fontsize=7)
        ax.set_title(f"{'AB'[prop == 'richness']}  {PT[prop]}", loc="left", fontsize=8); ax.grid(False)
    fig.colorbar(im, ax=axes, shrink=0.7, label="mean null log$_2$ lift")
    save(fig, "figS6_effort_index_null"); print("S6 done")

# ------------------------------------------------- Fig S7: data subsets -------
if want("S7") and os.path.exists(f"{RES}/variants.csv"):
    v = pd.read_csv(f"{RES}/variants.csv")
    v = v[v.focal8 & (v.method == "ll10000")]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), sharey=True)
    vs = [("full", "all records", INK, "o"), ("precise", "uncertainty ≤ 100 m", C["blue"], "s"), ("artportalen", "Artportalen only", C["orange"], "D"),
          ("early", "2000–2017", C["aqua"], "^"), ("late", "2018–2026", C["violet"], "v")]
    for ax, prop in zip(axes, ["red", "richness"]):
        y = np.arange(len(SP_ORDER))[::-1]; off = np.linspace(-0.3, 0.3, len(vs))
        for (var, lab, col, mk), o in zip(vs, off):
            for i, sp in enumerate(SP_ORDER):
                r = v[(v.variant == var) & (v.species == sp) & (v.prop == prop)]
                if not r.empty:
                    ax.plot(r.estimate.iloc[0], y[i] + o, marker=mk, color=col, ms=4, mec="white", mew=0.4, label=lab if i == 0 else None)
        ax.axvline(0, color=INK2, lw=0.8); ax.set_yticks(y); ax.set_yticklabels([SP_SHORT[s] for s in SP_ORDER], fontstyle="italic")
        ax.set_xlabel("logLift, K = 10,000"); ax.set_title(f"{'AB'[prop == 'richness']}  {PT[prop]}", loc="left", fontsize=8); ax.grid(axis="y", visible=False)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, fontsize=6.3, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.07, 1, 1)); save(fig, "figS7_data_subsets"); print("S7 done")
