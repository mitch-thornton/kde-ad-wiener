#!/usr/bin/env python3
"""make_crsp_figs.py -- build the CRSP real-data figures from results/*.json.

Reads the JSON written by analyze_crsp_adkde.py (placed under results/) and writes
figures/fig_crsp_dist.pdf (a representative return density on a log scale) and
figures/fig_crsp_tail.pdf (estimated vs historical one-percent Expected Shortfall across all
series). Also prints the aggregate tail-accuracy numbers used in the paper tables. Self-contained:
only numpy/matplotlib and the shipped results/ JSON are needed.
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
FIG = os.path.join(HERE, "..", "figures")
METHODS = ["Gaussian", "KDE", "AD-Wiener", "Mixture"]


def load(name):
    return json.load(open(os.path.join(RES, name)))

def series_list(d):
    return [k for k in d if not k.startswith("_")]


def fig_distribution(dist, label="SP500"):
    d = dist[label]
    xg = np.array(d["grid"]) * 100.0
    edges = np.array(d["hist_edges"]) * 100.0
    counts = np.array(d["hist_counts"]) / 100.0          # density per percent
    plt.rcParams.update({"font.size": 8.5, "axes.linewidth": 0.8})
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ax.bar(0.5 * (edges[:-1] + edges[1:]), counts, width=np.diff(edges),
           color="0.85", edgecolor="0.7", lw=0.2, label="empirical")
    sty = {"Gaussian": ("--", "0.55"), "AD-Wiener": (":", "0.4"), "Mixture": ("-.", "0.15")}
    for m in ["Gaussian", "AD-Wiener", "Mixture"]:
        f = np.array(d["density"][m]) / 100.0
        ls, c = sty[m]
        ax.plot(xg, f, ls, color=c, lw=1.1, label=m)
    ax.set_yscale("log"); ax.set_ylim(1e-3, 1e0)
    ax.set_xlim(-8, 8)
    ax.set_xlabel("daily return (%)"); ax.set_ylabel("density (log scale)")
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    fig.tight_layout()
    out = os.path.join(FIG, "fig_crsp_dist.pdf")
    fig.savefig(out); print("figure written:", os.path.basename(out), "(%s)" % label)


def fig_tail(tail):
    series = series_list(tail)
    hist = np.array([tail[s]["historical"]["0.010"] for s in series]) * 100.0
    plt.rcParams.update({"font.size": 8.5, "axes.linewidth": 0.8})
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    marks = {"Gaussian": ("o", "0.0"), "KDE": ("^", "0.5"),
             "AD-Wiener": ("s", "0.35"), "Mixture": ("D", "0.2")}
    lim = [min(hist) * 1.1, max(hist) * 0.9]
    ax.plot(lim, lim, color="0.7", lw=0.8, ls=":")
    for m in METHODS:
        est = np.array([tail[s]["methods"][m]["0.010"] for s in series]) * 100.0
        mk, c = marks[m]
        fc = c if m == "Gaussian" else "none"
        ax.scatter(hist, est, marker=mk, s=22, facecolors=fc, edgecolors=c, lw=0.9, label=m)
    ax.set_xlabel("historical ES$_{1\\%}$ (%)"); ax.set_ylabel("estimated ES$_{1\\%}$ (%)")
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout()
    out = os.path.join(FIG, "fig_crsp_tail.pdf")
    fig.savefig(out); print("figure written:", os.path.basename(out))


def aggregate_table(var, tail):
    series = series_list(var)
    print("\n=== mean |deviation from historical| across %d series (bps) ===" % len(series))
    print("%-8s %9s %9s %9s %9s" % ("metric", *METHODS))
    for src, name, a in [(var, "VaR 1%", "0.010"), (var, "VaR 5%", "0.050"),
                         (tail, "ES 1%", "0.010"), (tail, "ES 5%", "0.050")]:
        row = [np.mean([abs(src[s]["methods"][m][a] - src[s]["historical"][a]) * 1e4
                        for s in series]) for m in METHODS]
        print("%-8s %9.1f %9.1f %9.1f %9.1f" % (name, *row))


if __name__ == "__main__":
    dist, var, tail = load("return_distributions.json"), load("var.json"), load("tail_risk.json")
    label = "SP500" if "SP500" in dist else series_list(dist)[0]
    fig_distribution(dist, label)
    fig_tail(tail)
    aggregate_table(var, tail)
