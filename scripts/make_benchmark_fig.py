#!/usr/bin/env python3
"""Regenerate figures/fig_benchmark.pdf: average rank of seven estimators on the fifteen
Marron-Wand densities at n = 100, 500, 5000.

The per-density mean integrated-squared-error matrix below is exactly the body of
Tables tab:bench100 / tab:bench500 / tab:benchmark in the paper (mean over fifty
replications, scored by exact ISE against the closed-form Marron-Wand densities). The
average ranks are recomputed here from that matrix with average-tie ranking, so the
figure and the tables are guaranteed consistent and the figure needs no raw data.
"""
import numpy as np
from scipy.stats import rankdata
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

METHODS = ["Silverman", "ISJ/Botev", "LSCV", "Abramson", "GMM-BIC", "AD-Wiener", "super (GMM)"]

# 15 densities x 7 estimators, mean ISE x1e3 (== published tables)
ISE = {
 100: [[5.83,18.36,10.53,7.24,2.64,9.98,2.64],[9.05,29.52,13.38,9.57,12.18,11.15,13.30],
       [142.2,86.35,48.64,112.5,87.07,53.24,53.45],[101.7,97.43,52.40,58.83,79.23,55.80,56.22],
       [61.34,313.5,85.01,71.00,45.32,56.71,31.90],[8.73,21.18,11.02,8.68,16.96,12.46,17.54],
       [46.74,32.59,14.94,40.26,9.12,16.15,16.34],[12.57,22.48,12.99,12.75,22.89,13.61,21.21],
       [11.16,20.63,12.69,11.33,16.47,13.59,16.11],[53.17,44.85,42.47,54.59,57.88,45.19,55.26],
       [10.21,22.44,12.10,10.35,20.10,13.56,18.70],[27.81,31.55,28.18,28.39,33.50,26.30,31.43],
       [14.42,25.66,17.54,14.07,21.44,17.68,21.92],[85.06,51.44,39.10,79.85,45.18,41.65,41.91],
       [113.5,51.56,39.15,113.5,39.55,40.69,40.55]],
 500: [[1.86,4.73,2.54,2.24,0.48,2.37,0.48],[2.85,7.52,4.14,2.60,2.33,2.74,1.86],
       [106.3,30.09,15.69,73.70,42.54,16.71,16.59],[56.23,30.20,13.96,16.31,8.28,12.47,12.61],
       [19.62,59.86,23.53,21.54,6.17,15.45,6.15],[3.01,6.05,3.16,2.19,1.61,3.16,1.84],
       [20.86,8.76,3.97,13.02,1.66,3.69,3.75],[5.20,7.10,4.09,4.03,7.76,3.73,3.86],
       [4.81,7.05,3.62,3.71,5.42,3.96,5.67],[45.95,25.03,12.49,46.03,49.20,14.21,42.47],
       [4.43,7.26,4.32,3.60,3.07,4.52,3.32],[20.82,14.41,11.03,20.12,20.49,11.70,19.71],
       [7.87,8.61,7.28,6.76,6.08,7.42,6.23],[63.54,20.89,16.40,58.54,20.36,17.43,17.44],
       [88.28,19.43,16.06,83.11,22.44,16.43,16.20]],
 5000:[[0.31,0.69,0.47,0.37,0.06,0.28,0.06],[0.48,1.12,0.72,0.51,0.46,0.37,0.24],
       [58.90,5.04,3.96,30.74,9.42,2.32,2.32],[18.41,4.43,3.04,1.88,2.12,1.63,1.63],
       [2.89,7.29,4.17,3.28,0.62,1.55,0.62],[0.60,0.94,0.61,0.38,0.15,0.38,0.16],
       [5.07,1.35,0.83,1.49,0.15,0.46,0.15],[1.18,1.16,0.82,0.49,1.29,0.44,0.18],
       [1.35,1.16,0.89,0.75,1.28,0.53,1.29],[32.93,3.99,2.69,26.42,2.35,1.50,1.50],
       [2.09,2.07,2.06,1.85,1.66,1.85,1.66],[12.51,3.32,3.49,10.73,7.76,2.57,6.17],
       [4.74,2.59,2.77,4.23,4.58,2.32,4.59],[40.79,5.46,7.57,37.55,11.24,3.80,3.78],
       [47.65,5.39,7.46,39.19,14.92,2.43,2.42]],
}

def avg_ranks(M):
    A = np.array(M)
    return np.array([rankdata(r, method="average") for r in A]).mean(0)

def main():
    sizes = [100, 500, 5000]
    R = np.array([avg_ranks(ISE[n]) for n in sizes])           # 3 x 7
    here = os.path.dirname(os.path.abspath(__file__))
    outdir = os.path.join(os.path.dirname(here), "figures")
    os.makedirs(outdir, exist_ok=True)

    styles = {  # (color, linewidth, zorder, marker)
        "AD-Wiener":   ("#b00020", 2.6, 5, "o"),
        "super (GMM)": ("#1f5fb4", 2.6, 5, "s"),
        "GMM-BIC":     ("#2a8a2a", 1.6, 3, "^"),
        "LSCV":        ("#777777", 1.4, 2, "v"),
        "Silverman":   ("#999999", 1.2, 1, "D"),
        "Abramson":    ("#aaaaaa", 1.2, 1, "P"),
        "ISJ/Botev":   ("#bbbbbb", 1.2, 1, "X"),
    }
    x = np.arange(len(sizes))
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    for j, m in enumerate(METHODS):
        c, lw, z, mk = styles[m]
        ax.plot(x, R[:, j], color=c, lw=lw, marker=mk, ms=5, zorder=z, label=m)
    ax.set_xticks(x); ax.set_xticklabels([f"$n={s}$" for s in sizes])
    ax.set_ylabel("average rank (lower is better)")
    ax.invert_yaxis()
    ax.grid(True, axis="y", ls=":", alpha=0.5)
    ax.set_title("Average rank on the Marron-Wand benchmark")
    ax.legend(fontsize=7, ncol=2, loc="lower left", framealpha=0.9)
    fig.tight_layout()
    out = os.path.join(outdir, "fig_benchmark.pdf")
    fig.savefig(out)
    print("figure written:", os.path.basename(out))

if __name__ == "__main__":
    main()
