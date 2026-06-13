#!/usr/bin/env python3
"""exp_sdss_redshift.py -- galaxy-redshift density estimation on SDSS spectroscopic data.

In a sky cone the galaxy redshift distribution n(z) is a smooth selection envelope on which
large-scale structure (walls, clusters, voids) prints narrow overdensities at specific
redshifts. This is the same statistical geometry as the CMS dimuon spectrum, sharp features
that carry real probability mass sitting on a smooth background, but produced by an unrelated
physical mechanism (galaxy clustering rather than quantum resonances). No single bandwidth
resolves the narrow walls and the broad envelope at once, which is the adaptive regime the
AD-Wiener estimator targets. The true density is unknown, so the estimators are scored by
held-out fit: fit on a 70 percent training split and score the mean negative log-likelihood
on the disjoint 30 percent test split (lower is better).

DATA (SDSS DR18 SpecObj table, public; accessed via the SkyServer SQL interface
https://skyserver.sdss.org/dr18/SearchTools/sql with Output Format = CSV, press Submit):

  SELECT s.z
  FROM SpecObj AS s
  JOIN dbo.fGetNearbyObjEq(200.0, 0.0, 300) AS n ON n.objID = s.bestObjID
  WHERE s.class = 'GALAXY' AND s.zWarning = 0 AND s.z BETWEEN 0.01 AND 0.25

A five-degree-radius cone about (ra, dec) = (200, 0) returns a well-sampled n(z) in which the
large-scale walls are statistically resolved. The fGetNearbyObjEq function is built for small
crossmatch radii and can be slow at this size; a coordinate box over the same field,
"AND s.ra BETWEEN 197.5 AND 202.5 AND s.dec BETWEEN -2.5 AND 2.5" with the JOIN removed, is a
faster near-equivalent. The SkyServer CSV download begins with a "#Table1" comment line, so the
reader below skips comment lines. Save as sdss_z.csv and run from the bundle's scripts/ dir:

  python3 exp_sdss_redshift_v16.py --csv sdss_z.csv
"""
import os, sys, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ad_kde_v30 as K


def held_out_nll(train, test, xg, estimate):
    f = np.clip(estimate(train, xg), 1e-12, None)
    f = f / np.trapezoid(f, xg)
    ft = np.interp(test, xg, f)
    return -np.mean(np.log(np.clip(ft, 1e-12, None)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="sdss_z.csv")
    ap.add_argument("--col", default="z")
    ap.add_argument("--zmin", type=float, default=0.01)
    ap.add_argument("--zmax", type=float, default=0.22)
    args = ap.parse_args()

    df = pd.read_csv(args.csv, comment="#")
    df.columns = [c.strip().lower() for c in df.columns]
    z = df[args.col].to_numpy().astype(float)
    z = z[np.isfinite(z)]
    z = z[(z >= args.zmin) & (z <= args.zmax)]
    print("loaded %d SDSS galaxy redshifts in [%.3f, %.3f]" % (len(z), args.zmin, args.zmax))

    rng = np.random.default_rng(0)
    perm = rng.permutation(len(z))
    cut = int(0.7 * len(z))
    tr, te = z[perm[:cut]], z[perm[cut:]]
    xg = np.linspace(z.min(), z.max(), 2048)

    ests = {
        "naive KDE": lambda d, g: K.fft_kde(d, K.h_silverman(d), g),
        "AD-bw":     lambda d, g: K.ad_bw(d, g, strip="residue"),
        "AD-Wiener": lambda d, g: K.ad_wiener(d, g, strip="residue"),
    }
    print("\nHeld-out negative log-likelihood (lower is better):")
    for name, est in ests.items():
        print("    %-12s %.4f" % (name, held_out_nll(tr, te, xg, est)))

    # full-data estimates for the figure
    fN = K.fft_kde(z, K.h_silverman(z), xg)
    fW = K.ad_wiener(z, xg, strip="residue")
    nrm = lambda f, g: np.clip(f, 0, None) / np.trapezoid(np.clip(f, 0, None), g)

    # locate the dominant wall (mode of a fine histogram below z=0.15) for the zoom
    sel = z[z < 0.15]
    h, edges = np.histogram(sel, bins=120)
    zc = 0.5 * (edges[:-1] + edges[1:])
    zwall = float(zc[int(np.argmax(h))])
    z0, z1 = max(args.zmin, zwall - 0.035), zwall + 0.035

    plt.rcParams.update({"font.size": 8.5, "axes.linewidth": 0.8})
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.0, 2.7))

    axA.hist(z, bins=120, density=True, color="0.85", edgecolor="0.7", lw=0.1)
    axA.plot(xg, nrm(fN, xg), ":", color="0.55", lw=1.1, label="naive KDE")
    axA.plot(xg, nrm(fW, xg), "-", color="0.0", lw=1.0, label="AD-Wiener")
    axA.set_xlabel("redshift $z$"); axA.set_yticks([])
    axA.legend(frameon=False, fontsize=6.5, loc="upper right")
    axA.set_title("(A) full $n(z)$: walls on a smooth selection", fontsize=8.5)

    # zoom on the dominant wall in linear z
    mask = (z > z0) & (z < z1)
    xj = np.linspace(z0, z1, 1024)
    fjN = K.fft_kde(z[mask], K.h_silverman(z[mask]), xj)
    fjW = K.ad_wiener(z[mask], xj, strip="residue")
    axB.hist(z[mask], bins=70, density=True, color="0.85", edgecolor="0.7", lw=0.1)
    axB.plot(xj, nrm(fjN, xj), ":", color="0.55", lw=1.1, label="naive KDE")
    axB.plot(xj, nrm(fjW, xj), "-", color="0.0", lw=1.0, label="AD-Wiener")
    axB.axvline(zwall, color="0.4", ls="--", lw=0.5)
    axB.set_xlabel("redshift $z$"); axB.set_yticks([])
    axB.legend(frameon=False, fontsize=6.5, loc="upper right")
    axB.set_title("(B) dominant wall, $z\\approx%.2f$" % zwall, fontsize=8.5)

    fig.tight_layout(); fig.savefig("fig_sdss_redshift.pdf")
    print("\nfigure written: fig_sdss_redshift.pdf")


if __name__ == "__main__":
    main()
