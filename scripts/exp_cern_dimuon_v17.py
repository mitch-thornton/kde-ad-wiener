#!/usr/bin/env python3
"""exp_cern_dimuon.py -- invariant-mass spectrum density estimation on CMS Open Data.

The CMS dimuon spectrum (record 545, column M) is a multi-scale density: narrow resonances
(eta/rho/omega/phi ~0.6-1, J/psi 3.097, psi' 3.686, Upsilon ~9.5, Z 91.2 GeV) sitting on a smooth,
steeply falling combinatorial background. No single bandwidth resolves the narrow peaks and the broad
base at once, which is the adaptive regime the AD-Wiener estimator targets. There is no ground-truth
density, so the quantitative metric is held-out fit: fit on a training split and score the mean
negative log-likelihood on a disjoint test split (lower is better). Estimation is done in
x = log10(mass) so the decade-spanning resonances are comparably scaled; the held-out NLL is computed
consistently in that variable, so differences between estimators are meaningful.

DATA (CMS Open Data record 545, CC0):
  curl -L -o dimuon.csv https://opendata.cern.ch/record/545/files/Dimuon_DoubleMu.csv
Run from the bundle's scripts/ dir:  python3 exp_cern_dimuon.py --csv dimuon.csv
"""
import os, sys, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ad_kde_v30 as K

RESONANCES = [("$J/\\psi$", 3.097), ("$\\Upsilon$", 9.46), ("$Z$", 91.19)]


def held_out_nll(train, test, xg, estimate):
    f = np.clip(estimate(train, xg), 1e-12, None)
    f = f / np.trapezoid(f, xg)
    ft = np.interp(test, xg, f)
    return -np.mean(np.log(np.clip(ft, 1e-12, None)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="dimuon.csv")
    ap.add_argument("--mmin", type=float, default=0.4)
    ap.add_argument("--mmax", type=float, default=120.0)
    args = ap.parse_args()

    M = pd.read_csv(args.csv)["M"].to_numpy()
    M = M[np.isfinite(M)]
    M = M[(M >= args.mmin) & (M <= args.mmax)]
    x = np.log10(M)
    print("loaded %d opposite-sign dimuon events in [%.1f, %.0f] GeV" % (len(x), args.mmin, args.mmax))

    rng = np.random.default_rng(0)
    perm = rng.permutation(len(x))
    cut = int(0.7 * len(x))
    tr, te = x[perm[:cut]], x[perm[cut:]]
    xg = np.linspace(x.min(), x.max(), 2048)

    ests = {
        "naive KDE":   lambda d, g: K.fft_kde(d, K.h_silverman(d), g),
        "AD-bw":       lambda d, g: K.ad_bw(d, g, strip="residue"),
        "AD-Wiener":   lambda d, g: K.ad_wiener(d, g, strip="residue"),
    }
    print("\nHeld-out negative log-likelihood (log10-mass; lower is better):")
    nll = {}
    for name, est in ests.items():
        nll[name] = held_out_nll(tr, te, xg, est)
        print("    %-12s %.4f" % (name, nll[name]))

    # full-spectrum estimates (on all data) for the figure
    fkN = K.fft_kde(x, K.h_silverman(x), xg)
    fkW = K.ad_wiener(x, xg, strip="residue")
    nrm = lambda f: np.clip(f, 0, None) / np.trapezoid(np.clip(f, 0, None), xg)

    plt.rcParams.update({"font.size": 8.5, "axes.linewidth": 0.8})
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.0, 2.7))
    axA.hist(x, bins=240, density=True, color="0.85", edgecolor="0.7", lw=0.1)
    axA.plot(xg, nrm(fkN), ":", color="0.55", lw=1.1, label="naive KDE")
    axA.plot(xg, nrm(fkW), "-", color="0.0", lw=1.0, label="AD-Wiener")
    for lab, m in RESONANCES:
        lm = np.log10(m)
        axA.axvline(lm, color="0.4", ls="--", lw=0.5)
        axA.text(lm - 0.04, axA.get_ylim()[1]*0.92, lab, fontsize=6.5, ha="right", va="top")
    axA.set_xlabel("$\\log_{10}(m_{\\mu\\mu}/\\mathrm{GeV})$"); axA.set_yticks([])
    axA.legend(frameon=False, fontsize=6.5, loc="upper right", bbox_to_anchor=(0.88, 0.84),
               handlelength=1.3, handletextpad=0.4); axA.set_title("(A) full dimuon spectrum", fontsize=8.5)

    # J/psi zoom in linear mass
    jmask = (M > 2.6) & (M < 3.6)
    xj = np.linspace(2.6, 3.6, 1024)
    fjN = K.fft_kde(M[jmask], K.h_silverman(M[jmask]), xj)
    fjW = K.ad_wiener(M[jmask], xj, strip="residue")
    nrmj = lambda f: np.clip(f, 0, None) / np.trapezoid(np.clip(f, 0, None), xj)
    axB.hist(M[jmask], bins=80, density=True, color="0.85", edgecolor="0.7", lw=0.1)
    axB.plot(xj, nrmj(fjN), ":", color="0.55", lw=1.1, label="naive KDE")
    axB.plot(xj, nrmj(fjW), "-", color="0.0", lw=1.0, label="AD-Wiener")
    axB.axvline(3.097, color="0.4", ls="--", lw=0.5)
    axB.set_xlabel("$m_{\\mu\\mu}$ (GeV)"); axB.set_yticks([])
    axB.legend(frameon=False, fontsize=6.5, loc="upper right"); axB.set_title("(B) $J/\\psi$ region", fontsize=8.5)
    fig.tight_layout(); fig.savefig("fig_cern_dimuon.pdf"); print("\nfigure written: fig_cern_dimuon.pdf")


if __name__ == "__main__":
    main()
