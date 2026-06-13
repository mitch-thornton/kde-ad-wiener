#!/usr/bin/env python3
"""exp_cyber_unsw_v27.py -- network-traffic density estimation on UNSW-NB15.

Held-out negative log-likelihood of four estimators (Silverman global bandwidth, improved
Sheather-Jones, AD-Wiener with the residue floor, and the GMM + AD-Wiener superposition) on
continuous traffic features, on a benign-only model and on a benign-plus-attack mixture. The mixture
component is the bundled BIC Gaussian-mixture EM (adkde_plugins / exp_datagen), NOT an external
library. Produces fig_cyber.pdf and prints the NLL table.

DATA (public, not bundled): the standard UNSW-NB15 train/test partition
(UNSW_NB15_training-set.csv, UNSW_NB15_testing-set.csv), Moustafa & Slay 2015, Australian Centre for
Cyber Security. Place both CSVs in a directory and pass --data-dir. They are mirrored on GitHub, e.g.
raw.githubusercontent.com/Nir-J/ML-Projects/master/UNSW-Network_Packet_Classification/ .

USAGE: python3 exp_cyber_unsw_v27.py --data-dir /path/to/unsw [--out fig_cyber.pdf]
"""
import os, sys, argparse
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ad_kde_v30 as K
import adkde_plugins as P
import exp_datagen_v30 as E
from KDEpy.bw_selection import silvermans_rule, improved_sheather_jones

FEATURES = ["rate", "sload", "dload", "dmean"]   # continuous, multi-scale traffic features
FIG_FEATURES = ["rate", "sload"]
N_TRAIN_BEN, N_TEST_BEN, N_TRAIN_ATK, N_TEST_ATK = 6000, 6000, 3000, 3000


def _load(data_dir):
    import pandas as pd
    fr = os.path.join(data_dir, "UNSW_NB15_training-set.csv")
    fe = os.path.join(data_dir, "UNSW_NB15_testing-set.csv")
    df = pd.concat([pd.read_csv(fr), pd.read_csv(fe)], ignore_index=True)
    return df[df.label == 0], df[df.label == 1]


def _feat(s):
    x = np.log1p(np.clip(np.asarray(s, float), 0, None))
    return x[np.isfinite(x)]


def _silver(d, xg):
    h = max(silvermans_rule(d.reshape(-1, 1)), 1e-3)
    return np.mean(np.exp(-0.5*((xg[:, None]-d[None, :])/h)**2)/(h*np.sqrt(2*np.pi)), axis=1)
def _isj(d, xg):
    try: h = improved_sheather_jones(d.reshape(-1, 1))
    except Exception: h = silvermans_rule(d.reshape(-1, 1))
    h = max(h, 1e-3)
    return np.mean(np.exp(-0.5*((xg[:, None]-d[None, :])/h)**2)/(h*np.sqrt(2*np.pi)), axis=1)
def _adw(d, xg): return np.clip(K.ad_wiener(d, xg, strip="residue"), 0, None)
def _superpose(d, xg): return np.clip(E.superpose(d, xg)[0], 0, None)

EST = {"Silverman": _silver, "ISJ/Botev": _isj, "AD-Wiener": _adw, "superpose": _superpose}


def _nll(fg, xg, xt):
    f = np.clip(np.interp(xt, xg, fg), 1e-8, None)
    return -np.mean(np.log(f))


def run(data_dir, fig_out):
    ben_df, atk_df = _load(data_dir)
    rng = np.random.default_rng(0)
    rows = {}
    for f in FEATURES:
        b = _feat(ben_df[f].values); a = _feat(atk_df[f].values)
        rng.shuffle(b); rng.shuffle(a)
        btr, bte = b[:N_TRAIN_BEN], b[N_TRAIN_BEN:N_TRAIN_BEN+N_TEST_BEN]
        atr, ate = a[:N_TRAIN_ATK], a[N_TRAIN_ATK:N_TRAIN_ATK+N_TEST_ATK]
        lo = min(b.min(), a.min()); hi = np.percentile(np.concatenate([b, a]), 99.9)
        xg = np.linspace(lo, hi, 2048)
        mtr, mte = np.concatenate([btr, atr]), np.concatenate([bte, ate])
        for scen, (dtr, dte) in [("benign", (btr, bte)), ("benign+attack", (mtr, mte))]:
            rows[(f, scen)] = {e: _nll(EST[e](dtr, xg), xg, dte) for e in EST}

    print("\nUNSW-NB15 held-out NLL (lower is better)")
    print("%-10s %-14s %9s %9s %9s %9s" % ("feature", "scenario", *EST.keys()))
    for f in FEATURES:
        for scen in ("benign", "benign+attack"):
            r = rows[(f, scen)]
            best = min(r, key=r.get)
            print("%-10s %-14s " % (f, scen) +
                  " ".join(("%8.3f*" if e == best else "%8.3f ") % r[e] for e in EST))

    # figure: benign+attack density, Silverman (global) vs AD-Wiener (adaptive)
    fig, axes = plt.subplots(1, len(FIG_FEATURES), figsize=(7.2, 2.7))
    for ax, f in zip(np.atleast_1d(axes), FIG_FEATURES):
        b = _feat(ben_df[f].values); a = _feat(atk_df[f].values)
        rng2 = np.random.default_rng(2); rng2.shuffle(b); rng2.shuffle(a)
        mix = np.concatenate([b[:5000], a[:3000]])
        lo, hi = mix.min(), np.percentile(mix, 99.9); xg = np.linspace(lo, hi, 2048)
        ax.hist(mix, bins=80, density=True, color="0.85", label="benign+attack")
        ax.plot(xg, _silver(mix, xg), color="0.35", ls="--", lw=1.1, label="Silverman (global)")
        ax.plot(xg, _adw(mix, xg), color="0.0", lw=1.1, label="AD-Wiener")
        ax.set_title(r"$\log(1+\mathrm{%s})$" % f, fontsize=9)
        ax.tick_params(labelsize=7)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        ax.legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(fig_out, bbox_inches="tight")
    print("\nwrote", fig_out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", default="figures/fig_cyber.pdf")
    a = ap.parse_args()
    run(a.data_dir, a.out)
