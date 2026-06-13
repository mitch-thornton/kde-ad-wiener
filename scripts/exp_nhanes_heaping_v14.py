#!/usr/bin/env python3
"""exp_nhanes_heaping.py (v4) -- heaped-data robustness on NHANES; TWO separate figures.

Emits:
  fig_heaping_weight.pdf      controlled, ground-truthed: measured weight with imposed heaping
                              (A) robustness vs heaping coarseness   (B) rounded-to-30-lb mechanism
  fig_heaping_cigarettes.pdf  natural, qualitative: self-reported cigarettes/day (SMD650)
                              (C) de-heaping   (D) terminal-digit preference

PART 1 imposes known heaping on real continuous weights to MEASURE each estimator's response (a
controlled robustness study, not a claim that weight is reported at 40-lb resolution). PART 2 uses a
variable that heaps coarsely on its own -- the NHANES item states "1 pack equals 20 cigarettes" --
so no forcing is involved; counts have no continuous truth, hence this part is qualitative.
Refs: NHANES II digit preference (PubMed 7561984); Wang, Shiffman, Griffith & Heitjan,
Ann. Appl. Stat. 6(4):1689-1706 (2012).

Run from scripts/:  python3 exp_nhanes_heaping.py --data-dir nhanes
"""
import os, sys, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ad_kde_v30 as K
import exp_datagen_v30 as E

KG_TO_LB = 2.2046226218
XW = np.linspace(90, 360, 1024)
XC = np.linspace(0, 70, 1024)
GRIDS = [5, 10, 20, 30, 40]


def load(data_dir):
    demo = pd.read_sas(os.path.join(data_dir, "DEMO_J.xpt"))[["SEQN", "RIDAGEYR"]]
    bmx = pd.read_sas(os.path.join(data_dir, "BMX_J.xpt"))[["SEQN", "BMXWT"]]
    df = demo.merge(bmx, on="SEQN", how="inner")
    df = df[df["RIDAGEYR"] >= 20]
    df["meas_lb"] = df["BMXWT"] * KG_TO_LB
    meas = df.dropna(subset=["meas_lb"])
    meas = meas[(meas["meas_lb"] >= 80) & (meas["meas_lb"] <= 400)]["meas_lb"].to_numpy()
    smq = pd.read_sas(os.path.join(data_dir, "SMQ_J.xpt"))
    cig = smq["SMD650"].to_numpy()
    cig = cig[np.isfinite(cig)]
    cig = cig[(cig >= 1) & (cig <= 60)]
    return meas, cig


def norm(f, xg):
    f = np.clip(f, 0, None)
    return f / np.trapezoid(f, xg)

def gkde(d, h, xg):
    return np.mean(np.exp(-0.5 * ((xg[:, None] - d[None, :]) / h) ** 2) / (h * np.sqrt(2 * np.pi)), axis=1)

def ise(f, truth, xg):
    return np.trapezoid((norm(f, xg) - truth) ** 2, xg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=".")
    args = ap.parse_args()
    meas, cig = load(args.data_dir)
    print("loaded %d adult measured weights, %d genuine cigarette/day reports" % (len(meas), len(cig)))
    plt.rcParams.update({"font.size": 8.5, "axes.linewidth": 0.8})

    # ---------- FIGURE 1: controlled weight ----------
    truth = norm(gkde(meas, K.h_silverman(meas), XW), XW)
    print("\nPART 1 -- measured weight rounded to grid D; ISE x1e3 vs no-heaping density:")
    print("%6s %10s %10s %10s %12s" % ("D (lb)", "naive KDE", "AD simple", "AD residue", "superpose"))
    sweep = {"naive KDE": [], "AD simple floor": [], "AD residue floor": [], "superposition": []}
    for D in GRIDS:
        h = np.round(meas / D) * D
        e = {"naive KDE": gkde(h, K.h_silverman(h), XW),
             "AD simple floor": K.ad_wiener(h, XW, strip="simple"),
             "AD residue floor": K.ad_wiener(h, XW, strip="residue"),
             "superposition": E.superpose(h, XW)[0]}
        for k in sweep:
            sweep[k].append(1e3 * ise(e[k], truth, XW))
        print("%6d %10.3f %10.3f %10.3f %12.3f" % (D, sweep["naive KDE"][-1], sweep["AD simple floor"][-1],
              sweep["AD residue floor"][-1], sweep["superposition"][-1]))
    Dshow = 30
    hc = np.round(meas / Dshow) * Dshow
    naive_c = gkde(hc, K.h_silverman(hc), XW); resid_c = K.ad_wiener(hc, XW, strip="residue")
    super_c = E.superpose(hc, XW)[0]

    fig1, (axA, axB) = plt.subplots(1, 2, figsize=(6.6, 2.6))
    sty = {"naive KDE": ("o-", "0.0"), "AD simple floor": ("v:", "0.6"), "AD residue floor": ("s--", "0.2"),
           "superposition": ("D-", "0.45")}
    for k, (m, c) in sty.items():
        axA.loglog(GRIDS, np.clip(sweep[k], 1e-3, None), m, color=c, lw=1.2, ms=4, label=k)
    axA.set_xlabel("heaping grid $D$ (lb)"); axA.set_ylabel("ISE ($\\times10^{3}$) vs no-heaping")
    axA.legend(frameon=False, fontsize=6.5, loc="upper left"); axA.set_title("(A) robustness to heaping coarseness", fontsize=8.5)
    axB.plot(XW, truth, color="0.0", lw=1.3, label="no-heaping truth")
    axB.plot(XW, norm(naive_c, XW), ":", color="0.55", lw=1.1, label="naive KDE")
    axB.plot(XW, norm(resid_c, XW), "--", color="0.2", lw=1.1, label="AD residue")
    axB.plot(XW, norm(super_c, XW), "-.", color="0.45", lw=1.1, label="superposition")
    axB.set_xlim(110, 320); axB.set_yticks([]); axB.set_xlabel("weight (lb)")
    axB.legend(frameon=False, fontsize=6.5); axB.set_title("(B) measured weight rounded to %d lb" % Dshow, fontsize=8.5)
    fig1.suptitle("Controlled heaping (measured weight, ground-truthed)", fontsize=9, y=1.02)
    fig1.tight_layout(); fig1.savefig("fig_heaping_weight.pdf", bbox_inches="tight")

    # ---------- FIGURE 2: natural cigarettes ----------
    last = (np.round(cig).astype(int) % 10)
    digit_frac = np.array([np.mean(last == d) for d in range(10)])
    f10, f20 = np.mean(np.round(cig) % 10 == 0), np.mean(np.round(cig) % 20 == 0)
    print("\nPART 2 -- cigarettes/day: %.0f%% multiples of 10, %.0f%% multiples of 20; "
          "last-digit 0 share %.0f%% (uniform=10%%)." % (100*f10, 100*f20, 100*digit_frac[0]))
    naive_g = gkde(cig, K.h_silverman(cig), XC); resid_g = K.ad_wiener(cig, XC, strip="residue")

    fig2, (axC, axD) = plt.subplots(1, 2, figsize=(6.6, 2.6))
    axC.hist(cig, bins=np.arange(0, 62, 1), density=True, color="0.85", edgecolor="0.7", lw=0.1, label="reported")
    axC.plot(XC, norm(naive_g, XC), ":", color="0.55", lw=1.1, label="naive KDE")
    axC.plot(XC, norm(resid_g, XC), "--", color="0.2", lw=1.3, label="AD residue")
    axC.set_xlim(0, 60); axC.set_yticks([]); axC.set_xlabel("cigarettes/day")
    axC.legend(frameon=False, fontsize=6.5); axC.set_title("(C) de-heaping the natural comb", fontsize=8.5)
    axD.bar(range(10), digit_frac, color="0.6", edgecolor="0.3", lw=0.4)
    axD.axhline(0.1, color="0.0", ls=":", lw=0.9, label="no preference (10%)")
    axD.set_xticks(range(10)); axD.set_xlabel("last digit of report"); axD.set_ylabel("share of reports")
    axD.legend(frameon=False, fontsize=6.5); axD.set_title("(D) terminal-digit preference", fontsize=8.5)
    fig2.suptitle("Natural heaping (self-reported cigarettes/day, qualitative)", fontsize=9, y=1.02)
    fig2.tight_layout(); fig2.savefig("fig_heaping_cigarettes.pdf", bbox_inches="tight")
    print("\nfigures written: fig_heaping_weight.pdf, fig_heaping_cigarettes.pdf")


if __name__ == "__main__":
    main()
