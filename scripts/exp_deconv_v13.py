"""exp_deconv_v13.py -- density deconvolution under known measurement error.

We observe Y = X + e, where the measurement error e has a known distribution; the goal is the
density of X, not of Y. The observed density is the convolution f_Y = f_X * f_e, so an ordinary
kernel estimator faithfully recovers the blurred f_Y and is biased for f_X. In the characteristic
function domain phi_Y = phi_X * phi_e (a product), so deconvolution is division by the known noise
characteristic function phi_e, carried out in the same ECF domain in which the selector and the
AD-Wiener filter already operate.

Dividing by phi_e amplifies the sampling-noise floor: the flat 1/n floor on the empirical
characteristic function becomes the frequency-shaped floor (1/n)/|phi_e(t)|^2, which blows up where
phi_e decays. The AD-Wiener taper carries over unchanged except for this amplified floor, and the
cutoff is set where the deconvolved power meets the floor -- the residual-meets-floor stopping rule,
with no manual bandwidth. The error here is Laplace (ordinary smooth, phi_e(t) = 1/(1+b^2 t^2)); the
true density is bimodal so that the blur visibly merges the modes that deconvolution then recovers.

Baselines: an ordinary Gaussian KDE on Y (which estimates the blurred density), and a standard
deconvoluting-kernel estimator at its oracle bandwidth. Reproducible with numpy only.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
B = 0.7                                   # Laplace scale (known)
XG = np.linspace(-6, 6, 512)
T, L = 30.0, 2048
TG = np.linspace(-T, T, L)
DT = TG[1] - TG[0]
C0 = int(np.argmin(np.abs(TG)))
PE = 1.0 / (1.0 + (B * TG) ** 2)          # phi_e(t) for Laplace(0,B), known
INV = np.exp(-1j * np.outer(XG, TG)) * DT / (2 * np.pi)   # inverse-transform matrix (precomputed)


def f_X(x):
    g = lambda m: np.exp(-0.5 * ((x - m) / 0.5) ** 2) / (0.5 * np.sqrt(2 * np.pi))
    return 0.5 * g(-1.5) + 0.5 * g(1.5)

def sample_X(n, rng):
    s = rng.random(n) < 0.5
    return np.where(s, rng.normal(-1.5, 0.5, n), rng.normal(1.5, 0.5, n))

TRUE = f_X(XG); TRUE /= np.trapezoid(TRUE, XG)

def ise(f):
    f = np.clip(f, 0, None); f = f / np.trapezoid(f, XG)
    return np.trapezoid((f - TRUE) ** 2, XG)

def ecf(Y):
    return np.exp(1j * np.outer(TG, Y)).mean(axis=1)

def invert(phi):
    return (INV @ phi).real

def smooth(v, w=41):
    return np.convolve(v, np.ones(w) / w, mode="same")

def naive_kde(Y):
    h = 1.06 * np.std(Y) * len(Y) ** -0.2
    return np.mean(np.exp(-0.5 * ((XG[:, None] - Y[None, :]) / h) ** 2) / (h * np.sqrt(2 * np.pi)), axis=1)

def deconv_kde_fixed(pY, h):
    ker = np.where(np.abs(h * TG) < 1, (1 - (h * TG) ** 2) ** 3, 0.0)
    return invert(pY / PE * ker)

def first_sustained_cut(ratio, run=20):
    """First frequency each side where the deconvolved power stays below the floor: the stop."""
    hi = L
    for k in range(C0, L):
        if np.all(ratio[k:min(k + run, L)] < 1.0): hi = k; break
    lo = 0
    for k in range(C0, -1, -1):
        if np.all(ratio[max(k - run, 0):k + 1] < 1.0): lo = k; break
    return lo, hi

def ad_deconv(pY, n, return_diag=False):
    pX = pY / PE
    floor = (1.0 / n) / PE ** 2                       # amplified white floor (computed residual)
    P = smooth(np.abs(pX) ** 2)                       # smoothed deconvolved power (empirical residual)
    ratio = P / np.maximum(floor, 1e-30)
    W = np.clip(P - floor, 0, None) / np.maximum(P, 1e-30)
    lo, hi = first_sustained_cut(ratio)
    win = smooth(np.where((np.arange(L) >= lo) & (np.arange(L) < hi), 1.0, 0.0), 21)
    fhat = invert(pX * W * win)
    if return_diag:
        return fhat, P, floor, TG[hi]
    return fhat


def run(ns=(250, 500, 1000, 2000, 4000), reps=20, seed=2024):
    rng = np.random.default_rng(seed)
    rows = []
    for n in ns:
        iN, iD, iA = [], [], []
        for _ in range(reps):
            X = sample_X(n, rng); Y = X + rng.laplace(0, B, n)
            pY = ecf(Y)
            iN.append(ise(naive_kde(Y)))
            iD.append(min(ise(deconv_kde_fixed(pY, h)) for h in np.linspace(0.15, 1.2, 18)))
            iA.append(ise(ad_deconv(pY, n)))
        rows.append((n, np.mean(iN), np.mean(iD), np.mean(iA)))
    return rows


if __name__ == "__main__":
    rows = run()
    print("=== Deconvolution under known Laplace error (b=%.1f) ===" % B)
    print("%-6s %10s %14s %10s" % ("n", "naive(on Y)", "deconv-KDE(orc)", "AD-deconv"))
    for n, a, d, ad in rows:
        print("%-6d %10.2f %14.2f %10.2f" % (n, 1e3 * a, 1e3 * d, 1e3 * ad))

    # figure: recovery (left) + residual-vs-floor diagnostic (right)
    rng = np.random.default_rng(1)
    n = 2000; X = sample_X(n, rng); Y = X + rng.laplace(0, B, n)
    pY = ecf(Y)
    fad, P, floor, tstar = ad_deconv(pY, n, return_diag=True)
    fad = np.clip(fad, 0, None); fad /= np.trapezoid(fad, XG)
    nk = naive_kde(Y); nk /= np.trapezoid(nk, XG)

    plt.rcParams.update({"font.size": 8.5, "axes.linewidth": 0.8})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.5))
    ax1.plot(XG, TRUE, color="0.0", lw=1.4, label="true $f_X$")
    ax1.plot(XG, nk, ":", color="0.55", lw=1.2, label="naive KDE on $Y$")
    ax1.plot(XG, fad, "--", color="0.2", lw=1.2, label="AD deconvolution")
    ax1.set_xlim(-4, 4); ax1.set_xlabel("$x$"); ax1.set_ylabel("density")
    ax1.legend(frameon=False, fontsize=6.5, loc="lower left", bbox_to_anchor=(0.0,1.02), ncol=3)

    pos = TG[C0:]
    ax2.semilogy(pos, P[C0:], color="0.15", lw=1.2, label="deconvolved power")
    ax2.semilogy(pos, floor[C0:], "--", color="0.55", lw=1.1, label="computed floor")
    ax2.axvline(tstar, color="0.7", lw=0.9, ls=":")
    ax2.text(tstar + 0.3, P[C0] * 0.05, "$t^\\ast$", color="0.4", fontsize=8)
    ax2.set_xlim(0, 12); ax2.set_ylim(1e-6, 5)
    ax2.set_xlabel("frequency $t$"); ax2.set_ylabel("power (log)")
    ax2.legend(frameon=False, fontsize=7, loc="center right")
    fig.tight_layout()
    out = os.path.join(FIG, "fig_deconv.pdf")
    fig.savefig(out); print("figure written:", os.path.basename(out))
