#!/usr/bin/env python3
"""Extension A: AD effective bandwidth B_eff vs conventional 99%-power occupied
bandwidth B_99. Reproduces fig_b99_compare.pdf. Seed fixed (see DATA.md)."""
import os, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
SEED = 11; rng = np.random.default_rng(SEED)
FIG = os.path.join(os.path.dirname(__file__), "..", "figures")

def D_of(lam):
    lam = np.asarray(lam, float); s = lam.sum()
    return (s*s)/np.sum(lam*lam) if s > 0 else 0.0
def strip(P):
    return np.clip(P - np.median(P)/np.log(2.0), 0, None)
def B99(P, df, frac=0.99):
    # ITU-style contiguous occupied bandwidth: trim (1-frac)/2 power from each end
    c = np.cumsum(P); tot = c[-1]; lo = (1-frac)/2*tot; hi = (1+frac)/2*tot
    i = int(np.searchsorted(c, lo)); j = int(np.searchsorted(c, hi))
    return (j - i)*df

M, fs = 512, 512.0; df = fs/M; K, b0 = 64, 180
band = np.arange(b0, b0+K); B_support = K*df; T = 400
snrs = np.array([0,5,10,15,20,25,30])

def periodogram(kind, snr_lin):
    X = np.zeros(M, complex)
    X[band] = np.exp(1j*rng.uniform(0,2*np.pi,K)) if kind=="flat" else \
              (rng.standard_normal(K)+1j*rng.standard_normal(K))/np.sqrt(2)
    nu = K/(M*snr_lin); N = np.sqrt(nu/2)*(rng.standard_normal(M)+1j*rng.standard_normal(M))
    return np.abs(X+N)**2

def run(kind):
    be, b99 = [], []
    for s in snrs:
        sl = 10**(s/10.0); e, n = [], []
        for _ in range(T):
            P = periodogram(kind, sl); e.append(D_of(strip(P))*df); n.append(B99(P, df))
        be.append(np.mean(e)); b99.append(np.mean(n))
    return np.array(be), np.array(b99)

fe, f99 = run("flat"); re_, r99 = run("random")
print("True support=%.0f Hz  (random effective ~ %.0f Hz)" % (B_support, B_support/2))
print(" SNR | flat: B_eff  B_99 | random: B_eff  B_99")
for i, s in enumerate(snrs):
    print(" %3d |       %5.1f %5.1f |        %5.1f %5.1f" % (s, fe[i], f99[i], re_[i], r99[i]))

plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.8, "figure.dpi": 150})
fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
for ax, (be, b99v, ttl) in zip((a, b), ((fe, f99, "(a) flat band"), (re_, r99, "(b) random band"))):
    ax.axhline(B_support, ls='--', c='0.0', lw=0.9, label="true support (64 Hz)")
    ax.axhline(B_support/2, ls=':', c='0.5', lw=0.9, label="$K/2$ (32 Hz)")
    ax.plot(snrs, be, '-o', c='0.0', ms=4, label="$B_{\\mathrm{eff}}$ (AD, stripped)")
    ax.plot(snrs, b99v, '--s', c='0.5', ms=4, mfc='white', label="$B_{99}$ (raw periodogram)")
    ax.set_xlabel("input SNR (dB)"); ax.set_title(ttl, fontsize=9)
    ax.set_yscale('log'); ax.set_ylim(20, 600)
a.set_ylabel("bandwidth (Hz)"); a.legend(fontsize=6.6, frameon=False, loc="upper right")
plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig_b99_compare.pdf")); plt.close()
print("figure written: fig_b99_compare.pdf")
