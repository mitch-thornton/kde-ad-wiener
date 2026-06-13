#!/usr/bin/env python3
"""Extension B: chirps. The raw cyclic estimate measures SWEPT bandwidth; the
metaplectic chirp-rate (estimated by spectral-concentration maximization) de-chirps
to reveal the INSTANTANEOUS bandwidth, and reconstructs the swept bandwidth.
Reproduces fig_chirp.pdf. Seed fixed (see DATA.md)."""
import os, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
SEED = 19; rng = np.random.default_rng(SEED)
FIG = os.path.join(os.path.dirname(__file__), "..", "figures")

def D_of(lam):
    lam = np.asarray(lam, float); s = lam.sum()
    return (s*s)/np.sum(lam*lam) if s > 0 else 0.0
def strip(P):
    return np.clip(P - np.median(P)/np.log(2.0), 0, None)
def psi(P):
    return P.max()/P.sum()

M, fs = 512, 512.0; df = fs/M
fa, fb = 80.0, 220.0                      # instantaneous frequency sweep (Hz)
B_swept_true = fb - fa                     # 140 Hz
n = np.arange(M)
nu = (fa + (fb - fa)*n/(M-1))/fs          # instantaneous freq, cycles/sample
phase = 2*np.pi*np.cumsum(nu)
x = np.exp(1j*phase)
snr_db = 20.0; sl = 10**(snr_db/10.0)
nu_n = (np.sum(np.abs(x)**2)/M)/sl
x = x + np.sqrt(nu_n/2)*(rng.standard_normal(M)+1j*rng.standard_normal(M))

# raw cyclic estimate -> swept bandwidth
P_raw = np.abs(np.fft.fft(x))**2
B_raw = D_of(strip(P_raw))*df

# metaplectic chirp-rate estimate: maximize spectral concentration over de-chirp grid
c_true = (fb - fa)/(fs*(M-1))             # quadratic-phase coefficient (per sample^2), phase ~ pi*c*n^2
cs = np.linspace(0, 2*c_true, 601)
psis = np.array([psi(np.abs(np.fft.fft(x*np.exp(-1j*np.pi*c*n**2)))**2) for c in cs])
c_hat = cs[int(np.argmax(psis))]
x_dechirped = x*np.exp(-1j*np.pi*c_hat*n**2)
P_dech = np.abs(np.fft.fft(x_dechirped))**2
B_inst = D_of(strip(P_dech))*df
B_swept_est = c_hat*(M-1)*fs              # reconstruct swept bandwidth from chirp rate

print("Chirp: sweep %.0f->%.0f Hz, true swept bandwidth = %.1f Hz" % (fa, fb, B_swept_true))
print("  raw cyclic D*df (swept)      = %.1f Hz" % B_raw)
print("  chirp-rate estimate c_hat    = %.3e  (true %.3e)" % (c_hat, c_true))
print("  swept bandwidth from c_hat   = %.1f Hz" % B_swept_est)
print("  de-chirped D*df (instant.)   = %.1f Hz" % B_inst)
print("  psi raw=%.4f  psi dechirped=%.4f" % (psi(P_raw), psi(P_dech)))

plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.8, "figure.dpi": 150})
fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.9))
fr = np.arange(M)*df
a.plot(fr, P_raw/P_raw.max(), '-', c='0.5', lw=0.9, label="chirp (raw)")
a.plot(fr, P_dech/P_dech.max(), '-', c='0.0', lw=1.0, label="de-chirped")
a.axvspan(fa, fb, color='0.85', alpha=0.6, label="true swept band")
a.set_xlabel("frequency (Hz)"); a.set_ylabel("normalized $|X[k]|^2$")
a.set_title("(a) chirp vs de-chirped spectrum", fontsize=9); a.legend(fontsize=7, frameon=False, loc="lower center"); a.set_xlim(0, 300)
b.plot(cs/c_true, psis, '-', c='0.0', lw=1.0)
b.axvline(1.0, ls='--', c='0.5', lw=0.9, label="true chirp rate")
b.plot(c_hat/c_true, psis.max(), 'ko', ms=5, label="estimate")
b.set_xlabel("candidate chirp rate $/\\,c_{\\mathrm{true}}$"); b.set_ylabel("spectral concentration $\\psi$")
b.set_title("(b) chirp-rate estimation by $\\psi$-max", fontsize=9); b.legend(fontsize=7.5, frameon=False)
plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig_chirp.pdf")); plt.close()
print("figure written: fig_chirp.pdf")
