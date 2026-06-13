#!/usr/bin/env python3
"""AD bandwidth estimation - core experiments (mechanism, SNR sweep, multi-snapshot).
Reproduces fig_mechanism.pdf and fig_bw_sweep.pdf. Seed fixed (see DATA.md)."""
import os, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
SEED = 7; rng = np.random.default_rng(SEED)
FIG = os.path.join(os.path.dirname(__file__), "..", "figures")

def D_of(lam):
    lam = np.asarray(lam, float); s = lam.sum()
    return (s*s)/np.sum(lam*lam) if s > 0 else 0.0

# 1. MECHANISM: eigenvalues of R_G equal the periodogram
def cyclic_group_avg_cov(x):
    M = len(x); R = np.zeros((M, M), complex)
    for g in range(M):
        xs = np.roll(x, g); R += np.outer(xs, np.conj(xs))
    return R / M

Mc, b0c, Kc = 64, 22, 18
Xc = np.zeros(Mc, complex); Xc[b0c:b0c+Kc] = rng.standard_normal(Kc) + 1j*rng.standard_normal(Kc)
xc = np.fft.ifft(Xc) * np.sqrt(Mc)
R = cyclic_group_avg_cov(xc)
ev = np.sort(np.linalg.eigvalsh(R))[::-1]
pg = np.sort(np.abs(np.fft.fft(xc))**2 / Mc)[::-1]
scale = ev.sum()/pg.sum()
corr = float(np.corrcoef(ev, pg*scale)[0, 1])
maxrel = float(np.max(np.abs(ev - pg*scale))/ev.max())
print("MECHANISM (M=%d): corr=%.10f  max-rel-diff=%.2e  D_eig=%.3f D_pgram=%.3f"
      % (Mc, corr, maxrel, D_of(ev), D_of(pg)))

# 2. SNR SWEEP and MULTI-SNAPSHOT
M, fs = 512, 512.0; df = fs/M; K, b0 = 64, 180
band = np.arange(b0, b0+K); B_support = K*df; T = 400
snrs = np.array([-10,-5,0,5,10,15,20,25,30])

def periodogram(kind, snr_lin):
    X = np.zeros(M, complex)
    if kind == "flat":
        X[band] = np.exp(1j*rng.uniform(0, 2*np.pi, K))
    else:
        X[band] = (rng.standard_normal(K) + 1j*rng.standard_normal(K))/np.sqrt(2)
    nu = K/(M*snr_lin)
    N = np.sqrt(nu/2)*(rng.standard_normal(M) + 1j*rng.standard_normal(M))
    return np.abs(X + N)**2

def strip(P):
    return np.clip(P - np.median(P)/np.log(2.0), 0, None)

def sweep(kind):
    raw, stp = [], []
    for s in snrs:
        sl = 10**(s/10.0); dr, ds = [], []
        for _ in range(T):
            P = periodogram(kind, sl); dr.append(D_of(P)*df); ds.append(D_of(strip(P))*df)
        raw.append(np.mean(dr)); stp.append(np.mean(ds))
    return np.array(raw), np.array(stp)

flat_raw, flat_stp = sweep("flat"); rand_raw, rand_stp = sweep("random")
Ls = [1,2,4,8,16,32]; rec = {}
for snr in (10, 20):
    sl = 10**(snr/10.0); row = []
    for L in Ls:
        v = [D_of(strip(np.mean([periodogram("random", sl) for _ in range(L)], axis=0)))*df for _ in range(T)]
        row.append(np.mean(v))
    rec[snr] = row

print("\nB_eff (Hz)  true support=%.0f  random-effective~K/2=%.0f" % (B_support, B_support/2))
print(" SNR  flatRaw flatStp  randRaw randStp")
for i, s in enumerate(snrs):
    print(" %3d  %6.1f %6.1f   %6.1f %6.1f" % (s, flat_raw[i], flat_stp[i], rand_raw[i], rand_stp[i]))
print("multisnapshot (random, stripped):")
for snr in (10, 20):
    print("  %ddB: " % snr + " ".join("L%d=%.1f" % (L, v) for L, v in zip(Ls, rec[snr])))

# FIGURES (print-friendly, grayscale-safe)
plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.8, "figure.dpi": 150})
fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.7))
a.plot(np.arange(Mc)+1, ev, 'k-', lw=1.0, label="eigenvalues of $R_G$")
a.plot(np.arange(Mc)+1, pg*scale, 'o', ms=3, mfc='none', mec='0.45', label="periodogram $|X[k]|^2$")
a.set_xlabel("index (sorted)"); a.set_ylabel("value")
a.set_title("(a) $R_G$ spectrum $=$ periodogram", fontsize=9)
a.legend(fontsize=7, frameon=False); a.text(0.42, 0.6, "corr $=1.0000000$", transform=a.transAxes, fontsize=7.5)
Xex = np.zeros(M, complex); Xex[band] = np.exp(1j*rng.uniform(0,2*np.pi,K)); Pex = np.abs(Xex)**2 + 0.0
Pex = np.abs(Xex + np.sqrt((K/(M*100))/2)*(rng.standard_normal(M)+1j*rng.standard_normal(M)))**2
freqs = np.arange(M)*df; Deff = D_of(strip(Pex)); c0 = (b0+K/2)*df
b.plot(freqs, Pex, '-', color='0.25', lw=0.8)
b.axvspan(c0-Deff*df/2, c0+Deff*df/2, color='0.80', alpha=0.7, label="$B_{\\mathrm{eff}}=D f_s/M$")
b.set_xlabel("frequency (Hz)"); b.set_ylabel("$|X[k]|^2$")
b.set_title("(b) participation ratio as width", fontsize=9)
b.legend(fontsize=7.5, frameon=False); b.set_xlim(150, 270)
plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig_mechanism.pdf")); plt.close()

fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.9))
a.axhline(B_support, ls='--', c='0.0', lw=0.9, label="true support (64 Hz)")
a.axhline(B_support/2, ls=':', c='0.45', lw=0.9, label="random effective $\\approx K/2$")
a.plot(snrs, flat_raw, '-^', c='0.55', ms=3.5, label="flat, raw")
a.plot(snrs, flat_stp, '-^', c='0.0', ms=3.5, label="flat, stripped")
a.plot(snrs, rand_raw, '--s', c='0.55', ms=3.5, label="random, raw")
a.plot(snrs, rand_stp, '--s', c='0.0', ms=3.5, mfc='white', label="random, stripped")
a.set_ylim(0, 170); a.set_xlabel("input SNR (dB)"); a.set_ylabel("$B_{\\mathrm{eff}}$ (Hz)")
a.set_title("(a) single-observation estimate vs SNR", fontsize=9); a.legend(fontsize=6.6, frameon=False)
for snr, mk in zip((10, 20), ('o', 'D')):
    b.plot(Ls, rec[snr], '-'+mk, c='0.0' if snr == 20 else '0.5', ms=4, label="SNR %d dB" % snr)
b.axhline(B_support, ls='--', c='0.0', lw=0.9, label="true support")
b.set_xscale('log', base=2); b.set_xticks(Ls); b.set_xticklabels(Ls)
b.set_ylim(28, 68); b.set_xlabel("observations averaged $L$"); b.set_ylabel("$B_{\\mathrm{eff}}$ (Hz)")
b.set_title("(b) multi-observation recovery (random)", fontsize=9); b.legend(fontsize=7, frameon=False)
plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig_bw_sweep.pdf")); plt.close()
print("\nfigures written: fig_mechanism.pdf, fig_bw_sweep.pdf")
