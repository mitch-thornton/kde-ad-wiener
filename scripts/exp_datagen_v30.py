#!/usr/bin/env python3
"""exp_datagen.py -- a fidelity-measurement procedure for synthetic data generators.

USE-CASE. A practitioner specifies a target nonparametric density f and a sample count N, and a
generator emits N samples meant to follow f. Kernel density estimation then recovers the *realized*
density from those samples and measures its departure from f. This is a validation of the
measurement procedure, not a ranking of generators: the generator here is a parameterized reference
whose departure from the target is known by construction, so the question is whether the KDE-based
fidelity estimate tracks the true departure, and which estimator recovers it with least bias.

GENERATOR. Exact draws from an arbitrary target are produced by inverse-CDF sampling of the supplied
pdf grid. A controlled defect is introduced by Huber epsilon-contamination, the realized law being
    G_eps = (1 - eps) f + eps c,
so the departure from f is eps*TV(f,c) and eps is the literal error knob. An optional jitter convolves
the draws with a small Gaussian. Bounds: N in [2^9, 2^18].

ESTIMATOR SELECTION. The selected estimate defaults to AD-Wiener. '--method auto' routes by the global
coherent effective dimension D_2 (low -> mixture, high -> AD-Wiener), breaking ties toward AD-Wiener.
'--method {wiener,gmm}' forces one estimator on the whole support. '--route "lo:hi:method,..."' applies
a user-specified method on each named interval (a manual partition); boundary continuity between
pieces is NOT enforced here, which is exactly the matching problem the automatic mixed-mode estimator
must solve, and is left to that construction.

DIFFERENTIABILITY. The mixture estimate is a finite sum of Gaussians and is C-infinity everywhere with
closed-form gradients. The AD-Wiener estimate is band-limited (the Wiener taper zeroes frequencies
above the cutoff), so its un-clipped reconstruction is a trigonometric polynomial, also C-infinity;
the non-negativity clip introduces kinks only at the finite set of zero-crossings. Both therefore
support gradient-based use, e.g. digital-twin construction. '--diff-check' reports the cutoff and kinks.

Run from the bundle's scripts/ dir:  python3 exp_datagen_v30.py
"""
import os, sys, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ad_kde_v30 as K
import adkde_plugins as P

# Resolve the bundle figures/ directory relative to this script so build.sh regeneration
# updates the figures the manuscript actually includes, regardless of the working directory.
_FIGDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(_FIGDIR, exist_ok=True)
def _fig(name):
    return os.path.join(_FIGDIR, name)

SQRT2PI = np.sqrt(2 * np.pi)
def _n(x, m, s): return np.exp(-0.5 * ((x - m) / s) ** 2) / (s * SQRT2PI)

# ---------------- target library (name -> (pdf, support)) ----------------
_CLAW = [2.5 + (l - 2) * 0.6 for l in range(5)]
def _halfhalf(x):
    left = 0.5 * _n(x, -3.2, 0.5) + 0.5 * _n(x, -1.4, 0.5)              # smooth bimodal (mixture-optimal)
    right = 0.5 * _n(x, 2.5, 1.0) + sum(0.1 * _n(x, m, 0.08) for m in _CLAW)  # claw (AD-Wiener-optimal)
    return 0.5 * left + 0.5 * right
def _bimodal(x):  return 0.5 * _n(x, -1.5, 0.5) + 0.5 * _n(x, 1.5, 0.5)
def _kurtotic(x): return (2/3) * _n(x, 0, 1.0) + (1/3) * _n(x, 0, 0.1)
def _claw(x):     return 0.5 * _n(x, 0, 1.0) + sum(0.1 * _n(x, m, 0.1) for m in (-1.0, -0.5, 0.0, 0.5, 1.0))
def _alternating(x):
    # alternating smooth bumps and claw combs; the combs exceed what a bounded-order mixture can fit,
    # so the optimal partition genuinely alternates GMM (smooth) and AD-Wiener (combs)
    smooth = 0.32 * _n(x, -4.0, 0.6) + 0.32 * _n(x, 1.0, 0.7)
    comb1 = sum(0.045 * _n(x, m, 0.06) for m in (-1.8, -1.4, -1.0, -0.6))
    comb2 = sum(0.045 * _n(x, m, 0.06) for m in (3.4, 3.8, 4.2, 4.6))
    return smooth + comb1 + comb2
TARGETS = {"halfhalf": (_halfhalf, (-6.0, 6.0)),
           "bimodal":  (_bimodal,  (-4.0, 4.0)),
           "kurtotic": (_kurtotic, (-4.0, 4.0)),
           "claw":     (_claw,     (-4.0, 4.0)),
           "alternating": (_alternating, (-6.5, 6.5))}

# ---------------- generator ----------------
def inverse_cdf_sampler(pdf_grid, xg, n, rng):
    f = np.clip(pdf_grid, 0, None)
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(xg))])
    cdf /= cdf[-1]
    return np.interp(rng.random(n), cdf, xg)

def generate(target, N, eps=0.0, eps_j=0.0, contaminant=None, grid=4096, rng=None):
    rng = rng or np.random.default_rng(0)
    pdf, (lo, hi) = TARGETS[target]
    xg = np.linspace(lo, hi, grid)
    f = pdf(xg); f /= np.trapezoid(f, xg)
    if contaminant is None:                                   # default defect: a narrow spurious mode
        cm = lo + 0.75 * (hi - lo); c = _n(xg, cm, 0.04 * (hi - lo))
    else:
        c = contaminant(xg)
    c /= np.trapezoid(c, xg)
    g = (1 - eps) * f + eps * c
    d = inverse_cdf_sampler(g, xg, N, rng)
    if eps_j > 0:
        d = d + rng.normal(0, eps_j, size=N)
    return d, xg, f, g

# ---------------- estimators ----------------
def _nrm(f, xg):
    f = np.clip(f, 0, None); s = np.trapezoid(f, xg); return f / s if s > 0 else f
def est_naive(d, xg):  return _nrm(K.fft_kde(d, K.h_silverman(d), xg), xg)
def est_wiener(d, xg): return _nrm(K.ad_wiener(d, xg, strip="residue"), xg)
def est_gmm(d, xg, kmax=6): return _nrm(P.fit_mixture(d, kmax=kmax)(xg), xg)

def select_estimate(d, xg, method="wiener", kmax=6, d2_lo=2.0):
    """Global selected estimate. Default AD-Wiener. 'auto' routes by D_2, tie-break -> AD-Wiener."""
    if method == "gmm":    return est_gmm(d, xg, kmax)
    if method == "wiener": return est_wiener(d, xg)
    D2 = K.coherent_effdim(d, xg)                # auto
    return est_gmm(d, xg, kmax) if D2 < d2_lo else est_wiener(d, xg)

def routed_estimate(d, xg, route, kmax=6):
    """Manual partition: route = [(lo,hi,'gmm'|'wiener'), ...]. Pieces are NOT boundary-matched;
    the resulting kinks/jumps are the matching problem the automatic mixed-mode estimator solves."""
    out = est_wiener(d, xg).copy()               # AD-Wiener default outside any named interval
    full = {"gmm": est_gmm(d, xg, kmax), "wiener": est_wiener(d, xg)}
    for lo, hi, m in route:
        sel = (xg >= lo) & (xg < hi); out[sel] = full[m][sel]
    return _nrm(out, xg)

# ---------------- scoring ----------------
def tv(p, q, xg):                                 # total variation distance, 0.5 * integral|p-q|
    return 0.5 * float(np.trapezoid(np.abs(p - q), xg))
def ise(fh, ft, xg, lo=None, hi=None):
    m = np.ones_like(xg, bool)
    if lo is not None: m &= (xg >= lo)
    if hi is not None: m &= (xg < hi)
    return float(np.trapezoid((fh[m] - ft[m]) ** 2, xg[m]))

# ---------------- boundary-matched mixed-mode (locally partitioned) estimator ----------------
def partition_weight(xg, x0, delta):
    """Smooth partition of unity: 1 left of x0, 0 right of x0, C-infinity transition of half-width delta."""
    return 0.5 * (1.0 - np.tanh((xg - x0) / max(delta, 1e-9)))

def mixed_mode_estimate(d, xg, x0, m_left, m_right, delta=0.4, kmax=6):
    """Locally partitioned estimate: method m_left for x<x0, m_right for x>x0, joined by a smooth
    partition of unity so the result is continuous and differentiable at the boundary (boundary
    matching). Each method is fit on the full sample; the weight selects which governs each region."""
    full = {"gmm": est_gmm(d, xg, kmax), "wiener": est_wiener(d, xg)}
    w = partition_weight(xg, x0, delta)
    f = w * full[m_left] + (1.0 - w) * full[m_right]
    return _nrm(f, xg)

# ---------------- divergences (KL: Kullback-Leibler 1951; JS: Lin 1991) ----------------
def kl_div(p, q, xg, floor=1e-12):
    """KL(p || q) = integral p log(p/q). p=input (ideal), q=KDE output."""
    p = _nrm(p, xg); q = _nrm(q, xg)
    pe = np.clip(p, floor, None); qe = np.clip(q, floor, None)
    return float(np.trapezoid(pe * np.log(pe / qe), xg))

def js_div(p, q, xg, floor=1e-12):
    """Jensen-Shannon divergence (symmetric), base-2, in [0,1]."""
    p = _nrm(p, xg); q = _nrm(q, xg); m = 0.5 * (p + q)
    return float(0.5 * kl_div(p, m, xg, floor) + 0.5 * kl_div(q, m, xg, floor)) / np.log(2.0)

# ---------------- experiment 3: four routing combinations on the combined target ----------------
def four_combo(N=8000, x0=0.0, delta=0.4, kmax=6, seed=0):
    """All four method combinations on the concatenated smooth-mixture + claw target, comparing the
    specified (ideal) input density to the KDE-smoothed output by KL and JS divergence."""
    rng = np.random.default_rng(seed)
    d, xg, f, g = generate("halfhalf", N, eps=0.0, rng=rng)   # exact draws from the ideal target
    ft = _nrm(f, xg)
    combos = [("(1) GMM | GMM", "gmm", "gmm"),
              ("(2) AD-Wiener | AD-Wiener", "wiener", "wiener"),
              ("(3) AD-Wiener | GMM", "wiener", "gmm"),
              ("(4) GMM | AD-Wiener", "gmm", "wiener")]
    rows = []
    print("=== four routing combinations on the combined (smooth-mixture | claw) target ===")
    print("%-28s %10s %10s" % ("smooth-half | claw-half method", "KL(in||out)", "JS(in,out)"))
    ests = {}
    for label, ml, mr in combos:
        fh = mixed_mode_estimate(d, xg, x0, ml, mr, delta, kmax)
        ests[label] = fh
        kl = kl_div(ft, fh, xg); js = js_div(ft, fh, xg)
        rows.append((label, kl, js)); print("%-28s %10.4f %10.4f" % (label, kl, js))
    best = min(rows, key=lambda r: r[1]); worst = max(rows, key=lambda r: r[1])
    print("best (lowest KL): %s ; worst: %s" % (best[0], worst[0]))
    # 4-panel figure
    plt.rcParams.update({"font.size": 8.5, "axes.linewidth": 0.8})
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.6))
    for ax, (label, ml, mr) in zip(axes.ravel(), combos):
        ax.fill_between(xg, ft, color="0.88", lw=0)
        ax.plot(xg, ests[label], "-", color="0.0", lw=1.0)
        ax.axvline(x0, color="0.6", ls=":", lw=0.7)
        kl = dict((r[0], r[1]) for r in rows)[label]
        ax.set_title("%s   (KL %.3f)" % (label, kl), fontsize=8.0)
        ax.set_yticks([]); ax.set_xlabel("$x$")
    fig.tight_layout(); fig.savefig(_fig("fig_mixedmode.pdf")); print("\nfigure written: fig_mixedmode.pdf")
    return rows

# ---------------- optional per-subsection uniformity flattening ----------------
def local_uniform_gate(d, a, b, alpha=0.01, M=256):
    """Uniformity gate restricted to the data in [a,b], mapped to a local periodic [0,1)."""
    sub = d[(d >= a) & (d < b)]
    if sub.size < 30: return False
    xs = np.linspace(0.0, 1.0, M); ok, _, _ = K.uniform_gate((sub - a) / (b - a), xs, alpha)
    return ok

def flatten_uniform_segments(d, xg, f, segments, alpha=0.01, delta=0.4):
    """For each AD-Wiener segment whose local data is consistent with uniform, replace that stretch by
    its mass-preserving constant (a perfect-uniform subsection), joined by the partition of unity so
    the result stays differentiable. Mass on each flattened interval is preserved, so f stays valid."""
    out = f.copy()
    for a, b in segments:
        if not local_uniform_gate(d, a, b, alpha): continue
        m = (xg >= a) & (xg < b)
        if m.sum() < 2: continue
        mass = np.trapezoid(np.where(m, out, 0.0), xg); const = mass / (b - a)
        bump = np.clip(partition_weight(xg, b, delta) - partition_weight(xg, a, delta), 0, 1)
        out = (1 - bump) * out + bump * const
    return _nrm(out, xg)

# ---------------- superposition: GMM smooth base + AD-Wiener sharp residual ----------------
def _gmm_params(d, kmax=8):
    """BIC-selected Gaussian-mixture parameters (pi, mu, var)."""
    best = None
    for k in range(1, kmax + 1):
        params, bic = P._gmm_em(d, k)
        if best is None or bic < best[1]: best = (params, bic)
    return best[0]

def _wiener_filter_mass(rm, n, xg):
    """Apply the AD-Wiener filter to a binned mass signal rm (e.g. a residual). Keeps coherent structure
    above the 1/n floor, discards sub-floor noise, and returns a mass signal."""
    from scipy.ndimage import uniform_filter1d
    Phat = np.fft.rfft(rm); ecf2 = np.abs(Phat) ** 2
    Ssm = uniform_filter1d(ecf2, max(3, len(ecf2) // 128)); nu = 1.0 / n
    below = np.where(Ssm[1:] < nu)[0]; kc = (below[0] + 1) if len(below) else len(ecf2) - 1
    Shat = np.clip(Ssm - nu, 0, None); Shat[kc:] = 0.0; Wf = Shat / (Shat + nu)
    return np.fft.irfft(Wf * Phat, n=len(xg))

def superpose(d, xg, kmax=8, cwidth=1.5):
    """Additive decomposition with no spatial partition. A Gaussian mixture is fit and split by component
    width at a smoothness scale cwidth*h_silverman: the broad components form the smooth base, the narrow
    components are deferred to the residual. The AD-Wiener filter on the residual mass recovers the sharp
    structure the base omits, and the two are added back. Handles superimposed smooth-plus-sharp targets
    that a spatial boundary cannot separate."""
    pi, mu, var = _gmm_params(d, kmax); n = len(d); p, dx = K._binned(d, xg)
    theta = cwidth * K.h_silverman(d); broad = np.sqrt(var) >= theta
    if broad.any():
        base = np.sum([pi[j] * P._gauss(xg, mu[j], var[j]) for j in range(len(pi)) if broad[j]], axis=0)
    else:
        base = np.zeros_like(xg)
    base = np.asarray(base, dtype=float)
    sharp = _wiener_filter_mass(p - base * dx, n, xg) / dx                # AD-Wiener on the residual
    return _nrm(np.clip(base + sharp, 0, None), xg), base, np.clip(sharp, None, None)

def superposition_battery(reps=5, N=8000):
    """KL of GMM, AD-Wiener, and superposition on the five targets (mean over reps seeds)."""
    print("=== superposition battery: KL to target, mean over %d seeds ===" % reps)
    print("%-12s %9s %9s %9s" % ("target", "GMM", "AD-Wiener", "superpose"))
    rows = {}
    for name in ['bimodal', 'kurtotic', 'claw', 'halfhalf', 'alternating']:
        aG = []; aW = []; aS = []
        for seed in range(reps):
            d, xg, f, g = generate(name, N, eps=0.0, rng=np.random.default_rng(seed)); ft = _nrm(f, xg)
            aG.append(kl_div(ft, est_gmm(d, xg), xg)); aW.append(kl_div(ft, est_wiener(d, xg), xg))
            fS, _, _ = superpose(d, xg); aS.append(kl_div(ft, fS, xg))
        rows[name] = (np.mean(aG), np.mean(aW), np.mean(aS))
        print("%-12s %9.4f %9.4f %9.4f" % (name, *rows[name]))
    return rows

def superpose_showcase(seed=7, N=8000, path=None):
    """Three-panel claw showcase: GMM smooth base | AD-Wiener sharp residual | their sum vs the target."""
    d, xg, f, g = generate('claw', N, eps=0.0, rng=np.random.default_rng(seed)); ft = _nrm(f, xg)
    fS, base, sharp = superpose(d, xg)
    fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.3))
    ax[0].plot(xg, base, color='0.0', lw=1.0); ax[0].set_title("(A) GMM smooth base", fontsize=9)
    ax[1].plot(xg, sharp, color='0.0', lw=1.0); ax[1].axhline(0, color='0.7', lw=0.5)
    ax[1].set_title("(B) AD-Wiener sharp residual", fontsize=9)
    ax[2].plot(xg, ft, color='0.6', lw=2.2, label="target"); ax[2].plot(xg, fS, color='0.0', lw=1.0, label="superpose")
    ax[2].set_title("(C) base + residual", fontsize=9); ax[2].legend(fontsize=7, frameon=False)
    for a in ax: a.set_yticks([]); a.tick_params(labelsize=7)
    fig.tight_layout()
    out = path or _fig("fig_superpose.pdf"); fig.savefig(out, bbox_inches='tight'); plt.close(fig)
    return out

# ---------------- join robustness (lemma check) ----------------
def join_robustness(N=8000, seed=0):
    """Place the boundary at adversarial locations and widths; verify the convex-combination blend
    never exceeds max(f_L,f_R) or drops below min (no spurious spike or dropout) and stays non-negative."""
    rng = np.random.default_rng(seed)
    d, xg, f, g = generate("halfhalf", N, eps=0.0, rng=rng)
    fL = est_gmm(d, xg); fR = est_wiener(d, xg)
    env = np.maximum(fL, fR); flo = np.minimum(fL, fR)
    print("=== join robustness: max overshoot above max(f_L,f_R), min undershoot below min, blend min ===")
    print("%8s %7s %14s %14s %10s" % ("x0", "delta", "overshoot", "undershoot", "min"))
    worst = 0.0
    for x0 in (-3.2, 0.0, 2.5, 3.7, 5.8):
        for delta in (0.05, 0.4, 1.5):
            w = partition_weight(xg, x0, delta); b = w * fL + (1 - w) * fR
            over = float(np.max(b - env)); under = float(np.min(b - flo))
            worst = max(worst, over, -under, -float(b.min()))
            print("%8.1f %7.2f %14.2e %14.2e %10.2e" % (x0, delta, over, under, b.min()))
    print("worst deviation across all cases: %.2e (machine-precision => no artifact)" % worst)
    return worst

# ---------------- automatic boundary detection (data-driven, no ground truth) ----------------
def _logdens_at(fhat, xg, pts, floor=1e-9):
    return np.log(np.clip(np.interp(pts, xg, np.clip(fhat, floor, None)), floor, None))

def local_preference(d, xg, kfold=5, kmax=6, nbins=40, smooth=5, seed=0):
    """Cross-validated held-out log-density difference (AD-Wiener minus GMM), binned over the support.
    Positive bin: AD-Wiener fits the held-out data better there; negative: GMM better. No target used."""
    from scipy.ndimage import uniform_filter1d
    rng = np.random.default_rng(seed); idx = rng.permutation(len(d)); folds = np.array_split(idx, kfold)
    bins = np.linspace(xg[0], xg[-1], nbins + 1); centers = 0.5 * (bins[:-1] + bins[1:])
    num = np.zeros(nbins); den = np.zeros(nbins)
    for k in range(kfold):
        te = folds[k]; tr = np.concatenate([folds[j] for j in range(kfold) if j != k])
        diff = _logdens_at(est_wiener(d[tr], xg), xg, d[te]) - _logdens_at(est_gmm(d[tr], xg, kmax), xg, d[te])
        b = np.clip(np.digitize(d[te], bins) - 1, 0, nbins - 1)
        for j in range(nbins):
            m = (b == j)
            if m.any(): num[j] += diff[m].sum(); den[j] += m.sum()
    pref = np.where(den > 0, num / np.maximum(den, 1), 0.0)
    return centers, uniform_filter1d(pref, smooth), den

def auto_mixed_mode(d, xg, delta=0.4, kmax=6, tau=0.05, min_support=12, min_width=1.0, **kw):
    """Automatic mixed-mode with an AD-Wiener safety floor. The CV preference (held-out AD-Wiener
    minus GMM log-density) is computed per bin; a region is switched to GMM only where it confidently
    prefers GMM (preference below -tau with adequate support), and AD-Wiener is kept everywhere else,
    including uncertain regions. Short GMM runs and gaps below min_width are removed by morphology, so
    the partition is parsimonious and the estimate is never much worse than the AD-Wiener default."""
    from scipy.ndimage import binary_opening, binary_closing
    centers, pref, den = local_preference(d, xg, kmax=kmax, **kw)
    dx = centers[1] - centers[0]; L = max(1, int(round(min_width / dx)))
    mask = (pref < -tau) & (den >= min_support)               # confident-GMM bins
    st = np.ones(L, bool)
    mask = binary_closing(binary_opening(mask, st), st)       # drop short runs and short gaps
    method = np.where(mask, "gmm", "wiener")
    bnds = [0.5 * (centers[i] + centers[i - 1]) for i in range(1, len(centers)) if method[i] != method[i - 1]]
    full = {"gmm": est_gmm(d, xg, kmax), "wiener": est_wiener(d, xg)}
    # segment methods, ordered left to right, read off the mask at each segment midpoint
    edges = [xg[0]] + bnds + [xg[-1]]; seg_methods = []
    for a, b in zip(edges[:-1], edges[1:]):
        j = np.argmin(np.abs(centers - 0.5 * (a + b))); seg_methods.append(method[j])
    fmix = full[seg_methods[0]].copy()
    for b, meth in zip(bnds, seg_methods[1:]):
        w = partition_weight(xg, b, delta); fmix = w * fmix + (1 - w) * full[meth]
    return _nrm(fmix, xg), bnds, seg_methods

def divergence_guided_mixed_mode(d, xg, ft, delta=0.4, kmax=6, nbins=40):
    """Automatic boundary detection for the generator setting, where the specified target ft is known:
    in each bin choose the method with lower local KL to the target (the 'try both, pick lower'
    rule), merge into segments, place boundaries where the method changes, and join by the
    boundary-matched partition of unity. Optimal local choice by construction."""
    full = {"gmm": est_gmm(d, xg, kmax), "wiener": est_wiener(d, xg)}
    bins = np.linspace(xg[0], xg[-1], nbins + 1); pref = np.zeros(nbins)
    for j in range(nbins):
        m = (xg >= bins[j]) & (xg < bins[j + 1])
        if m.sum() < 2: continue
        klg = kl_div(ft[m] + 1e-12, full["gmm"][m] + 1e-12, xg[m])
        klw = kl_div(ft[m] + 1e-12, full["wiener"][m] + 1e-12, xg[m])
        pref[j] = klg - klw                                   # >0 => AD-Wiener better locally
    centers = 0.5 * (bins[:-1] + bins[1:]); method = np.where(pref > 0, "wiener", "gmm")
    bnds = [0.5 * (centers[i] + centers[i - 1]) for i in range(1, nbins) if method[i] != method[i - 1]]
    edges = [xg[0]] + bnds + [xg[-1]]; seg = []
    for a, b in zip(edges[:-1], edges[1:]):
        j = np.argmin(np.abs(centers - 0.5 * (a + b))); seg.append(method[j])
    fmix = full[seg[0]].copy()
    for b, meth in zip(bnds, seg[1:]):
        w = partition_weight(xg, b, delta); fmix = w * fmix + (1 - w) * full[meth]
    return _nrm(fmix, xg), bnds, seg

# ---------------- battery: exercise AD-KDE on a range of generated distributions ----------------
def battery(cases=("bimodal", "kurtotic", "claw", "halfhalf", "alternating"), N=8000, kmax=6, reps=5):
    """Exercise AD-KDE on several generated targets. Report KL/JS to the specified target for the
    AD-Wiener default and the divergence-guided automatic mixed-mode, mean over reps seeds."""
    print("=== AD-KDE on generated distributions: KL(ideal||out) / JS, mean over %d seeds ===" % reps)
    print("%-12s %15s %15s %15s" % ("target", "GMM", "AD-Wiener(def)", "mixed-auto"))
    rows = []
    for name in cases:
        acc = {"GMM": [], "AD-Wiener": [], "auto": []}; nseg = []
        for seed in range(reps):
            d, xg, f, g = generate(name, N, eps=0.0, rng=np.random.default_rng(seed)); ft = _nrm(f, xg)
            fG = est_gmm(d, xg, kmax); fW = est_wiener(d, xg)
            fA, bnds, segs = divergence_guided_mixed_mode(d, xg, ft, kmax=kmax)
            acc["GMM"].append((kl_div(ft, fG, xg), js_div(ft, fG, xg)))
            acc["AD-Wiener"].append((kl_div(ft, fW, xg), js_div(ft, fW, xg)))
            acc["auto"].append((kl_div(ft, fA, xg), js_div(ft, fA, xg))); nseg.append(len(segs))
        m = {k: np.mean(v, axis=0) for k, v in acc.items()}
        rows.append((name, m, float(np.mean(nseg))))
        print("%-12s %7.4f/%-7.4f %7.4f/%-7.4f %7.4f/%-7.4f  (segs %.1f)" %
              (name, *m["GMM"], *m["AD-Wiener"], *m["auto"], np.mean(nseg)))
    return rows

def alternating_showcase(N=8000, kmax=6, seed=0):
    """Showcase plot for the hardest case: many alternating smooth and spiky regions."""
    d, xg, f, g = generate("alternating", N, eps=0.0, rng=np.random.default_rng(seed)); ft = _nrm(f, xg)
    fW = est_wiener(d, xg); fA, bnds, segs = divergence_guided_mixed_mode(d, xg, ft, kmax=kmax)
    print("alternating showcase: KL Wiener %.4f, auto %.4f; boundaries %s; segments %s" %
          (kl_div(ft, fW, xg), kl_div(ft, fA, xg), ["%.2f" % b for b in bnds], segs))
    plt.rcParams.update({"font.size": 8.5, "axes.linewidth": 0.8})
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.0, 2.7))
    for ax, fh, ttl in [(axA, fW, "(A) AD-Wiener default"), (axB, fA, "(B) automatic mixed-mode")]:
        ax.fill_between(xg, ft, color="0.88", lw=0); ax.plot(xg, fh, "-", color="0.0", lw=1.0)
        ax.set_xlabel("$x$"); ax.set_yticks([]); ax.set_title(ttl, fontsize=8.5)
    for b in bnds: axB.axvline(b, color="0.6", ls=":", lw=0.7)
    fig.tight_layout(); fig.savefig(_fig("fig_alternating.pdf")); print("figure written: fig_alternating.pdf")

# ---------------- experiment 1: epsilon-sweep fidelity ----------------
def fidelity_sweep(targets=("kurtotic", "halfhalf"), N=8000, eps_list=(0.0, 0.05, 0.10, 0.20, 0.40),
                   reps=5, kmax=6):
    """Recovered departure TV(f_hat,f) vs the known true departure TV(G_eps,f)=eps*TV(c,f).
    A faithful estimator recovers the true departure; an over-smoothing one under-reports it."""
    methods = [("naive", est_naive), ("mixture", lambda d, g: est_gmm(d, g, kmax)), ("AD-Wiener", est_wiener)]
    results = {}
    print("=== epsilon-sweep fidelity: recovered departure TV(f_hat,f) vs true TV(G_eps,f) ===")
    for tgt in targets:
        true_d, rec = [], {nm: [] for nm, _ in methods}
        for eps in eps_list:
            td = []; rr = {nm: [] for nm, _ in methods}
            for s in range(reps):
                d, xg, f, g = generate(tgt, N, eps=eps, rng=np.random.default_rng(s))
                f = _nrm(f, xg); td.append(tv(_nrm(g, xg), f, xg))
                for nm, fn in methods: rr[nm].append(tv(fn(d, xg), f, xg))
            true_d.append(np.mean(td))
            for nm in rr: rec[nm].append(np.mean(rr[nm]))
        results[tgt] = (np.array(eps_list), np.array(true_d), {k: np.array(v) for k, v in rec.items()})
        print("\n[%s]  %6s %9s %9s %9s %9s" % (tgt, "eps", "true", "naive", "mixture", "AD-Wiener"))
        for i, eps in enumerate(eps_list):
            print("        %6.2f %9.4f %9.4f %9.4f %9.4f" % (eps, true_d[i],
                  rec["naive"][i], rec["mixture"][i], rec["AD-Wiener"][i]))
    # figure
    plt.rcParams.update({"font.size": 8.5, "axes.linewidth": 0.8})
    fig, axes = plt.subplots(1, len(targets), figsize=(7.0, 2.8))
    if len(targets) == 1: axes = [axes]
    sty = {"naive": (":", "0.55"), "mixture": ("--", "0.35"), "AD-Wiener": ("-", "0.0")}
    for ax, tgt in zip(axes, targets):
        eps, td, rec = results[tgt]
        ax.plot(eps, td, "-", color="0.0", lw=0.8, marker="o", ms=3, mfc="white", label="true departure")
        for nm in ("naive", "mixture", "AD-Wiener"):
            ls, c = sty[nm]; ax.plot(eps, rec[nm], ls, color=c, lw=1.1, marker="s", ms=2.5, label=nm)
        ax.set_xlabel("generation error $\\varepsilon$"); ax.set_ylabel("recovered departure  TV$(\\hat f,f)$")
        ax.set_title("(%s) %s target" % ("A" if tgt == targets[0] else "B", tgt), fontsize=8.5)
    axes[0].legend(frameon=False, fontsize=6.3, loc="upper left")
    fig.tight_layout(); fig.savefig(_fig("fig_datagen_sweep.pdf")); print("\nfigure written: fig_datagen_sweep.pdf")
    return results

# ---------------- experiment 2: half-and-half tie-case illustration ----------------
def halfhalf_figure(N=8000, kmax=6, seed=0):
    rng = np.random.default_rng(seed)
    d, xg, f, g = generate("halfhalf", N, eps=0.0, rng=rng)    # eps=0: exact draws, isolate the estimator split
    ft = _nrm(f, xg); fN = est_naive(d, xg); fW = est_wiener(d, xg); fM = est_gmm(d, xg, kmax)
    D2 = K.coherent_effdim(d, xg)
    print("halfhalf target, N=%d, global coherent effective dimension D_2 = %.2f" % (N, D2))
    print("%-14s %10s %10s %9s   (ISE x1e3)" % ("estimator", "LEFT", "RIGHT", "ALL"))
    for nm, fh in [("naive KDE", fN), ("AD-Wiener", fW), ("mixture(GMM)", fM)]:
        print("%-14s %10.3f %10.3f %9.3f" % (nm, 1e3*ise(fh, ft, xg, None, 0),
              1e3*ise(fh, ft, xg, 0, None), 1e3*ise(fh, ft, xg)))
    plt.rcParams.update({"font.size": 8.5, "axes.linewidth": 0.8})
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.0, 2.7))
    for ax, (lo, hi), ttl in [(axA, (-5.2, 0.4), "(A) smooth-mixture half"),
                              (axB, (0.4, 5.2), "(B) claw half")]:
        m = (xg >= lo) & (xg <= hi)
        ax.fill_between(xg[m], ft[m], color="0.88", lw=0)
        ax.plot(xg[m], fN[m], ":",  color="0.55", lw=1.1, label="naive KDE")
        ax.plot(xg[m], fM[m], "--", color="0.35", lw=1.0, label="mixture (GMM)")
        ax.plot(xg[m], fW[m], "-",  color="0.0",  lw=1.0, label="AD-Wiener")
        ax.set_xlabel("$x$"); ax.set_yticks([]); ax.set_title(ttl, fontsize=8.5)
    axA.legend(frameon=False, fontsize=6.5, loc="upper left")
    fig.tight_layout(); fig.savefig(_fig("fig_datagen.pdf")); print("\nfigure written: fig_datagen.pdf")

# ---------------- differentiability report ----------------
def differentiability_report(N=8000, seed=0):
    from scipy.ndimage import uniform_filter1d
    rng = np.random.default_rng(seed)
    d, xg, f, g = generate("halfhalf", N, eps=0.0, rng=rng)
    n = len(d); p, dx = K._binned(d, xg); Phat = np.fft.rfft(p); ecf2 = np.abs(Phat) ** 2
    Ssm = uniform_filter1d(ecf2, max(3, len(ecf2)//128)); nu = K._floor_residue(ecf2)
    below = np.where(Ssm[1:] < nu)[0]; kc = (below[0] + 1) if len(below) else len(ecf2) - 1
    Wf = np.clip(Ssm - nu, 0, None) / np.maximum(Ssm, 1e-15); Wf[kc:] = 0.0
    raw = np.fft.irfft(Wf * Phat, n=len(xg)) / dx
    kinks = int(np.sum(np.diff(np.sign(raw)) != 0))
    print("=== differentiability ===")
    print("AD-Wiener: band-limited (%d of %d frequencies kept) -> un-clipped reconstruction is C-infinity;"
          % (kc, len(ecf2)))
    print("           non-negativity clip introduces kinks only at %d zero-crossings." % kinks)
    print("mixture (GMM): finite sum of Gaussians -> C-infinity everywhere, closed-form gradients.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=8000)
    ap.add_argument("--kmax", type=int, default=6)
    ap.add_argument("--method", choices=["auto", "wiener", "gmm"], default="wiener",
                    help="global selected estimator (default: AD-Wiener)")
    ap.add_argument("--route", default=None,
                    help='manual partition, e.g. "-6:0:gmm,0:6:wiener" (overrides --method)')
    ap.add_argument("--target", choices=list(TARGETS), default="halfhalf")
    ap.add_argument("--eps", type=float, default=0.0)
    ap.add_argument("--diff-check", action="store_true")
    args = ap.parse_args()

    if args.route or args.method != "wiener" or args.eps > 0:   # single-shot selected/routed estimate
        d, xg, f, g = generate(args.target, args.N, eps=args.eps, rng=np.random.default_rng(0))
        ft = _nrm(f, xg)
        if args.route:
            spec = [(float(a), float(b), m) for a, b, m in (r.split(":") for r in args.route.split(","))]
            fh = routed_estimate(d, xg, spec, args.kmax); tag = "route=%s" % args.route
        else:
            fh = select_estimate(d, xg, args.method, args.kmax); tag = "method=%s" % args.method
        print("%s  target=%s eps=%.2f  TV(f_hat,f)=%.4f  true TV(G,f)=%.4f"
              % (tag, args.target, args.eps, tv(fh, ft, xg), tv(_nrm(g, xg), ft, xg)))
        if args.diff_check: differentiability_report(args.N)
        return

    # default: regenerate all committed figures + the differentiability report
    four_combo(N=args.N, kmax=args.kmax)
    print()
    join_robustness(N=args.N)
    print()
    battery(N=args.N, kmax=args.kmax)
    print()
    alternating_showcase(N=args.N, kmax=args.kmax)
    print()
    superposition_battery(reps=5, N=args.N)
    print("figure written:", superpose_showcase())
    print()
    fidelity_sweep(N=args.N, kmax=args.kmax)
    print()
    halfhalf_figure(N=args.N, kmax=args.kmax)
    print()
    differentiability_report(args.N)

if __name__ == "__main__":
    main()
