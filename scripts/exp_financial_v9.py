"""exp_financial_v9.py -- a leptokurtic financial-returns application (the kurtotic case).

Daily returns are modeled as a calm/turbulent normal mixture, the canonical leptokurtic
(sharp-peak, heavy-tail) shape of asset returns. The density and its left-tail risk
(Value-at-Risk and Expected Shortfall) are estimated from a Gaussian fit (the RiskMetrics-style
baseline), an ordinary KDE, the spectral AD-Wiener estimator, and an adaptive mixture (the
fit_mixture plugin; default Gaussian-mixture EM with BIC, an external backend when registered). The Gaussian
baseline underestimates tail risk; the mixture, which matches the generating structure, recovers
the density and the tail risk most accurately. Reproducible with the self-contained default.
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import adkde_plugins as P
import ad_kde_v30 as K

FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
W, S1, S2 = 0.9, 0.008, 0.030                # 90% calm (0.8%/day), 10% turbulent (3%/day)

def _gauss(x, m, s):
    return np.exp(-0.5 * ((x - m) / s) ** 2) / (s * np.sqrt(2 * np.pi))

def true_pdf(x):
    return W * _gauss(x, 0, S1) + (1 - W) * _gauss(x, 0, S2)

def sample_returns(n, rng):
    calm = rng.random(n) < W
    return np.where(calm, rng.normal(0, S1, n), rng.normal(0, S2, n))

def excess_kurtosis():
    m2 = W * S1 ** 2 + (1 - W) * S2 ** 2
    m4 = W * 3 * S1 ** 4 + (1 - W) * 3 * S2 ** 4
    return m4 / m2 ** 2 - 3.0

def _quantile(cdf, xg, a):
    return xg[min(np.searchsorted(cdf, a), len(xg) - 1)]

def tail_risk(f, xg, alphas=(0.01, 0.05)):
    f = np.clip(f, 0, None); f = f / np.trapezoid(f, xg)
    cdf = np.cumsum(f); cdf /= cdf[-1]; out = {}
    for a in alphas:
        v = _quantile(cdf, xg, a); m = xg <= v
        es = np.trapezoid(xg[m] * f[m], xg[m]) / max(np.trapezoid(f[m], xg[m]), 1e-12)
        out[a] = (v, es)
    return out

def run(n=1000, reps=200, seed=2024):
    xg = np.linspace(-0.2, 0.2, 20001)
    F = np.cumsum(true_pdf(xg)); F /= F[-1]
    VaR_t = {a: _quantile(F, xg, a) for a in (0.01, 0.05)}
    truth = tail_risk(true_pdf(xg), xg)
    ES_t = {a: truth[a][1] for a in (0.01, 0.05)}
    methods = ["Gaussian", "KDE", "AD-Wiener", "Mixture"]
    agg = {m: {"ise": [], 0.01: ([], []), 0.05: ([], [])} for m in methods}
    rng = np.random.default_rng(seed)
    last = {}
    for _ in range(reps):
        d = sample_returns(n, rng)
        ests = {
            "Gaussian":  _gauss(xg, d.mean(), d.std()),
            "KDE":       K.fft_kde(d, K.h_silverman(d), xg),
            "AD-Wiener": K.ad_wiener(d, xg, "simple"),
            "Mixture":   P.fit_mixture(d, kmax=4)(xg),
        }
        last = ests
        for m, f in ests.items():
            fn = np.clip(f, 0, None); fn = fn / np.trapezoid(fn, xg)
            agg[m]["ise"].append(np.trapezoid((fn - true_pdf(xg)) ** 2, xg))
            tr = tail_risk(fn, xg)
            for a in (0.01, 0.05):
                agg[m][a][0].append(tr[a][0]); agg[m][a][1].append(tr[a][1])
    return xg, VaR_t, ES_t, agg, last

if __name__ == "__main__":
    xg, VaR_t, ES_t, agg, last = run()
    print("=== Leptokurtic returns / tail risk (mixture backend: %s) ===" % P.active("mixture"))
    print("excess kurtosis = %.1f ;  true VaR1%%=%.4f  VaR5%%=%.4f  ES1%%=%.4f"
          % (excess_kurtosis(), VaR_t[0.01], VaR_t[0.05], ES_t[0.01]))
    print("%-10s %14s %10s %10s %10s" % ("method", "ISE(1e3)+-sd", "VaR1%err", "VaR5%err", "ES1%err"))
    rows = {}
    for m in agg:
        ise = 1e3 * np.mean(agg[m]["ise"]); ise_sd = 1e3 * np.std(agg[m]["ise"])
        v1 = 1e4 * abs(np.mean(agg[m][0.01][0]) - VaR_t[0.01]); v1sd = 1e4 * np.std(agg[m][0.01][0])
        v5 = 1e4 * abs(np.mean(agg[m][0.05][0]) - VaR_t[0.05]); v5sd = 1e4 * np.std(agg[m][0.05][0])
        e1 = 1e4 * abs(np.mean(agg[m][0.01][1]) - ES_t[0.01]); e1sd = 1e4 * np.std(agg[m][0.01][1])
        rows[m] = (ise, ise_sd, v1, v1sd, v5, v5sd, e1, e1sd)
        print("%-10s %8.1f+-%-4.1f %10.1f %10.1f %10.1f" % (m, ise, ise_sd, v1, v5, e1))
    print("  (per-replication sd of VaR1%%/VaR5%%/ES1%% estimates, x1e4: "
          + ", ".join("%s %.0f/%.0f/%.0f" % (m, rows[m][3], rows[m][5], rows[m][7]) for m in agg) + ")")

    plt.rcParams.update({"font.size": 8.5, "axes.linewidth": 0.8})
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ax.plot(xg * 100, true_pdf(xg), color="0.0", lw=1.4, label="true")
    sty = {"Gaussian": ("--", "0.55"), "Mixture": ("-.", "0.2"), "AD-Wiener": (":", "0.4")}
    for m in ["Gaussian", "AD-Wiener", "Mixture"]:
        ls, c = sty[m]
        ax.plot(xg * 100, np.clip(last[m], 0, None) / np.trapezoid(np.clip(last[m], 0, None), xg),
                ls, color=c, lw=1.1, label=m)
    ax.axvline(VaR_t[0.01] * 100, color="0.7", lw=0.8, ls=":")
    ax.set_yscale("log"); ax.set_ylim(1e-1, 1e2)
    ax.set_xlim(-12, 12); ax.set_xlabel("daily return (%)"); ax.set_ylabel("density (log scale)")
    ax.text(VaR_t[0.01] * 100 - 0.3, 0.2, "1% VaR", color="0.5", fontsize=7, ha="right")
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    fig.tight_layout()
    out = os.path.join(FIG, "fig_financial.pdf")
    fig.savefig(out); print("figure written:", os.path.basename(out))
