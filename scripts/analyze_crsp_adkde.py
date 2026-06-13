#!/usr/bin/env python3
"""analyze_crsp_adkde.py -- AD-KDE analysis of CRSP daily returns.

Reads the CSVs written by wrds_extract_crsp.py from a `data/` subdirectory, runs the AD-KDE
estimators on each return series, and writes results into a `results/` subdirectory as JSON.
No WRDS connection is needed here; only numpy/scipy and this bundle's estimators are used. Upload
the contents of `results/` and they will be turned into the tables and plots of the paper.

Four study outputs, one JSON each:
    results/return_distributions.json   per series: grid + density for each method (plot-ready)
    results/risk_estimates.json         per series: mean, volatility, annualized vol, skew, kurtosis
    results/var.json                    per series: Value-at-Risk by method and historical, several levels
    results/tail_risk.json              per series: Expected Shortfall, Hill tail index, exceedances

Each estimator (Gaussian, KDE, AD-Wiener, adaptive Mixture) is compared against the historical
(empirical) reference, since the true law is unknown for real data.
"""
import os, sys, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ad_kde_v30 as K
import adkde_plugins as P

DATADIR = "data"
RESDIR = "results"
ALPHAS = [0.005, 0.01, 0.05]            # tail probabilities for VaR / ES
GRID_N = 512
ANNUALIZE = np.sqrt(252.0)


def _read_csv(path):
    import csv
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    return rows

def load_series():
    """Return {label: 1-D numpy array of daily returns}."""
    series = {}
    mpath = os.path.join(DATADIR, "crsp_market_daily.csv")
    if os.path.exists(mpath):
        rows = _read_csv(mpath)
        for col, lab in [("vwretd", "MKT_vw"), ("ewretd", "MKT_ew"), ("sprtrn", "SP500")]:
            vals = [float(r[col]) for r in rows if r.get(col) not in (None, "", "NA")]
            if vals:
                series[lab] = np.asarray(vals, float)
    spath = os.path.join(DATADIR, "crsp_stock_daily.csv")
    if os.path.exists(spath):
        rows = _read_csv(spath)
        byt = {}
        for r in rows:
            v = r.get("ret")
            if v in (None, "", "NA"):
                continue
            key = r.get("ticker") or ("permno_" + str(r.get("permno")))
            byt.setdefault(key, []).append(float(v))
        for k, v in byt.items():
            if len(v) >= 250:                 # at least ~1 trading year
                series[k] = np.asarray(v, float)
    return series


def _grid(x):
    lo, hi = np.percentile(x, 0.1), np.percentile(x, 99.9)
    pad = 0.25 * (hi - lo + 1e-9)
    return np.linspace(lo - pad, hi + pad, GRID_N)

def _normalize(f, xg):
    f = np.clip(np.asarray(f, float), 0, None)
    z = np.trapezoid(f, xg)
    return f / z if z > 0 else f

def densities(x, xg):
    g = _normalize(np.exp(-0.5 * ((xg - x.mean()) / x.std()) ** 2), xg)
    kde = _normalize(K.fft_kde(x, K.h_silverman(x), xg), xg)
    adw = _normalize(K.ad_wiener(x, xg, "simple"), xg)
    mix = _normalize(P.fit_mixture(x, kmax=4)(xg), xg)
    return {"Gaussian": g, "KDE": kde, "AD-Wiener": adw, "Mixture": mix}

def var_es_from_density(f, xg, alpha):
    f = _normalize(f, xg)
    cdf = np.cumsum(f); cdf = cdf / cdf[-1]
    i = int(np.searchsorted(cdf, alpha)); i = min(max(i, 1), len(xg) - 1)
    v = xg[i]
    m = xg <= v
    es = np.trapezoid(xg[m] * f[m], xg[m]) / max(np.trapezoid(f[m], xg[m]), 1e-12)
    return float(v), float(es)

def hill_tail_index(x, frac=0.05):
    """Hill estimator of the tail index on the left tail (loss magnitudes)."""
    losses = -x[x < 0]
    losses = np.sort(losses)[::-1]
    k = max(10, int(frac * len(losses)))
    k = min(k, len(losses) - 1)
    if k < 10:
        return None
    top = losses[:k]
    xk = losses[k]
    alpha = 1.0 / np.mean(np.log(top / xk))
    return float(alpha)


def main():
    os.makedirs(RESDIR, exist_ok=True)
    series = load_series()
    if not series:
        print("No data found under %s/. Run wrds_extract_crsp.py first." % DATADIR)
        return
    print("analyzing %d series: %s" % (len(series), ", ".join(series)))

    dist_out, risk_out, var_out, tail_out = {}, {}, {}, {}
    for lab, x in series.items():
        xg = _grid(x)
        dens = densities(x, xg)

        dist_out[lab] = {
            "n": int(len(x)),
            "grid": xg.tolist(),
            "hist_edges": np.histogram_bin_edges(x, bins=60).tolist(),
            "hist_counts": np.histogram(x, bins=60, density=True)[0].tolist(),
            "density": {m: f.tolist() for m, f in dens.items()},
        }

        sd = float(x.std(ddof=1))
        risk_out[lab] = {
            "n": int(len(x)),
            "mean_daily": float(x.mean()),
            "vol_daily": sd,
            "vol_annual": float(sd * ANNUALIZE),
            "skew": float(((x - x.mean()) ** 3).mean() / sd ** 3),
            "excess_kurtosis": float(((x - x.mean()) ** 4).mean() / sd ** 4 - 3.0),
            "min": float(x.min()), "max": float(x.max()),
        }

        var_out[lab] = {"levels": ALPHAS, "historical": {}, "methods": {m: {} for m in dens}}
        tail_out[lab] = {"levels": ALPHAS, "historical": {}, "methods": {m: {} for m in dens},
                         "hill_tail_index": hill_tail_index(x)}
        for a in ALPHAS:
            var_out[lab]["historical"]["%.3f" % a] = float(np.quantile(x, a))
            cut = x[x <= np.quantile(x, a)]
            tail_out[lab]["historical"]["%.3f" % a] = float(cut.mean()) if len(cut) else None
            for m, f in dens.items():
                v, es = var_es_from_density(f, xg, a)
                var_out[lab]["methods"][m]["%.3f" % a] = v
                tail_out[lab]["methods"][m]["%.3f" % a] = es

    meta = {"source": "CRSP daily returns via wrds_extract_crsp.py",
            "estimators": ["Gaussian", "KDE (Silverman)", "AD-Wiener", "adaptive Mixture"],
            "reference": "historical (empirical) quantile / shortfall",
            "alphas": ALPHAS, "annualization": "sqrt(252)"}
    for name, obj in [("return_distributions", dist_out), ("risk_estimates", risk_out),
                      ("var", var_out), ("tail_risk", tail_out)]:
        obj["_meta"] = meta
        with open(os.path.join(RESDIR, name + ".json"), "w") as fh:
            json.dump(obj, fh)
        print("wrote %s/%s.json" % (RESDIR, name))


if __name__ == "__main__":
    main()
