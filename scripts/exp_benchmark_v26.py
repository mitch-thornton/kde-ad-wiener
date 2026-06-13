#!/usr/bin/env python3
"""Marron-Wand benchmark: AD-Wiener / superposition vs strong baselines, exact ISE on known densities."""
import os, sys, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ad_kde_v30 as K
import exp_datagen_v30 as E
import adkde_plugins as P
from KDEpy.bw_selection import improved_sheather_jones, silvermans_rule
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV

# ---------------- Marron-Wand 15 densities as (weight, mean, sd) lists ----------------
def _mw():
    d = {}
    d["1 Gaussian"] = [(1, 0, 1)]
    d["2 Skewed unimodal"] = [(1/5,0,1),(1/5,1/2,2/3),(3/5,13/12,5/9)]
    d["3 Strongly skewed"] = [(1/8, 3*((2/3)**l - 1), (2/3)**l) for l in range(8)]
    d["4 Kurtotic unimodal"] = [(2/3,0,1),(1/3,0,1/10)]
    d["5 Outlier"] = [(1/10,0,1),(9/10,0,1/10)]
    d["6 Bimodal"] = [(1/2,-1,2/3),(1/2,1,2/3)]
    d["7 Separated bimodal"] = [(1/2,-3/2,1/2),(1/2,3/2,1/2)]
    d["8 Skewed bimodal"] = [(3/4,0,1),(1/4,3/2,1/3)]
    d["9 Trimodal"] = [(9/20,-6/5,3/5),(9/20,6/5,3/5),(1/10,0,1/4)]
    d["10 Claw"] = [(1/2,0,1)] + [(1/10, l/2 - 1, 1/10) for l in range(5)]
    d["11 Double claw"] = [(49/100,-1,2/3),(49/100,1,2/3)] + [(1/350,(l-3)/2,1/100) for l in range(7)]
    d["12 Asymmetric claw"] = [(1/2,0,1)] + [(2**(1-l)/31, l+1/2, 2**(-l)/10) for l in range(-2,3)]
    d["13 Asym double claw"] = [(46/100,2*l-1,2/3) for l in range(2)] + \
        [(1/300,-l/2,1/100) for l in range(1,4)] + [(7/300,l/2,7/100) for l in range(1,4)]
    d["14 Smooth comb"] = [(2**(5-l)/63, (65-96*(1/2)**l)/21, (32/63)/2**l) for l in range(6)]
    d["15 Discrete comb"] = [(2/7,(12*l-15)/7,2/7) for l in range(3)] + [(1/21,2*l/7,1/21) for l in range(8,11)]
    return d

MW = _mw()

def true_pdf(comps, x):
    return sum(w * np.exp(-0.5*((x-m)/s)**2)/(s*np.sqrt(2*np.pi)) for w,m,s in comps)

def sample(comps, n, rng):
    w = np.array([c[0] for c in comps]); w = w/w.sum()
    idx = rng.choice(len(comps), size=n, p=w)
    return np.array([rng.normal(comps[i][1], comps[i][2]) for i in idx])

def grid_for(comps):
    lo = min(m-6*s for _,m,s in comps); hi = max(m+6*s for _,m,s in comps)
    return np.linspace(lo, hi, 2048)

# ---------------- estimators: data -> density on xg ----------------
def _nrm(f, xg):
    f = np.clip(f, 0, None); z = np.trapezoid(f, xg); return f/z if z>0 else f

def kde_fixed_bw(d, xg, h):
    return _nrm(np.mean(np.exp(-0.5*((xg[:,None]-d[None,:])/h)**2)/(h*np.sqrt(2*np.pi)), axis=1), xg)

def est_silverman(d, xg):
    return kde_fixed_bw(d, xg, max(silvermans_rule(d.reshape(-1,1)), 1e-3))

def est_isj(d, xg):
    try: h = improved_sheather_jones(d.reshape(-1,1))
    except Exception: h = silvermans_rule(d.reshape(-1,1))
    return kde_fixed_bw(d, xg, max(h, 1e-3))

def est_cv(d, xg):
    sub = d if len(d) <= 800 else d[np.random.default_rng(0).choice(len(d),800,replace=False)]
    hs = np.linspace(0.05, 1.0, 12) * sub.std()
    gs = GridSearchCV(KernelDensity(kernel="gaussian"), {"bandwidth": hs}, cv=3)
    gs.fit(sub.reshape(-1,1)); h = gs.best_params_["bandwidth"]
    return kde_fixed_bw(d, xg, max(h, 1e-3))

def est_abramson(d, xg):
    h0 = max(silvermans_rule(d.reshape(-1,1)), 1e-3)
    pilot = np.clip(np.mean(np.exp(-0.5*((d[:,None]-d[None,:])/h0)**2)/(h0*np.sqrt(2*np.pi)), axis=1), 1e-12, None)
    g = np.exp(np.mean(np.log(pilot))); hi = h0*np.sqrt(g/pilot)
    f = np.mean(np.exp(-0.5*((xg[:,None]-d[None,:])/hi[None,:])**2)/(hi[None,:]*np.sqrt(2*np.pi)), axis=1)
    return _nrm(f, xg)

def est_gmm(d, xg):
    from sklearn.mixture import GaussianMixture
    best = None
    for k in range(1, 11):
        gm = GaussianMixture(k, covariance_type="full", reg_covar=1e-5, max_iter=100,
                             random_state=0).fit(d.reshape(-1,1))
        b = gm.bic(d.reshape(-1,1))
        if best is None or b < best[1]: best = (gm, b)
    f = np.exp(best[0].score_samples(xg.reshape(-1,1)))
    return _nrm(f, xg)

def est_adwiener(d, xg):
    return _nrm(K.ad_wiener(d, xg, strip="residue"), xg)

def est_superpose(d, xg):
    return _nrm(E.superpose(d, xg)[0], xg)

ESTS = {"Silverman":est_silverman, "ISJ/Botev":est_isj, "LSCV":est_cv, "Abramson":est_abramson,
        "GMM-BIC":est_gmm, "AD-Wiener":est_adwiener, "superpose":est_superpose}

def ise(fhat, ftrue, xg):
    return np.trapezoid((fhat-ftrue)**2, xg)

def run(densities, ns, reps, ests=ESTS):
    out = {}  # (dname, n) -> {est: [ise per rep]}
    for dname in densities:
        comps = MW[dname]; xg = grid_for(comps); ft = true_pdf(comps, xg)
        for n in ns:
            key = (dname, n); out[key] = {e: [] for e in ests}
            for r in range(reps):
                rng = np.random.default_rng(1000*r + n)
                d = sample(comps, n, rng); d = np.clip(d, xg[0], xg[-1])
                for ename, efn in ests.items():
                    try: out[key][ename].append(ise(efn(d, xg), ft, xg))
                    except Exception: out[key][ename].append(np.nan)
    return out

if __name__ == "__main__":
    import argparse, pickle
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--reps", type=int, default=40)
    ap.add_argument("--lo", type=int, default=0); ap.add_argument("--hi", type=int, default=15)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    if a.quick:
        t0=time.time(); res = run(["1 Gaussian","10 Claw","15 Discrete comb"], [500,5000], 5)
        print("quick test %.1fs"%(time.time()-t0))
        for (dn,n),r in res.items():
            print("%-20s n=%-5d "%(dn,n)+" ".join("%s=%.3f"%(e,1e3*np.nanmean(v)) for e,v in r.items()))
    else:
        names = list(MW.keys())[a.lo:a.hi]; t0=time.time()
        res = run(names, [a.n], a.reps)
        print("ran %d densities n=%d reps=%d in %.0fs"%(len(names), a.n, a.reps, time.time()-t0))
        if a.out:
            with open(a.out,"wb") as f: pickle.dump(res, f)
            print("saved", a.out)
