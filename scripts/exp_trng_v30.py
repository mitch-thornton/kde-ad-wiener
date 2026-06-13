"""TRNG case: the NIST randomness beacon (Beacon 2.0) bitstream, verified uncorrupted by the core
SP 800-22 tests (sts_runner.py), is read as uniform deviates and estimated by AD-Wiener and GMM.
The specified target is the uniform density on [0,1); KL and JS are reported against it."""
import numpy as np, os, exp_datagen_v30 as E
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def uniform_deviates_from_bytes(path=None):
    by = np.load(path or os.path.join(_ROOT, "data", "beacon_bytes.npy")); u = by[:(by.size // 4) * 4].reshape(-1, 4).astype(np.uint32)
    return (u[:, 0] << 24 | u[:, 1] << 16 | u[:, 2] << 8 | u[:, 3]).astype(np.float64) / 2 ** 32

def trng_density(N=8000, seed=0, flatten=False):
    try:
        vals = uniform_deviates_from_bytes(); d = np.random.default_rng(seed).choice(vals, N, replace=False)
    except FileNotFoundError:
        d = np.load(os.path.join(_ROOT, "data", "beacon_uniform_8000.npy"))[:N]      # vendored derived sample
    xg = np.linspace(0.0, 1.0, 1024); ft = np.ones_like(xg)  # uniform target on [0,1)
    rows = [("naive KDE", E.est_naive(d, xg)), ("GMM", E.est_gmm(d, xg)), ("AD-Wiener", E.est_wiener(d, xg))]
    if flatten:
        rows.append(("AD-Wiener+flatten", E._nrm(E.K.ad_wiener(d, xg, strip="residue", flatten=True), xg)))
    print("=== beacon uniform deviates, KL/JS to the uniform target (N=%d) ===" % N)
    for name, fh in rows:
        print("%-18s KL %.6f  JS %.6f" % (name, E.kl_div(ft, fh, xg), E.js_div(ft, fh, xg)))

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--flatten", action="store_true",
        help="Also report AD-Wiener with the uniformity gate on (snaps to a perfect uniform).")
    trng_density(flatten=ap.parse_args().flatten)
