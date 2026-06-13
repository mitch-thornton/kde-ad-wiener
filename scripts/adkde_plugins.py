"""adkde_plugins.py -- plugin interface for the AD-KDE multivariate and mixture extensions.

Three plugin hooks, each with a self-contained default so every experiment in this bundle is
reproducible WITHOUT any private engine or GPL dependency:

    group_library(M)     -> list of candidate groups          (default: cyclic/translation group)
    match_group(R)       -> the matched group for covariance R (default: assume cyclic)
    fit_mixture(d, kmax) -> adaptive mixture density (callable) (default: Gaussian-mixture EM + BIC)

To use a richer backend, register it before running an experiment:

    import adkde_plugins as P
    P.register("bgm",           my_blind_group_matcher)   # e.g. the AD BMG reference pipeline
    P.register("group_library", my_group_library)         # a user-defined candidate library
    P.register("mixture",       my_mixture_backend)         # any external mixture library

The released bundle ships this file (defaults + registry). The default mixture backend is a
self-contained Gaussian-mixture EM with BIC order selection, with no external dependency; a
practitioner may register their own blind-group-matching backend, candidate group library, or
external mixture estimator in its place.
"""
import numpy as np

_REGISTRY = {"group_library": None, "bgm": None, "mixture": None}

def register(kind, fn):
    if kind not in _REGISTRY:
        raise KeyError("unknown plugin '%s' (expected one of %s)" % (kind, list(_REGISTRY)))
    _REGISTRY[kind] = fn

def active(kind):
    """Report whether a hook is using a registered backend or the built-in default."""
    return "custom" if _REGISTRY[kind] is not None else "default"

# ---------------------------------------------------------------------------
# A "group" is a small dict {"name": str, "reynolds": callable S -> R_G}.
# The Reynolds projector averages a covariance over the group, i.e. projects it
# onto the commutant; for the cyclic group this is the circulant projection,
# whose eigenbasis is the DFT (the matched transform used throughout the paper).
# ---------------------------------------------------------------------------
def _circulant_from_diag(diag):
    d = len(diag)
    return np.array([[diag[(i - j) % d] for j in range(d)] for i in range(d)])

def _reynolds_cyclic(S):
    d = len(S)
    avg = np.array([np.mean([S[i, (i + k) % d] for i in range(d)]) for k in range(d)])
    return _circulant_from_diag(avg)

def cyclic_group(M):
    return {"name": "cyclic", "reynolds": _reynolds_cyclic}

def group_library(M):
    if _REGISTRY["group_library"] is not None:
        return _REGISTRY["group_library"](M)
    return [cyclic_group(M)]

def match_group(data):
    """Blind group matching from snapshot data (an (n, d) array). Default: assume the cyclic
    (translation) group used throughout the paper. Register a 'bgm' backend for blind discovery
    (the backend may use a train/test split of the snapshots, as the AD reference pipeline does)."""
    data = np.asarray(data)
    if _REGISTRY["bgm"] is not None:
        return _REGISTRY["bgm"](data)
    d = data.shape[1] if data.ndim == 2 else len(data)
    return cyclic_group(d)

def reynolds(S, group=None):
    """Group-averaged covariance (Reynolds projection onto the matched commutant)."""
    if group is None:
        group = cyclic_group(len(S))
    return group["reynolds"](S)

# ---------------------------------------------------------------------------
# Mixture plugin. Default: a self-contained Gaussian-mixture EM with BIC order
# selection (no external dependency). Register an external mixture backend for a
# richer cross-family library if one is available.
# ---------------------------------------------------------------------------
def _gauss(x, m, v):
    return np.exp(-0.5 * (x - m) ** 2 / v) / np.sqrt(2 * np.pi * v)

def _gmm_em(d, k, iters=200, seed=0):
    rng = np.random.default_rng(seed)
    mu = rng.choice(d, size=k, replace=k > len(d))
    var = np.full(k, max(d.var(), 1e-6))
    pi = np.ones(k) / k
    for _ in range(iters):
        P = np.array([pi[j] * _gauss(d, mu[j], var[j]) for j in range(k)]) + 1e-300
        P /= P.sum(0)
        Nk = P.sum(1) + 1e-12
        pi = Nk / len(d)
        mu = (P @ d) / Nk
        var = np.array([(P[j] * (d - mu[j]) ** 2).sum() / Nk[j] for j in range(k)]) + 1e-8
    ll = np.sum(np.log(np.sum([pi[j] * _gauss(d, mu[j], var[j]) for j in range(k)], 0) + 1e-300))
    bic = -2 * ll + (3 * k - 1) * np.log(len(d))
    return (pi, mu, var), bic

def fit_mixture(d, kmax=6):
    """Adaptive mixture density estimate (returns a callable pdf). Default: Gaussian-mixture
    EM with BIC-selected order. Register a 'mixture' backend for a richer cross-family
    library that also covers skew-normal and heavy-tailed components."""
    if _REGISTRY["mixture"] is not None:
        return _REGISTRY["mixture"](d, kmax)
    best = None
    for k in range(1, kmax + 1):
        params, bic = _gmm_em(d, k)
        if best is None or bic < best[1]:
            best = (params, bic)
    pi, mu, var = best[0]
    return lambda x: sum(pi[j] * _gauss(x, mu[j], var[j]) for j in range(len(pi)))
