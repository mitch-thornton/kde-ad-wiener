"""NIST SP 800-22 verification the standard way: split the stream into k sub-sequences, run each test
per sub-sequence, and report the proportion passing at alpha=0.01. For k sequences the acceptable
proportion is 0.99 +/- 3*sqrt(0.99*0.01/k); a stream is uncorrupted when every test lands in range."""
import numpy as np, sts_tests as T
from math import sqrt

def longest_run_fast(arr):
    p = np.concatenate([[0], arr, [0]]); d = np.diff(p)
    s = np.where(d == 1)[0]; e = np.where(d == -1)[0]
    return int((e - s).max()) if s.size else 0

def longest_run_p(b):
    M = 10000; N = b.size // M; pi = np.array([0.0882, 0.2092, 0.2483, 0.1933, 0.1208, 0.0675, 0.0727])
    nu = np.zeros(7)
    for blk in b[:N * M].reshape(N, M):
        mx = longest_run_fast(blk); cls = 0 if mx <= 10 else (6 if mx >= 16 else mx - 10); nu[cls] += 1
    from scipy.special import gammaincc
    chi2 = (((nu - N * pi) ** 2) / (N * pi)).sum(); return float(gammaincc(3.0, chi2 / 2.0))

b = np.unpackbits(np.load("data/beacon_bytes.npy"))
L = 1 << 20; k = b.size // L                    # 2^20 bits per sequence
seqs = b[:k * L].reshape(k, L)
print("beacon: %d bits -> %d sequences of %d bits" % (b.size, k, L))
tests = {"Monobit": T.monobit, "BlockFreq": lambda s: T.block_frequency(s, 128), "Runs": T.runs,
         "LongestRun": longest_run_p, "Spectral": T.spectral, "Cusum-fwd": lambda s: T.cusum(s, 0),
         "Cusum-bwd": lambda s: T.cusum(s, 1), "ApproxEnt": lambda s: T.approximate_entropy(s, 10),
         "Serial1": lambda s: T.serial(s, 12)[0], "Serial2": lambda s: T.serial(s, 12)[1]}
lo = 0.99 - 3 * sqrt(0.99 * 0.01 / k); hi = 1.0
print("acceptable pass proportion for k=%d: >= %.4f\n" % (k, lo))
print("%-12s %8s  %s" % ("test", "pass/k", "verdict"))
allok = True
for name, fn in tests.items():
    npass = sum(fn(s) >= 0.01 for s in seqs); prop = npass / k; ok = prop >= lo; allok &= ok
    print("%-12s %4d/%-3d  %.4f  %s" % (name, npass, k, prop, "OK" if ok else "BELOW"))
print("\nStream uncorrupted (all tests within acceptable proportion):", allok)
