"""Core NIST SP 800-22 statistical tests (a self-contained Python implementation of the tests in the
NIST STS suite, used by the SMU-DDI STEER framework) for verifying a binary stream is uncorrupted.
Each test returns a p-value; the stream passes a test at significance 0.01 when p >= 0.01."""
import numpy as np
from scipy.special import erfc, gammaincc
from math import sqrt, log, fabs
from numpy.fft import rfft

def monobit(b):
    n = b.size; s = np.abs((2 * b.astype(np.int64) - 1).sum()) / sqrt(n)
    return float(erfc(s / sqrt(2)))

def block_frequency(b, M=128):
    n = b.size; N = n // M
    pis = b[:N * M].reshape(N, M).mean(1)
    chi2 = 4.0 * M * ((pis - 0.5) ** 2).sum()
    return float(gammaincc(N / 2.0, chi2 / 2.0))

def runs(b):
    n = b.size; pi = b.mean()
    if fabs(pi - 0.5) >= 2.0 / sqrt(n): return 0.0
    v = 1 + int((b[1:] != b[:-1]).sum())
    num = fabs(v - 2.0 * n * pi * (1 - pi))
    return float(erfc(num / (2.0 * sqrt(2.0 * n) * pi * (1 - pi))))

def longest_run(b):
    # n >= 750,000: M=10^4, K=6, classes <=10..>=16
    M = 10000; n = b.size; N = n // M
    pi = [0.0882, 0.2092, 0.2483, 0.1933, 0.1208, 0.0675, 0.0727]
    nu = np.zeros(7)
    blocks = b[:N * M].reshape(N, M)
    for blk in blocks:
        # longest run of ones
        mx = run = 0
        for bit in blk:
            run = run + 1 if bit else 0
            if run > mx: mx = run
        cls = min(max(mx - 10, 0), 6) if mx >= 10 else 0
        cls = 0 if mx <= 10 else (6 if mx >= 16 else mx - 10)
        nu[cls] += 1
    chi2 = (((nu - N * np.array(pi)) ** 2) / (N * np.array(pi))).sum()
    return float(gammaincc(6 / 2.0, chi2 / 2.0))

def spectral(b):
    n = b.size; x = 2.0 * b - 1.0
    mod = np.abs(rfft(x))[:n // 2]
    T = sqrt(log(1.0 / 0.05) * n); N0 = 0.95 * n / 2.0
    N1 = float((mod < T).sum())
    d = (N1 - N0) / sqrt(n * 0.95 * 0.05 / 4.0)
    return float(erfc(fabs(d) / sqrt(2)))

def cusum(b, mode=0):
    x = 2 * b.astype(np.int64) - 1
    s = np.cumsum(x if mode == 0 else x[::-1]); z = int(np.abs(s).max()); n = b.size
    from scipy.stats import norm
    k1 = np.arange((-n // z + 1) // 4, n // z // 4 + 1)
    t1 = (norm.cdf((4 * k1 + 1) * z / sqrt(n)) - norm.cdf((4 * k1 - 1) * z / sqrt(n))).sum()
    k2 = np.arange((-n // z - 3) // 4, n // z // 4 + 1)
    t2 = (norm.cdf((4 * k2 + 3) * z / sqrt(n)) - norm.cdf((4 * k2 + 1) * z / sqrt(n))).sum()
    return float(max(0.0, min(1.0, 1.0 - t1 + t2)))

def _phi(b, m):
    n = b.size
    if m == 0: return 0.0
    # rolling integer of m bits over circular extension
    ext = np.concatenate([b, b[:m - 1]]).astype(np.int64)
    val = np.zeros(n, dtype=np.int64)
    for j in range(m):
        val = (val << 1) | ext[j:j + n]
    counts = np.bincount(val, minlength=1 << m).astype(np.float64)
    c = counts / n
    nz = c[c > 0]
    return float((nz * np.log(nz)).sum())

def approximate_entropy(b, m=10):
    n = b.size; apen = _phi(b, m) - _phi(b, m + 1)
    chi2 = 2.0 * n * (log(2.0) - apen)
    return float(gammaincc(2 ** (m - 1), chi2 / 2.0))

def serial(b, m=12):
    n = b.size
    p0 = _psi2(b, m); p1 = _psi2(b, m - 1); p2 = _psi2(b, m - 2)
    d1 = p0 - p1; d2 = p0 - 2 * p1 + p2
    return float(gammaincc(2 ** (m - 2), d1 / 2.0)), float(gammaincc(2 ** (m - 3), d2 / 2.0))

def _psi2(b, m):
    n = b.size
    if m <= 0: return 0.0
    ext = np.concatenate([b, b[:m - 1]]).astype(np.int64)
    val = np.zeros(n, dtype=np.int64)
    for j in range(m):
        val = (val << 1) | ext[j:j + n]
    counts = np.bincount(val, minlength=1 << m).astype(np.float64)
    return float((counts ** 2).sum() * (1 << m) / n - n)

if __name__ == "__main__":
    b = np.unpackbits(np.load("data/beacon_bytes.npy"))
    print("beacon bitstream: %d bits" % b.size)
    SPECT = b[:1 << 22]                       # 4,194,304 bits for the FFT-based spectral test
    res = [("Frequency (Monobit)", monobit(b)),
           ("Block Frequency (M=128)", block_frequency(b)),
           ("Runs", runs(b)),
           ("Longest Run of Ones", longest_run(b)),
           ("Spectral (DFT)", spectral(SPECT)),
           ("Cumulative Sums (fwd)", cusum(b, 0)),
           ("Cumulative Sums (bwd)", cusum(b, 1)),
           ("Approximate Entropy (m=10)", approximate_entropy(b, 10))]
    s1, s2 = serial(b, 12)
    res += [("Serial 1 (m=12)", s1), ("Serial 2 (m=12)", s2)]
    print("%-30s %12s  %s" % ("test", "p-value", "result"))
    allpass = True
    for name, p in res:
        ok = p >= 0.01; allpass &= ok
        print("%-30s %12.5f  %s" % (name, p, "PASS" if ok else "FAIL"))
    print("\nAll core SP 800-22 tests pass at alpha=0.01:", allpass)
