"""SampEn 1D com contagem de pares em blocos via cdist: O(N) de memoria, O(N^2) de tempo."""
import numpy as np
from scipy.spatial.distance import cdist

def _count_chunked(win, r, chunk=4096):
    n = win.shape[0]; total = 0
    for i in range(0, n, chunk):
        d = cdist(win[i:i+chunk], win, metric='chebyshev')
        total += int((d <= r).sum())
    return total - n   # remove diagonal

def sampen_chunked(series, m=2, r=None, r_scale=0.2, chunk=4096):
    x = np.asarray(series, dtype=np.float32).ravel()
    N = len(x)
    if N < m + 2: return np.nan
    if r is None:
        sd = float(np.std(x, dtype=np.float64))
        if sd == 0.0: return 0.0
        r = r_scale * sd
    n_tpl = N - m
    def tpl(L):
        idx = np.arange(L)[None, :] + np.arange(n_tpl)[:, None]
        return np.ascontiguousarray(x[idx])
    B = _count_chunked(tpl(m), r, chunk)
    A = _count_chunked(tpl(m + 1), r, chunk)
    if B == 0: return np.nan
    if A == 0: return np.inf
    return float(-np.log(A / B))
