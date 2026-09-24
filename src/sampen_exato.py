"""
SampEn 1D exata sobre a populacao completa de pesos, sem subamostragem.

A SampEn precisa CONTAR pares de templates a distancia de Chebyshev <= r,
nao enumera-los.  A contagem sai de uma arvore k-d (scipy cKDTree) com
metrica L-infinito, evitando a matriz N x N da implementacao de referencia.

O valor retornado e IDENTICO ao de src/complexity.sample_entropy_1d --
validado bit a bit em ruido gaussiano, serie periodica, uniforme e pesos.
"""
import numpy as np
from scipy.spatial import cKDTree


def _templates(x, length, n_tpl):
    idx = np.arange(length)[None, :] + np.arange(n_tpl)[:, None]
    return np.ascontiguousarray(x[idx])


def _pares(tpl, r):
    """Pares ordenados (i, j) com i != j a distancia L-inf <= r."""
    tree = cKDTree(tpl)                      # uma arvore so, reutilizada
    return tree.count_neighbors(tree, r, p=np.inf) - len(tpl)


def sampen_exato(series, m=2, r=None, r_scale=0.2, verbose=False):
    x = np.asarray(series, dtype=np.float64).ravel()
    N = len(x)
    if N < m + 2:
        return np.nan
    if r is None:
        sd = float(np.std(x))                # ddof=0, igual a referencia
        if sd == 0.0:
            return 0.0
        r = r_scale * sd

    n_tpl = N - m
    B = _pares(_templates(x, m,     n_tpl), r)
    if verbose:
        print(f"    B contado (n={N:,})", flush=True)
    A = _pares(_templates(x, m + 1, n_tpl), r)

    if B == 0:
        return np.nan
    if A == 0:
        return np.inf
    return float(-np.log(A / B))
