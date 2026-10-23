# ==============================================================================
# ARQUIVO GERADO AUTOMATICAMENTE -- nao edite aqui.
# Fonte: src/complexity.py  (repositorio Masters Claude)
# Regenerar com: python deliverables/build_deliverable.py
# ==============================================================================

"""
src/complexity.py — Medidas de complexidade estatistica para analise de pesos de redes neurais.

FONTE UNICA DE VERDADE deste repositorio. Nenhum outro arquivo deve redefinir
`lmc_complexity` ou `sample_entropy_1d` — importe daqui.

Implementa:
  - Complexidade estatistica LMC     (Lopez-Ruiz, Mancini & Calbet, 1995)
  - Entropia amostral 1D (SampEn)    (Richman & Moorman, 2000; Pincus, 1991)
  - MSE 1D                           (Costa, Goldberger & Peng, 2002/2005)
  - Localizacao e achatamento da CAMADA DENSA (ver secao "Camada densa" abaixo)

Alinhado com `thesis-complexity-nn/src/complexity.py` — as convencoes numericas
sao identicas, para que os numeros da dissertacao e deste repositorio batam.

Convencoes padrao (validadas pelo grupo de pesquisa):
  - SampEn: m=2, r = 0.2 x SD(serie, ddof=0)
  - LMC:    entropia de Shannon em bits (log2), produto NAO normalizado H x D,
            bins = min(100, max(10, n // 5))
  - MSE:    max_scale=10, com r fixo calculado da serie original (tau=1)

Dependencias: apenas numpy e scipy (sem torch), para que este arquivo possa ser
entregue de forma autocontida.

--------------------------------------------------------------------------------
Camada densa: o "caminho" e a ordem do achatamento
--------------------------------------------------------------------------------
Seja uma camada densa com M entradas x_1..x_M e N neuronios n_1..n_N. A matriz de
pesos e W com forma [N, M], e w_ji e o peso do caminho x_i -> n_j. No PyTorch,
`nn.Linear.weight` ja tem exatamente essa forma [out_features, in_features].

Duas ordens de percurso sao possiveis ao achatar W em um vetor:

  (A) "n_major" — por neuronio  (n1x1, n1x2, ..., n1xM, n2x1, ...)
      Le todos os pesos que CHEGAM em n1, depois em n2, etc.
      Equivale a W.reshape(-1) — o achatamento nativo do PyTorch.

  (B) "x_major" — por entrada   (x1n1, x1n2, ..., x1nN, x2n1, ...)
      Le todos os pesos que SAEM de x1, depois de x2, etc.
      Equivale a W.T.reshape(-1).

Consequencia importante:
  - A LMC e INVARIANTE a ordem (so usa o histograma dos valores):
        lmc(v_A) == lmc(v_B)  exatamente.
  - A SampEn NAO e invariante (compara janelas consecutivas):
        sampen(v_A) != sampen(v_B), e a diferenca carrega significado.

Por isso `flatten_dense` expoe a ordem como parametro explicito, com default
"n_major" (convencao adotada no projeto).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from scipy.spatial.distance import cdist

__all__ = [
    # LMC
    "compute_normalized_histogram",
    "shannon_entropy_from_hist",
    "disequilibrium_from_hist",
    "lmc_complexity",
    # SampEn
    "sample_entropy_1d",
    "mse_1d",
    # Camada densa
    "FLATTEN_ORDERS",
    "flatten_dense",
    "find_dense_layers",
    "dense_complexity",
]

FLATTEN_ORDERS = ("n_major", "x_major")


# ============================================================================
# Helpers privados
# ============================================================================

def _to_numpy(x) -> np.ndarray:
    """Converte tensor torch / array / lista para np.ndarray float64, sem importar torch."""
    if hasattr(x, "detach"):          # torch.Tensor (duck typing)
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def _coarse_grain_1d(x: np.ndarray, tau: int) -> np.ndarray:
    """Granulacao grossa 1D: media em blocos nao sobrepostos de tamanho tau."""
    if tau == 1:
        return x
    n_trim = (len(x) // tau) * tau
    return x[:n_trim].reshape(-1, tau).mean(axis=1)


def _chebyshev_match_count(windows: np.ndarray, r: float) -> int:
    """
    Conta pares ordenados (i, j), i != j, com distancia de Chebyshev <= r.

    windows : array (N, d), cada linha e um template achatado.
    Retorna o dobro do numero de pares nao ordenados — o fator 2 se cancela na razao A/B.
    """
    dists = cdist(windows.astype(np.float32), windows.astype(np.float32), metric="chebyshev")
    matches = dists <= r
    np.fill_diagonal(matches, False)
    return int(matches.sum())


# ============================================================================
# Complexidade estatistica LMC
# ============================================================================

def compute_normalized_histogram(data, bins: Optional[int] = None) -> np.ndarray:
    """
    Histograma normalizado (distribuicao de probabilidade p) dos valores de `data`.

    Se `bins` for None, usa a regra adaptativa validada pelo grupo:
        bins = min(100, max(10, n // 5))
    """
    flat = _to_numpy(data).ravel()
    n = len(flat)
    if bins is None:
        bins = min(100, max(10, n // 5))
    counts, _ = np.histogram(flat, bins=bins)
    total = counts.sum()
    if total == 0:
        return np.zeros(bins, dtype=np.float64)
    return counts / float(total)


def shannon_entropy_from_hist(hist: np.ndarray) -> float:
    """Entropia de Shannon em bits (log2) a partir de uma distribuicao ja normalizada."""
    p = np.asarray(hist, dtype=np.float64)
    nz = p > 0
    # o "+ 0.0" evita devolver -0.0 no caso degenerado (distribuicao delta),
    # que apareceria como "-0.00000" nos CSVs e graficos
    return float(-np.sum(p[nz] * np.log2(p[nz])) + 0.0)


def disequilibrium_from_hist(hist: np.ndarray) -> float:
    """
    Desequilibrio D = soma((p_i - 1/N)^2): distancia L2 (ao quadrado) da
    distribuicao uniforme. Zero quando p e exatamente uniforme.
    """
    p = np.asarray(hist, dtype=np.float64)
    N = len(p)
    if N == 0:
        return 0.0
    return float(np.sum((p - 1.0 / N) ** 2))


def lmc_complexity(weights, n_bins: Optional[int] = None) -> Dict[str, float]:
    """
    Complexidade estatistica LMC:  C = H x D

        H = -soma(p_i * log2(p_i))      entropia de Shannon, em bits
        D =  soma((p_i - 1/N)^2)        desequilibrio (distancia L2 da uniforme)
        C =  H x D                      produto NAO normalizado

    C e baixa tanto no "cristal" (ordem total: H ~ 0) quanto no "gas"
    (desordem total: D ~ 0), e alta em estruturas intermediarias.

    ATENCAO: H e D NAO sao normalizados para [0, 1]. A escala de C depende do
    numero de bins e da distribuicao dos pesos. Ao comparar entre camadas, epocas
    ou runs, use SEMPRE o mesmo `n_bins`. (Para a camada densa de 3.072 pesos a
    regra adaptativa ja resulta em 100 bins fixos.)

    Parametros
    ----------
    weights : array-like (qualquer forma; e achatado internamente)
    n_bins  : numero de bins; se None, usa min(100, max(10, n // 5))

    Retorna
    -------
    dict com as chaves:
        entropy        : H (bits)
        disequilibrium : D
        complexity     : C = H x D
        n_bins         : numero de bins efetivamente usado
    """
    flat = _to_numpy(weights).ravel()
    n = len(flat)
    if n_bins is None:
        n_bins = min(100, max(10, n // 5))

    p = compute_normalized_histogram(flat, bins=n_bins)
    if p.sum() == 0:
        return dict(entropy=0.0, disequilibrium=0.0, complexity=0.0, n_bins=float(n_bins))

    H = shannon_entropy_from_hist(p)
    D = disequilibrium_from_hist(p)
    return dict(entropy=H, disequilibrium=D, complexity=H * D, n_bins=float(n_bins))


# ============================================================================
# Entropia amostral 1D  (Richman & Moorman, 2000)
# ============================================================================

def sample_entropy_1d(
    series,
    m: int = 2,
    r: Optional[float] = None,
    r_scale: float = 0.2,
) -> float:
    """
    Entropia amostral 1D:  SampEn(m, r) = -ln(A / B)

        B = #{pares ordenados (i,j), i != j, com dist_Chebyshev entre templates
              de comprimento m <= r}
        A = idem, para templates de comprimento m+1

    Ambos os contadores usam o MESMO conjunto de N-m posicoes iniciais, que e a
    convencao padrao de Richman & Moorman.

    Interpretacao: 0 = perfeitamente regular/previsivel; valores altos =
    irregular/imprevisivel.

    IMPORTANTE: a SampEn depende da ORDEM da serie. Ao aplica-la a uma matriz de
    pesos, a ordem do achatamento e uma escolha metodologica — ver `flatten_dense`.

    Parametros
    ----------
    series  : array 1D de comprimento >= m+2
    m       : dimensao de imersao (default 2)
    r       : tolerancia absoluta; se None, usa r_scale * SD(series, ddof=0)
    r_scale : fracao do desvio padrao (default 0.2)

    Retorna
    -------
    float — np.nan se a serie for curta demais ou se B = 0 (indefinido);
            np.inf se B > 0 mas A = 0 (irregularidade maxima);
            0.0 se a serie for constante.
    """
    x = _to_numpy(series).ravel()
    N = len(x)
    if N < m + 2:
        return np.nan

    if r is None:
        sd = float(np.std(x))  # ddof=0
        if sd == 0.0:
            return 0.0
        r = r_scale * sd

    n_tpl = N - m

    def _template_matrix(length: int) -> np.ndarray:
        idx = np.arange(length)[None, :] + np.arange(n_tpl)[:, None]
        return x[idx]

    B = _chebyshev_match_count(_template_matrix(m), r)
    A = _chebyshev_match_count(_template_matrix(m + 1), r)

    if B == 0:
        return np.nan
    if A == 0:
        return np.inf
    # "+ 0.0" evita -0.0 quando A == B (serie perfeitamente previsivel, p.ex. rampa linear)
    return float(-np.log(A / B) + 0.0)


# ============================================================================
# Multiscale entropy (MSE)
# ============================================================================

def mse_1d(series, max_scale: int = 10, m: int = 2, r_scale: float = 0.2) -> np.ndarray:
    """
    MSE 1D (Costa et al., 2002/2005). Para cada escala tau = 1..max_scale:
      1. granulacao grossa (media em blocos de tau amostras);
      2. SampEn da serie granulada, com r FIXO calculado da serie original.
    """
    x = _to_numpy(series).ravel()
    sd = float(np.std(x))
    if sd == 0.0:
        return np.zeros(max_scale)
    r = r_scale * sd

    result = np.empty(max_scale)
    for tau in range(1, max_scale + 1):
        result[tau - 1] = sample_entropy_1d(_coarse_grain_1d(x, tau), m=m, r=r)
    return result


# ============================================================================
# Camada densa: localizacao e achatamento
# ============================================================================

def flatten_dense(weight, order: str = "n_major") -> np.ndarray:
    """
    Achata a matriz de pesos de uma camada densa em um vetor 1D, na ordem pedida.

    Seja W [N, M] com w_ji = peso do caminho x_i -> n_j (forma nativa de
    `nn.Linear.weight`, isto e, [out_features, in_features]).

      order="n_major" (default) -> (w_11, w_12, ..., w_1M, w_21, ...)
          "por neuronio": n1x1, n1x2, ... — equivale a W.reshape(-1)

      order="x_major"           -> (w_11, w_21, ..., w_N1, w_12, ...)
          "por entrada": x1n1, x1n2, ... — equivale a W.T.reshape(-1)

    Os dois vetores sao permutacoes um do outro: a LMC e identica, a SampEn nao.

    Levanta ValueError se `weight` nao for 2D ou se `order` for desconhecida.
    """
    W = _to_numpy(weight)
    if W.ndim != 2:
        raise ValueError(
            f"flatten_dense espera a matriz 2D de uma camada densa, recebi forma {W.shape}"
        )
    if order == "n_major":
        return np.ascontiguousarray(W.reshape(-1))
    if order == "x_major":
        return np.ascontiguousarray(W.T.reshape(-1))
    raise ValueError(f"order desconhecida: {order!r}. Use uma de {FLATTEN_ORDERS}.")


def find_dense_layers(
    state_dict,
    param_types: Optional[Dict[str, str]] = None,
    include_bias: bool = False,
) -> List[str]:
    """
    Localiza os pesos das camadas densas em um state_dict.

    Criterio (todos precisam valer):
      - o tensor tem exatamente 2 dimensoes  [out_features, in_features];
      - o nome termina em ".weight" (ou e "weight");
      - se `param_types` for fornecido, param_types[nome] == "Linear".

    O bias e EXCLUIDO por padrao: o projeto analisa *caminhos* entre neuronios,
    e o bias nao e um caminho. Passe include_bias=True para relaxar (o bias e 1D
    e nao passa no criterio de 2D, entao isso so afeta a listagem).

    Retorna a lista de nomes, na ordem do state_dict. Nunca concatena camadas
    distintas — cada camada densa deve ser analisada separadamente.
    """
    names: List[str] = []
    for name, tensor in state_dict.items():
        if not hasattr(tensor, "ndim"):
            continue
        if tensor.ndim != 2:
            continue
        if not (name == "weight" or name.endswith(".weight")):
            if not include_bias:
                continue
        if param_types is not None and param_types.get(name) != "Linear":
            continue
        names.append(name)
    return names


def dense_complexity(
    weight,
    order: str = "n_major",
    n_bins: Optional[int] = None,
    m: int = 2,
    r_scale: float = 0.2,
) -> Dict[str, float]:
    """
    Conveniencia: achata a camada densa na ordem pedida e devolve os DOIS
    indicadores do projeto de uma vez.

    Retorna dict com: lmc, entropy, disequilibrium, sampen, n_weights, order.
    Valores nao finitos da SampEn (inf de irregularidade maxima) sao convertidos
    para np.nan, para que os consumidores possam usar dropna() com seguranca.
    """
    v = flatten_dense(weight, order=order)
    lmc = lmc_complexity(v, n_bins=n_bins)
    se = sample_entropy_1d(v, m=m, r_scale=r_scale)
    if not np.isfinite(se):
        se = np.nan
    return dict(
        lmc=lmc["complexity"],
        entropy=lmc["entropy"],
        disequilibrium=lmc["disequilibrium"],
        sampen=float(se),
        n_weights=float(v.size),
        order=order,
    )
