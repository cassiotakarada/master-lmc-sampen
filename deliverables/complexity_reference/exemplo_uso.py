"""
Exemplo minimo: calcula LMC e SampEn sobre a camada densa de uma ResNet-18 real.

Autocontido -- precisa apenas de numpy e scipy (nao precisa de torch nem do
repositorio). Rode de dentro desta pasta:

    python exemplo_uso.py
"""
import json
import pathlib

import numpy as np

from complexity import (
    dense_complexity,
    flatten_dense,
    lmc_complexity,
    sample_entropy_1d,
)

AQUI = pathlib.Path(__file__).parent


def main() -> None:
    W = np.load(AQUI / "fc_weight_exemplo.npy")
    meta = json.loads((AQUI / "fc_weight_exemplo.json").read_text(encoding="utf-8"))

    print("=" * 74)
    print("CAMADA DENSA ANALISADA")
    print("=" * 74)
    print(f"  origem : {meta['origem']}")
    print(f"  modelo : {meta['modelo']}")
    print(f"  tensor : {meta['parametro']}  forma {tuple(W.shape)}  = {W.size} pesos")
    print(f"  leitura: W[j, i] = peso do caminho  x_i -> n_j")
    print(f"           N = {W.shape[0]} neuronios,  M = {W.shape[1]} entradas")

    print()
    print("=" * 74)
    print("AS DUAS ORDENS DE PERCURSO (achatamento do vetor)")
    print("=" * 74)
    v_n = flatten_dense(W, order="n_major")   # n1x1, n1x2, ...  == W.reshape(-1)
    v_x = flatten_dense(W, order="x_major")   # x1n1, x1n2, ...  == W.T.reshape(-1)

    print(f"  n_major (por neuronio): n1x1, n1x2, n1x3, ...   primeiros 4: {np.round(v_n[:4], 5)}")
    print(f"  x_major (por entrada) : x1n1, x1n2, x1n3, ...   primeiros 4: {np.round(v_x[:4], 5)}")
    print(f"  sao permutacoes um do outro? {np.array_equal(np.sort(v_n), np.sort(v_x))}")

    print()
    print("=" * 74)
    print("OS DOIS INDICADORES")
    print("=" * 74)
    for ordem, v in (("n_major", v_n), ("x_major", v_x)):
        lmc = lmc_complexity(v, n_bins=100)
        se = sample_entropy_1d(v, m=2, r_scale=0.2)
        print(f"  [{ordem}]")
        print(f"      entropia de Shannon  H = {lmc['entropy']:.6f} bits")
        print(f"      desequilibrio        D = {lmc['disequilibrium']:.6e}")
        print(f"      complexidade LMC   C=HxD = {lmc['complexity']:.6f}")
        print(f"      entropia amostral  SampEn = {se:.6f}")

    print()
    print("  >>> Note: a LMC e IDENTICA nas duas ordens (so usa o histograma dos")
    print("      valores). A SampEn DIFERE, porque compara janelas consecutivas.")
    print("      Por isso a ordem do achatamento e uma escolha metodologica que")
    print("      precisa ser declarada -- e so afeta a SampEn.")

    print()
    print("=" * 74)
    print("ATALHO: dense_complexity() devolve tudo de uma vez")
    print("=" * 74)
    for k, val in dense_complexity(W, order="n_major").items():
        print(f"  {k:>16} : {val}")


if __name__ == "__main__":
    main()
