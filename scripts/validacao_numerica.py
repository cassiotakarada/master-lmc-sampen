"""
scripts/validacao_numerica.py — Validação numérica das medidas de complexidade
contra casos analiticamente conhecidos (objetivo específico 1 da dissertação).

Exercita src/complexity.py em distribuições/séries cujo valor esperado é conhecido
em forma fechada ou qualitativamente, para verificar a corretude da implementação:

  LMC (C = H × D, H em bits, D = L2 da uniforme, n_bins = N):
    - Uniforme   → H = log2(N),  D ≈ 0,        C ≈ 0
    - Degenerada → H = 0,         D = 1 - 1/N,  C = 0
    - Gaussiana  → 0 < H < log2(N), D > 0,      C > 0 (regime intermediário)

  SampEn 1D (m=2, r=0.2·σ):
    - Constante  → 0 (perfeitamente regular, por definição)
    - Periódica  → ≈ 0 (alta previsibilidade)
    - Ruído branco gaussiano → finito e elevado (alta irregularidade)

Uso:  python scripts/validacao_numerica.py
"""

from __future__ import annotations
import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.complexity import lmc_complexity, sample_entropy_1d

SEED = 42
N_BINS = 100


def banner(t):
    print("\n" + "=" * 72)
    print(t)
    print("=" * 72)


def main():
    rng = np.random.default_rng(SEED)

    # ────────────────────────────────────────────────────────────────────
    banner("LMC — C = H × D  (n_bins = N = %d ; H esperado da uniforme = log2(N) = %.4f)" % (N_BINS, np.log2(N_BINS)))
    print(f"{'caso':<28}{'H (bits)':>12}{'D':>12}{'C':>12}   esperado")
    print("-" * 84)

    # Uniforme: rampa linear → histograma uniforme
    uni = np.linspace(0.0, 1.0, 100_000)
    r = lmc_complexity(uni, n_bins=N_BINS)
    print(f"{'uniforme':<28}{r['entropy']:>12.4f}{r['disequilibrium']:>12.5f}{r['complexity']:>12.5f}   H=log2(N)=%.4f, D≈0, C≈0" % np.log2(N_BINS))

    # Degenerada: todos os valores num único bin
    deg = np.full(100_000, 0.5)
    r = lmc_complexity(deg, n_bins=N_BINS)
    print(f"{'degenerada (constante)':<28}{r['entropy']:>12.4f}{r['disequilibrium']:>12.5f}{r['complexity']:>12.5f}   H=0, D=1-1/N=%.4f, C=0" % (1 - 1 / N_BINS))

    # Gaussiana: regime intermediário
    gau = rng.normal(0.0, 1.0, 100_000)
    r = lmc_complexity(gau, n_bins=N_BINS)
    print(f"{'gaussiana':<28}{r['entropy']:>12.4f}{r['disequilibrium']:>12.5f}{r['complexity']:>12.5f}   0<H<log2(N), D>0, C>0")

    # ────────────────────────────────────────────────────────────────────
    banner("SampEn 1D  (m=2, r=0.2·σ)")
    print(f"{'caso':<28}{'SampEn':>12}   esperado")
    print("-" * 64)

    cst = np.ones(1000)
    print(f"{'constante':<28}{sample_entropy_1d(cst):>12.4f}   = 0 (regular)")

    # Periódica: padrão senoidal repetido
    base = np.sin(np.linspace(0, 2 * np.pi, 20, endpoint=False))
    per = np.tile(base, 50)
    print(f"{'periódica (sin, T=20)':<28}{sample_entropy_1d(per):>12.4f}   ≈ 0 (previsível)")

    noi = rng.normal(0.0, 1.0, 1000)
    print(f"{'ruído branco gaussiano':<28}{sample_entropy_1d(noi):>12.4f}   finito e elevado (irregular)")

    print()


if __name__ == "__main__":
    main()
