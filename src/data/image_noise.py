"""
Ruido colorido nas IMAGENS de treino: branco e rosa (pedido do orientador, 2026-10-15).

COMO O RUIDO E GERADO (dominio da frequencia, com scipy.fft)
-------------------------------------------------------------
1. Sorteia-se ruido branco gaussiano z[y, x] ~ N(0, 1), do tamanho da imagem.
2. Z = FFT2(z). No ruido branco todas as frequencias tem a mesma potencia, em media.
3. Multiplica-se Z pelo filtro radial 1 / f^(beta/2), com f = sqrt(fx^2 + fy^2).
   Como potencia = amplitude^2, o espectro de POTENCIA fica proporcional a 1/f^beta:
       beta = 0  ->  branco  (potencia plana)
       beta = 1  ->  rosa    (potencia ~ 1/f: mais intensidade nas baixas frequencias)
       beta = 2  ->  marrom  (potencia ~ 1/f^2; e o que se obtem multiplicando a
                              AMPLITUDE pela hiperbole 1/f)
   O termo f = 0 (brilho medio) e zerado, para o ruido nao deslocar a imagem toda.
4. z' = Re(IFFT2(Z filtrado)), normalizado para desvio padrao 1.
5. imagem_ruidosa = clip(imagem + sigma * z', 0, 1). As imagens ja chegam em [0, 1]
   (ToTensor), entao `sigma` e a intensidade do ruido na mesma escala do pixel.

FIXO POR IMAGEM
---------------
Cada imagem de treino recebe SEMPRE o mesmo ruido, em todas as epocas: a semente do
sorteio e (noise_seed, indice da imagem). Assim a rede pode decorar o ruido, que e o
regime escolhido. Nada e guardado em memoria -- o ruido e regerado de forma
determinista a cada acesso (uma FFT 64x64 custa microssegundos).

O mesmo campo 2D e somado aos 3 canais. No MedNIST isso e natural (escala de cinza
replicada em RGB, ver src/data/transforms.py); no PathMNIST, que e colorido de verdade,
foi mantido igual por decisao de 2026-10-23, para que o ruido seja o mesmo nos dois
datasets.

SOMENTE O TREINO e afetado. Validacao e teste ficam intactos.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
from scipy import fft
from torch.utils.data import Dataset

# nome do ruido -> expoente beta do espectro de potencia (potencia ~ 1/f^beta)
CORES = {"branco": 0.0, "rosa": 1.0, "marrom": 2.0}


def ruido_colorido(altura: int, largura: int, beta: float,
                   rng: np.random.Generator) -> np.ndarray:
    """Campo 2D com espectro de potencia ~ 1/f^beta, media 0 e desvio padrao 1."""
    branco = rng.standard_normal((altura, largura))
    if beta == 0.0:
        return (branco - branco.mean()) / branco.std()
    espectro = fft.fft2(branco)
    fy = fft.fftfreq(altura)[:, None]
    fx = fft.fftfreq(largura)[None, :]
    f = np.sqrt(fx ** 2 + fy ** 2)
    f[0, 0] = np.inf                      # zera o componente constante (brilho medio)
    espectro = espectro / f ** (beta / 2.0)
    campo = np.real(fft.ifft2(espectro))
    return campo / campo.std()


def resolver_beta(cor: str, beta: Optional[float] = None) -> float:
    """`beta` explicito tem prioridade; senao, o da cor nomeada."""
    if beta is not None:
        return float(beta)
    if cor not in CORES:
        raise ValueError(f"ruido desconhecido {cor!r}; use um de {sorted(CORES)}")
    return CORES[cor]


class RuidoNaImagem(Dataset):
    """Envolve um dataset e soma ruido colorido FIXO a cada imagem; rotulos intactos."""

    def __init__(self, base: Dataset, beta: float, sigma: float, seed: int) -> None:
        if sigma < 0:
            raise ValueError(f"sigma deve ser >= 0, recebi {sigma}")
        self.base = base
        self.beta = float(beta)
        self.sigma = float(sigma)
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.base)

    def ruido(self, idx: int, altura: int, largura: int) -> np.ndarray:
        """O ruido da imagem `idx` -- sempre o mesmo para o mesmo (seed, idx)."""
        rng = np.random.default_rng([self.seed, int(idx)])
        return ruido_colorido(altura, largura, self.beta, rng)

    def __getitem__(self, idx: int) -> Dict:
        item = self.base[idx]
        img = item["image"]                                   # tensor [C, H, W] em [0, 1]
        campo = self.ruido(idx, img.shape[-2], img.shape[-1])
        img = img + self.sigma * torch.from_numpy(campo).to(img.dtype)
        item["image"] = img.clamp_(0.0, 1.0)
        return item

    def rotulos(self):
        return self.base.rotulos()
