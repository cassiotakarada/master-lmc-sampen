"""Testes do ruido colorido nas imagens (src/data/image_noise.py)."""
import numpy as np
import pytest
import torch
from scipy import fft

from src.data.image_noise import CORES, RuidoNaImagem, resolver_beta, ruido_colorido


def _inclinacao_espectral(beta: float, n: int = 128, repeticoes: int = 20) -> float:
    """Inclinacao de log(potencia) contra log(f), media sobre varios sorteios."""
    rng = np.random.default_rng(0)
    pot = np.zeros((n, n))
    for _ in range(repeticoes):
        pot += np.abs(fft.fft2(ruido_colorido(n, n, beta, rng))) ** 2
    f = np.sqrt(fft.fftfreq(n)[:, None] ** 2 + fft.fftfreq(n)[None, :] ** 2)
    sel = (f > 0.02) & (f < 0.45)
    return np.polyfit(np.log(f[sel]), np.log(pot[sel]), 1)[0]


@pytest.mark.parametrize("cor", ["branco", "rosa", "marrom"])
def test_espectro_de_potencia_segue_1_sobre_f_beta(cor):
    beta = CORES[cor]
    assert _inclinacao_espectral(beta) == pytest.approx(-beta, abs=0.1)


@pytest.mark.parametrize("beta", [0.0, 1.0, 2.0])
def test_media_zero_desvio_um(beta):
    z = ruido_colorido(64, 64, beta, np.random.default_rng(3))
    assert z.mean() == pytest.approx(0.0, abs=1e-9)
    assert z.std() == pytest.approx(1.0, rel=1e-9)


def test_rosa_concentra_potencia_nas_baixas_frequencias():
    """Rosa: metade inferior do espectro com mais potencia que no branco."""
    def fracao_baixa(beta):
        pot = np.abs(fft.fft2(ruido_colorido(64, 64, beta, np.random.default_rng(1)))) ** 2
        f = np.sqrt(fft.fftfreq(64)[:, None] ** 2 + fft.fftfreq(64)[None, :] ** 2)
        return pot[f < 0.1].sum() / pot.sum()
    assert fracao_baixa(1.0) > 2 * fracao_baixa(0.0)


class _Base(torch.utils.data.Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, idx):
        return {"image": torch.full((3, 64, 64), 0.5), "label": torch.tensor(idx % 2)}

    def rotulos(self):
        return [i % 2 for i in range(4)]


def test_ruido_fixo_por_imagem_e_diferente_entre_imagens():
    ds = RuidoNaImagem(_Base(), beta=1.0, sigma=0.1, seed=42)
    a1, a2, b = ds[0]["image"], ds[0]["image"], ds[1]["image"]
    assert torch.equal(a1, a2)             # mesma imagem -> mesmo ruido em toda epoca
    assert not torch.equal(a1, b)          # imagens diferentes -> ruidos diferentes


def test_canais_recebem_o_mesmo_campo_e_rotulo_intacto():
    ds = RuidoNaImagem(_Base(), beta=1.0, sigma=0.1, seed=42)
    item = ds[2]
    assert torch.equal(item["image"][0], item["image"][1])
    assert torch.equal(item["image"][0], item["image"][2])
    assert int(item["label"]) == 0
    assert ds.rotulos() == _Base().rotulos()


def test_imagem_fica_em_0_1():
    img = RuidoNaImagem(_Base(), beta=0.0, sigma=2.0, seed=1)[0]["image"]
    assert float(img.min()) >= 0.0 and float(img.max()) <= 1.0


def test_sigma_zero_nao_altera_a_imagem():
    assert torch.equal(RuidoNaImagem(_Base(), beta=1.0, sigma=0.0, seed=1)[0]["image"],
                       _Base()[0]["image"])


def test_resolver_beta():
    assert resolver_beta("rosa") == 1.0
    assert resolver_beta("branco") == 0.0
    assert resolver_beta("rosa", 1.5) == 1.5
    with pytest.raises(ValueError):
        resolver_beta("azul")
