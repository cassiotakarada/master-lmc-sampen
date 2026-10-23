"""Testes do leitor do PathMNIST com um .npz falso, do mesmo formato do oficial."""
import argparse
import os

import numpy as np
import pytest
import torch

from src.config import TrainConfig
from src.data.image_noise import RuidoNaImagem
from src.data.pathmnist import CLASSES, ArquivoPathMNIST, PathMNISTDataset
from src.data.split import stratified_subset_indices
from src.data.transforms import build_transforms
from src.main_train import NUM_CLASSES_POR_DATASET, merge_config


@pytest.fixture
def npz_falso(tmp_path):
    rng = np.random.default_rng(0)
    caminho = tmp_path / "pathmnist_64.npz"
    dados = {}
    for secao, n in (("train", 90), ("val", 18), ("test", 9)):
        dados[f"{secao}_images"] = rng.integers(0, 256, (n, 64, 64, 3), dtype=np.uint8)
        dados[f"{secao}_labels"] = (np.arange(n) % 9).reshape(-1, 1).astype(np.uint8)
    np.savez(caminho, **dados)
    return str(caminho)


def test_tamanhos_e_classes(npz_falso):
    arq = ArquivoPathMNIST(npz_falso)
    assert [len(PathMNISTDataset(s, arquivo=arq)) for s in ("train", "val", "test")] == [90, 18, 9]
    assert len(CLASSES) == 9 == NUM_CLASSES_POR_DATASET["pathmnist"]


def test_imagem_rgb_preserva_as_cores(npz_falso):
    """O PathMNIST e colorido: os 3 canais NAO podem virar copias do cinza."""
    arq = ArquivoPathMNIST(npz_falso)
    ds = PathMNISTDataset("train", transform=build_transforms(size=(64, 64)), arquivo=arq)
    item = ds[0]
    assert item["image"].shape == (3, 64, 64)
    assert item["label"].dtype == torch.long
    esperado = torch.from_numpy(arq.imagens["train"][0]).permute(2, 0, 1).float() / 255
    assert torch.allclose(item["image"], esperado)
    assert not torch.equal(item["image"][0], item["image"][1])


def test_rotulos_na_ordem_do_arquivo(npz_falso):
    ds = PathMNISTDataset("val", arquivo=ArquivoPathMNIST(npz_falso))
    assert ds.rotulos() == [i % 9 for i in range(18)]
    assert int(ds[5]["label"]) == 5


def test_subamostragem_estratificada_mantem_as_9_classes(npz_falso):
    ds = PathMNISTDataset("train", arquivo=ArquivoPathMNIST(npz_falso))
    idx = stratified_subset_indices(ds.rotulos(), 0.2, seed=0)
    assert len(idx) == 18
    assert sorted({ds.rotulos()[i] for i in idx}) == list(range(9))


def test_ruido_funciona_em_imagem_colorida(npz_falso):
    """Mesmo campo 2D nos 3 canais (decisao de 2026-10-23), fixo por imagem."""
    base = PathMNISTDataset("train", transform=build_transforms(size=(64, 64)),
                            arquivo=ArquivoPathMNIST(npz_falso))
    ruidoso = RuidoNaImagem(base, beta=1.0, sigma=0.1, seed=3)
    a, b = ruidoso[4]["image"], ruidoso[4]["image"]
    assert torch.equal(a, b)
    assert a.shape == (3, 64, 64)
    assert not torch.equal(a, base[4]["image"])


def test_carregar_reaproveita_a_leitura(npz_falso):
    assert ArquivoPathMNIST.carregar(npz_falso) is ArquivoPathMNIST.carregar(npz_falso)


def test_arquivo_ausente_diz_onde_baixar(tmp_path):
    with pytest.raises(FileNotFoundError, match="zenodo"):
        ArquivoPathMNIST(os.path.join(tmp_path, "nao_existe.npz"))


def _args(**kw):
    return argparse.Namespace(**kw)


def test_num_classes_segue_o_dataset():
    assert merge_config(_args(dataset="pathmnist")).num_classes == 9
    assert merge_config(_args()).num_classes == 6
    with pytest.raises(ValueError, match="nao bate"):
        merge_config(_args(dataset="pathmnist", num_classes=6))


def test_padrao_continua_mednist():
    assert TrainConfig().dataset == "mednist"
