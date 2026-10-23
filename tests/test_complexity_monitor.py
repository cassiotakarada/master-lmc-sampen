"""Testes do monitor ao vivo e do detector de overfitting (fases F4/F5 do plan.md)."""
import pathlib
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.training.complexity_monitor import (  # noqa: E402
    ComplexityMonitor,
    encontrar_camadas_densas,
)
from src.training.overfit import (  # noqa: E402
    detect_overfit_onset,
    epocas_sem_melhora,
    epocas_usaveis,
)


class ModeloFake(nn.Module):
    """Mesma forma da cabeca da ResNet-18 do projeto: 512 entradas -> 6 neuronios."""

    def __init__(self, n_in=512, n_out=6, extra_linear=False):
        super().__init__()
        self.conv = nn.Conv2d(1, 8, 3)
        self.bn = nn.BatchNorm2d(8)
        if extra_linear:
            self.oculta = nn.Linear(n_in, 128)
            self.fc = nn.Linear(128, n_out)
        else:
            self.fc = nn.Linear(n_in, n_out)


# ==========================================================================
# Deteccao de overfitting
# ==========================================================================

class TestDeteccaoOverfit:
    def test_v_classico(self):
        r = detect_overfit_onset([1.0, 0.6, 0.3, 0.4, 0.5, 0.6, 0.7], patience=3)
        assert r["melhor_epoca"] == 3
        assert r["inicio_overfit"] == 4
        assert r["epoca_deteccao"] == 6
        assert r["confirmado"] is True
        assert r["n_epocas_usaveis"] == 3
        assert r["n_epocas_descartadas"] == 4

    def test_uma_epoca_ruim_isolada_nao_dispara(self):
        """Robustez a ruido: sem patience, um azar no sorteio de lotes declararia overfit."""
        r = detect_overfit_onset([1.0, 0.6, 0.3, 0.4, 0.25, 0.2, 0.15], patience=3)
        assert r["confirmado"] is False
        assert r["melhor_epoca"] == 7

    def test_grokking_mergulho_tardio_nao_e_descartado(self):
        """A validacao fica parada e despenca no fim: nada pode ser cortado."""
        vl = [1.0, 0.9, 0.9, 0.9, 0.9, 0.9, 0.2]
        r = detect_overfit_onset(vl, patience=3)
        assert r["confirmado"] is False
        assert r["melhor_epoca"] == 7
        assert epocas_usaveis(vl) == list(range(1, 8))

    def test_treino_curto_demais_nao_confirma(self):
        r = detect_overfit_onset([1.0, 0.5, 0.6], patience=3)
        assert r["confirmado"] is False
        assert "confirmar" in r["motivo"]

    def test_delta_ignora_piora_dentro_do_ruido(self):
        vl = [1.0, 0.5, 0.501, 0.502, 0.503]
        assert detect_overfit_onset(vl, patience=3, delta=0.0)["confirmado"] is True
        assert detect_overfit_onset(vl, patience=3, delta=0.01)["confirmado"] is False

    def test_epocas_usaveis_corta_no_melhor(self):
        assert epocas_usaveis([1.0, 0.6, 0.3, 0.4, 0.5, 0.6], patience=3) == [1, 2, 3]

    def test_epocas_sem_melhora(self):
        assert epocas_sem_melhora([1.0, 0.5, 0.6, 0.7]) == 2
        assert epocas_sem_melhora([1.0, 0.5, 0.4]) == 0

    def test_lista_vazia_nao_quebra(self):
        assert detect_overfit_onset([])["confirmado"] is False

    def test_patience_invalido(self):
        with pytest.raises(ValueError, match="patience"):
            detect_overfit_onset([1.0, 0.5], patience=0)


# ==========================================================================
# Localizacao da camada densa no modelo vivo
# ==========================================================================

class TestLocalizacaoNoModelo:
    def test_acha_a_linear_ignorando_conv_e_bn(self):
        nomes = [n for n, _ in encontrar_camadas_densas(ModeloFake())]
        assert nomes == ["fc"]

    def test_com_duas_densas_usa_a_mais_proxima_da_saida(self):
        model = ModeloFake(extra_linear=True)
        assert [n for n, _ in encontrar_camadas_densas(model)] == ["oculta", "fc"]
        m = ComplexityMonitor().medir(model)
        assert m["camada"] == "fc", "o status deve usar a cabeca de classificacao"

    def test_modelo_sem_densa_levanta_erro(self):
        with pytest.raises(ValueError, match="nenhuma camada densa"):
            ComplexityMonitor().medir(nn.Sequential(nn.Conv2d(1, 4, 3)))


# ==========================================================================
# Medicao
# ==========================================================================

class TestMedicao:
    def test_usa_todos_os_pesos_sem_subamostrar(self):
        m = ComplexityMonitor().medir(ModeloFake())
        assert m["n_pesos"] == 512 * 6 == 3072
        assert m["n_neuronios"] == 6 and m["n_entradas"] == 512

    def test_produz_os_indicadores(self):
        m = ComplexityMonitor(calcular_ambas_ordens=True).medir(ModeloFake())
        for chave in ("lmc", "sampen", "sampen_x_major"):
            assert chave in m, f"faltou {chave}"
            assert np.isfinite(m[chave]), f"{chave} nao e finito"

    def test_lmc_nao_depende_da_ordem_mas_sampen_sim(self):
        model = ModeloFake()
        a = ComplexityMonitor(order="n_major").medir(model)
        b = ComplexityMonitor(order="x_major").medir(model)
        assert a["lmc"] == b["lmc"]
        assert a["sampen"] != b["sampen"]

# ==========================================================================
# Historico por epoca
# ==========================================================================

class TestHistorico:
    def test_registro_nao_tem_status(self):
        """O monitor so mede: o status online saiu em 2026-10-23 (escopo descritivo)."""
        reg = ComplexityMonitor(calcular_ambas_ordens=False).on_epoch_end(ModeloFake(), 1, 1.0, 0.9)
        assert "status" not in reg
        assert reg["val_accuracy"] == 0.9

    def test_historico_cresce_uma_entrada_por_epoca(self):
        mon = ComplexityMonitor()
        model = ModeloFake()
        for i in range(1, 6):
            mon.on_epoch_end(model, i, 1.0 / i)
        assert len(mon.historico) == 5
        assert [h["epoch"] for h in mon.historico] == [1, 2, 3, 4, 5]

    def test_d_lmc_e_a_diferenca_para_a_epoca_anterior(self):
        mon = ComplexityMonitor()
        model = ModeloFake()
        r1 = mon.on_epoch_end(model, 1, 1.0)
        model.fc.weight.data.add_(torch.randn_like(model.fc.weight) * 0.05)
        r2 = mon.on_epoch_end(model, 2, 0.9)
        assert r1["d_lmc"] == 0.0
        assert r2["d_lmc"] == pytest.approx(r2["lmc"] - r1["lmc"])

    def test_janela_da_inclinacao_e_configuravel(self):
        assert ComplexityMonitor().janela == 3
        assert ComplexityMonitor(janela=4).janela == 4
