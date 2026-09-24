"""Testes do monitor ao vivo e do detector de overfitting (fases F4/F5 do plan.md)."""
import pathlib
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.complexity import sample_entropy_2d  # noqa: E402
from src.training.complexity_monitor import (  # noqa: E402
    ComplexityMonitor,
    LimiaresStatus,
    TrainingStatus,
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

    def test_produz_os_tres_indicadores(self):
        m = ComplexityMonitor(calcular_2d=True, calcular_ambas_ordens=True).medir(ModeloFake())
        for chave in ("lmc", "sampen", "sampen_x_major", "sampen2d"):
            assert chave in m, f"faltou {chave}"
            assert np.isfinite(m[chave]), f"{chave} nao e finito"

    def test_lmc_nao_depende_da_ordem_mas_sampen_sim(self):
        model = ModeloFake()
        a = ComplexityMonitor(order="n_major").medir(model)
        b = ComplexityMonitor(order="x_major").medir(model)
        assert a["lmc"] == b["lmc"]
        assert a["sampen"] != b["sampen"]

    def test_sampen2d_e_igual_nas_duas_ordens(self):
        """SampEn2D opera na matriz: nao existe escolha de ordem para influenciar."""
        model = ModeloFake()
        a = ComplexityMonitor(order="n_major").medir(model)
        b = ComplexityMonitor(order="x_major").medir(model)
        assert a["sampen2d"] == b["sampen2d"]
        assert np.isfinite(a["sampen2d"])

    def test_sampen2d_usa_m1_por_padrao(self):
        """Com m=2 numa densa de 6 linhas a medida fica indefinida -- ver docstring."""
        assert ComplexityMonitor().sampen2d_m == 1

    def test_sampen2d_com_m2_fica_indefinida_nesta_matriz(self):
        """Documenta a limitacao real: janelas 3x3 nao se repetem numa matriz 6x512."""
        m = ComplexityMonitor(sampen2d_m=2).medir(ModeloFake())
        assert np.isnan(m["sampen2d"])

    @pytest.mark.parametrize("n_in,n_out", [(512, 6), (48, 32), (64, 64)])
    def test_m2_fica_indefinida_em_pesos_de_densa(self, n_in, n_out):
        """A causa NAO e o tamanho da matriz: m=2 falha em todas as formas testadas."""
        torch.manual_seed(0)
        m = ComplexityMonitor(sampen2d_m=2).medir(ModeloFake(n_in=n_in, n_out=n_out))
        assert np.isnan(m["sampen2d"])

    def test_m2_funciona_quando_ha_estrutura_espacial(self):
        """A causa real: janelas 3x3 so casam se padroes locais se repetirem.

        Pesos de camada densa sao quase independentes -- cada peso liga um par
        (entrada, neuronio) e nao guarda relacao com o vizinho na matriz -- entao nao
        ha nada que se repita. Numa matriz COM estrutura, m=2 funciona normalmente.
        """
        rng = np.random.default_rng(0)

        # padrao 4x4 ladrilhado: vizinhancas locais se repetem o tempo todo
        estruturada = np.tile(rng.normal(size=(4, 4)), (16, 16))
        assert np.isfinite(sample_entropy_2d(estruturada, m=2, r_scale=0.20, max_dim=64))

        # mesmo tamanho, mesma escala, mas sem estrutura: indefinida
        independente = rng.normal(size=(64, 64))
        assert not np.isfinite(sample_entropy_2d(independente, m=2, r_scale=0.20, max_dim=64))

    def test_pode_desligar_o_2d(self):
        assert "sampen2d" not in ComplexityMonitor(calcular_2d=False).medir(ModeloFake())


# ==========================================================================
# Maquina de estados
# ==========================================================================

class TestStatus:
    def _rodar(self, val_losses, mexer_pesos, limiares=None, val_accs=None):
        """Roda o monitor epoca a epoca.

        `val_accs` e o que decide o status desde a recalibracao de 2026-09-22. Quando
        nao for passado, deriva-se uma acuracia plausivel do val_loss apenas para manter
        os testes legiveis -- acuracia alta quando a perda e baixa.
        """
        torch.manual_seed(0)
        model = ModeloFake()
        mon = ComplexityMonitor(calcular_2d=False, calcular_ambas_ordens=False,
                                limiares=limiares or LimiaresStatus())
        if val_accs is None:
            pior = max(val_losses)
            val_accs = [1.0 - 0.5 * (vl / pior) for vl in val_losses]
        estados = []
        for i, (vl, va) in enumerate(zip(val_losses, val_accs), start=1):
            mexer_pesos(model, i)
            estados.append(mon.on_epoch_end(model, i, vl, va)["status"])
        return estados

    def test_primeiras_epocas_sao_inicializando(self):
        est = self._rodar([1.0, 0.9, 0.8, 0.7, 0.6], lambda m, i: None)
        assert est[0] == est[1] == TrainingStatus.INICIALIZANDO.value

    def test_acuracia_caida_leva_a_overfitting(self):
        """Depois de `patience` epocas abaixo do pico, o status tem de acusar."""
        est = self._rodar([1.0] * 6, lambda m, i: None,
                          val_accs=[0.90, 0.99, 0.95, 0.94, 0.93, 0.92])
        assert est[-1] == TrainingStatus.OVERFITTING.value

    def test_acuracia_saturada_nao_leva_a_overfitting(self):
        """O caso que motivou a recalibracao (F6).

        Com o val_loss, um treino saudavel numa metrica saturada era rotulado
        OVERFITTING em metade das epocas: "a perda parou de melhorar" e verdade cedo
        quando ela ja esta no chao. A acuracia presa em 99,9 %% NAO e degradacao.
        """
        est = self._rodar([0.01, 0.009, 0.011, 0.010, 0.012, 0.011], lambda m, i: None,
                          val_accs=[0.998, 0.999, 0.999, 0.998, 0.999, 0.999])
        assert TrainingStatus.OVERFITTING.value not in est

    def test_pesos_congelados_e_acuracia_plana_dao_estagnado(self):
        est = self._rodar([1.0] * 6, lambda m, i: None,
                          val_accs=[0.90, 0.90, 0.90, 0.90, 0.90, 0.90])
        assert TrainingStatus.ESTAGNADO.value in est

    def test_sem_acuracia_o_status_fica_indisponivel(self):
        """Sem o dado que decide, o monitor admite que nao sabe -- nao chuta."""
        torch.manual_seed(0)
        model = ModeloFake()
        mon = ComplexityMonitor(calcular_2d=False, calcular_ambas_ordens=False)
        est = [mon.on_epoch_end(model, i, 1.0 / i)["status"] for i in range(1, 7)]
        assert set(est) == {TrainingStatus.INICIALIZANDO.value}

    def test_status_sempre_e_valor_valido_do_enum(self):
        validos = {s.value for s in TrainingStatus}
        est = self._rodar([1.0, 0.8, 0.6, 0.5, 0.45, 0.44, 0.43],
                          lambda m, i: m.fc.weight.data.add_(torch.randn_like(m.fc.weight) * 0.01))
        assert set(est) <= validos

    def test_historico_cresce_uma_entrada_por_epoca(self):
        mon = ComplexityMonitor(calcular_2d=False)
        model = ModeloFake()
        for i in range(1, 6):
            mon.on_epoch_end(model, i, 1.0 / i)
        assert len(mon.historico) == 5
        assert [h["epoch"] for h in mon.historico] == [1, 2, 3, 4, 5]

    def test_d_lmc_e_a_diferenca_para_a_epoca_anterior(self):
        mon = ComplexityMonitor(calcular_2d=False)
        model = ModeloFake()
        r1 = mon.on_epoch_end(model, 1, 1.0)
        model.fc.weight.data.add_(torch.randn_like(model.fc.weight) * 0.05)
        r2 = mon.on_epoch_end(model, 2, 0.9)
        assert r1["d_lmc"] == 0.0
        assert r2["d_lmc"] == pytest.approx(r2["lmc"] - r1["lmc"])


class TestLimiares:
    def test_defaults_documentados(self):
        lim = LimiaresStatus()
        assert lim.patience == 3 and lim.janela == 3
        assert lim.var_relativa_estavel == 0.02
        # Valores escolhidos pela calibracao de 2026-09-22 (scripts/calibrar_status.py):
        # ajustados em 3+3 runs e verificados em 2+2 que a busca nunca viu.
        assert lim.queda_acc == 0.005
        assert lim.afast_k == 5.0 and lim.afast_p == 3 and lim.afast_base == 5
        assert not hasattr(lim, "melhora_val_minima"), (
            "o status deixou de ser ancorado no val_loss na recalibracao de 2026-09-22")

    def test_sao_configuraveis(self):
        """A F6 precisa calibra-los; nao podem estar fixos no codigo."""
        mon = ComplexityMonitor(limiares=LimiaresStatus(patience=5, janela=4))
        assert mon.limiares.patience == 5 and mon.limiares.janela == 4
