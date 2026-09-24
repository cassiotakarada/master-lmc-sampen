"""Testes da aplicacao do corte por overfitting nas analises (fase F5 do plan.md)."""
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.training.overfit import (  # noqa: E402
    epoca_de_afastamento,
    epoca_de_queda_relativa,
    epocas_abaixo_do_pico,
)
from src.analysis.overfit_cut import (  # noqa: E402
    apenas_usaveis,
    carregar_relatorio,
    marcar_pos_overfit,
    resumo_texto,
    sombrear_regiao_descartada,
)

REL_COM_OVERFIT = {"confirmado": True, "melhor_epoca": 3, "n_epocas": 7,
                   "n_epocas_descartadas": 4, "inicio_overfit": 4}
REL_SEM_OVERFIT = {"confirmado": False, "melhor_epoca": 7, "n_epocas": 7,
                   "n_epocas_descartadas": 0, "inicio_overfit": None}


@pytest.fixture
def df():
    return pd.DataFrame({"epoch": range(1, 8), "LMC": [0.1 * i for i in range(7)]})


class TestMarcacao:
    def test_marca_apenas_epocas_apos_a_melhor(self, df):
        out = marcar_pos_overfit(df, REL_COM_OVERFIT)
        assert out["pos_overfit"].tolist() == [False, False, False, True, True, True, True]

    def test_sem_overfit_confirmado_nada_e_descartado(self, df):
        assert marcar_pos_overfit(df, REL_SEM_OVERFIT)["pos_overfit"].sum() == 0

    def test_sem_relatorio_nada_e_descartado(self, df):
        assert marcar_pos_overfit(df, None)["pos_overfit"].sum() == 0

    def test_nao_muta_o_dataframe_original(self, df):
        marcar_pos_overfit(df, REL_COM_OVERFIT)
        assert "pos_overfit" not in df.columns

    def test_a_melhor_epoca_e_usavel(self, df):
        """O corte e <= melhor_epoca: a propria melhor epoca ENTRA nas conclusoes."""
        out = marcar_pos_overfit(df, REL_COM_OVERFIT)
        assert out.loc[out["epoch"] == 3, "pos_overfit"].item() is False or \
               not out.loc[out["epoch"] == 3, "pos_overfit"].item()

    def test_apenas_usaveis_corta_e_remove_a_coluna(self, df):
        out = apenas_usaveis(df, REL_COM_OVERFIT)
        assert out["epoch"].tolist() == [1, 2, 3]
        assert "pos_overfit" not in out.columns


class TestCarregamento:
    def test_le_o_json_quando_existe(self, tmp_path):
        (tmp_path / "overfit_report.json").write_text(json.dumps(REL_COM_OVERFIT))
        rel = carregar_relatorio(str(tmp_path))
        assert rel["melhor_epoca"] == 3
        assert rel["_origem"] == "overfit_report.json"

    def test_recalcula_do_history_quando_falta_o_json(self, tmp_path):
        """Runs anteriores a F4 nao tem o relatorio; exigir re-treino seria absurdo."""
        pd.DataFrame({"epoch": range(1, 8),
                      "val_loss": [1.0, 0.6, 0.3, 0.4, 0.5, 0.6, 0.7]}).to_csv(
            tmp_path / "history.csv", index=False)
        rel = carregar_relatorio(str(tmp_path))
        assert rel["melhor_epoca"] == 3
        assert rel["confirmado"] is True
        assert rel["_origem"] == "recalculado de history.csv"

    def test_o_json_tem_prioridade_sobre_o_history(self, tmp_path):
        (tmp_path / "overfit_report.json").write_text(
            json.dumps({"confirmado": True, "melhor_epoca": 99}))
        pd.DataFrame({"epoch": [1, 2], "val_loss": [1.0, 0.5]}).to_csv(
            tmp_path / "history.csv", index=False)
        assert carregar_relatorio(str(tmp_path))["melhor_epoca"] == 99

    def test_diretorio_vazio_devolve_none(self, tmp_path):
        assert carregar_relatorio(str(tmp_path)) is None

    def test_history_sem_val_loss_devolve_none(self, tmp_path):
        pd.DataFrame({"epoch": [1, 2]}).to_csv(tmp_path / "history.csv", index=False)
        assert carregar_relatorio(str(tmp_path)) is None


class TestSombreamento:
    def test_sombreia_quando_ha_overfitting(self):
        fig, ax = plt.subplots()
        ax.plot(range(1, 8), range(7))
        assert sombrear_regiao_descartada(ax, REL_COM_OVERFIT) is True
        rotulos = ax.get_legend_handles_labels()[1]   # ja sao strings
        assert any("descartado" in r for r in rotulos)
        assert any("melhor época" in r for r in rotulos)
        plt.close(fig)

    def test_nao_sombreia_run_sem_overfitting(self):
        """Sombrear um run que nunca entrou em overfitting seria mentir sobre o dado."""
        fig, ax = plt.subplots()
        ax.plot(range(1, 8), range(7))
        assert sombrear_regiao_descartada(ax, REL_SEM_OVERFIT) is False
        plt.close(fig)

    def test_nao_quebra_sem_relatorio(self):
        fig, ax = plt.subplots()
        assert sombrear_regiao_descartada(ax, None) is False
        plt.close(fig)


class TestResumo:
    def test_com_overfitting(self):
        t = resumo_texto(REL_COM_OVERFIT)
        assert "melhor época 3" in t and "4 de 7" in t

    def test_sem_overfitting_diz_que_nada_foi_descartado(self):
        assert "nenhuma época descartada" in resumo_texto(REL_SEM_OVERFIT)

    def test_sem_relatorio(self):
        assert resumo_texto(None) == "sem relatório de overfitting"


class TestRunsReais:
    """Verifica contra os runs que existem no repositorio."""

    @pytest.mark.parametrize("run,melhor,confirmado", [
        ("run_mednist_resnet18_64", 20, False),
        ("run_mednist_resnet18_64_seed1", 4, True),
        ("run_mednist_resnet18_64_seed7", 14, True),
    ])
    def test_corte_dos_runs_do_repositorio(self, run, melhor, confirmado):
        rd = pathlib.Path(__file__).resolve().parent.parent / "monai_weights" / run
        if not (rd / "history.csv").is_file():
            pytest.skip(f"run ausente: {run}")
        rel = carregar_relatorio(str(rd))
        assert rel["melhor_epoca"] == melhor
        assert rel["confirmado"] is confirmado


class TestDetectoresOnline:
    """As duas funcoes que a maquina de estados do monitor usa (recalibracao 2026-09-22).

    Elas so podem olhar o passado: sao chamadas durante o treino, epoca a epoca.
    """

    def test_acuracia_sempre_subindo_nunca_acusa(self):
        assert epocas_abaixo_do_pico([0.7, 0.8, 0.9, 0.95, 0.99], queda=0.01) == 0

    def test_conta_epocas_consecutivas_abaixo_do_pico(self):
        # pico 0.99 na 2a epoca; as 3 seguintes ficam >1pp abaixo
        assert epocas_abaixo_do_pico([0.90, 0.99, 0.95, 0.94, 0.93], queda=0.01) == 3

    def test_queda_isolada_nao_acumula(self):
        """Uma epoca ruim no meio nao pode somar com outra la na frente."""
        assert epocas_abaixo_do_pico([0.90, 0.99, 0.95, 0.99, 0.95], queda=0.01) == 1

    def test_oscilacao_na_terceira_casa_nao_conta(self):
        """O caso do MedNIST: acuracia saturada oscila, mas nao degrada."""
        assert epocas_abaixo_do_pico([0.998, 0.999, 0.9985, 0.999, 0.9988],
                                     queda=0.01) == 0

    def test_usa_o_pico_CORRENTE_e_nao_o_global(self):
        """Um pico futuro nao pode influenciar o juizo de uma epoca passada."""
        serie = [0.90, 0.85, 0.84, 0.83, 0.99]
        # ate a 4a epoca o pico corrente e 0.90, e as 3 seguintes estao abaixo dele
        assert epocas_abaixo_do_pico(serie[:4], queda=0.01) == 3
        # a ultima epoca reestabelece o pico e zera a contagem
        assert epocas_abaixo_do_pico(serie, queda=0.01) == 0

    def test_afastamento_sinaliza_saida_do_plato(self):
        serie = [1.0, 1.01, 0.99, 1.0, 1.005] + [2.0] * 5
        assert epoca_de_afastamento(serie, base_epocas=5, k=4.0, p=3) == 6

    def test_afastamento_devolve_none_em_serie_estavel(self):
        rng = np.random.default_rng(0)
        serie = 1.0 + rng.normal(0, 0.01, 40)
        assert epoca_de_afastamento(serie, base_epocas=5, k=4.0, p=3) is None

    def test_afastamento_exige_persistencia(self):
        """Um unico ponto fora da faixa nao e afastamento -- p=3 exige 3 seguidos."""
        serie = [1.0, 1.01, 0.99, 1.0, 1.005, 5.0, 1.0, 1.0, 1.0, 1.0]
        assert epoca_de_afastamento(serie, base_epocas=5, k=4.0, p=3) is None

    def test_afastamento_e_causal(self):
        """A decisao ate a epoca k nao pode mudar por causa do que vem depois."""
        serie = [1.0, 1.01, 0.99, 1.0, 1.005] + [2.0] * 5
        cedo = epoca_de_afastamento(serie[:8], base_epocas=5, k=4.0, p=3)
        tarde = epoca_de_afastamento(serie + [9.0] * 5, base_epocas=5, k=4.0, p=3)
        assert cedo == tarde == 6

    def test_afastamento_sem_historico_suficiente(self):
        assert epoca_de_afastamento([1.0, 1.0, 1.0], base_epocas=5, k=4.0, p=3) is None


class TestQuedaRelativa:
    """Criterio do ALERTA desde o redesenho de 2026-09-24.

    O que se exige dele: ser INDEPENDENTE DE ESCALA. Foi a dependencia da escala (o
    sigma do plato das epocas 1-5, que variava 47x entre runs) que fez o criterio
    anterior quebrar na troca de ResNet-18 para DenseNet-121.
    """

    def test_serie_plana_nunca_alerta(self):
        assert epoca_de_queda_relativa([5.0] * 20, queda_rel=0.10, p=2) is None

    def test_queda_sustentada_alerta(self):
        # cai 20 % a partir da 4a epoca e fica la
        assert epoca_de_queda_relativa([10, 10, 10, 8, 8, 8], queda_rel=0.10, p=2) == 5

    def test_queda_menor_que_o_limiar_nao_alerta(self):
        assert epoca_de_queda_relativa([10, 10, 10, 9.5, 9.5], queda_rel=0.10, p=2) is None

    def test_queda_isolada_nao_alerta(self):
        """Uma epoca ruim nao e declinio: p=2 exige persistencia."""
        assert epoca_de_queda_relativa([10, 10, 8, 10, 10], queda_rel=0.10, p=2) is None

    def test_e_invariante_a_escala(self):
        """O MESMO formato de serie, multiplicado por 1000, alerta na MESMA epoca.

        Este e o teste que o criterio antigo nao passaria de forma util: ele dependia do
        desvio absoluto das primeiras epocas.
        """
        base = [10, 10, 10, 8, 8, 8]
        assert (epoca_de_queda_relativa(base, 0.10, 2)
                == epoca_de_queda_relativa([v * 1000 for v in base], 0.10, 2)
                == epoca_de_queda_relativa([v * 0.001 for v in base], 0.10, 2))

    def test_nao_depende_de_quao_quieto_o_inicio_e(self):
        """Duas series com o MESMO declinio, uma agitada no comeco e outra nao.

        Reproduz em miniatura o que quebrou na DenseNet: la o inicio agitado inflava o
        sigma e o alerta nunca saia, enquanto um inicio quieto demais disparava sozinho.
        """
        quieta = [10.0, 10.0, 10.0, 10.0, 10.0, 8.0, 8.0]
        agitada = [10.0, 11.0, 9.0, 11.0, 10.0, 8.0, 8.0]
        assert (epoca_de_queda_relativa(quieta, 0.10, 2)
                == epoca_de_queda_relativa(agitada, 0.10, 2) == 7)

    def test_usa_o_maximo_CORRENTE(self):
        """Um pico tardio nao pode reclassificar epocas passadas."""
        # sobe ate 20 na 5a epoca; so entao o 10 do inicio ficaria 50% abaixo do pico
        serie = [10, 10, 10, 10, 20, 20]
        assert epoca_de_queda_relativa(serie[:4], 0.10, 2) is None

    def test_ignora_valores_nao_finitos(self):
        assert epoca_de_queda_relativa([10, float("nan"), 10, 8, 8], 0.10, 2) == 5
