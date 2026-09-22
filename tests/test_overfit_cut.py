"""Testes da aplicacao do corte por overfitting nas analises (fase F5 do plan.md)."""
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

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
