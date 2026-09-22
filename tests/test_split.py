"""Testes do particionamento e da subamostragem (fase F3 do plan.md)."""
import collections
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.config import TrainConfig  # noqa: E402
from src.data.split import resumo_particao, stratified_subset_indices  # noqa: E402


class TestSubamostragemEstratificada:
    @pytest.fixture
    def rotulos(self):
        return [0] * 100 + [1] * 50 + [2] * 10   # desbalanceado de proposito

    def test_e_determinista(self, rotulos):
        a = stratified_subset_indices(rotulos, 0.2, seed=42)
        b = stratified_subset_indices(rotulos, 0.2, seed=42)
        assert a == b, "mesma semente tem de dar exatamente o mesmo subconjunto"

    def test_sementes_diferentes_dao_subconjuntos_diferentes(self, rotulos):
        assert stratified_subset_indices(rotulos, 0.2, 42) != stratified_subset_indices(rotulos, 0.2, 7)

    def test_preserva_a_proporcao_entre_classes(self, rotulos):
        idx = stratified_subset_indices(rotulos, 0.2, seed=0)
        cont = collections.Counter(rotulos[i] for i in idx)
        assert cont[0] == 20 and cont[1] == 10 and cont[2] == 2

    def test_nenhuma_classe_desaparece_com_fracao_minuscula(self, rotulos):
        """O ponto do 'estratificado': sortear sem cuidado zeraria a classe rara."""
        idx = stratified_subset_indices(rotulos, 0.01, seed=0)
        cont = collections.Counter(rotulos[i] for i in idx)
        assert set(cont) == {0, 1, 2}, f"classe sumiu: {cont}"
        assert all(v >= 1 for v in cont.values())

    def test_fracao_um_devolve_tudo(self, rotulos):
        assert stratified_subset_indices(rotulos, 1.0, seed=0) == list(range(len(rotulos)))

    def test_indices_ordenados_e_sem_repeticao(self, rotulos):
        idx = stratified_subset_indices(rotulos, 0.3, seed=1)
        assert idx == sorted(idx)
        assert len(idx) == len(set(idx))

    def test_indices_sao_validos(self, rotulos):
        idx = stratified_subset_indices(rotulos, 0.3, seed=1)
        assert all(0 <= i < len(rotulos) for i in idx)

    @pytest.mark.parametrize("f", [0.0, -0.1, 1.5])
    def test_rejeita_fracao_invalida(self, rotulos, f):
        with pytest.raises(ValueError, match="fraction"):
            stratified_subset_indices(rotulos, f, seed=0)


class TestResumoParticao:
    def test_mostra_contagens_e_percentuais(self):
        txt = resumo_particao(80, 10, 10)
        assert "treino=80" in txt and "80.0%" in txt
        assert "validacao=10" in txt and "teste=10" in txt
        assert "total=100" in txt

    def test_particao_vazia_nao_divide_por_zero(self):
        assert resumo_particao(0, 0, 0) == "particao vazia"


class TestConfigDaParticao:
    def test_val_split_morto_foi_removido(self):
        """P2 do plan.md: `val_split` era declarado e nunca usado."""
        assert not hasattr(TrainConfig(), "val_split")

    def test_tem_os_campos_novos_com_defaults_sensatos(self):
        cfg = TrainConfig()
        assert cfg.data_seed == 0
        assert cfg.val_frac == 0.1
        assert cfg.test_frac == 0.1
        assert cfg.train_fraction == 1.0

    def test_data_seed_e_independente_da_seed_de_treino(self):
        """A particao nao pode mudar quando a semente do treino muda."""
        cfg = TrainConfig()
        cfg.seed = 999
        assert cfg.data_seed == 0, "mudar a semente do treino nao pode mexer na particao"
