"""
Validacao numerica de src/complexity.py (fase F2 do plan.md).

Estrategia: em vez de confiar numa biblioteca externa, este arquivo contem uma
implementacao INGENUA de referencia -- lacos explicitos, escrita diretamente a
partir das definicoes dos artigos originais, sem nenhuma otimizacao -- e verifica
que a implementacao rapida de src/complexity.py concorda com ela.

Isso valida duas coisas de uma vez:
  1. que a formula implementada e a formula publicada;
  2. que as otimizacoes (cdist, stride tricks) nao introduziram erro.

Referencias:
  Lopez-Ruiz, Mancini & Calbet (1995), Phys. Lett. A 209:321-326.
  Richman & Moorman (2000), Am. J. Physiol. 278:H2039-H2049.
"""
import json
import math
import os
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.complexity import (  # noqa: E402
    FLATTEN_ORDERS,
    compute_normalized_histogram,
    dense_complexity,
    disequilibrium_from_hist,
    find_dense_layers,
    flatten_dense,
    lmc_complexity,
    mse_1d,
    sample_entropy_1d,
    sample_entropy_2d,
    shannon_entropy_from_hist,
)

REPO = pathlib.Path(__file__).resolve().parent.parent
CKPT_REAL = REPO / "monai_weights" / "run_mednist_resnet18_64" / "epoch_005.pth"
FC_EXEMPLO = REPO / "deliverables" / "complexity_reference" / "fc_weight_exemplo.npy"


# ==========================================================================
# Implementacoes INGENUAS de referencia (lentas de proposito, O(N^2) explicito)
# ==========================================================================

def lmc_ingenua(data, n_bins):
    """LMC pela definicao, com lacos explicitos. Lopez-Ruiz et al. (1995)."""
    flat = list(np.asarray(data, dtype=np.float64).ravel())
    counts, _ = np.histogram(flat, bins=n_bins)
    total = sum(int(c) for c in counts)
    p = [int(c) / total for c in counts]

    H = 0.0
    for pi in p:
        if pi > 0:
            H -= pi * math.log2(pi)

    D = 0.0
    for pi in p:
        D += (pi - 1.0 / n_bins) ** 2

    return H, D, H * D


def sampen_ingenua(serie, m=2, r=None, r_scale=0.2):
    """
    SampEn pela definicao de Richman & Moorman (2000), com lacos explicitos.

        B = #{pares nao ordenados (i,j), i<j, com d_Chebyshev(templates m)   <= r}
        A = #{pares nao ordenados (i,j), i<j, com d_Chebyshev(templates m+1) <= r}
        SampEn = -ln(A / B)

    Ambos os contadores usam as mesmas N-m posicoes iniciais.
    """
    x = [float(v) for v in np.asarray(serie, dtype=np.float64).ravel()]
    N = len(x)
    if N < m + 2:
        return float("nan")
    if r is None:
        sd = float(np.std(np.array(x)))
        if sd == 0.0:
            return 0.0
        r = r_scale * sd

    n_tpl = N - m

    def chebyshev(i, j, length):
        maior = 0.0
        for k in range(length):
            d = abs(x[i + k] - x[j + k])
            if d > maior:
                maior = d
        return maior

    B = 0
    A = 0
    for i in range(n_tpl):
        for j in range(i + 1, n_tpl):
            if chebyshev(i, j, m) <= r:
                B += 1
            if chebyshev(i, j, m + 1) <= r:
                A += 1

    if B == 0:
        return float("nan")
    if A == 0:
        return float("inf")
    return -math.log(A / B)


# ==========================================================================
# LMC — propriedades fundamentais
# ==========================================================================

class TestLMCPropriedades:
    def test_entropia_de_uniforme_exata_e_log2_nbins(self):
        """H e maxima e vale exatamente log2(N) quando p e perfeitamente uniforme."""
        for n_bins in (10, 50, 100):
            p = np.ones(n_bins) / n_bins
            assert shannon_entropy_from_hist(p) == pytest.approx(math.log2(n_bins))

    def test_desequilibrio_de_uniforme_exata_e_zero(self):
        """D = 0 exatamente quando p e a distribuicao uniforme."""
        for n_bins in (10, 50, 100):
            p = np.ones(n_bins) / n_bins
            assert disequilibrium_from_hist(p) == pytest.approx(0.0, abs=1e-15)

    def test_entropia_de_delta_e_zero(self):
        """H = 0 quando toda a massa esta num unico bin."""
        p = np.zeros(50)
        p[7] = 1.0
        assert shannon_entropy_from_hist(p) == 0.0

    def test_lmc_do_cristal_e_zero(self):
        """'Cristal': todos os pesos iguais -> H = 0 -> C = 0 (e nunca -0.0)."""
        res = lmc_complexity(np.full(5000, 0.3), n_bins=50)
        assert res["entropy"] == 0.0
        assert res["complexity"] == 0.0
        assert not math.copysign(1, res["complexity"]) < 0, "nao deve retornar -0.0"

    def test_lmc_do_gas_e_quase_zero(self):
        """'Gas': pesos uniformemente espalhados -> D ~ 0 -> C ~ 0."""
        rng = np.random.default_rng(0)
        res = lmc_complexity(rng.uniform(-1, 1, 20000), n_bins=50)
        assert res["entropy"] > 0.98 * math.log2(50)   # entropia quase maxima
        assert res["disequilibrium"] < 1e-3            # desequilibrio quase nulo
        assert res["complexity"] < 0.01

    def test_lmc_e_alta_no_regime_intermediario(self):
        """O 'morro': C intermediaria > C do cristal E > C do gas."""
        rng = np.random.default_rng(0)
        c_cristal = lmc_complexity(np.full(20000, 0.3), n_bins=50)["complexity"]
        c_gas = lmc_complexity(rng.uniform(-1, 1, 20000), n_bins=50)["complexity"]
        c_meio = lmc_complexity(rng.normal(0, 1, 20000), n_bins=50)["complexity"]
        assert c_meio > c_cristal
        assert c_meio > c_gas
        assert c_meio > 20 * c_gas, "a separacao deve ser folgada, nao marginal"

    def test_lmc_e_invariante_a_permutacao(self):
        """A LMC so olha o histograma: embaralhar o vetor nao pode mudar nada."""
        rng = np.random.default_rng(1)
        x = rng.normal(size=3072)
        a = lmc_complexity(x, n_bins=100)["complexity"]
        b = lmc_complexity(rng.permutation(x), n_bins=100)["complexity"]
        assert a == b

    def test_histograma_soma_um(self):
        rng = np.random.default_rng(2)
        p = compute_normalized_histogram(rng.normal(size=1000), bins=37)
        assert len(p) == 37
        assert p.sum() == pytest.approx(1.0)

    def test_regra_adaptativa_de_bins(self):
        """bins = min(100, max(10, n // 5)) quando n_bins nao e informado."""
        rng = np.random.default_rng(3)
        assert lmc_complexity(rng.normal(size=20))["n_bins"] == 10      # max(10, 4)
        assert lmc_complexity(rng.normal(size=250))["n_bins"] == 50     # 250 // 5
        assert lmc_complexity(rng.normal(size=3072))["n_bins"] == 100   # min(100, 614)


class TestLMCContraReferencia:
    @pytest.mark.parametrize("n_bins", [10, 50, 100])
    @pytest.mark.parametrize("dist", ["normal", "uniform", "exponential", "bimodal"])
    def test_bate_com_implementacao_ingenua(self, n_bins, dist):
        rng = np.random.default_rng(hash((n_bins, dist)) % 2**32)
        if dist == "normal":
            x = rng.normal(0, 1, 2000)
        elif dist == "uniform":
            x = rng.uniform(-2, 2, 2000)
        elif dist == "exponential":
            x = rng.exponential(1.0, 2000)
        else:
            x = np.concatenate([rng.normal(-3, 0.5, 1000), rng.normal(3, 0.5, 1000)])

        H_ref, D_ref, C_ref = lmc_ingenua(x, n_bins)
        res = lmc_complexity(x, n_bins=n_bins)

        assert res["entropy"] == pytest.approx(H_ref, rel=1e-12)
        assert res["disequilibrium"] == pytest.approx(D_ref, rel=1e-12)
        assert res["complexity"] == pytest.approx(C_ref, rel=1e-12)


# ==========================================================================
# SampEn — propriedades fundamentais
# ==========================================================================

class TestSampEnPropriedades:
    def test_serie_constante_e_zero(self):
        assert sample_entropy_1d(np.ones(200)) == 0.0

    def test_rampa_linear_e_zero(self):
        """Serie perfeitamente previsivel -> SampEn = 0 (e nunca -0.0)."""
        se = sample_entropy_1d(np.linspace(0, 1, 500))
        assert se == pytest.approx(0.0, abs=1e-12)
        assert not math.copysign(1, se) < 0, "nao deve retornar -0.0"

    def test_periodica_e_muito_menor_que_ruido(self):
        """O teste central: sinal regular tem SampEn baixa, ruido tem alta."""
        rng = np.random.default_rng(0)
        N = 500
        seno = np.sin(np.linspace(0, 20 * np.pi, N))
        ruido = rng.normal(size=N)
        se_seno = sample_entropy_1d(seno)
        se_ruido = sample_entropy_1d(ruido)
        assert se_seno < 0.5
        assert se_ruido > 1.8
        assert se_ruido > 4 * se_seno

    def test_ruido_adicionado_aumenta_sampen(self):
        rng = np.random.default_rng(1)
        N = 500
        seno = np.sin(np.linspace(0, 20 * np.pi, N))
        limpo = sample_entropy_1d(seno)
        sujo = sample_entropy_1d(seno + 0.05 * rng.normal(size=N))
        assert sujo > limpo

    def test_tolerancia_maior_reduz_sampen(self):
        """r maior = criterio de 'parecido' mais frouxo = serie parece mais regular."""
        rng = np.random.default_rng(2)
        x = rng.normal(size=400)
        valores = [sample_entropy_1d(x, r_scale=f) for f in (0.1, 0.2, 0.5, 1.0)]
        assert valores == sorted(valores, reverse=True), f"nao monotonico: {valores}"

    def test_serie_curta_retorna_nan(self):
        assert np.isnan(sample_entropy_1d([1.0, 2.0, 3.0], m=2))

    def test_depende_da_ordem(self):
        """Contraponto direto a LMC: embaralhar MUDA a SampEn."""
        rng = np.random.default_rng(3)
        seno = np.sin(np.linspace(0, 20 * np.pi, 500))
        assert sample_entropy_1d(seno) != sample_entropy_1d(rng.permutation(seno))


class TestSampEnContraReferencia:
    @pytest.mark.parametrize("m", [1, 2, 3])
    @pytest.mark.parametrize("sinal", ["normal", "seno", "seno_ruidoso", "uniforme"])
    def test_bate_com_implementacao_ingenua(self, m, sinal):
        rng = np.random.default_rng(hash((m, sinal)) % 2**32)
        N = 220          # pequeno: a referencia ingenua e O(N^2 * m)
        if sinal == "normal":
            x = rng.normal(size=N)
        elif sinal == "seno":
            x = np.sin(np.linspace(0, 8 * np.pi, N))
        elif sinal == "seno_ruidoso":
            x = np.sin(np.linspace(0, 8 * np.pi, N)) + 0.1 * rng.normal(size=N)
        else:
            x = rng.uniform(-1, 1, N)

        ref = sampen_ingenua(x, m=m)
        got = sample_entropy_1d(x, m=m)
        assert got == pytest.approx(ref, rel=1e-6), f"rapida={got} ingenua={ref}"

    def test_bate_com_ingenua_em_pesos_reais(self):
        """Validacao sobre a propria camada densa do projeto, nao so sinais sinteticos."""
        if not FC_EXEMPLO.is_file():
            pytest.skip(f"matriz de exemplo ausente: {FC_EXEMPLO}")
        W = np.load(FC_EXEMPLO)
        v = flatten_dense(W, order="n_major")[:300]   # recorte, pela referencia O(N^2)
        assert sample_entropy_1d(v) == pytest.approx(sampen_ingenua(v), rel=1e-6)

    @pytest.mark.skipif(
        __import__("importlib").util.find_spec("antropy") is None,
        reason="antropy nao instalado (comparacao opcional com biblioteca externa)",
    )
    def test_bate_com_antropy(self):
        import antropy
        rng = np.random.default_rng(0)
        x = rng.normal(size=500)
        r = 0.2 * float(np.std(x))
        assert sample_entropy_1d(x, m=2, r=r) == pytest.approx(
            antropy.sample_entropy(x, order=2, metric="chebyshev"), rel=1e-3
        )


# ==========================================================================
# Camada densa: o "caminho" e a ordem do achatamento
# ==========================================================================

class TestCamadaDensa:
    @pytest.fixture
    def W(self):
        """Matriz [N neuronios, M entradas]; w[j, i] = peso do caminho x_i -> n_j."""
        rng = np.random.default_rng(42)
        return rng.normal(size=(6, 512))

    def test_n_major_e_o_reshape_nativo(self, W):
        """n1x1, n1x2, ... : le todos os pesos que CHEGAM em n1, depois em n2."""
        v = flatten_dense(W, order="n_major")
        assert np.array_equal(v, W.reshape(-1))
        assert np.array_equal(v[:512], W[0, :]), "os 512 primeiros sao o neuronio n1"
        assert v[0] == W[0, 0] and v[1] == W[0, 1]   # n1x1, n1x2

    def test_x_major_e_a_transposta(self, W):
        """x1n1, x1n2, ... : le todos os pesos que SAEM de x1, depois de x2."""
        v = flatten_dense(W, order="x_major")
        assert np.array_equal(v, W.T.reshape(-1))
        assert np.array_equal(v[:6], W[:, 0]), "os 6 primeiros sao a entrada x1"
        assert v[0] == W[0, 0] and v[1] == W[1, 0]   # x1n1, x1n2

    def test_as_duas_ordens_sao_permutacoes(self, W):
        a = flatten_dense(W, "n_major")
        b = flatten_dense(W, "x_major")
        assert a.size == b.size == W.size
        assert np.array_equal(np.sort(a), np.sort(b))
        assert not np.array_equal(a, b), "nao podem ser identicas numa matriz nao quadrada"

    def test_lmc_identica_nas_duas_ordens(self, W):
        """Propriedade central: a LMC nao ve a ordem."""
        a = lmc_complexity(flatten_dense(W, "n_major"), n_bins=100)
        b = lmc_complexity(flatten_dense(W, "x_major"), n_bins=100)
        assert a["complexity"] == b["complexity"]
        assert a["entropy"] == b["entropy"]
        assert a["disequilibrium"] == b["disequilibrium"]

    def test_sampen_difere_entre_as_ordens(self, W):
        """Propriedade central: a SampEn ve a ordem -- por isso ela precisa ser declarada."""
        a = sample_entropy_1d(flatten_dense(W, "n_major"))
        b = sample_entropy_1d(flatten_dense(W, "x_major"))
        assert a != b

    def test_propriedades_valem_em_pesos_reais(self):
        if not FC_EXEMPLO.is_file():
            pytest.skip(f"matriz de exemplo ausente: {FC_EXEMPLO}")
        W = np.load(FC_EXEMPLO)
        assert W.shape == (6, 512)
        a, b = flatten_dense(W, "n_major"), flatten_dense(W, "x_major")
        assert lmc_complexity(a, n_bins=100)["complexity"] == lmc_complexity(b, n_bins=100)["complexity"]
        assert sample_entropy_1d(a) != sample_entropy_1d(b)

    def test_rejeita_matriz_nao_2d(self):
        with pytest.raises(ValueError, match="2D"):
            flatten_dense(np.zeros((2, 3, 4)))
        with pytest.raises(ValueError, match="2D"):
            flatten_dense(np.zeros(10))

    def test_rejeita_ordem_desconhecida(self):
        with pytest.raises(ValueError, match="order desconhecida"):
            flatten_dense(np.zeros((3, 4)), order="row_major")

    def test_ordens_declaradas(self):
        assert FLATTEN_ORDERS == ("n_major", "x_major")

    def test_dense_complexity_devolve_os_dois_indicadores(self, W):
        d = dense_complexity(W, order="n_major")
        assert set(d) == {"lmc", "entropy", "disequilibrium", "sampen", "n_weights", "order"}
        assert d["n_weights"] == 3072
        assert d["order"] == "n_major"
        assert d["lmc"] == pytest.approx(lmc_complexity(flatten_dense(W), n_bins=100)["complexity"])
        assert np.isfinite(d["sampen"])


class TestLocalizacaoDaCamadaDensa:
    def test_encontra_apenas_o_peso_2d_da_linear(self):
        state = {
            "conv1.weight": np.zeros((64, 3, 7, 7)),   # 4D -> conv, fora
            "bn1.weight": np.zeros(64),                # 1D -> batchnorm, fora
            "bn1.bias": np.zeros(64),
            "fc.weight": np.zeros((6, 512)),           # 2D + Linear -> DENTRO
            "fc.bias": np.zeros(6),                    # bias -> fora (nao e um caminho)
        }
        param_types = {
            "conv1.weight": "Convolutional",
            "bn1.weight": "BatchNorm",
            "bn1.bias": "BatchNorm",
            "fc.weight": "Linear",
            "fc.bias": "Linear",
        }
        assert find_dense_layers(state, param_types) == ["fc.weight"]

    def test_encontra_multiplas_densas_sem_concatenar(self):
        state = {
            "classifier.0.weight": np.zeros((128, 512)),
            "classifier.3.weight": np.zeros((6, 128)),
            "conv.weight": np.zeros((8, 3, 3, 3)),
        }
        param_types = {
            "classifier.0.weight": "Linear",
            "classifier.3.weight": "Linear",
            "conv.weight": "Convolutional",
        }
        nomes = find_dense_layers(state, param_types)
        assert nomes == ["classifier.0.weight", "classifier.3.weight"]
        assert len(nomes) == 2, "cada densa e analisada separadamente, nunca concatenada"

    def test_funciona_sem_param_types(self):
        """Sem param_types, o criterio 2D + sufixo .weight ainda isola a densa."""
        state = {
            "conv1.weight": np.zeros((64, 3, 7, 7)),
            "bn1.weight": np.zeros(64),
            "fc.weight": np.zeros((6, 512)),
        }
        assert find_dense_layers(state) == ["fc.weight"]

    def test_encontra_no_checkpoint_real(self):
        if not CKPT_REAL.is_file():
            pytest.skip(f"checkpoint real ausente: {CKPT_REAL}")
        torch = pytest.importorskip("torch")
        sd = torch.load(CKPT_REAL, map_location="cpu", weights_only=True)
        pt_path = CKPT_REAL.parent / "param_types.json"
        param_types = json.loads(pt_path.read_text()) if pt_path.is_file() else None

        nomes = find_dense_layers(sd, param_types)
        assert nomes == ["fc.weight"]
        assert tuple(sd[nomes[0]].shape) == (6, 512)


# ==========================================================================
# Sanidade de 2D e multiescala (usados na F9)
# ==========================================================================

class TestBidimensionalEMultiescala:
    def test_sampen2d_de_matriz_constante_e_zero(self):
        assert sample_entropy_2d(np.ones((32, 32))) == 0.0

    def test_sampen2d_ruido_maior_que_padrao_regular(self):
        rng = np.random.default_rng(0)
        regular = np.tile(np.array([[0.0, 1.0], [1.0, 0.0]]), (16, 16))
        ruido = rng.normal(size=(32, 32))
        assert sample_entropy_2d(ruido) > sample_entropy_2d(regular)

    def test_sampen2d_e_invariante_a_ordem_de_achatamento(self):
        """SampEn2D opera na matriz, entao nao exige escolher n_major/x_major."""
        rng = np.random.default_rng(1)
        W = rng.normal(size=(32, 32))
        assert sample_entropy_2d(W) == sample_entropy_2d(W.T.T)

    def test_sampen2d_rejeita_nao_2d(self):
        with pytest.raises(ValueError, match="2D"):
            sample_entropy_2d(np.zeros(10))

    def test_mse_devolve_um_valor_por_escala(self):
        rng = np.random.default_rng(2)
        vals = mse_1d(rng.normal(size=2000), max_scale=5)
        assert vals.shape == (5,)
        assert np.isfinite(vals).all()

    def test_mse_de_serie_constante_e_zero(self):
        assert np.all(mse_1d(np.ones(1000), max_scale=5) == 0.0)
