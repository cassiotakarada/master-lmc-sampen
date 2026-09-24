"""
Monitor de complexidade ao vivo (fase F4 do plan.md).

Calcula LMC e SampEn da CAMADA DENSA ao fim de cada epoca e traduz as trajetorias em
um STATUS legivel do treinamento. O objetivo pratico, descrito na justificativa do
projeto: dar ao desenvolvedor uma leitura do treino que nao depende de mais dados
rotulados -- os dois indicadores saem so dos pesos.

O que e calculado a cada epoca (sobre TODOS os pesos da densa, sem subamostragem):
  - LMC                 invariante a ordem do achatamento
  - SampEn 1D n_major   ordem "por neuronio":  n1x1, n1x2, ...
  - SampEn 1D x_major   ordem "por entrada":   x1n1, x1n2, ...
  - SampEn 2D           sobre a matriz [neuronios, entradas], sem achatar

POR QUE A SampEn2D USA m=1 (e nao o m=2 padrao)
-----------------------------------------------
Na SampEn2D a janela (m+1)x(m+1) tem (m+1)^2 elementos, e TODOS precisam casar dentro
de r. Com m=2 sao 9 elementos -- e duas vizinhancas 3x3 so casam se a matriz tiver
ESTRUTURA ESPACIAL, isto e, se padroes locais se repetirem.

Pesos de camada densa sao aproximadamente independentes entre si: nao ha vizinhanca,
cada peso liga um par (entrada, neuronio) sem relacao espacial com o vizinho na matriz.
Resultado medido: o contador A zera e a SampEn2D fica indefinida em 10 de 10 epocas de
um run real. Verificado que isso NAO depende do tamanho da matriz -- (32,48), (64,64),
(64,128), (64,256) e (6,512) falham todas igualmente. O que muda o resultado e a
presenca de estrutura: uma matriz de ruido SUAVIZADO da SampEn2D finita (7,00), e um
gradiente suave da 0,0.

Com m=1 as janelas sao 1x1 contra 2x2 (4 elementos), as contagens ficam robustas
(B~7e5, A~1e3) e a trajetoria e utilizavel. Por isso m=1 e o padrao aqui.

Consequencia para a interpretacao: a SampEn2D desta camada mede o quanto vizinhancas
2x2 de pesos se repetem -- nao "textura" no sentido de imagem, que a camada densa nao tem.

O STATUS ONLINE E PROVISORIO E PODE SER RETIRADO
------------------------------------------------
`TrainingStatus.OVERFITTING` aqui significa "ate esta epoca, o val_loss nao melhora ha
`patience` epocas" -- um juizo ONLINE, feito sem conhecer o futuro. Se o val_loss voltar
a cair depois (ruido, ou grokking), a epoca deixa de ser pos-overfitting.

Quem manda no corte dos dados (regra D6) e o `overfit_report.json`, calculado por
`overfit.detect_overfit_onset` sobre a trajetoria COMPLETA, ao fim do treino. E normal e
esperado que o status online tenha dito OVERFITTING em epocas que o relatorio final
considera perfeitamente usaveis. Os dois nao se contradizem: um responde "o que eu sei
agora?", o outro "o que aconteceu de fato?".

>>> AVISO SOBRE OS LIMIARES <<<
Os limiares de `LimiaresStatus` sao PROVISORIOS. Eles so podem ser fixados depois de
olhar trajetorias reais (F6). Ajustar limiar olhando o resultado e depois anunciar que
"funcionou" e circular. O protocolo da F6 e: calibrar em 2 sementes, VERIFICAR na
terceira, e declarar isso no texto. Ate la, trate o status como leitura auxiliar, nao
como resultado.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import numpy as np
import torch.nn as nn

from ..complexity import dense_complexity, flatten_dense, lmc_complexity, sample_entropy_1d, sample_entropy_2d
from ..utils.logging import get_logger
from .overfit import (epoca_de_afastamento, epoca_de_queda_relativa,
                      epocas_abaixo_do_pico, epocas_sem_melhora)


class TrainingStatus(str, Enum):
    """Leitura do estado do treino a partir dos pesos + acuracia de validacao."""
    INICIALIZANDO = "INICIALIZANDO"      # historico curto demais para concluir algo
    APRENDENDO = "APRENDENDO"            # acuracia subindo e pesos ainda se reorganizando
    CONVERGINDO = "CONVERGINDO"          # acuracia subindo, mas a complexidade estabilizou
    ESTAGNADO = "ESTAGNADO"              # acuracia parada e complexidade plana
    ALERTA_OVERFIT = "ALERTA_OVERFIT"    # SampEn2D saiu do plato ANTES de a acuracia cair
    OVERFITTING = "OVERFITTING"          # acuracia abaixo do pico ha `patience` epocas


@dataclass
class LimiaresStatus:
    """Limiares da maquina de estados.

    RECALIBRADOS EM 2026-09-22 PARA A ACURACIA (antes: val_loss). O protocolo declarado
    no plan.md foi seguido: ajustados em 3 sementes de ruido + 3 de controle e
    VERIFICADOS nas 2+2 restantes, que nao participaram do ajuste. Ver
    `scripts/calibrar_status.py`, que refaz a busca e a verificacao a partir dos CSVs.

    Resultado na VERIFICACAO (os runs que a busca nunca viu): 2/2 runs com ruido
    detectados, 0/2 falsos positivos, alerta 5,0 epocas antes da degradacao real.
    Um dos 2 controles alertou, na epoca 31 -- tarde, 21 epocas depois do ultimo alerta
    de um run com ruido.

    Por que a troca: com o val_loss o status dizia OVERFITTING em 10 de 10 runs (51 % das
    epocas ate nos controles saudaveis). O MedNIST satura na 1a epoca, entao "o val_loss
    parou de melhorar" e verdade cedo mesmo com a rede sa. Ver a secao da F6 no plan.md.
    """
    patience: int = 3                    # epocas consecutivas abaixo do pico p/ confirmar
    janela: int = 3                      # epocas da media movel
    queda_acc: float = 0.005             # 0,5 ponto percentual abaixo do pico = degradacao
    melhora_acc_minima: float = 0.001    # ganho de acuracia na janela abaixo disso = "plano"
    var_relativa_estavel: float = 0.02   # |inclinacao|/media abaixo disso = "estavel"
    # Aviso antecipado, sobre a SampEn2D (a medida que a F6 mostrou discriminar: 5,3x de
    # contraste contra 1,0x da LMC, que ocupava este lugar ate 2026-09-22).
    #
    # CRITERIO TROCADO EM 2026-09-24: era "afastou-se k desvios do plato das epocas 1-5";
    # passou a ser "caiu `queda_rel` abaixo do maximo corrente por `queda_p` epocas". O
    # anterior falhava na DenseNet-121 nas duas pontas (ver epoca_de_queda_relativa).
    # Ajustados SO em runs de ResNet-18 (14 com ruido, 8 controles, 3 niveis de ruido) e
    # testados as cegas na DenseNet-121: 3/3 detectados, margem de +17 epocas.
    queda_rel: float = 0.10              # 10 % abaixo do maximo corrente
    queda_p: int = 2                     # epocas consecutivas para confirmar
    # Mantidos para reproduzir a analise anterior (scripts/ablacao_flatten.py, F6). O
    # status NAO os usa mais.
    afast_base: int = 5
    afast_k: float = 5.0
    afast_p: int = 3


def _inclinacao_relativa(serie: List[float], janela: int) -> float:
    """Inclinacao dos ultimos `janela` pontos, normalizada pela media (adimensional).

    Normalizar e essencial: a LMC vive na casa de 0,05 e a SampEn na de 2,2. Sem
    normalizar, o mesmo limiar significaria coisas completamente diferentes nas duas.
    """
    y = [v for v in serie[-janela:] if v is not None and np.isfinite(v)]
    if len(y) < 2:
        return 0.0
    x = np.arange(len(y), dtype=float)
    coef = np.polyfit(x, np.asarray(y, dtype=float), 1)[0]
    media = float(np.mean(np.abs(y)))
    return float(coef / media) if media > 0 else 0.0


def encontrar_camadas_densas(model: nn.Module) -> List[tuple]:
    """Devolve [(nome, modulo)] de cada nn.Linear do modelo, na ordem de definicao.

    Resolve por tipo, nao por nome fixo: funciona com `fc` da ResNet e com
    `class_layers.out` da DenseNet sem precisar de caso especial.
    """
    return [(nome, mod) for nome, mod in model.named_modules() if isinstance(mod, nn.Linear)]


class ComplexityMonitor:
    """Calcula os indicadores a cada epoca e mantem o historico para o status."""

    def __init__(
        self,
        order: str = "n_major",
        n_bins: int = 100,
        r_scale: float = 0.20,
        calcular_2d: bool = True,
        sampen2d_m: int = 1,
        calcular_ambas_ordens: bool = True,
        limiares: Optional[LimiaresStatus] = None,
        logger=None,
    ) -> None:
        self.order = order
        self.n_bins = n_bins
        self.r_scale = r_scale
        self.calcular_2d = calcular_2d
        self.sampen2d_m = sampen2d_m
        self.calcular_ambas_ordens = calcular_ambas_ordens
        self._avisou_2d_indefinida = False
        self._avisou_sem_acuracia = False
        self.limiares = limiares or LimiaresStatus()
        self.logger = logger or get_logger()
        self.historico: List[Dict] = []
        self._nome_densa: Optional[str] = None

    # ------------------------------------------------------------------ calculo
    def medir(self, model: nn.Module) -> Dict[str, float]:
        """Calcula os indicadores da camada densa primaria (a mais proxima da saida)."""
        densas = encontrar_camadas_densas(model)
        if not densas:
            raise ValueError("o modelo nao tem nenhuma camada densa (nn.Linear)")
        if len(densas) > 1 and self._nome_densa is None:
            self.logger.info(
                "%d camadas densas encontradas (%s); o status usa a mais proxima da saida: %s",
                len(densas), [n for n, _ in densas], densas[-1][0],
            )
        nome, modulo = densas[-1]
        self._nome_densa = nome

        W = modulo.weight.detach().cpu().numpy()   # [neuronios, entradas]
        v = flatten_dense(W, order=self.order)

        m: Dict[str, float] = {
            "camada": nome,
            "n_pesos": int(v.size),
            "n_neuronios": int(W.shape[0]),
            "n_entradas": int(W.shape[1]),
        }
        lmc = lmc_complexity(v, n_bins=self.n_bins)
        m["lmc"] = lmc["complexity"]
        m["entropia"] = lmc["entropy"]
        m["desequilibrio"] = lmc["disequilibrium"]
        m["sampen"] = _finito(sample_entropy_1d(v, m=2, r_scale=self.r_scale))

        if self.calcular_ambas_ordens:
            outra = "x_major" if self.order == "n_major" else "n_major"
            m[f"sampen_{outra}"] = _finito(
                sample_entropy_1d(flatten_dense(W, order=outra), m=2, r_scale=self.r_scale)
            )
        if self.calcular_2d:
            # max_dim = maior dimensao => NENHUMA subamostragem da matriz
            bruto = sample_entropy_2d(W, m=self.sampen2d_m, r_scale=self.r_scale,
                                      max_dim=int(max(W.shape)))
            m["sampen2d"] = _finito(bruto)
            if not np.isfinite(bruto) and not self._avisou_2d_indefinida:
                self._avisou_2d_indefinida = True
                self.logger.warning(
                    "SampEn2D indefinida na camada %s %dx%d com m=%d: as janelas "
                    "(m+1)x(m+1) = %d elementos so casam se houver estrutura espacial, e "
                    "pesos de camada densa sao quase independentes. Aumentar a matriz NAO "
                    "resolve. Use sampen2d_m=1 (padrao) ou aumente sampen_r_factor=%.2f.",
                    nome, W.shape[0], W.shape[1], self.sampen2d_m,
                    (self.sampen2d_m + 1) ** 2, self.r_scale,
                )
        return m

    # ------------------------------------------------------------------ status
    def _classificar(self, val_accs: List[float]) -> TrainingStatus:
        """Estado do treino com o que se sabe ate esta epoca -- so olha o passado.

        Ordem das perguntas (a primeira que responder "sim" vence):
          1. ja ha historico?                        -> INICIALIZANDO
          2. a acuracia ja caiu e ficou caida?       -> OVERFITTING   (o fato)
          3. a SampEn2D caiu do seu maximo?           -> ALERTA_OVERFIT (o aviso)
          4. acuracia parada + complexidade parada?  -> ESTAGNADO
          5. acuracia subindo + complexidade parada? -> CONVERGINDO
          6. caso contrario                          -> APRENDENDO

        O 2 vem antes do 3 de proposito: depois que a degradacao ja e fato, chamar de
        "alerta" seria mentir sobre o tempo verbal.
        """
        lim = self.limiares
        if len(self.historico) < max(3, lim.janela) or not val_accs:
            return TrainingStatus.INICIALIZANDO

        if epocas_abaixo_do_pico(val_accs, lim.queda_acc) >= lim.patience:
            return TrainingStatus.OVERFITTING

        # O aviso antecipado sai da SampEn2D -- a unica das tres medidas que discriminou
        # overfitting na F6 (contraste 5,3x contra 1,0x da LMC, que ocupava este lugar
        # ate 2026-09-22). Cai para a LMC apenas se a 2D nao estiver sendo calculada.
        chave = "sampen2d" if self.calcular_2d else "lmc"
        serie = [h[chave] for h in self.historico if np.isfinite(h.get(chave, np.nan))]
        if len(serie) == len(self.historico):
            if epoca_de_queda_relativa(serie, lim.queda_rel, lim.queda_p):
                return TrainingStatus.ALERTA_OVERFIT

        incl_lmc = _inclinacao_relativa([h["lmc"] for h in self.historico], lim.janela)
        complexidade_estavel = abs(incl_lmc) < lim.var_relativa_estavel

        # Acuracia ainda subindo? Ganho ABSOLUTO na janela, em pontos de acuracia --
        # ganho relativo nao serve numa metrica que vive em 0,999, onde todo movimento
        # real e da terceira casa decimal.
        va = val_accs[-lim.janela:]
        acc_melhorando = (va[-1] - va[0]) > lim.melhora_acc_minima

        if complexidade_estavel:
            return (TrainingStatus.CONVERGINDO if acc_melhorando
                    else TrainingStatus.ESTAGNADO)
        return TrainingStatus.APRENDENDO

    # ------------------------------------------------------------------ API
    def on_epoch_end(self, model: nn.Module, epoch: int, val_loss: float,
                     val_accuracy: Optional[float] = None) -> Dict:
        """Mede, classifica e registra. Devolve o dict da epoca.

        `val_accuracy` e o que decide o status desde a recalibracao de 2026-09-22. Sem
        ela o monitor ainda mede tudo, mas nao ha status: mentir dizendo APRENDENDO sem
        base seria pior do que admitir que falta o dado.
        """
        reg = self.medir(model)
        reg["epoch"] = epoch
        reg["val_loss"] = float(val_loss)
        if val_accuracy is not None:
            reg["val_accuracy"] = float(val_accuracy)

        anterior = self.historico[-1] if self.historico else None
        reg["d_lmc"] = reg["lmc"] - anterior["lmc"] if anterior else 0.0
        reg["d_sampen"] = (reg["sampen"] - anterior["sampen"]
                           if anterior and np.isfinite(reg["sampen"]) and np.isfinite(anterior["sampen"])
                           else 0.0)
        self.historico.append(reg)

        val_accs = [h["val_accuracy"] for h in self.historico if "val_accuracy" in h]
        if len(val_accs) == len(self.historico):
            reg["status"] = self._classificar(val_accs).value
        else:
            if not self._avisou_sem_acuracia:
                self._avisou_sem_acuracia = True
                self.logger.warning(
                    "Sem val_accuracy: o status fica indisponivel. Desde a recalibracao "
                    "de 2026-09-22 a maquina de estados e ancorada na acuracia -- com o "
                    "val_loss ela dizia OVERFITTING em 10 de 10 runs da F6. Passe "
                    "val_accuracy em on_epoch_end para reativa-la."
                )
            reg["status"] = TrainingStatus.INICIALIZANDO.value
        reg["incl_lmc_rel"] = _inclinacao_relativa([h["lmc"] for h in self.historico],
                                                   self.limiares.janela)
        return reg

    def resumo_log(self, reg: Dict) -> str:
        """Linha curta para o log da epoca."""
        partes = [f"LMC={reg['lmc']:.5f}", f"SampEn={reg['sampen']:.4f}"]
        if "sampen2d" in reg:
            partes.append(f"SampEn2D={reg['sampen2d']:.4f}")
        partes.append(f"status={reg['status']}")
        return " | ".join(partes)


def _finito(v: float) -> float:
    """inf (irregularidade maxima) vira nan, para os consumidores usarem dropna()."""
    return float(v) if np.isfinite(v) else float("nan")
