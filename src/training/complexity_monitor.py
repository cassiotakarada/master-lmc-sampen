"""
Monitor de complexidade ao vivo (fase F4 do plan.md).

Calcula LMC e SampEn da CAMADA DENSA ao fim de cada epoca e guarda o historico. So
mede e registra: o trabalho e descritivo, entao nao ha detector, limiar nem "status"
do treino (o status online saiu em 2026-10-23 junto com a SampEn2D, que o alimentava).

O que e calculado a cada epoca (sobre TODOS os pesos da densa, sem subamostragem):
  - LMC                 invariante a ordem do achatamento
  - SampEn 1D n_major   ordem "por neuronio":  n1x1, n1x2, ...
  - SampEn 1D x_major   ordem "por entrada":   x1n1, x1n2, ...

Quem decide o corte dos dados (regra D6) e o `overfit_report.json`, calculado por
`overfit.detect_overfit_onset` sobre a trajetoria COMPLETA, ao fim do treino.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch.nn as nn

from ..complexity import flatten_dense, lmc_complexity, sample_entropy_1d
from ..utils.logging import get_logger


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
    """Calcula os indicadores a cada epoca e mantem o historico."""

    def __init__(
        self,
        order: str = "n_major",
        n_bins: int = 100,
        r_scale: float = 0.20,
        calcular_ambas_ordens: bool = True,
        janela: int = 3,
        logger=None,
    ) -> None:
        self.order = order
        self.n_bins = n_bins
        self.r_scale = r_scale
        self.calcular_ambas_ordens = calcular_ambas_ordens
        self.janela = janela              # epocas da inclinacao relativa da LMC
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
                "%d camadas densas encontradas (%s); o monitor usa a mais proxima da saida: %s",
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
        return m

    # ------------------------------------------------------------------ API
    def on_epoch_end(self, model: nn.Module, epoch: int, val_loss: float,
                     val_accuracy: Optional[float] = None) -> Dict:
        """Mede e registra. Devolve o dict da epoca."""
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

        reg["incl_lmc_rel"] = _inclinacao_relativa([h["lmc"] for h in self.historico],
                                                   self.janela)
        return reg

    def resumo_log(self, reg: Dict) -> str:
        """Linha curta para o log da epoca."""
        return f"LMC={reg['lmc']:.5f} | SampEn={reg['sampen']:.4f}"


def _finito(v: float) -> float:
    """inf (irregularidade maxima) vira nan, para os consumidores usarem dropna()."""
    return float(v) if np.isfinite(v) else float("nan")
