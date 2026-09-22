"""
Deteccao do inicio do overfitting (fase F5 do plan.md, antecipada porque a F4 precisa).

Fonte unica desta regra: o monitor ao vivo (F4) e os scripts de analise (F5) consultam
esta funcao. Nao reimplemente o criterio em outro lugar.

TRES EPOCAS DIFERENTES, que e facil confundir
---------------------------------------------
  melhor_epoca    -- epoca de MENOR val_loss em todo o treino. E o ponto de corte:
                     as analises usam as epocas <= melhor_epoca.
  inicio_overfit  -- melhor_epoca + 1: a primeira epoca da regiao descartada.
  epoca_deteccao  -- melhor_epoca + patience: quando um monitor ONLINE teria certeza
                     suficiente para anunciar. Sempre posterior ao inicio, por
                     construcao -- so da para saber que o pico passou depois de passar.

GROKKING
--------
Grokking e quando a validacao fica parada e melhora muito tarde. Um criterio online que
para no primeiro sinal de piora mataria esse caso. Por isso o padrao do projeto e treinar
ate o fim (`--no_early_stop`) e calcular `melhor_epoca` sobre a trajetoria COMPLETA: um
mergulho tardio do val_loss ainda e capturado como o minimo global.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence


def epocas_sem_melhora(val_losses: Sequence[float]) -> int:
    """Quantas epocas se passaram desde o menor val_loss ate agora (uso online)."""
    if not val_losses:
        return 0
    melhor = min(val_losses)
    idx_melhor = list(val_losses).index(melhor)
    return len(val_losses) - 1 - idx_melhor


def detect_overfit_onset(
    val_losses: Sequence[float],
    patience: int = 3,
    delta: float = 0.0,
) -> Dict[str, Optional[float]]:
    """
    Localiza a melhor epoca e, se houve piora sustentada, o inicio do overfitting.

    A piora so e considerada sustentada se as `patience` epocas imediatamente apos a
    melhor ficarem TODAS acima de `melhor_val_loss + delta`. Sem essa exigencia, uma
    unica epoca ruim por azar do sorteio de lotes ja declararia overfitting.

    `delta` existe para nao confundir ruido com piora: aumentos menores que ele nao
    contam. delta=0.0 aceita qualquer aumento.

    Parametros
    ----------
    val_losses : val_loss por epoca, na ordem (indice 0 = epoca 1)
    patience   : epocas consecutivas de piora exigidas para confirmar
    delta      : aumento minimo, em valor absoluto de loss, para contar como piora

    Retorna dict com:
        melhor_epoca      : 1-indexada
        melhor_val_loss
        inicio_overfit    : 1-indexada, ou None se nao confirmado
        epoca_deteccao    : 1-indexada, ou None
        n_epocas          : total de epocas observadas
        n_epocas_usaveis  : quantas ficam ate o corte (= melhor_epoca)
        n_epocas_descartadas
        confirmado        : bool
        motivo            : explicacao em texto, para o relatorio
    """
    if patience < 1:
        raise ValueError(f"patience deve ser >= 1, recebi {patience}")
    vl = [float(v) for v in val_losses]
    n = len(vl)
    if n == 0:
        return dict(melhor_epoca=None, melhor_val_loss=None, inicio_overfit=None,
                    epoca_deteccao=None, n_epocas=0, n_epocas_usaveis=0,
                    n_epocas_descartadas=0, confirmado=False,
                    motivo="nenhuma epoca observada")

    idx_melhor = min(range(n), key=lambda i: vl[i])
    melhor_epoca = idx_melhor + 1
    melhor = vl[idx_melhor]
    posteriores = vl[idx_melhor + 1:]

    base = dict(melhor_epoca=melhor_epoca, melhor_val_loss=melhor,
                n_epocas=n, n_epocas_usaveis=melhor_epoca)

    if len(posteriores) < patience:
        return dict(**base, inicio_overfit=None, epoca_deteccao=None,
                    n_epocas_descartadas=0, confirmado=False,
                    motivo=(f"treino terminou {len(posteriores)} epoca(s) apos a melhor; "
                            f"sao necessarias {patience} para confirmar"))

    if all(v > melhor + delta for v in posteriores[:patience]):
        return dict(**base, inicio_overfit=melhor_epoca + 1,
                    epoca_deteccao=melhor_epoca + patience,
                    n_epocas_descartadas=n - melhor_epoca, confirmado=True,
                    motivo=(f"val_loss ficou acima de {melhor:.6f}+{delta} nas {patience} "
                            f"epocas seguintes a {melhor_epoca}"))

    return dict(**base, inicio_overfit=None, epoca_deteccao=None,
                n_epocas_descartadas=0, confirmado=False,
                motivo=(f"a piora apos a epoca {melhor_epoca} nao se sustentou por "
                        f"{patience} epocas (delta={delta})"))


def epocas_usaveis(val_losses: Sequence[float], patience: int = 3,
                   delta: float = 0.0) -> List[int]:
    """Lista 1-indexada das epocas que as analises podem usar (regra D6 do plan.md).

    Se o overfitting foi confirmado, devolve 1..melhor_epoca. Caso contrario, todas --
    nao ha regiao a descartar.
    """
    r = detect_overfit_onset(val_losses, patience=patience, delta=delta)
    if r["confirmado"]:
        return list(range(1, int(r["melhor_epoca"]) + 1))
    return list(range(1, len(val_losses) + 1))


def detect_degradacao_acuracia(
    val_accs: Sequence[float],
    queda: float = 0.01,
    patience: int = 3,
) -> Dict[str, Optional[float]]:
    """Mesma pergunta de `detect_overfit_onset`, mas ancorada na ACURACIA.

    POR QUE EXISTE
    --------------
    O val_loss e a ancora padrao de early stopping, mas fica fragil quando a metrica
    satura. Medido no run de ruido de rotulo da F6: o minimo do val_loss cai na epoca 4,
    quando a acuracia de validacao ainda era 99,75%% e o modelo estava perfeitamente
    saudavel -- a "piora" eram flutuacoes de terceira casa decimal. A degradacao real
    so comeca na epoca 15. Ancorar no val_loss ali descartaria 56 de 60 epocas por ruido.

    Criterio: primeira epoca a partir da qual a acuracia fica `queda` abaixo do seu pico
    por `patience` epocas CONSECUTIVAS. Exigir persistencia evita que uma unica epoca
    ruim dispare o corte.

    Devolve o mesmo formato de `detect_overfit_onset`, para que os consumidores
    (sombreamento, marcacao de epocas) funcionem sem alteracao.
    """
    va = [float(v) for v in val_accs]
    n = len(va)
    vazio = dict(melhor_epoca=None, melhor_val_loss=None, inicio_overfit=None,
                 epoca_deteccao=None, n_epocas=n, n_epocas_usaveis=n,
                 n_epocas_descartadas=0, confirmado=False, criterio="val_accuracy")
    if n == 0:
        return {**vazio, "motivo": "nenhuma epoca observada"}

    pico = max(va)
    idx_pico = va.index(pico)
    for i in range(n - patience + 1):
        if all(v < pico - queda for v in va[i:i + patience]):
            melhor = i          # ultima epoca antes da degradacao (0-based) -> 1-based = i
            melhor_epoca = max(1, melhor)
            return dict(melhor_epoca=melhor_epoca, melhor_val_loss=None,
                        inicio_overfit=melhor_epoca + 1,
                        epoca_deteccao=melhor_epoca + patience,
                        n_epocas=n, n_epocas_usaveis=melhor_epoca,
                        n_epocas_descartadas=n - melhor_epoca, confirmado=True,
                        criterio="val_accuracy",
                        motivo=(f"acuracia ficou {queda:.3f} abaixo do pico ({pico:.4f}) "
                                f"por {patience} epocas consecutivas a partir da epoca "
                                f"{melhor_epoca + 1}"))
    return {**vazio, "melhor_epoca": idx_pico + 1,
            "motivo": (f"a acuracia nunca ficou {queda:.3f} abaixo do pico por "
                       f"{patience} epocas consecutivas")}
