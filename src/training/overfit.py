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

import numpy as np


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


# ---------------------------------------------------------------------------
# Versoes ONLINE, para a maquina de estados do monitor (F4/§4.2)
# ---------------------------------------------------------------------------
# As funcoes acima olham a trajetoria COMPLETA -- sao o veredito final. As duas abaixo
# respondem a pergunta do monitor: "com o que sei ate a epoca k, o que esta acontecendo?".
# Por isso so consultam o passado, nunca o futuro.


def epocas_abaixo_do_pico(val_accs: Sequence[float], queda: float = 0.01) -> int:
    """Ha quantas epocas CONSECUTIVAS a acuracia esta `queda` abaixo do pico ate agora.

    Analogo online de `detect_degradacao_acuracia`. O pico e o maximo ATE cada epoca
    (pico corrente), nao o maximo global -- este ultimo so se conhece no fim do treino.

    Substitui `epocas_sem_melhora(val_losses)` na classificacao de status. Motivo,
    medido nos 10 runs da F6: com o val_loss, o status dizia OVERFITTING em 10 de 10
    runs -- 51 %% das epocas ate nos 5 controles saudaveis -- porque numa metrica que
    satura na 1a epoca "o val_loss parou de melhorar" acontece cedo mesmo com a rede
    perfeitamente sa. Com a acuracia e uma queda de 1 ponto percentual, o mesmo criterio
    dispara em 5 de 5 runs com ruido e em 0 de 5 controles.
    """
    va = [float(v) for v in val_accs if v is not None and np.isfinite(v)]
    if not va:
        return 0
    pico = -np.inf
    seq = 0
    for v in va:
        pico = max(pico, v)
        seq = seq + 1 if v < pico - queda else 0
    return seq


def epoca_de_afastamento(serie: Sequence[float], base_epocas: int = 5, k: float = 4.0,
                         p: int = 3) -> Optional[int]:
    """Primeira epoca em que a serie se afasta, de forma sustentada, do seu plato inicial.

    Por que nao "inversao de sinal": o evento relevante num indicador nao e
    necessariamente um pico. Na SampEn2D do run de ruido, por exemplo, a serie parte de
    um plato e comeca um declinio sustentado -- nao ha inversao nenhuma, e um detector
    de pico nao veria nada.

    Plato = media +- k*desvio das `base_epocas` primeiras epocas. O sinal e a primeira
    epoca a partir da qual a serie fica fora dessa faixa por `p` epocas consecutivas.

    E CAUSAL: cada decisao usa apenas epocas anteriores ou a corrente, entao serve tanto
    para a analise offline (F6) quanto para o status online (F4). Fonte unica -- os
    scripts de analise importam daqui.

    Devolve a epoca (1-indexada) ou None se nunca se afastar -- o resultado esperado num
    run sem overfitting.
    """
    y = np.asarray([float(v) for v in serie], dtype=float)
    if len(y) < base_epocas + p:
        return None
    base = y[:base_epocas]
    mu, sd = float(np.mean(base)), float(np.std(base))
    if sd == 0 or not np.isfinite(sd):
        return None
    fora = np.abs(y - mu) > k * sd
    for i in range(base_epocas, len(y) - p + 1):
        if np.all(fora[i:i + p]):
            return int(i + 1)
    return None


def epoca_de_queda_relativa(serie: Sequence[float], queda_rel: float = 0.10,
                            p: int = 2) -> Optional[int]:
    """Primeira epoca em que a serie fica `queda_rel` abaixo do seu MAXIMO CORRENTE,
    por `p` epocas consecutivas.

    SUBSTITUI `epoca_de_afastamento` COMO CRITERIO DO ALERTA (2026-09-24)
    ---------------------------------------------------------------------
    O criterio anterior media o afastamento em desvios-padrao do plato das 5 primeiras
    epocas. Isso quebrou na DenseNet-121, nas duas pontas:

      - nos runs de ruido, as epocas 1-5 nao eram plato nenhum (a dinamica da DenseNet e
        ~2x mais lenta e a serie ainda se movia la), o sigma inflava, e a faixa k*sigma
        ficava tao larga que o declinio posterior nunca a cruzava -- alertou em 1 de 3;
      - num controle o sigma deu 0,006, quase zero, e qualquer flutuacao virava alerta --
        disparou na epoca 6, ANTES do unico alerta legitimo.

    O sigma do plato variou 47x entre os 12 runs observados. Normalizar por ele e
    normalizar por uma grandeza instavel.

    Aqui a escala e o PROPRIO MAXIMO da serie, que nao depende de quantas epocas o treino
    leva para assentar nem de quao quieto ele comeca. E a mesma forma de
    `epocas_abaixo_do_pico`, usada para a acuracia -- e foi justamente aquele criterio que
    atravessou a troca de arquitetura sem ajuste.

    PRESSUPOSTO: a SampEn2D CAI durante a memorizacao. Verificado nos 12 runs das duas
    arquiteturas (queda de 24-29 % nos runs com ruido contra 5-16 % nos controles). Um
    indicador que subisse precisaria do espelho desta funcao.

    `max(pico, v)` usa o maximo ATE a epoca corrente, nunca o global: a funcao roda
    durante o treino e nao pode enxergar o futuro.

    O QUE ESTE CRITERIO CUSTA, E O TERCEIRO DESENHO QUE FOI DESCARTADO
    ------------------------------------------------------------------
    Ele detecta ACUMULO de queda, entao chega mais tarde que o antigo: epoca 11-14 contra
    7-9 na ResNet. Com 50 % de ruido a antecedencia cai para +1, +1 e -1 epoca -- ou seja,
    nesse regime extremo o aviso deixa de ser aviso. Esta limitacao e real e esta no texto.

    Um terceiro criterio foi medido: "a serie caiu em p epocas consecutivas" (so o SINAL da
    variacao, sem sigma e sem nivel). Ele detecta INICIO em vez de acumulo e avisa bem mais
    cedo -- antecedencia media 8,6 na ResNet (minimo 6) e 24-28 na DenseNet. Foi descartado
    por especificidade: dispara em 7 dos 8 controles da ResNet, com margem de separacao de
    apenas 2 epocas (3 na DenseNet). Margem fina foi exatamente o que quebrou na troca de
    arquitetura da primeira vez; trocar 8 epocas de aviso por uma margem que provavelmente
    nao sobrevive a um dataset novo seria repetir o erro com outro nome.
    """
    pico, seq = -np.inf, 0
    for i, v in enumerate(serie):
        v = float(v)
        if not np.isfinite(v):
            continue
        pico = max(pico, v)
        if pico <= 0 or not np.isfinite(pico):
            continue
        seq = seq + 1 if (pico - v) / abs(pico) >= queda_rel else 0
        if seq >= p:
            return int(i + 1)
    return None
