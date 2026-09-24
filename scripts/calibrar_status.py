"""
Recalibracao dos limiares do status para a ACURACIA (§4.2 do plan.md).

POR QUE RECALIBRAR
------------------
A `status_timeline.png` mostrou que a maquina de estados ancorada no `val_loss` nao
discrimina nada: dizia OVERFITTING em 10 de 10 runs da F6, ocupando 51 % das epocas ate
nos 5 controles saudaveis. Nao era bug -- era a definicao: "o val_loss nao melhora ha
`patience` epocas". Num problema que satura na 1a epoca, isso e verdade cedo mesmo com a
rede perfeitamente sa.

O PROTOCOLO (e por que ele importa mais que o resultado)
--------------------------------------------------------
Escolher limiar olhando os 10 runs e depois anunciar que "funcionou" seria circular --
o proprio plan.md avisa disso. Entao:

  CALIBRACAO  3 runs de ruido + 3 controles  -> a busca em grade so enxerga estes
  VERIFICACAO 2 runs de ruido + 2 controles  -> nunca participam da escolha

O numero que vale e o da VERIFICACAO. Se ele desabar, o limiar decorou a calibracao.

COMO O REPLAY FUNCIONA
----------------------
Nao reimplementa a classificacao: instancia o `ComplexityMonitor` de producao, enche o
historico dele com as linhas ja gravadas em `complexity_live.csv` e chama o mesmo
`_classificar`. Se o codigo de producao mudar, este script muda junto -- e o que se mede
aqui e exatamente o que roda no treino.
"""
from __future__ import annotations

import argparse
import itertools
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.training.complexity_monitor import ComplexityMonitor, LimiaresStatus, TrainingStatus
from src.training.overfit import detect_degradacao_acuracia

# Divisao FIXA, escolhida antes de olhar qualquer resultado: as duas sementes mais
# recentes (13 e 23) ficam de fora da calibracao, em ambos os grupos.
VERIFICACAO = ("f6_ruido30_seed13", "f6_ruido30_seed23", "f6_seed13", "f6_seed23")


def carregar(run_dir: str) -> dict | None:
    """Junta complexity_live.csv (indicadores) e history.csv (acuracia)."""
    vivo = os.path.join(run_dir, "complexity_live.csv")
    hist = os.path.join(run_dir, "history.csv")
    if not (os.path.isfile(vivo) and os.path.isfile(hist)):
        return None
    df = pd.read_csv(vivo).sort_values("epoch").reset_index(drop=True)
    h = pd.read_csv(hist)
    if "val_accuracy" not in h.columns:
        return None
    df = df.merge(h[["epoch", "val_accuracy"]], on="epoch", how="left", suffixes=("", "_h"))
    nome = os.path.basename(run_dir.rstrip("/"))
    # Ancora de VERDADE: quando a acuracia de fato degradou, medida sobre a trajetoria
    # completa. E contra ela que a antecedencia tem de ser medida -- nunca contra o
    # proprio rotulo OVERFITTING do status, que se move quando os limiares mudam.
    rel = detect_degradacao_acuracia(df["val_accuracy"].tolist(), queda=0.01, patience=3)
    return {"nome": nome, "df": df, "ruido": "ruido" in nome,
            "verificacao": nome in VERIFICACAO,
            "ancora": rel.get("inicio_overfit") if rel.get("confirmado") else None}


def replay(run: dict, lim: LimiaresStatus) -> list[str]:
    """Reproduz o status epoca a epoca com o classificador DE PRODUCAO."""
    mon = ComplexityMonitor(limiares=lim, calcular_2d=True)
    df = run["df"]
    historico, accs, saida = [], [], []
    for _, linha in df.iterrows():
        reg = {"lmc": float(linha["lmc"]),
               "sampen2d": float(linha.get("sampen2d", np.nan)),
               "epoch": int(linha["epoch"])}
        historico.append(reg)
        accs.append(float(linha["val_accuracy"]))
        mon.historico = historico
        saida.append(mon._classificar(accs).value)
    return saida


def avaliar(runs: list[dict], lim: LimiaresStatus) -> dict:
    """Mede o que interessa: detecta os runs com ruido? poupa os controles? avisa cedo?

    A antecedencia e medida contra a ANCORA REAL (a degradacao observada na trajetoria
    completa), nao contra o rotulo OVERFITTING do proprio status. Medir contra o proprio
    rotulo premiaria limiares frouxos: bastaria atrasar a confirmacao para o numero
    subir, sem que o aviso tivesse chegado um minuto antes.

    `margem` = (primeiro ALERTA entre os controles) - (ultimo ALERTA entre os runs com
    ruido). E ela que diz se existe um limiar de epoca capaz de separar os dois grupos.
    """
    det, falsos = 0, 0
    n_ruido = n_ctrl = 0
    epocas_of, antecedencias = [], []
    alerta_ruido, alerta_ctrl = [], []
    for r in runs:
        st = replay(r, lim)
        of = next((i + 1 for i, s in enumerate(st) if s == TrainingStatus.OVERFITTING.value), None)
        al = next((i + 1 for i, s in enumerate(st) if s == TrainingStatus.ALERTA_OVERFIT.value), None)
        if r["ruido"]:
            n_ruido += 1
            if of:
                det += 1
                epocas_of.append(of)
            if al:
                alerta_ruido.append(al)
                if r["ancora"]:
                    antecedencias.append(r["ancora"] - al)
        else:
            n_ctrl += 1
            if of:
                falsos += 1
            if al:
                alerta_ctrl.append(al)
    margem = (min(alerta_ctrl) - max(alerta_ruido)) if (alerta_ruido and alerta_ctrl) else None
    return {
        "detectados": det, "n_ruido": n_ruido,
        "falsos_positivos": falsos, "n_ctrl": n_ctrl,
        "alerta_em_controles": len(alerta_ctrl),
        "epoca_of_media": float(np.mean(epocas_of)) if epocas_of else None,
        "epoca_alerta_media": float(np.mean(alerta_ruido)) if alerta_ruido else None,
        "antecedencia_media": float(np.mean(antecedencias)) if antecedencias else None,
        "margem": margem,
    }


def busca(calib: list[dict]) -> tuple[LimiaresStatus, list[dict]]:
    """Grade pequena e declarada. Restricao dura primeiro, preferencia depois.

    DURA: detectar TODOS os runs com ruido e NENHUM controle. Um limiar que erra isso
    esta fora, por melhor que sejam os outros numeros -- de nada serve avisar cedo se o
    aviso tambem sai em treino saudavel.

    PREFERENCIA, entre os que passam: alerta mais cedo (antecedencia maior). Empate
    resolvido pelo que confirma o overfitting mais cedo.
    """
    grade = {
        "queda_acc": (0.005, 0.01, 0.02),
        "patience": (2, 3, 4),
        "afast_k": (3.0, 4.0, 5.0),
        "afast_p": (2, 3, 4),
    }
    nomes = list(grade)
    passaram = []
    for combo in itertools.product(*(grade[n] for n in nomes)):
        lim = LimiaresStatus(**dict(zip(nomes, combo)))
        m = avaliar(calib, lim)
        if m["detectados"] != m["n_ruido"] or m["falsos_positivos"]:
            continue
        # O alerta tambem tem de separar: se um controle alerta antes de um run com
        # ruido, nao existe limiar de epoca que salve -- os grupos se misturam.
        if m["margem"] is not None and m["margem"] <= 0:
            continue
        passaram.append({**dict(zip(nomes, combo)), **m})
    if not passaram:
        return LimiaresStatus(), []
    # Preferencia, nesta ordem:
    #   1. margem maior -- e ela que sobrevive a dados novos. `margem is None` significa
    #      que NENHUM controle alertou: separacao perfeita, tratada como margem infinita.
    #   2. aviso mais cedo em relacao a degradacao REAL;
    #   3. confirmacao mais cedo, para desempatar.
    def preferencia(d):
        margem = np.inf if d["margem"] is None else d["margem"]
        return (-margem, -(d["antecedencia_media"] or -99), d["epoca_of_media"] or 99)

    passaram.sort(key=preferencia)
    melhor = passaram[0]
    return LimiaresStatus(**{n: melhor[n] for n in nomes}), passaram


def tabela(runs: list[dict], lim: LimiaresStatus) -> pd.DataFrame:
    linhas = []
    for r in runs:
        st = replay(r, lim)
        of = next((i + 1 for i, s in enumerate(st) if s == TrainingStatus.OVERFITTING.value), None)
        al = next((i + 1 for i, s in enumerate(st) if s == TrainingStatus.ALERTA_OVERFIT.value), None)
        linhas.append({
            "run": r["nome"],
            "grupo": "ruido" if r["ruido"] else "controle",
            "conjunto": "VERIFICACAO" if r["verificacao"] else "calibracao",
            "1o_ALERTA": al, "1o_OVERFITTING": of,
            "ancora_real": r["ancora"],
            "antecedencia": (r["ancora"] - al) if (r["ancora"] and al) else None,
            "frac_OVERFITTING": sum(1 for s in st if s == TrainingStatus.OVERFITTING.value) / len(st),
            "correto": bool(of) == r["ruido"],
        })
    return pd.DataFrame(linhas)


def main() -> int:
    p = argparse.ArgumentParser(description="Recalibracao do status para acuracia")
    p.add_argument("--root", default="monai_weights")
    p.add_argument("--output_dir", default="monai_weights/f6_analise")
    args = p.parse_args()

    alvos = sorted(os.path.join(args.root, d) for d in os.listdir(args.root)
                   if d.startswith("f6_"))
    runs = [r for r in (carregar(a) for a in alvos) if r]
    calib = [r for r in runs if not r["verificacao"]]
    verif = [r for r in runs if r["verificacao"]]
    if not calib or not verif:
        print("Preciso de runs nos dois conjuntos (calibracao e verificacao).")
        return 1

    print("=" * 98)
    print("PROTOCOLO — a busca so enxerga o conjunto de calibracao")
    print("=" * 98)
    print(f"  calibracao ({len(calib)}): " + ", ".join(r["nome"] for r in calib))
    print(f"  VERIFICACAO ({len(verif)}): " + ", ".join(r["nome"] for r in verif))

    lim, passaram = busca(calib)
    print(f"\n{len(passaram)} combinacoes passaram a restricao dura na calibracao "
          f"(detectar 3/3 ruido, 0/3 falsos).")
    if not passaram:
        print("NENHUMA passou -- o criterio por acuracia nao separa estes runs. Pare aqui.")
        return 1
    print(f"escolhida: queda_acc={lim.queda_acc} patience={lim.patience} "
          f"afast_k={lim.afast_k} afast_p={lim.afast_p}")

    print("\n" + "=" * 98)
    print("RESULTADO POR RUN")
    print("=" * 98)
    tab = tabela(runs, lim)
    print(tab.to_string(index=False))

    print("\n" + "=" * 98)
    print("O NUMERO QUE VALE — desempenho no conjunto de VERIFICACAO")
    print("=" * 98)
    for rotulo, sub in (("calibracao", calib), ("VERIFICACAO", verif)):
        m = avaliar(sub, lim)
        ant = f"{m['antecedencia_media']:.1f}" if m["antecedencia_media"] is not None else "-"
        mg = f"{m['margem']:+d}" if m["margem"] is not None else "-"
        print(f"  {rotulo:<12} detectou {m['detectados']}/{m['n_ruido']} runs com ruido | "
              f"{m['falsos_positivos']}/{m['n_ctrl']} falsos OVERFITTING | "
              f"alerta {ant} epocas antes da degradacao real | "
              f"alerta tardio em {m['alerta_em_controles']}/{m['n_ctrl']} controles "
              f"(margem {mg} epocas)")

    os.makedirs(args.output_dir, exist_ok=True)
    saida = os.path.join(args.output_dir, "calibracao_status.csv")
    tab.to_csv(saida, index=False)
    pd.DataFrame(passaram).to_csv(
        os.path.join(args.output_dir, "calibracao_status_grade.csv"), index=False)
    print(f"\nTabela: {saida}")
    print("Grade completa: calibracao_status_grade.csv")
    print("\nPara adotar, edite os padroes de LimiaresStatus em "
          "src/training/complexity_monitor.py.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
