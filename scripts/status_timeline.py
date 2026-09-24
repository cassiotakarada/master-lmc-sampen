"""
Linha do tempo do status do monitor (figura pendente da F6, §4.2 do plan.md).

O QUE A FIGURA MOSTRA
---------------------
Uma faixa colorida por epoca, sob a curva de `val_loss`, dizendo em que estado o monitor
da F4 classificou o treino naquele momento: INICIALIZANDO, APRENDENDO, CONVERGINDO,
ESTAGNADO, ALERTA_OVERFIT ou OVERFITTING.

POR QUE ELA IMPORTA MAIS DO QUE PARECE
--------------------------------------
O status e um juizo ONLINE: a cada epoca ele responde "com o que sei ate agora, em que
pe esta este treino?". Ele nao conhece o futuro. A figura serve justamente para expor,
lado a lado, o que o monitor DIZIA na epoca k e o que de fato aconteceu depois.

A leitura honesta destes 10 runs e desconfortavel e precisa aparecer no texto: o status
diz OVERFITTING em TODOS os runs, inclusive nos 5 controles saudaveis, ocupando de 45%
a 60% das epocas deles. Isso nao e bug -- e a definicao operacional em uso: OVERFITTING
ali significa "o val_loss nao melhora ha `patience` epocas". Num problema em que a
metrica satura na primeira epoca (MedNIST), ela para de melhorar cedo mesmo com a rede
perfeitamente saudavel. A conclusao e que o STATUS, como esta calibrado, nao serve para
discriminar overfitting -- quem discrimina sao os indicadores (SampEn2D), e e por isso
que o resultado da F6 nao se apoia no status.

Cores: escala de severidade, validada com `validate_palette.js --pairs all` (all-pairs
porque os 10 runs empilhados sao pequenos multiplos, e qualquer par de estados pode ser
comparado a distancia). Legenda sempre presente -- cor nunca carrega sentido sozinha.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from src.training.complexity_monitor import LimiaresStatus

from analise_f6 import GRID, SURFACE, TEXT_1, TEXT_2, marcos_de_degradacao
from calibrar_status import replay as replay_status

# Ordem = severidade crescente. A legenda segue esta ordem, nao a de aparicao.
# Paleta: cinza neutro para "ainda sem historico" + 5 slots validados all-pairs
# (CVD dE 9.1, visao normal 16.3). Aqua e amarelo ficam abaixo de 3:1 contra a
# superficie: a regra de alivio exige rotulo visivel -- daí a legenda obrigatoria.
# Descricoes conforme a recalibracao de 2026-09-22: a maquina de estados passou a ser
# ancorada na ACURACIA, e o aviso antecipado saiu da LMC para a SampEn2D.
STATUS = [
    ("INICIALIZANDO",  "#9a9894", "sem histórico suficiente"),
    ("APRENDENDO",     "#2a78d6", "acurácia subindo, pesos se reorganizando"),
    ("CONVERGINDO",    "#1baf7a", "acurácia subindo, complexidade estabilizou"),
    ("ESTAGNADO",      "#4a3aa7", "acurácia parada e complexidade plana"),
    ("ALERTA_OVERFIT", "#eda100", "SampEn2D saiu do platô ANTES de a acurácia cair"),
    ("OVERFITTING",    "#d03b3b", "acurácia abaixo do pico há `patience` épocas"),
]
COR_STATUS = {nome: cor for nome, cor, _ in STATUS}
TingeOF = "OVERFITTING"   # atalho de leitura nas checagens do subtitulo
GAP = 0.06   # respiro de superficie entre segmentos vizinhos (~2px na escala usada)


def carregar(run_dir: str, fonte: str = "replay") -> dict | None:
    """`fonte="csv"` le o status como foi gravado no treino; `"replay"` recalcula.

    O replay usa o classificador DE PRODUCAO sobre os mesmos dados, entao mostra o que o
    monitor diria HOJE. E o padrao porque os CSVs dos runs da F6 foram gravados antes da
    recalibracao de 2026-09-22 e carregam o status antigo, ancorado no val_loss.
    """
    caminho = os.path.join(run_dir, "complexity_live.csv")
    if not os.path.isfile(caminho):
        return None
    df = pd.read_csv(caminho).sort_values("epoch").reset_index(drop=True)
    hist = os.path.join(run_dir, "history.csv")
    if os.path.isfile(hist):
        h = pd.read_csv(hist)
        cols = [c for c in ("val_accuracy", "train_eval_accuracy") if c in h.columns]
        if cols:
            df = df.merge(h[["epoch"] + cols], on="epoch", how="left", suffixes=("", "_h"))
    nome = os.path.basename(run_dir.rstrip("/"))
    run = {"nome": nome, "df": df, "ruido": "ruido" in nome,
           "ancora": marcos_de_degradacao(df).get("val_acc_cai")}
    if fonte == "replay":
        if "val_accuracy" not in df.columns:
            return None
        df["status"] = replay_status(run, LimiaresStatus())
    return run


def faixa(ax, df: pd.DataFrame, y: float, altura: float) -> None:
    """Desenha a faixa de status de um run como retangulos por epoca."""
    for _, linha in df.iterrows():
        cor = COR_STATUS.get(str(linha["status"]), "#dddddd")
        ax.barh(y, 1 - GAP, left=linha["epoch"] - 0.5 + GAP / 2, height=altura,
                color=cor, edgecolor="none")


def figura(runs: list[dict], out: str) -> None:
    ruido = next((r for r in runs if r["ruido"]), None)
    ctrl = next((r for r in runs if not r["ruido"]), None)
    ordenados = ([r for r in runs if r["ruido"]] + [r for r in runs if not r["ruido"]])

    fig = plt.figure(figsize=(15.5, 9.2), facecolor=SURFACE)
    gs = fig.add_gridspec(3, 2, height_ratios=[2.4, 0.42, 3.6], hspace=0.42, wspace=0.14,
                          left=0.085, right=0.935, top=0.855, bottom=0.175)

    # --- A e B: val_loss com a faixa de status logo abaixo, mesmo eixo x ---
    for col, (res, titulo) in enumerate(((ruido, "A. Run COM overfitting"),
                                         (ctrl, "B. Run de controle (saudável)"))):
        if res is None:
            continue
        df = res["df"]
        x = df["epoch"].to_numpy()

        ax = fig.add_subplot(gs[0, col])
        ax.plot(x, df["val_loss"], color=TEXT_1, linewidth=2.0, label="val_loss")
        if res["ancora"]:
            ax.axvline(res["ancora"], color=TEXT_1, linestyle="--", linewidth=1.2,
                       alpha=0.65, label=f"val_acc cai (época {res['ancora']})")
        ax.set_facecolor(SURFACE)
        ax.set_ylabel("Perda de validação", color=TEXT_2, fontsize=9)
        ax.set_title(f"{titulo} — {res['nome']}", color=TEXT_1, fontsize=10.5,
                     loc="left", pad=8)
        ax.grid(True, color=GRID, linewidth=0.8, alpha=0.9)
        ax.set_axisbelow(True)
        ax.tick_params(colors=TEXT_2, labelsize=8.5, labelbottom=False)
        for lado in ("top", "right"):
            ax.spines[lado].set_visible(False)
        for lado in ("left", "bottom"):
            ax.spines[lado].set_color(GRID)
        ax.set_xlim(0.5, x.max() + 0.5)
        ax.legend(frameon=False, fontsize=8.5, labelcolor=TEXT_2)

        axf = fig.add_subplot(gs[1, col])
        faixa(axf, df, 0, 0.75)
        axf.set_facecolor(SURFACE)
        axf.set_xlim(0.5, x.max() + 0.5)
        axf.set_ylim(-0.5, 0.5)
        axf.set_yticks([])
        axf.set_xlabel("Época", color=TEXT_2, fontsize=9)
        axf.tick_params(colors=TEXT_2, labelsize=8.5)
        for lado in axf.spines:
            axf.spines[lado].set_visible(False)
        axf.text(0, 1.30, "status do monitor, época a época", transform=axf.transAxes,
                 color=TEXT_2, fontsize=8.5, va="bottom")

    # --- C: os 10 runs empilhados, para comparar de relance ---
    ax = fig.add_subplot(gs[2, :])
    x_max = max(r["df"]["epoch"].max() for r in ordenados)
    for i, res in enumerate(ordenados):
        faixa(ax, res["df"], i, 0.68)
        if res["ancora"]:
            ax.plot([res["ancora"]], [i], marker="v", markersize=7, color=TEXT_1,
                    markeredgecolor=SURFACE, markeredgewidth=1.0, zorder=5)
    n_ruido = sum(1 for r in ordenados if r["ruido"])
    ax.axhline(n_ruido - 0.5, color=TEXT_2, linewidth=1.0, alpha=0.6)
    ax.text(1.012, 1 - (n_ruido - 1) / 2 / len(ordenados) - 0.5 / len(ordenados),
            "COM\nruído", transform=ax.transAxes, color=TEXT_2, fontsize=8.5,
            va="center", ha="left", linespacing=1.3, clip_on=False)
    ax.text(1.012, 1 - (n_ruido + (len(ordenados) - n_ruido - 1) / 2) / len(ordenados)
            - 0.5 / len(ordenados), "controles", transform=ax.transAxes, color=TEXT_2,
            fontsize=8.5, va="center", ha="left", clip_on=False)

    ax.set_facecolor(SURFACE)
    ax.set_yticks(range(len(ordenados)))
    ax.set_yticklabels([r["nome"].replace("f6_", "") for r in ordenados], fontsize=8.5,
                       color=TEXT_2)
    ax.invert_yaxis()
    ax.set_xlim(0.5, x_max + 0.5)
    ax.set_xlabel("Época", color=TEXT_2, fontsize=9)
    ax.set_title("C. Todos os runs — faixa de status época a época",
                 color=TEXT_1, fontsize=10.5, loc="left", pad=8)
    ax.tick_params(colors=TEXT_2, labelsize=8.5)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    for lado in ("top", "right", "left"):
        ax.spines[lado].set_visible(False)
    ax.spines["bottom"].set_color(GRID)

    handles = [Patch(facecolor=cor, edgecolor="none", label=f"{nome} — {desc}")
               for nome, cor, desc in STATUS]
    handles.append(plt.Line2D([], [], marker="v", linestyle="none", color=TEXT_1,
                              markersize=7, label="época em que a acurácia de validação cai"))
    ax.legend(handles=handles, frameon=False, fontsize=8.5, labelcolor=TEXT_2,
              loc="upper center", bbox_to_anchor=(0.5, -0.155), ncol=3, handlelength=1.6,
              columnspacing=1.4)

    fig.suptitle("Linha do tempo do status do monitor — o que ele dizia, época a época",
                 color=TEXT_1, fontsize=13, x=0.005, ha="left", y=0.985, weight="bold")
    # O subtitulo descreve o que ESTA no grafico, calculado dele -- nunca uma conclusao
    # cravada no codigo, que envelheceria mal na proxima recalibracao.
    n_ruido_det = sum(1 for r in ordenados if r["ruido"]
                      and TingeOF in [s for s in r["df"]["status"]])
    n_ctrl_det = sum(1 for r in ordenados if not r["ruido"]
                     and TingeOF in [s for s in r["df"]["status"]])
    n_r = sum(1 for r in ordenados if r["ruido"])
    n_c = len(ordenados) - n_r
    fig.text(0.005, 0.938,
             "O status é um juízo ONLINE (\"com o que sei até agora\"), não o veredito final — "
             "quem corta os dados é o overfit_report.json, calculado sobre a trajetória completa.",
             color=TEXT_2, fontsize=9, ha="left", va="top")
    fig.text(0.005, 0.903,
             f"Nestes runs: OVERFITTING em {n_ruido_det}/{n_r} com ruído e "
             f"{n_ctrl_det}/{n_c} controles. "
             + ("Os dois grupos ficam separados." if n_ruido_det == n_r and n_ctrl_det == 0
                else "Os grupos NÃO ficam separados — o status acusa treino saudável também."),
             color=TEXT_2, fontsize=9, ha="left", va="top")
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description="Linha do tempo do status (F6)")
    p.add_argument("--root", default="monai_weights")
    p.add_argument("--output_dir", default="monai_weights/f6_analise")
    p.add_argument("--fonte", choices=("replay", "csv"), default="replay",
                   help="replay = recalcula com o classificador atual; csv = como foi gravado")
    p.add_argument("--sufixo", default="", help="sufixo do nome dos arquivos de saida")
    args = p.parse_args()

    alvos = sorted(
        os.path.join(args.root, d) for d in os.listdir(args.root)
        if d.startswith("f6_") and os.path.isfile(os.path.join(args.root, d, "complexity_live.csv"))
    )
    runs = [r for r in (carregar(a, args.fonte) for a in alvos) if r]
    if not runs:
        print("Nenhum run f6_* com complexity_live.csv encontrado.")
        return 1

    linhas = []
    for r in runs:
        d = r["df"]
        cont = d["status"].value_counts()
        prim = d[d["status"] == "OVERFITTING"]["epoch"]
        alerta = d[d["status"] == "ALERTA_OVERFIT"]["epoch"]
        linhas.append({
            "run": r["nome"], "ruido": r["ruido"], "n_epocas": len(d),
            "ancora": r["ancora"],
            "1o_OVERFITTING": int(prim.iloc[0]) if len(prim) else None,
            "frac_OVERFITTING": len(prim) / len(d),
            "1o_ALERTA": int(alerta.iloc[0]) if len(alerta) else None,
            **{f"n_{k}": int(v) for k, v in cont.items()},
        })
    tab = pd.DataFrame(linhas)
    os.makedirs(args.output_dir, exist_ok=True)
    tab.to_csv(os.path.join(args.output_dir, f"status_timeline{args.sufixo}.csv"), index=False)

    print("=" * 96)
    print("STATUS DO MONITOR — quanto de cada run ele classificou como OVERFITTING")
    print("=" * 96)
    print(tab[["run", "ruido", "n_epocas", "ancora", "1o_OVERFITTING",
               "frac_OVERFITTING", "1o_ALERTA"]].to_string(index=False))

    com, ctrl = tab[tab["ruido"]], tab[~tab["ruido"]]
    print(f"\nruns com ruido:  OVERFITTING em {com['frac_OVERFITTING'].mean():.0%} das epocas, "
          f"primeiro na epoca {com['1o_OVERFITTING'].mean():.1f} (media)")
    print(f"controles:       OVERFITTING em {ctrl['frac_OVERFITTING'].mean():.0%} das epocas, "
          f"primeiro na epoca {ctrl['1o_OVERFITTING'].mean():.1f} (media)")
    n_det = int((com["1o_OVERFITTING"].notna()).sum())
    n_falso = int((ctrl["1o_OVERFITTING"].notna()).sum())
    print(f"\ndetectou {n_det}/{len(com)} runs com ruido | {n_falso}/{len(ctrl)} falsos positivos")
    if n_det == len(com) and n_falso == 0:
        print("O status SEPARA os dois grupos: acusa todos os runs com ruido e nenhum controle.")
    else:
        print("O status NAO separa os dois grupos -- acusa runs saudaveis tambem.")
    print("Ainda assim e leitura auxiliar, nao resultado: quem corta os dados (regra D6)")
    print("e o overfit_report.json, calculado sobre a trajetoria completa.")

    saida = os.path.join(args.output_dir, f"status_timeline{args.sufixo}.png")
    figura(runs, saida)
    print(f"\nFigura: {saida}")
    print(f"Tabela: {os.path.join(args.output_dir, f'status_timeline{args.sufixo}.csv')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
