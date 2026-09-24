"""
O indicador vale so para 30 % de ruido? (ultima lacuna da F6)

A PERGUNTA
----------
Todo o resultado ate aqui veio de UM nivel de ruido. Sem variar isso, "a SampEn2D
antecipa o overfitting" e uma afirmacao sobre 30 % de rotulos sorteados, nao sobre
overfitting. Este script cruza tres niveis -- 15 %, 30 % e 50 % -- nas MESMAS tres
sementes (42, 1, 7), de modo que a unica coisa que muda entre eles e o ruido.

O QUE OS DADOS MOSTRAM
----------------------
A epoca do ALERTA nao se mexe: ~10 nos tres niveis. Quem se mexe e a epoca em que o
estrago aparece na validacao, que chega cada vez mais cedo conforme o ruido aumenta.
A antecedencia encolhe (7,7 -> 5,3 -> 3,0 epocas) por causa disso, e nao porque o
indicador piore.

Lendo de outro jeito: a SampEn2D parece marcar o INICIO DA MEMORIZACAO, um evento que
acontece na mesma altura do treino independentemente de quanto ruido existe. O que o
ruido controla e a velocidade com que essa memorizacao vira dano visivel.
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
from scipy import stats

from analise_f6 import COR, GRID, SURFACE, TEXT_1, TEXT_2, _eixo
from calibrar_status import carregar, replay
from src.training.complexity_monitor import LimiaresStatus
from src.training.overfit import detect_degradacao_acuracia

# (nivel sorteado, prefixo do run_id). A fracao efetivamente ERRADA e f*5/6, porque o
# sorteio inclui a classe verdadeira (convencao de Zhang et al., 2017).
NIVEIS = [(15, "f6_ruido15_seed{s}"), (30, None), (50, "f6_ruido50_seed{s}")]
SEMENTES = (42, 1, 7)


def caminho(nivel: int, molde: str | None, semente: int) -> str:
    if molde is None:   # os runs de 30 % vieram antes e usam outro padrao de nome
        nome = "f6_ruido30" if semente == 42 else f"f6_ruido30_seed{semente}"
    else:
        nome = molde.format(s=semente)
    return os.path.join("monai_weights", nome)


def medir(run_dir: str) -> dict | None:
    if not os.path.isdir(run_dir):
        return None
    run = carregar(run_dir)
    if run is None:
        return None
    st = replay(run, LimiaresStatus())
    d = pd.read_csv(os.path.join(run_dir, "history.csv"))
    rel = detect_degradacao_acuracia(d["val_accuracy"].tolist(), queda=0.01, patience=3)
    anc = rel["inicio_overfit"] if rel["confirmado"] else None
    al = next((i + 1 for i, x in enumerate(st) if x == "ALERTA_OVERFIT"), None)
    of = next((i + 1 for i, x in enumerate(st) if x == "OVERFITTING"), None)
    return {"run": os.path.basename(run_dir), "alerta": al, "overfit": of, "ancora": anc,
            "antecedencia": (anc - al) if (anc and al) else None,
            "treino_final": float(d["train_eval_accuracy"].iloc[-1]),
            "val_final": float(d["val_accuracy"].iloc[-1])}


def figura(tab: pd.DataFrame, out: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6), facecolor=SURFACE)
    niveis = sorted(tab["errados"].unique())

    # A. as duas epocas, lado a lado: uma nao se mexe, a outra desce
    ax = axes[0]
    for col, rotulo, cor, marca in (("alerta", "alerta da SampEn2D", COR["sampen2d"], "o"),
                                    ("ancora", "degradação visível na validação", COR["val"], "s")):
        m = tab.groupby("errados")[col].mean()
        sd = tab.groupby("errados")[col].std()
        ax.errorbar(niveis, m.values, yerr=sd.values, color=cor, linewidth=2.0,
                    marker=marca, markersize=8, markeredgecolor=SURFACE,
                    markeredgewidth=1.2, capsize=4, label=rotulo)
    # A seta aponta para o VAO entre as duas curvas no nivel mais baixo -- e esse vao
    # que o painel B transforma em barra.
    ax.annotate("a distância entre as\nduas linhas É a antecedência",
                xy=(niveis[0] + 0.6, (tab[tab.errados == niveis[0]].alerta.mean()
                                      + tab[tab.errados == niveis[0]].ancora.mean()) / 2),
                xytext=(niveis[0] + 9, 12.0), color=TEXT_2, fontsize=8.5,
                arrowprops=dict(arrowstyle="->", color=TEXT_2, lw=1.0),
                linespacing=1.4, ha="left")
    _eixo(ax, "Rótulos efetivamente errados (%)", "Época",
          "A. O alerta não se move; o estrago chega antes")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=TEXT_2, loc="upper right")

    # B. a antecedencia, que e a diferenca entre as duas curvas de A
    ax = axes[1]
    m = tab.groupby("errados")["antecedencia"].mean()
    sd = tab.groupby("errados")["antecedencia"].std()
    ax.bar(niveis, m.values, yerr=sd.values, width=6, color=COR["sampen2d"],
           edgecolor=SURFACE, linewidth=1.5, capsize=5,
           error_kw=dict(ecolor=TEXT_2, elinewidth=1.2))
    for n, v in zip(niveis, m.values):
        ax.text(n, v + 0.55, f"{v:.1f}", ha="center", color=TEXT_1, fontsize=9.5)
    ax.axhline(0, color=TEXT_2, linewidth=1.0)
    _eixo(ax, "Rótulos efetivamente errados (%)", "Épocas de antecedência",
          "B. Quanto o aviso chega antes")

    # C. a checagem que impede confundir overfitting com artefato
    ax = axes[2]
    larg = 2.6
    tr = tab.groupby("errados")["treino_final"].mean() * 100
    va = tab.groupby("errados")["val_final"].mean() * 100
    ax.bar(np.array(niveis) - larg / 2, tr.values, width=larg, color=COR["train"],
           edgecolor=SURFACE, linewidth=1.2)
    ax.bar(np.array(niveis) + larg / 2, va.values, width=larg, color=COR["val"],
           edgecolor=SURFACE, linewidth=1.2)
    # Rotulo direto na primeira dupla, em vez de caixa de legenda: com so duas series a
    # legenda roubaria area util e taparia barras que vao ate 100.
    ax.text(niveis[0] - larg / 2, tr.values[0] + 2.5, "treino\nem eval()", ha="center",
            va="bottom", color=COR["train"], fontsize=8.5, linespacing=1.3)
    ax.text(niveis[0] + larg / 2, va.values[0] - 4, "validação", ha="center", va="top",
            color=SURFACE, fontsize=8.5, rotation=90)
    for n, v in zip(niveis, va.values):
        ax.text(n + larg / 2, v + 2.0, f"{v:.0f}", ha="center", color=TEXT_1, fontsize=8.5)
    _eixo(ax, "Rótulos efetivamente errados (%)", "Acurácia final (%)",
          "C. Memorização real nos três níveis")
    ax.set_ylim(0, 118)

    fig.suptitle("A antecedência depende do nível de ruído — mas o alerta, não",
                 color=TEXT_1, fontsize=13, x=0.005, ha="left", y=0.985, weight="bold")
    fig.text(0.005, 0.925,
             "3 níveis × as mesmas 3 sementes (42, 1, 7): só o ruído muda entre eles. "
             "A SampEn2D marca o início da memorização por volta da época 10 nos três — "
             "o que o ruído controla é a rapidez com que isso vira dano visível.",
             color=TEXT_2, fontsize=9, ha="left", va="top")
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description="Antecedencia x nivel de ruido")
    p.add_argument("--output_dir", default="monai_weights/f6_analise")
    args = p.parse_args()

    linhas = []
    for nivel, molde in NIVEIS:
        for s in SEMENTES:
            m = medir(caminho(nivel, molde, s))
            if m:
                linhas.append({"nivel": nivel, "errados": nivel * 5 / 6,
                               "semente": s, **m})
    if not linhas:
        print("Nenhum run encontrado.")
        return 1
    tab = pd.DataFrame(linhas)

    print("=" * 96)
    print("ANTECEDENCIA x NIVEL DE RUIDO — as mesmas 3 sementes nos 3 niveis")
    print("=" * 96)
    print(tab[["nivel", "errados", "semente", "alerta", "overfit", "ancora",
               "antecedencia", "treino_final", "val_final"]].to_string(index=False))

    print("\n" + "=" * 96)
    print("O QUE MUDA E O QUE NAO MUDA")
    print("=" * 96)
    print(f"{'ruido':>7} {'alerta':>14} {'ancora':>14} {'antecedencia':>14} {'detectou':>9}")
    for nivel in sorted(tab["nivel"].unique()):
        g = tab[tab["nivel"] == nivel]
        det = f"{g['overfit'].notna().sum()}/{len(g)}"
        print(f"{nivel:>6}% {g.alerta.mean():>8.1f} ± {g.alerta.std():.1f} "
              f"{g.ancora.mean():>8.1f} ± {g.ancora.std():.1f} "
              f"{g.antecedencia.mean():>8.1f} ± {g.antecedencia.std():.1f} {det:>9}")

    x = tab["errados"].to_numpy()
    print("\nregressao contra a fracao de rotulos errados:")
    for col, rotulo in (("alerta", "epoca do ALERTA"), ("ancora", "epoca da ANCORA"),
                        ("antecedencia", "ANTECEDENCIA")):
        r = stats.linregress(x, tab[col].to_numpy())
        marca = "NAO depende do ruido" if r.pvalue > 0.05 else "depende do ruido"
        print(f"  {rotulo:<16} {r.slope:+.4f} epoca por ponto de ruido | "
              f"r={r.rvalue:+.3f} p={r.pvalue:.4f}  -> {marca}")

    os.makedirs(args.output_dir, exist_ok=True)
    tab.to_csv(os.path.join(args.output_dir, "niveis_de_ruido.csv"), index=False)
    saida = os.path.join(args.output_dir, "niveis_de_ruido.png")
    figura(tab, saida)
    print(f"\nFigura: {saida}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
