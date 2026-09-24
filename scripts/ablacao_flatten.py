"""
Ablacao da ordem do achatamento (decisao D2 do orientador, item da F6).

A PERGUNTA
----------
A matriz de pesos da camada densa e 6x512. Para calcular a SampEn 1D, que e definida
sobre uma SEQUENCIA, e preciso achatar essa matriz -- e ha duas formas de percorre-la:

  n_major  (padrao, preferencia do orientador): n1x1, n1x2, ..., n1x512, n2x1, ...
           -- le neuronio por neuronio, cada um dos 512 pesos de um, depois o proximo
  x_major: x1n1, x1n2, ..., x1n6, x2n1, ...
           -- le entrada por entrada, o peso que cada um dos 6 neuronios da a ela

Os dois vetores contem EXATAMENTE os mesmos 3.072 numeros, em ordem diferente. Logo:

  - a LMC, que so olha o histograma, TEM de coincidir -- isso e a checagem de sanidade;
  - a SampEn, que olha a vizinhanca temporal, PODE mudar muito -- e a ablacao de verdade.

Nao e um detalhe estetico: e um grau de liberdade metodologico. Se a conclusao do
trabalho dependesse da ordem escolhida, seria preciso justificar a escolha; e se uma
ordem for claramente melhor, escolher a outra por inercia desperdicaria sinal.

O QUE ESTE SCRIPT NAO FAZ
-------------------------
Nao recalcula nada a partir dos pesos: o monitor da F4 ja gravou as DUAS ordens em
`complexity_live.csv` a cada epoca (colunas `sampen` e `sampen_x_major`). Aqui so se
aplica a ambas o MESMO detector usado na F6 -- importado de `analise_f6`, nao copiado,
para que nao existam dois detectores divergentes.
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

from analise_f6 import (COR, GRID, SURFACE, TEXT_1, TEXT_2, _eixo,
                        epoca_de_afastamento, marcos_de_degradacao)

# As duas ordens, com a cor que as identifica em todas as figuras.
ORDENS = [("sampen", "n-major (padrão)", COR["sampen"]),
          ("sampen_x_major", "x-major", "#8b5cf6")]


def checagem_lmc_invariante(run_dir: str, n_epocas: int = 3) -> list[dict]:
    """A LMC tem de dar o MESMO numero nas duas ordens. Se nao der, ha bug.

    Vale como teste de integridade do pipeline inteiro: se a LMC divergisse, o
    `flatten_dense` estaria perdendo ou duplicando pesos em alguma das ordens, e todos
    os resultados da F6 estariam contaminados.
    """
    import torch
    from src.complexity import flatten_dense, lmc_complexity

    saida = []
    for ep in range(1, n_epocas + 1):
        ckpt = os.path.join(run_dir, f"epoch_{ep:03d}.pth")
        if not os.path.isfile(ckpt):
            continue
        sd = torch.load(ckpt, map_location="cpu", weights_only=True)
        densas = [(k, v) for k, v in sd.items() if v.ndim == 2]
        if not densas:
            continue
        nome, W = densas[-1]
        W = W.numpy()
        a = lmc_complexity(flatten_dense(W, order="n_major"), n_bins=100)["complexity"]
        b = lmc_complexity(flatten_dense(W, order="x_major"), n_bins=100)["complexity"]
        saida.append({"epoca": ep, "camada": nome, "lmc_n_major": a, "lmc_x_major": b,
                      "diferenca": abs(a - b)})
    return saida


def medir(run_dir: str) -> dict | None:
    """Aplica o detector da F6 as duas ordens do mesmo run."""
    caminho = os.path.join(run_dir, "complexity_live.csv")
    if not os.path.isfile(caminho):
        return None
    df = pd.read_csv(caminho).sort_values("epoch").reset_index(drop=True)
    hist = os.path.join(run_dir, "history.csv")
    if os.path.isfile(hist):
        h = pd.read_csv(hist)
        cols = [c for c in ("val_accuracy", "train_eval_accuracy") if c in h.columns]
        df = df.merge(h[["epoch"] + cols], on="epoch", how="left", suffixes=("", "_h"))

    ancora = marcos_de_degradacao(df).get("val_acc_cai")
    nome = os.path.basename(run_dir.rstrip("/"))
    linha = {"run": nome, "ruido": "ruido" in nome, "ancora": ancora}

    for chave, rotulo, _ in ORDENS:
        if chave not in df.columns:
            continue
        s = df[chave].to_numpy()
        sinal = epoca_de_afastamento(s)
        fin = s[np.isfinite(s)]
        linha[f"{chave}_sinal"] = sinal
        linha[f"{chave}_antec"] = (ancora - sinal) if (ancora and sinal) else None
        linha[f"{chave}_amplitude"] = (
            (fin.max() - fin.min()) / abs(fin.mean()) if len(fin) and fin.mean() else None
        )
    return {"linha": linha, "df": df, "nome": nome}


def figura(resultados: list[dict], tab: pd.DataFrame, out: str) -> None:
    """Quatro paineis: as duas series num run de cada tipo, amplitude e antecedencia.

    Painel por painel, e NUNCA dois eixos y (anti-padrao registrado no plan.md).
    """
    ruido = next((r for r in resultados if r["linha"]["ruido"]), None)
    ctrl = next((r for r in resultados if not r["linha"]["ruido"]), None)

    fig, axes = plt.subplots(1, 4, figsize=(17.5, 4.5), facecolor=SURFACE)

    for ax, res, titulo in ((axes[0], ruido, "A. Run COM overfitting"),
                            (axes[1], ctrl, "B. Run de controle")):
        if res is None:
            ax.set_visible(False)
            continue
        df, x = res["df"], res["df"]["epoch"].to_numpy()
        for chave, rotulo, cor in ORDENS:
            if chave not in df.columns:
                continue
            ax.plot(x, df[chave], color=cor, linewidth=1.8, label=rotulo)
            sinal = res["linha"].get(f"{chave}_sinal")
            if sinal:
                ax.axvline(sinal, color=cor, linestyle=":", linewidth=1.5)
        anc = res["linha"].get("ancora")
        if anc:
            ax.axvline(anc, color=COR["val"], linestyle="--", linewidth=1.2, alpha=0.7,
                       label=f"val_acc cai (época {anc})")
        _eixo(ax, "Época", "SampEn 1D", f"{titulo} — {res['nome']}")
        ax.legend(frameon=False, fontsize=8.5, labelcolor=TEXT_2)

    # C. amplitude por run, as duas ordens lado a lado
    ax = axes[2]
    t = tab.sort_values(["ruido", "run"], ascending=[False, True])
    y = np.arange(len(t))
    alt = 0.38
    for i, (chave, rotulo, cor) in enumerate(ORDENS):
        col = f"{chave}_amplitude"
        if col not in t.columns:
            continue
        ax.barh(y + (i - 0.5) * alt, t[col] * 100, height=alt, color=cor,
                label=rotulo, edgecolor=SURFACE, linewidth=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels([n.replace("f6_", "") for n in t["run"]], fontsize=7.5)
    ax.invert_yaxis()
    _eixo(ax, "Amplitude relativa (%)", "", "C. Amplitude — quanto a medida se move")
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
    ax.legend(frameon=False, fontsize=8.5, labelcolor=TEXT_2)

    # D. antecedencia, so nos runs com overfitting (nos controles nao ha o que antecipar)
    ax = axes[3]
    com = tab[tab["ruido"]]
    for i, (chave, rotulo, cor) in enumerate(ORDENS):
        col = f"{chave}_antec"
        if col not in com.columns:
            continue
        v = com[col].dropna().astype(float)
        if not len(v):
            continue
        ax.barh([i], [v.mean()], height=0.45, color=cor, edgecolor=SURFACE, linewidth=0.6)
        ax.errorbar([v.mean()], [i], xerr=[v.std(ddof=1) if len(v) > 1 else 0],
                    fmt="none", ecolor=TEXT_2, elinewidth=1.2, capsize=4)
        ax.text(v.mean() + 0.25, i, f"{v.mean():.1f} ± {v.std(ddof=1):.1f}",
                va="center", fontsize=8.5, color=TEXT_1)
    ax.set_yticks(range(len(ORDENS)))
    ax.set_yticklabels([r for _, r, _ in ORDENS], fontsize=9)
    ax.invert_yaxis()
    ax.axvline(0, color=TEXT_2, linewidth=1.0)
    _eixo(ax, "Épocas de antecedência (média ± dp)", "",
          "D. Antecedência — quanto avisa antes")

    fig.suptitle("Ablação da ordem do achatamento — n-major × x-major (SampEn 1D)",
                 color=TEXT_1, fontsize=13, x=0.005, ha="left", y=0.985, weight="bold")
    fig.text(0.005, 0.935,
             "Mesmos 3.072 pesos, percorridos em ordens diferentes. A LMC coincide (só "
             "olha o histograma); a SampEn não — e é por isso que a ordem precisa ser "
             "declarada na metodologia.",
             color=TEXT_2, fontsize=8.5, ha="left", va="top")
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description="Ablacao da ordem do flatten (D2)")
    p.add_argument("--root", default="monai_weights")
    p.add_argument("--output_dir", default="monai_weights/f6_analise")
    args = p.parse_args()

    alvos = sorted(
        os.path.join(args.root, d) for d in os.listdir(args.root)
        if d.startswith("f6_") and os.path.isfile(os.path.join(args.root, d, "complexity_live.csv"))
    )
    if not alvos:
        print("Nenhum run f6_* com complexity_live.csv encontrado.")
        return 1

    print("=" * 96)
    print("SANIDADE — a LMC tem de ser IDENTICA nas duas ordens (so olha o histograma)")
    print("=" * 96)
    checagens = checagem_lmc_invariante(alvos[0])
    for c in checagens:
        ok = "OK" if c["diferenca"] < 1e-12 else "DIVERGE!"
        print(f"  epoca {c['epoca']} | {c['camada']:<12} n_major={c['lmc_n_major']:.12f} "
              f"x_major={c['lmc_x_major']:.12f} | dif={c['diferenca']:.2e}  {ok}")
    if checagens and max(c["diferenca"] for c in checagens) >= 1e-12:
        print("\n  A LMC DIVERGIU entre as ordens -- ha bug no flatten_dense. Pare aqui.")
        return 1

    resultados = [r for r in (medir(a) for a in alvos) if r]
    tab = pd.DataFrame([r["linha"] for r in resultados])

    os.makedirs(args.output_dir, exist_ok=True)
    tab.to_csv(os.path.join(args.output_dir, "ablacao_ordem_flatten.csv"), index=False)

    print("\n" + "=" * 96)
    print("ABLACAO — mesmo detector da F6 aplicado as duas ordens")
    print("=" * 96)
    print(tab.to_string(index=False))

    com, ctrl = tab[tab["ruido"]], tab[~tab["ruido"]]
    print("\n" + "=" * 96)
    print("CONTRASTE ruido x controle, por ordem")
    print("=" * 96)
    print(f"{'ordem':<18} {'amp. ruido':>11} {'amp. ctrl':>10} {'razao':>7} {'p':>9} "
          f"{'antec (media+-dp)':>19} {'falsos+':>9}")
    for chave, rotulo, _ in ORDENS:
        a = com[f"{chave}_amplitude"].dropna().values
        b = ctrl[f"{chave}_amplitude"].dropna().values
        pv = stats.ttest_ind(a, b, equal_var=False).pvalue if len(a) > 1 and len(b) > 1 else float("nan")
        an = com[f"{chave}_antec"].dropna().astype(float)
        fp = ctrl[f"{chave}_sinal"].dropna()
        antec = f"{an.mean():.1f} +- {an.std(ddof=1):.1f}" if len(an) > 1 else "-"
        print(f"{rotulo:<18} {a.mean():>10.1%} {b.mean():>10.1%} {a.mean()/b.mean():>6.1f}x "
              f"{pv:>9.4f} {antec:>19} {len(fp):>4}/{len(ctrl)}")

    print("\nepoca do sinal e MARGEM DE SEGURANCA:")
    print("  margem = (primeiro sinal entre os controles) - (ultimo sinal entre os runs de")
    print("  ruido). E ela, e nao a contagem de falsos positivos, que diz se existe um")
    print("  limiar de epoca capaz de separar os dois grupos -- e com quanta folga.")
    for chave, rotulo, _ in ORDENS:
        a = sorted(int(v) for v in com[f"{chave}_sinal"].dropna())
        b = sorted(int(v) for v in ctrl[f"{chave}_sinal"].dropna())
        if a and b:
            margem = min(b) - max(a)
            sep = f"margem {margem:+d} epocas" if margem > 0 else "SOBREPOE"
        else:
            sep = "-"
        print(f"  {rotulo:<18} ruido {str(a):<24} controle {str(b):<24} {sep}")

    saida = os.path.join(args.output_dir, "ablacao_ordem_flatten.png")
    figura(resultados, tab, saida)
    print(f"\nFigura: {saida}")
    print(f"Tabela: {os.path.join(args.output_dir, 'ablacao_ordem_flatten.csv')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
