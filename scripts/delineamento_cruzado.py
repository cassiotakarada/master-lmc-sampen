"""
Delineamento cruzado: a divergencia da LMC vem do GRUPO DE PESOS ou do MECANISMO?

A dissertacao tem duas celulas preenchidas e duas vazias:

                    | subamostragem      | corrupcao de rotulos
  LMC nos bn_gamma  | funciona (cap. 4)  | NUNCA TESTADO      <- esta celula
  SampEn2D na densa | NUNCA TESTADO      | funciona (cap. 4)

Este script preenche a celula de cima a direita: calcula LMC e SampEn 1D sobre
o grupo bn_gamma, epoca a epoca, nos runs com corrupcao de rotulos e nos
controles PAREADOS. Nao exige re-treino -- usa os checkpoints epoch_*.pth.

Se a LMC dos bn_gamma discriminar sob corrupcao, a divergencia relatada no
capitulo 4 e atribuivel ao GRUPO DE PESOS (a camada densa e que e opaca a LMC).
Se nao discriminar, a divergencia e atribuivel ao MECANISMO (a LMC responde a
escassez de dados, nao a memorizacao).
"""
from __future__ import annotations
import argparse, glob, json, os, pathlib, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch

from src.complexity import lmc_complexity, sample_entropy_1d


def extrai_bn_gamma(caminho: str) -> np.ndarray:
    """Concatena os tensores gamma (weight) de todas as BatchNorm do checkpoint."""
    sd = torch.load(caminho, map_location="cpu", weights_only=True)
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    partes = []
    for k, v in sd.items():
        if not torch.is_tensor(v):
            continue
        # gamma da BatchNorm: vetor 1D terminando em .weight, com num_batches_tracked
        # e running_* no mesmo prefixo. Exclui fc.weight (2D) e conv (4D).
        if v.ndim == 1 and k.endswith(".weight"):
            prefixo = k.rsplit(".", 1)[0]
            if f"{prefixo}.running_mean" in sd:
                partes.append(v.detach().float().numpy().ravel())
    if not partes:
        raise RuntimeError(f"nenhum gamma de BatchNorm encontrado em {caminho}")
    return np.concatenate(partes)


def serie_do_run(run_dir: str, com_sampen: bool, bins: int, r_scale: float) -> pd.DataFrame:
    linhas = []
    for ck in sorted(glob.glob(os.path.join(run_dir, "epoch_*.pth"))):
        ep = int(os.path.basename(ck).split("_")[1].split(".")[0])
        g = extrai_bn_gamma(ck)
        d = {"epoch": ep, "n_pesos": len(g), "lmc": lmc_complexity(g, n_bins=bins)["complexity"]}
        if com_sampen:
            d["sampen"] = sample_entropy_1d(g, m=2, r_scale=r_scale)
        linhas.append(d)
    return pd.DataFrame(linhas).sort_values("epoch").reset_index(drop=True)


def amplitude_relativa(v: np.ndarray) -> float:
    """Mesma definicao do scripts/analise_f6.py: (max-min)/|media|."""
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return float("nan")
    m = np.mean(v)
    return float((v.max() - v.min()) / abs(m)) if m != 0 else float("nan")


def main() -> int:
    p = argparse.ArgumentParser(description="Delineamento cruzado: LMC nos bn_gamma")
    p.add_argument("--root", default="monai_weights")
    p.add_argument("--ruido", nargs="+", required=True, help="run_dirs com corrupcao")
    p.add_argument("--controle", nargs="+", required=True, help="run_dirs de controle PAREADO")
    p.add_argument("--output_dir", default="monai_weights/f7_cruzado")
    p.add_argument("--bins", type=int, default=100)
    p.add_argument("--r_scale", type=float, default=0.20)
    p.add_argument("--sem_sampen", action="store_true", help="calcula so a LMC (mais rapido)")
    a = p.parse_args()

    os.makedirs(a.output_dir, exist_ok=True)
    com_sampen = not a.sem_sampen
    registros = []

    for grupo, runs in (("ruido", a.ruido), ("controle", a.controle)):
        for rd in runs:
            nome = os.path.basename(rd.rstrip("/"))
            df = serie_do_run(rd, com_sampen, a.bins, a.r_scale)
            df.to_csv(os.path.join(a.output_dir, f"serie_{nome}.csv"), index=False)
            reg = {"run": nome, "grupo": grupo, "n_epocas": len(df),
                   "n_pesos": int(df.n_pesos.iloc[0]),
                   "lmc_amplitude": amplitude_relativa(df.lmc.values),
                   "lmc_epoca_pico": int(df.loc[df.lmc.idxmax(), "epoch"])}
            if com_sampen:
                reg["sampen_amplitude"] = amplitude_relativa(df.sampen.values)
            registros.append(reg)
            print(f"  [{grupo:9s}] {nome:24s} n={len(df):2d} "
                  f"lmc_ampl={reg['lmc_amplitude']:.4f} pico@{reg['lmc_epoca_pico']}")

    res = pd.DataFrame(registros)
    res.to_csv(os.path.join(a.output_dir, "cruzado_resumo.csv"), index=False)

    from scipy import stats
    print("\n" + "=" * 78)
    print("CELULA DO DELINEAMENTO: LMC (e SampEn) sobre bn_gamma SOB CORRUPCAO")
    print("=" * 78)
    r, c = res[res.grupo == "ruido"], res[res.grupo == "controle"]
    cols = ["lmc_amplitude"] + (["sampen_amplitude"] if com_sampen else [])
    for col in cols:
        A, B = r[col].dropna().values, c[col].dropna().values
        if len(A) < 2 or len(B) < 2:
            continue
        u, pv = stats.mannwhitneyu(A, B, alternative="two-sided")
        razao = A.mean() / B.mean() if B.mean() else float("nan")
        veredito = "DISCRIMINA" if (razao >= 2 and pv < 0.05) else \
                   ("fraco" if razao >= 1.5 else "NAO DISCRIMINA")
        nome = col.replace("_amplitude", "")
        print(f"{nome:10s} ruido={A.mean():.4f}+-{A.std(ddof=1):.4f}  "
              f"ctrl={B.mean():.4f}+-{B.std(ddof=1):.4f}  "
              f"razao={razao:.2f}x  p={pv:.5f}  {veredito}")
    print(f"\nSalvo em {a.output_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
