"""Resumo da melhor epoca por run, com o corte de overfitting (regra D6 do plan.md).

Uso:
    myenv/bin/python best_epochs.py                      # varre monai_weights/
    myenv/bin/python best_epochs.py --run_dirs a b c
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.analysis.overfit_cut import carregar_relatorio  # noqa: E402


def coletar(run_dir: str) -> dict:
    """Le history.csv de um run e devolve uma linha do resumo, ou None."""
    hist = os.path.join(run_dir, "history.csv")
    if not os.path.isfile(hist):
        return None
    df = pd.read_csv(hist)
    loss_col = next((c for c in ("val_loss", "test_loss") if c in df.columns), None)
    acc_col = next((c for c in ("val_accuracy", "test_accuracy") if c in df.columns), None)
    if "epoch" not in df.columns or loss_col is None:
        print(f"[skip] {hist}: faltam colunas obrigatorias")
        return None

    # Runs interrompidos deixam history.csv so com o cabecalho.
    df = df.dropna(subset=[loss_col])
    if df.empty:
        print(f"[skip] {hist}: nenhuma epoca registrada (run interrompido?)")
        return None

    melhor = df.loc[df[loss_col].idxmin()]
    rel = carregar_relatorio(run_dir) or {}

    return {
        "run": os.path.basename(run_dir.rstrip("/")),
        "n_epocas": len(df),
        "melhor_epoca": int(melhor["epoch"]),
        "train_loss": round(float(melhor.get("train_loss", float("nan"))), 5),
        "val_loss": round(float(melhor[loss_col]), 6),
        "val_acc": round(float(melhor[acc_col]), 5) if acc_col else None,
        # --- corte de overfitting (D6) ---
        "overfit": bool(rel.get("confirmado", False)),
        "inicio_overfit": rel.get("inicio_overfit"),
        "epoca_deteccao": rel.get("epoca_deteccao"),
        "usaveis": rel.get("n_epocas_usaveis"),
        "descartadas": rel.get("n_epocas_descartadas", 0),
        "origem": rel.get("_origem", "-"),
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Melhor epoca + corte de overfitting por run")
    p.add_argument("--run_dirs", nargs="+", default=None)
    p.add_argument("--root", default="monai_weights",
                   help="raiz a varrer quando --run_dirs nao e informado")
    args = p.parse_args()

    if args.run_dirs:
        alvos = args.run_dirs
    else:
        alvos = sorted(
            os.path.join(args.root, d) for d in os.listdir(args.root)
            if os.path.isfile(os.path.join(args.root, d, "history.csv"))
        ) if os.path.isdir(args.root) else []

    if not alvos:
        print(f"Nenhum run com history.csv encontrado em {args.root}")
        return 1

    linhas = [l for l in (coletar(rd) for rd in alvos) if l]
    if not linhas:
        print("Nenhum run utilizavel")
        return 1

    df = pd.DataFrame(linhas)
    print("\nMELHOR EPOCA E CORTE DE OVERFITTING (regra D6)")
    print("=" * 110)
    print(df.to_string(index=False))

    n_of = int(df["overfit"].sum())
    print(f"\n{n_of} de {len(df)} run(s) com overfitting confirmado.")
    if n_of:
        print(f"Total de epocas descartadas: {int(df['descartadas'].sum())} "
              f"de {int(df['n_epocas'].sum())}.")
    print("Epocas apos a melhor NAO devem entrar em conclusoes, medias ou correlacoes.")
    print("Elas seguem gravadas e aparecem nos graficos, sombreadas.")

    destino = os.path.join(args.root if not args.run_dirs else ".", "best_epoch_summary.csv")
    df.to_csv(destino, index=False)
    print(f"\nSalvo em: {destino}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
