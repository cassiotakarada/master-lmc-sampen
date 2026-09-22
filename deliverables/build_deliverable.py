"""Gera o pacote autocontido `deliverables/complexity_reference/` a partir de src/complexity.py.

Rode a partir da raiz do repositorio:  myenv/bin/python deliverables/build_deliverable.py

Por que um script e nao uma copia manual: manter uma copia editavel a mao
recriaria exatamente o problema que a F1 veio resolver (a mesma funcao definida
em varios lugares). O pacote de entrega e um ARTEFATO GERADO.
"""
import json
import pathlib
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "deliverables" / "complexity_reference"
SRC = ROOT / "src" / "complexity.py"

HEADER = """# ==============================================================================
# ARQUIVO GERADO AUTOMATICAMENTE -- nao edite aqui.
# Fonte: src/complexity.py  (repositorio Masters Claude)
# Regenerar com: python deliverables/build_deliverable.py
# ==============================================================================

"""


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    # 1. modulo (copia carimbada)
    dest = OUT / "complexity.py"
    dest.write_text(HEADER + SRC.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"[ok] {dest.relative_to(ROOT)}")

    # 2. matriz real da camada densa, para o exemplo rodar sem torch
    try:
        import numpy as np
        import torch

        ckpt = ROOT / "monai_weights" / "run_mednist_resnet18_64" / "epoch_005.pth"
        if ckpt.is_file():
            sd = torch.load(ckpt, map_location="cpu", weights_only=True)
            W = sd["fc.weight"].detach().cpu().numpy()
            np.save(OUT / "fc_weight_exemplo.npy", W)
            meta = {
                "origem": str(ckpt.relative_to(ROOT)),
                "parametro": "fc.weight",
                "forma": list(W.shape),
                "n_pesos": int(W.size),
                "modelo": "ResNet-18, 6 classes, MedNIST 64x64 (checkpoint anterior a F3b, treinado com MONAI)",
                "descricao": "W[j, i] = peso do caminho x_i -> n_j (forma [neuronios, entradas])",
            }
            (OUT / "fc_weight_exemplo.json").write_text(
                json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            print(f"[ok] fc_weight_exemplo.npy  forma={W.shape}")
        else:
            print(f"[!!] checkpoint nao encontrado: {ckpt}")
    except Exception as exc:  # noqa: BLE001
        print(f"[!!] nao foi possivel extrair a matriz de exemplo: {exc}")

    print(f"\nPacote pronto em: {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
