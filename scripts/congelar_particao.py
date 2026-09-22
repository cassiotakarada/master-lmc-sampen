"""
Congela a particao treino/validacao/teste do MedNIST em disco (fase F3b do plan.md).

POR QUE ISTO EXISTE
-------------------
Hoje quem decide quem cai em cada conjunto e o `MedNISTDataset` do MONAI, com o
embaralhamento semeado interno dele. Ao remover o MONAI, essa regra sumiria junto e a
particao mudaria -- o que quebraria a comparabilidade com os runs ja feitos.

A solucao e gravar a lista exata de arquivos de cada conjunto num JSON, ANTES de tirar
o MONAI. A partir dai a particao deixa de depender da implementacao interna de uma
biblioteca e passa a ser um dado versionado do projeto, o que e melhor para
reprodutibilidade: qualquer pessoa consegue refazer exatamente a mesma divisao.

RODE ESTE SCRIPT ENQUANTO O MONAI AINDA ESTIVER INSTALADO.

Uso:
    myenv/bin/python scripts/congelar_particao.py
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

DESTINO = "data/mednist_split.json"
DATA_SEED = 0
VAL_FRAC = 0.1
TEST_FRAC = 0.1


def main() -> int:
    try:
        from monai.apps import MedNISTDataset
    except ImportError:
        print("ERRO: MONAI nao esta instalado. Este script precisa rodar ANTES da remocao.")
        return 1

    comum = dict(root_dir="data/MedNIST", download=True, transform=None, cache_rate=0.0,
                 num_workers=0, seed=DATA_SEED, val_frac=VAL_FRAC, test_frac=TEST_FRAC)

    particao = {}
    for secao, nome in (("training", "train"), ("validation", "val"), ("test", "test")):
        ds = MedNISTDataset(section=secao, **comum)
        itens = [{"image": d["image"], "label": int(d["label"]),
                  "class_name": d.get("class_name", "")} for d in ds.data]
        particao[nome] = itens
        print(f"  {nome:<6} {len(itens):>6} amostras")

    classes = sorted({d["class_name"] for d in particao["train"]})
    rotulo_por_classe = {}
    for d in particao["train"]:
        rotulo_por_classe.setdefault(d["class_name"], d["label"])

    # Assinatura de integridade: detecta se os arquivos em disco mudaram.
    h = hashlib.sha256()
    for nome in ("train", "val", "test"):
        for d in particao[nome]:
            h.update(f"{nome}|{d['image']}|{d['label']}\n".encode())

    saida = {
        "_descricao": ("Particao treino/validacao/teste do MedNIST, congelada a partir do "
                       "MedNISTDataset do MONAI antes da migracao para PyTorch puro (F3b). "
                       "Esta e agora a fonte de verdade da particao -- nao regerar sem motivo."),
        "gerado_por": "scripts/congelar_particao.py",
        "origem": f"monai.apps.MedNISTDataset(seed={DATA_SEED}, val_frac={VAL_FRAC}, test_frac={TEST_FRAC})",
        "data_seed": DATA_SEED,
        "val_frac": VAL_FRAC,
        "test_frac": TEST_FRAC,
        "classes": classes,
        "rotulo_por_classe": rotulo_por_classe,
        "n_train": len(particao["train"]),
        "n_val": len(particao["val"]),
        "n_test": len(particao["test"]),
        "sha256": h.hexdigest(),
        "splits": particao,
    }

    os.makedirs(os.path.dirname(DESTINO), exist_ok=True)
    with open(DESTINO, "w", encoding="utf-8") as f:
        json.dump(saida, f, indent=1, ensure_ascii=False)

    tam = os.path.getsize(DESTINO) / 1e6
    print(f"\nParticao congelada em {DESTINO} ({tam:.1f} MB)")
    print(f"  classes: {classes}")
    print(f"  rotulos: {rotulo_por_classe}")
    print(f"  sha256 : {saida['sha256'][:16]}...")

    # Verificacao: nenhuma amostra pode aparecer em dois conjuntos
    conjuntos = {n: {d["image"] for d in particao[n]} for n in ("train", "val", "test")}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        comuns = conjuntos[a] & conjuntos[b]
        assert not comuns, f"VAZAMENTO: {len(comuns)} amostras em {a} E {b}"
    print("  verificado: nenhuma amostra aparece em dois conjuntos")
    return 0


if __name__ == "__main__":
    sys.exit(main())
