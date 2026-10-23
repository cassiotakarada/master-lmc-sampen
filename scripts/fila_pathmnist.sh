#!/usr/bin/env bash
# Dispara os 73 runs do PathMNIST (prefixo pm_) nas duas GPUs, um worker por GPU.
# Cada worker pega o proximo run livre (reserva em monai_weights/.fila/), comecando
# pelos mais longos, entao a fila se equilibra sozinha. Seguro para relancar: runs
# completos sao pulados. Para refazer um run que falhou, apague monai_weights/.fila/<run_id>.
#
# Uso: scripts/fila_pathmnist.sh            (roda em segundo plano, sobrevive ao terminal)
#      tail -f monai_weights/pm_filas/gpu0.log
set -u
cd "$(dirname "$0")/.."
mkdir -p monai_weights/pm_filas
for gpu in 0 1; do
  CUDA_VISIBLE_DEVICES=$gpu setsid nohup myenv/bin/python -u run_sweep.py \
    --only pm_ --claim --longest_first \
    > "monai_weights/pm_filas/gpu${gpu}.log" 2>&1 < /dev/null &
  echo "GPU $gpu -> worker PID $! (log: monai_weights/pm_filas/gpu${gpu}.log)"
  sleep 5
done
