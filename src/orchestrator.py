"""
src/orchestrator.py — YAML-driven, resumable experiment runner.

Usage:
  python -m src.orchestrator --config config/experiment_grid.yaml
  python -m src.orchestrator --config config/experiment_grid.yaml --resume
  python -m src.orchestrator --config config/experiment_grid.yaml --dry-run

Output layout:
  results/
    runs/
      {run_id}/         ← per-run directory (training.py layout)
    results.csv         ← one summary row per completed run
    status.json         ← {run_id: "pending"|"running"|"completed"|"failed"}

Resume behaviour:
  Runs already marked "completed" in status.json are skipped.
  Runs marked "running" or "failed" (from a crashed previous attempt) are
  retried from their last completed epoch via train_run(resume=True).

Writer trigger:
  After every writer_trigger_every completed runs the orchestrator checks for
  src/writer.py and invokes it as a subprocess.  If the file does not yet exist
  (deliverable 7 not yet implemented) the trigger is silently skipped.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import yaml

from src.training import train_run

logger = logging.getLogger(__name__)


# ============================================================================
# Dataset name normalisation
# ============================================================================

_DATASET_CLASS = {
    'pathmnist':    'PathMNIST',
    'organamnist':  'OrganAMNIST',
    'organcmnist':  'OrganCMNIST',
    'organsmnist':  'OrganSMNIST',
    'bloodmnist':   'BloodMNIST',
    'chestmnist':   'ChestMNIST',
    'dermamnist':   'DermaMNIST',
    'retinamnist':  'RetinaMNIST',
    'pneumoniamnist': 'PneumoniaMNIST',
    'octmnist':     'OCTMNIST',
    'breastmnist':  'BreastMNIST',
    'tissuemnist':  'TissueMNIST',
}


def normalise_dataset(name: str) -> str:
    """Return the medmnist class name (PascalCase) for any casing of name."""
    return _DATASET_CLASS.get(name.lower(), name)


# ============================================================================
# Grid expansion
# ============================================================================

def load_grid(config_path: str | Path) -> tuple[dict, list[dict]]:
    """
    Parse the YAML config file.

    Returns (defaults, run_configs) where run_configs is a flat list of dicts
    — one per (dataset, epochs, seed) cell — with defaults merged in and
    run_id populated.
    """
    path = Path(config_path)
    with open(path) as f:
        raw = yaml.safe_load(f)

    defaults = raw.get('defaults', {})
    experiments = raw.get('experiments', [])
    runs = _expand(experiments, defaults)
    return defaults, runs


DEFAULT_WEIGHT_DECAY = 1e-4    # supervisor's validated baseline
DEFAULT_ARCHITECTURE = "resnet18"


def _expand(experiments: list[dict], defaults: dict) -> list[dict]:
    """Flatten experiments × seeds × train_fractions × weight_decays × architectures.

    Each experiment cell may declare:
      - seeds:           list of model/training seeds (required or via defaults)
      - train_fractions: list of fractions in (0, 1] (optional; defaults to [1.0])
      - train_fraction:  scalar shortcut (ignored if train_fractions is set)
      - weight_decays:   list of WD values (optional; defaults to [defaults.weight_decay])
      - weight_decay:    scalar shortcut (ignored if weight_decays is set)
      - architectures:   list of architecture names (optional; defaults to ['resnet18'])
      - architecture:    scalar shortcut (ignored if architectures is set)
      - data_seed:       int for deterministic subsample (optional; defaults to 0)
    """
    runs: list[dict] = []
    default_wd = defaults.get('weight_decay', DEFAULT_WEIGHT_DECAY)
    default_arch = defaults.get('architecture', DEFAULT_ARCHITECTURE)

    for exp in experiments:
        dataset = normalise_dataset(exp['dataset'])
        epochs  = exp['epochs']
        seeds   = exp.get('seeds', [defaults.get('seed', 42)])

        fractions = exp.get('train_fractions')
        if fractions is None:
            fractions = [exp.get('train_fraction',
                                 defaults.get('train_fraction', 1.0))]

        wds = exp.get('weight_decays')
        if wds is None:
            wds = [exp.get('weight_decay', default_wd)]

        archs = exp.get('architectures')
        if archs is None:
            archs = [exp.get('architecture', default_arch)]

        data_seed = exp.get('data_seed', defaults.get('data_seed', 0))

        for seed in seeds:
            for frac in fractions:
                for wd in wds:
                    for arch in archs:
                        cfg = {
                            **defaults,
                            'dataset':        dataset,
                            'epochs':         epochs,
                            'seed':           seed,
                            'train_fraction': float(frac),
                            'data_seed':      int(data_seed),
                            'weight_decay':   float(wd),
                            'architecture':   str(arch),
                        }
                        cfg['run_id'] = make_run_id(
                            dataset, epochs, seed,
                            train_fraction=float(frac),
                            weight_decay=float(wd),
                            architecture=str(arch),
                        )
                        runs.append(cfg)
    return runs


def make_run_id(dataset: str, epochs: int, seed: int,
                train_fraction: float = 1.0,
                weight_decay: float = DEFAULT_WEIGHT_DECAY,
                architecture: str = DEFAULT_ARCHITECTURE) -> str:
    """Canonical run identifier.

    Examples:
        'PathMNIST_30ep_seed42'                          (full data, default WD+arch)
        'PathMNIST_100ep_seed42_frac0.10'                (subsampled)
        'PathMNIST_100ep_seed42_wd0'                     (full data, WD=0)
        'PathMNIST_100ep_seed42_wd1e-03'                 (full data, WD=1e-3)
        'PathMNIST_100ep_seed42_arch-densenet121'        (different architecture)
    """
    base = f"{dataset}_{epochs}ep_seed{seed}"
    if train_fraction < 1.0:
        base += f"_frac{train_fraction:.2f}"
    if abs(weight_decay - DEFAULT_WEIGHT_DECAY) > 1e-12:
        if weight_decay == 0.0:
            base += "_wd0"
        else:
            exp = int(round(math.log10(weight_decay))) if weight_decay > 0 else 0
            mant = weight_decay / (10 ** exp)
            if abs(mant - 1.0) < 1e-6:
                base += f"_wd1e{exp:+03d}"
            else:
                base += f"_wd{weight_decay:.0e}"
    if architecture.lower() != DEFAULT_ARCHITECTURE:
        base += f"_arch-{architecture.lower()}"
    return base


# ============================================================================
# Status tracking
# ============================================================================

_STATUS_FILE = 'status.json'
VALID_STATUSES = ('pending', 'running', 'completed', 'failed')


def load_status(results_dir: Path) -> dict[str, str]:
    path = results_dir / _STATUS_FILE
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def save_status(status: dict[str, str], results_dir: Path) -> None:
    (results_dir / _STATUS_FILE).write_text(json.dumps(status, indent=2))


def set_run_status(run_id: str, new_status: str, results_dir: Path) -> None:
    status = load_status(results_dir)
    status[run_id] = new_status
    save_status(status, results_dir)


def get_run_status(run_id: str, results_dir: Path) -> str:
    return load_status(results_dir).get(run_id, 'pending')


# ============================================================================
# Results summary CSV
# ============================================================================

_RESULTS_HEADER = [
    'run_id', 'dataset', 'epochs', 'seed',
    'final_val_acc', 'final_val_loss',
    'completed_at',
]


def append_result(result: dict, run_cfg: dict, results_dir: Path) -> None:
    path = results_dir / 'results.csv'
    write_header = not path.exists()
    with open(path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=_RESULTS_HEADER)
        if write_header:
            w.writeheader()
        w.writerow({
            'run_id':         result['run_id'],
            'dataset':        run_cfg['dataset'],
            'epochs':         run_cfg['epochs'],
            'seed':           run_cfg['seed'],
            'final_val_acc':  result.get('final_val_acc'),
            'final_val_loss': result.get('final_val_loss'),
            'completed_at':   datetime.now(timezone.utc).isoformat(),
        })


# ============================================================================
# Writer trigger
# ============================================================================

def maybe_trigger_writer(
    results_dir: Path,
    n_completed: int,
    trigger_every: int,
) -> None:
    """
    Invoke src/writer.py when n_completed is a multiple of trigger_every.
    Silently skips if the file does not yet exist (deliverable 7 pending).
    """
    if trigger_every <= 0 or n_completed % trigger_every != 0:
        return
    writer = Path('src') / 'writer.py'
    if not writer.exists():
        logger.debug("Writer not yet available — trigger skipped (n_completed=%d)",
                     n_completed)
        return
    logger.info("Triggering writer (n_completed=%d)", n_completed)
    subprocess.run(
        [sys.executable, str(writer), '--results-dir', str(results_dir),
         '--n-completed', str(n_completed)],
        check=False,
    )


# ============================================================================
# Main orchestration loop
# ============================================================================

def orchestrate(
    config_path: str | Path,
    results_dir: str | Path = 'results',
    resume: bool = True,
    dry_run: bool = False,
    _train_fn: Optional[Callable] = None,
) -> dict:
    """
    Run the full experiment grid described in config_path.

    Parameters
    ----------
    config_path : path to the YAML grid config
    results_dir : root output directory (default: 'results/')
    resume      : pass resume=True to each train_run call
    dry_run     : print the planned runs without executing them
    _train_fn   : injectable replacement for train_run — used in unit tests only

    Returns
    -------
    dict with 'n_total', 'n_completed', 'n_skipped', 'n_failed'
    """
    results_dir   = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    runs_dir      = results_dir / 'runs'
    runs_dir.mkdir(exist_ok=True)

    defaults, run_configs = load_grid(config_path)
    trigger_every = int(defaults.get('writer_trigger_every', 5))
    train_fn      = _train_fn if _train_fn is not None else train_run

    n_completed = n_skipped = n_failed = 0
    n_total     = len(run_configs)

    logger.info("Grid loaded: %d runs from %s", n_total, config_path)

    for i, cfg in enumerate(run_configs, start=1):
        run_id  = cfg['run_id']
        out_dir = runs_dir / run_id

        status = get_run_status(run_id, results_dir)

        if dry_run:
            print(f"[{i:02d}/{n_total}]  {run_id}  (status={status})")
            continue

        if status == 'completed':
            logger.info("[%d/%d] Skipping %s — already complete",
                        i, n_total, run_id)
            n_skipped += 1
            continue

        logger.info("[%d/%d] Starting %s", i, n_total, run_id)
        set_run_status(run_id, 'running', results_dir)

        try:
            result = train_fn(cfg, output_dir=out_dir, resume=resume)
        except Exception as exc:
            logger.error("Run %s FAILED: %s", run_id, exc, exc_info=True)
            set_run_status(run_id, 'failed', results_dir)
            n_failed += 1
            continue

        if result.get('final_val_acc') is None:
            # train_run returned early because the run was already complete
            logger.info("[%d/%d] %s was already complete (detected by state.json)",
                        i, n_total, run_id)
        else:
            logger.info("[%d/%d] %s finished — val_acc=%.4f",
                        i, n_total, run_id, result['final_val_acc'])

        set_run_status(run_id, 'completed', results_dir)
        append_result(result, cfg, results_dir)
        n_completed += 1

        maybe_trigger_writer(results_dir, n_completed, trigger_every)

    summary = {
        'n_total':     n_total,
        'n_completed': n_completed,
        'n_skipped':   n_skipped,
        'n_failed':    n_failed,
    }
    logger.info("Grid done — %s", summary)
    return summary


# ============================================================================
# CLI entry-point
# ============================================================================

def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description='Run the thesis experiment grid from a YAML config.'
    )
    p.add_argument('--config',      default='config/experiment_grid.yaml',
                   help='Path to the YAML grid config (default: config/experiment_grid.yaml)')
    p.add_argument('--results-dir', default='results',
                   help='Root directory for all run outputs (default: results/)')
    p.add_argument('--resume',      action='store_true', default=True,
                   help='Resume interrupted runs from their last epoch (default: True)')
    p.add_argument('--no-resume',   action='store_false', dest='resume',
                   help='Restart every run from scratch')
    p.add_argument('--dry-run',     action='store_true',
                   help='Print the planned runs without executing them')
    p.add_argument('--log-level',   default='INFO',
                   choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                   help='Logging verbosity (default: INFO)')
    return p.parse_args(argv)


if __name__ == '__main__':
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s  %(levelname)s  %(name)s  %(message)s',
        datefmt='%H:%M:%S',
    )
    orchestrate(
        config_path=args.config,
        results_dir=args.results_dir,
        resume=args.resume,
        dry_run=args.dry_run,
    )
