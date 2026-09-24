"""
src/training.py — Training loop with dense metric collection.

Responsibilities:
  1. Load a MedMNIST dataset (PathMNIST / OrganAMNIST / BloodMNIST).
  2. Build a ResNet-18 adapted to the dataset's number of classes.
  3. Train with SGD (supervisor's validated hyperparameters).
  4. Checkpoint at K log-spaced global steps (denser early).
  5. At each checkpoint: build WeightSnapshot in-memory → compute LMC +
     SampEn₁D per layer group → append to metrics.csv.
     No per-step .pt file is written; only one resume checkpoint per epoch.
  6. Support resuming from the last completed epoch via state.json.

Output layout (one run):
  output_dir/
    checkpoints/
      resume_epoch_NNN.pt   ← latest epoch only (previous deleted)
    metrics.csv             ← one row per (checkpoint × group)
    history.csv             ← one row per epoch (train + val)
    config.json             ← frozen config for provenance
    state.json              ← mutable resume state
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms

from src.complexity import lmc_complexity, sample_entropy_1d
from src.weights import build_param_type_map, snapshot_from_state_dict

logger = logging.getLogger(__name__)


# ============================================================================
# CSV schemas
# ============================================================================

METRICS_HEADER = [
    'run_id', 'step', 'epoch',
    'group', 'n_flat_samples', 'n_units',
    'lmc_entropy', 'lmc_disequilibrium', 'lmc_complexity',
    'sampen_1d',
    'l2_mean', 'l2_std',
]

HISTORY_HEADER = ['step', 'epoch', 'phase', 'loss', 'acc']


# ============================================================================
# Dataset / model builders
# ============================================================================

def make_loaders(
    dataset_name: str,
    image_size: int = 64,
    batch_size: int = 64,
    num_workers: int = 4,
    data_root: str = './data',
    train_fraction: float = 1.0,
    data_seed: int = 0,
) -> tuple[DataLoader, DataLoader, int]:
    """
    Build train and val DataLoaders for a MedMNIST dataset.

    Returns (train_loader, val_loader, n_classes).
    Images are resized to image_size × image_size and normalised with
    ImageNet statistics (grey → RGB by channel repeat before norm).

    Parameters
    ----------
    train_fraction : float in (0, 1]
        Fraction of the training set to use. Values < 1.0 induce overfit
        regimes by reducing data while keeping model capacity fixed. The
        validation set is never subsampled.
    data_seed : int
        Seed for the (deterministic) subsampling RNG. Two runs with the same
        (dataset_name, train_fraction, data_seed) see exactly the same
        training indices, regardless of the model/training seed.
    """
    import medmnist
    from torch.utils.data import Subset

    if not 0.0 < train_fraction <= 1.0:
        raise ValueError(
            f"train_fraction must be in (0, 1], got {train_fraction}"
        )

    info_key = dataset_name.lower()
    info = medmnist.INFO.get(info_key)
    if info is None:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Available keys: {sorted(medmnist.INFO.keys())}"
        )
    n_classes = len(info['label'])

    to_rgb = transforms.Lambda(
        lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x
    )
    norm = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        to_rgb,
        norm,
    ])
    val_tf = transforms.Compose([
        transforms.ToTensor(),
        to_rgb,
        norm,
    ])

    DatasetClass = getattr(medmnist, dataset_name)
    train_ds = DatasetClass(split='train', transform=train_tf,
                             download=True, root=data_root, size=image_size)
    val_ds   = DatasetClass(split='val',   transform=val_tf,
                             download=True, root=data_root, size=image_size)

    # Deterministic subsampling of the training set.  The RNG is local to this
    # call (independent of torch.manual_seed) so that two runs with the same
    # (dataset, train_fraction, data_seed) see exactly the same indices.
    if train_fraction < 1.0:
        n_total = len(train_ds)
        n_keep = max(1, int(round(n_total * train_fraction)))
        rng = np.random.default_rng(data_seed)
        idx = rng.permutation(n_total)[:n_keep]
        idx.sort()  # preserve original order for any downstream sequential ops
        train_ds = Subset(train_ds, idx.tolist())
        logger.info(
            "make_loaders: subsampled train set to %d/%d (%.1f%%, data_seed=%d)",
            n_keep, n_total, 100.0 * train_fraction, data_seed,
        )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size * 2, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader, n_classes


def make_model(n_classes: int, architecture: str = "resnet18") -> nn.Module:
    """Build a torchvision classifier with the final head adapted to n_classes.

    Supported architectures:
      - "resnet18"    (default; baseline used in Phases 0–2)
      - "resnet34"    (deeper residual, same family — Phase 3)
      - "densenet121" (dense connectivity — Phase 3)

    Pre-trained weights are NOT loaded; all models train from scratch.
    """
    arch = architecture.lower()
    if arch == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, n_classes)
    elif arch == "resnet34":
        model = models.resnet34(weights=None)
        model.fc = nn.Linear(model.fc.in_features, n_classes)
    elif arch == "densenet121":
        model = models.densenet121(weights=None)
        model.classifier = nn.Linear(model.classifier.in_features, n_classes)
    else:
        raise ValueError(
            f"Unsupported architecture '{architecture}'. "
            f"Choose from: resnet18, resnet34, densenet121."
        )
    return model


# ============================================================================
# Checkpoint schedule
# ============================================================================

def checkpoint_schedule(total_steps: int, n_checkpoints: int) -> set[int]:
    """
    Return a set of global step numbers at which to record snapshots.

    Steps are logarithmically spaced so that early training (where weight
    distributions change fastest) is sampled more densely.  Step 1 is always
    included.  The returned set has at most n_checkpoints elements.
    """
    if total_steps <= n_checkpoints:
        return set(range(1, total_steps + 1))
    log_steps = np.unique(
        np.round(np.logspace(0, np.log10(total_steps), n_checkpoints)).astype(int)
    )
    log_steps = log_steps[log_steps <= total_steps]
    return {int(s) for s in log_steps}


# ============================================================================
# Helpers
# ============================================================================

def _accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Fraction of correctly classified examples in a batch."""
    preds = logits.argmax(dim=1)
    return float((preds == labels.view(-1)).float().mean())


def _append_csv(path: Path, rows: list[dict], header: list[str]) -> None:
    write_header = not path.exists()
    with open(path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=header)
        if write_header:
            w.writeheader()
        w.writerows(rows)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Return (mean_loss, mean_acc) over the loader."""
    model.eval()
    total_loss = total_acc = 0.0
    n_batches = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.view(-1).long().to(device)
        logits = model(images)
        total_loss += criterion(logits, labels).item()
        total_acc  += _accuracy(logits, labels)
        n_batches  += 1
    model.train()
    if n_batches == 0:
        return 0.0, 0.0
    return total_loss / n_batches, total_acc / n_batches


# ============================================================================
# Metric extraction
# ============================================================================

def extract_metrics_rows(snapshot, run_id: str) -> list[dict]:
    """
    Compute LMC + SampEn₁D + L2 norm statistics for every group in
    the snapshot.  Returns one dict per group, ready for metrics.csv.
    """
    rows = []
    for grp in snapshot.groups():
        flat = snapshot.flat_weights[grp]
        l2   = snapshot.l2_norms[grp]

        lmc  = lmc_complexity(flat)
        se1d = sample_entropy_1d(flat)

        rows.append({
            'run_id':             run_id,
            'step':               snapshot.step,
            'epoch':              round(snapshot.epoch, 6),
            'group':              grp,
            'n_flat_samples':     len(flat),
            'n_units':            len(l2),
            'lmc_entropy':        lmc['entropy'],
            'lmc_disequilibrium': lmc['disequilibrium'],
            'lmc_complexity':     lmc['complexity'],
            'sampen_1d':          float(se1d) if np.isfinite(se1d) else None,
            'l2_mean':            float(np.mean(l2)),
            'l2_std':             float(np.std(l2)),
        })
    return rows


# ============================================================================
# Main training entry-point
# ============================================================================

def train_run(
    config: dict,
    output_dir: str | Path,
    resume: bool = True,
    _loaders: Optional[tuple] = None,
) -> dict:
    """
    Run one training experiment and collect weight-complexity metrics.

    Parameters
    ----------
    config       : experiment configuration (see keys below)
    output_dir   : root directory for this run's outputs
    resume       : if True, resume from last completed epoch when state.json exists
    _loaders     : optional (train_loader, val_loader, n_classes) tuple that
                   bypasses make_loaders — used in unit tests only

    Required config keys
    --------------------
    run_id         : str
    dataset        : str  — 'PathMNIST' | 'OrganAMNIST' | 'BloodMNIST'
    epochs         : int
    seed           : int

    Optional config keys (with defaults matching supervisor's experiments)
    -----------------------------------------------------------------------
    lr             : float  (0.001)
    momentum       : float  (0.9)
    weight_decay   : float  (0.0001)
    batch_size     : int    (64)
    image_size     : int    (64)
    n_checkpoints  : int    (100)
    max_flat_sample: int    (10_000)
    num_workers    : int    (4)
    data_root      : str    ('./data')

    Returns
    -------
    dict with 'run_id', 'final_val_acc', 'final_val_loss', 'metrics_path',
    'history_path'.  'final_val_acc' / 'final_val_loss' are None if training
    was skipped because it had already completed.
    """
    out      = Path(output_dir)
    ckpt_dir = out / 'checkpoints'
    out.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(exist_ok=True)

    run_id        = config['run_id']
    total_epochs  = config['epochs']
    seed          = config['seed']
    lr            = config.get('lr',             0.001)
    momentum      = config.get('momentum',       0.9)
    weight_decay  = config.get('weight_decay',   0.0001)
    batch_size    = config.get('batch_size',     64)
    image_size    = config.get('image_size',     64)
    n_checkpoints = config.get('n_checkpoints',  100)
    max_flat      = config.get('max_flat_sample', 10_000)
    num_workers   = config.get('num_workers',    4)
    data_root     = config.get('data_root',      './data')
    train_fraction = config.get('train_fraction', 1.0)
    data_seed     = config.get('data_seed',      0)
    architecture  = config.get('architecture',   'resnet18')

    metrics_path = out / 'metrics.csv'
    history_path = out / 'history.csv'
    state_path   = out / 'state.json'
    config_path  = out / 'config.json'

    if not config_path.exists():
        config_path.write_text(json.dumps(config, indent=2))

    # Seeds — set before building model / dataloaders so shuffling is reproducible
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info("run_id=%s  device=%s", run_id, device)

    # Data
    if _loaders is not None:
        train_loader, val_loader, n_classes = _loaders
    else:
        train_loader, val_loader, n_classes = make_loaders(
            config['dataset'], image_size, batch_size, num_workers, data_root,
            train_fraction=train_fraction, data_seed=data_seed,
        )

    # Model + optimizer + loss
    model         = make_model(n_classes, architecture=architecture).to(device)
    optimizer     = torch.optim.SGD(model.parameters(), lr=lr,
                                     momentum=momentum, weight_decay=weight_decay)
    criterion     = nn.CrossEntropyLoss()
    param_type_map = build_param_type_map(model)

    # Log-spaced checkpoint schedule
    steps_per_epoch = len(train_loader)
    total_steps     = total_epochs * steps_per_epoch
    ckpt_steps      = checkpoint_schedule(total_steps, n_checkpoints)

    # Resume state
    start_epoch:     int      = 0
    global_step:     int      = 0
    processed_steps: set[int] = set()

    if resume and state_path.exists():
        state = json.loads(state_path.read_text())
        if state.get('last_completed_epoch', 0) >= total_epochs:
            logger.info("run_id=%s already complete — skipping", run_id)
            return {
                'run_id':         run_id,
                'final_val_acc':  None,
                'final_val_loss': None,
                'metrics_path':   str(metrics_path),
                'history_path':   str(history_path),
            }
        start_epoch     = state.get('last_completed_epoch', 0)
        global_step     = state.get('last_step', 0)
        processed_steps = set(state.get('processed_steps', []))

        resume_ckpt = ckpt_dir / f'resume_epoch_{start_epoch:03d}.pt'
        if resume_ckpt.exists():
            raw = torch.load(resume_ckpt, map_location=device, weights_only=True)
            model.load_state_dict(raw['model'])
            optimizer.load_state_dict(raw['optimizer'])
            logger.info("Resumed from epoch %d (step %d)", start_epoch, global_step)

    # Training
    model.train()
    val_loss = val_acc = 0.0

    for epoch in range(start_epoch, total_epochs):
        epoch_loss = epoch_acc = 0.0

        for batch_idx, (images, labels) in enumerate(train_loader):
            global_step += 1
            images = images.to(device)
            labels = labels.view(-1).long().to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc  += _accuracy(logits, labels)

            # In-memory metric snapshot (no .pt file written per step)
            if global_step in ckpt_steps and global_step not in processed_steps:
                epoch_frac = epoch + (batch_idx + 1) / steps_per_epoch
                snap = snapshot_from_state_dict(
                    model.state_dict(), param_type_map,
                    step=global_step, epoch=epoch_frac,
                    run_id=run_id,
                    max_flat_sample=max_flat,
                    rng=rng,
                    include_bias_group=False,  # biases excluded per supervisor's convention
                )
                _append_csv(metrics_path,
                            extract_metrics_rows(snap, run_id),
                            METRICS_HEADER)
                processed_steps.add(global_step)
                logger.debug("step=%d  epoch=%.3f  groups=%s",
                             global_step, epoch_frac, snap.groups())

        # End-of-epoch evaluation
        n_batches  = len(train_loader)
        train_loss = epoch_loss / n_batches
        train_acc  = epoch_acc  / n_batches
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        logger.info(
            "epoch=%d/%d  train_loss=%.4f  train_acc=%.4f  "
            "val_loss=%.4f  val_acc=%.4f",
            epoch + 1, total_epochs, train_loss, train_acc, val_loss, val_acc,
        )

        _append_csv(history_path, [
            {'step': global_step, 'epoch': epoch + 1, 'phase': 'train',
             'loss': train_loss, 'acc': train_acc},
            {'step': global_step, 'epoch': epoch + 1, 'phase': 'val',
             'loss': val_loss,   'acc': val_acc},
        ], HISTORY_HEADER)

        # Persist resume checkpoint — keep only the latest epoch to save disk
        new_ckpt  = ckpt_dir / f'resume_epoch_{epoch + 1:03d}.pt'
        prev_ckpt = ckpt_dir / f'resume_epoch_{epoch:03d}.pt'
        torch.save({'model': model.state_dict(),
                    'optimizer': optimizer.state_dict()}, new_ckpt)
        if prev_ckpt.exists():
            prev_ckpt.unlink()

        state_path.write_text(json.dumps({
            'last_completed_epoch': epoch + 1,
            'last_step':            global_step,
            'processed_steps':      sorted(processed_steps),
        }, indent=2))

    return {
        'run_id':         run_id,
        'final_val_acc':  val_acc,
        'final_val_loss': val_loss,
        'metrics_path':   str(metrics_path),
        'history_path':   str(history_path),
    }
