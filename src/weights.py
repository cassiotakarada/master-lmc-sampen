"""
src/weights.py — Layer grouping and weight snapshot logic.

Responsibilities:
  1. Identify the group (conv | dense | bn_gamma | bn_beta | bias) of every
     parameter in an nn.Module by inspecting the actual module type — no
     name-pattern matching.
  2. Build a `WeightSnapshot` from a state_dict: stores the data needed to
     compute all complexity measures without keeping the full checkpoint in
     memory.

Group definitions (compatible with supervisor's param_types.json):
  conv     — weight tensors of Conv1d / Conv2d / Conv3d layers
  dense    — weight tensors of Linear layers
  bn_gamma — weight (γ) tensors of BatchNorm layers
  bn_beta  — bias  (β) tensors of BatchNorm layers
  bias     — all other bias parameters (conv biases, linear biases)

Biases of non-BN layers are explicitly separated into their own group.
Activation / Pooling / Flatten layers have no learnable parameters and are
ignored automatically.

WeightSnapshot contents (in-memory, not serialised):
  flat_weights  — systematic sample (≤ max_flat_sample, fixed step, original
                  flatten order preserved — see decision D3) of concatenated
                  weights per group.  Used for LMC and SampEn₁D.
  l2_norms      — per-filter (conv) or per-neuron (dense / bn) L2 norms,
                  concatenated across all parameters in the group.
                  Used to build temporal series across checkpoints.
  weight_2d     — 2D matrix representation of each parameter:
                    conv  (C_out, C_in, kH, kW) → (C_out, C_in·kH·kW)
                    dense (out, in)              → (out, in)  [unchanged]
                    bn    (C,)                   → (1, C)
                  Used for SampEn₂D.
  param_names   — dict of group → [param_name, ...] for provenance tracking.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


# ============================================================================
# Constants
# ============================================================================

GROUPS = ('conv', 'dense', 'bn_gamma', 'bn_beta', 'bias')


# ============================================================================
# Module-type identification (mirrors supervisor's identify_layer_type)
# ============================================================================

def identify_module_type(module: nn.Module) -> str:
    """Return a human-readable string for the module type."""
    if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        return 'Convolutional'
    if isinstance(module, nn.Linear):
        return 'Linear'
    if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                           nn.GroupNorm, nn.LayerNorm, nn.InstanceNorm2d)):
        return 'BatchNorm'
    if isinstance(module, (nn.ReLU, nn.GELU, nn.SiLU, nn.Sigmoid, nn.Tanh)):
        return 'Activation'
    if isinstance(module, (nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d,
                           nn.AvgPool2d, nn.AdaptiveAvgPool2d,
                           nn.AdaptiveAvgPool3d)):
        return 'Pooling'
    if isinstance(module, nn.Flatten):
        return 'Flatten'
    if isinstance(module, nn.Embedding):
        return 'Embedding'
    return 'Other'


def build_param_type_map(model: nn.Module) -> dict[str, str]:
    """
    Walk model.named_modules() and map every parameter full-name to its
    module's type string.  Equivalent to the supervisor's param_type_mapping
    construction in medmnist_weights_extraction.py.

    Returns a dict {full_param_name: module_type_string}.
    """
    mapping: dict[str, str] = {}
    for module_name, module in model.named_modules():
        mtype = identify_module_type(module)
        for param_name, _ in module.named_parameters(recurse=False):
            full = f"{module_name}.{param_name}" if module_name else param_name
            mapping[full] = mtype
    return mapping


def save_param_type_map(mapping: dict[str, str], path: str | Path) -> None:
    """Persist param_type_map to a JSON file (mirrors supervisor's run output)."""
    with open(path, 'w') as f:
        json.dump(mapping, f, indent=2)


def load_param_type_map(path: str | Path) -> dict[str, str]:
    """Load a param_type_map from a JSON file."""
    with open(path) as f:
        return json.load(f)


# ============================================================================
# Parameter → group assignment
# ============================================================================

def param_group(name: str, module_type: str) -> str:
    """
    Map a single parameter to one of GROUPS based on its module type and name.

    Parameters
    ----------
    name        : full parameter name, e.g. 'layer2.0.downsample.0.bias'
    module_type : string from identify_module_type / param_type_map

    Returns
    -------
    One of GROUPS, or 'other' for unrecognised parameters.
    """
    is_bias = name.endswith('.bias') or name == 'bias'

    if module_type == 'Convolutional':
        return 'bias' if is_bias else 'conv'
    if module_type == 'Linear':
        return 'bias' if is_bias else 'dense'
    if module_type == 'BatchNorm':
        return 'bn_beta' if is_bias else 'bn_gamma'

    # Activation, Pooling, Flatten: no learnable parameters in standard nets.
    # Anything else with a bias still goes into the catch-all group.
    return 'bias' if is_bias else 'other'


# ============================================================================
# Tensor transformations
# ============================================================================

def to_2d(tensor: torch.Tensor) -> np.ndarray:
    """
    Reshape a parameter tensor to a 2D numpy array for SampEn₂D.

      Conv  (C_out, C_in, kH, kW) → (C_out, C_in·kH·kW)
      Dense (out, in)              → (out, in)   [unchanged]
      1-D   (C,)                   → (1, C)
      Other                        → (prod(dims[:-1]), last_dim)
    """
    arr = tensor.detach().cpu().numpy().astype(np.float64)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    if arr.ndim == 2:
        return arr
    # ndim >= 3: fold all dims except the first (output dimension) into one
    return arr.reshape(arr.shape[0], -1)


def systematic_subsample(arr: np.ndarray, max_size: int) -> np.ndarray:
    """
    Subsample a 1-D array to at most `max_size` elements with a FIXED STEP,
    preserving the original ordering (thesis methodology, decision D3).

    Ordering matters because SampEn₁D is a measure of sequential structure:
    the flatten order groups Conv2d weights by output neuron (D2), so each
    contiguous block of C_in·kH·kW values is one whole filter.  Random
    subsampling shuffles the vector and collapses SampEn₁D into a statistic
    of the marginal weight distribution, discarding that structure.

    Returns `arr` unchanged when it already fits within `max_size`.
    """
    n = len(arr)
    if n <= max_size:
        return arr
    step = n // max_size
    return arr[::step][:max_size]


def l2_norm_per_unit(tensor: torch.Tensor) -> np.ndarray:
    """
    Compute per-filter (conv) or per-neuron (dense / bn) L2 norms.

      Conv  (C_out, C_in, kH, kW) → L2 over (C_in, kH, kW) → shape (C_out,)
      Dense (out, in)              → L2 over (in,)            → shape (out,)
      1-D   (C,)                   → absolute values           → shape (C,)
    """
    arr = tensor.detach().cpu().numpy().astype(np.float64)
    if arr.ndim == 1:
        return np.abs(arr)   # BN gamma/beta: per-channel magnitude
    # ndim ≥ 2: norm of each row of the 2D view
    mat = arr.reshape(arr.shape[0], -1)
    return np.linalg.norm(mat, axis=1)


# ============================================================================
# WeightSnapshot dataclass
# ============================================================================

@dataclass
class WeightSnapshot:
    """
    Lightweight in-memory snapshot of a model's weights at one checkpoint.

    Fields
    ------
    step         : global training step number
    epoch        : epoch number (may be fractional if checkpointing mid-epoch)
    run_id       : identifier of the experiment run (for provenance)
    flat_weights : dict[group → 1-D float64 array] — systematic sample of
                   concatenated weights (≤ max_flat_sample per group, fixed
                   step, original flatten order preserved).
                   Used for LMC and SampEn₁D.
    l2_norms     : dict[group → 1-D float64 array] — per-filter / per-neuron
                   L2 norms, concatenated across all parameters in the group.
                   Used to build temporal series across snapshots.
    weight_2d    : dict[group → list[(param_name, 2-D float64 array)]]
                   2D views of each parameter.  Used for SampEn₂D.
    param_names  : dict[group → list[str]] — names for provenance tracking.
    """
    step: int
    epoch: float
    run_id: str
    flat_weights: dict[str, np.ndarray] = field(default_factory=dict)
    l2_norms: dict[str, np.ndarray] = field(default_factory=dict)
    weight_2d: dict[str, list[tuple[str, np.ndarray]]] = field(default_factory=dict)
    param_names: dict[str, list[str]] = field(default_factory=dict)

    def groups(self) -> list[str]:
        """Return the non-empty groups present in this snapshot."""
        return list(self.flat_weights.keys())

    def n_params(self, group: str) -> int:
        """Total number of weight elements (before sampling) would need recount;
        returns the flat sample size instead."""
        return len(self.flat_weights.get(group, []))

    def summary(self) -> dict[str, dict]:
        """Return a compact dict describing each group's sizes."""
        out = {}
        for g in self.groups():
            out[g] = {
                'flat_sample_size': len(self.flat_weights[g]),
                'n_units': len(self.l2_norms.get(g, [])),
                'n_matrices': len(self.weight_2d.get(g, [])),
                'params': self.param_names.get(g, []),
            }
        return out


# ============================================================================
# Snapshot builder
# ============================================================================

def snapshot_from_state_dict(
    state_dict: dict[str, torch.Tensor],
    param_type_map: dict[str, str],
    step: int,
    epoch: float,
    run_id: str = '',
    max_flat_sample: int = 10_000,
    rng: Optional[np.random.Generator] = None,
    include_bias_group: bool = True,
) -> WeightSnapshot:
    """
    Build a WeightSnapshot from a loaded state_dict.

    Parameters
    ----------
    state_dict      : model.state_dict() — maps param_name → tensor
    param_type_map  : maps param_name → module type string
                      (built by build_param_type_map or loaded from JSON)
    step            : global training step at this checkpoint
    epoch           : epoch number (can be fractional)
    run_id          : experiment run identifier
    max_flat_sample : maximum number of weight elements to store per group
                      for LMC / SampEn₁D.  Mirrors supervisor's convention
                      of sampling up to 10,000 elements.  Subsampling is
                      systematic (fixed step), preserving flatten order (D3).
    rng             : accepted for backward compatibility; no longer used for
                      the flat sample, which is now deterministic.
    include_bias_group : if False, skip the 'bias' group entirely
                         (matching supervisor's original code that skips biases)

    Returns
    -------
    WeightSnapshot with flat_weights, l2_norms, weight_2d, param_names populated
    for each non-empty group.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # Accumulate tensors per group
    buckets: dict[str, list[tuple[str, torch.Tensor]]] = {g: [] for g in GROUPS}

    for name, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        if tensor.numel() == 0:
            continue
        mtype = param_type_map.get(name, 'Other')
        grp = param_group(name, mtype)
        if grp == 'other':
            continue
        if grp == 'bias' and not include_bias_group:
            continue
        buckets[grp].append((name, tensor))

    snap = WeightSnapshot(step=step, epoch=epoch, run_id=run_id)

    for grp, items in buckets.items():
        if not items:
            continue

        snap.param_names[grp] = [name for name, _ in items]

        # Flat sample: concatenate all weights, then subsample if needed.
        # Subsampling is SYSTEMATIC (fixed step), preserving the original
        # flatten order — see thesis methodology, decision D3.  Random
        # subsampling would shuffle the vector and reduce SampEn₁D to a
        # measure of the marginal weight distribution.
        all_flat = np.concatenate([
            t.detach().cpu().numpy().ravel().astype(np.float64)
            for _, t in items
        ])
        snap.flat_weights[grp] = systematic_subsample(all_flat, max_flat_sample)

        # L2 norms: per-filter or per-neuron, concatenated across all params
        snap.l2_norms[grp] = np.concatenate([
            l2_norm_per_unit(t) for _, t in items
        ])

        # 2D weight matrices: one per parameter
        snap.weight_2d[grp] = [
            (name, to_2d(t)) for name, t in items
        ]

    return snap


def snapshot_from_checkpoint(
    checkpoint_path: str | Path,
    param_type_map: dict[str, str],
    step: int,
    epoch: float,
    run_id: str = '',
    max_flat_sample: int = 10_000,
    rng: Optional[np.random.Generator] = None,
    map_location: str = 'cpu',
) -> WeightSnapshot:
    """
    Load a .pt/.pth checkpoint from disk and build a WeightSnapshot.

    The checkpoint must contain either a raw state_dict or a dict with
    a 'state_dict' key (both conventions are handled).
    """
    raw = torch.load(checkpoint_path, map_location=map_location,
                     weights_only=True)
    # Handle wrapped checkpoints {'state_dict': {...}, 'epoch': ..., ...}
    state_dict = raw.get('state_dict', raw) if isinstance(raw, dict) else raw
    return snapshot_from_state_dict(
        state_dict=state_dict,
        param_type_map=param_type_map,
        step=step,
        epoch=epoch,
        run_id=run_id,
        max_flat_sample=max_flat_sample,
        rng=rng,
    )
