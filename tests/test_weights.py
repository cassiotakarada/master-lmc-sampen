"""
tests/test_weights.py — Unit tests for src/weights.py.

Covers:
  - identify_module_type: all recognized module types → 'Other'
  - param_group: correct group for every (module_type, param_name) pair
  - build_param_type_map: correct mapping for a small model
  - save_param_type_map / load_param_type_map: JSON round-trip
  - to_2d: correct reshape for 1-D, 2-D, 4-D tensors
  - l2_norm_per_unit: correct norms for 1-D, 2-D, 4-D tensors
  - WeightSnapshot: fields, groups(), n_params(), summary()
  - snapshot_from_state_dict: full pipeline with known model
  - snapshot_from_checkpoint: raw and wrapped .pt files
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.weights import (
    GROUPS,
    build_param_type_map,
    identify_module_type,
    l2_norm_per_unit,
    load_param_type_map,
    param_group,
    save_param_type_map,
    snapshot_from_checkpoint,
    snapshot_from_state_dict,
    to_2d,
    WeightSnapshot,
)


# ============================================================================
# Shared tiny model
# ============================================================================

class SmallNet(nn.Module):
    """Conv2d + BatchNorm2d + Linear — covers all non-bias groups."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=3, bias=True)
        self.bn   = nn.BatchNorm2d(8)
        self.fc   = nn.Linear(8, 4, bias=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.fc(x.mean(dim=(2, 3)))


# ============================================================================
# identify_module_type
# ============================================================================

class TestIdentifyModuleType:

    def test_conv1d(self):
        assert identify_module_type(nn.Conv1d(1, 4, 3)) == 'Convolutional'

    def test_conv2d(self):
        assert identify_module_type(nn.Conv2d(1, 4, 3)) == 'Convolutional'

    def test_conv3d(self):
        assert identify_module_type(nn.Conv3d(1, 4, 3)) == 'Convolutional'

    def test_linear(self):
        assert identify_module_type(nn.Linear(4, 2)) == 'Linear'

    def test_batchnorm1d(self):
        assert identify_module_type(nn.BatchNorm1d(4)) == 'BatchNorm'

    def test_batchnorm2d(self):
        assert identify_module_type(nn.BatchNorm2d(4)) == 'BatchNorm'

    def test_batchnorm3d(self):
        assert identify_module_type(nn.BatchNorm3d(4)) == 'BatchNorm'

    def test_groupnorm(self):
        assert identify_module_type(nn.GroupNorm(2, 4)) == 'BatchNorm'

    def test_layernorm(self):
        assert identify_module_type(nn.LayerNorm(4)) == 'BatchNorm'

    def test_instancenorm2d(self):
        assert identify_module_type(nn.InstanceNorm2d(4)) == 'BatchNorm'

    def test_relu(self):
        assert identify_module_type(nn.ReLU()) == 'Activation'

    def test_gelu(self):
        assert identify_module_type(nn.GELU()) == 'Activation'

    def test_silu(self):
        assert identify_module_type(nn.SiLU()) == 'Activation'

    def test_sigmoid(self):
        assert identify_module_type(nn.Sigmoid()) == 'Activation'

    def test_tanh(self):
        assert identify_module_type(nn.Tanh()) == 'Activation'

    def test_maxpool1d(self):
        assert identify_module_type(nn.MaxPool1d(2)) == 'Pooling'

    def test_maxpool2d(self):
        assert identify_module_type(nn.MaxPool2d(2)) == 'Pooling'

    def test_avgpool2d(self):
        assert identify_module_type(nn.AvgPool2d(2)) == 'Pooling'

    def test_adaptive_avgpool2d(self):
        assert identify_module_type(nn.AdaptiveAvgPool2d((1, 1))) == 'Pooling'

    def test_flatten(self):
        assert identify_module_type(nn.Flatten()) == 'Flatten'

    def test_embedding(self):
        assert identify_module_type(nn.Embedding(10, 4)) == 'Embedding'

    def test_dropout_returns_other(self):
        assert identify_module_type(nn.Dropout()) == 'Other'


# ============================================================================
# param_group
# ============================================================================

class TestParamGroup:

    def test_conv_weight(self):
        assert param_group('conv.weight', 'Convolutional') == 'conv'

    def test_conv_bias(self):
        assert param_group('conv.bias', 'Convolutional') == 'bias'

    def test_linear_weight(self):
        assert param_group('fc.weight', 'Linear') == 'dense'

    def test_linear_bias(self):
        assert param_group('fc.bias', 'Linear') == 'bias'

    def test_bn_weight_is_gamma(self):
        assert param_group('bn.weight', 'BatchNorm') == 'bn_gamma'

    def test_bn_bias_is_beta(self):
        assert param_group('bn.bias', 'BatchNorm') == 'bn_beta'

    def test_other_bias_goes_to_bias_group(self):
        assert param_group('custom.bias', 'Other') == 'bias'

    def test_other_weight_goes_to_other(self):
        assert param_group('custom.weight', 'Other') == 'other'

    def test_toplevel_bias_param_name(self):
        # Param named literally 'bias' (no dot prefix)
        assert param_group('bias', 'Linear') == 'bias'

    def test_activation_has_no_params_falls_to_other(self):
        assert param_group('act.scale', 'Activation') == 'other'

    def test_all_valid_groups_in_GROUPS(self):
        # Every group returned by param_group (except 'other') must be in GROUPS
        cases = [
            ('conv.weight', 'Convolutional'),
            ('conv.bias',   'Convolutional'),
            ('fc.weight',   'Linear'),
            ('fc.bias',     'Linear'),
            ('bn.weight',   'BatchNorm'),
            ('bn.bias',     'BatchNorm'),
            ('x.bias',      'Other'),
        ]
        for name, mtype in cases:
            grp = param_group(name, mtype)
            assert grp in GROUPS, f"param_group('{name}', '{mtype}') = '{grp}' not in GROUPS"


# ============================================================================
# build_param_type_map
# ============================================================================

class TestBuildParamTypeMap:

    def test_all_learnable_params_mapped(self):
        model = SmallNet()
        mapping = build_param_type_map(model)
        for name, _ in model.named_parameters():
            assert name in mapping, f"'{name}' missing from param_type_map"

    def test_conv_params_are_convolutional(self):
        mapping = build_param_type_map(SmallNet())
        assert mapping['conv.weight'] == 'Convolutional'
        assert mapping['conv.bias']   == 'Convolutional'

    def test_bn_params_are_batchnorm(self):
        mapping = build_param_type_map(SmallNet())
        assert mapping['bn.weight'] == 'BatchNorm'
        assert mapping['bn.bias']   == 'BatchNorm'

    def test_linear_params_are_linear(self):
        mapping = build_param_type_map(SmallNet())
        assert mapping['fc.weight'] == 'Linear'
        assert mapping['fc.bias']   == 'Linear'

    def test_buffers_not_in_map(self):
        # Buffers (running_mean, running_var, num_batches_tracked) are NOT params
        mapping = build_param_type_map(SmallNet())
        assert 'bn.running_mean' not in mapping
        assert 'bn.running_var'  not in mapping


# ============================================================================
# save_param_type_map / load_param_type_map
# ============================================================================

class TestParamTypeMapIO:

    def test_round_trip(self):
        mapping = build_param_type_map(SmallNet())
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = Path(f.name)
        try:
            save_param_type_map(mapping, path)
            assert load_param_type_map(path) == mapping
        finally:
            path.unlink(missing_ok=True)

    def test_saved_file_is_valid_json(self):
        mapping = build_param_type_map(SmallNet())
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = Path(f.name)
        try:
            save_param_type_map(mapping, path)
            with open(path) as fh:
                data = json.load(fh)
            assert isinstance(data, dict)
        finally:
            path.unlink(missing_ok=True)


# ============================================================================
# to_2d
# ============================================================================

class TestTo2D:

    def test_1d_tensor_becomes_row_vector(self):
        t = torch.tensor([1.0, 2.0, 3.0, 4.0])
        out = to_2d(t)
        assert out.shape == (1, 4)

    def test_2d_tensor_unchanged(self):
        t = torch.randn(8, 16)
        assert to_2d(t).shape == (8, 16)

    def test_4d_conv_tensor_folded(self):
        # (C_out, C_in, kH, kW) → (C_out, C_in*kH*kW)
        t = torch.randn(8, 3, 3, 3)
        assert to_2d(t).shape == (8, 3 * 3 * 3)

    def test_output_is_float64(self):
        assert to_2d(torch.randn(4, 4).float()).dtype == np.float64

    def test_values_preserved_2d(self):
        t = torch.arange(6.0).reshape(2, 3)
        np.testing.assert_array_almost_equal(to_2d(t), t.numpy())

    def test_values_preserved_4d(self):
        t = torch.ones(2, 3, 2, 2)
        out = to_2d(t)
        assert (out == 1.0).all()


# ============================================================================
# l2_norm_per_unit
# ============================================================================

class TestL2NormPerUnit:

    def test_1d_returns_absolute_values(self):
        t = torch.tensor([-3.0, 0.0, 4.0])
        np.testing.assert_array_almost_equal(l2_norm_per_unit(t), [3.0, 0.0, 4.0])

    def test_2d_row_norms(self):
        # Each row = one neuron; expected: [5.0, 1.0]
        t = torch.tensor([[3.0, 4.0], [0.0, 1.0]])
        np.testing.assert_array_almost_equal(l2_norm_per_unit(t), [5.0, 1.0])

    def test_4d_conv_per_filter_norms(self):
        # (C_out=4, C_in=3, kH=2, kW=2)
        t = torch.zeros(4, 3, 2, 2)
        t[0] = 1.0              # filter 0: 12 ones → norm = sqrt(12)
        t[1, 0, 0, 0] = 5.0    # filter 1: one 5   → norm = 5
        out = l2_norm_per_unit(t)
        assert out.shape == (4,)
        assert abs(out[0] - np.sqrt(12)) < 1e-6
        assert abs(out[1] - 5.0) < 1e-6
        assert out[2] == 0.0 and out[3] == 0.0

    def test_output_dtype_float64(self):
        assert l2_norm_per_unit(torch.randn(8, 3, 3, 3)).dtype == np.float64

    def test_output_nonnegative(self):
        assert (l2_norm_per_unit(torch.randn(16, 8, 3, 3)) >= 0).all()

    def test_output_shape_1d(self):
        assert l2_norm_per_unit(torch.randn(12)).shape == (12,)

    def test_output_shape_2d(self):
        assert l2_norm_per_unit(torch.randn(4, 8)).shape == (4,)

    def test_output_shape_4d(self):
        assert l2_norm_per_unit(torch.randn(8, 3, 3, 3)).shape == (8,)


# ============================================================================
# WeightSnapshot
# ============================================================================

class TestWeightSnapshot:

    def _make_snap(self):
        snap = WeightSnapshot(step=10, epoch=1.0, run_id='test_run')
        snap.flat_weights['conv']  = np.array([1.0, 2.0, 3.0])
        snap.flat_weights['dense'] = np.array([4.0, 5.0])
        snap.l2_norms['conv']      = np.array([1.5, 2.5])
        snap.l2_norms['dense']     = np.array([0.7])
        snap.weight_2d['conv']     = [('conv.weight', np.zeros((4, 9)))]
        snap.weight_2d['dense']    = [('fc.weight', np.zeros((2, 8)))]
        snap.param_names['conv']   = ['conv.weight']
        snap.param_names['dense']  = ['fc.weight']
        return snap

    def test_groups_matches_flat_weight_keys(self):
        snap = self._make_snap()
        assert set(snap.groups()) == {'conv', 'dense'}

    def test_n_params_returns_sample_size(self):
        snap = self._make_snap()
        assert snap.n_params('conv')  == 3
        assert snap.n_params('dense') == 2

    def test_n_params_missing_group_returns_zero(self):
        assert self._make_snap().n_params('bn_gamma') == 0

    def test_summary_flat_sample_size(self):
        s = self._make_snap().summary()
        assert s['conv']['flat_sample_size'] == 3

    def test_summary_n_units(self):
        s = self._make_snap().summary()
        assert s['conv']['n_units'] == 2

    def test_summary_n_matrices(self):
        s = self._make_snap().summary()
        assert s['conv']['n_matrices'] == 1

    def test_summary_params(self):
        s = self._make_snap().summary()
        assert s['conv']['params'] == ['conv.weight']

    def test_default_fields_are_empty_dicts(self):
        snap = WeightSnapshot(step=0, epoch=0.0, run_id='x')
        assert snap.flat_weights == {}
        assert snap.l2_norms     == {}
        assert snap.weight_2d    == {}
        assert snap.param_names  == {}


# ============================================================================
# snapshot_from_state_dict
# ============================================================================

class TestSnapshotFromStateDict:

    def setup_method(self):
        torch.manual_seed(0)
        self.model          = SmallNet()
        self.state_dict     = self.model.state_dict()
        self.param_type_map = build_param_type_map(self.model)
        self.rng            = np.random.default_rng(42)

    def _snap(self, **kw):
        defaults = dict(state_dict=self.state_dict,
                        param_type_map=self.param_type_map,
                        step=1, epoch=0.5, run_id='test', rng=self.rng)
        defaults.update(kw)
        return snapshot_from_state_dict(**defaults)

    def test_step_epoch_run_id_stored(self):
        snap = self._snap(step=5, epoch=2.0, run_id='myrun')
        assert snap.step == 5 and snap.epoch == 2.0 and snap.run_id == 'myrun'

    def test_conv_group_present(self):
        assert 'conv' in self._snap().flat_weights

    def test_dense_group_present(self):
        assert 'dense' in self._snap().flat_weights

    def test_bn_gamma_group_present(self):
        assert 'bn_gamma' in self._snap().flat_weights

    def test_bn_beta_group_present(self):
        assert 'bn_beta' in self._snap().flat_weights

    def test_bias_group_present_by_default(self):
        assert 'bias' in self._snap().flat_weights

    def test_bias_group_absent_when_excluded(self):
        assert 'bias' not in self._snap(include_bias_group=False).flat_weights

    def test_flat_weights_are_1d_float64(self):
        for g, arr in self._snap().flat_weights.items():
            assert arr.ndim == 1,            f"Group {g}: expected 1-D"
            assert arr.dtype == np.float64,  f"Group {g}: expected float64"

    def test_l2_norms_are_1d_nonnegative(self):
        for g, arr in self._snap().l2_norms.items():
            assert arr.ndim == 1,           f"Group {g}: expected 1-D"
            assert (arr >= 0).all(),        f"Group {g}: norms must be ≥ 0"

    def test_weight_2d_entries_are_2d_float64(self):
        for g, pairs in self._snap().weight_2d.items():
            for name, mat in pairs:
                assert mat.ndim == 2,           f"{name}: expected 2-D"
                assert mat.dtype == np.float64, f"{name}: expected float64"

    def test_conv_2d_shape_folded(self):
        # Conv2d(3, 8, 3) weight: (8, 3, 3, 3) → (8, 27)
        mats = dict(self._snap().weight_2d['conv'])
        assert mats['conv.weight'].shape == (8, 27)

    def test_dense_2d_shape_unchanged(self):
        # Linear(8, 4) weight: (4, 8) → (4, 8)
        mats = dict(self._snap().weight_2d['dense'])
        assert mats['fc.weight'].shape == (4, 8)

    def test_bn_gamma_2d_is_row_vector(self):
        # BatchNorm2d(8) weight: (8,) → (1, 8)
        mats = dict(self._snap().weight_2d['bn_gamma'])
        assert mats['bn.weight'].shape == (1, 8)

    def test_l2_norms_conv_shape(self):
        # Conv2d(3, 8, 3) weight → 8 filters → 8 norms
        assert self._snap().l2_norms['conv'].shape == (8,)

    def test_l2_norms_dense_shape(self):
        # Linear(8, 4) weight → 4 neurons → 4 norms
        assert self._snap().l2_norms['dense'].shape == (4,)

    def test_flat_sample_all_included_when_small(self):
        # conv.weight has 8*3*3*3=216 elements < 10 000 → no subsampling
        n_conv = self.state_dict['conv.weight'].numel()
        assert len(self._snap().flat_weights['conv']) == n_conv

    def test_flat_sample_capped_at_max_flat_sample(self):
        model = nn.Sequential(nn.Linear(200, 200))
        sd    = model.state_dict()
        ptm   = build_param_type_map(model)
        snap  = snapshot_from_state_dict(sd, ptm, 0, 0,
                                         max_flat_sample=100, rng=self.rng)
        assert len(snap.flat_weights['dense']) <= 100

    def test_param_names_provenance(self):
        snap = self._snap()
        assert 'conv.weight' in snap.param_names.get('conv', [])
        assert 'fc.weight'   in snap.param_names.get('dense', [])

    def test_bn_buffers_skipped_without_crash(self):
        # running_mean / running_var are in state_dict but NOT in param_type_map
        snap = self._snap()
        assert snap is not None   # no KeyError / crash

    def test_same_rng_seed_gives_same_flat_sample(self):
        model = nn.Sequential(nn.Linear(300, 300))
        sd    = model.state_dict()
        ptm   = build_param_type_map(model)
        snap1 = snapshot_from_state_dict(sd, ptm, 0, 0, max_flat_sample=50,
                                         rng=np.random.default_rng(99))
        snap2 = snapshot_from_state_dict(sd, ptm, 0, 0, max_flat_sample=50,
                                         rng=np.random.default_rng(99))
        np.testing.assert_array_equal(snap1.flat_weights['dense'],
                                      snap2.flat_weights['dense'])


# ============================================================================
# snapshot_from_checkpoint
# ============================================================================

class TestSnapshotFromCheckpoint:

    def setup_method(self):
        torch.manual_seed(0)
        self.model  = SmallNet()
        self.ptm    = build_param_type_map(self.model)

    def _save(self, payload, suffix='.pt'):
        f = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        path = Path(f.name)
        f.close()
        torch.save(payload, path)
        return path

    def test_raw_state_dict_checkpoint(self):
        path = self._save(self.model.state_dict())
        try:
            snap = snapshot_from_checkpoint(path, self.ptm, step=0, epoch=0.0)
            assert 'conv' in snap.flat_weights
        finally:
            path.unlink(missing_ok=True)

    def test_wrapped_state_dict_checkpoint(self):
        # {'state_dict': ..., 'epoch': 5} format used by many training scripts
        path = self._save({'state_dict': self.model.state_dict(), 'epoch': 5})
        try:
            snap = snapshot_from_checkpoint(path, self.ptm, step=0, epoch=0.0)
            assert 'conv' in snap.flat_weights
        finally:
            path.unlink(missing_ok=True)

    def test_step_and_epoch_propagated(self):
        path = self._save(self.model.state_dict())
        try:
            snap = snapshot_from_checkpoint(path, self.ptm, step=42, epoch=3.5)
            assert snap.step == 42 and snap.epoch == 3.5
        finally:
            path.unlink(missing_ok=True)

    def test_pth_extension_accepted(self):
        path = self._save(self.model.state_dict(), suffix='.pth')
        try:
            snap = snapshot_from_checkpoint(path, self.ptm, step=0, epoch=0.0)
            assert snap is not None
        finally:
            path.unlink(missing_ok=True)


# ============================================================================
# Systematic subsampling (decision D3)
# ============================================================================

class TestSystematicSubsample:
    """The flat sample must preserve flatten order — SampEn₁D depends on it."""

    def test_returns_array_unchanged_when_it_fits(self):
        from src.weights import systematic_subsample
        a = np.arange(50.0)
        np.testing.assert_array_equal(systematic_subsample(a, 100), a)

    def test_respects_max_size(self):
        from src.weights import systematic_subsample
        assert len(systematic_subsample(np.arange(10_000.0), 100)) == 100

    def test_preserves_original_order(self):
        from src.weights import systematic_subsample
        out = systematic_subsample(np.arange(10_000.0), 100)
        assert np.all(np.diff(out) > 0), "systematic sampling must not shuffle"

    def test_uses_a_fixed_step(self):
        from src.weights import systematic_subsample
        out = systematic_subsample(np.arange(10_000.0), 100)
        assert len(set(np.diff(out))) == 1, "step must be constant"

    def test_is_deterministic_regardless_of_rng(self):
        from src.weights import systematic_subsample
        a = np.random.default_rng(0).standard_normal(10_000)
        np.testing.assert_array_equal(systematic_subsample(a, 100),
                                      systematic_subsample(a, 100))

    def test_snapshot_flat_sample_keeps_order(self):
        """End-to-end: a subsampled group must come out monotonic when the
        underlying weights are monotonic."""
        import torch.nn as nn
        from src.weights import build_param_type_map, snapshot_from_state_dict
        model = nn.Sequential(nn.Linear(400, 400))
        sd = {'0.weight': torch.arange(160_000.0).reshape(400, 400),
              '0.bias':   torch.zeros(400)}
        ptm = build_param_type_map(model)
        snap = snapshot_from_state_dict(sd, ptm, step=0, epoch=0.0,
                                        max_flat_sample=1_000)
        flat = snap.flat_weights['dense']
        assert len(flat) == 1_000
        assert np.all(np.diff(flat) > 0), "flatten order was not preserved"
