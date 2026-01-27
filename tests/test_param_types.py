import json
import tempfile

from src.training.model import build_model
from src.utils.param_types import save_param_types


def test_param_types_saved():
    model = build_model("densenet121", in_channels=1, num_classes=2)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = f"{tmpdir}/param_types.json"
        save_param_types(model, path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    assert isinstance(data, dict)
    assert len(data) > 0
    valid_categories = {"Convolutional", "Linear", "Embedding", "BatchNorm", "Activation", "Pooling", "Flatten", "Other"}
    assert set(data.values()).issubset(valid_categories)
