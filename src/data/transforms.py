import os
from typing import Sequence

from monai.transforms import (
    EnsureChannelFirstd,
    ScaleIntensityd,
    ResizeD,
    EnsureTyped,
    Compose,
    Lambdad,
    LoadImage,
)


def build_transforms(size: Sequence[int] = (224, 224)):
    loader = LoadImage(image_only=True)

    def _load_if_path(x):
        if isinstance(x, (str, os.PathLike)):
            return loader(x)
        return x

    return Compose(
        [
            Lambdad(keys=["image"], func=_load_if_path),
            EnsureChannelFirstd(keys=["image"]),
            ScaleIntensityd(keys=["image"]),
            ResizeD(keys=["image"], spatial_size=size),
            EnsureTyped(keys=["image", "label"]),
        ]
    )
