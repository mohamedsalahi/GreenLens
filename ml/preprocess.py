"""Image loading and preprocessing shared by training and inference.

Both the training pipeline and the inference worker must feed the model
exactly the same kind of input, so both go through this module:

  1. decode with Pillow, apply EXIF orientation, convert to RGB
  2. bilinear resize to 224x224
  3. the standard VGG16 caffe-style preprocessing: RGB to BGR conversion
     and subtraction of the ImageNet channel means (via
     keras.applications.vgg16.preprocess_input)

Anything that changes how pixels reach the network belongs here, not in
the callers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image, ImageOps

IMAGE_SIZE = (224, 224)
PathLike = Union[str, Path]


def load_image(path: PathLike) -> np.ndarray:
    """Decode an image file into a 224x224 RGB uint8 array."""
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")
        if img.size != IMAGE_SIZE:
            img = img.resize(IMAGE_SIZE, Image.BILINEAR)
        return np.asarray(img, dtype=np.uint8)


def image_dimensions(path: PathLike) -> tuple[int, int]:
    """Original pixel dimensions of the file (width, height)."""
    with Image.open(path) as img:
        return img.size


def preprocess_array(image: np.ndarray) -> np.ndarray:
    """Apply VGG16 preprocessing to a uint8 RGB image array."""
    from tensorflow.keras.applications.vgg16 import preprocess_input

    return preprocess_input(np.asarray(image, dtype=np.float32))


def preprocess_path(path: PathLike) -> np.ndarray:
    """Full pipeline for one image file: float32 array ready for the model.

    Accepts bytes as well as str/Path because tf.data passes python
    strings to numpy_function as bytes.
    """
    if isinstance(path, bytes):
        path = path.decode()
    return preprocess_array(load_image(path))
