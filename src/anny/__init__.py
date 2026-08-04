# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
from anny.models import (
    Anny,
    create_fullbody_model,
    create_hand_model,
    create_head_model,
)
from anny.anthropometry import Anthropometry
from anny.anny_inverter import AnnyInverter
from anny.keypoints import KeypointsRegressor
import importlib.metadata

try:
    __version__ = importlib.metadata.version("anny")
except importlib.metadata.PackageNotFoundError:
    # The source directory was imported without installing the package.
    __version__ = "0.0.0+unknown"

__all__ = [
    "Anny",
    "create_fullbody_model",
    "create_hand_model",
    "create_head_model",
    "AnnyInverter",
    "KeypointsRegressor",
    "Anthropometry",
]
