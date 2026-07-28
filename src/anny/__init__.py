# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
from anny.models import *
from anny.anthropometry import Anthropometry
from anny.parameters_regressor import ParametersRegressor
from anny.keypoints import KeypointsRegressor
import importlib.metadata

try:
    __version__ = importlib.metadata.version("anny")
except importlib.metadata.PackageNotFoundError:
    # The source directory was imported without installing the package.
    __version__ = "0.0.0+unknown"
