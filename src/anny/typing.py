# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
from typing import Literal, Sequence
import os

Topology = Literal["default", "makehuman", "smplx", "soma", "notoes", "notoes_collapse3pc", "notoes_collapse5pc", "notoes_collapse10pc", "anny_from_soma"]
RigPreset = Literal["default", "default_no_toes", "cmu_mb", "game_engine", "mixamo"]
SkinningMethod = Literal["lbs", "dqs", "warp_lbs"]

PoseParameterization = Literal["world", "local-bone-world", "local-bone", "local-ref", "world-orient", "root_relative_world", "root_relative"]
BoneOrientation = Literal["blender", "gramschmidtyx", "gramschmidtyz", "blender-rootidentity", "procrustes"]

# `local_changes` selector for create_model / create_fullbody_model:
#   "none"    -> no local change blend shapes
#   "default" -> all local change blend shapes except nipple-related ones
#   "all"     -> every local change blend shape
#   Sequence[str] -> exactly the listed labels (must match `local_change_labels`)
LocalChanges = Literal["none", "default", "all"] | Sequence[str]

PathLike = os.PathLike | str