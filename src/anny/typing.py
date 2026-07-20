# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
from __future__ import annotations
from typing import Literal, Sequence, TypeAlias
from pathlib import Path


PathLike: TypeAlias = Path | str


AlternativeTopology: TypeAlias = Literal["smplx", "smpl", "soma",  "anny_from_soma", "notoes", "notoes_collapse3pc", "notoes_collapse5pc", "notoes_collapse10pc", "legacy_default"]
Submodel: TypeAlias = Literal["body", "head", "hand.L", "hand.R"]
Rig: TypeAlias = Literal["anny", "makehuman", "cmu_mb", "game_engine", "mixamo", "soma"]


SkinningMethod: TypeAlias = Literal["lbs", "dqs", "warp_lbs"]
PoseParameterization: TypeAlias = Literal["world", "local-bone-world", "local-bone", "local-ref", "world-orient"]
BoneOrientation: TypeAlias = Literal["blender", "procrustes"]
LocalChanges: TypeAlias = Literal["none", "default", "all"] | Sequence[str]