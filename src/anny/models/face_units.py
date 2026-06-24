# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch

from anny.paths import ANNY_ROOT_DIR, PathLike


FACE_UNIT_LABELS: list[str] = [
    "browDownLeft",
    "browDownRight",
    "browInnerUp",
    "browOuterUpLeft",
    "browOuterUpRight",
    "cheekPuff",
    "cheekSquintLeft",
    "cheekSquintRight",
    "eyeBlinkLeft",
    "eyeBlinkRight",
    "eyeLookDownLeft",
    "eyeLookDownRight",
    "eyeLookInLeft",
    "eyeLookInRight",
    "eyeLookOutLeft",
    "eyeLookOutRight",
    "eyeLookUpLeft",
    "eyeLookUpRight",
    "eyeSquintLeft",
    "eyeSquintRight",
    "eyeWideLeft",
    "eyeWideRight",
    "jawForward",
    "jawLeft",
    "jawOpen",
    "jawRight",
    "mouthClose",
    "mouthDimpleLeft",
    "mouthDimpleRight",
    "mouthFrownLeft",
    "mouthFrownRight",
    "mouthFunnel",
    "mouthLeft",
    "mouthLowerDownLeft",
    "mouthLowerDownRight",
    "mouthPressLeft",
    "mouthPressRight",
    "mouthPucker",
    "mouthRight",
    "mouthRollLower",
    "mouthRollUpper",
    "mouthShrugLower",
    "mouthShrugUpper",
    "mouthSmileLeft",
    "mouthSmileRight",
    "mouthStretchLeft",
    "mouthStretchRight",
    "mouthUpperUpLeft",
    "mouthUpperUpRight",
    "noseSneerLeft",
    "noseSneerRight",
    "tongueOut",
]




def load_plain_target(
    filename: PathLike,
    vertices_count: int,
    world_transformation,
    dtype: torch.dtype,
) -> torch.Tensor:
    blend_shape = torch.zeros((vertices_count, 3), dtype=dtype)
    with open(filename, "r") as target_file:
        for line_number, line in enumerate(target_file, start=1):
            stripped = line.strip()
            if stripped == "" or stripped.startswith("#"):
                continue
            data = stripped.split()
            if len(data) != 4:
                raise ValueError(
                    f"Invalid target line in {filename}:{line_number}; expected vertex id and 3 offsets."
                )
            vertex_id = int(data[0])
            if vertex_id < 0 or vertex_id >= vertices_count:
                raise ValueError(
                    f"Invalid vertex id {vertex_id} in {filename}:{line_number}; "
                    f"expected 0 <= id < {vertices_count}."
                )
            blend_shape[vertex_id, :] = torch.as_tensor(
                [float(x) for x in data[1:]],
                dtype=dtype,
            )
    return world_transformation.apply(blend_shape)


def load_face_unit_blendshapes(
    root_dirname: PathLike = ANNY_ROOT_DIR,
    vertices_count: int | None = None,
    world_transformation=None,
    dtype: torch.dtype = torch.float64,
) -> tuple[list[str], torch.Tensor]:
    if vertices_count is None:
        raise ValueError("vertices_count must be provided to load face units.")
    if world_transformation is None:
        raise ValueError("world_transformation must be provided to load face units.")

    faceunit_dir = Path(root_dirname) / "data/faceunits01/targets/faceunits"
    blendshapes: list[torch.Tensor] = []
    missing_files: list[str] = []
    for label in FACE_UNIT_LABELS:
        filename = faceunit_dir / f"{label}.target"
        if not filename.exists():
            missing_files.append(str(filename))
            continue
        blendshapes.append(
            load_plain_target(
                filename=filename,
                vertices_count=vertices_count,
                world_transformation=world_transformation,
                dtype=dtype,
            )
        )

    if missing_files:
        joined = "\n".join(missing_files)
        raise FileNotFoundError(f"Missing faceunit target files:\n{joined}")

    if len(blendshapes) != len(FACE_UNIT_LABELS):
        raise ValueError(
            f"Expected {len(FACE_UNIT_LABELS)} faceunit targets, "
            f"loaded {len(blendshapes)}."
        )

    return list(FACE_UNIT_LABELS), torch.stack(blendshapes)
