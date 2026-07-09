# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
import dataclasses
from typing import Literal
import warnings

from anny.models.model_data import RigConfig
from anny.typing import BoneOrientation, PoseParameterization

LegacyBoneOrientation = BoneOrientation | Literal["blender-rootidentity"]
LegacyPoseParameterization = PoseParameterization | Literal[
    "root_relative", "root_relative_world"
]


def legacy_bone_orientation_to_rig_options(
    bone_orientation: LegacyBoneOrientation,
) -> tuple[BoneOrientation, bool]:
    """
    Returns a tuple of (bone_orientation, root_identity_orientation) for the given legacy bone orientation.
    """
    if bone_orientation == "blender-rootidentity":
        return "blender", True
    if bone_orientation == "blender":
        return "blender", False
    if bone_orientation == "procrustes":
        return "procrustes", False
    raise ValueError(
        "bone_orientation must be 'blender', 'blender-rootidentity', or 'procrustes'."
    )

def check_legacy_pose_parameterization(
    pose_parameterization: LegacyPoseParameterization,
    bone_orientation: LegacyBoneOrientation,
) -> tuple[PoseParameterization, LegacyBoneOrientation]:
    if pose_parameterization == "root_relative":
        warnings.warn(
            "pose_parameterization='root_relative' is deprecated, use 'local-bone' instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        if bone_orientation != "blender":
            raise ValueError(
                "pose_parameterization='root_relative' requires "
                "bone_orientation='blender'."
            )
        return "local-bone", "blender"
    if pose_parameterization == "root_relative_world":
        warnings.warn(
            "pose_parameterization='root_relative_world' is deprecated, "
            "use 'local-bone' with bone_orientation='blender-rootidentity' instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        if bone_orientation not in ["blender", "blender-rootidentity"]:
            raise ValueError(
                "pose_parameterization='root_relative_world' requires 'blender' or 'blender-rootidentity' bone orientation."
            )
        return "local-bone", bone_orientation
    return pose_parameterization, bone_orientation

def legacy_topology_to_anny(
    topology: str = "default",
    remove_unattached_vertices: bool = True,
    triangulate_faces: bool = False) -> str:
    new_topology = topology.replace("default", "anny")
    if not triangulate_faces and new_topology not in ["smpl", "smplx", "soma"]:
        new_topology = new_topology + "-quads"
    if not remove_unattached_vertices:
        new_topology = new_topology + "-full"
    return new_topology

def legacy_rig_to_anny(
    rig: str,
    bone_orientation: LegacyBoneOrientation,
) -> "RigConfig | str":
    if rig == "soma":
        return rig
    # Legacy rig="default" is the full MakeHuman rig (weights.default.json, no
    # pruning): "makehuman-symmetric" resolves to base_rig="anny" with an empty
    # bones_to_remove, unlike the bare "anny" spec which prunes tongue/expression
    # /zero-weight bones. bone_orientation and root_identity_orientation cannot be
    # expressed via the string spec for the procrustes case, so we override them
    # on the resolved RigConfig to preserve the exact legacy behavior.
    spec = rig.replace("default", "makehuman-symmetric")
    resolved = RigConfig.from_string(spec)
    resolved_bone_orientation, root_identity_orientation = (
        legacy_bone_orientation_to_rig_options(bone_orientation)
    )
    return dataclasses.replace(
        resolved,
        bone_orientation=resolved_bone_orientation,
        root_identity_orientation=root_identity_orientation,
    )
