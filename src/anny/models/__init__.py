# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
from typing import Literal
import warnings


from anny.models.model_data import (
    ModelData,
    ModelMetadata,
    AnnyModelConfig,
    RigConfig,
    TopologyConfig,
    cache_builder,
)


from anny.models.phenotype import Anny

from anny.models.legacy import (
    LegacyBoneOrientation,
    LegacyPoseParameterization,
    check_legacy_pose_parameterization,
    legacy_rig_to_anny,
    legacy_topology_to_anny,
)

from anny.typing import LocalChanges, PoseParameterization, SkinningMethod


@cache_builder
def build_model_data(
    rig: RigConfig,
    topology: TopologyConfig,
    local_changes: LocalChanges,
    facial_actions: bool,
) -> ModelData:
    if rig.base_rig == "soma":
        import anny.models.soma

        return anny.models.soma.build_soma_rig_model_data(
            topology=topology,
            local_changes=local_changes,
            facial_actions=facial_actions,
        )
    if topology.base_mesh == "makehuman":
        import anny.models.full_model

        return anny.models.full_model.build_anny_model_data(
            rig=rig,
            topology=topology,
            local_changes=local_changes,
            facial_actions=facial_actions,
        )
    # Alternative topologies
    import anny.models.retopology

    if topology.base_mesh == "smplx":
        return anny.models.retopology.build_smplx_topology_model_data(
            rig=rig,
            local_changes=local_changes,
            facial_actions=facial_actions,
        )
    if topology.base_mesh == "smpl":
        return anny.models.retopology.build_smpl_topology_model_data(
            rig=rig,
            local_changes=local_changes,
            facial_actions=facial_actions,
        )

    return anny.models.retopology.build_alternative_topology_model_data(
        rig=rig,
        topology=topology,
        local_changes=local_changes,
        facial_actions=facial_actions,
        reference_topology="anny_from_soma" if topology.base_mesh == "soma" else "anny",
    )


def create_fullbody_model(
    rig: str = "default",
    topology: str = "default",
    local_changes: LocalChanges = "none",
    remove_unattached_vertices: bool = True,
    triangulate_faces: bool = False,
    pose_parameterization: LegacyPoseParameterization = "local-bone",
    bone_orientation: LegacyBoneOrientation = "blender-rootidentity",
    extrapolate_phenotypes: bool = False,
    all_phenotypes: bool = False,
    skinning_method: SkinningMethod | None = None,
):
    warnings.warn(
        "create_fullbody_model() is deprecated and preserves legacy full-body "
        "defaults, including rig='default'. Use Anny(...) "
        "for the current defaults.",
        DeprecationWarning,
        stacklevel=2,
    )
    if type(local_changes) is bool:
        warnings.warn(
            "Passing local_changes as a bool is deprecated, "
            "use 'default' or 'none' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        local_changes = "default" if local_changes else "none"
    pose_parameterization, bone_orientation = check_legacy_pose_parameterization(
        pose_parameterization,
        bone_orientation,
    )
    anny_rig = legacy_rig_to_anny(rig, bone_orientation)
    anny_topology = legacy_topology_to_anny(
        topology, remove_unattached_vertices, triangulate_faces
    )

    return Anny(
        rig=anny_rig,
        topology=anny_topology,
        local_changes=local_changes,
        extrapolate_phenotypes=extrapolate_phenotypes,
        all_phenotypes=all_phenotypes,
        facial_actions=False,
        skinning_method=skinning_method,
        pose_parameterization=pose_parameterization,
    )


def create_hand_model(
    side: Literal["R", "L"] = "R",
    local_changes: LocalChanges = "none",
    remove_unattached_vertices: bool = True,
    triangulate_faces: bool = False,
    pose_parameterization: PoseParameterization = "local-bone",
    extrapolate_phenotypes: bool = False,
    all_phenotypes: bool = False,
):
    warnings.warn(
        "create_hand_model() is deprecated. Use Anny(...) with 'hand.R' or 'hand.L' topology",
        DeprecationWarning,
        stacklevel=2,
    )
    topology = f"hand.{side}"
    topology = legacy_topology_to_anny(
        topology=topology,
        remove_unattached_vertices=remove_unattached_vertices,
        triangulate_faces=triangulate_faces,
    )
    # Keep the full rig: head/hand part models need the expression/eye/tongue
    # bones that the default "anny" pruning would otherwise strip.
    rig = RigConfig(
        base_rig="anny", root_identity_orientation=True, bones_to_remove=frozenset()
    )
    return Anny(
        rig=rig,
        topology=topology,
        local_changes=local_changes,
        pose_parameterization=pose_parameterization,
        extrapolate_phenotypes=extrapolate_phenotypes,
        all_phenotypes=all_phenotypes,
    )


def create_head_model(
    eyes: bool = True,
    tongue: bool = True,
    local_changes: LocalChanges = "none",
    facial_actions: bool = False,
    pose_parameterization: PoseParameterization = "local-bone",
    extrapolate_phenotypes: bool = False,
    all_phenotypes: bool = False,
    remove_unattached_vertices: bool = True,
    triangulate_faces: bool = False,
):
    warnings.warn(
        "create_head_model() is deprecated. Use Anny(...) with 'head' topology",
        DeprecationWarning,
        stacklevel=2,
    )
    topology = "head"
    if not eyes:
        topology += "-noeyes"
    if not tongue:
        topology += "-notongue"
    topology = legacy_topology_to_anny(
        topology=topology,
        remove_unattached_vertices=remove_unattached_vertices,
        triangulate_faces=triangulate_faces,
    )
    # The head part model keeps the full (unpruned) rig with its expression/eye/tongue bones and uses
    # the legacy blender (tail-based) orientation. Those facial bones are absent from the precomputed
    # procrustes covariance (built on the pruned full-body anny rig), so the head model does not use it.
    rig = RigConfig(
        base_rig="anny",
        bone_orientation="blender",
        root_identity_orientation=True,
        bones_to_remove=frozenset(),
    )
    return Anny(
        rig=rig,
        topology=topology,
        local_changes=local_changes,
        facial_actions=facial_actions,
        pose_parameterization=pose_parameterization,
        extrapolate_phenotypes=extrapolate_phenotypes,
        all_phenotypes=all_phenotypes,
    )


__all__ = [
    "Anny",
    "create_fullbody_model",
    "create_hand_model",
    "create_head_model",
    "ModelData",
    "ModelMetadata",
    "AnnyModelConfig",
]
