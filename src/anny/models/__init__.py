# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
import warnings
from typing import Literal

import anny.models.full_model
from anny.models.full_model import (
    RigPreset,
    SkinningMethod,
    LocalChanges,
    load_data,
    _filenames_from_rig,
    ANNY_ROOT_DIR,
)

from anny.models.model_data import ModelData, ModelMetadata, AnnyModelMetadata, cache_builder
from anny.paths import PathLike
from anny.face_segmentation import get_face_segmentation_mask
import anny.models.model_transforms as model_transforms
import anny.models.retopology
from anny.models.retopology import Topology
from anny.models.rigged_model import PoseParameterization, BoneOrientation
from anny.models.phenotype import Anny
import anny.models.soma

_eye_bone_labels = {"eye.L", "eye.R"}
_tongue_bone_labels = {
    "tongue00",
    "tongue01",
    "tongue02",
    "tongue03",
    "tongue04",
    "tongue05.L",
    "tongue05.R",
    "tongue06.L",
    "tongue06.R",
    "tongue07.L",
    "tongue07.R",
}
_facial_expression_bone_labels = {
    "jaw",
    "special04",
    "oris02",
    "oris01",
    "oris06.L",
    "oris07.L",
    "oris06.R",
    "oris07.R",
    "levator02.L",
    "levator03.L",
    "levator04.L",
    "levator05.L",
    "levator02.R",
    "levator03.R",
    "levator04.R",
    "levator05.R",
    "special01",
    "oris04.L",
    "oris03.L",
    "oris04.R",
    "oris03.R",
    "oris06",
    "oris05",
    "special03",
    "levator06.L",
    "levator06.R",
    "special06.L",
    "special05.L",
    "orbicularis03.L",
    "orbicularis04.L",
    "special06.R",
    "special05.R",
    "orbicularis03.R",
    "orbicularis04.R",
    "temporalis01.L",
    "oculi02.L",
    "oculi01.L",
    "temporalis01.R",
    "oculi02.R",
    "oculi01.R",
    "temporalis02.L",
    "risorius02.L",
    "risorius03.L",
    "temporalis02.R",
    "risorius02.R",
    "risorius03.R",
}

# Bones with zero influence on the mesh in the default skinning (excluding the root pose)
_zero_weight_bone_labels = {
    "oris02",
    "oris06.L",
    "oris06.R",
    "levator02.L",
    "levator03.L",
    "levator04.L",
    "levator02.R",
    "levator03.R",
    "levator04.R",
    "special01",
    "oris04.L",
    "oris04.R",
    "oris06",
    "special03",
    "special06.L",
    "special06.R",
    "temporalis01.L",
    "oculi02.L",
    "temporalis01.R",
    "oculi02.R",
    "temporalis02.L",
    "risorius02.L",
    "temporalis02.R",
    "risorius02.R",
}

_toe_bone_labels = {
    "toe1-1.L",
    "toe1-2.L",
    "toe2-1.L",
    "toe2-2.L",
    "toe2-3.L",
    "toe3-1.L",
    "toe3-2.L",
    "toe3-3.L",
    "toe4-1.L",
    "toe4-2.L",
    "toe4-3.L",
    "toe5-1.L",
    "toe5-2.L",
    "toe5-3.L",
    "toe1-1.R",
    "toe1-2.R",
    "toe2-1.R",
    "toe2-2.R",
    "toe2-3.R",
    "toe3-1.R",
    "toe3-2.R",
    "toe3-3.R",
    "toe4-1.R",
    "toe4-2.R",
    "toe4-3.R",
    "toe5-1.R",
    "toe5-2.R",
    "toe5-3.R",
}
_hand_bone_labels = {
    "metacarpal1.L",
    "finger1-1.L",
    "finger1-2.L",
    "finger1-3.L",
    "metacarpal2.L",
    "finger2-1.L",
    "finger2-2.L",
    "finger2-3.L",
    "metacarpal3.L",
    "finger3-1.L",
    "finger3-2.L",
    "finger3-3.L",
    "metacarpal4.L",
    "finger4-1.L",
    "finger4-2.L",
    "finger4-3.L",
    "finger5-1.L",
    "finger5-2.L",
    "finger5-3.L",
    "metacarpal1.R",
    "finger1-1.R",
    "finger1-2.R",
    "finger1-3.R",
    "metacarpal2.R",
    "finger2-1.R",
    "finger2-2.R",
    "finger2-3.R",
    "metacarpal3.R",
    "finger3-1.R",
    "finger3-2.R",
    "finger3-3.R",
    "metacarpal4.R",
    "finger4-1.R",
    "finger4-2.R",
    "finger4-3.R",
    "finger5-1.R",
    "finger5-2.R",
    "finger5-3.R",
}
_breast_bone_labels = {"breast.L", "breast.R"}


    
    
@cache_builder
def build_fullbody_model_data(
    rig: RigPreset | PathLike = "default",
    topology: Topology = "default",
    local_changes: LocalChanges = "none",
    remove_unattached_vertices: bool = True,
    triangulate_faces: bool = False,
    pose_parameterization: str = "local-bone",
    bone_orientation: str = "blender-rootidentity",
    extrapolate_phenotypes: bool = False,
    all_phenotypes: bool = False,
    skinning_method: SkinningMethod | None = None,
    weights_filename: PathLike | None = None,
) -> ModelData:
    
    # Legacy names
    if pose_parameterization == "root_relative":
        warnings.warn(
            "pose_parameterization='root_relative' is deprecated, use 'local-bone' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        pose_parameterization = "local-bone"
        assert bone_orientation == "blender-rootidentity"
        # Override default
        bone_orientation = "blender"
    elif pose_parameterization == "root_relative_world":
        warnings.warn(
            "pose_parameterization='root_relative_world' is deprecated, "
            "use 'local-bone' with bone_orientation='blender-rootidentity' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        pose_parameterization = "local-bone"
        assert bone_orientation == "blender-rootidentity"

    if type(local_changes) == bool:
        warnings.warn(
            "Passing local_changes as a bool is deprecated, "
            "use 'default' or 'none' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        local_changes = "default" if local_changes else "none"

    bones_to_remove = set()
    if isinstance(rig, str) and rig.startswith("default-"):
        rig_specs = rig.split("-")
        assert rig_specs[0] == "default"
        for spec in rig_specs[1:]:
            if spec == "pruned":
                bones_to_remove.update(_zero_weight_bone_labels)
            elif spec == "noeyes":
                bones_to_remove.update(_eye_bone_labels)
            elif spec == "notongue":
                bones_to_remove.update(_tongue_bone_labels)
            elif spec == "noexpression":
                bones_to_remove.update(_facial_expression_bone_labels)
                bones_to_remove.update(_eye_bone_labels)
                bones_to_remove.update(_tongue_bone_labels)
            elif spec == "notoes":
                bones_to_remove.update(_toe_bone_labels)
            elif spec == "nohands":
                bones_to_remove.update(_hand_bone_labels)
            elif spec == "nobreasts":
                bones_to_remove.update(_breast_bone_labels)
            else:
                raise ValueError(f"Unknown rig specifier: {spec}")
        rig = "default"

    if rig == "soma":
        assert len(bones_to_remove) == 0, (
            "The 'soma' rig does not support removing bones from the default rig."
        )
        return anny.models.soma.build_soma_rig_model_data(
            topology=topology,
            local_changes=local_changes,
            pose_parameterization=pose_parameterization,
            extrapolate_phenotypes=extrapolate_phenotypes,
            all_phenotypes=all_phenotypes,
            remove_unattached_vertices=remove_unattached_vertices,
            triangulate_faces=triangulate_faces,
            skinning_method=skinning_method,
        )

    if topology.startswith("default") or topology.startswith("makehuman"):
        topology_specs = topology.split("-")
        assert topology_specs[0] in ("default", "makehuman")

        eyes = True
        tongue = True
        for spec in topology_specs[1:]:
            if spec == "noeyes":
                eyes = False
            elif spec == "notongue":
                tongue = False
            else:
                raise ValueError(f"Unknown topology specifier: {spec}")
        topology = topology_specs[0]

        return anny.models.full_model.build_model_data(
            rig=rig,
            topology=topology,
            eyes=eyes,
            tongue=tongue,
            local_changes=local_changes,
            bones_to_remove=bones_to_remove,
            pose_parameterization=pose_parameterization,
            extrapolate_phenotypes=extrapolate_phenotypes,
            all_phenotypes=all_phenotypes,
            remove_unattached_vertices=remove_unattached_vertices,
            triangulate_faces=triangulate_faces,
            skinning_method=skinning_method,
            bone_orientation=bone_orientation,
            weights_filename=weights_filename,
        )
    else:
        if topology == "smplx":
            return anny.models.retopology.build_smplx_topology_model_data(
                rig=rig,
                all_phenotypes=all_phenotypes,
                bones_to_remove=bones_to_remove,
                pose_parameterization=pose_parameterization,
                extrapolate_phenotypes=extrapolate_phenotypes,
                local_changes=local_changes,
                skinning_method=skinning_method,
                bone_orientation=bone_orientation,
                remove_skinning_islands=remove_skinning_islands,
                weights_filename=weights_filename,
            )
        elif topology == "smpl":
            return anny.models.retopology.build_smpl_topology_model_data(
                            rig=rig,
                            all_phenotypes=all_phenotypes,
                            bones_to_remove=bones_to_remove,
                            pose_parameterization=pose_parameterization,
                            extrapolate_phenotypes=extrapolate_phenotypes,
                            local_changes=local_changes,
                            skinning_method=skinning_method,
                            bone_orientation=bone_orientation,
                        )
        elif topology == "soma":
            return anny.models.retopology.build_soma_topology_model_data(
                rig=rig,
                all_phenotypes=all_phenotypes,
                bones_to_remove=bones_to_remove,
                pose_parameterization=pose_parameterization,
                extrapolate_phenotypes=extrapolate_phenotypes,
                local_changes=local_changes,
                skinning_method=skinning_method,
                bone_orientation=bone_orientation,
                weights_filename=weights_filename,
            )
        else:
            return anny.models.retopology.build_alternative_topology_model_data(
                rig=rig,
                topology=topology,
                all_phenotypes=all_phenotypes,
                bones_to_remove=bones_to_remove,
                pose_parameterization=pose_parameterization,
                extrapolate_phenotypes=extrapolate_phenotypes,
                local_changes=local_changes,
                skinning_method=skinning_method,
                bone_orientation=bone_orientation,
                weights_filename=weights_filename,
            )

def create_fullbody_model(
    rig: RigPreset | PathLike = "default",
    topology: Topology = "default",
    local_changes: LocalChanges = "none",
    remove_unattached_vertices: bool = True,
    triangulate_faces: bool = False,
    pose_parameterization: PoseParameterization = "local-bone",
    bone_orientation: BoneOrientation = "blender-rootidentity",
    extrapolate_phenotypes: bool = False,
    all_phenotypes: bool = False,
    skinning_method: SkinningMethod | None = None,
    weights_filename: PathLike | None = None,
):
    return Anny(
        rig=rig,
        topology=topology,
        local_changes=local_changes,
        remove_unattached_vertices=remove_unattached_vertices,
        triangulate_faces=triangulate_faces,
        pose_parameterization=pose_parameterization,
        bone_orientation=bone_orientation,
        extrapolate_phenotypes=extrapolate_phenotypes,
        all_phenotypes=all_phenotypes,
        skinning_method=skinning_method,
        weights_filename=weights_filename,
    )


@cache_builder
def build_hand_model_data(
    side: Literal["R", "L"] = "R",
    local_changes: LocalChanges = "none",
    remove_unattached_vertices: bool = True,
    triangulate_faces: bool = False,
    pose_parameterization: str = "root_relative",
    extrapolate_phenotypes: bool = False,
    all_phenotypes: bool = False,
    bone_orientation: str = "blender-rootidentity",
) -> ModelData:
    hand_bones = {
        f"wrist.{side}",
        f"finger1-1.{side}",
        f"finger1-2.{side}",
        f"finger1-3.{side}",
        f"metacarpal1.{side}",
        f"finger2-1.{side}",
        f"finger2-2.{side}",
        f"finger2-3.{side}",
        f"metacarpal2.{side}",
        f"finger3-1.{side}",
        f"finger3-2.{side}",
        f"finger3-3.{side}",
        f"metacarpal3.{side}",
        f"finger4-1.{side}",
        f"finger4-2.{side}",
        f"finger4-3.{side}",
        f"metacarpal4.{side}",
        f"finger5-1.{side}",
        f"finger5-2.{side}",
        f"finger5-3.{side}",
    }

    rig_filename, weights_filename = _filenames_from_rig("default", None, ANNY_ROOT_DIR)

    # Load base data to resolve which bones and faces to keep (fast: reads from cache)
    base_data = load_data(
        rig_filename=rig_filename,
        weights_filename=weights_filename,
    )
    bones_to_remove = {
        label for label in base_data.metadata.bone_labels if label not in hand_bones
    }
    faces_to_keep = get_face_segmentation_mask(
        model_transforms.edit_mesh(base_data), [f"hand.{side}"]
    )

    data = anny.models.full_model.build_model_data(
        bones_to_remove=bones_to_remove,
        faces_to_keep=faces_to_keep,
        local_changes=local_changes,
        remove_unattached_vertices=remove_unattached_vertices,
        triangulate_faces=triangulate_faces,
        pose_parameterization=pose_parameterization,
        extrapolate_phenotypes=extrapolate_phenotypes,
        all_phenotypes=all_phenotypes,
        bone_orientation=bone_orientation,
    )
    return data


def create_hand_model(
    side: Literal["R", "L"] = "R",
    local_changes: LocalChanges = "none",
    remove_unattached_vertices: bool = True,
    triangulate_faces: bool = False,
    pose_parameterization: str = "root_relative",
    extrapolate_phenotypes: bool = False,
    all_phenotypes: bool = False,
    bone_orientation: str = "blender-rootidentity",
):

    return Anny.from_model_data(
        build_hand_model_data(
            side=side,
            local_changes=local_changes,
            remove_unattached_vertices=remove_unattached_vertices,
            triangulate_faces=triangulate_faces,
            pose_parameterization=pose_parameterization,
            extrapolate_phenotypes=extrapolate_phenotypes,
            all_phenotypes=all_phenotypes,
            bone_orientation=bone_orientation,
        )
    )


@cache_builder
def build_head_model_data(
    eyes: bool = True,
    tongue: bool = True,
    local_changes: LocalChanges = "none",
    pose_parameterization: str = "root_relative",
    extrapolate_phenotypes: bool = False,
    all_phenotypes: bool = False,
    remove_unattached_vertices: bool = True,
    triangulate_faces: bool = False,
    bone_orientation: str = "blender-rootidentity",
):
    face_bones = {"neck01", "neck02", "neck03", "head"}
    face_bones.update(_facial_expression_bone_labels)
    if eyes:
        face_bones.update(_eye_bone_labels)
    if tongue:
        face_bones.update(_tongue_bone_labels)

    rig_filename, weights_filename = _filenames_from_rig("default", None, ANNY_ROOT_DIR)

    # Load base data to resolve which bones and faces to keep (fast: reads from cache)
    base_data = load_data(
        rig_filename=rig_filename,
        weights_filename=weights_filename,
        eyes=eyes,
        tongue=tongue,
    )
    bones_to_remove = {
        label for label in base_data.metadata.bone_labels if label not in face_bones
    }
    faces_to_keep = get_face_segmentation_mask(
        model_transforms.edit_mesh(base_data),
        [
            "head",
            "eye_cavity.R",
            "eye_cavity.L",
            "mouth_cavity",
            "eye_front.L",
            "eye_back.L",
            "eye_front.R",
            "eye_back.L",
            "tongue",
        ],
    )

    data = anny.models.full_model.build_model_data(
        eyes=eyes,
        tongue=tongue,
        bones_to_remove=bones_to_remove,
        faces_to_keep=faces_to_keep,
        local_changes=local_changes,
        remove_unattached_vertices=remove_unattached_vertices,
        triangulate_faces=triangulate_faces,
        pose_parameterization=pose_parameterization,
        extrapolate_phenotypes=extrapolate_phenotypes,
        all_phenotypes=all_phenotypes,
        bone_orientation=bone_orientation,
    )
    return data


def create_head_model(
    eyes: bool = True,
    tongue: bool = True,
    local_changes: LocalChanges = "none",
    pose_parameterization: str = "root_relative",
    extrapolate_phenotypes: bool = False,
    all_phenotypes: bool = False,
    remove_unattached_vertices: bool = True,
    triangulate_faces: bool = False,
    bone_orientation: str = "blender-rootidentity",
):
    return Anny.from_model_data(
        build_head_model_data(
            eyes=eyes,
            tongue=tongue,
            local_changes=local_changes,
            pose_parameterization=pose_parameterization,
            extrapolate_phenotypes=extrapolate_phenotypes,
            all_phenotypes=all_phenotypes,
            remove_unattached_vertices=remove_unattached_vertices,
            triangulate_faces=triangulate_faces,
            bone_orientation=bone_orientation,
        )
    )


__all__ = [
    "build_fullbody_model_data",
    "create_fullbody_model",
    "Anny",
    "build_hand_model_data",
    "create_hand_model",
    "build_head_model_data",
    "create_head_model",
    "ModelData",
    "ModelMetadata",
    "AnnyModelMetadata"
]
