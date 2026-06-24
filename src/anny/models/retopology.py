# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
import torch
from typing import Literal
from anny.typing import Topology
from anny.utils import obj_utils
from anny.models.full_model import RigPreset, build_model_data
from anny.models.model_transforms import apply_retopology, apply_retopology_from_mesh
import os
from anny.paths import ANNY_ROOT_DIR, get_anny2smplx_data_path, get_anny2smpl_data_path, PathLike
import roma
import logging
import math

logger = logging.getLogger(__name__)


def _load_target_topology_mesh(root_dirname: PathLike, topology: Topology):
    if topology == "soma":
        filename = "data/soma/SOMA_wrap.obj"
    elif topology == "anny_from_soma": # The base body (default phenotypes) from SOMA-X repo
        filename = "data/soma/base_body.obj"
    else:
        filename = f"data/topology/{topology}.obj"
    vertices, _, groups = obj_utils.load_obj_file(os.path.join(root_dirname, filename), dtype=torch.float64)
    transformation = roma.Rotation(roma.euler_to_rotmat("x", [math.pi/2], dtype=torch.float64)[None])
    vertices = transformation.apply(vertices)
    faces = groups['noname']['face_vertex_indices']
    return vertices, faces

def build_smplx_topology_model_data(rig: RigPreset | PathLike = "default",
                                bones_to_remove=set(),
                                all_phenotypes=False,
                                skinning_method=None,
                                pose_parameterization: str = "local-bone",
                                extrapolate_phenotypes=False,
                                local_changes="none",
                                bone_orientation="default-rootidentity",
                                root_dirname=ANNY_ROOT_DIR,
                                weights_filename: PathLike | None = None):
    ref_data = build_model_data(rig=rig,
                                        eyes=True,
                                        tongue=False,
                                        bones_to_remove=bones_to_remove,
                                        remove_unattached_vertices=False,
                                        all_phenotypes=all_phenotypes,
                                        skinning_method=skinning_method,
                                        pose_parameterization=pose_parameterization,
                                        extrapolate_phenotypes=extrapolate_phenotypes,
                                        local_changes=local_changes,
                                        bone_orientation=bone_orientation,
                                        root_dirname=root_dirname,
                                        weights_filename=weights_filename)

    # Load the SMPL-X topology
    state_dict = torch.load(get_anny2smplx_data_path(),
                            map_location="cpu",
                            weights_only=True)
    barycentric_coordinates = state_dict["anny2dst_barycentric_coordinates"]
    reference_vertex_indices = state_dict["anny2dst_vertex_indices"]
    vertices = barycentric_coordinates[0][:,None] * ref_data.template_vertices[reference_vertex_indices[:,0]] + \
               barycentric_coordinates[1][:,None] * ref_data.template_vertices[reference_vertex_indices[:,1]] + \
               barycentric_coordinates[2][:,None] * ref_data.template_vertices[reference_vertex_indices[:,2]]
    faces = state_dict["dst_faces"]
    data = apply_retopology(
        ref_data,
        vertices=vertices,
        faces=faces,
        reference_vertex_indices=reference_vertex_indices,
        barycentric_coordinates=barycentric_coordinates,
    )
    return data

def build_smpl_topology_model_data(rig: RigPreset | PathLike = "default",
                                bones_to_remove=set(),
                                all_phenotypes=False,
                                skinning_method=None,
                                pose_parameterization: str = "root_relative_world",
                                extrapolate_phenotypes=False,
                                local_changes="none",
                                bone_orientation="default",
                                root_dirname=ANNY_ROOT_DIR,
                                weights_filename: PathLike | None = None):
    ref_data = build_model_data(rig=rig,
                                        eyes=True,
                                        tongue=False,
                                        bones_to_remove=bones_to_remove,
                                        remove_unattached_vertices=False,
                                        all_phenotypes=all_phenotypes,
                                        skinning_method=skinning_method,
                                        pose_parameterization=pose_parameterization,
                                        extrapolate_phenotypes=extrapolate_phenotypes,
                                        local_changes=local_changes,
                                        bone_orientation=bone_orientation,
                                        root_dirname=root_dirname,
                                        weights_filename=weights_filename)

    # Load the SMPL topology
    state_dict = torch.load(get_anny2smpl_data_path(),
                            map_location="cpu",
                            weights_only=True)
    barycentric_coordinates = state_dict["anny2dst_barycentric_coordinates"]
    reference_vertex_indices = state_dict["anny2dst_vertex_indices"]
    vertices = barycentric_coordinates[0][:,None] * ref_data.template_vertices[reference_vertex_indices[:,0]] + \
               barycentric_coordinates[1][:,None] * ref_data.template_vertices[reference_vertex_indices[:,1]] + \
               barycentric_coordinates[2][:,None] * ref_data.template_vertices[reference_vertex_indices[:,2]]
    faces = state_dict["dst_faces"]
    data = apply_retopology(
        ref_data,
        vertices=vertices,
        faces=faces,
        reference_vertex_indices=reference_vertex_indices,
        barycentric_coordinates=barycentric_coordinates,
    )
    return data

def build_soma_topology_model_data(rig: RigPreset | PathLike ="default",
                                bones_to_remove=set(),
                                all_phenotypes=False,
                                skinning_method=None,
                                pose_parameterization: str = "local-bone",
                                extrapolate_phenotypes=False,
                                local_changes="none",
                                bone_orientation="default-rootidentity",
                                root_dirname=ANNY_ROOT_DIR,
                                weights_filename: PathLike | None = None):
    return build_alternative_topology_model_data(rig=rig,
                                      topology="soma",
                                      bones_to_remove=bones_to_remove,
                                      all_phenotypes=all_phenotypes,
                                      skinning_method=skinning_method,
                                      pose_parameterization=pose_parameterization,
                                      extrapolate_phenotypes=extrapolate_phenotypes,
                                      local_changes=local_changes,
                                      bone_orientation=bone_orientation,
                                      root_dirname=root_dirname,
                                      weights_filename=weights_filename,
                                      reference_topology="anny_from_soma")


def build_alternative_topology_model_data(rig: RigPreset | PathLike ="default",
                                      topology: Topology="default",
                                bones_to_remove=set(),
                                all_phenotypes=False,
                                skinning_method=None,
                                pose_parameterization: str = "local-bone",
                                extrapolate_phenotypes=False,
                                local_changes="none",
                                bone_orientation="default-rootidentity",
                                root_dirname=ANNY_ROOT_DIR,
                                weights_filename: PathLike | None = None,
                                reference_topology: Literal["default", "anny_from_soma"]="default"):
    # For soma, the template mesh has only attached vertices and eyes+tongue
    is_soma = reference_topology == "anny_from_soma"
    if is_soma:
        assert topology == "soma", "The 'anny_from_soma' reference topology can only be used with the 'soma' target topology."

    ref_data = build_model_data(rig=rig,
                                eyes=is_soma,
                                tongue=is_soma,
                                bones_to_remove=bones_to_remove,
                                all_phenotypes=all_phenotypes,
                                skinning_method=skinning_method,
                                pose_parameterization=pose_parameterization,
                                extrapolate_phenotypes=extrapolate_phenotypes,
                                local_changes=local_changes,
                                remove_unattached_vertices=is_soma,
                                bone_orientation=bone_orientation,
                                root_dirname=root_dirname, weights_filename=weights_filename)

    reference_vertices, reference_faces = _load_target_topology_mesh(root_dirname, reference_topology)
    vertices, faces = _load_target_topology_mesh(root_dirname, topology)

    data = apply_retopology_from_mesh(
        ref_data,
        target_vertices=vertices,
        target_faces=faces,
        source_vertices=reference_vertices,
        source_faces=reference_faces,
    )
    return data
