# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
from dataclasses import replace
import torch
from typing import Literal
from anny.typing import AlternativeTopology, LocalChanges
from anny.utils import obj_utils
from anny.models.full_model import build_anny_model_data
from anny.models.model_data import with_bone_orientation, RigConfig, TopologyConfig
from anny.models.model_transforms import (
    apply_anny_cached_orientation,
    apply_procrustes_orientation,
    apply_retopology,
    apply_retopology_from_mesh,
    triangulate
)
import os
from anny.paths import get_anny2smplx_data_path, get_anny2smpl_data_path, get_anny_root_dir
import roma
import logging
import math

logger = logging.getLogger(__name__)


def _load_target_topology_mesh(target_topology: AlternativeTopology):
    if target_topology == "soma":
        filename = "data/soma/SOMA_wrap.obj"
    elif target_topology == "anny_from_soma": # The base body (default phenotypes) from SOMA-X repo
        filename = "data/soma/base_body.obj"
    else:
        filename = f"data/topology/{target_topology}.obj"
    vertices, _, groups = obj_utils.load_obj_file(os.path.join(get_anny_root_dir(), filename), dtype=torch.float64)
    transformation = roma.Rotation(roma.euler_to_rotmat("x", [math.pi/2], dtype=torch.float64)[None])
    vertices = transformation.apply(vertices)
    faces = groups['noname']['face_vertex_indices']
    return vertices, faces

def build_smplx_topology_model_data(
                                rig: RigConfig, local_changes: LocalChanges,
                                facial_actions: bool):
    source_rig = with_bone_orientation(rig, "blender")
    source_topology = TopologyConfig(
        base_mesh="makehuman",
        nudity_edits=False,
        eyes=True,
        tongue=False,
        remove_unattached_vertices=False,
        triangulate_faces=True,
    )
    ref_data = build_anny_model_data(rig=source_rig,
                                topology=source_topology,local_changes=local_changes,
                                facial_actions=facial_actions)

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
    if rig.bone_orientation == "procrustes":
        data = apply_procrustes_orientation(data)
    elif rig.bone_orientation == "cached":
        data = apply_anny_cached_orientation(data)
    return data

def build_smpl_topology_model_data(
                                rig: RigConfig, local_changes: LocalChanges,
                                facial_actions: bool):
    source_rig = with_bone_orientation(rig, "blender")
    source_topology = TopologyConfig(
        base_mesh="makehuman",
        nudity_edits=False,
        eyes=True,
        tongue=False,
        remove_unattached_vertices=False,
        triangulate_faces=True,
    )
    ref_data = build_anny_model_data(rig=source_rig,
                                topology=source_topology, local_changes=local_changes,
                                facial_actions=facial_actions)

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
    if rig.bone_orientation == "procrustes":
        data = apply_procrustes_orientation(data)
    elif rig.bone_orientation == "cached":
        data = apply_anny_cached_orientation(data)
    return data

def build_alternative_topology_model_data(
                                      rig: RigConfig,
                                      topology: TopologyConfig,
                                      local_changes: LocalChanges,
                                      facial_actions: bool,
                                      reference_topology: Literal["legacy_default", "anny_from_soma", "anny"]="anny"):
    # For soma, the template mesh has only attached vertices and eyes+tongue
    source_rig = with_bone_orientation(rig, "blender")
    source_topology = TopologyConfig.from_string("anny")
    if reference_topology == "anny_from_soma":
        source_topology = replace(source_topology, remove_unattached_vertices=True, eyes=True, tongue=True)
    if reference_topology == "legacy_default":
        source_topology = replace(source_topology, remove_unattached_vertices=False, eyes=False, tongue=False)

    ref_data = build_anny_model_data(rig=source_rig,
                                topology=source_topology, local_changes=local_changes,
                                facial_actions=facial_actions)
    if reference_topology == "anny":
        reference_vertices = ref_data.template_vertices
        reference_faces = ref_data.faces
    else:
        reference_vertices, reference_faces = _load_target_topology_mesh(reference_topology)
    if topology.base_mesh == "makehuman":
        raise ValueError("Alternative topologies must have base_mesh other than 'makehuman'.")
    vertices, faces = _load_target_topology_mesh(topology.base_mesh)

    data = apply_retopology_from_mesh(
        ref_data,
        target_vertices=vertices,
        target_faces=faces,
        source_vertices=reference_vertices,
        source_faces=reference_faces,
    )
    if topology.triangulate_faces:
        data = triangulate(data)
    if rig.bone_orientation == "procrustes":
        data = apply_procrustes_orientation(data)
    elif rig.bone_orientation == "cached":
        data = apply_anny_cached_orientation(data)
    return data
