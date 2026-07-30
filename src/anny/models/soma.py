# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0

import dataclasses
import os

from anny.typing import LocalChanges, FacialActions
import torch

from anny.models.model_data import RigConfig, TopologyConfig
from anny.models.model_transforms import (
    apply_procrustes_retopology,
    apply_soma_rig,
)
from anny.utils.mesh_utils import triangulate_faces as _triangulate_faces
from anny.utils.warp_mesh_utils import point_to_mesh_distance_and_face_uvs
from anny.paths import get_anny_root_dir
from anny.models import retopology


def _load_soma_rig():
    """Load soma rig data, preferring .safetensors and falling back to legacy .pt."""
    pt_path = os.path.join(get_anny_root_dir(), "data/soma/soma_rig.pt")
    return torch.load(pt_path, weights_only=True)


def _load_cached_orientation_data():
    """Load the precomputed procrustes orientation data for the SOMA rig."""
    path = os.path.join(get_anny_root_dir(), "data/cached/soma.pth")
    return torch.load(path, weights_only=True)


def _load_soma_faces():
    """Load the canonical SOMA-X triangulation of the SOMA mesh.

    ``data/soma/SOMA_wrap.obj`` is a quad mesh; anny's on-the-fly shortest-diagonal
    triangulation picks per-quad diagonals that differ from the SOMA-X package's baked
    triangles (``SOMA_neutral.npz['triangles']``) on ~18% of quads. That diagonal choice is
    authored data, not a rule, so we bundle the canonical triangulation (a triangulation of the
    already-bundled quad mesh) and use it verbatim so ``topology='soma'`` faces match the
    ``soma`` package exactly.
    """
    path = os.path.join(get_anny_root_dir(), "data/soma/soma_faces.pt")
    return torch.load(path, weights_only=True)


def build_soma_rig_and_topology_model_data(
    local_changes: LocalChanges, facial_actions: FacialActions
):
    soma_rig_data = _load_soma_rig()
    procrustes_orientation_data = _load_cached_orientation_data()
    soma_data = retopology.build_alternative_topology_model_data(
        rig=RigConfig.from_string("anny"),
        topology=TopologyConfig.from_string("soma"),
        local_changes=local_changes,
        facial_actions=facial_actions,
        reference_topology="anny_from_soma",
    )
    data = apply_soma_rig(soma_data, soma_rig_data, procrustes_orientation_data)
    # Use the canonical SOMA-X triangulation so the SOMA-topology mesh (and the retopology
    # source it provides for other topologies) matches the soma package exactly (see
    # _load_soma_faces).
    data = dataclasses.replace(data, faces=_load_soma_faces())
    return data


def build_soma_rig_model_data(
    topology: TopologyConfig, local_changes: LocalChanges, facial_actions: FacialActions
):
    soma_data = build_soma_rig_and_topology_model_data(
        local_changes=local_changes,
        facial_actions=facial_actions,
    )

    if topology.base_mesh == "soma":
        return soma_data

    source_vertices = soma_data.template_vertices
    source_triangular_faces = torch.tensor(
        _triangulate_faces(soma_data.template_vertices, soma_data.faces.cpu().tolist()),
        dtype=torch.int64,
    )

    # Lazy import to avoid circular dependency with models/__init__.py
    from anny.models import build_model_data

    target_data = build_model_data(
        rig=RigConfig.from_string("anny"),
        local_changes=local_changes,
        facial_actions=facial_actions,
        topology=topology,
    )

    vertices = target_data.template_vertices

    _, target2source_face_ids, uvs = point_to_mesh_distance_and_face_uvs(
        points=vertices.to(dtype=torch.float32),
        vertices=source_vertices.to(dtype=torch.float32),
        faces=source_triangular_faces,
        max_dist=1000.0,
    )

    uvs = uvs.to(dtype=source_vertices.dtype)

    u, v = uvs[:, 0], uvs[:, 1]
    w = 1.0 - u - v
    target2source_barycentric_coordinates = torch.stack([u, v, w], dim=0)
    reference_vertex_indices = source_triangular_faces[target2source_face_ids]

    ref_data = soma_data
    data = apply_procrustes_retopology(
        ref_data,
        vertices=vertices,
        faces=target_data.faces,
        source_model=soma_data,
        reference_vertex_indices=reference_vertex_indices,
        barycentric_coordinates=target2source_barycentric_coordinates,
        base_mesh_vertex_indices=target_data.base_mesh_vertex_indices,
    )
    return data
