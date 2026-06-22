# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
from __future__ import annotations

import collections
import dataclasses
import logging

import roma
import torch
import trimesh
import trimesh.graph

from anny.models.phenotype import Anny
from anny.models.model_data import ModelData
from anny.typing import LocalChanges
from anny.utils.mesh_utils import (
    get_edge_vertex_indices,
    get_symmetric_vertex_indices,
    triangulate_faces,
    triangulate_faces_with_texture_coordinates,
)

logger = logging.getLogger(__name__)

def _get_symmetric_bone_name(bone_name: str) -> str:
    """Return the mirror counterpart of a bone name across the body's symmetry plane.

    Supports the built-in MakeHuman ``.L``/``.R`` suffixes and rigs that use
    ``Left``/``Right`` tokens, such as Mixamo and SOMA. Central bones are their
    own mirror and returned unchanged.
    """
    if bone_name.endswith(".L"):
        return bone_name[:-2] + ".R"
    if bone_name.endswith(".R"):
        return bone_name[:-2] + ".L"
    if "Left" in bone_name:
        return bone_name.replace("Left", "Right", 1)
    if "Right" in bone_name:
        return bone_name.replace("Right", "Left", 1)
    return bone_name


# ---------------------------------------------------------------------------
# Blend shape filtering
# ---------------------------------------------------------------------------


def filter_local_changes(data: ModelData, local_changes: LocalChanges) -> ModelData:
    assert data.metadata is not None, "ModelData.metadata must be set to filter local changes"
    labels = data.metadata.local_change_labels
    if local_changes == "none":
        local_changes_mask = [False] * len(labels)
    elif local_changes == "default":
        local_changes_mask = ["nipple" not in label.lower() for label in labels]
    elif local_changes == "all":
        local_changes_mask = [True] * len(labels)
    elif isinstance(local_changes, str):
        raise ValueError(
            f"Unknown local_changes preset {local_changes!r}. "
            "Expected 'none', 'default', 'all', or a sequence of label strings."
        )
    else:
        label_to_idx = {label: i for i, label in enumerate(labels)}
        local_changes_mask = [False] * len(labels)
        for label in local_changes:
            local_changes_mask[label_to_idx[label]] = True

    local_change_labels = [
        label for i, label in enumerate(labels) if local_changes_mask[i]
    ]
    non_local_count = len(data.blendshapes) - 2 * len(labels)
    blend_shapes_mask = torch.concatenate(
        (
            torch.ones(non_local_count, dtype=torch.bool),
            torch.as_tensor(local_changes_mask, dtype=torch.bool).repeat_interleave(2),
        )
    )
    extra = {}
    if data.bone_tails_blendshapes is not None:
        extra["bone_tails_blendshapes"] = data.bone_tails_blendshapes[blend_shapes_mask]
    return dataclasses.replace(
        data,
        blendshapes=data.blendshapes[blend_shapes_mask],
        bone_heads_blendshapes=data.bone_heads_blendshapes[blend_shapes_mask],
        metadata=dataclasses.replace(
            data.metadata, local_change_labels=local_change_labels
        ),
        **extra,
    )


# ---------------------------------------------------------------------------
# Mesh topology operations
# ---------------------------------------------------------------------------


def edit_mesh(data: ModelData) -> ModelData:
    """Apply the minor MakeHuman mesh edits (nudity-related face removals/caps)."""
    from anny.models.full_model import get_edited_mesh_faces

    new_faces, new_ftci = get_edited_mesh_faces(
        data.faces, data.face_texture_coordinate_indices
    )
    return dataclasses.replace(
        data, faces=new_faces, face_texture_coordinate_indices=new_ftci
    )


def filter_faces(data: ModelData, faces_to_keep: torch.Tensor) -> ModelData:
    return dataclasses.replace(
        data,
        faces=data.faces[faces_to_keep, :],
        face_texture_coordinate_indices=data.face_texture_coordinate_indices[
            faces_to_keep, :
        ],
    )


def triangulate(data: ModelData) -> ModelData:
    tri_faces, tri_ftci = triangulate_faces_with_texture_coordinates(
        vertices=data.template_vertices,
        faces=data.faces.detach().cpu().numpy().tolist(),
        face_texture_coordinate_indices=data.face_texture_coordinate_indices.detach()
        .cpu()
        .numpy()
        .tolist(),
    )
    return dataclasses.replace(
        data,
        faces=torch.as_tensor(tri_faces, dtype=torch.int64),
        face_texture_coordinate_indices=torch.as_tensor(tri_ftci, dtype=torch.int64),
    )


# ---------------------------------------------------------------------------
# Vertex / skinning operations
# ---------------------------------------------------------------------------


def remove_unattached_vertices(data: ModelData) -> ModelData:
    base_mesh_vertex_indices = torch.unique(data.faces.flatten(), sorted=True)
    old_to_new = torch.full(
        (len(data.template_vertices),), fill_value=-1, dtype=torch.int64
    )
    old_to_new[base_mesh_vertex_indices] = torch.arange(len(base_mesh_vertex_indices))
    _, verts_per_face = data.faces.shape
    new_faces = old_to_new[data.faces.flatten()].reshape(-1, verts_per_face)
    assert torch.all(new_faces >= 0)
    return dataclasses.replace(
        data,
        template_vertices=data.template_vertices[base_mesh_vertex_indices],
        vertex_bone_weights=data.vertex_bone_weights[base_mesh_vertex_indices],
        vertex_bone_indices=data.vertex_bone_indices[base_mesh_vertex_indices],
        blendshapes=data.blendshapes[:, base_mesh_vertex_indices, :],
        faces=new_faces,
        base_mesh_vertex_indices=base_mesh_vertex_indices,
    )


def symmetrize_skinning_weights(data: ModelData) -> ModelData:
    """Average per-vertex skinning weights across the body's YZ symmetry plane."""
    template_vertices = data.template_vertices
    vertex_bone_indices = data.vertex_bone_indices
    vertex_bone_weights = data.vertex_bone_weights
    bone_labels = data.metadata.bone_labels
    N = template_vertices.shape[0]
    B = len(bone_labels)

    sym = get_symmetric_vertex_indices(template_vertices, axis=0, threshold=1e-4)

    name_to_id = {n: i for i, n in enumerate(bone_labels)}
    bone_mirror = []
    for name in bone_labels:
        mate = _get_symmetric_bone_name(name)
        assert mate in name_to_id, (
            f"Bone {name!r} has no symmetric counterpart {mate!r}; "
            "remove L/R bones in pairs to keep symmetry well-defined."
        )
        bone_mirror.append(name_to_id[mate])
    bone_mirror = torch.as_tensor(bone_mirror, dtype=torch.int64)

    dense = torch.zeros(N, B, dtype=vertex_bone_weights.dtype)
    dense.scatter_add_(1, vertex_bone_indices, vertex_bone_weights)
    dense = 0.5 * (dense + dense[sym][:, bone_mirror])

    sorted_weights, sorted_indices = dense.sort(dim=-1, descending=True)
    new_K = int((sorted_weights > 0).sum(dim=-1).max().item())
    new_weights = sorted_weights[:, :new_K].contiguous()
    new_indices = sorted_indices[:, :new_K].contiguous()
    new_indices = torch.where(
        new_weights > 0, new_indices, torch.zeros_like(new_indices)
    )
    return dataclasses.replace(
        data, vertex_bone_weights=new_weights, vertex_bone_indices=new_indices
    )


def remove_skinning_islands(data: ModelData) -> ModelData:
    """Zero out per-bone skinning influence outside the largest connected component per bone."""
    edges = get_edge_vertex_indices(data.faces).cpu().numpy()
    vertex_bone_indices = data.vertex_bone_indices
    vertex_bone_weights = data.vertex_bone_weights.clone()

    mesh_components = trimesh.graph.connected_components(edges=edges)
    body_vertex_indices = torch.as_tensor(
        max(mesh_components, key=len), dtype=torch.int64
    )
    body_vertex_mask = torch.zeros(vertex_bone_weights.shape[0], dtype=torch.bool)
    body_vertex_mask[body_vertex_indices] = True

    for bone_id in range(len(data.metadata.bone_labels)):
        bone_vertex_mask = (
            (vertex_bone_indices == bone_id) & (vertex_bone_weights > 0)
        ).any(dim=-1) & body_vertex_mask
        if not bone_vertex_mask.any():
            continue
        skinned_nodes = bone_vertex_mask.cpu().numpy().nonzero()[0]
        components = trimesh.graph.connected_components(
            edges=edges, nodes=skinned_nodes
        )
        if len(components) <= 1:
            continue
        largest = max(components, key=len)
        drop_mask = bone_vertex_mask.clone()
        drop_mask[torch.as_tensor(largest, dtype=torch.int64)] = False
        slot_mask = (vertex_bone_indices == bone_id) & drop_mask[:, None]
        vertex_bone_weights[slot_mask] = 0.0

    row_sums = vertex_bone_weights.sum(dim=-1, keepdim=True)
    assert (row_sums > 0).all(), (
        "Some vertices have zero total skinning weight after island filtering."
    )
    return dataclasses.replace(data, vertex_bone_weights=vertex_bone_weights / row_sums)


def compact_skinning_weights(data: ModelData) -> ModelData:
    """Iteratively drop the trailing zero-weight bone slot until all slots are positive."""
    while True:
        vertex_bone_weights = data.vertex_bone_weights
        vertex_bone_indices = data.vertex_bone_indices
        index = torch.argmin(vertex_bone_weights, dim=-1)
        smallest_weight = torch.gather(
            vertex_bone_weights, dim=-1, index=index[:, None]
        )
        if torch.any(smallest_weight > 0):
            break
        vertices_count, max_bones = vertex_bone_weights.shape
        logger.info("Reducing the number of influencing bones to %d", max_bones - 1)
        mask = torch.arange(max_bones)[None, :] != index[:, None]
        data = dataclasses.replace(
            data,
            vertex_bone_weights=vertex_bone_weights[mask].reshape(
                vertices_count, max_bones - 1
            ),
            vertex_bone_indices=vertex_bone_indices[mask].reshape(
                vertices_count, max_bones - 1
            ),
        )
    return data


# ---------------------------------------------------------------------------
# Metadata update
# ---------------------------------------------------------------------------


def set_metadata(
    data: ModelData,
    *,
    skinning_method,
    pose_parameterization,
    all_phenotypes,
    extrapolate_phenotypes,
    bone_orientation,
) -> ModelData:
    return dataclasses.replace(
        data,
        metadata=dataclasses.replace(
            data.metadata,
            skinning_method=skinning_method,
            pose_parameterization=pose_parameterization,
            all_phenotypes=all_phenotypes,
            extrapolate_phenotypes=extrapolate_phenotypes,
            bone_orientation=bone_orientation,
        ),
    )


# ---------------------------------------------------------------------------
# Retopology
# ---------------------------------------------------------------------------


def interpolate_skinning_weights(
    data: ModelData,
    source_model: ModelData,
    reference_vertex_indices: torch.Tensor,
    barycentric_coordinates,
) -> ModelData:
    """Barycentric-interpolate skinning weights from *source_model* onto the target topology in *data*."""
    vertex_bone_weights = []
    vertex_bone_indices = []
    for vertex_id in range(len(reference_vertex_indices)):
        new_weights: dict[int, float] = collections.defaultdict(lambda: 0.0)
        for i in range(3):
            ref_vid = reference_vertex_indices[vertex_id, i].item()
            coeff = barycentric_coordinates[i][vertex_id].item()
            for bone_idx, bone_weight in zip(
                source_model.vertex_bone_indices[ref_vid],
                source_model.vertex_bone_weights[ref_vid],
            ):
                new_weights[bone_idx.item()] += coeff * bone_weight.item()
        new_weights = {k: v for k, v in new_weights.items() if v > 0.0}
        vertex_bone_weights.append(list(new_weights.values()))
        vertex_bone_indices.append(list(new_weights.keys()))

    max_bones_per_vertex = max(len(idx) for idx in vertex_bone_indices)
    logger.info(f"{max_bones_per_vertex=}")
    for indices, weights in zip(vertex_bone_indices, vertex_bone_weights):
        while len(indices) < max_bones_per_vertex:
            indices.append(0)
            weights.append(0.0)
    vbi = torch.as_tensor(vertex_bone_indices, dtype=torch.int64)
    vbw = torch.as_tensor(vertex_bone_weights, dtype=torch.float64)
    vbw /= torch.sum(vbw, dim=-1, keepdim=True)
    return dataclasses.replace(data, vertex_bone_weights=vbw, vertex_bone_indices=vbi)


def apply_retopology(
    data: ModelData,
    vertices: torch.Tensor,
    faces: torch.Tensor,
    reference_vertex_indices: torch.Tensor,
    barycentric_coordinates,
    base_mesh_vertex_indices=None,
) -> ModelData:
    """Apply a new mesh topology, interpolating blendshapes and skinning weights barycentrically."""
    if base_mesh_vertex_indices is None:
        base_mesh_vertex_indices = torch.arange(len(vertices), dtype=torch.int64)
    blendshapes = sum(
        data.blendshapes[:, reference_vertex_indices[:, i]]
        * barycentric_coordinates[i][None, :, None]
        for i in range(3)
    )
    target_data = dataclasses.replace(
        data,
        template_vertices=vertices,
        faces=faces,
        blendshapes=blendshapes,
        texture_coordinates=None,
        face_texture_coordinate_indices=None,
        base_mesh_vertex_indices=base_mesh_vertex_indices,
        metadata=dataclasses.replace(
            data.metadata,
            bone_orientation=(
                data.metadata.bone_orientation
                if data.metadata.bone_orientation != "procrustes"
                else "blender-rootidentity"
            ),
        ),
    )
    # data is used as source_model: ModelData exposes .vertex_bone_indices/.vertex_bone_weights directly
    return interpolate_skinning_weights(
        target_data,
        source_model=data,
        reference_vertex_indices=reference_vertex_indices,
        barycentric_coordinates=barycentric_coordinates,
    )


def apply_retopology_from_mesh(
    data: ModelData,
    target_vertices: torch.Tensor,
    target_faces,
    source_vertices: torch.Tensor,
    source_faces,
    base_mesh_vertex_indices=None,
) -> ModelData:
    """Apply retopology by projecting target vertices onto a source mesh to compute barycentric coordinates.

    source_vertices and source_faces define the reference mesh for projection; their vertex indices
    must share the same ordering as data. source_faces may be pre-triangulated (shape (N, 3)) or not.
    The new topology's template vertices are derived by bary-interpolating data.template_vertices.
    """
    assert data.template_vertices.shape[0] == len(source_vertices), (
        "source_vertices must have the same number of vertices as data.template_vertices"
    )

    def _triangulate(verts, faces):
        if isinstance(faces, torch.Tensor) and faces.ndim == 2 and faces.shape[1] == 3:
            return faces
        faces_list = faces.cpu().tolist() if isinstance(faces, torch.Tensor) else faces
        return torch.tensor(triangulate_faces(verts, faces_list), dtype=torch.int64)

    source_triangular_faces = _triangulate(source_vertices, source_faces)

    distances, face_ids, uvs = point_to_mesh_distance_and_face_uvs(
        points=target_vertices.to(dtype=torch.float32),
        vertices=source_vertices.to(dtype=torch.float32),
        faces=source_triangular_faces,
        max_dist=1000.0,
    )
    assert distances.max() < 1.5e-2, (
        "Some vertices are too far from the reference model."
    )

    uvs = uvs.to(dtype=torch.float64)
    u, v = uvs[:, 0], uvs[:, 1]
    w = 1.0 - u - v
    barycentric_coords = [u, v, w]
    reference_vertex_indices = source_triangular_faces[face_ids]
    vertices = (
        u[:, None] * data.template_vertices[reference_vertex_indices[:, 0]]
        + v[:, None] * data.template_vertices[reference_vertex_indices[:, 1]]
        + w[:, None] * data.template_vertices[reference_vertex_indices[:, 2]]
    )

    faces = _triangulate(target_vertices, target_faces)

    return apply_retopology(
        data,
        vertices=vertices,
        faces=faces,
        reference_vertex_indices=reference_vertex_indices,
        barycentric_coordinates=barycentric_coords,
        base_mesh_vertex_indices=base_mesh_vertex_indices,
    )


def apply_procrustes_retopology(
    data: ModelData,
    vertices: torch.Tensor,
    faces: torch.Tensor,
    source_model: ModelData,
    reference_vertex_indices: torch.Tensor,
    barycentric_coordinates,
    base_mesh_vertex_indices=None,
) -> ModelData:
    """Apply a new topology with procrustes-based bone orientation, reusing *source_model* buffers."""
    if base_mesh_vertex_indices is None:
        base_mesh_vertex_indices = torch.arange(len(vertices), dtype=torch.int64)
    blendshapes = sum(
        data.blendshapes[:, reference_vertex_indices[:, i]]
        * barycentric_coordinates[i][None, :, None]
        for i in range(3)
    )
    target_data = dataclasses.replace(
        data,
        template_vertices=vertices,
        faces=faces,
        blendshapes=blendshapes,
        texture_coordinates=None,
        face_texture_coordinate_indices=None,
        base_mesh_vertex_indices=base_mesh_vertex_indices,
        metadata=dataclasses.replace(
            data.metadata,
            bone_orientation="procrustes",
        ),
    )
    target_data = interpolate_skinning_weights(
        target_data,
        source_model=source_model,
        reference_vertex_indices=reference_vertex_indices,
        barycentric_coordinates=barycentric_coordinates,
    )

    bone_nonzeroweight_mask = source_model.bone_nonzeroweight_mask
    bvi_list: list[list[int]] = []
    bvw_list: list[list[float]] = []
    tbv_list: list[list[list[float]]] = []

    triangular_faces = torch.tensor(
        triangulate_faces(vertices, faces.detach().cpu().numpy().tolist()),
        dtype=torch.int64,
    )
    _, ref2target_face_ids, uvs = point_to_mesh_distance_and_face_uvs(
        points=source_model.template_vertices.to(dtype=torch.float32),
        vertices=vertices.to(dtype=torch.float32),
        faces=triangular_faces,
        max_dist=1000.0,
    )
    uvs = uvs.to(dtype=vertices.dtype)
    u, v = uvs[:, 0], uvs[:, 1]
    w = 1.0 - u - v
    ref2target_bary = torch.stack([u, v, w], dim=0)

    source = Anny.from_model_data(source_model)
    rest_bone_poses = source.get_rest_model(
        torch.zeros(
            (1, source_model.blendshapes.shape[0]),
            dtype=source_model.template_vertices.dtype,
            device=source_model.template_vertices.device,
        )
    )["rest_bone_poses"]
    for bone_nonzero_idx, bone_idx in enumerate(
        torch.nonzero(bone_nonzeroweight_mask).squeeze().numpy().tolist()
    ):
        weights: dict[int, float] = collections.defaultdict(lambda: 0)
        for i in range(source_model.bone_vertex_indices.shape[1]):
            ref_id = source_model.bone_vertex_indices[bone_nonzero_idx, i].item()
            ref_weight = torch.sqrt(
                source_model.bone_vertex_weights[bone_nonzero_idx, i]
            )
            face = triangular_faces[ref2target_face_ids[ref_id]]
            for k in range(3):
                u_k = ref2target_bary[k][ref_id]
                idx = face[k].item()
                weights[idx] += (u_k * ref_weight).sum().item()
        v_indices = list(weights.keys())
        bvi_list.append(v_indices)
        bvw_list.append(list(weights.values()))
        bone_verts = (
            roma.Rigid.from_homogeneous(rest_bone_poses[:, bone_idx, :, :])
            .inverse()
            .apply(vertices[v_indices])
            .numpy()
            .tolist()
        )
        tbv_list.append(bone_verts)

    k = max(len(idx) for idx in bvi_list)
    for indices, bweights, bone_verts in zip(bvi_list, bvw_list, tbv_list):
        while len(indices) < k:
            indices.append(0)
            bweights.append(0.0)
            bone_verts.append([0.0, 0.0, 0.0])

    bone_vertex_indices = torch.as_tensor(bvi_list, dtype=torch.int64)
    bone_vertex_weights = torch.square(torch.as_tensor(bvw_list, dtype=torch.float64))
    template_bone_vertices = torch.tensor(tbv_list, dtype=vertices.dtype)

    return dataclasses.replace(
        target_data,
        bone_nonzeroweight_mask=bone_nonzeroweight_mask,
        bone_vertex_indices=bone_vertex_indices,
        bone_vertex_weights=bone_vertex_weights,
        template_bone_vertices=template_bone_vertices,
    )


# ---------------------------------------------------------------------------
# SOMA rig replacement
# ---------------------------------------------------------------------------


def apply_soma_rig(data: ModelData, soma_rig_data: dict) -> ModelData:
    """Replace the rig in *data* with the SOMA rig while keeping the mesh and blendshapes."""

    sparse_rbf_matrix = soma_rig_data["sparse_rbf_matrix"]
    skinning_weights = soma_rig_data["skinning_weights"]
    bind_world_transforms = soma_rig_data["bind_world_transforms"]
    t_pose_world = soma_rig_data["t_pose_world"]
    bone_parents = soma_rig_data["bone_parents"]
    bone_labels = soma_rig_data["bone_labels"]
    bind_shape = soma_rig_data["bind_shape"]
    dtype = data.template_vertices.dtype

    vertex_count = data.template_vertices.shape[0]
    bone_positions = torch.mm(
        sparse_rbf_matrix, data.template_vertices.reshape(vertex_count, -1)
    )

    n_blendshapes = data.blendshapes.shape[0]
    blendshapes_flat = data.blendshapes.permute(1, 0, 2).reshape(vertex_count, -1)
    bone_heads_blendshapes = torch.mm(sparse_rbf_matrix, blendshapes_flat)
    bone_heads_blendshapes = bone_heads_blendshapes.reshape(
        -1, n_blendshapes, 3
    ).permute(1, 0, 2)

    template_bone_heads = bone_positions.clone()
    template_bone_heads[0] = template_bone_heads[1]
    bone_heads_blendshapes[:, 0] = bone_heads_blendshapes[:, 1]

    # LBS skinning weights: (vertex_count, k) format
    raw_weights, raw_indices = torch.sort(
        skinning_weights, dim=1, descending=True, stable=True
    )
    k_lbs = torch.count_nonzero(raw_weights > 0, dim=1).max().item()
    vertex_bone_weights = raw_weights[:, :k_lbs].to(dtype=dtype)
    vertex_bone_indices = raw_indices[:, :k_lbs]

    # Procrustes buffers: (bone_count, k) format
    bvw_sorted, bvi_sorted = torch.sort(
        skinning_weights, dim=0, descending=True, stable=True
    )
    bvw_sorted = bvw_sorted.t()
    bvi_sorted = bvi_sorted.t()

    weight_threshold = 0.01
    k_proc = torch.count_nonzero(bvw_sorted > weight_threshold, dim=-1).max().item()
    bone_nonzeroweight_mask = bvw_sorted.sum(dim=-1) > 0.0
    bone_vertex_indices = bvi_sorted[:, :k_proc][bone_nonzeroweight_mask].clone()
    bone_vertex_weights = (
        (bvw_sorted > weight_threshold)
        .to(dtype=dtype)[:, :k_proc][bone_nonzeroweight_mask]
        .clone()
    )
    template_bone_vertices = (
        roma.Rigid.from_homogeneous(bind_world_transforms)
        .inverse()[bone_nonzeroweight_mask, None]
        .apply(bind_shape[bone_vertex_indices])
    )
    reference_bone_orientations = t_pose_world[:, :3, :3].to(
        dtype=dtype, device=data.template_vertices.device
    )

    return dataclasses.replace(
        data,
        metadata=dataclasses.replace(
            data.metadata,
            bone_parents=bone_parents,
            bone_labels=bone_labels,
            bone_orientation="procrustes",
        ),
        template_bone_heads=template_bone_heads,
        bone_heads_blendshapes=bone_heads_blendshapes,
        vertex_bone_weights=vertex_bone_weights,
        vertex_bone_indices=vertex_bone_indices,
        bone_nonzeroweight_mask=bone_nonzeroweight_mask,
        bone_vertex_indices=bone_vertex_indices,
        bone_vertex_weights=bone_vertex_weights,
        template_bone_vertices=template_bone_vertices,
        reference_bone_orientations=reference_bone_orientations,
        # Tail-based orientation fields don't apply when bone_orientation="procrustes"
        template_bone_tails=None,
        bone_tails_blendshapes=None,
        bone_rolls_rotmat=None,
    )
