# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
from __future__ import annotations

import collections
import dataclasses
import logging
import os

import roma
import torch
import trimesh
import trimesh.graph

from anny.models.rigged_model import RiggedModelWithLinearBlendShapes
from anny.models.model_data import ModelData
from anny.paths import get_anny_root_dir
from anny.utils.mesh_utils import (
    get_edge_vertex_indices,
    get_symmetric_vertex_indices,
    triangulate_faces,
    triangulate_faces_with_texture_coordinates,
)
from anny.utils.warp_mesh_utils import point_to_mesh_distance_and_face_uvs

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class _ProcrustesBuffers:
    bone_nonzeroweight_mask: torch.Tensor
    bone_vertex_indices: torch.Tensor
    bone_vertex_weights: torch.Tensor
    template_bone_vertices: torch.Tensor


def _build_procrustes_buffers(
    *,
    vertices: torch.Tensor,
    bone_world_transforms: torch.Tensor,
    bone_nonzeroweight_mask: torch.Tensor,
    per_bone_vertex_indices: list[torch.Tensor],
    per_bone_vertex_weights: list[torch.Tensor],
    weight_dtype: torch.dtype | None = None,
) -> _ProcrustesBuffers:
    """Pack per-bone Procrustes samples and express them in each bone's local frame."""
    active_bone_indices = torch.nonzero(
        bone_nonzeroweight_mask,
        as_tuple=False,
    ).squeeze(1)
    if active_bone_indices.numel() == 0:
        raise ValueError("Cannot build procrustes buffers without active bones.")
    if len(per_bone_vertex_indices) != active_bone_indices.numel():
        raise ValueError("Expected one vertex-index tensor per active bone.")
    if len(per_bone_vertex_weights) != active_bone_indices.numel():
        raise ValueError("Expected one vertex-weight tensor per active bone.")

    device = vertices.device
    vertex_dtype = vertices.dtype
    if weight_dtype is None:
        weight_dtype = vertex_dtype

    bone_nonzeroweight_mask = bone_nonzeroweight_mask.to(device=device)
    active_bone_indices = active_bone_indices.to(device=device)
    bone_world_transforms = bone_world_transforms.to(dtype=vertex_dtype, device=device)

    max_vertices_per_bone = max(indices.numel() for indices in per_bone_vertex_indices)
    active_bone_count = active_bone_indices.numel()
    bone_vertex_indices = torch.zeros(
        (active_bone_count, max_vertices_per_bone),
        dtype=torch.int64,
        device=device,
    )
    bone_vertex_weights = torch.zeros(
        (active_bone_count, max_vertices_per_bone),
        dtype=weight_dtype,
        device=device,
    )

    for row, (indices, weights) in enumerate(
        zip(per_bone_vertex_indices, per_bone_vertex_weights)
    ):
        if indices.numel() != weights.numel():
            raise ValueError(
                "Procrustes vertex indices and weights must have matching lengths."
            )
        count = indices.numel()
        bone_vertex_indices[row, :count] = indices.to(dtype=torch.int64, device=device)
        bone_vertex_weights[row, :count] = weights.to(dtype=weight_dtype, device=device)

    if max_vertices_per_bone == 0:
        template_bone_vertices = torch.zeros(
            (active_bone_count, 0, 3),
            dtype=vertex_dtype,
            device=device,
        )
    else:
        template_bone_vertices = (
            roma.Rigid.from_homogeneous(bone_world_transforms[active_bone_indices])
            .inverse()[:, None]
            .apply(vertices[bone_vertex_indices])
        )

    return _ProcrustesBuffers(
        bone_nonzeroweight_mask=bone_nonzeroweight_mask,
        bone_vertex_indices=bone_vertex_indices,
        bone_vertex_weights=bone_vertex_weights,
        template_bone_vertices=template_bone_vertices,
    )


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


def filter_blendshapes(
    data: ModelData,
    local_changes_mask: list[bool],
    facial_actions: bool,
) -> ModelData:
    if len(local_changes_mask) != len(data.metadata.local_change_labels):
        raise ValueError(
            f"local_changes_mask length {len(local_changes_mask)} does not match "
            f"the number of local change labels {len(data.metadata.local_change_labels)}"
        )
    facial_count = len(data.metadata.facial_action_labels)
    non_local_count = len(data.blendshapes) - facial_count - 2 * len(local_changes_mask)
    blend_shapes_mask = torch.cat(
        (
            data.blendshapes.new_ones((non_local_count,), dtype=torch.bool),
            data.blendshapes.new_full(
                (facial_count,), facial_actions, dtype=torch.bool
            ),
            torch.tensor(
                local_changes_mask,
                dtype=torch.bool,
                device=data.blendshapes.device,
            ).repeat_interleave(2),
        )
    )
    extra = {}
    if data.bone_tails_blendshapes is not None:
        extra["bone_tails_blendshapes"] = data.bone_tails_blendshapes[blend_shapes_mask]
    local_change_labels = [
        data.metadata.local_change_labels[i]
        for i in range(len(data.metadata.local_change_labels))
        if local_changes_mask[i]
    ]
    blendshape_labels = [
        label
        for label, keep in zip(
            data.metadata.blendshape_labels, blend_shapes_mask.tolist()
        )
        if keep
    ]
    return dataclasses.replace(
        data,
        metadata=dataclasses.replace(
            data.metadata,
            local_change_labels=local_change_labels,
            facial_action_labels=data.metadata.facial_action_labels
            if facial_actions
            else [],
            blendshape_labels=blendshape_labels,
        ),
        blendshapes=data.blendshapes[blend_shapes_mask],
        bone_heads_blendshapes=data.bone_heads_blendshapes[blend_shapes_mask],
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
    if data.face_texture_coordinate_indices is None:
        tri_faces = triangulate_faces(
            vertices=data.template_vertices,
            faces=data.faces.detach().cpu().numpy().tolist(),
        )
        return dataclasses.replace(
            data, faces=torch.as_tensor(tri_faces, dtype=torch.int64)
        )
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
    bone_labels = data.metadata.bone_labels
    template_vertices = data.template_vertices
    vertex_bone_indices = data.vertex_bone_indices
    vertex_bone_weights = data.vertex_bone_weights

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
    bone_labels = data.metadata.bone_labels
    edges = get_edge_vertex_indices(data.faces).cpu().numpy()
    vertex_bone_indices = data.vertex_bone_indices
    original_vertex_bone_weights = data.vertex_bone_weights
    vertex_bone_weights = original_vertex_bone_weights.clone()

    mesh_components = trimesh.graph.connected_components(edges=edges)
    body_vertex_indices = torch.as_tensor(
        max(mesh_components, key=len), dtype=torch.int64
    )
    body_vertex_mask = torch.zeros(vertex_bone_weights.shape[0], dtype=torch.bool)
    body_vertex_mask[body_vertex_indices] = True

    for bone_id in range(len(bone_labels)):
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
    zero_weight_rows = row_sums.squeeze(-1) <= 0
    if zero_weight_rows.any():
        vertex_bone_weights[zero_weight_rows] = original_vertex_bone_weights[
            zero_weight_rows
        ]
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


def apply_procrustes_orientation(data: ModelData) -> ModelData:
    """Convert final tail-oriented ModelData into procrustes-oriented ModelData."""
    required_tail_fields = (
        data.template_bone_tails,
        data.bone_tails_blendshapes,
        data.bone_rolls_rotmat,
    )
    if any(field is None for field in required_tail_fields):
        raise ValueError(
            "Cannot convert ModelData to bone_orientation='procrustes' because "
            "tail orientation buffers are missing."
        )

    source = RiggedModelWithLinearBlendShapes(data, bone_orientation="blender")
    blendshape_coeffs = torch.zeros(
        (1, data.blendshapes.shape[0]),
        dtype=data.template_vertices.dtype,
        device=data.template_vertices.device,
    )
    rest_bone_poses = source.get_rest_model(blendshape_coeffs)["rest_bone_poses"]

    bone_count = len(data.metadata.bone_parents)
    bone_nonzeroweight_mask = torch.zeros(
        bone_count,
        dtype=torch.bool,
        device=data.vertex_bone_weights.device,
    )
    per_bone_vertex_indices: list[torch.Tensor] = []
    per_bone_vertex_weights: list[torch.Tensor] = []

    for bone_idx in range(bone_count):
        slot_mask = data.vertex_bone_indices == bone_idx
        vertex_weights = torch.where(
            slot_mask,
            data.vertex_bone_weights,
            torch.zeros_like(data.vertex_bone_weights),
        ).sum(dim=-1)
        vertex_mask = vertex_weights > 0
        if not vertex_mask.any():
            continue

        vertex_indices = torch.nonzero(vertex_mask, as_tuple=False).squeeze(1)
        weights = vertex_weights[vertex_indices]
        weights = weights / weights.sum()

        bone_nonzeroweight_mask[bone_idx] = True
        per_bone_vertex_indices.append(vertex_indices)
        per_bone_vertex_weights.append(weights)

    if not per_bone_vertex_indices:
        raise ValueError(
            "Cannot convert ModelData to bone_orientation='procrustes' because "
            "no bone has positive skinning influence."
        )

    procrustes_buffers = _build_procrustes_buffers(
        vertices=data.template_vertices,
        bone_world_transforms=rest_bone_poses[0],
        bone_nonzeroweight_mask=bone_nonzeroweight_mask,
        per_bone_vertex_indices=per_bone_vertex_indices,
        per_bone_vertex_weights=per_bone_vertex_weights,
    )

    return dataclasses.replace(
        data,
        bone_nonzeroweight_mask=procrustes_buffers.bone_nonzeroweight_mask,
        bone_vertex_indices=procrustes_buffers.bone_vertex_indices,
        bone_vertex_weights=procrustes_buffers.bone_vertex_weights,
        template_bone_vertices=procrustes_buffers.template_bone_vertices,
        template_bone_tails=None,
        bone_tails_blendshapes=None,
        bone_rolls_rotmat=None,
    )


def apply_anny_cached_orientation(
    data: ModelData, root_bone_source_label: str | None = None
) -> ModelData:
    """Apply the precomputed tail-aimed procrustes covariance (``data/cached/anny.pth``) to an
    anny-family rig, replacing the legacy runtime registration.

    Bones are selected by label (asserting full coverage); blendshape rows are selected by label. The
    facial-action orientation rows are then nulled: facial actions deform the face mesh, not the
    skeleton, so their coefficients must not reorient bones. No runtime child-offset refiner is used:
    tail-aiming is already baked into the covariance.
    """
    orientation_data = torch.load(
        os.path.join(get_anny_root_dir(), "data/cached/anny.pth"), weights_only=True
    )
    dtype, device = data.template_vertices.dtype, data.template_vertices.device

    source_bone_labels = [str(label) for label in orientation_data["bone_labels"]]
    source_index = {label: i for i, label in enumerate(source_bone_labels)}
    target_bone_labels = list(data.metadata.bone_labels)
    source_labels_for_target = target_bone_labels.copy()
    if root_bone_source_label is not None:
        assert target_bone_labels[0] == "root"
        source_labels_for_target[0] = root_bone_source_label
    missing = [label for label in source_labels_for_target if label not in source_index]
    assert not missing, (
        "Precomputed anny orientation data (data/cached/anny.pth) does not cover bones: "
        f"{missing}."
    )
    bone_selection = [source_index[label] for label in source_labels_for_target]

    bone_template_orientation_matrices = orientation_data[
        "bone_template_orientation_matrices"
    ][bone_selection].to(dtype=dtype, device=device)
    reference_bone_orientations = orientation_data["reference_bone_orientations"][
        bone_selection
    ].to(dtype=dtype, device=device)
    bone_orientation_blendshapes = _select_blendshape_rows(
        orientation_data["bone_orientation_blendshapes"][:, bone_selection],
        source_labels=orientation_data["blendshape_labels"],
        target_labels=data.metadata.blendshape_labels,
    ).to(dtype=dtype, device=device)

    # Bone origins ("heads") from the precomputed data (with the root realigned to the pelvis), kept
    # consistent with the frames the covariance was computed against.
    template_bone_heads = orientation_data["template_bone_heads"][bone_selection].to(
        dtype=dtype, device=device
    )
    bone_heads_blendshapes = _select_blendshape_rows(
        orientation_data["bone_heads_blendshapes"][:, bone_selection],
        source_labels=orientation_data["blendshape_labels"],
        target_labels=data.metadata.blendshape_labels,
    ).to(dtype=dtype, device=device)

    return dataclasses.replace(
        data,
        template_bone_heads=template_bone_heads,
        bone_heads_blendshapes=bone_heads_blendshapes,
        bone_template_orientation_matrices=bone_template_orientation_matrices,
        bone_orientation_blendshapes=bone_orientation_blendshapes,
        reference_bone_orientations=reference_bone_orientations,
        # Tail-aiming is baked into the covariance: no runtime child-offset refiner.
        bone_children_indices=None,
        bone_children_mask=None,
        bone_children_local_offsets=None,
        # Runtime-registration buffers are replaced by the covariance.
        bone_nonzeroweight_mask=None,
        bone_vertex_indices=None,
        bone_vertex_weights=None,
        template_bone_vertices=None,
        # Tail-based orientation fields don't apply when bone_orientation="procrustes".
        template_bone_tails=None,
        bone_tails_blendshapes=None,
        bone_rolls_rotmat=None,
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
    """Apply a new topology with procrustes-based bone orientation.

    Precomputed orientation matrices depend only on the blendshape coefficients, hence are
    topology-independent and carried over from *data* unchanged. With the legacy
    runtime-registration variant, the per-bone vertex buffers of *source_model* are instead
    rebuilt on the target topology.
    """
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
    )
    target_data = interpolate_skinning_weights(
        target_data,
        source_model=source_model,
        reference_vertex_indices=reference_vertex_indices,
        barycentric_coordinates=barycentric_coordinates,
    )

    if source_model.bone_template_orientation_matrices is not None:
        return target_data

    # Legacy runtime-registration variant: rebuild the per-bone vertex buffers on the target topology.
    if (
        source_model.bone_nonzeroweight_mask is None
        or source_model.bone_vertex_indices is None
        or source_model.bone_vertex_weights is None
    ):
        raise ValueError("source_model must contain procrustes buffers.")

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

    source = RiggedModelWithLinearBlendShapes(
        source_model, bone_orientation="procrustes"
    )
    rest_bone_poses = source.get_rest_model(
        torch.zeros(
            (1, source_model.blendshapes.shape[0]),
            dtype=source_model.template_vertices.dtype,
            device=source_model.template_vertices.device,
        )
    )["rest_bone_poses"]

    per_bone_vertex_indices: list[torch.Tensor] = []
    per_bone_vertex_weights: list[torch.Tensor] = []
    for bone_nonzero_idx in range(source_model.bone_vertex_indices.shape[0]):
        weights: dict[int, float] = collections.defaultdict(lambda: 0.0)
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
        per_bone_vertex_indices.append(
            torch.as_tensor(list(weights.keys()), dtype=torch.int64)
        )
        per_bone_vertex_weights.append(
            torch.square(torch.as_tensor(list(weights.values()), dtype=torch.float64))
        )

    procrustes_buffers = _build_procrustes_buffers(
        vertices=vertices,
        bone_world_transforms=rest_bone_poses[0],
        bone_nonzeroweight_mask=source_model.bone_nonzeroweight_mask,
        per_bone_vertex_indices=per_bone_vertex_indices,
        per_bone_vertex_weights=per_bone_vertex_weights,
        weight_dtype=torch.float64,
    )

    return dataclasses.replace(
        target_data,
        bone_nonzeroweight_mask=procrustes_buffers.bone_nonzeroweight_mask,
        bone_vertex_indices=procrustes_buffers.bone_vertex_indices,
        bone_vertex_weights=procrustes_buffers.bone_vertex_weights,
        template_bone_vertices=procrustes_buffers.template_bone_vertices,
    )


# ---------------------------------------------------------------------------
# SOMA rig replacement
# ---------------------------------------------------------------------------


def regress_soma_bone_origins(
    sparse_rbf_matrix: torch.Tensor,
    template_vertices: torch.Tensor,
    blendshapes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """RBF-regress SOMA bone origins and their blendshape displacements from the mesh.

    The root bone (index 0) has no dedicated regressor and reuses the Hips (index 1) position.
    """
    vertex_count = template_vertices.shape[0]
    template_bone_origins = torch.mm(
        sparse_rbf_matrix, template_vertices.reshape(vertex_count, -1)
    )

    n_blendshapes = blendshapes.shape[0]
    blendshapes_flat = blendshapes.permute(1, 0, 2).reshape(vertex_count, -1)
    bone_origins_blendshapes = torch.mm(sparse_rbf_matrix, blendshapes_flat)
    bone_origins_blendshapes = bone_origins_blendshapes.reshape(
        -1, n_blendshapes, 3
    ).permute(1, 0, 2)

    template_bone_origins[0] = template_bone_origins[1]
    bone_origins_blendshapes[:, 0] = bone_origins_blendshapes[:, 1]
    return template_bone_origins, bone_origins_blendshapes


def _select_blendshape_rows(
    orientation_blendshapes: torch.Tensor,
    source_labels: list[str],
    target_labels: list[str],
) -> torch.Tensor:
    """Copy the blendshape rows matching a target blendshape stack, identified by their labels.
    Use zero for facial actions."""
    source_index = {label: i for i, label in enumerate(source_labels)}
    missing = [
        label
        for label in target_labels
        if label not in source_index and not label.startswith("facial_action:")
    ]
    if missing:
        raise ValueError(
            f"Precomputed orientation data does not cover the following blend shapes: {missing}."
        )
    zero = orientation_blendshapes.new_zeros((1, *orientation_blendshapes.shape[1:]))
    padded = torch.cat((orientation_blendshapes, zero))
    return padded[[source_index.get(label, -1) for label in target_labels]]


def _build_bone_children_buffers(
    bone_parents,
    bind_world_transforms: torch.Tensor,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack, for each bone, its children indices and their bind-pose offsets expressed in the
    bone's bind frame. Used to refine procrustes orientations like the SOMA skeleton fit."""
    bone_count = len(bone_parents)
    children: list[list[int]] = [[] for _ in range(bone_count)]
    for bone_idx in range(bone_count):
        parent_idx = int(bone_parents[bone_idx])
        if parent_idx >= 0:
            children[parent_idx].append(bone_idx)

    max_children = max(len(c) for c in children)
    bone_children_indices = torch.zeros((bone_count, max_children), dtype=torch.int64)
    bone_children_mask = torch.zeros((bone_count, max_children), dtype=dtype)
    for bone_idx, bone_children in enumerate(children):
        count = len(bone_children)
        bone_children_indices[bone_idx, :count] = torch.tensor(
            bone_children, dtype=torch.int64
        )
        bone_children_mask[bone_idx, :count] = 1.0

    bind = bind_world_transforms.to(dtype=dtype)
    offsets = bind[bone_children_indices, :3, 3] - bind[:, None, :3, 3]
    bone_children_local_offsets = torch.einsum("kji,kcj->kci", bind[:, :3, :3], offsets)
    bone_children_local_offsets = (
        bone_children_local_offsets * bone_children_mask[:, :, None]
    )
    return bone_children_indices, bone_children_mask, bone_children_local_offsets


def apply_soma_rig(
    data: ModelData, soma_rig_data: dict, procrustes_orientation_data: dict
) -> ModelData:
    """Replace the rig in *data* with the SOMA rig while keeping the mesh and blendshapes.

    Bone orientations come from the precomputed procrustes orientation data generated by
    ``scripts/precompute_procrustes_rig_weights.py --rig soma``.
    """
    sparse_rbf_matrix = soma_rig_data["sparse_rbf_matrix"]
    skinning_weights = soma_rig_data["skinning_weights"]
    bind_world_transforms = soma_rig_data["bind_world_transforms"]
    t_pose_world = soma_rig_data["t_pose_world"]
    bone_parents = soma_rig_data["bone_parents"]
    bone_labels = soma_rig_data["bone_labels"]
    dtype = data.template_vertices.dtype

    template_bone_heads, bone_heads_blendshapes = regress_soma_bone_origins(
        sparse_rbf_matrix, data.template_vertices, data.blendshapes
    )

    # LBS skinning weights: (vertex_count, k) format
    raw_weights, raw_indices = torch.sort(
        skinning_weights, dim=1, descending=True, stable=True
    )
    k_lbs = torch.count_nonzero(raw_weights > 0, dim=1).max().item()
    vertex_bone_weights = raw_weights[:, :k_lbs].to(dtype=dtype)
    vertex_bone_indices = raw_indices[:, :k_lbs]

    # Precomputed procrustes orientation data, with blendshape rows matching the model configuration.
    if list(procrustes_orientation_data["bone_labels"]) != [
        str(label) for label in bone_labels
    ]:
        raise ValueError(
            "Precomputed procrustes orientation data does not match the SOMA rig bones."
        )
    bone_orientation_blendshapes = _select_blendshape_rows(
        procrustes_orientation_data["bone_orientation_blendshapes"],
        source_labels=procrustes_orientation_data["blendshape_labels"],
        target_labels=data.metadata.blendshape_labels,
    ).to(dtype=dtype, device=data.template_vertices.device)
    bone_template_orientation_matrices = procrustes_orientation_data[
        "bone_template_orientation_matrices"
    ].to(dtype=dtype, device=data.template_vertices.device)

    reference_bone_orientations = t_pose_world[:, :3, :3].to(
        dtype=dtype, device=data.template_vertices.device
    )

    # Children structure and bind-frame offsets, used to refine the orientations like the SOMA skeleton fit.
    bone_children_indices, bone_children_mask, bone_children_local_offsets = (
        _build_bone_children_buffers(bone_parents, bind_world_transforms, dtype=dtype)
    )

    return dataclasses.replace(
        data,
        metadata=dataclasses.replace(
            data.metadata,
            bone_parents=bone_parents,
            bone_labels=bone_labels,
        ),
        template_bone_heads=template_bone_heads,
        bone_heads_blendshapes=bone_heads_blendshapes,
        vertex_bone_weights=vertex_bone_weights,
        vertex_bone_indices=vertex_bone_indices,
        bone_template_orientation_matrices=bone_template_orientation_matrices,
        bone_orientation_blendshapes=bone_orientation_blendshapes,
        reference_bone_orientations=reference_bone_orientations,
        bone_children_indices=bone_children_indices,
        bone_children_mask=bone_children_mask,
        bone_children_local_offsets=bone_children_local_offsets,
        # Runtime-registration buffers of the source rig don't apply to the SOMA rig
        bone_nonzeroweight_mask=None,
        bone_vertex_indices=None,
        bone_vertex_weights=None,
        template_bone_vertices=None,
        # Tail-based orientation fields don't apply when bone_orientation="procrustes"
        template_bone_tails=None,
        bone_tails_blendshapes=None,
        bone_rolls_rotmat=None,
    )
