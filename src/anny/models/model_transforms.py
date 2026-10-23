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
from anny.typing import Phenotypes
from anny.utils.mesh_utils import (
    get_edge_vertex_indices,
    get_symmetric_vertex_indices,
    triangulate_faces,
    triangulate_faces_with_texture_coordinates,
)

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


def filter_blendshapes(data: ModelData, mask: list[bool]) -> ModelData:
    if len(mask) != len(data.metadata.blendshape_labels):
        raise ValueError(
            f"mask length {len(mask)} does not match "
            f"the number of blendshape labels {len(data.metadata.blendshape_labels)}"
        )
    blendshape_mask = torch.tensor(mask, dtype=torch.bool)
    extra = {}
    if data.bone_tails_blendshapes is not None:
        extra["bone_tails_blendshapes"] = data.bone_tails_blendshapes[blendshape_mask]
    return dataclasses.replace(
        data,
        metadata=dataclasses.replace(
            data.metadata,
            blendshape_labels=[
                data.metadata.blendshape_labels[i] for i in range(len(mask)) if mask[i]
            ],
        ),
        blendshapes=data.blendshapes[blendshape_mask],
        bone_heads_blendshapes=data.bone_heads_blendshapes[blendshape_mask],
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
    regression_weights,
    check_weights: bool = True,
) -> ModelData:
    """Linearly interpolate skinning weights from *source_model* onto the target topology in *data*.

    K = reference_vertex_indices.shape[1] source vertices are combined per target vertex;
    regression_weights has the same (N, K) shape as reference_vertex_indices.

    Bones are accumulated densely per target vertex, then repacked keeping only the bones whose
    total influence is non-negligible, ordered by first appearance in the source packing. Targets
    are processed in chunks so that the dense (chunk, bone_count) accumulator stays bounded for
    large target counts.

    The interpolation coordinates need not be barycentric: negative values are supported, and they
    are used as given, without renormalization (the caller applies the same coordinates verbatim to
    the template vertices and the blendshapes). Only the skinning weights they produce are
    renormalized, per target vertex, because linear blend skinning requires them to sum to 1. A
    target whose accumulated weights cancel out has no meaningful binding and is rejected when
    check_weights is True (the default); with check_weights=False such targets keep zero weights.
    """
    num_targets, num_references = reference_vertex_indices.shape
    bone_count = int(source_model.vertex_bone_indices.max()) + 1
    chunk_size = max(1, _SKINNING_CHUNK_ELEMENTS // bone_count)

    chunks = [
        _interpolate_skinning_weights_chunk(
            source_model=source_model,
            reference_vertex_indices=reference_vertex_indices[
                start : start + chunk_size
            ],
            regression_weights=regression_weights[start : start + chunk_size],
            bone_count=bone_count,
        )
        for start in range(0, num_targets, chunk_size)
    ]

    max_bones_per_vertex = max(indices.shape[1] for indices, _ in chunks)
    logger.info(f"{max_bones_per_vertex=}")
    vbi = torch.cat(
        [_pad_columns(indices, max_bones_per_vertex) for indices, _ in chunks]
    )
    vbw = torch.cat(
        [_pad_columns(weights, max_bones_per_vertex) for _, weights in chunks]
    )

    row_sums = vbw.sum(dim=-1, keepdim=True)
    degenerate = (row_sums.abs() < _MIN_SKINNING_WEIGHT).squeeze(-1)
    if degenerate.any():
        if check_weights:
            raise ValueError(
                f"{int(degenerate.sum())} of {num_targets} target vertices have skinning weights "
                "summing to zero, leaving them unbound. Their interpolation coordinates cancel out "
                f"the source bindings; first offender is target {int(degenerate.nonzero()[0])}."
            )
        row_sums = torch.where(degenerate[:, None], torch.ones_like(row_sums), row_sums)
        vbw = torch.where(degenerate[:, None], torch.zeros_like(vbw), vbw)
    # The accumulation runs in float64 for precision; the result follows the source skinning
    # weights so that the retopologized data keeps a single floating dtype.
    vertex_bone_weights = (vbw / row_sums).to(
        dtype=source_model.vertex_bone_weights.dtype,
        device=source_model.vertex_bone_weights.device,
    )
    return dataclasses.replace(
        data, vertex_bone_weights=vertex_bone_weights, vertex_bone_indices=vbi
    )


# Element budget of the dense per-chunk bone accumulator (~32 MB in float64).
_SKINNING_CHUNK_ELEMENTS = 4 * 1024 * 1024

# Below this magnitude an accumulated bone weight is treated as cancellation noise rather than as
# an influence. Non-barycentric coordinates make such near-cancellations common; keeping them would
# cost a packed column each.
_MIN_SKINNING_WEIGHT = 1e-12


def _pad_columns(tensor: torch.Tensor, width: int) -> torch.Tensor:
    """Right-pad a (N, W) tensor with zeros up to *width* columns."""
    if tensor.shape[1] == width:
        return tensor
    padding = torch.zeros(
        (tensor.shape[0], width - tensor.shape[1]),
        dtype=tensor.dtype,
        device=tensor.device,
    )
    return torch.cat([tensor, padding], dim=1)


def _interpolate_skinning_weights_chunk(
    *,
    source_model: ModelData,
    reference_vertex_indices: torch.Tensor,
    regression_weights: torch.Tensor,
    bone_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Packed (indices, weights) of one chunk of target vertices, trimmed to its own width."""
    num_targets, num_references = reference_vertex_indices.shape
    device = source_model.vertex_bone_weights.device
    slot_count = source_model.vertex_bone_indices.shape[1]
    unset = num_references * slot_count

    accumulated = torch.zeros(
        (num_targets, bone_count), dtype=torch.float64, device=device
    )
    # Position of each bone's first appearance, so that the packed column order matches the
    # insertion order of the per-vertex accumulation.
    first_appearance = torch.full(
        (num_targets, bone_count), unset, dtype=torch.int64, device=device
    )
    for i in range(num_references):
        references = reference_vertex_indices[:, i]
        source_indices = source_model.vertex_bone_indices[references]
        source_weights = source_model.vertex_bone_weights[references].to(torch.float64)
        coefficients = regression_weights[:, i].to(torch.float64)
        accumulated.scatter_add_(
            1, source_indices, coefficients[:, None] * source_weights
        )
        slots = torch.arange(
            i * slot_count, (i + 1) * slot_count, device=device
        ).expand(num_targets, slot_count)
        first_appearance.scatter_reduce_(1, source_indices, slots, reduce="amin")

    # Pack, per target vertex, the bones that actually influence it, in first-appearance order.
    # The test is on the magnitude, not on the sign: a negative total is a genuine influence, since
    # the interpolation coordinates may be negative, while a total that cancelled out to ~0 adds
    # nothing to the skinning sum. The rig is untouched.
    ranks = torch.where(
        accumulated.abs() > _MIN_SKINNING_WEIGHT, first_appearance, unset
    )
    ranks, bone_order = torch.sort(ranks, dim=1)
    width = int((ranks < unset).sum(dim=1).max())
    ranks, bone_order = ranks[:, :width], bone_order[:, :width]

    kept = ranks < unset
    gathered = accumulated.gather(1, bone_order)
    return (
        torch.where(kept, bone_order, torch.zeros_like(bone_order)),
        torch.where(kept, gathered, torch.zeros_like(gathered)),
    )


def _apply_interpolate_model_data(
    data: ModelData,
    *,
    reference_vertex_indices: torch.Tensor,
    regression_weights: torch.Tensor,
    faces: torch.Tensor,
    base_mesh_vertex_indices,
    vertices: torch.Tensor | None,
    check_weights: bool = True,
) -> ModelData:
    """Shared core of apply_retopology and interpolate_model_data.

    reference_vertex_indices is (N, K); vertices may be None, in which case the target template
    vertices are interpolated from data.template_vertices.
    """
    expected_weights_shape = reference_vertex_indices.shape
    if regression_weights.shape != expected_weights_shape:
        raise ValueError(
            "regression_weights must have the same (N, K) shape as "
            f"reference_vertex_indices; expected {expected_weights_shape}, got "
            f"{tuple(regression_weights.shape)}."
        )

    if check_weights:
        if float(regression_weights.min()) < -1e-6:
            raise ValueError(
                f"weights must be non-negative, got a minimum of {float(regression_weights.min())}."
            )
        weight_sum_error = float((regression_weights.sum(dim=1) - 1.0).abs().max())
        if weight_sum_error > 1e-6:
            raise ValueError(
                "weights rows must sum to 1 (convex combination), largest deviation is "
                f"{weight_sum_error}. Renormalize them explicitly if that is intended."
            )
    num_references = reference_vertex_indices.shape[1]
    if base_mesh_vertex_indices is None:
        base_mesh_vertex_indices = torch.arange(
            reference_vertex_indices.shape[0], dtype=torch.int64
        )
    if vertices is None:
        vertices = sum(
            data.template_vertices[reference_vertex_indices[:, i]]
            * regression_weights[:, i, None]
            for i in range(num_references)
        )
    blendshapes = sum(
        data.blendshapes[:, reference_vertex_indices[:, i]]
        * regression_weights[None, :, i, None]
        for i in range(num_references)
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
        regression_weights=regression_weights,
        check_weights=check_weights,
    )


def apply_retopology(
    data: ModelData,
    vertices: torch.Tensor,
    faces: torch.Tensor,
    reference_vertex_indices: torch.Tensor,
    barycentric_coordinates,
    base_mesh_vertex_indices=None,
    check_weights=True,
) -> ModelData:
    """Apply a new mesh topology, interpolating blendshapes and skinning weights barycentrically.

    Caller-supplied target vertices, typically K=3 triangle barycentrics from a surface projection.
    Barycentric coordinates should have the same (N, K) shape as reference_vertex_indices. The
    historical sequence of K tensors of shape (N,), and tensor form (K, N), remain accepted.
    """
    if isinstance(barycentric_coordinates, torch.Tensor):
        regression_weights = barycentric_coordinates
        if regression_weights.shape != reference_vertex_indices.shape:
            regression_weights = regression_weights.transpose(0, 1)
    else:
        regression_weights = torch.stack(tuple(barycentric_coordinates), dim=1)

    return _apply_interpolate_model_data(
        data,
        reference_vertex_indices=reference_vertex_indices,
        regression_weights=regression_weights,
        faces=faces,
        base_mesh_vertex_indices=base_mesh_vertex_indices,
        vertices=vertices,
        check_weights=check_weights,
    )


# Vertex-indexed buffers of the legacy runtime-registration procrustes variant.
_PROCRUSTES_RUNTIME_FIELDS = (
    "bone_nonzeroweight_mask",
    "bone_vertex_indices",
    "bone_vertex_weights",
    "template_bone_vertices",
)


def _check_linear_interpolation_inputs(
    data: ModelData, indices: torch.Tensor, weights: torch.Tensor
) -> None:
    if indices.ndim != 2:
        raise ValueError(
            f"indices must be a 2-D (N, K) tensor, got shape {tuple(indices.shape)}."
        )
    if not torch.is_floating_point(weights):
        raise ValueError(f"weights must have a floating dtype, got {weights.dtype}.")
    if indices.dtype.is_floating_point or indices.dtype.is_complex:
        raise ValueError(f"indices must have an integer dtype, got {indices.dtype}.")
    if weights.shape != indices.shape:
        raise ValueError(
            f"weights and indices must have the same shape, got {tuple(weights.shape)} "
            f"and {tuple(indices.shape)}."
        )
    num_targets, num_references = indices.shape
    if num_references < 1:
        raise ValueError(
            "indices must combine at least one source vertex per target (K >= 1)."
        )
    if num_targets < 1:
        raise ValueError(
            "apply_barycentric_retopology requires at least one output vertex, got indices "
            f"of shape {tuple(indices.shape)}."
        )
    num_source_vertices = data.template_vertices.shape[0]
    if int(indices.min()) < 0 or int(indices.max()) >= num_source_vertices:
        raise ValueError(
            f"indices must be in [0, {num_source_vertices}), got values in "
            f"[{int(indices.min())}, {int(indices.max())}]."
        )


def interpolate_model_data(
    data: ModelData,
    indices: torch.Tensor,
    weights: torch.Tensor,
    faces: torch.Tensor | None = None,
    base_mesh_vertex_indices: torch.Tensor | None = None,
) -> ModelData:
    """Resample a model onto points given as convex combinations of its template vertices.

    Output vertex n is ``sum_k weights[n, k] * data.template_vertices[indices[n, k]]``, with indices
    of shape (N, K) and non-negative weights of shape (N, K) summing to 1 per row. Unlike
    apply_retopology_from_mesh, no target mesh, surface projection nor warp is needed.

    faces are optional and index the new vertices; without them the result is a point set, which
    rules out everything needing connectivity (triangulate, remove_unattached_vertices, volume
    computation, parameter regression, self-interpenetration). base_mesh_vertex_indices defaults to
    arange(N), meaningless for a resampled mesh; for a plain subsampling (K=1) pass
    data.base_mesh_vertex_indices[indices[:, 0]].

    Bone- and blendshape-indexed buffers are topology-independent and carried over, so bone poses
    match the source exactly; data must therefore come from Anny.to_model_data() and use precomputed
    orientation matrices (the vertex-indexed runtime procrustes buffers cannot be retargeted here).
    """
    _check_linear_interpolation_inputs(data, indices, weights)

    if data.bone_template_orientation_matrices is None and any(
        getattr(data, field) is not None for field in _PROCRUSTES_RUNTIME_FIELDS
    ):
        raise ValueError(
            "apply_barycentric_retopology cannot retarget the legacy runtime-registration "
            f"procrustes buffers ({', '.join(_PROCRUSTES_RUNTIME_FIELDS)}): they index source "
            "vertices, and rebuilding them requires projecting onto a target surface (see "
            "apply_procrustes_retopology), which an arbitrary point set does not have. Use a rig "
            "with precomputed bone_template_orientation_matrices (e.g. bone_orientation='cached', "
            "or apply_anny_cached_orientation(data))."
        )

    dtype, device = data.template_vertices.dtype, data.device
    indices = indices.to(dtype=torch.int64, device=device)
    weights = weights.to(dtype=dtype, device=device)

    if faces is None:
        faces = torch.empty((0, 3), dtype=torch.int64, device=device)
    else:
        faces = faces.to(dtype=torch.int64, device=device)

    if base_mesh_vertex_indices is None:
        base_mesh_vertex_indices = torch.arange(
            indices.shape[0], dtype=torch.int64, device=device
        )
    else:
        base_mesh_vertex_indices = base_mesh_vertex_indices.to(
            dtype=torch.int64, device=device
        )

    result = _apply_interpolate_model_data(
        data,
        reference_vertex_indices=indices,
        regression_weights=weights,
        faces=faces,
        base_mesh_vertex_indices=base_mesh_vertex_indices,
        vertices=None,
    )
    return dataclasses.replace(
        result, **{field: None for field in _PROCRUSTES_RUNTIME_FIELDS}
    )


def _project_onto_symmetry_plane(
    points: torch.Tensor,
    vertices: torch.Tensor,
    triangular_faces: torch.Tensor,
    axis: int,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project points onto the intersection of a triangle mesh with the plane ``x[axis] == 0``.

    Each triangle meets the plane along a segment, possibly reduced to a point: an edge lying in the
    plane, or the chord between two crossing points. Vertices closer than eps to the plane are
    considered on it. Returns (reference_vertex_indices, weights, distances), where the (M, 3)
    indices are the vertices of the triangle holding the closest point, and the (M, 3) weights are
    its barycentric coordinates within that triangle.
    """
    offsets = vertices[triangular_faces][..., axis]
    offsets = torch.where(offsets.abs() < eps, torch.zeros_like(offsets), offsets)
    if (offsets == 0).all(dim=1).any():
        raise ValueError(
            f"Some triangles lie within the symmetry plane (axis={axis}), which is not supported."
        )

    # Candidate intersection points, as barycentric coordinates within their triangle: the three
    # vertices, then the three edge crossings.
    num_triangles = len(triangular_faces)
    candidates = torch.zeros(
        (num_triangles, 6, 3), dtype=vertices.dtype, device=vertices.device
    )
    candidates[:, :3] = torch.eye(3, dtype=vertices.dtype, device=vertices.device)
    valid = torch.zeros((num_triangles, 6), dtype=torch.bool, device=vertices.device)
    valid[:, :3] = offsets == 0
    for i in range(3):
        j = (i + 1) % 3
        crossing = offsets[:, i] * offsets[:, j] < 0
        t = offsets[:, i] / torch.where(
            crossing, offsets[:, i] - offsets[:, j], torch.ones_like(offsets[:, i])
        )
        candidates[:, 3 + i, i] = 1.0 - t
        candidates[:, 3 + i, j] = t
        valid[:, 3 + i] = crossing

    # Reduce each triangle's candidates to the two endpoints of its intersection segment. A triangle
    # not lying in the plane meets it in at most two points: 0 (no intersection), 1 (it touches the
    # plane at a single vertex), or 2 (an in-plane edge, a vertex and the crossing of the opposite
    # edge, or two edge crossings).
    count = valid.sum(dim=1)
    assert int(count.max()) <= 2
    # Stable descending sort of the validity flags moves the valid candidates first, in their
    # original order; the first two are the segment endpoints, as barycentrics of shape (T, 2, 3).
    order = valid.to(torch.int8).argsort(dim=1, descending=True, stable=True)[:, :2]
    ends = candidates.gather(1, order[..., None].expand(-1, -1, 3))
    # A single touching vertex is treated as a zero-length segment.
    ends[:, 1] = torch.where(count[:, None] == 1, ends[:, 0], ends[:, 1])
    # Keep only the S triangles that meet the plane.
    kept = count > 0
    if not kept.any():
        raise ValueError(
            f"The source mesh does not meet the symmetry plane (axis={axis})."
        )
    faces, ends = triangular_faces[kept], ends[kept]

    # 3D segments [a, a + ab] of shape (S, 3), from their barycentric endpoints.
    triangle_vertices = vertices[faces]
    a = (ends[:, 0, :, None] * triangle_vertices).sum(dim=1)
    ab = (ends[:, 1, :, None] * triangle_vertices).sum(dim=1) - a

    # Point-to-segment projection of every point onto every segment, with the segment parameter t
    # of shape (M, S) clamped to [0, 1]. Zero-length segments get t = 0, i.e. their single point.
    squared_lengths = (ab * ab).sum(dim=-1)
    t = (
        torch.einsum("msd,sd->ms", points[:, None] - a[None], ab)
        / torch.where(
            squared_lengths > 0, squared_lengths, torch.ones_like(squared_lengths)
        )[None]
    )
    t = t.clamp(0.0, 1.0)
    distances = (points[:, None] - (a[None] + t[..., None] * ab[None])).norm(dim=-1)

    # Closest segment for each point.
    distances, segment_ids = distances.min(dim=1)
    t = t.gather(1, segment_ids[:, None])
    # Barycentric coordinates are affine, so interpolating those of the segment endpoints gives the
    # coordinates of the projected point within the triangle holding that segment.
    weights = (1.0 - t) * ends[segment_ids, 0] + t * ends[segment_ids, 1]
    return faces[segment_ids], weights, distances


def apply_retopology_from_mesh(
    data: ModelData,
    target_vertices: torch.Tensor,
    target_faces,
    source_vertices: torch.Tensor,
    source_faces,
    base_mesh_vertex_indices=None,
    symmetry_axis: int | None = None,
) -> ModelData:
    """Apply retopology by projecting target vertices onto a source mesh to compute barycentric coordinates.

    source_vertices and source_faces define the reference mesh for projection; their vertex indices
    must share the same ordering as data. source_faces may be pre-triangulated (shape (N, 3)) or not.
    The new topology's template vertices are derived by bary-interpolating data.template_vertices.

    With symmetry_axis set, target vertices lying in the plane ``target_vertices[:, symmetry_axis]
    == 0`` are projected onto the intersection of the source surface with that plane, instead of
    onto the whole surface. The closest surface point of such a vertex is otherwise ambiguous above
    a concave midline region (two mirror candidates at the same distance), which breaks the
    symmetry of the result. In convex regions both projections give the same point.
    """
    try:
        from anny.utils.warp_mesh_utils import point_to_mesh_distance_and_face_uvs
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "Warp not found. Install warp extra to use retopology."
        )

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

    uvs = uvs.to(dtype=torch.float64)
    u, v = uvs[:, 0], uvs[:, 1]
    barycentric_coords = torch.stack([u, v, 1.0 - u - v], dim=1)
    reference_vertex_indices = source_triangular_faces[face_ids]

    if symmetry_axis is not None:
        on_plane = target_vertices[:, symmetry_axis].abs() < 1e-6
        if on_plane.any():
            plane_indices, plane_coords, plane_distances = _project_onto_symmetry_plane(
                points=target_vertices[on_plane].to(dtype=torch.float64),
                vertices=source_vertices.to(dtype=torch.float64),
                triangular_faces=source_triangular_faces,
                axis=symmetry_axis,
            )
            reference_vertex_indices = reference_vertex_indices.clone()
            reference_vertex_indices[on_plane] = plane_indices
            barycentric_coords[on_plane] = plane_coords
            distances = distances.clone()
            distances[on_plane] = plane_distances.to(dtype=distances.dtype)

    assert distances.max() < 1.5e-2, (
        "Some vertices are too far from the reference model."
    )

    vertices = sum(
        barycentric_coords[:, i, None]
        * data.template_vertices[reference_vertex_indices[:, i]]
        for i in range(3)
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
    barycentric_coordinates: torch.Tensor,
    base_mesh_vertex_indices=None,
) -> ModelData:
    """Apply a new topology with procrustes-based bone orientation.

    Precomputed orientation matrices depend only on the blendshape coefficients, hence are
    topology-independent and carried over from *data* unchanged. With the legacy
    runtime-registration variant, the per-bone vertex buffers of *source_model* are instead
    rebuilt on the target topology.
    """
    try:
        from anny.utils.warp_mesh_utils import point_to_mesh_distance_and_face_uvs
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "Warp not found. Install warp extra to use retopology."
        )
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
        regression_weights=torch.stack(tuple(barycentric_coordinates), dim=1)
        if not isinstance(barycentric_coordinates, torch.Tensor)
        else barycentric_coordinates.transpose(0, 1),
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


def _with_nearest_rotations(transforms: torch.Tensor) -> torch.Tensor:
    """Homogeneous transforms (..., 4, 4) in float64, with their rotation blocks made orthonormal."""
    transforms = transforms.double().clone()
    transforms[..., :3, :3] = roma.special_procrustes(transforms[..., :3, :3])
    return transforms


def apply_soma_rig(
    data: ModelData, soma_rig_data: dict, procrustes_orientation_data: dict
) -> ModelData:
    """Replace the rig in *data* with the SOMA rig while keeping the mesh and blendshapes.

    Bone orientations come from the precomputed procrustes orientation data generated by
    ``scripts/precompute_procrustes_rig_weights.py --rig soma``.
    """
    sparse_rbf_matrix = soma_rig_data["sparse_rbf_matrix"]
    skinning_weights = soma_rig_data["skinning_weights"]
    # SOMA's frames are float32 rotations (to ~6e-8), which the kinematics invert by transposition.
    bind_world_transforms = _with_nearest_rotations(
        soma_rig_data["bind_world_transforms"]
    )
    t_pose_world = _with_nearest_rotations(soma_rig_data["t_pose_world"])
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


# ---------------------------------------------------------------------------
# Phenotype blendshape compaction
# ---------------------------------------------------------------------------

# Blocks of blendshape labels driven by the phenotype parameters.
_PHENOTYPE_BLENDSHAPE_BLOCKS = ("universal", "race", "height", "proportions", "breast")
_RACE_LABELS = ("african", "asian", "caucasian")


def _phenotype_corners(
    phenotype_labels: list[str], anchors, dtype: torch.dtype
) -> torch.Tensor:
    """(P, n) phenotype vectors whose blendshape coefficients affinely span the coefficients
    of every reachable phenotype: every combination of the interpolation anchors of the
    interpolated phenotypes, and of the one-hot race vectors when race is exposed."""
    import itertools

    axes = []
    race_indices = []
    for i, label in enumerate(phenotype_labels):
        if label in _RACE_LABELS:
            race_indices.append(i)
            axes.append([0.0])
        else:
            axes.append(anchors[label].tolist())
    corners = torch.tensor(list(itertools.product(*axes)), dtype=dtype)
    if race_indices:
        one_hot = torch.eye(len(race_indices), dtype=dtype)
        corners = corners.repeat_interleave(len(race_indices), dim=0)
        corners[:, race_indices] = one_hot.repeat(
            corners.shape[0] // len(race_indices), 1
        )
    return corners


def compact_phenotype_blendshapes(
    data: ModelData,
    phenotypes: Phenotypes = "default",
    rank: int | None = None,
    head_weight: float = 1.0,
    orientation_weight: float = 1e-2,
) -> ModelData:
    """Replace the phenotype blendshapes by a minimal basis of the rest states that the
    ``phenotypes`` reach.

    The rest state (vertices, bone heads, and bone tails or orientation matrices when present)
    is affine in the ``K`` phenotype blendshape coefficients ``c``, and ``c`` is an affine
    combination of its values at the corners of the phenotype anchor grid. The reachable rest
    states therefore span, around the template, a linear subspace of dimension ``r <= K`` (136
    for the default model, against 624 blendshapes), of which this transform fits an
    orthonormal basis by SVD. The compacted model keeps the template and uses the ``r`` basis
    directions as phenotype blendshapes, with coefficients ``c @ projection`` stored in the
    ``phenotype_blendshape_projection`` field and applied by
    :class:`anny.models.phenotype.Anny`. Facial action and local change blendshapes are kept
    unchanged.

    With ``rank=None`` the numerical rank is kept and the compacted model reproduces the
    original one up to round-off (float64 fit); the weights below then have no effect on the
    outputs. A smaller ``rank`` keeps the best approximation of the rest states over the
    anchor grid corners, in the metric defined by the weights: mean squared vertex
    displacement (m²), plus ``head_weight`` times the mean squared bone head (and tail)
    displacement (m²), plus ``orientation_weight`` (m²) times the mean squared Frobenius norm
    of the deltas of the (unit-scale) bone orientation matrices. A unit coefficient of the
    compacted basis is one meter of RMS displacement in this metric.

    The result depends on the phenotype set, recorded in ``metadata.phenotype_labels``, and
    is valid for both values of ``extrapolate_phenotypes``.
    """
    from anny.models.model_data import resolve_phenotypes
    from anny.models.phenotype import (
        make_phenotype_anchors,
        phenotype_blendshape_weights,
    )

    if data.stacked_phenotype_blend_shapes_mask is None:
        raise ValueError("Model data has no phenotype blendshapes to compact.")
    if data.metadata.phenotype_labels is not None:
        raise ValueError(
            "Model data is already restricted to the phenotypes "
            f"{data.metadata.phenotype_labels} (e.g. already compacted)."
        )
    labels = data.metadata.blendshape_labels
    K = data.stacked_phenotype_blend_shapes_mask.shape[0]
    if any(
        label.split(":")[0] not in _PHENOTYPE_BLENDSHAPE_BLOCKS for label in labels[:K]
    ):
        raise ValueError(
            "Phenotype blendshapes are expected to be the first blendshapes."
        )
    if any(label.split(":")[0] in _PHENOTYPE_BLENDSHAPE_BLOCKS for label in labels[K:]):
        raise ValueError(
            "Phenotype blendshapes are expected to be the first blendshapes."
        )

    dtype, device = data.template_vertices.dtype, data.template_vertices.device
    fit_dtype = torch.float64
    phenotype_labels = resolve_phenotypes(phenotypes)
    anchors = make_phenotype_anchors(fit_dtype, device)
    mask = data.stacked_phenotype_blend_shapes_mask.to(dtype=fit_dtype, device=device)

    def coefficients(phenotype_parameters: torch.Tensor) -> torch.Tensor:
        return phenotype_blendshape_weights(
            phenotype_parameters.to(device),
            phenotype_labels,
            anchors,
            mask,
            extrapolate=False,
        )

    corners = _phenotype_corners(phenotype_labels, anchors, fit_dtype)
    C = coefficients(corners)  # (P, K)

    # Rest-state blocks, as (K, d) deltas with their metric weights.
    V, B = data.template_vertices.shape[0], data.template_bone_heads.shape[0]
    blocks: dict[str, tuple[torch.Tensor, torch.Tensor, float]] = {
        "template_vertices": (
            data.template_vertices,
            data.blendshapes,
            1.0 / V,
        ),
        "template_bone_heads": (
            data.template_bone_heads,
            data.bone_heads_blendshapes,
            head_weight / B,
        ),
    }
    if data.bone_tails_blendshapes is not None:
        assert data.template_bone_tails is not None
        blocks["template_bone_tails"] = (
            data.template_bone_tails,
            data.bone_tails_blendshapes,
            head_weight / B,
        )
    if data.bone_orientation_blendshapes is not None:
        assert data.bone_template_orientation_matrices is not None
        blocks["bone_template_orientation_matrices"] = (
            data.bone_template_orientation_matrices,
            data.bone_orientation_blendshapes,
            orientation_weight / B,
        )
    _blendshape_field = {
        "template_vertices": "blendshapes",
        "template_bone_heads": "bone_heads_blendshapes",
        "template_bone_tails": "bone_tails_blendshapes",
        "bone_template_orientation_matrices": "bone_orientation_blendshapes",
    }
    names = list(blocks)
    sqrt_weights = [blocks[name][2] ** 0.5 for name in names]
    deltas = {name: blocks[name][1][:K].to(fit_dtype).flatten(1) for name in names}
    D = torch.cat([s * deltas[name] for name, s in zip(names, sqrt_weights)], dim=1)

    # Right singular vectors of Y = C D (rest-state offsets from the template at the corners)
    # without forming the (P, N) matrix: with the thin SVD C = U S Vh, Y and Z = (S Vh) D
    # share them (U has orthonormal columns).
    _, Sc, Vhc = torch.linalg.svd(C, full_matrices=False)
    q = int((Sc > max(C.shape) * torch.finfo(fit_dtype).eps * Sc[0]).sum())
    Z = (Sc[:q, None] * Vhc[:q]) @ D
    _, s, Vh = torch.linalg.svd(Z, full_matrices=False)
    numerical_rank = int(
        (s > max(C.shape[0], D.shape[1]) * torch.finfo(fit_dtype).eps * s[0]).sum()
    )
    if rank is None:
        rank = numerical_rank
    elif not 0 < rank <= numerical_rank:
        raise ValueError(
            f"rank must be in [1, {numerical_rank}] (the numerical rank), got {rank}."
        )
    E = Vh[:rank]
    # Deterministic signs: the largest-magnitude entry of each direction is positive.
    peak = E.abs().argmax(dim=1, keepdim=True)
    E = E * E.gather(1, peak).sign()
    projection = D @ E.T  # (K, rank)
    logger.info(
        f"Compacted {K} phenotype blendshapes into {rank} (numerical rank {numerical_rank})."
    )

    replacements = {}
    for name, split, sqrt_weight in zip(
        names, E.split([deltas[name].shape[1] for name in names], dim=1), sqrt_weights
    ):
        template, blendshapes, _ = blocks[name]
        replacements[_blendshape_field[name]] = torch.cat(
            (
                (split / sqrt_weight).reshape(rank, *template.shape).to(dtype),
                blendshapes[K:],
            )
        )
    return dataclasses.replace(
        data,
        metadata=dataclasses.replace(
            data.metadata,
            blendshape_labels=[f"phenotype_basis:{i:03d}" for i in range(rank)]
            + labels[K:],
            phenotype_labels=phenotype_labels,
        ),
        phenotype_blendshape_projection=projection.to(dtype),
        **replacements,
    )
