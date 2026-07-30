# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
import json
import logging
import os
import gzip
from dataclasses import dataclass, replace
from typing import cast

import roma
import torch

from anny.models.model_transforms import (
    edit_mesh,
    filter_faces,
    triangulate,
    compact_skinning_weights,
    apply_anny_cached_orientation,
    apply_procrustes_orientation,
)
import anny.utils.obj_utils
from anny.models.facial_actions import load_facial_action_blendshapes
from anny.models.phenotype import PHENOTYPE_VARIATIONS
from anny.models.model_data import (
    ModelData,
    ModelMetadata,
    cache_builder,
    TopologyConfig,
    RigConfig,
    resolve_blendshape_mask,
)
from anny.paths import get_anny_root_dir, PathLike
import anny.models.model_transforms as model_transforms
from anny.typing import FacialActions, LocalChanges, Submodel
from anny.face_segmentation import get_face_segmentation_mask

logger = logging.getLogger(__name__)


def _faces_to_keep_from_submodel(
    base_data: ModelData, submodel: Submodel
) -> torch.Tensor | None:
    if submodel == "body":
        return None
    if submodel == "head":
        return get_face_segmentation_mask(
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
    submodel_split, side = submodel.split(".")
    if submodel_split == "hand":
        if side not in ["L", "R"]:
            raise ValueError(f"Unknown side: {side}")
        return get_face_segmentation_mask(
            model_transforms.edit_mesh(base_data), [f"hand.{side}"]
        )
    raise ValueError(f"Unknown body part: {submodel}")


def load_blend_shape(filename, vertices_count, world_transformation, dtype):
    blend_shape = torch.zeros((vertices_count, 3), dtype=dtype)
    with gzip.open(filename, "rt") as archive:
        for line in archive.readlines():
            data = line.strip().split()
            # Indexing starting at 0
            id = int(data[0])
            assert id >= 0 and id < vertices_count
            offset = [float(x) for x in data[1:]]
            assert len(offset) == 3
            blend_shape[id, :] = torch.as_tensor(offset, dtype=dtype)
    # Blend shapes were expressed in decimeters
    return world_transformation.apply(blend_shape)


def load_macrodetails(template_vertices, world_transformation, dtype):
    root_dirname = get_anny_root_dir()
    vertices_count = len(template_vertices)
    macrodetails_components = PHENOTYPE_VARIATIONS

    # Newborn blend shapes are created as a scaled down version of the baby blend shapes
    newborn_blend_shape_scaling = torch.as_tensor(
        [0.922, 0.922, 0.75], dtype=dtype
    )  # Empirical values to scale down the body for newborns
    normalizing_factor = 3.0  # the cumulated weight of newborn blend shapes when the age is set to newborn

    logger.info("Loading macrodetails blend shapes...")

    # Load macrodetails_components
    macrodetails_dir = os.path.join(root_dirname, "data/mpfb2/targets/macrodetails")
    # Universal macrodetails_components
    universal_blend_shapes = dict()
    for gender in macrodetails_components["gender"]:
        for age in macrodetails_components["age"]:
            for muscle in macrodetails_components["muscle"]:
                for weight in macrodetails_components["weight"]:
                    age_to_load = age if age != "newborn" else "baby"
                    filename = os.path.join(
                        macrodetails_dir,
                        f"universal-{gender}-{age_to_load}-{muscle}-{weight}.target.gz",
                    )
                    blend_shape = load_blend_shape(
                        filename,
                        vertices_count=vertices_count,
                        world_transformation=world_transformation,
                        dtype=dtype,
                    )
                    if age == "newborn":
                        blend_shape = (
                            newborn_blend_shape_scaling[None, :] * blend_shape
                            + (
                                (newborn_blend_shape_scaling[None, :] - 1)
                                / normalizing_factor
                            )
                            * template_vertices
                        )
                    universal_blend_shapes[(gender, age, muscle, weight)] = blend_shape

    # 'Race'-based components
    race_blend_shapes = dict()
    for race in macrodetails_components["race"]:
        for gender in macrodetails_components["gender"]:
            for age in macrodetails_components["age"]:
                age_to_load = age if age != "newborn" else "baby"
                filename = os.path.join(
                    macrodetails_dir, f"{race}-{gender}-{age_to_load}.target.gz"
                )
                blend_shape = load_blend_shape(
                    filename,
                    vertices_count=vertices_count,
                    world_transformation=world_transformation,
                    dtype=dtype,
                )
                if age == "newborn":
                    blend_shape = (
                        newborn_blend_shape_scaling[None, :] * blend_shape
                        + (
                            (newborn_blend_shape_scaling[None, :] - 1)
                            / normalizing_factor
                        )
                        * template_vertices
                    )
                race_blend_shapes[(race, gender, age)] = blend_shape

    # Height based components
    height_blend_shape = dict()
    for gender in macrodetails_components["gender"]:
        for age in macrodetails_components["age"]:
            for muscle in macrodetails_components["muscle"]:
                for weight in macrodetails_components["weight"]:
                    for height in macrodetails_components["height"]:
                        age_to_load = age if age != "newborn" else "baby"
                        filename = os.path.join(
                            macrodetails_dir,
                            "height",
                            f"{gender}-{age_to_load}-{muscle}-{weight}-{height}.target.gz",
                        )
                        blend_shape = load_blend_shape(
                            filename,
                            vertices_count=vertices_count,
                            world_transformation=world_transformation,
                            dtype=dtype,
                        )
                        if age == "newborn":
                            blend_shape = (
                                newborn_blend_shape_scaling[None, :] * blend_shape
                                + (
                                    (newborn_blend_shape_scaling[None, :] - 1)
                                    / normalizing_factor
                                )
                                * template_vertices
                            )
                        height_blend_shape[(gender, age, muscle, weight, height)] = (
                            blend_shape
                        )

    # Proportions based components
    proportions_blend_shapes = dict()
    for gender in macrodetails_components["gender"]:
        for age in macrodetails_components["age"]:
            if age not in ["newborn", "baby"]:
                for muscle in macrodetails_components["muscle"]:
                    for weight in macrodetails_components["weight"]:
                        for proportions in macrodetails_components["proportions"]:
                            filename = os.path.join(
                                macrodetails_dir,
                                "proportions",
                                f"{gender}-{age}-{muscle}-{weight}-{proportions}.target.gz",
                            )
                            blend_shape = load_blend_shape(
                                filename,
                                vertices_count=vertices_count,
                                world_transformation=world_transformation,
                                dtype=dtype,
                            )
                            proportions_blend_shapes[
                                (gender, age, muscle, weight, proportions)
                            ] = blend_shape

    # Breast related blend shapes
    breast_macrodetails_dir = os.path.join(root_dirname, "data/mpfb2/targets/breast")
    breast_blend_shapes = dict()
    gender = "female"
    for age in macrodetails_components["age"]:
        for muscle in macrodetails_components["muscle"]:
            for weight in macrodetails_components["weight"]:
                for cupsize in macrodetails_components["cupsize"]:
                    for firmness in macrodetails_components["firmness"]:
                        filename = os.path.join(
                            breast_macrodetails_dir,
                            f"{gender}-{age}-{muscle}-{weight}-{cupsize}-{firmness}.target.gz",
                        )
                        if os.path.exists(filename):
                            assert age not in ["newborn", "baby"]
                            blend_shape = load_blend_shape(
                                filename,
                                vertices_count=vertices_count,
                                world_transformation=world_transformation,
                                dtype=dtype,
                            )
                            breast_blend_shapes[
                                (gender, age, muscle, weight, cupsize, firmness)
                            ] = blend_shape
    return (
        universal_blend_shapes,
        race_blend_shapes,
        height_blend_shape,
        proportions_blend_shapes,
        breast_blend_shapes,
    )


def _get_coordinates_regressor(groups, data):
    """
    Parse some rig data to return a list of vertex indices to average in order to compute a joint location
    """
    if data["strategy"] == "VERTEX":
        return [data["vertex_index"]]
    elif data["strategy"] == "CUBE":
        group = groups[data["cube_name"]]
        # Return cube center
        return group["vertex_unique_indices"]
    elif data["strategy"] == "MEAN":
        return data["vertex_indices"]
    else:
        raise NotImplementedError


@dataclass
class BlendshapeData:
    blendshapes: torch.Tensor
    stacked_phenotype_blend_shapes_mask: torch.Tensor
    local_change_labels: list[str]
    facial_action_labels: list[str]
    blendshape_labels: list[str]


@dataclass
class MeshData:
    template_vertices: torch.Tensor
    texture_coordinates: torch.Tensor
    groups: dict[str, object]
    faces: torch.Tensor
    face_texture_coordinate_indices: torch.Tensor


@dataclass
class RigData:
    bone_labels: list[str]
    bone_parents: list[int]
    template_bone_heads: torch.Tensor
    template_bone_tails: torch.Tensor
    bone_heads_blendshapes: torch.Tensor
    bone_tails_blendshapes: torch.Tensor
    bone_rolls_rotmat: torch.Tensor
    vertex_bone_weights: torch.Tensor
    vertex_bone_indices: torch.Tensor


def load_all_blendshapes(
    template_vertices: torch.Tensor,
    world_transformation,
    dtype: torch.dtype,
) -> BlendshapeData:
    root_dirname = get_anny_root_dir()

    (
        universal_blend_shapes,
        race_blend_shapes,
        height_blend_shapes,
        proportions_blend_shapes,
        breast_blend_shapes,
    ) = load_macrodetails(
        template_vertices=template_vertices,
        world_transformation=world_transformation,
        dtype=dtype,
    )

    # Stack all macrodetails blend shapes together for better vectorization and efficiency at runtime.
    l_macrodetails = []
    for detail_type, values in PHENOTYPE_VARIATIONS.items():
        for z in values:
            l_macrodetails.append(z)
    assert len(set(l_macrodetails)) == len(l_macrodetails), "Non unique keys"

    l_blend_shape = []
    l_mask = []
    # Unique label associated with each blend shape, e.g. to identify corresponding rows across configurations.
    blendshape_labels = []
    for block_name, blend_shapes in [
        ("universal", universal_blend_shapes),
        ("race", race_blend_shapes),
        ("height", height_blend_shapes),
        ("proportions", proportions_blend_shapes),
        ("breast", breast_blend_shapes),
    ]:
        for components, blend_shape in blend_shapes.items():
            l_blend_shape.append(blend_shape)
            blendshape_labels.append(f"{block_name}:{'-'.join(components)}")
            mask = torch.zeros(len(l_macrodetails), dtype=dtype)
            for x in components:
                idx = l_macrodetails.index(x)
                mask[idx] = 1
            l_mask.append(mask)

    local_blend_shapes = []
    local_change_labels = []
    local_blendshape_labels = []
    with open(os.path.join(root_dirname, "data/mpfb2/targets/target.json"), "r") as f:
        targets_metadata = json.load(f)

    for key, metadata in targets_metadata.items():
        if key != "genitals":
            for category in metadata["categories"]:
                for side in ["left", "right", "unsided"]:
                    if "opposites" in category:
                        neg, pos = (
                            category["opposites"][f"negative-{side}"],
                            category["opposites"][f"positive-{side}"],
                        )
                        if len(neg) > 0 and len(pos) > 0:
                            neg_blend_shape = load_blend_shape(
                                os.path.join(
                                    root_dirname,
                                    "data/mpfb2/targets",
                                    key,
                                    neg + ".target.gz",
                                ),
                                vertices_count=len(template_vertices),
                                world_transformation=world_transformation,
                                dtype=dtype,
                            )
                            pos_blend_shape = load_blend_shape(
                                os.path.join(
                                    root_dirname,
                                    "data/mpfb2/targets",
                                    key,
                                    pos + ".target.gz",
                                ),
                                vertices_count=len(template_vertices),
                                world_transformation=world_transformation,
                                dtype=dtype,
                            )
                            local_change_labels.append(pos)
                            local_blend_shapes.append(pos_blend_shape)
                            local_blend_shapes.append(neg_blend_shape)
                            local_blendshape_labels.append(f"local_change:{pos}")
                            local_blendshape_labels.append(f"local_change:{neg}")

    facial_action_labels, facial_action_blend_shape_tensor = (
        load_facial_action_blendshapes(
            vertices_count=len(template_vertices),
            world_transformation=world_transformation,
            dtype=dtype,
        )
    )
    facial_action_blend_shapes = list(facial_action_blend_shape_tensor)

    logger.info(
        f"{len(universal_blend_shapes)=}, {len(race_blend_shapes)=}, "
        f"{len(height_blend_shapes)=}, {len(proportions_blend_shapes)=}, "
        f"{len(breast_blend_shapes)=}, {len(facial_action_blend_shapes)=}, "
        f"{len(local_blend_shapes)=}"
    )

    blendshape_labels = (
        blendshape_labels
        + [f"facial_action:{label}" for label in facial_action_labels]
        + local_blendshape_labels
    )
    blendshapes = torch.stack(
        l_blend_shape + facial_action_blend_shapes + local_blend_shapes
    )
    assert (
        len(set(blendshape_labels)) == len(blendshape_labels) == blendshapes.shape[0]
    ), "Blend shape labels are not unique"

    return BlendshapeData(
        blendshapes=blendshapes,
        stacked_phenotype_blend_shapes_mask=torch.stack(l_mask),
        local_change_labels=local_change_labels,
        facial_action_labels=facial_action_labels,
        blendshape_labels=blendshape_labels,
    )


def load_mesh(
    eyes: bool,
    tongue: bool,
    world_transformation,
    dtype: torch.dtype,
) -> MeshData:
    root_dirname = get_anny_root_dir()
    base_mesh_filename = os.path.join(root_dirname, "data/mpfb2/3dobjs/base.obj")
    template_vertices, texture_coordinates, groups = anny.utils.obj_utils.load_obj_file(
        base_mesh_filename, dtype=dtype
    )
    template_vertices = world_transformation.apply(template_vertices)

    for group in groups.values():
        group["vertex_unique_indices"] = torch.unique(
            group["face_vertex_indices"].flatten()
        )

    face_vertex_indices = groups["body"]["face_vertex_indices"]
    face_texture_coordinate_indices = groups["body"]["face_texture_coordinate_indices"]

    if eyes:
        face_vertex_indices = torch.concatenate(
            [
                face_vertex_indices,
                groups["helper-l-eye"]["face_vertex_indices"],
                groups["helper-r-eye"]["face_vertex_indices"],
            ],
            dim=0,
        )
        face_texture_coordinate_indices = torch.concatenate(
            [
                face_texture_coordinate_indices,
                groups["helper-l-eye"]["face_texture_coordinate_indices"],
                groups["helper-r-eye"]["face_texture_coordinate_indices"],
            ],
            dim=0,
        )
    if tongue:
        face_vertex_indices = torch.concatenate(
            [face_vertex_indices, groups["helper-tongue"]["face_vertex_indices"]], dim=0
        )
        face_texture_coordinate_indices = torch.concatenate(
            [
                face_texture_coordinate_indices,
                groups["helper-tongue"]["face_texture_coordinate_indices"],
            ],
            dim=0,
        )

    return MeshData(
        template_vertices=template_vertices,
        texture_coordinates=cast(torch.Tensor, texture_coordinates),
        groups=cast(dict[str, object], groups),
        faces=cast(torch.Tensor, face_vertex_indices),
        face_texture_coordinate_indices=cast(
            torch.Tensor, face_texture_coordinate_indices
        ),
    )


def load_rig(
    rig_filename: PathLike,
    weights_filename: PathLike,
    groups: dict[str, object],
    template_vertices: torch.Tensor,
    blendshapes: torch.Tensor,
    dtype: torch.dtype,
) -> RigData:
    assert rig_filename is not None
    assert weights_filename is not None

    with open(rig_filename, "r") as f:
        rig_data = json.load(f)

    if "bones" in rig_data.keys():
        rig_data = rig_data["bones"]

    root_joints = [
        node
        for node in rig_data.keys()
        if ("parent" not in rig_data[node].keys() or rig_data[node]["parent"] == "")
    ]
    assert len(root_joints) == 1
    root_joint = root_joints[0]

    with open(weights_filename) as f:
        weights_data = json.load(f)

    bone_tail_offsets = [torch.zeros(3, dtype=dtype) for _ in range(len(rig_data))]

    bone_labels = []
    bone_parents = []

    def parse_recursively(bone_label, parent_id):
        bone_id = len(bone_labels)
        bone_labels.append(bone_label)
        bone_parents.append(parent_id)
        for node in rig_data.keys():
            if (node not in bone_labels) and rig_data[node]["parent"] == bone_label:
                parse_recursively(node, parent_id=bone_id)

    parse_recursively(root_joint, parent_id=-1)
    assert len(bone_labels) == len(rig_data)

    bone_head_regressor_indices = []
    bone_tail_regressor_indices = []
    bone_rolls = []

    for bone_name in bone_labels:
        bone_head_regressor_indices.append(
            torch.as_tensor(
                _get_coordinates_regressor(groups, rig_data[bone_name]["head"]),
                dtype=torch.int64,
            )
        )
        bone_tail_regressor_indices.append(
            torch.as_tensor(
                _get_coordinates_regressor(groups, rig_data[bone_name]["tail"]),
                dtype=torch.int64,
            )
        )
        bone_rolls.append(rig_data[bone_name]["roll"])

    vertices_count = len(template_vertices)
    vertex_bone_indices = [[] for _ in range(vertices_count)]
    vertex_bone_weights = [[] for _ in range(vertices_count)]
    for bone_id, bone_label in enumerate(bone_labels):
        joint_weight_data = sorted(weights_data["weights"].get(bone_label, []))
        for vertex_idx, vertex_weight in joint_weight_data:
            vertex_bone_indices[vertex_idx].append(bone_id)
            vertex_bone_weights[vertex_idx].append(vertex_weight)

    max_bones_per_vertex = max([len(indices) for indices in vertex_bone_indices])
    logger.info(f"{max_bones_per_vertex=}")
    for indices, weights in zip(vertex_bone_indices, vertex_bone_weights):
        while len(indices) < max_bones_per_vertex:
            indices.append(0)
            weights.append(0.0)
    vertex_bone_indices = torch.as_tensor(vertex_bone_indices, dtype=torch.int64)
    vertex_bone_weights = torch.as_tensor(vertex_bone_weights, dtype=dtype)
    vertex_bone_weights /= torch.sum(vertex_bone_weights, dim=-1, keepdim=True)

    bones_count = len(bone_labels)
    template_bone_tails = []
    tails_blend_shapes = []
    template_bone_heads = []
    heads_blend_shapes = []
    for bone_id in range(bones_count):
        template_bone_tails.append(
            torch.mean(template_vertices[bone_tail_regressor_indices[bone_id]], dim=0)
        )
        tails_blend_shapes.append(
            torch.mean(blendshapes[:, bone_tail_regressor_indices[bone_id], :], dim=1)
            + bone_tail_offsets[bone_id]
        )
        template_bone_heads.append(
            torch.mean(template_vertices[bone_head_regressor_indices[bone_id]], dim=0)
        )
        heads_blend_shapes.append(
            torch.mean(blendshapes[:, bone_head_regressor_indices[bone_id], :], dim=1)
        )
    template_bone_heads = torch.stack(template_bone_heads)
    heads_blend_shapes = torch.stack(heads_blend_shapes, dim=1)
    template_bone_tails = torch.stack(template_bone_tails)
    tails_blend_shapes = torch.stack(tails_blend_shapes, dim=1)
    bone_rolls_rotmat = roma.euler_to_rotmat("Y", [torch.tensor([bone_rolls])]).to(
        dtype
    )

    return RigData(
        bone_labels=bone_labels,
        bone_parents=bone_parents,
        template_bone_heads=template_bone_heads,
        template_bone_tails=template_bone_tails,
        bone_heads_blendshapes=heads_blend_shapes,
        bone_tails_blendshapes=tails_blend_shapes,
        bone_rolls_rotmat=bone_rolls_rotmat,
        vertex_bone_weights=vertex_bone_weights,
        vertex_bone_indices=vertex_bone_indices,
    )


def _filter_rig(
    data: ModelData,
    bones_to_remove: set[str] | frozenset[str],
    subtree_root: str | None,
) -> ModelData:
    """
    Edits rigs to remove bones and/or keep only a selected subtree.

    Every original bone is reparented to its nearest retained ancestor, or made a root if no such bone exists.
    All skinning weight from an original bone are assigned to its nearest retained ancestor. Weights
    that converge on the same target and vertex are aggregated.

    When subtree_root is not None, only descendants of subtree_root are kept, and
    this function produces a new root bone named `root`. The new root has the same rest
    transform as the selected `subtree_root`, and vertices influenced by deleted
    or out-of-subtree bones fall back to it.
    """
    if not bones_to_remove and subtree_root is None:
        return data

    source_labels = data.metadata.bone_labels
    source_parents = data.metadata.bone_parents

    candidate_indices = set(range(len(source_labels)))
    if subtree_root is not None:
        if subtree_root not in source_labels:
            raise ValueError(
                f"Selected subtree root {subtree_root!r} is not in the rig."
            )
        subtree_root_index = source_labels.index(subtree_root)
        candidate_indices = set()
        for bone_index, parent_index in enumerate(source_parents):
            if bone_index == subtree_root_index or parent_index in candidate_indices:
                candidate_indices.add(bone_index)
        if not any(
            source_labels[index] not in bones_to_remove for index in candidate_indices
        ):
            raise ValueError(
                f"Rig filtering removed every bone from selected subtree {subtree_root!r}."
            )

    retained_indices = [
        bone_index
        for bone_index, bone_label in enumerate(source_labels)
        if bone_index in candidate_indices and bone_label not in bones_to_remove
    ]
    if not retained_indices:
        selected = f" from selected subtree {subtree_root!r}" if subtree_root else ""
        raise ValueError(f"Rig filtering removed every bone{selected}.")

    retained_set = set(retained_indices)
    index_offset = int(subtree_root is not None)
    output_bone_count = len(retained_indices) + index_offset
    old_to_new = {
        old_index: new_index
        for new_index, old_index in enumerate(retained_indices, start=index_offset)
    }

    def nearest_retained_ancestor(old_index: int) -> int | None:
        parent_index = source_parents[old_index]
        while parent_index >= 0:
            if parent_index in retained_set:
                return parent_index
            parent_index = source_parents[parent_index]
        return None

    if subtree_root is None:
        root_indices = [
            old_index
            for old_index in retained_indices
            if nearest_retained_ancestor(old_index) is None
        ]
        if len(root_indices) != 1:
            root_labels = [source_labels[index] for index in root_indices]
            raise ValueError(f"Rig filtering must produce one root, got {root_labels}.")
        fallback_root_index = old_to_new[root_indices[0]]
    else:
        fallback_root_index = 0

    def retained_target(old_index: int) -> int:
        current_index = old_index
        while current_index >= 0:
            if current_index in retained_set:
                return old_to_new[current_index]
            current_index = source_parents[current_index]
        return fallback_root_index

    target_indices = torch.as_tensor(
        [retained_target(index) for index in range(len(source_labels))],
        dtype=torch.int64,
        device=data.vertex_bone_indices.device,
    )
    remapped_indices = target_indices[data.vertex_bone_indices]
    dense_weights = torch.zeros(
        (len(data.template_vertices), output_bone_count),
        dtype=data.vertex_bone_weights.dtype,
        device=data.vertex_bone_weights.device,
    )
    dense_weights.scatter_add_(1, remapped_indices, data.vertex_bone_weights)

    positive = dense_weights > 0
    max_bones_per_vertex = int(positive.sum(dim=-1).max().item())
    bone_indices = torch.arange(
        output_bone_count, device=data.vertex_bone_indices.device
    ).expand_as(dense_weights)
    packed_indices = (
        torch.where(positive, bone_indices, output_bone_count)
        .sort(dim=-1)
        .values[:, :max_bones_per_vertex]
    )
    valid = packed_indices < output_bone_count
    packed_indices = torch.where(valid, packed_indices, 0)
    packed_weights = torch.where(
        valid,
        dense_weights.gather(1, packed_indices),
        0,
    )

    transform_indices = (
        [subtree_root_index, *retained_indices]
        if subtree_root is not None
        else retained_indices
    )
    retained = torch.as_tensor(transform_indices, dtype=torch.int64)
    template_bone_tails = (
        None if data.template_bone_tails is None else data.template_bone_tails[retained]
    )
    bone_tails_blendshapes = (
        None
        if data.bone_tails_blendshapes is None
        else data.bone_tails_blendshapes[:, retained]
    )
    bone_rolls_rotmat = (
        None if data.bone_rolls_rotmat is None else data.bone_rolls_rotmat[:, retained]
    )
    bone_labels = [source_labels[index] for index in retained_indices]
    bone_parents = []
    for old_index in retained_indices:
        parent = nearest_retained_ancestor(old_index)
        bone_parents.append(
            (
                -1
                if parent is None and subtree_root is None
                else fallback_root_index
                if parent is None
                else old_to_new[parent]
            )
        )
    if subtree_root is not None:
        bone_labels.insert(0, "root")
        bone_parents.insert(0, -1)

    if (
        data.bone_template_orientation_matrices is not None
        or data.bone_orientation_blendshapes is not None
    ):
        raise ValueError("Use _filter_rig only before loading rig cache.")

    return replace(
        data,
        metadata=replace(
            data.metadata,
            bone_labels=bone_labels,
            bone_parents=bone_parents,
        ),
        template_bone_heads=data.template_bone_heads[retained],
        bone_heads_blendshapes=data.bone_heads_blendshapes[:, retained],
        template_bone_tails=template_bone_tails,
        bone_tails_blendshapes=bone_tails_blendshapes,
        bone_rolls_rotmat=bone_rolls_rotmat,
        vertex_bone_indices=packed_indices,
        vertex_bone_weights=packed_weights,
    )


@cache_builder
def load_data(
    weights_filename: PathLike,
    rig_filename: PathLike,
    eyes: bool = False,
    tongue: bool = False,
    remove_zero_weights_bones: bool = False,
) -> ModelData:
    logger.info(
        "Cache not found, loading data from source files and caching it for future use..."
    )
    dtype = torch.float64
    # Consider a world transformation to use a "Z up" coordinate system with meter as unit for consistency with Blender.
    # Do not mess with this, or it will change the bone orientations.
    world_transformation = roma.Linear(
        0.1 * roma.euler_to_rotmat("X", [90], degrees=True, dtype=dtype)
    )[None]

    mesh_data = load_mesh(
        eyes=eyes,
        tongue=tongue,
        world_transformation=world_transformation,
        dtype=dtype,
    )
    blendshape_data = load_all_blendshapes(
        template_vertices=mesh_data.template_vertices,
        world_transformation=world_transformation,
        dtype=mesh_data.template_vertices.dtype,
    )
    rig_data = load_rig(
        rig_filename=rig_filename,
        weights_filename=weights_filename,
        groups=mesh_data.groups,
        template_vertices=mesh_data.template_vertices,
        blendshapes=blendshape_data.blendshapes,
        dtype=mesh_data.template_vertices.dtype,
    )

    data = ModelData(
        metadata=ModelMetadata(
            bone_labels=rig_data.bone_labels,
            bone_parents=rig_data.bone_parents,
            blendshape_labels=blendshape_data.blendshape_labels,
        ),
        template_vertices=mesh_data.template_vertices,
        faces=mesh_data.faces,
        texture_coordinates=mesh_data.texture_coordinates,
        face_texture_coordinate_indices=mesh_data.face_texture_coordinate_indices,
        blendshapes=blendshape_data.blendshapes,
        stacked_phenotype_blend_shapes_mask=blendshape_data.stacked_phenotype_blend_shapes_mask,
        template_bone_heads=rig_data.template_bone_heads,
        bone_heads_blendshapes=rig_data.bone_heads_blendshapes,
        vertex_bone_weights=rig_data.vertex_bone_weights,
        vertex_bone_indices=rig_data.vertex_bone_indices,
        base_mesh_vertex_indices=torch.arange(
            len(mesh_data.template_vertices), dtype=torch.int64
        ),
        template_bone_tails=rig_data.template_bone_tails,
        bone_tails_blendshapes=rig_data.bone_tails_blendshapes,
        bone_rolls_rotmat=rig_data.bone_rolls_rotmat,
    )
    if remove_zero_weights_bones:
        weighted_bones = set(
            data.vertex_bone_indices[data.vertex_bone_weights > 0].tolist()
        )
        zero_weight_bones = {
            label
            for index, label in enumerate(data.metadata.bone_labels)
            if index not in weighted_bones
        }
        data = _filter_rig(data, zero_weight_bones, subtree_root=None)
    return data


def get_edited_mesh_faces(
    faces: torch.Tensor, face_texture_coordinate_indices: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Minor edits of the MakeHuman mesh topology to satisfy nudity criteria of most contexts.
    """
    device = faces.device
    dtype = faces.dtype

    # --- Vertex indices belonging to faces we want to discard
    vertex_indices_to_discard_l = torch.arange(
        1778, 1794, dtype=dtype, device=device
    )  # 1778..1793
    vertex_indices_to_discard_r = torch.arange(
        8450, 8466, dtype=dtype, device=device
    )  # 8450..8465
    vertex_indices_to_discard = torch.cat(
        [vertex_indices_to_discard_l, vertex_indices_to_discard_r], dim=0
    )

    faces_to_keep_mask = ~torch.isin(faces, vertex_indices_to_discard).any(dim=1)
    faces_kept = faces[faces_to_keep_mask]

    face_texture_coordinate_indices_kept = face_texture_coordinate_indices[
        faces_to_keep_mask
    ]

    # Retrieve texture coordinates used by the vertex indices we want to discard
    ignored_face_ids = torch.nonzero(~faces_to_keep_mask, as_tuple=False).squeeze(1)
    vertex_texture_coordinates = dict()
    for face_id in ignored_face_ids:
        for vertex_id, uv_id in zip(
            faces[face_id], face_texture_coordinate_indices[face_id]
        ):
            vid = vertex_id.item()
            uv_id = uv_id.item()
            if vid in vertex_texture_coordinates:
                assert vertex_texture_coordinates[vid] == uv_id, (
                    f"Vertex {vid} has inconsistent texture coordinates {vertex_texture_coordinates[vid]} vs {uv_id}"
                )
            else:
                vertex_texture_coordinates[vid] = uv_id

    # Add new faces to close the holes left by the discarded faces
    f_l = torch.tensor(
        [
            [8437, 8438, 8439, 8440],
            [8436, 8437, 8440, 8441],
            [8435, 8436, 8441, 8442],
            [8434, 8435, 8442, 8443],
            [8449, 8434, 8443, 8444],
            [8448, 8449, 8444, 8445],
            [8447, 8448, 8445, 8446],
        ],
        dtype=dtype,
        device=device,
    )
    t_l = torch.tensor(
        [vertex_texture_coordinates[vid.item()] for vid in f_l.flatten()]
    ).reshape_as(f_l)

    f_r = torch.tensor(
        [
            [1762, 1771, 1770, 1763],
            [1763, 1770, 1769, 1764],
            [1764, 1769, 1768, 1765],
            [1765, 1768, 1767, 1766],
            [1762, 1777, 1772, 1771],
            [1777, 1776, 1773, 1772],
            [1776, 1775, 1774, 1773],
        ],
        dtype=dtype,
        device=device,
    )
    t_r = torch.tensor(
        [vertex_texture_coordinates[vid.item()] for vid in f_r.flatten()]
    ).reshape_as(f_r)

    # Safety check: ensure caps don't reference vertices to discard
    if (
        torch.isin(f_l, vertex_indices_to_discard).any()
        or torch.isin(f_r, vertex_indices_to_discard).any()
    ):
        raise ValueError(
            "Cap faces (f_l/f_r) reference vertices to discard; please fix the indices."
        )

    # Append new quads
    faces_out = torch.cat([faces_kept, f_l, f_r], dim=0)
    face_texture_coordinate_indices_out = torch.cat(
        [face_texture_coordinate_indices_kept, t_l, t_r], dim=0
    )

    return faces_out, face_texture_coordinate_indices_out


def build_anny_model_data(
    rig: RigConfig,
    topology: TopologyConfig,
    local_changes: LocalChanges,
    facial_actions: FacialActions,
) -> ModelData:
    if topology.base_mesh != "makehuman":
        raise ValueError(
            f"build_anny_model_data only supports 'makehuman' base mesh, got {topology.base_mesh}"
        )
    rig_filename, weights_filename = rig.resolve_filenames()
    if rig_filename is None or weights_filename is None:
        raise ValueError(
            "build_model_data requires a resolved MPFB rig with rig and weights filenames."
        )
    data = load_data(
        rig_filename=rig_filename,
        weights_filename=weights_filename,
        eyes=topology.eyes,
        tongue=topology.tongue,
    )
    data = _filter_rig(data, rig.bones_to_remove, rig.subtree_root)

    mask = resolve_blendshape_mask(
        local_changes, facial_actions, data.metadata.blendshape_labels
    )
    data = model_transforms.filter_blendshapes(data, mask)

    faces_to_keep = _faces_to_keep_from_submodel(data, topology.submodel)

    if (
        topology.nudity_edits or faces_to_keep is not None
    ):  # Filter faces is applied on edited mesh
        data = edit_mesh(data)

    if faces_to_keep is not None:
        data = filter_faces(data, faces_to_keep)

    if topology.remove_unattached_vertices:
        data = model_transforms.remove_unattached_vertices(data)

    data = compact_skinning_weights(data)

    if topology.triangulate_faces:
        data = triangulate(data)

    if rig.bone_orientation == "procrustes":
        data = apply_procrustes_orientation(data)
    elif rig.bone_orientation == "cached":
        data = apply_anny_cached_orientation(
            data, root_bone_source_label=rig.subtree_root
        )

    return data
