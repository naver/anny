# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
import json
import logging
import os
import pathlib
from typing import Literal
import warnings
import gzip

import roma
import torch

from anny.models.model_transforms import (
    LocalChanges,
    filter_local_changes,
    edit_mesh,
    filter_faces,
    triangulate,
    compact_skinning_weights,
    set_metadata,
)
import anny.utils.obj_utils
from anny.models.phenotype import PHENOTYPE_VARIATIONS
from anny.models.model_data import ModelData, AnnyModelMetadata, cache_builder
from anny.paths import ANNY_ROOT_DIR, PathLike
import anny.models.model_transforms as model_transforms
from anny.typing import RigPreset, SkinningMethod

logger = logging.getLogger(__name__)

RigPreset = Literal["default", "makehuman", "cmu_mb", "game_engine", "mixamo"]
SkinningMethod = Literal["lbs", "dqs", "warp_lbs"]



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
            blend_shape[id,:] = torch.as_tensor(offset, dtype=dtype)
    # Blend shapes were expressed in decimeters
    return world_transformation.apply(blend_shape)


def load_macrodetails(root_dirname,
                    template_vertices,
                    world_transformation,
                    dtype):
        vertices_count = len(template_vertices)
        macrodetails_components = PHENOTYPE_VARIATIONS

        # Newborn blend shapes are created as a scaled down version of the baby blend shapes
        newborn_blend_shape_scaling = torch.as_tensor([0.922,0.922,0.75], dtype=dtype) # Empirical values to scale down the body for newborns
        normalizing_factor = 3. # the cumulated weight of newborn blend shapes when the age is set to newborn

        logger.info("Loading macrodetails blend shapes...")

        # Load macrodetails_components
        macrodetails_dir=os.path.join(root_dirname, "data/mpfb2/targets/macrodetails")
        # Universal macrodetails_components
        universal_blend_shapes = dict()
        for gender in macrodetails_components["gender"]:
            for age in macrodetails_components["age"]:
                for muscle in macrodetails_components["muscle"]:
                    for weight in macrodetails_components["weight"]:
                        age_to_load = age if age != "newborn" else "baby"
                        filename = os.path.join(macrodetails_dir, f"universal-{gender}-{age_to_load}-{muscle}-{weight}.target.gz")
                        blend_shape = load_blend_shape(filename, vertices_count=vertices_count, world_transformation=world_transformation, dtype=dtype)
                        if age == "newborn":
                            blend_shape = newborn_blend_shape_scaling[None,:] * blend_shape + ((newborn_blend_shape_scaling[None,:] - 1) / normalizing_factor) * template_vertices
                        universal_blend_shapes[(gender, age, muscle, weight)] = blend_shape

        # 'Race'-based components
        race_blend_shapes = dict()
        for race in macrodetails_components["race"]:
            for gender in macrodetails_components["gender"]:
                for age in macrodetails_components["age"]:
                        age_to_load = age if age != "newborn" else "baby"
                        filename = os.path.join(macrodetails_dir, f"{race}-{gender}-{age_to_load}.target.gz")
                        blend_shape = load_blend_shape(filename, vertices_count=vertices_count, world_transformation=world_transformation, dtype=dtype)
                        if age == "newborn":
                            blend_shape = newborn_blend_shape_scaling[None,:] * blend_shape + ((newborn_blend_shape_scaling[None,:] - 1) / normalizing_factor) * template_vertices
                        race_blend_shapes[(race, gender, age)] = blend_shape

        # Height based components
        height_blend_shape = dict()
        for gender in macrodetails_components["gender"]:
            for age in macrodetails_components["age"]:
                for muscle in macrodetails_components["muscle"]:
                    for weight in macrodetails_components["weight"]:
                        for height in macrodetails_components["height"]:
                            age_to_load = age if age != "newborn" else "baby"
                            filename = os.path.join(macrodetails_dir, "height", f"{gender}-{age_to_load}-{muscle}-{weight}-{height}.target.gz")
                            blend_shape = load_blend_shape(filename, vertices_count=vertices_count, world_transformation=world_transformation, dtype=dtype)
                            if age == "newborn":
                                blend_shape = newborn_blend_shape_scaling[None,:] * blend_shape + ((newborn_blend_shape_scaling[None,:] - 1) / normalizing_factor) * template_vertices
                            height_blend_shape[(gender, age, muscle, weight, height)] = blend_shape

        # Proportions based components
        proportions_blend_shapes = dict()
        for gender in macrodetails_components["gender"]:
            for age in macrodetails_components["age"]:
                if age not in ["newborn", "baby"]:
                    for muscle in macrodetails_components["muscle"]:
                        for weight in macrodetails_components["weight"]:
                            for proportions in macrodetails_components["proportions"]:
                                filename = os.path.join(macrodetails_dir, "proportions", f"{gender}-{age}-{muscle}-{weight}-{proportions}.target.gz")
                                blend_shape = load_blend_shape(filename, vertices_count=vertices_count, world_transformation=world_transformation, dtype=dtype)
                                proportions_blend_shapes[(gender, age, muscle, weight, proportions)] = blend_shape

        # Breast related blend shapes
        breast_macrodetails_dir = os.path.join(root_dirname, "data/mpfb2/targets/breast")
        breast_blend_shapes = dict()
        gender = "female"
        for age in macrodetails_components["age"]:
            for muscle in macrodetails_components["muscle"]:
                    for weight in macrodetails_components["weight"]:
                        for cupsize in macrodetails_components["cupsize"]:
                            for firmness in macrodetails_components["firmness"]:
                                filename = os.path.join(breast_macrodetails_dir, f"{gender}-{age}-{muscle}-{weight}-{cupsize}-{firmness}.target.gz")
                                if os.path.exists(filename):
                                    assert age not in ["newborn", "baby"]
                                    blend_shape = load_blend_shape(filename, vertices_count=vertices_count, world_transformation=world_transformation, dtype=dtype)
                                    breast_blend_shapes[(gender, age, muscle, weight, cupsize, firmness)] = blend_shape
        return universal_blend_shapes, race_blend_shapes, height_blend_shape, proportions_blend_shapes, breast_blend_shapes

def _get_coordinates_regressor(groups, data):
    """
    Parse some rig data to return a list of vertex indices to average in order to compute a joint location
    """
    if data['strategy'] == 'VERTEX':
        return [data['vertex_index']]
    elif data['strategy'] == "CUBE":
        group = groups[data['cube_name']]
        # Return cube center
        return group["vertex_unique_indices"]
    elif data['strategy'] == 'MEAN':
        return data['vertex_indices']
    else:
        raise NotImplementedError



def _build_model_data_from_raw(d: dict, bone_labels, bone_parents, local_change_labels) -> ModelData:
    """Assemble a ModelData from the raw tensors computed in load_data."""
    metadata = AnnyModelMetadata(
        bone_parents=bone_parents,
        bone_labels=bone_labels,
        local_change_labels=local_change_labels,
        pose_parameterization="local-bone",
        skinning_method=None,
        all_phenotypes=False,
        extrapolate_phenotypes=False,
        bone_orientation="blender-rootidentity",
    )
    return ModelData(
        metadata=metadata,
        template_vertices=d["template_vertices"],
        faces=d["faces"],
        texture_coordinates=d["texture_coordinates"],
        face_texture_coordinate_indices=d["face_texture_coordinate_indices"],
        blendshapes=d["blendshapes"],
        stacked_phenotype_blend_shapes_mask=d["stacked_phenotype_blend_shapes_mask"],
        template_bone_heads=d["template_bone_heads"],
        bone_heads_blendshapes=d["bone_heads_blendshapes"],
        vertex_bone_weights=d["vertex_bone_weights"],
        vertex_bone_indices=d["vertex_bone_indices"],
        base_mesh_vertex_indices=torch.arange(len(d["template_vertices"]), dtype=torch.int64),
        template_bone_tails=d["template_bone_tails"],
        bone_tails_blendshapes=d["bone_tails_blendshapes"],
        bone_rolls_rotmat=d["bone_rolls_rotmat"],
    )

@cache_builder
def load_data(
            weights_filename: PathLike,
            rig_filename: PathLike,
            eyes: bool = False,
            tongue : bool = False,
            remove_zero_weights_bones : bool = False,
            bones_to_remove: set[str] = set(),
            root_dirname : PathLike = ANNY_ROOT_DIR,
) -> ModelData:
    # Copy so we never mutate a caller-owned set, and so the shared default never accumulates state across calls.
    bones_to_remove = set(bones_to_remove)

    logger.info("Cache not found, loading data from source files and caching it for future use...")
    dtype = torch.float64
    # Consider a world transformation to use a "Z up" coordinate system with meter as unit for consistency with Blender.
    # Do not mess with this, or it will change the bone orientations.
    world_transformation = roma.Linear(0.1 * roma.euler_to_rotmat("X", [90], degrees=True, dtype=dtype))[None]

    # Load the base mesh
    base_mesh_filename = os.path.join(root_dirname, "data/mpfb2/3dobjs/base.obj")
    template_vertices, texture_coordinates, groups = anny.utils.obj_utils.load_obj_file(base_mesh_filename, dtype=dtype)
    template_vertices = world_transformation.apply(template_vertices)
    # For each group, compute vertex unique ids
    for group in groups.values():
        group["vertex_unique_indices"] = torch.unique(group["face_vertex_indices"].flatten())

    # These are quad faces
    face_vertex_indices = groups["body"]["face_vertex_indices"]
    face_texture_coordinate_indices = groups["body"]["face_texture_coordinate_indices"]
    # Get texture coordinates as well

    # Add eyes faces
    if eyes:
        face_vertex_indices = torch.concatenate([face_vertex_indices, groups["helper-l-eye"]["face_vertex_indices"], groups["helper-r-eye"]["face_vertex_indices"]], dim=0)
        face_texture_coordinate_indices = torch.concatenate([face_texture_coordinate_indices, groups["helper-l-eye"]["face_texture_coordinate_indices"], groups["helper-r-eye"]["face_texture_coordinate_indices"]], dim=0)
    if tongue:
        face_vertex_indices = torch.concatenate([face_vertex_indices, groups["helper-tongue"]["face_vertex_indices"]], dim=0)
        face_texture_coordinate_indices = torch.concatenate([face_texture_coordinate_indices, groups["helper-tongue"]["face_texture_coordinate_indices"]], dim=0)

    assert rig_filename is not None
    assert weights_filename is not None


    with open(rig_filename, "r") as f:
        rig_data = json.load(f)

    if "bones" in rig_data.keys():
        rig_data = rig_data["bones"]

    # Look for a bone that has no parent and consider it as the root
    root_joints = [node for node in rig_data.keys() if ('parent' not in rig_data[node].keys() or rig_data[node]['parent'] == "")]
    assert len(root_joints) == 1
    root_joint = root_joints[0]

    # Load a sparse encoding of bones and weights associated to each vertex
    with open(weights_filename) as f:
        weights_data = json.load(f)

    # Offsets are used to define the orientation of the bones.
    bone_tail_offsets = [torch.zeros(3, dtype=dtype) for _ in range(len(rig_data))]

    # Order joints to ensure that parents are indexed before children when processing them sequentially
    bone_labels = []
    bone_parents = []
    def parse_recursively(bone_label, parent_id):
        bone_id = len(bone_labels)
        bone_labels.append(bone_label)
        bone_parents.append(parent_id)
        for node in rig_data.keys():
            if not (node in bone_labels) and rig_data[node]['parent'] == bone_label:
                parse_recursively(node, parent_id=bone_id)
    parse_recursively(root_joint, parent_id=-1)
    assert len(bone_labels) == len(rig_data)

    if remove_zero_weights_bones:
        for bone_label in bone_labels:
            if len(weights_data["weights"][bone_label]) == 0:
                bones_to_remove.add(bone_label)

    # Remove some bones
    for bone_label in bones_to_remove:
        idx = bone_labels.index(bone_label)
        parent_idx = bone_parents[idx]
        # Assign vertices to the parent bone
        weights_data["weights"][bone_labels[parent_idx]].extend(weights_data["weights"][bone_label])
        weights_data["weights"].pop(bone_label)
        # Skip this bone in the kinematic tree
        for i in range(len(bone_parents)):
            if bone_parents[i] == idx:
                bone_parents[i] = parent_idx
            elif bone_parents[i] > idx:
                bone_parents[i] -= 1 # update indices to account for the pop just below
        bone_labels.pop(idx)
        bone_parents.pop(idx)

    # Reset the root joint to the first bone
    root_joints = [node for node in rig_data.keys() if rig_data[node]['parent'] == ""]
    assert len(root_joints) == 1
    root_joint = root_joints[0]

    # Load bone keypoints parameters (head tail and roll parametrization)
    bone_head_regressor_indices = []
    bone_tail_regressor_indices = []
    bone_rolls = []

    for bone_name in bone_labels:
        bone_head_regressor_indices.append(torch.as_tensor(_get_coordinates_regressor(groups, rig_data[bone_name]["head"]), dtype=torch.int64))
        bone_tail_regressor_indices.append(torch.as_tensor(_get_coordinates_regressor(groups, rig_data[bone_name]["tail"]), dtype=torch.int64))
        bone_rolls.append(rig_data[bone_name]["roll"])

    vertices_count = len(template_vertices)
    vertex_bone_indices = [[] for _ in range(vertices_count)]
    vertex_bone_weights = [[] for _ in range(vertices_count)]
    for bone_id, bone_label in enumerate(bone_labels):
        if bone_label not in weights_data["weights"]:
            warnings.warn("Remove joints without associated weights")
            continue
        joint_weight_data = weights_data['weights'][bone_label]
        if len(joint_weight_data) == 0:
            warnings.warn("Remove joints without associated weights")
        else:
            for vertex_idx, vertex_weight in joint_weight_data:
                vertex_bone_indices[vertex_idx].append(bone_id)
                vertex_bone_weights[vertex_idx].append(vertex_weight)
    # Pad the lists to have the same length for each vertex
    max_bones_per_vertex = max([len(indices) for indices in vertex_bone_indices])
    logger.info(f"{max_bones_per_vertex=}")
    for indices, weights in zip(vertex_bone_indices, vertex_bone_weights):
        while len(indices) < max_bones_per_vertex:
            indices.append(0)
            weights.append(0.)
    vertex_bone_indices = torch.as_tensor(vertex_bone_indices, dtype=torch.int64)
    vertex_bone_weights = torch.as_tensor(vertex_bone_weights, dtype=dtype)
    vertex_bone_weights /= torch.sum(vertex_bone_weights, dim=-1, keepdim=True)

    # Load blend shapes
    universal_blend_shapes, race_blend_shapes, height_blend_shapes, proportions_blend_shapes, breast_blend_shapes = load_macrodetails(root_dirname=root_dirname, template_vertices=template_vertices, world_transformation=world_transformation, dtype=template_vertices.dtype)

    # Stack all macrodetails blend shapes together for better vectorization and efficiency at runtime.
    # List of stacked macrodetails keys
    l_macrodetails = []
    for detail_type, values in PHENOTYPE_VARIATIONS.items():
        for z in values:
            l_macrodetails.append(z)
    assert len(set(l_macrodetails)) == len(l_macrodetails), "Non unique keys"
    l_blend_shape = []
    l_mask = []
    for blend_shapes in [universal_blend_shapes,
                        race_blend_shapes,
                        height_blend_shapes,
                        proportions_blend_shapes,
                        breast_blend_shapes]:
        for components, blend_shape in blend_shapes.items():
            l_blend_shape.append(blend_shape)
            mask = torch.zeros(len(l_macrodetails), dtype=dtype)
            for x in components:
                idx = l_macrodetails.index(x)
                mask[idx] = 1
            l_mask.append(mask)

    # Append local changes blend shapes as well
    local_blend_shapes = []
    local_change_labels = []
    # Load local blend shapes
    with open(os.path.join(root_dirname, "data/mpfb2/targets/target.json"), "r") as f:
        targets_metadata = json.load(f)

    for key, metadata in targets_metadata.items():
        if key != "genitals":
            for category in metadata["categories"]:
                for side in ["left", "right", "unsided"]:
                    if "opposites" in category:
                        neg, pos = category["opposites"][f"negative-{side}"], category["opposites"][f"positive-{side}"]
                        if len(neg) > 0 and len(pos) > 0:
                            neg_blend_shape = load_blend_shape(os.path.join(root_dirname, "data/mpfb2/targets", key, neg + ".target.gz"), vertices_count=len(template_vertices), world_transformation=world_transformation, dtype=template_vertices.dtype)
                            pos_blend_shape = load_blend_shape(os.path.join(root_dirname, "data/mpfb2/targets", key, pos + ".target.gz"), vertices_count=len(template_vertices), world_transformation=world_transformation, dtype=template_vertices.dtype)
                            local_change_labels.append(pos)
                            local_blend_shapes.append(pos_blend_shape)
                            local_blend_shapes.append(neg_blend_shape)

    logger.info(f"{len(universal_blend_shapes)=}, {len(race_blend_shapes)=}, {len(height_blend_shapes)=}, {len(proportions_blend_shapes)=}, {len(breast_blend_shapes)=}, {len(local_blend_shapes)=}")
    stacked_phenotype_blend_shapes = torch.stack(l_blend_shape + local_blend_shapes) # [564,19158,3]
    stacked_phenotype_blend_shapes_mask = torch.stack(l_mask) # [564,25]

    bones_count = len(bone_labels)

    # Precompute bones head and tail locations, as well as corresponding blendshapes
    template_bone_tails = []
    tails_blend_shapes = []
    template_bone_heads = []
    heads_blend_shapes = []
    for bone_id in range(bones_count):
        template_bone_tails.append(torch.mean(template_vertices[bone_tail_regressor_indices[bone_id]], dim=0))
        tails_blend_shapes.append(torch.mean(stacked_phenotype_blend_shapes[:,bone_tail_regressor_indices[bone_id],:], dim=1) + bone_tail_offsets[bone_id])
        template_bone_heads.append(torch.mean(template_vertices[bone_head_regressor_indices[bone_id]], dim=0))
        heads_blend_shapes.append(torch.mean(stacked_phenotype_blend_shapes[:,bone_head_regressor_indices[bone_id],:], dim=1))
    template_bone_heads = torch.stack(template_bone_heads)
    heads_blend_shapes = torch.stack(heads_blend_shapes, dim=1)
    template_bone_tails = torch.stack(template_bone_tails)
    tails_blend_shapes = torch.stack(tails_blend_shapes, dim=1)
    bone_rolls_rotmat = roma.euler_to_rotmat('Y', [torch.tensor([bone_rolls])]).to(dtype) # [1,K,3,3]

    data = _build_model_data_from_raw(
        dict(
            template_vertices=template_vertices,
            faces=face_vertex_indices,
            texture_coordinates=texture_coordinates,
            face_texture_coordinate_indices=face_texture_coordinate_indices,
            blendshapes=stacked_phenotype_blend_shapes,
            template_bone_heads=template_bone_heads,
            template_bone_tails=template_bone_tails,
            bone_heads_blendshapes=heads_blend_shapes,
            bone_tails_blendshapes=tails_blend_shapes,
            bone_rolls_rotmat=bone_rolls_rotmat,
            vertex_bone_weights=vertex_bone_weights,
            vertex_bone_indices=vertex_bone_indices,
            stacked_phenotype_blend_shapes_mask=stacked_phenotype_blend_shapes_mask,
        ),
        bone_labels=bone_labels,
        bone_parents=bone_parents,
        local_change_labels=local_change_labels,
    )
    return data

def get_edited_mesh_faces(faces: torch.Tensor, face_texture_coordinate_indices: torch.Tensor) -> torch.Tensor:
    """
    Minor edits of the MakeHuman mesh topology to satisfy nudity criteria of most contexts.
    """
    device = faces.device
    dtype = faces.dtype

    # --- Vertex indices belonging to faces we want to discard
    vertex_indices_to_discard_l = torch.arange(1778, 1794, dtype=dtype, device=device)  # 1778..1793
    vertex_indices_to_discard_r = torch.arange(8450, 8466, dtype=dtype, device=device)  # 8450..8465
    vertex_indices_to_discard = torch.cat([vertex_indices_to_discard_l, vertex_indices_to_discard_r], dim=0)

    faces_to_keep_mask = ~torch.isin(faces, vertex_indices_to_discard).any(dim=1)
    faces_kept = faces[faces_to_keep_mask]

    face_texture_coordinate_indices_kept = face_texture_coordinate_indices[faces_to_keep_mask]

    # Retrieve texture coordinates used by the vertex indices we want to discard
    ignored_face_ids = torch.nonzero(~faces_to_keep_mask, as_tuple=False).squeeze(1)
    vertex_texture_coordinates = dict()
    for face_id in ignored_face_ids:
        for vertex_id, uv_id in zip(faces[face_id], face_texture_coordinate_indices[face_id]):
            vid = vertex_id.item()
            uv_id = uv_id.item()
            if vid in vertex_texture_coordinates:
                assert vertex_texture_coordinates[vid] == uv_id, f"Vertex {vid} has inconsistent texture coordinates {vertex_texture_coordinates[vid]} vs {uv_id}"
            else:
                vertex_texture_coordinates[vid] = uv_id

    # Add new faces to close the holes left by the discarded faces
    f_l = torch.tensor([
        [8437, 8438, 8439, 8440],
        [8436, 8437, 8440, 8441],
        [8435, 8436, 8441, 8442],
        [8434, 8435, 8442, 8443],
        [8449, 8434, 8443, 8444],
        [8448, 8449, 8444, 8445],
        [8447, 8448, 8445, 8446],
    ], dtype=dtype, device=device)
    t_l = torch.tensor([vertex_texture_coordinates[vid.item()] for vid in f_l.flatten()]).reshape_as(f_l)

    f_r = torch.tensor([
        [1762, 1771, 1770, 1763],
        [1763, 1770, 1769, 1764],
        [1764, 1769, 1768, 1765],
        [1765, 1768, 1767, 1766],
        [1762, 1777, 1772, 1771],
        [1777, 1776, 1773, 1772],
        [1776, 1775, 1774, 1773],
    ], dtype=dtype, device=device)
    t_r = torch.tensor([vertex_texture_coordinates[vid.item()] for vid in f_r.flatten()]).reshape_as(f_r)

    # Safety check: ensure caps don't reference vertices to discard
    if torch.isin(f_l, vertex_indices_to_discard).any() or torch.isin(f_r, vertex_indices_to_discard).any():
        raise ValueError("Cap faces (f_l/f_r) reference vertices to discard; please fix the indices.")

    # Append new quads
    faces_out = torch.cat([faces_kept, f_l, f_r], dim=0)
    face_texture_coordinate_indices_out = torch.cat([face_texture_coordinate_indices_kept, t_l, t_r], dim=0)

    return faces_out, face_texture_coordinate_indices_out


# Maps a `RigPreset` name to (rig basename, weights basename) under data/mpfb2/rigs/standard/.
# `default` uses the cleaned weights baked by scripts/compute_skinning_weights.py;
# `makehuman` uses the original default MakeHuman weights (no symmetry/island cleanup).
_RIG_PRESET_FILES: dict[str, tuple[str, str]] = {
    "default":     ("rig.default.json",     "weights.default.json"),
    "makehuman":      ("rig.default.json",     "weights.makehuman.json"),
    "cmu_mb":      ("rig.cmu_mb.json",      "weights.cmu_mb.json"),
    "game_engine": ("rig.game_engine.json", "weights.game_engine.json"),
    "mixamo":      ("rig.mixamo.json",      "weights.mixamo.json"),
}


def _filenames_from_rig(rig: RigPreset | PathLike, weights_filename: PathLike | None, root_dirname: PathLike):
    """Resolve a rig preset name (or a custom rig JSON path) to (rig_filename, weights_filename)."""
    standard_dir = os.path.join(root_dirname, "data/mpfb2/rigs/standard")
    if rig in _RIG_PRESET_FILES:
        rig_basename, weights_basename = _RIG_PRESET_FILES[rig]
        rig_filename = os.path.join(standard_dir, rig_basename)
        if weights_filename is None:
            weights_filename = os.path.join(standard_dir, weights_basename)
    else:
        rig_filename = str(rig)

    if not pathlib.Path(rig_filename).exists():
        raise FileNotFoundError(f"Rig file not found: {rig_filename}")
    if weights_filename is None:
        raise ValueError("weights_filename must be provided when using a custom rig path")
    if not pathlib.Path(weights_filename).exists():
        raise FileNotFoundError(f"Weights file not found: {weights_filename}")
    return rig_filename, str(weights_filename)

def build_model_data(rig: RigPreset | PathLike = "default",
                 topology: str = "default",
                 eyes: bool = False,
                 tongue: bool = False,
                 bones_to_remove: set[str] = set(),
                 faces_to_keep: torch.Tensor | None = None,
                 local_changes: LocalChanges = "none",
                 skinning_method: SkinningMethod | None = None,
                 remove_unattached_vertices: bool = False,
                 triangulate_faces: bool = False,
                 all_phenotypes: bool = False,
                 pose_parameterization: str = "local-bone",
                 extrapolate_phenotypes: bool = False,
                 bone_orientation: str = "blender-rootidentity",
                 root_dirname: PathLike = ANNY_ROOT_DIR,
                 weights_filename: PathLike | None = None)-> ModelData:
    if rig == "default_no_toes":
        standard_dir = os.path.join(root_dirname, "data/mpfb2/rigs/standard")
        with open(os.path.join(standard_dir, "rig.default.json"), "r") as f:
            default_rig_data = json.load(f)
        with open(os.path.join(standard_dir, "rig.default_no_toes.json"), "r") as f:
            no_toes_rig_data = json.load(f)
        bones_to_remove = set(bones_to_remove)
        bones_to_remove.update(set(default_rig_data) - set(no_toes_rig_data))
        rig = "default"

    rig_filename, weights_filename = _filenames_from_rig(rig, weights_filename, root_dirname)
    data = load_data(
        rig_filename=rig_filename,
        weights_filename=weights_filename,
        eyes=eyes,
        tongue=tongue,
        bones_to_remove=bones_to_remove,
        root_dirname=root_dirname,
    )

    data = filter_local_changes(data, local_changes)

    if topology == "default":
        data = edit_mesh(data)
    else:
        assert topology == "makehuman", "Invalid topology option"

    if faces_to_keep is not None:
        data = filter_faces(data, faces_to_keep)

    if remove_unattached_vertices:
        data = model_transforms.remove_unattached_vertices(data)

    data = compact_skinning_weights(data)

    if triangulate_faces:
        data = triangulate(data)

    data = set_metadata(
        data,
        skinning_method=skinning_method,
        pose_parameterization=pose_parameterization,
        all_phenotypes=all_phenotypes,
        extrapolate_phenotypes=extrapolate_phenotypes,
        bone_orientation=bone_orientation,
    )
    return data
