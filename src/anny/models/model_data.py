# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
from __future__ import annotations

import dataclasses
import functools
import importlib.metadata
import logging
import hashlib
import json
import inspect
import os
from pathlib import Path
from typing import Callable, Literal

import torch

from anny.paths import get_anny_cache_path, get_anny_root_dir
from anny.typing import (
    PathLike,
    AlternativeTopology,
    BoneOrientation,
    PoseParameterization,
    SkinningMethod,
    Submodel,
    Rig,
    LocalChanges,
)

ANNY_VERSION = importlib.metadata.version("anny")
# Increase this if there are any non-backwards-compatible changes to the data/metadata format
CURRENT_DATA_VERSION = 9

logger = logging.getLogger(__name__)

PHENOTYPE_VARIATIONS = dict(
    race=["african", "asian", "caucasian"],
    gender=["male", "female"],
    age=["newborn", "baby", "child", "young", "old"],
    muscle=["minmuscle", "averagemuscle", "maxmuscle"],
    weight=["minweight", "averageweight", "maxweight"],
    height=["minheight", "maxheight"],
    proportions=["idealproportions", "uncommonproportions"],
    cupsize=["mincup", "averagecup", "maxcup"],
    firmness=["minfirmness", "averagefirmness", "maxfirmness"],
)

PHENOTYPE_LABELS = [
    key for key in PHENOTYPE_VARIATIONS.keys() if key != "race"
] + PHENOTYPE_VARIATIONS["race"]
EXCLUDED_PHENOTYPES = ["cupsize", "firmness"] + PHENOTYPE_VARIATIONS["race"]


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


_RIG_PRESET_FILES: dict[str, tuple[str, str]] = {
    "anny": (
        "rig.default.json",
        "weights.default.json",
    ),  # default weights: see scripts/compute_skinning_weights.py
    "makehuman": ("rig.default.json", "weights.default.json"),
    "cmu_mb": ("rig.cmu_mb.json", "weights.cmu_mb.json"),
    "game_engine": ("rig.game_engine.json", "weights.game_engine.json"),
    "mixamo": ("rig.mixamo.json", "weights.mixamo.json"),
}


@dataclasses.dataclass(frozen=True)
class RigConfig:
    base_rig: Rig | Path
    bone_orientation: BoneOrientation = "cached"
    root_identity_orientation: bool = True
    weights_filename: Path | None = None
    bones_to_remove: frozenset[str] = dataclasses.field(default_factory=frozenset)

    def resolve_filenames(self) -> tuple[str | None, str | None]:
        return _resolve_rig_filenames(self)

    @classmethod
    def from_string(cls, spec: str) -> RigConfig:
        return _parse_rig_spec(spec)


@dataclasses.dataclass(frozen=True)
class TopologyConfig:
    base_mesh: AlternativeTopology | Literal["makehuman"]
    nudity_edits: bool = True
    remove_unattached_vertices: bool = True
    triangulate_faces: bool = True
    eyes: bool = True
    tongue: bool = True
    submodel: Submodel = "body"

    @classmethod
    def from_string(cls, spec: str) -> TopologyConfig:
        return _parse_topology_spec(spec)


@dataclasses.dataclass(frozen=True)
class AnnyModelConfig:
    all_phenotypes: bool
    local_changes: LocalChanges
    extrapolate_phenotypes: bool
    facial_actions: bool

    pose_parameterization: PoseParameterization
    skinning_method: SkinningMethod | None

    rig: RigConfig
    topology: TopologyConfig


@dataclasses.dataclass(frozen=True)
class ModelMetadata:
    bone_labels: list[str]
    bone_parents: list[int]

    local_change_labels: list[str] = dataclasses.field(default_factory=list)
    facial_action_labels: list[str] = dataclasses.field(default_factory=list)
    # Unique label per blend shape, identifying corresponding rows across configurations
    blendshape_labels: list[str] = dataclasses.field(default_factory=list)


@dataclasses.dataclass(frozen=True)
class ModelData:
    """Typed, immutable container for all data needed to construct any RiggedModelWithLinearBlendShapes model."""

    # Always present
    metadata: ModelMetadata

    template_vertices: torch.Tensor
    faces: torch.Tensor
    blendshapes: torch.Tensor
    stacked_phenotype_blend_shapes_mask: torch.Tensor | None
    template_bone_heads: torch.Tensor
    bone_heads_blendshapes: torch.Tensor
    vertex_bone_weights: torch.Tensor
    vertex_bone_indices: torch.Tensor
    base_mesh_vertex_indices: torch.Tensor
    # Optional / topology-dependent
    texture_coordinates: torch.Tensor | None = None
    face_texture_coordinate_indices: torch.Tensor | None = None
    # Tail-based orientation (bone_orientation != "procrustes")
    template_bone_tails: torch.Tensor | None = None
    bone_tails_blendshapes: torch.Tensor | None = None
    bone_rolls_rotmat: torch.Tensor | None = None
    # Procrustes-based orientation (bone_orientation == "procrustes")
    bone_nonzeroweight_mask: torch.Tensor | None = None
    bone_vertex_indices: torch.Tensor | None = None
    bone_vertex_weights: torch.Tensor | None = None
    template_bone_vertices: torch.Tensor | None = None
    reference_bone_orientations: torch.Tensor | None = None
    # Precomputed procrustes-based orientation, taking precedence over the runtime registration above
    bone_template_orientation_matrices: torch.Tensor | None = None
    bone_orientation_blendshapes: torch.Tensor | None = None
    # Optional SOMA-style refinement of the precomputed orientations (child-offset alignment,
    # end joints copying their parent orientation)
    bone_children_indices: torch.Tensor | None = None
    bone_children_mask: torch.Tensor | None = None
    bone_children_local_offsets: torch.Tensor | None = None

    @property
    def device(self) -> torch.device:
        return self.template_vertices.device

    def save_safetensors(self, path: PathLike) -> None:
        """Serialize to a safetensors file.  Non-None tensors are stored as-is; ``metadata``
        is packed as a JSON string in the safetensors header under the key ``"metadata"``."""
        import safetensors.torch

        tensors = {
            f.name: getattr(self, f.name).contiguous()
            for f in dataclasses.fields(self)
            if f.name != "metadata" and getattr(self, f.name) is not None
        }
        meta_json = json.dumps(dataclasses.asdict(self.metadata))
        safetensors.torch.save_file(
            tensors,
            path,
            metadata={
                "metadata": meta_json,
                "metadata_class": self.metadata.__class__.__name__,
                "data_version": str(CURRENT_DATA_VERSION),
                "anny_version": ANNY_VERSION,
            },
        )

    @classmethod
    def load_safetensors(cls, path: PathLike) -> ModelData:
        """Deserialize from a safetensors file previously written by :meth:`save_safetensors`."""
        from safetensors import safe_open

        tensors = {}

        with safe_open(path, framework="pt") as f:
            data_version = f.metadata().get("data_version", 1)
            if int(data_version) != CURRENT_DATA_VERSION:
                raise ValueError(
                    f"Data version mismatch: file {path} has data_version={data_version}, but current code expects data_version={CURRENT_DATA_VERSION}"
                )

            meta_str = f.metadata().get("metadata", "{}")
            for k in f.keys():
                tensors[k] = f.get_tensor(k)

        meta_dict = json.loads(meta_str)
        metadata = ModelMetadata(**meta_dict)

        return cls(**tensors, metadata=metadata)


def _get_builder_metadata(
    f: Callable[..., ModelData], *args, **kwargs
) -> dict[str, str | int | bool]:
    all_kwargs = {}

    def _to_valid(x):
        if dataclasses.is_dataclass(x):
            return {k: _to_valid(v) for k, v in dataclasses.asdict(x).items()}
        if isinstance(x, Path):
            return str(x)
        if isinstance(x, (set, frozenset)):
            return sorted(_to_valid(v) for v in x)
        if isinstance(x, dict):
            return {k: _to_valid(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [_to_valid(v) for v in x]
        return x

    for i, param in enumerate(inspect.signature(f).parameters.values()):
        if i < len(args):
            value = _to_valid(args[i])
        elif kwargs.get(param.name) is not None:
            value = _to_valid(kwargs[param.name])
        elif param.default is not param.empty:
            value = _to_valid(param.default)
        else:
            raise ValueError(
                f"Missing value for parameter {param.name} of builder function {f.__name__}"
            )

        if param.name == "facial_actions" and value == "none":
            continue

        all_kwargs[param.name] = value
    return all_kwargs


def cache_builder(f: Callable[..., ModelData]) -> Callable[..., ModelData]:
    """Decorator to add metadata about the model-building function and its arguments to the resulting ModelData."""

    @functools.wraps(f)
    def wrapper(*args, **kwargs) -> ModelData:
        cache_path = get_anny_cache_path()
        if cache_path is None:
            logger.info(
                "No cache directory specified, building model data without caching..."
            )
            return f(*args, **kwargs)

        metadata = _get_builder_metadata(f, *args, **kwargs)
        hex = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()[
            :32
        ]
        cache_path = (
            Path(cache_path)
            / f"v{CURRENT_DATA_VERSION}"
            / f"{f.__name__}_{hex}.safetensors"
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if cache_path.exists():
            logger.info(f"Loading cached model data from {cache_path}")
            data = ModelData.load_safetensors(cache_path)
        else:
            logger.info(
                f"No cached model data found at {cache_path}, building model data..."
            )
            data = f(*args, **kwargs)
            data.save_safetensors(cache_path)
            logger.info(f"Saved built model data to cache at {cache_path}")
        return data

    return wrapper


def _standard_rig_dir(root_dirname: PathLike) -> str:
    return os.path.join(root_dirname, "data/mpfb2/rigs/standard")


def _preset_filenames(preset: str, root_dirname: PathLike) -> tuple[str, str]:
    rig_basename, weights_basename = _RIG_PRESET_FILES[preset]
    standard_dir = _standard_rig_dir(root_dirname)
    return (
        os.path.join(standard_dir, rig_basename),
        os.path.join(standard_dir, weights_basename),
    )


def _bones_to_remove_from_modifier(modifier: str) -> set[str]:
    if modifier == "pruned":
        return set(_zero_weight_bone_labels)
    if modifier == "noeyes":
        return set(_eye_bone_labels)
    if modifier == "notongue":
        return set(_tongue_bone_labels)
    if modifier == "nofacialexpression":
        return set(_facial_expression_bone_labels)
    if modifier == "noexpression":
        return (
            set(_facial_expression_bone_labels)
            | set(_eye_bone_labels)
            | set(_tongue_bone_labels)
        )
    if modifier == "notoes":
        return set(_toe_bone_labels)
    if modifier == "nohands":
        return set(_hand_bone_labels)
    if modifier == "nobreasts":
        return set(_breast_bone_labels)
    raise ValueError(f"Unknown rig specifier: {modifier}")


def _bones_to_remove_from_modifiers(modifiers: list[str]) -> set[str]:
    bones_to_remove: set[str] = set()
    for modifier in modifiers:
        bones_to_remove.update(_bones_to_remove_from_modifier(modifier))
    return bones_to_remove


# Modifiers applied to the bare "anny" rig by default (pruned procrustes preset).
_ANNY_RIG_MODIFIERS = ["notongue", "nobreasts", "nofacialexpression", "pruned"]


def _validate_files(
    rig_filename: PathLike, weights_filename: PathLike
) -> tuple[str, str]:
    rig_filename = str(rig_filename)
    weights_filename = str(weights_filename)
    if not Path(rig_filename).exists():
        raise FileNotFoundError(f"Rig file not found: {rig_filename}")
    if not Path(weights_filename).exists():
        raise FileNotFoundError(f"Weights file not found: {weights_filename}")
    return rig_filename, weights_filename


def _parse_rig_spec(spec: str) -> RigConfig:
    base_rig = spec.split("-")[0]
    modifiers = spec.split("-")[1:]
    if base_rig == "soma":
        if len(modifiers) > 0:
            raise ValueError(
                f"Invalid rig spec: {spec}. 'soma' rig does not support modifiers."
            )
        return RigConfig(
            base_rig="soma",
            weights_filename=None,
            bone_orientation="cached",
            root_identity_orientation=False,
        )
    if base_rig not in _RIG_PRESET_FILES:
        raise ValueError(f"Invalid rig spec: {spec}. Unknown base rig: {base_rig}")

    rig_spec = RigConfig(
        base_rig=base_rig,
        bone_orientation="cached" if base_rig == "anny" else "blender",
        root_identity_orientation=True,
    )

    if "procrustes" in modifiers:
        assert "blender" not in modifiers
        rig_spec = dataclasses.replace(rig_spec, bone_orientation="procrustes")
        modifiers.remove("procrustes")

    if "blender" in modifiers:
        assert "procrustes" not in modifiers
        rig_spec = dataclasses.replace(
            rig_spec, bone_orientation="blender", root_identity_orientation=False
        )
        modifiers.remove("blender")

    if "rootidentity" in modifiers:
        rig_spec = dataclasses.replace(rig_spec, root_identity_orientation=True)
        modifiers.remove("rootidentity")

    if base_rig in ["anny", "makehuman"]:
        modifiers += _ANNY_RIG_MODIFIERS if base_rig == "anny" else []
        rig_spec = dataclasses.replace(
            rig_spec,
            bones_to_remove=frozenset(_bones_to_remove_from_modifiers(modifiers)),
        )
    return rig_spec


def _parse_topology_spec(spec: str) -> TopologyConfig:
    parts = spec.split("-")
    spec_base = parts[0]
    modifiers = parts[1:]
    if spec_base in AlternativeTopology.__args__:
        obj = TopologyConfig(
            base_mesh=spec_base,
            submodel="body",
            nudity_edits=False,
            eyes=True,
            tongue=True,
            remove_unattached_vertices=True,
            triangulate_faces=True,
        )
        if len(modifiers) > 0:
            if len(modifiers) > 1 or modifiers[0] != "quads":
                raise ValueError(f"Unknown topology specifier: {spec}")
            if spec_base in ["smplx", "smpl", "soma"]:
                raise ValueError(
                    f"Topology specifier '{spec_base}' does not support 'quads' modifier."
                )
            obj = dataclasses.replace(obj, triangulate_faces=False)
        return obj
    if spec_base == "head":
        obj = TopologyConfig(
            base_mesh="makehuman",
            submodel="head",
            nudity_edits=False,
            eyes=True,
            tongue=True,
            remove_unattached_vertices=True,
            triangulate_faces=True,
        )
    elif spec_base == "hand.L":
        obj = TopologyConfig(
            base_mesh="makehuman",
            submodel="hand.L",
            nudity_edits=False,
            eyes=False,
            tongue=False,
            remove_unattached_vertices=True,
            triangulate_faces=True,
        )
    elif spec_base == "hand.R":
        obj = TopologyConfig(
            base_mesh="makehuman",
            submodel="hand.R",
            nudity_edits=False,
            eyes=False,
            tongue=False,
            remove_unattached_vertices=True,
            triangulate_faces=True,
        )
    elif spec_base == "anny":
        obj = TopologyConfig(
            base_mesh="makehuman",
            submodel="body",
            nudity_edits=True,
            eyes=True,
            tongue=True,
            remove_unattached_vertices=True,
            triangulate_faces=True,
        )
    elif spec_base == "makehuman":
        obj = TopologyConfig(
            base_mesh="makehuman",
            submodel="body",
            nudity_edits=False,
            eyes=False,
            tongue=False,
            remove_unattached_vertices=False,
            triangulate_faces=False,
        )
    elif spec_base == "default":
        raise ValueError(
            "Topology specifier 'default' is only valid in legacy create_fullbody_model, use 'anny' or 'makehuman' instead."
        )
    else:
        raise ValueError(f"Unknown Anny topology: {spec_base}")

    for modifier in modifiers:
        if spec_base in ["anny", "makehuman", "head", "hand.R", "hand.L"]:
            if modifier == "noeyes":
                obj = dataclasses.replace(obj, eyes=False)
                continue
            elif modifier == "notongue":
                obj = dataclasses.replace(obj, tongue=False)
                continue
            elif modifier == "quads":
                obj = dataclasses.replace(obj, triangulate_faces=False)
                continue
        if spec_base == "anny":
            if modifier == "full":
                obj = dataclasses.replace(
                    obj, remove_unattached_vertices=False, nudity_edits=False
                )
                continue
        if spec_base == "makehuman":
            if modifier == "sfw":
                obj = dataclasses.replace(obj, nudity_edits=True)
                continue

        raise ValueError(f"Unknown topology specifier: {modifier} for {spec_base}")

    return obj


def _resolve_rig_filenames(
    spec: RigConfig,
) -> tuple[str | None, str | None]:
    root_dirname = get_anny_root_dir()
    if isinstance(spec.base_rig, str) and spec.base_rig in _RIG_PRESET_FILES:
        rig_filename, preset_weights_filename = _preset_filenames(
            spec.base_rig, root_dirname
        )
        weights_filename = spec.weights_filename or preset_weights_filename
        return _validate_files(rig_filename, weights_filename)

    if spec.weights_filename is None:
        raise ValueError(
            "weights_filename must be provided when using a custom rig path"
        )
    return _validate_files(spec.base_rig, spec.weights_filename)


def with_bone_orientation(
    rig: RigConfig, bone_orientation: BoneOrientation
) -> RigConfig:
    return dataclasses.replace(rig, bone_orientation=bone_orientation)


def resolve_local_change_mask(
    local_changes: LocalChanges, local_changes_labels: list[str]
) -> list[bool]:
    if local_changes == "none":
        local_changes_mask = [False] * len(local_changes_labels)
    elif local_changes == "default":
        local_changes_mask = [
            "nipple" not in label.lower() for label in local_changes_labels
        ]
    elif local_changes == "all":
        local_changes_mask = [True] * len(local_changes_labels)
    elif isinstance(local_changes, str):
        raise ValueError(
            f"Unknown local_changes preset {local_changes!r}. "
            "Expected 'none', 'default', 'all', or a sequence of label strings."
        )
    else:
        label_to_idx = {label: i for i, label in enumerate(local_changes_labels)}
        local_changes_mask = [False] * len(local_changes_labels)
        for label in local_changes:
            local_changes_mask[label_to_idx[label]] = True
    return local_changes_mask


def resolve_phenotypes(all_phenotypes: bool) -> list[str]:
    if all_phenotypes:
        return PHENOTYPE_LABELS
    else:
        return [x for x in PHENOTYPE_LABELS if x not in EXCLUDED_PHENOTYPES]
