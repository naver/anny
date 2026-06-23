# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
from __future__ import annotations

import dataclasses
import logging
import hashlib
import json
import inspect
from pathlib import Path
from typing import Callable
import importlib.metadata

import torch

from anny.paths import PathLike, get_anny_cache_path
from anny.typing import BoneOrientation, PoseParameterization, SkinningMethod

ANNY_VERSION = importlib.metadata.version("anny")
# Increase this if there are any non-backwards-compatible changes to the data/metadata format
CURRENT_DATA_VERSION = 4

logger = logging.getLogger(__name__)

@dataclasses.dataclass(frozen=True)
class ModelMetadata:
    """Non-tensor configuration for a RiggedModelWithLinearBlendShapes."""
    bone_parents: list[int]
    bone_labels: list[str]
    pose_parameterization: PoseParameterization
    skinning_method: SkinningMethod | None
    bone_orientation: BoneOrientation

@dataclasses.dataclass(frozen=True)
class AnnyModelMetadata(ModelMetadata):
    """Non-tensor configuration for an Anny model"""
    local_change_labels: list
    all_phenotypes: bool
    extrapolate_phenotypes: bool


@dataclasses.dataclass(frozen=True)
class ModelData:
    """Typed, immutable container for all data needed to construct any RiggedModelWithLinearBlendShapes model.

    Tensor fields are stored directly; non-tensor configuration lives in ``metadata``.
    Use :meth:`save_safetensors` / :meth:`load_safetensors` for portable serialization 
    """
    metadata: ModelMetadata
    # Always present
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
        safetensors.torch.save_file(tensors, path,
            metadata={
                "metadata": meta_json,
                "metadata_class": self.metadata.__class__.__name__,
                "data_version": str(CURRENT_DATA_VERSION),
                "anny_version": ANNY_VERSION}
        )

    @classmethod
    def load_safetensors(cls, path: PathLike) -> ModelData:
        """Deserialize from a safetensors file previously written by :meth:`save_safetensors`."""
        from safetensors import safe_open
        tensors = {}
        
        with safe_open(path, framework="pt") as f:
            data_version = f.metadata().get("data_version", 1)
            if int(data_version) != CURRENT_DATA_VERSION:
                raise ValueError(f"Data version mismatch: file {path} has data_version={data_version}, but current code expects data_version={CURRENT_DATA_VERSION}")

            metadata_class = f.metadata().get("metadata_class", "AnnyModelMetadata")
            if metadata_class not in ["ModelMetadata", "AnnyModelMetadata"]:
                raise ValueError(f"Unknown metadata class {metadata_class} in safetensors file {path}")
                
            meta_str = f.metadata().get("metadata", "{}")
            for k in f.keys():
                tensors[k] = f.get_tensor(k)
            
        meta_dict = json.loads(meta_str)
        metadata = ModelMetadata(**meta_dict) if metadata_class == "ModelMetadata" else AnnyModelMetadata(**meta_dict)
        
        return cls(metadata=metadata, **tensors)

def _get_builder_metadata(f: Callable[..., ModelData], *args, **kwargs) -> dict[str, str | int | bool]:
    all_kwargs = {}
    def _to_valid(x):
        if isinstance(x, Path):
            return str(x)
        if isinstance(x, (set, frozenset)):
            return sorted(x)
        return x
    for i, param in enumerate(inspect.signature(f).parameters.values()):
        if i < len(args):
            all_kwargs[param.name] = _to_valid(args[i])
            continue
        if kwargs.get(param.name) is not None:
            all_kwargs[param.name] = _to_valid(kwargs[param.name])
            continue
        if param.default is not param.empty:
            all_kwargs[param.name] = _to_valid(param.default)
            continue
        raise ValueError(f"Missing value for parameter {param.name} of builder function {f.__name__}")
    return all_kwargs


def cache_builder(f: Callable[..., ModelData]) -> Callable[..., ModelData]:
    """Decorator to add metadata about the model-building function and its arguments to the resulting ModelData."""
    def wrapper(*args, **kwargs) -> ModelData:
        cache_path = get_anny_cache_path()
        if cache_path is None:
            logger.info("No cache directory specified, building model data without caching...")
            return f(*args, **kwargs)
            
        metadata = _get_builder_metadata(f, *args, **kwargs)
        hex = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()[:32]
        cache_path = Path(cache_path) / f"v{CURRENT_DATA_VERSION}" / f"{f.__name__}_{hex}.safetensors"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if cache_path.exists():
            logger.info(f"Loading cached model data from {cache_path}")
            data = ModelData.load_safetensors(cache_path)
        else:
            logger.info(f"No cached model data found at {cache_path}, building model data...")
            data = f(*args, **kwargs)
            data.save_safetensors(cache_path)
            logger.info(f"Saved built model data to cache at {cache_path}")
        return data
    return wrapper