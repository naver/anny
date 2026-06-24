# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
from __future__ import annotations
import dataclasses
from typing import TYPE_CHECKING, Union

import torch

from anny.torch_compat import make_buffer
from anny.models.rigged_model import BoneOrientation, PoseParameterization, RiggedModelWithLinearBlendShapes

if TYPE_CHECKING:
    from anny.models.full_model import RigPreset, SkinningMethod
    from anny.models.model_transforms import LocalChanges
    from anny.models.retopology import Topology
    from anny.models.model_data import ModelData
    from anny.paths import PathLike
from anny.models.model_data import AnnyModelMetadata
import anny.utils.interpolation
import anny.utils.relu


class BufferDict(torch.nn.Module):
    def __init__(self, input_dict):
        super().__init__()
        for k,v in input_dict.items():
            self.register_buffer(k, v)

    def __getitem__(self, key):
        return getattr(self, key)


def to_batched_tensor(value, device, dtype):
    """
    Helper function to accept float inputs
    """
    value = torch.as_tensor(value, device=device, dtype=dtype)
    if value.dim() == 0:
        return value.unsqueeze(dim=0)
    return value

PHENOTYPE_VARIATIONS = dict(
            race=["african", "asian", "caucasian"],
            gender=["male", "female"],
            age=["newborn", "baby", "child", "young", "old"],
            muscle=["minmuscle", "averagemuscle", "maxmuscle"],
            weight=["minweight", "averageweight", "maxweight"],
            height=["minheight", "maxheight"],
            proportions=["idealproportions", "uncommonproportions"],
            cupsize=["mincup", "averagecup", "maxcup"],
            firmness=["minfirmness", "averagefirmness", "maxfirmness"])

PHENOTYPE_LABELS = [key for key in PHENOTYPE_VARIATIONS.keys() if key != "race"] + PHENOTYPE_VARIATIONS["race"]
EXCLUDED_PHENOTYPES = ['cupsize', 'firmness'] + PHENOTYPE_VARIATIONS["race"]


class Anny(RiggedModelWithLinearBlendShapes):
    """Phenotype-aware Anny model and public full-body constructor."""

    def __init__(
        self,
        rig: "RigPreset | PathLike" = "default",
        topology: "Topology" = "default",
        local_changes: "LocalChanges" = "none",
        face_units: bool = False,
        remove_unattached_vertices: bool = True,
        triangulate_faces: bool = False,
        pose_parameterization: PoseParameterization = "local-bone",
        bone_orientation: BoneOrientation = "blender-rootidentity",
        extrapolate_phenotypes: bool = False,
        all_phenotypes: bool = False,
        skinning_method: "SkinningMethod | None" = None,
        weights_filename: "PathLike | None" = None,
    ) -> None:
        from anny.models import build_fullbody_model_data
        if bone_orientation == "procrustes" and rig != "soma":
            # TODO: fix this, procrustes bone orientation should be supported for all rigs, but currently it is only implemented for the soma rig
            raise NotImplementedError(
                "bone_orientation='procrustes' is only supported for rig='soma'."
            )

        data = build_fullbody_model_data(
            rig=rig,
            topology=topology,
            local_changes=local_changes,
            face_units=face_units,
            remove_unattached_vertices=remove_unattached_vertices,
            triangulate_faces=triangulate_faces,
            pose_parameterization=pose_parameterization,
            bone_orientation=bone_orientation,
            extrapolate_phenotypes=extrapolate_phenotypes,
            all_phenotypes=all_phenotypes,
            skinning_method=skinning_method,
            weights_filename=weights_filename,
        )
        self._init_from_model_data(data)
       

    def _init_from_model_data(self, data: "ModelData") -> None:
        if not isinstance(data.metadata, AnnyModelMetadata):
            raise ValueError("ModelData must have metadata to be loaded into Anny model.")
        super()._init_from_model_data(data)
        self._init_phenotype_parameters(
            stacked_phenotype_blend_shapes_mask=data.stacked_phenotype_blend_shapes_mask,
            local_change_labels=data.metadata.local_change_labels,
            face_unit_labels=data.metadata.face_unit_labels,
            base_mesh_vertex_indices=data.base_mesh_vertex_indices,
            extrapolate_phenotypes=data.metadata.extrapolate_phenotypes,
            all_phenotypes=data.metadata.all_phenotypes,
        )

    def _init_phenotype_parameters(self,
                                   stacked_phenotype_blend_shapes_mask,
                                   local_change_labels,
                                   face_unit_labels,
                                   base_mesh_vertex_indices,
                                   extrapolate_phenotypes,
                                   all_phenotypes):
        self.stacked_phenotype_blend_shapes_mask = make_buffer(self, "stacked_phenotype_blend_shapes_mask", stacked_phenotype_blend_shapes_mask, persistent=False)
        self.local_change_labels = local_change_labels
        self.face_unit_labels = face_unit_labels
        self.base_mesh_vertex_indices = base_mesh_vertex_indices
        self.extrapolate_phenotypes = extrapolate_phenotypes
        self.all_phenotypes = all_phenotypes

        self.phenotype_labels = PHENOTYPE_LABELS if self.all_phenotypes else [x for x in PHENOTYPE_LABELS if x not in EXCLUDED_PHENOTYPES]

        self.anchors = BufferDict(self._make_phenotype_anchors())

    def _make_phenotype_anchors(self) -> dict:
        anchors = {'age': torch.linspace(-1/3, 1., len(PHENOTYPE_VARIATIONS['age']), dtype=self.dtype, device=self.device)}
        for label in ['gender', 'muscle', 'weight', 'height', 'proportions', 'cupsize', 'firmness']:
            anchors[label] = torch.linspace(0., 1., len(PHENOTYPE_VARIATIONS[label]), dtype=self.dtype, device=self.device)
        return anchors

    def _as_1d_parameter_tensor(self, value, name: str) -> torch.Tensor:
        tensor = to_batched_tensor(value, self.device, self.dtype)
        if tensor.dim() != 1:
            raise ValueError(f"{name} must be a scalar or a 1-D tensor, got shape {tuple(tensor.shape)}.")
        return tensor

    def _merge_batch_size(self, batch_size: int, candidate: int, name: str) -> int:
        if candidate == 1 or candidate == batch_size:
            return max(batch_size, candidate)
        if batch_size == 1:
            return candidate
        raise ValueError(
            f"Batch size of {name} ({candidate}) must be 1 or match inferred batch size ({batch_size})."
        )

    def _expand_vector_batch(self, value: torch.Tensor, batch_size: int, name: str) -> torch.Tensor:
        if value.shape[0] == batch_size:
            return value
        if value.shape[0] == 1:
            return value.expand(batch_size)
        raise ValueError(
            f"Batch size of {name} ({value.shape[0]}) must be 1 or match inferred batch size ({batch_size})."
        )

    def _expand_matrix_batch(self, value: torch.Tensor, batch_size: int, name: str) -> torch.Tensor:
        if value.shape[0] == batch_size:
            return value
        if value.shape[0] == 1:
            return value.expand(batch_size, -1)
        raise ValueError(
            f"Batch size of {name} ({value.shape[0]}) must be 1 or match inferred batch size ({batch_size})."
        )

    def _validate_face_unit_range(self, values: torch.Tensor) -> None:
        if values.numel() > 0 and not torch.all((values >= 0) & (values <= 1)):
            raise ValueError("Face unit values must be in [0, 1].")

    def _parse_face_units(self, face_units: dict | torch.Tensor | None) -> torch.Tensor:
        face_unit_count = len(self.face_unit_labels)
        if face_unit_count == 0:
            if face_units is None:
                return torch.zeros((1, 0), dtype=self.dtype, device=self.device)
            if isinstance(face_units, dict) and len(face_units) == 0:
                return torch.zeros((1, 0), dtype=self.dtype, device=self.device)
            raise ValueError("face_units were passed, but this model was built with face_units='none'.")

        if face_units is None:
            return torch.zeros((1, face_unit_count), dtype=self.dtype, device=self.device)

        if isinstance(face_units, torch.Tensor):
            values = torch.as_tensor(face_units, dtype=self.dtype, device=self.device)
            if values.dim() != 2 or values.shape[1] != face_unit_count:
                raise ValueError(
                    f"face_units tensor must have shape [B, {face_unit_count}], got {tuple(values.shape)}."
                )
            self._validate_face_unit_range(values)
            return values

        if isinstance(face_units, dict):
            unknown = sorted(set(face_units) - set(self.face_unit_labels))
            if unknown:
                raise ValueError(
                    f"Unknown face unit labels {unknown}; available labels are {self.face_unit_labels}."
                )

            coerced: dict[str, torch.Tensor] = {}
            batch_size = 1
            for label, raw_value in face_units.items():
                value = self._as_1d_parameter_tensor(raw_value, f"face_units[{label!r}]")
                self._validate_face_unit_range(value)
                batch_size = self._merge_batch_size(batch_size, value.shape[0], f"face_units[{label!r}]")
                coerced[label] = value

            values = torch.zeros((batch_size, face_unit_count), dtype=self.dtype, device=self.device)
            for i, label in enumerate(self.face_unit_labels):
                if label in coerced:
                    values[:, i] = self._expand_vector_batch(coerced[label], batch_size, f"face_units[{label!r}]")
            return values

        raise ValueError(
            f"face_units must be None, a dict, or a tensor, got {type(face_units)}."
        )

    def parse_phenotype_kwargs(self, phenotype_kwargs):
        if type(phenotype_kwargs) is torch.Tensor:
            assert phenotype_kwargs.shape[1] == len(self.phenotype_labels), f"phenotype_kwargs tensor must have shape [bs, {len(self.phenotype_labels)}], got {phenotype_kwargs.shape}"
            phenotype_kwargs = {key: phenotype_kwargs[:,i] for i, key in enumerate(self.phenotype_labels)}
        return phenotype_kwargs

    def get_phenotype_blendshape_coefficients(self,
        gender: Union[float, torch.Tensor] = 0.5,
        age: Union[float, torch.Tensor] = 0.5,
        muscle: Union[float, torch.Tensor] = 0.5,
        weight: Union[float, torch.Tensor] = 0.5,
        height: Union[float, torch.Tensor] = 0.5,
        proportions: Union[float, torch.Tensor] = 0.5,
        cupsize: Union[float, torch.Tensor] = 0.5,
        firmness: Union[float, torch.Tensor] = 0.5,
        african: Union[float, torch.Tensor] = 0.5,
        asian: Union[float, torch.Tensor] = 0.5,
        caucasian: Union[float, torch.Tensor] = 0.5,
        local_changes: dict = dict(),
        face_units: dict | torch.Tensor | None = None):
        """Return blendshape coefficients corresponding to the input phenotype description."""
        dtype = self.dtype
        device = self.device
        anchors = self.anchors

        phenotype_inputs = {
            "age": age,
            "gender": gender,
            "muscle": muscle,
            "weight": weight,
            "height": height,
            "proportions": proportions,
            "cupsize": cupsize,
            "firmness": firmness,
            "african": african,
            "asian": asian,
            "caucasian": caucasian,
        }
        phenotype_tensors = {
            key: self._as_1d_parameter_tensor(value, key)
            for key, value in phenotype_inputs.items()
        }
        face_unit_weights = self._parse_face_units(face_units)

        local_change_tensors: dict[str, torch.Tensor] = {}
        batch_size = 1
        for key, value in phenotype_tensors.items():
            batch_size = self._merge_batch_size(batch_size, value.shape[0], key)
        batch_size = self._merge_batch_size(batch_size, face_unit_weights.shape[0], "face_units")
        for key in self.local_change_labels:
            if key in local_changes:
                value = self._as_1d_parameter_tensor(local_changes[key], f"local_changes[{key!r}]")
                batch_size = self._merge_batch_size(batch_size, value.shape[0], f"local_changes[{key!r}]")
                local_change_tensors[key] = value

        weight_dicts = {}
        for feature in ['age', 'gender', 'muscle', 'weight', 'height', 'proportions', 'cupsize', 'firmness']:
            value = self._expand_vector_batch(phenotype_tensors[feature], batch_size, feature)
            interpolation_coeffs = anny.utils.interpolation.linear_interpolation_coefficients(
                value, anchors[feature], extrapolate=self.extrapolate_phenotypes)
            weight_dicts[feature] = {key: interpolation_coeffs[:, i] for i, key in enumerate(PHENOTYPE_VARIATIONS[feature])}

        race_values = torch.stack(
            [
                self._expand_vector_batch(phenotype_tensors[key], batch_size, key)
                for key in ("african", "asian", "caucasian")
            ],
            dim=1,
        )
        race_weights = torch.nan_to_num(race_values / torch.sum(race_values, dim=1, keepdim=True), 1/3, 1/3, 1/3)

        dict_phens = {
            **weight_dicts['age'], **weight_dicts['gender'], **weight_dicts['muscle'],
            **weight_dicts['weight'], **weight_dicts['height'], **weight_dicts['proportions'],
            **weight_dicts['cupsize'], **weight_dicts['firmness'],
            'african': race_weights[:, 0], 'asian': race_weights[:, 1], 'caucasian': race_weights[:, 2],
        }
        phens = torch.stack(
            [dict_phens[key] for key_list in PHENOTYPE_VARIATIONS.values() for key in key_list],
            dim=1,
        )

        masked_phens = phens.unsqueeze(1) * self.stacked_phenotype_blend_shapes_mask.unsqueeze(0)
        wi = torch.prod(masked_phens + (1 - self.stacked_phenotype_blend_shapes_mask.unsqueeze(0)), dim=-1)

        coefficient_groups = [wi]
        if len(self.face_unit_labels) > 0:
            coefficient_groups.append(self._expand_matrix_batch(face_unit_weights, batch_size, "face_units"))

        if len(self.local_change_labels) > 0:
            local_weights = torch.zeros((batch_size, 2 * len(self.local_change_labels)), device=device, dtype=dtype)
            for i, key in enumerate(self.local_change_labels):
                if key in local_change_tensors:
                    value = self._expand_vector_batch(local_change_tensors[key], batch_size, f"local_changes[{key!r}]")
                    local_weights[:, 2*i] = anny.utils.relu.relu_with_gradient_at_zero(value)
                    local_weights[:, 2*i+1] = anny.utils.relu.relu_with_gradient_at_zero(-value)
            coefficient_groups.append(local_weights)

        return torch.cat(coefficient_groups, dim=1)

    def _get_pose_batch_size(self, pose_parameters: torch.Tensor | dict | tuple | None) -> int:
        if pose_parameters is None:
            return 1
        if isinstance(pose_parameters, torch.Tensor):
            return pose_parameters.shape[0]
        if isinstance(pose_parameters, dict):
            return next(iter(pose_parameters.values())).shape[0]
        if isinstance(pose_parameters, tuple):
            return pose_parameters[0].shape[0]
        raise ValueError(f"Invalid pose_parameters type: {type(pose_parameters)}")

    def forward(
        self,
        pose_parameters: torch.Tensor | dict | tuple | None = None,
        phenotype_kwargs: dict | torch.Tensor = dict(),
        local_changes_kwargs: dict = dict(),
        face_units: dict | torch.Tensor | None = None,
        pose_parameterization: PoseParameterization | None = None,
        return_bone_ends: bool = False,
    ) -> dict[str, torch.Tensor]:
        phenotype_kwargs = self.parse_phenotype_kwargs(phenotype_kwargs)
        assert set(phenotype_kwargs) <= set(self.phenotype_labels), f"Invalid phenotype: {set(phenotype_kwargs) - set(self.phenotype_labels)}; available: {self.phenotype_labels}"
        blendshape_coeffs = self.get_phenotype_blendshape_coefficients(
            **phenotype_kwargs,
            local_changes=local_changes_kwargs,
            face_units=face_units,
        )
            
        return super().forward(pose_parameters, blendshape_coeffs, pose_parameterization=pose_parameterization, return_bone_ends=return_bone_ends)

    def to_model_data(self) -> "ModelData":
        model_data = super().to_model_data()
        model_data = dataclasses.replace(model_data, stacked_phenotype_blend_shapes_mask=self.stacked_phenotype_blend_shapes_mask)
        model_data = dataclasses.replace(model_data, metadata=AnnyModelMetadata(
            **dataclasses.asdict(model_data.metadata),
            local_change_labels=self.local_change_labels,
            face_unit_labels=self.face_unit_labels,
            all_phenotypes=self.all_phenotypes,
            extrapolate_phenotypes=self.extrapolate_phenotypes,
        ))
        return model_data
