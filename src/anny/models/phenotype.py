# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
from __future__ import annotations
from typing import TYPE_CHECKING, final

import torch

from anny.torch_compat import make_buffer
from anny.models.rigged_model import PoseParameterization, RiggedModelWithLinearBlendShapes
if TYPE_CHECKING:
    from anny.typing import LocalChanges, SkinningMethod
from anny.models.model_data import AnnyModelConfig, PHENOTYPE_VARIATIONS, RigConfig, TopologyConfig, resolve_phenotypes
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
        value = value.unsqueeze(dim=0)
    if value.dim() != 1:
        raise ValueError(f"Must be a scalar or a 1-D tensor, got shape {tuple(value.shape)}.")
    return value

@final
class Anny(RiggedModelWithLinearBlendShapes):
    """Phenotype-aware Anny model and public full-body constructor."""

    def __init__(
        self,
        rig: str | RigConfig = "anny",
        topology: str | TopologyConfig = "anny",
        local_changes: "LocalChanges" = "none",
        facial_actions: bool = False,
        extrapolate_phenotypes: bool = False,
        all_phenotypes: bool = False,
        pose_parameterization: PoseParameterization = "local-ref",
        skinning_method: SkinningMethod | None = None,
    ) -> None:
        from anny.models import build_model_data
        rig_config = RigConfig.from_string(rig) if isinstance(rig, str) else rig
        topology_config = TopologyConfig.from_string(topology) if isinstance(topology, str) else topology
        self.config = AnnyModelConfig(
            rig=rig_config,
            topology=topology_config,
            local_changes=local_changes,
            facial_actions=facial_actions,
            extrapolate_phenotypes=extrapolate_phenotypes,
            all_phenotypes=all_phenotypes,
            pose_parameterization=pose_parameterization,
            skinning_method=skinning_method,
        )
        data = build_model_data(
            rig=rig_config,
            topology=topology_config,
            local_changes=local_changes
        )

        super().__init__(data,
            pose_parameterization=self.config.pose_parameterization,
            skinning_method=skinning_method,
            bone_orientation=rig_config.bone_orientation,
            root_identity_orientation=rig_config.root_identity_orientation)
        if data.stacked_phenotype_blend_shapes_mask is None:
            raise ValueError("Model data does not contain stacked_phenotype_blend_shapes_mask, cannot initialize Anny model.")
        self._init_phenotype_parameters(
            stacked_phenotype_blend_shapes_mask=data.stacked_phenotype_blend_shapes_mask,
            local_change_labels=data.metadata.local_change_labels,
            facial_action_labels=data.metadata.facial_action_labels,
            base_mesh_vertex_indices=data.base_mesh_vertex_indices,
            extrapolate_phenotypes=extrapolate_phenotypes,
            phenotype_labels=resolve_phenotypes(
                all_phenotypes=self.config.all_phenotypes),
        )

    def _init_phenotype_parameters(self,
                                   stacked_phenotype_blend_shapes_mask: torch.Tensor,
                                   local_change_labels: list[str],
                                   facial_action_labels: list[str],
                                   base_mesh_vertex_indices: torch.Tensor,
                                   extrapolate_phenotypes: bool,
                                   phenotype_labels: list[str]):
        self.stacked_phenotype_blend_shapes_mask = make_buffer(self, "stacked_phenotype_blend_shapes_mask", stacked_phenotype_blend_shapes_mask, persistent=False)
        self.local_change_labels = local_change_labels
        self.facial_action_labels = facial_action_labels
        self.base_mesh_vertex_indices = base_mesh_vertex_indices
        self.extrapolate_phenotypes = extrapolate_phenotypes
        self.phenotype_labels = phenotype_labels
        self.anchors = BufferDict(self._make_phenotype_anchors())

    def _make_phenotype_anchors(self) -> dict[str, torch.Tensor]:
        anchors = {'age': torch.linspace(-1/3, 1., len(PHENOTYPE_VARIATIONS['age']), dtype=self.dtype, device=self.device)}
        for label in ['gender', 'muscle', 'weight', 'height', 'proportions', 'cupsize', 'firmness']:
            anchors[label] = torch.linspace(0., 1., len(PHENOTYPE_VARIATIONS[label]), dtype=self.dtype, device=self.device)
        return anchors

    def _parse_facial_actions(self, facial_actions: dict[str, torch.Tensor] | torch.Tensor | None) -> torch.Tensor:
        facial_action_count = len(self.facial_action_labels)
        if facial_action_count == 0:
            if facial_actions is None:
                return torch.zeros((1, 0), dtype=self.dtype, device=self.device)
            if isinstance(facial_actions, dict) and len(facial_actions) == 0:
                return torch.zeros((1, 0), dtype=self.dtype, device=self.device)
            raise ValueError("facial_actions were passed, but this model was built with facial_actions='none'.")

        if facial_actions is None:
            return torch.zeros((1, facial_action_count), dtype=self.dtype, device=self.device)

        if isinstance(facial_actions, torch.Tensor):
            values = torch.as_tensor(facial_actions, dtype=self.dtype, device=self.device)
            if values.dim() != 2 or values.shape[1] != facial_action_count:
                raise ValueError(
                    f"facial_actions tensor must have shape [B, {facial_action_count}], got {tuple(values.shape)}."
                )
            return values

        if isinstance(facial_actions, dict):
            unknown = sorted(set(facial_actions) - set(self.facial_action_labels))
            if unknown:
                raise ValueError(
                    f"Unknown face unit labels {unknown}; available labels are {self.facial_action_labels}."
                )

            coerced: dict[str, torch.Tensor] = {}
            batch_size = 1
            for label, raw_value in facial_actions.items():
                value = to_batched_tensor(raw_value, self.device, self.dtype)
                batch_size = max(batch_size, value.shape[0])
                coerced[label] = value

            values = torch.zeros((batch_size, facial_action_count), dtype=self.dtype, device=self.device)
            for i, label in enumerate(self.facial_action_labels):
                if label in coerced:
                    values[:, i] = coerced[label].expand(batch_size)
            return values

        raise ValueError(
            f"facial_actions must be None, a dict, or a tensor, got {type(facial_actions)}."
        )

    def parse_phenotype_kwargs(self, phenotype_kwargs: dict[str, torch.Tensor] | torch.Tensor) -> dict[str, torch.Tensor]:
        if isinstance(phenotype_kwargs, torch.Tensor):
            assert phenotype_kwargs.shape[1] == len(self.phenotype_labels), f"phenotype_kwargs tensor must have shape [bs, {len(self.phenotype_labels)}], got {phenotype_kwargs.shape}"
            phenotype_kwargs = {key: phenotype_kwargs[:,i] for i, key in enumerate(self.phenotype_labels)}
        return phenotype_kwargs

    def get_phenotype_blendshape_coefficients(self,
        gender: float | torch.Tensor = 0.5,
        age: float | torch.Tensor = 0.5,
        muscle: float | torch.Tensor = 0.5,
        weight: float | torch.Tensor = 0.5,
        height: float | torch.Tensor = 0.5,
        proportions: float | torch.Tensor = 0.5,
        cupsize: float | torch.Tensor = 0.5,
        firmness: float | torch.Tensor = 0.5,
        african: float | torch.Tensor = 0.5,
        asian: float | torch.Tensor = 0.5,
        caucasian: float | torch.Tensor = 0.5,
        local_changes: dict[str, torch.Tensor] | None = None,
        facial_actions: dict[str, torch.Tensor] | torch.Tensor | None = None):
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
            key: to_batched_tensor(value, self.device, self.dtype)
            for key, value in phenotype_inputs.items()
        }
        facial_action_weights = self._parse_facial_actions(facial_actions)

        local_change_tensors: dict[str, torch.Tensor] = {}
        batch_size = 1
        for key, value in phenotype_tensors.items():
            batch_size = max(batch_size, value.shape[0])
        batch_size = max(batch_size, facial_action_weights.shape[0])
        for key in self.local_change_labels:
            if local_changes is not None and key in local_changes:
                value = to_batched_tensor(local_changes[key], self.device, self.dtype)
                batch_size = max(batch_size, value.shape[0])
                local_change_tensors[key] = value

        weight_dicts = {}
        for feature in ['age', 'gender', 'muscle', 'weight', 'height', 'proportions', 'cupsize', 'firmness']:
            value = phenotype_tensors[feature].expand(batch_size)
            interpolation_coeffs = anny.utils.interpolation.linear_interpolation_coefficients(
                value, anchors[feature], extrapolate=self.extrapolate_phenotypes)
            weight_dicts[feature] = {key: interpolation_coeffs[:, i] for i, key in enumerate(PHENOTYPE_VARIATIONS[feature])}

        race_values = torch.stack(
            [
                phenotype_tensors[key].expand(batch_size)
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
        if len(self.facial_action_labels) > 0:
            coefficient_groups.append(facial_action_weights.expand(batch_size, -1))

        if len(self.local_change_labels) > 0:
            local_weights = torch.zeros((batch_size, 2 * len(self.local_change_labels)), device=device, dtype=dtype)
            for i, key in enumerate(self.local_change_labels):
                if key in local_change_tensors:
                    value = local_change_tensors[key].expand(batch_size)
                    local_weights[:, 2*i] = anny.utils.relu.relu_with_gradient_at_zero(value)
                    local_weights[:, 2*i+1] = anny.utils.relu.relu_with_gradient_at_zero(-value)
            coefficient_groups.append(local_weights)

        return torch.cat(coefficient_groups, dim=1)

    def forward(
        self,
        pose_parameters: torch.Tensor | dict[str, torch.Tensor] | tuple[torch.Tensor, ...] | None = None,
        phenotype_kwargs: dict[str, torch.Tensor] | torch.Tensor  | None = None,
        local_changes_kwargs: dict[str, torch.Tensor] | None = None,
        facial_actions: dict[str, torch.Tensor]  | torch.Tensor | None = None,
        pose_parameterization: PoseParameterization | None = None,
        return_bone_ends: bool = False,
    ) -> dict[str, torch.Tensor]:
        if phenotype_kwargs is None:
            phenotype_kwargs = {}
        if local_changes_kwargs is None:
            local_changes_kwargs = {}
        phenotype_kwargs = self.parse_phenotype_kwargs(phenotype_kwargs)
        assert set(phenotype_kwargs) <= set(self.phenotype_labels), f"Invalid phenotype: {set(phenotype_kwargs) - set(self.phenotype_labels)}; available: {self.phenotype_labels}"
        blendshape_coeffs = self.get_phenotype_blendshape_coefficients(
            **phenotype_kwargs,
            local_changes=local_changes_kwargs,
            facial_actions=facial_actions,
        )

        return super().forward(pose_parameters, blendshape_coeffs, pose_parameterization=pose_parameterization, return_bone_ends=return_bone_ends)
