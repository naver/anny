# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
from __future__ import annotations
import dataclasses
from typing import TYPE_CHECKING, final

import torch

from anny.torch_compat import make_buffer
from anny.models.rigged_model import (
    PoseParameterization,
    RiggedModelWithLinearBlendShapes,
)

if TYPE_CHECKING:
    from anny.models.model_data import ModelData
    from anny.typing import LocalChanges, SkinningMethod, BoneOrientation
from anny.models.model_data import (
    AnnyModelConfig,
    PHENOTYPE_LABELS,
    PHENOTYPE_VARIATIONS,
    RigConfig,
    TopologyConfig,
    resolve_phenotypes,
)
import anny.utils.interpolation
import anny.utils.relu


class BufferDict(torch.nn.Module):
    def __init__(self, input_dict):
        super().__init__()
        for k, v in input_dict.items():
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
        raise ValueError(
            f"Must be a scalar or a 1-D tensor, got shape {tuple(value.shape)}."
        )
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
        topology_config = (
            TopologyConfig.from_string(topology)
            if isinstance(topology, str)
            else topology
        )
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
            local_changes=local_changes,
            facial_actions=facial_actions,
        )
        self._init_from_model_data(
            data,
            pose_parameterization=self.config.pose_parameterization,
            skinning_method=skinning_method,
            bone_orientation=rig_config.bone_orientation,
            root_identity_orientation=rig_config.root_identity_orientation,
            all_phenotypes=all_phenotypes,
            extrapolate_phenotypes=extrapolate_phenotypes,
        )

    @staticmethod
    def from_model_data(
        data: ModelData,
        skinning_method: SkinningMethod | None = None,
        pose_parameterization: PoseParameterization = "local-bone",
        bone_orientation: BoneOrientation = "blender",
        root_identity_orientation: bool = True,
        all_phenotypes: bool = False,
        extrapolate_phenotypes: bool = False,
    ):
        """Construct an Anny model from a ModelData object."""
        model = Anny.__new__(Anny)
        model.config = None
        model._init_from_model_data(
            data,
            pose_parameterization=pose_parameterization,
            skinning_method=skinning_method,
            bone_orientation=bone_orientation,
            root_identity_orientation=root_identity_orientation,
            all_phenotypes=all_phenotypes,
            extrapolate_phenotypes=extrapolate_phenotypes,
        )
        return model

    def _init_from_model_data(
        self,
        data: ModelData,
        skinning_method: SkinningMethod | None = None,
        pose_parameterization: PoseParameterization = "local-bone",
        bone_orientation: BoneOrientation = "blender",
        root_identity_orientation: bool = True,
        all_phenotypes: bool = False,
        extrapolate_phenotypes: bool = False,
    ):
        super().__init__(
            data,
            pose_parameterization=pose_parameterization,
            skinning_method=skinning_method,
            bone_orientation=bone_orientation,
            root_identity_orientation=root_identity_orientation,
        )
        if data.stacked_phenotype_blend_shapes_mask is None:
            raise ValueError(
                "Model data does not contain stacked_phenotype_blend_shapes_mask, cannot initialize Anny model."
            )
        self._init_phenotype_parameters(
            stacked_phenotype_blend_shapes_mask=data.stacked_phenotype_blend_shapes_mask,
            local_change_labels=data.metadata.local_change_labels,
            facial_action_labels=data.metadata.facial_action_labels,
            base_mesh_vertex_indices=data.base_mesh_vertex_indices,
            extrapolate_phenotypes=extrapolate_phenotypes,
            phenotype_labels=resolve_phenotypes(all_phenotypes=all_phenotypes),
        )

    def _init_phenotype_parameters(
        self,
        stacked_phenotype_blend_shapes_mask: torch.Tensor,
        local_change_labels: list[str],
        facial_action_labels: list[str],
        base_mesh_vertex_indices: torch.Tensor,
        extrapolate_phenotypes: bool,
        phenotype_labels: list[str],
    ):
        self.stacked_phenotype_blend_shapes_mask = make_buffer(
            self,
            "stacked_phenotype_blend_shapes_mask",
            stacked_phenotype_blend_shapes_mask,
            persistent=False,
        )
        self.local_change_labels = local_change_labels
        self.facial_action_labels = facial_action_labels
        self.base_mesh_vertex_indices = base_mesh_vertex_indices
        self.extrapolate_phenotypes = extrapolate_phenotypes
        self.phenotype_labels = phenotype_labels
        self.anchors = BufferDict(self._make_phenotype_anchors())

    def _make_phenotype_anchors(self) -> dict[str, torch.Tensor]:
        anchors = {
            "age": torch.linspace(
                -1 / 3,
                1.0,
                len(PHENOTYPE_VARIATIONS["age"]),
                dtype=self.dtype,
                device=self.device,
            )
        }
        for label in [
            "gender",
            "muscle",
            "weight",
            "height",
            "proportions",
            "cupsize",
            "firmness",
        ]:
            anchors[label] = torch.linspace(
                0.0,
                1.0,
                len(PHENOTYPE_VARIATIONS[label]),
                dtype=self.dtype,
                device=self.device,
            )
        return anchors

    def _parse_parameter_kwargs(
        self,
        kwargs: dict[str, float | torch.Tensor] | torch.Tensor | None,
        labels: list[str],
        default: float,
        name: str,
    ) -> torch.Tensor:
        if kwargs is None:
            return default * torch.ones(
                (1, len(labels)), dtype=self.dtype, device=self.device
            )
        if isinstance(kwargs, dict):
            unknown = set(kwargs) - set(labels)
            if unknown:
                raise ValueError(f"Invalid {name}: {unknown}; available: {labels}")
        if isinstance(kwargs, torch.Tensor):
            if kwargs.dim() != 2 or kwargs.shape[1] != len(labels):
                raise ValueError(
                    f"{name} tensor must have shape [B, {len(labels)}], got {tuple(kwargs.shape)}."
                )
            return kwargs
        values = [
            to_batched_tensor(kwargs.get(label, default), self.device, self.dtype)
            for label in labels
        ]
        batch_size = max((value.shape[0] for value in values), default=1)
        if not values:
            return torch.empty((batch_size, 0), dtype=self.dtype, device=self.device)
        return torch.stack([value.expand(batch_size) for value in values], dim=1)

    def get_phenotype_blendshape_coefficients(
        self,
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
        local_changes: dict[str, float | torch.Tensor] | None = None,
    ):
        """
        Return blendshape coefficients corresponding to the input phenotype description.
        Deprecated but kept for compatibility with SOMA.
        """
        phenotype_parameters = self._parse_parameter_kwargs(
            {
                "gender": gender,
                "age": age,
                "muscle": muscle,
                "weight": weight,
                "height": height,
                "proportions": proportions,
                "cupsize": cupsize,
                "firmness": firmness,
                "african": african,
                "asian": asian,
                "caucasian": caucasian,
            },
            PHENOTYPE_LABELS,
            0.5,
            "phenotype_kwargs",
        )
        local_changes_parameters = self._parse_parameter_kwargs(
            local_changes, self.local_change_labels, 0.0, "local_changes_kwargs"
        )
        return self._get_phenotype_blendshape_coefficients(
            phenotype_parameters,
            local_changes_parameters,
            torch.zeros(
                phenotype_parameters.shape[0],
                len(self.facial_action_labels),
                dtype=phenotype_parameters.dtype,
                device=phenotype_parameters.device,
            ),
        )

    def _get_phenotype_blendshape_coefficients(
        self,
        phenotype_parameters: torch.Tensor,
        local_changes: torch.Tensor,
        facial_actions: torch.Tensor,
    ) -> torch.Tensor:
        phenotype_labels = self.phenotype_labels
        batch_size = max(
            phenotype_parameters.shape[0],
            local_changes.shape[0],
            facial_actions.shape[0],
        )
        phenotype_parameters = phenotype_parameters.expand(batch_size, -1)
        local_changes = local_changes.expand(batch_size, -1)

        def phenotype_value(label: str) -> torch.Tensor:
            if label in phenotype_labels:
                return phenotype_parameters[:, phenotype_labels.index(label)]
            return phenotype_parameters.new_full((batch_size,), 0.5)

        weight_dicts = {}
        for feature in [
            "age",
            "gender",
            "muscle",
            "weight",
            "height",
            "proportions",
            "cupsize",
            "firmness",
        ]:
            interpolation_coeffs = (
                anny.utils.interpolation.linear_interpolation_coefficients(
                    phenotype_value(feature),
                    self.anchors[feature],
                    extrapolate=self.extrapolate_phenotypes,
                )
            )
            weight_dicts[feature] = {
                key: interpolation_coeffs[:, i]
                for i, key in enumerate(PHENOTYPE_VARIATIONS[feature])
            }

        race_values = torch.stack(
            [phenotype_value(key) for key in ("african", "asian", "caucasian")],
            dim=1,
        )
        race_weights = torch.nan_to_num(
            race_values / torch.sum(race_values, dim=1, keepdim=True),
            1 / 3,
            1 / 3,
            1 / 3,
        )

        dict_phens = {
            **weight_dicts["age"],
            **weight_dicts["gender"],
            **weight_dicts["muscle"],
            **weight_dicts["weight"],
            **weight_dicts["height"],
            **weight_dicts["proportions"],
            **weight_dicts["cupsize"],
            **weight_dicts["firmness"],
            "african": race_weights[:, 0],
            "asian": race_weights[:, 1],
            "caucasian": race_weights[:, 2],
        }
        phens = torch.stack(
            [
                dict_phens[key]
                for key_list in PHENOTYPE_VARIATIONS.values()
                for key in key_list
            ],
            dim=1,
        )

        mask = self.stacked_phenotype_blend_shapes_mask.unsqueeze(0)
        wi = torch.prod(phens.unsqueeze(1) * mask + (1 - mask), dim=-1)

        coefficient_groups = [wi]
        if self.facial_action_labels:
            coefficient_groups.append(facial_actions.expand(batch_size, -1))

        if self.local_change_labels:
            local_weights = torch.zeros(
                (batch_size, 2 * len(self.local_change_labels)),
                device=self.device,
                dtype=self.dtype,
            )
            for i in range(len(self.local_change_labels)):
                value = local_changes[:, i]
                local_weights[:, 2 * i] = anny.utils.relu.relu_with_gradient_at_zero(
                    value
                )
                local_weights[:, 2 * i + 1] = (
                    anny.utils.relu.relu_with_gradient_at_zero(-value)
                )
            coefficient_groups.append(local_weights)

        return torch.cat(coefficient_groups, dim=1)

    def parse_phenotype_kwargs(
        self, phenotype_kwargs: dict[str, float | torch.Tensor] | torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """
        For backward compatibility only. Used by SOMA.
        """
        tensor = self._parse_parameter_kwargs(
            phenotype_kwargs, self.phenotype_labels, 0.5, "phenotype_kwargs"
        )
        return {k: tensor[:, i] for (i, k) in enumerate(self.phenotype_labels)}

    def get_tensor_inputs(
        self,
        pose_parameters: dict[str, torch.Tensor]
        | torch.Tensor
        | tuple[torch.Tensor, ...]
        | None,
        phenotype_kwargs: dict[str, float | torch.Tensor] | torch.Tensor | None,
        local_changes_kwargs: dict[str, float | torch.Tensor] | torch.Tensor | None,
        facial_actions: dict[str, float | torch.Tensor] | torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pose_parameters = self.parse_delta_transforms_dict(pose_parameters)
        phenotype_parameters = self._parse_parameter_kwargs(
            phenotype_kwargs,
            self.phenotype_labels,
            0.5,
            "phenotype_kwargs",
        )
        local_change_parameters = self._parse_parameter_kwargs(
            local_changes_kwargs,
            self.local_change_labels,
            0.0,
            "local_changes_kwargs",
        )
        facial_action_parameters = self._parse_parameter_kwargs(
            facial_actions,
            self.facial_action_labels,
            0.0,
            "facial_actions",
        )
        return (
            pose_parameters,
            phenotype_parameters,
            local_change_parameters,
            facial_action_parameters,
        )

    def forward(
        self,
        pose_parameters: torch.Tensor
        | dict[str, torch.Tensor]
        | tuple[torch.Tensor, ...]
        | None = None,
        phenotype_kwargs: dict[str, float | torch.Tensor] | torch.Tensor | None = None,
        local_changes_kwargs: dict[str, float | torch.Tensor]
        | torch.Tensor
        | None = None,
        facial_actions: dict[str, float | torch.Tensor] | torch.Tensor | None = None,
        pose_parameterization: PoseParameterization | None = None,
        return_bone_ends: bool = False,
    ) -> dict[str, torch.Tensor]:
        (
            pose_parameters,
            phenotype_parameters,
            local_change_parameters,
            facial_action_parameters,
        ) = self.get_tensor_inputs(
            pose_parameters,
            phenotype_kwargs,
            local_changes_kwargs,
            facial_actions,
        )
        blendshape_coeffs = self._get_phenotype_blendshape_coefficients(
            phenotype_parameters,
            local_change_parameters,
            facial_action_parameters,
        )
        return super().forward(
            pose_parameters,
            blendshape_coeffs,
            pose_parameterization=pose_parameterization,
            return_bone_ends=return_bone_ends,
        )

    def to_model_data(self) -> "ModelData":
        model_data = super().to_model_data()
        return dataclasses.replace(
            model_data,
            stacked_phenotype_blend_shapes_mask=self.stacked_phenotype_blend_shapes_mask,
        )
