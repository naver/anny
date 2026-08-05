# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0

import dataclasses
import inspect
from unittest.mock import patch
import warnings

import numpy as np
import roma
import torch

from anny.models.rigged_model import RiggedModelWithLinearBlendShapes
import anny.models.model_transforms
from anny.models.model_data import ModelData, ModelMetadata
from anny.paths import get_anny2smpl_data_path, get_anny2smplx_data_path


with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    # Patching because smplx uses deprecated numpy types and inspect.getargspec, which
    # are removed in newer versions of numpy and Python.
    with (
        patch.object(np, "int", np.int_, create=True),
        patch.object(np, "float", np.float64, create=True),
        patch.object(np, "bool", np.bool_, create=True),
        patch.object(np, "complex", np.complex128, create=True),
        patch.object(np, "object", np.object_, create=True),
        patch.object(np, "str", np.str_, create=True),
        patch.object(np, "unicode", np.str_, create=True),
        patch.object(inspect, "getargspec", inspect.getfullargspec, create=True),
    ):
        import smplx

        # Unused here, but importing chumpy under the patches above is what lets smplx
        # unpickle legacy SMPL .pkl files.
        import chumpy  # noqa: F401


def _synthetic_tail_identity_rolls(
    reference: torch.Tensor, bone_count: int
) -> torch.Tensor:
    # get_bone_poses treats synthetic +Y tails as degenerate and applies this
    # rotation; using it as the roll keeps SMPL joint rest frames at identity.
    roll = reference.new_tensor([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    return roll.unsqueeze(0).unsqueeze(0).expand(1, bone_count, 3, 3).clone()


def _add_bone_tail_blendshapes(data: ModelData) -> ModelData:
    template_bone_tails = data.template_bone_heads + torch.tensor(
        [0.0, 1.0, 0.0],
        dtype=data.template_bone_heads.dtype,
        device=data.template_bone_heads.device,
    )
    bone_tails_blendshapes = data.bone_heads_blendshapes.clone()
    bone_rolls_rotmat = _synthetic_tail_identity_rolls(
        data.template_bone_heads,
        len(data.metadata.bone_labels),
    )
    return dataclasses.replace(
        data,
        template_bone_tails=template_bone_tails,
        bone_tails_blendshapes=bone_tails_blendshapes,
        bone_rolls_rotmat=bone_rolls_rotmat,
    )


class SMPLX(RiggedModelWithLinearBlendShapes):
    def __init__(
        self,
        *smplx_args,
        model_type="smplx",
        pose_corrective=True,
        topology="smplx",
        **smplx_kwargs,
    ):
        # Original model
        model = smplx.create(*smplx_args, model_type="smplx", **smplx_kwargs)

        template_vertices = model.v_template
        base_blendshapes = model.shapedirs.permute(2, 0, 1)

        template_bone_heads = model.J_regressor @ template_vertices

        blendshapes = torch.concatenate(
            (base_blendshapes, model.expr_dirs.permute(2, 0, 1)), dim=0
        )

        if pose_corrective:
            pose_corrective_blendshapes = model.posedirs.reshape(
                (len(model.posedirs), -1, 3)
            )
            blendshapes = torch.concatenate(
                (blendshapes, pose_corrective_blendshapes), dim=0
            )

        masked_blendshapes = blendshapes.clone()
        masked_blendshapes[len(base_blendshapes) :] = 0.0
        bone_heads_blendshapes = torch.einsum(
            "JV,SVD->SJD", model.J_regressor, masked_blendshapes
        )

        bone_count = template_bone_heads.shape[0]
        vertex_bone_weights = model.lbs_weights
        vertex_bone_indices = (
            torch.arange(bone_count)
            .unsqueeze(0)
            .expand(vertex_bone_weights.shape[0], -1)
        )
        bone_labels = [f"bone_{i}" for i in range(bone_count)]

        self.pose_mean = model.pose_mean.reshape(1, -1, 3)
        self.use_pca = model.use_pca
        self.left_hand_components: torch.Tensor | None = None
        self.right_hand_components: torch.Tensor | None = None
        if self.use_pca:
            self.left_hand_components = model.left_hand_components
            self.right_hand_components = model.right_hand_components
        self.pose_corrective = pose_corrective
        metadata = ModelMetadata(
            bone_parents=model.parents,
            bone_labels=bone_labels,
        )
        data = ModelData(
            metadata=metadata,
            template_vertices=model.v_template,
            faces=model.faces_tensor,
            texture_coordinates=None,
            face_texture_coordinate_indices=None,
            blendshapes=blendshapes,
            template_bone_heads=template_bone_heads,
            bone_heads_blendshapes=bone_heads_blendshapes,
            vertex_bone_weights=vertex_bone_weights,
            vertex_bone_indices=vertex_bone_indices,
            reference_bone_orientations=None,
            base_mesh_vertex_indices=torch.arange(
                len(template_vertices), dtype=torch.int64
            ),
            stacked_phenotype_blend_shapes_mask=None,
        )
        if topology == "smplx":
            pass
        else:
            # Load the SMPL-X/Anny correspondences
            anny2smplx_state_dict = torch.load(
                get_anny2smplx_data_path(), map_location="cpu", weights_only=True
            )
            barycentric_coordinates = anny2smplx_state_dict[
                "dst2anny_barycentric_coordinates"
            ]
            reference_vertex_indices = anny2smplx_state_dict["dst2anny_vertex_indices"]
            vertices = (
                barycentric_coordinates[0][:, None]
                * data.template_vertices[reference_vertex_indices[:, 0]]
                + barycentric_coordinates[1][:, None]
                * data.template_vertices[reference_vertex_indices[:, 1]]
                + barycentric_coordinates[2][:, None]
                * data.template_vertices[reference_vertex_indices[:, 2]]
            )
            faces = anny2smplx_state_dict["anny_faces"]
            anny_data = anny.models.model_transforms.apply_retopology(
                data,
                vertices=vertices,
                faces=faces,
                reference_vertex_indices=reference_vertex_indices,
                barycentric_coordinates=barycentric_coordinates,
                check_weights=False,
            )
        if topology == "anny":
            data = anny.models.model_transforms.remove_unattached_vertices(anny_data)
        elif topology == "smpl":
            # Load the SMPL topology
            state_dict = torch.load(
                get_anny2smpl_data_path(), map_location="cpu", weights_only=True
            )
            barycentric_coordinates = state_dict["anny2dst_barycentric_coordinates"]
            reference_vertex_indices = state_dict["anny2dst_vertex_indices"]
            vertices = (
                barycentric_coordinates[0][:, None]
                * anny_data.template_vertices[reference_vertex_indices[:, 0]]
                + barycentric_coordinates[1][:, None]
                * anny_data.template_vertices[reference_vertex_indices[:, 1]]
                + barycentric_coordinates[2][:, None]
                * anny_data.template_vertices[reference_vertex_indices[:, 2]]
            )
            faces = state_dict["dst_faces"]
            data = anny.models.model_transforms.apply_retopology(
                anny_data,
                vertices=vertices,
                faces=faces,
                reference_vertex_indices=reference_vertex_indices,
                barycentric_coordinates=barycentric_coordinates,
            )
            data = anny.models.model_transforms.remove_unattached_vertices(data)
        else:
            assert topology == "smplx"

        bone_labels = [f"bone_{i}" for i in range(bone_count)]

        metadata = dataclasses.replace(
            data.metadata, bone_labels=bone_labels, bone_parents=model.parents.tolist()
        )
        data = dataclasses.replace(data, metadata=metadata)
        data = _add_bone_tail_blendshapes(data)
        super().__init__(
            data=data,
            skinning_method=None,
            pose_parameterization="local-bone-world",
            bone_orientation="blender",
            root_identity_orientation=False,
        )

    def forward(
        self,
        betas,
        expression,
        global_orient,
        transl,
        body_pose,
        leye_pose,
        reye_pose,
        left_hand_pose,
        right_hand_pose,
        jaw_pose,
    ):
        if self.use_pca:
            left_hand_pose = torch.einsum(
                "bi,ij->bj", [left_hand_pose, self.left_hand_components]
            )
            right_hand_pose = torch.einsum(
                "bi,ij->bj", [right_hand_pose, self.right_hand_components]
            )

        rotvec = torch.cat(
            [
                global_orient.reshape(-1, 1, 3),
                body_pose.reshape(-1, 21, 3),
                jaw_pose.reshape(-1, 1, 3),
                leye_pose.reshape(-1, 1, 3),
                reye_pose.reshape(-1, 1, 3),
                left_hand_pose.reshape(-1, 15, 3),
                right_hand_pose.reshape(-1, 15, 3),
            ],
            dim=1,
        )

        batch_size = rotvec.shape[0]
        pose_parameters = (
            torch.eye(4, dtype=rotvec.dtype, device=rotvec.device)
            .unsqueeze(0)
            .expand(batch_size, self.bone_count, 4, 4)
            .clone()
        )
        pose_parameters[:, :, :3, :3] = roma.rotvec_to_rotmat(
            rotvec + self.pose_mean.to(dtype=rotvec.dtype, device=rotvec.device)
        )
        pose_parameters[:, 0, :3, 3] = transl
        blendshape_coeffs = torch.cat((betas, expression), dim=1)

        if self.pose_corrective:
            batch_size = len(pose_parameters)
            identity_rotmat = torch.eye(
                3, dtype=pose_parameters.dtype, device=pose_parameters.device
            )
            pose_corrective_blendshape_coeffs = (
                pose_parameters[:, 1:, :3, :3] - identity_rotmat[None, None]
            ).view(batch_size, -1)
            full_blendshape_coeffs = torch.concatenate(
                (blendshape_coeffs, pose_corrective_blendshape_coeffs), dim=-1
            )
        else:
            full_blendshape_coeffs = blendshape_coeffs

        return super().forward(
            pose_parameters=pose_parameters, blendshape_coeffs=full_blendshape_coeffs
        )


class SMPL(RiggedModelWithLinearBlendShapes):
    def __init__(
        self, *smpl_args, pose_corrective=True, topology="smpl", **smpl_kwargs
    ):
        # Original model
        model = smplx.create(*smpl_args, model_type="smpl", **smpl_kwargs)

        template_vertices = model.v_template
        base_blendshapes = model.shapedirs.permute(2, 0, 1)

        template_bone_heads = model.J_regressor @ template_vertices

        if pose_corrective:
            pose_corrective_blendshapes = model.posedirs.reshape(
                (len(model.posedirs), -1, 3)
            )
            blendshapes = torch.concatenate(
                (base_blendshapes, pose_corrective_blendshapes), dim=0
            )
        else:
            blendshapes = base_blendshapes

        masked_blendshapes = blendshapes.clone()
        masked_blendshapes[len(base_blendshapes) :] = 0.0
        bone_heads_blendshapes = torch.einsum(
            "JV,SVD->SJD", model.J_regressor, masked_blendshapes
        )

        bone_count = template_bone_heads.shape[0]
        vertex_bone_weights = model.lbs_weights
        vertex_bone_indices = (
            torch.arange(bone_count)
            .unsqueeze(0)
            .expand(vertex_bone_weights.shape[0], -1)
        )
        bone_labels = [f"bone_{i}" for i in range(bone_count)]
        self.pose_corrective = pose_corrective
        metadata = ModelMetadata(
            bone_parents=model.parents,
            bone_labels=bone_labels,
        )
        data = ModelData(
            metadata=metadata,
            template_vertices=model.v_template,
            faces=model.faces_tensor,
            texture_coordinates=None,
            face_texture_coordinate_indices=None,
            blendshapes=blendshapes,
            template_bone_heads=template_bone_heads,
            bone_heads_blendshapes=bone_heads_blendshapes,
            vertex_bone_weights=vertex_bone_weights,
            vertex_bone_indices=vertex_bone_indices,
            reference_bone_orientations=None,
            base_mesh_vertex_indices=torch.arange(
                len(template_vertices), dtype=torch.int64
            ),
            stacked_phenotype_blend_shapes_mask=None,
        )
        if topology != "smpl":
            # Load the SMPL/Anny correspondences
            anny_state_dict = torch.load(
                get_anny2smpl_data_path(), map_location="cpu", weights_only=True
            )
            barycentric_coordinates = anny_state_dict[
                "dst2anny_barycentric_coordinates"
            ]
            reference_vertex_indices = anny_state_dict["dst2anny_vertex_indices"]
            vertices = (
                barycentric_coordinates[0][:, None]
                * data.template_vertices[reference_vertex_indices[:, 0]]
                + barycentric_coordinates[1][:, None]
                * data.template_vertices[reference_vertex_indices[:, 1]]
                + barycentric_coordinates[2][:, None]
                * data.template_vertices[reference_vertex_indices[:, 2]]
            )
            faces = anny_state_dict["anny_faces"]
            anny_data = anny.models.model_transforms.apply_retopology(
                data,
                vertices=vertices,
                faces=faces,
                reference_vertex_indices=reference_vertex_indices,
                barycentric_coordinates=barycentric_coordinates,
                check_weights=False,
            )

            if topology == "anny":
                data = anny.models.model_transforms.remove_unattached_vertices(
                    anny_data
                )
            elif topology == "smplx":
                # Load the SMPLX topology
                state_dict = torch.load(
                    get_anny2smplx_data_path(), map_location="cpu", weights_only=True
                )
                barycentric_coordinates = state_dict["anny2dst_barycentric_coordinates"]
                reference_vertex_indices = state_dict["anny2dst_vertex_indices"]
                vertices = (
                    barycentric_coordinates[0][:, None]
                    * anny_data.template_vertices[reference_vertex_indices[:, 0]]
                    + barycentric_coordinates[1][:, None]
                    * anny_data.template_vertices[reference_vertex_indices[:, 1]]
                    + barycentric_coordinates[2][:, None]
                    * anny_data.template_vertices[reference_vertex_indices[:, 2]]
                )
                faces = state_dict["dst_faces"]
                data = anny.models.model_transforms.apply_retopology(
                    anny_data,
                    vertices=vertices,
                    faces=faces,
                    reference_vertex_indices=reference_vertex_indices,
                    barycentric_coordinates=barycentric_coordinates,
                )
                data = anny.models.model_transforms.remove_unattached_vertices(data)
            else:
                raise ValueError()
        data = _add_bone_tail_blendshapes(data)
        super().__init__(
            data=data,
            pose_parameterization="local-bone-world",
            skinning_method=None,
            bone_orientation="blender",
            root_identity_orientation=False,
        )

    def forward(self, betas, global_orient, transl, body_pose):
        rotvec = torch.cat(
            [
                global_orient.reshape(-1, 1, 3),
                body_pose.reshape(-1, self.bone_count - 1, 3),
            ],
            dim=1,
        )

        batch_size = rotvec.shape[0]
        pose_parameters = (
            torch.eye(4, dtype=rotvec.dtype, device=rotvec.device)
            .unsqueeze(0)
            .expand(batch_size, self.bone_count, 4, 4)
            .clone()
        )
        pose_parameters[:, :, :3, :3] = roma.rotvec_to_rotmat(rotvec)
        pose_parameters[:, 0, :3, 3] = transl
        blendshape_coeffs = betas

        if self.pose_corrective:
            batch_size = len(pose_parameters)
            identity_rotmat = torch.eye(
                3, dtype=pose_parameters.dtype, device=pose_parameters.device
            )
            pose_corrective_blendshape_coeffs = (
                pose_parameters[:, 1:, :3, :3] - identity_rotmat[None, None]
            ).view(batch_size, -1)
            full_blendshape_coeffs = torch.concatenate(
                (blendshape_coeffs, pose_corrective_blendshape_coeffs), dim=-1
            )
        else:
            full_blendshape_coeffs = blendshape_coeffs

        return super().forward(
            pose_parameters=pose_parameters, blendshape_coeffs=full_blendshape_coeffs
        )
