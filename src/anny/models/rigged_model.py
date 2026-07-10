# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
import warnings
from typing import TYPE_CHECKING

import torch
import roma

import anny.skinning.skinning as skinning
from anny.torch_compat import make_buffer
from anny.typing import PoseParameterization, BoneOrientation, SkinningMethod
import anny.utils.kinematics as kinematics
from anny.utils.mesh_utils import triangulate_faces

if TYPE_CHECKING:
    from anny.models.model_data import ModelData

def _make_optional_buffer(module: torch.nn.Module, name: str, tensor: torch.Tensor | None) -> torch.Tensor | None:
    if tensor is None:
        return None
    return make_buffer(module, name, tensor, persistent=False)    

class RiggedModelWithLinearBlendShapes(torch.nn.Module):
    def __init__(
        self,
        data: "ModelData",
        skinning_method: SkinningMethod | None = None,
        pose_parameterization: PoseParameterization = "local-bone",
        bone_orientation: BoneOrientation = "blender",
        root_identity_orientation: bool = True,
    ) -> None:
        super().__init__()
        self.template_vertices = make_buffer(self, "template_vertices", data.template_vertices, persistent=False)
        self.faces = data.faces
        self.texture_coordinates = _make_optional_buffer(self, "texture_coordinates", data.texture_coordinates)
        self.face_texture_coordinate_indices = make_buffer(self, "face_texture_coordinate_indices", data.face_texture_coordinate_indices, persistent=False) if data.face_texture_coordinate_indices is not None else None
        self.blendshapes = make_buffer(self, "blendshapes", data.blendshapes, persistent=False)
        self.template_bone_heads = make_buffer(self, "template_bone_heads", data.template_bone_heads, persistent=False)
        self.bone_heads_blendshapes = make_buffer(self, "bone_heads_blendshapes", data.bone_heads_blendshapes, persistent=False)
        self.reference_bone_orientations = _make_optional_buffer(self, "reference_bone_orientations", data.reference_bone_orientations)
       
        self.vertex_bone_weights = make_buffer(self, "vertex_bone_weights", data.vertex_bone_weights, persistent=False)
        self.vertex_bone_indices = make_buffer(self, "vertex_bone_indices", data.vertex_bone_indices, persistent=False)
        self.base_mesh_vertex_indices = make_buffer(self, "base_mesh_vertex_indices", data.base_mesh_vertex_indices, persistent=False)
        self.set_skinning_method(skinning_method)
        self.pose_parameterization: PoseParameterization = pose_parameterization

        self.bone_orientation: BoneOrientation = bone_orientation
        self.root_identity_orientation = root_identity_orientation
        
        self.bone_labels = data.metadata.bone_labels
        self.bone_parents = data.metadata.bone_parents

        self.kinematic_propagation_fronts = kinematics.get_kinematic_propagation_fronts(self.bone_parents)

        
        if self.bone_orientation == "blender":
            assert data.template_bone_tails is not None
            assert data.bone_tails_blendshapes is not None
            assert data.bone_rolls_rotmat is not None
            
        elif data.bone_template_orientation_matrices is not None:
            assert data.bone_orientation_blendshapes is not None
        else:
            assert data.bone_nonzeroweight_mask is not None
            assert data.bone_vertex_indices is not None
            assert data.bone_vertex_weights is not None
            assert data.template_bone_vertices is not None

        self.template_bone_tails = _make_optional_buffer(self, "template_bone_tails", data.template_bone_tails)
        self.bone_tails_blendshapes = _make_optional_buffer(self, "bone_tails_blendshapes", data.bone_tails_blendshapes)
        self.y_axis = make_buffer(self, "y_axis", torch.as_tensor([0.0, 1.0, 0.0], dtype=self.template_vertices.dtype), persistent=False)
        self.degenerate_rotation = make_buffer(
            self,
            "degenerate_rotation",
            torch.tensor([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]], dtype=self.template_vertices.dtype),
            persistent=False,
        )
        self.bone_rolls_rotmat = _make_optional_buffer(self, "bone_rolls_rotmat", data.bone_rolls_rotmat)
        self.bone_nonzeroweight_mask = _make_optional_buffer(self, "bone_nonzeroweight_mask", data.bone_nonzeroweight_mask)
        self.bone_vertex_indices = _make_optional_buffer(self, "bone_vertex_indices", data.bone_vertex_indices)
        self.bone_vertex_weights = _make_optional_buffer(self, "bone_vertex_weights", data.bone_vertex_weights)
        self.template_bone_vertices = _make_optional_buffer(self, "template_bone_vertices", data.template_bone_vertices)
        self.bone_template_orientation_matrices = _make_optional_buffer(self, "bone_template_orientation_matrices", data.bone_template_orientation_matrices)
        self.bone_orientation_blendshapes = _make_optional_buffer(self, "bone_orientation_blendshapes", data.bone_orientation_blendshapes)

    @property
    def bone_count(self) -> int:
        return len(self.bone_labels)

    @property
    def dtype(self) -> torch.dtype:
        return self.template_vertices.dtype

    @property
    def device(self) -> torch.device:
        return self.template_vertices.device

    def get_triangular_faces(self) -> torch.Tensor:
        """
        Return a triangulated version of the faces, splitting quads when needed.
        """
        triangular_faces = torch.tensor(triangulate_faces(vertices=self.template_vertices, faces=self.faces.detach().cpu().numpy().tolist()), device=self.device)
        return triangular_faces

    def set_skinning_method(self, skinning_method: SkinningMethod | None) -> None:
        self._skinning_method_parameter: SkinningMethod | None = skinning_method  # preserve original (None = auto-detect)
        if skinning_method is None:
            # Default skinning settings.
            try:
                import anny.skinning.warp_skinning
                skinning_method = "warp_lbs"
            except ImportError:
                warnings.warn("Fallback to default lbs skinning. Consider installing NVidia Warp for lower memory footprint.")
                skinning_method = "lbs"
        if skinning_method == "lbs":
            self._skinning_method = skinning.linear_blend_skinning
        elif skinning_method == "dqs":
            self._skinning_method = skinning.dual_quaternion_skinning
        elif skinning_method == "warp_lbs":
            import anny.skinning.warp_skinning
            self._skinning_method = anny.skinning.warp_skinning.linear_blend_skinning
        else:
            raise NotImplementedError


    def to_model_data(self) -> "ModelData":
        from anny.models.model_data import ModelData, ModelMetadata
        return ModelData(
            metadata=ModelMetadata(
                bone_parents=self.bone_parents,
                bone_labels=self.bone_labels,
            ),
            template_vertices=self.template_vertices,
            faces=self.faces,
            texture_coordinates=self.texture_coordinates,
            face_texture_coordinate_indices=self.face_texture_coordinate_indices,
            blendshapes=self.blendshapes,
            stacked_phenotype_blend_shapes_mask=None,
            template_bone_heads=self.template_bone_heads,
            bone_heads_blendshapes=self.bone_heads_blendshapes,
            vertex_bone_weights=self.vertex_bone_weights,
            vertex_bone_indices=self.vertex_bone_indices,
            base_mesh_vertex_indices=self.base_mesh_vertex_indices,
            template_bone_tails=self.template_bone_tails,
            bone_tails_blendshapes=self.bone_tails_blendshapes,
            bone_rolls_rotmat=self.bone_rolls_rotmat,
            bone_nonzeroweight_mask=self.bone_nonzeroweight_mask,
            bone_vertex_indices=self.bone_vertex_indices,
            bone_vertex_weights=self.bone_vertex_weights,
            template_bone_vertices=self.template_bone_vertices,
            reference_bone_orientations=self.reference_bone_orientations,
            bone_template_orientation_matrices=self.bone_template_orientation_matrices,
            bone_orientation_blendshapes=self.bone_orientation_blendshapes,
        )


    def get_rest_vertices(self, blendshape_coeffs: torch.Tensor) -> torch.Tensor:
        return skinning.apply_linear_blendshape(self.template_vertices, self.blendshapes, blendshape_coeffs)

    def get_rest_model(self, blendshape_coeffs: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.bone_orientation == "blender":
            return self._get_tail_rest_model(blendshape_coeffs)
        if self.bone_orientation == "procrustes":
            return self._get_procrustes_rest_model(blendshape_coeffs)
        raise ValueError(f"Unknown bone orientation method: {self.bone_orientation!r}")

    def _get_tail_rest_model(self, blendshape_coeffs: torch.Tensor) -> dict[str, torch.Tensor]:
        assert self.template_bone_tails is not None
        assert self.bone_tails_blendshapes is not None
        assert self.bone_rolls_rotmat is not None
        assert self.y_axis is not None
        assert self.degenerate_rotation is not None

        rest_vertices = self.get_rest_vertices(blendshape_coeffs)
        rest_bone_heads = skinning.apply_linear_blendshape(self.template_bone_heads, self.bone_heads_blendshapes, blendshape_coeffs)
        rest_bone_tails = skinning.apply_linear_blendshape(self.template_bone_tails, self.bone_tails_blendshapes, blendshape_coeffs)

        if self.bone_orientation == "blender":
            rest_bone_poses = kinematics.get_bone_poses(rest_bone_heads, rest_bone_tails, self.bone_rolls_rotmat, y_axis=self.y_axis, degenerate_rotation=self.degenerate_rotation)
        else:
            raise NotImplementedError(f"Bone orientation {self.bone_orientation} not implemented. Supported orientation is 'blender'.")

        if self.root_identity_orientation:
            rest_bone_poses[:, 0, :3, :3] = torch.eye(3, device=rest_bone_poses.device, dtype=rest_bone_poses.dtype)

        return dict(rest_vertices=rest_vertices, rest_bone_heads=rest_bone_heads, rest_bone_tails=rest_bone_tails, rest_bone_poses=rest_bone_poses)

    def _get_procrustes_rest_model(self, blendshape_coeffs: torch.Tensor) -> dict[str, torch.Tensor]:
        rest_vertices = self.get_rest_vertices(blendshape_coeffs)
        rest_bone_heads = skinning.apply_linear_blendshape(self.template_bone_heads, self.bone_heads_blendshapes, blendshape_coeffs)
        batch_size = rest_vertices.shape[0]
        if self.bone_template_orientation_matrices is not None:
            # Precomputed variant: the cross-covariance matrices are linear in the blendshape coefficients.
            # b: batch, a: blend shape, k: bone
            M = self.bone_template_orientation_matrices[None] + torch.einsum(
                "ba,akij->bkij", blendshape_coeffs, self.bone_orientation_blendshapes
            )
            rest_bone_orientation = roma.special_procrustes(M)
        else:
            assert self.bone_nonzeroweight_mask is not None
            assert self.bone_vertex_indices is not None
            assert self.bone_vertex_weights is not None
            assert self.template_bone_vertices is not None

            bone_vertices = torch.gather(
                rest_vertices[:, None].expand(-1, self.bone_vertex_indices.shape[0], -1, -1),
                dim=2,
                index=self.bone_vertex_indices[None, :, :, None].expand(batch_size, -1, -1, 3),
            )
            bone_vertices = bone_vertices - rest_bone_heads[:, self.bone_nonzeroweight_mask, None, :]
            R = roma.rigid_vectors_registration(self.template_bone_vertices[None], bone_vertices, weights=self.bone_vertex_weights[None])
            rest_bone_orientation = torch.eye(3, device=rest_bone_heads.device, dtype=rest_bone_heads.dtype).expand(batch_size, self.bone_count, 3, 3).clone()
            rest_bone_orientation[:, self.bone_nonzeroweight_mask] = R
        rest_bone_poses = roma.Rigid(linear=rest_bone_orientation, translation=rest_bone_heads).to_homogeneous()
        if self.root_identity_orientation:
            rest_bone_poses[:, 0, :3, :3] = torch.eye(3, device=rest_bone_poses.device, dtype=rest_bone_poses.dtype)
        return dict(rest_vertices=rest_vertices, rest_bone_heads=rest_bone_heads, rest_bone_poses=rest_bone_poses)


    def parse_delta_transforms_dict(self, delta_transforms_dict):
        """
        Converts a dictionary, namedtuple, or tensor representation of delta transforms
        into a batched tensor of homogeneous transformation matrices.

        This function supports the following input formats:
        - A `dict` or `namedtuple` mapping `bone_label` strings to per-bone delta transforms
        (either `torch.Tensor` or `roma.Rigid` objects), where each transform is of shape `(B, 4, 4)`.
        - A full `torch.Tensor` of shape `(B, N, 4, 4)` representing the full batch of transforms.

        Any bones missing from the input dict/namedtuple are automatically filled with identity transforms.

        Args:
            delta_transforms_dict (dict | namedtuple | torch.Tensor):
                A dictionary or namedtuple mapping bone labels (from `self.bone_labels`)
                to delta transform tensors or `roma.Rigid` objects of shape `(B, 4, 4)`,
                or a tensor of shape `(B, N, 4, 4)` representing the full batch directly.

        Returns:
            torch.Tensor: A tensor of shape `(B, N, 4, 4)`, where `B` is the batch size and
                        `N` is the number of joints (length of `self.bone_labels`), representing
                        the batched homogeneous transformation matrices.

        Raises:
            NameError: If `delta_transforms_dict` is not a supported type.
            AssertionError: If any provided transform does not have the expected shape `(B, 4, 4)`.
        """

        if isinstance(delta_transforms_dict, tuple) and hasattr(delta_transforms_dict, '_fields'):
            delta_transforms_dict = delta_transforms_dict._asdict()

        if isinstance(delta_transforms_dict, dict):
            batch_size = len(next(iter(delta_transforms_dict.values())))
            identity = torch.eye(4, dtype=self.template_vertices.dtype, device=self.template_vertices.device)[None].repeat(batch_size, 1, 1)
            delta_transforms = []
            for bone_id, bone_label in enumerate(self.bone_labels):
                if bone_label in delta_transforms_dict:
                    delta = delta_transforms_dict[bone_label]
                    if isinstance(delta, roma.Rigid):
                        delta = delta.to_homogeneous()
                    assert delta.shape == (batch_size, 4, 4), f"Invalid shape {delta.shape} for bone '{bone_label}', shape should be {(batch_size, 4, 4)}"
                else:
                    delta = identity
                delta_transforms.append(delta)
            return torch.stack(delta_transforms, dim=1)

        elif delta_transforms_dict is None:
            # No pose supplied: return batch-1 identity deltas.
            identity = torch.eye(4, dtype=self.template_vertices.dtype, device=self.template_vertices.device)[None].repeat(1, len(self.bone_labels), 1, 1)
            return identity

        elif isinstance(delta_transforms_dict, torch.Tensor):
            return delta_transforms_dict

        else:
            raise NameError(f"delta_transforms_dict should be a dict, a namedtuple or a tensor, but got {type(delta_transforms_dict)}")

    def get_bone_ends(
        self,
        rest_bone_heads: torch.Tensor,
        rest_bone_tails: torch.Tensor,
        rest_bone_poses: torch.Tensor,
        bone_poses: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        relative_transform = roma.Rigid.from_homogeneous(bone_poses) @ roma.Rigid.from_homogeneous(rest_bone_poses).inverse()
        bone_heads = relative_transform.apply(rest_bone_heads)
        bone_tails = relative_transform.apply(rest_bone_tails)
        return bone_heads, bone_tails

    def get_skinned_vertices(self, rest_vertices: torch.Tensor, bone_transforms) -> torch.Tensor:
        """
        Args:
            - rest_vertices: BxVx3
            - bone_transforms: list of J batch of transformations
        """
        if isinstance(bone_transforms, list) and isinstance(bone_transforms[0], roma.Rigid):
            bone_transforms = roma.Rigid(torch.stack([t.linear for t in bone_transforms], dim=1), torch.stack([t.translation for t in bone_transforms], dim=1))
            bone_transforms = bone_transforms.to_homogeneous()
        elif isinstance(bone_transforms, torch.Tensor):
            pass
        vertices = self._skinning_method(rest_vertices,
                                        bone_weights=self.vertex_bone_weights.unsqueeze(dim=0),
                                        bone_indices=self.vertex_bone_indices.unsqueeze(dim=0),
                                        bone_transforms=bone_transforms)
        return vertices

    def _expand_batch_size(self, bone_transforms: torch.Tensor, rest_bone_poses: torch.Tensor):
        bone_batch_size = bone_transforms.shape[0]
        blendshape_batch_size = rest_bone_poses.shape[0]
        if bone_batch_size == blendshape_batch_size:
            return bone_transforms, rest_bone_poses
        if blendshape_batch_size > 1 and bone_batch_size > 1:
            raise ValueError(f"Batch size of pose_parameters ({bone_batch_size}) and blendshape_coeffs ({blendshape_batch_size}) must match, one if the two must have batch size 1.")

        new_batch_size = max(bone_batch_size, blendshape_batch_size)
        rest_bone_poses = rest_bone_poses.expand(new_batch_size, -1, 4, 4)
        bone_transforms = bone_transforms.expand(new_batch_size, -1, 4, 4)
        return bone_transforms, rest_bone_poses


    def get_bone_transforms(
        self,
        pose_parameters,
        rest_bone_poses: torch.Tensor,
        pose_parameterization: PoseParameterization | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pose_parameterization = pose_parameterization or self.pose_parameterization
        assert pose_parameterization is not None
        delta_transforms = self.parse_delta_transforms_dict(pose_parameters)

        # Expand the rest bone poses to match the batch size of the delta transforms if rest data has batch size 1
        delta_transforms, rest_bone_poses = self._expand_batch_size(delta_transforms, rest_bone_poses)

        bone_transforms = None
        if pose_parameterization == "world":
            bone_poses = delta_transforms
        else:
            if self.reference_bone_orientations is not None:
                # Use the reference bone orientations
                ref_bone_poses, _ = kinematics.parallel_forward_kinematic_absolute_orientations(self.kinematic_propagation_fronts, rest_bone_poses=rest_bone_poses, absolute_orientations=self.reference_bone_orientations[None])
            else:
                ref_bone_poses = rest_bone_poses
            if pose_parameterization == "local-bone-world":
                base_transform = None
                bone_poses, _ = kinematics.parallel_forward_kinematic(self.kinematic_propagation_fronts, rest_bone_poses=ref_bone_poses, delta_transforms=delta_transforms, base_transform=base_transform)
            elif pose_parameterization == "local-bone":
                # Pose is parameterized as local transforms relative to the reference pose, expressed in bone space.
                # The reference bone is the origin
                base_transform = roma.Rigid.from_homogeneous(ref_bone_poses[:,0]).inverse().to_homogeneous()
                bone_poses, _ = kinematics.parallel_forward_kinematic(self.kinematic_propagation_fronts, rest_bone_poses=ref_bone_poses, delta_transforms=delta_transforms, base_transform=base_transform)
            elif pose_parameterization == "local-ref":
                # Pose is parameterized as local transforms relative to the reference pose, expressed in the reference pose space.
                # The reference bone is the origin
                base_transform = roma.Rigid.from_homogeneous(ref_bone_poses[:,0]).inverse().to_homogeneous()
                reference_orientations = roma.Rigid(ref_bone_poses[:,:,:3,:3], translation=None)
                T = reference_orientations.inverse().to_homogeneous() @ delta_transforms @ reference_orientations.to_homogeneous()
                bone_poses, _ = kinematics.parallel_forward_kinematic(self.kinematic_propagation_fronts, rest_bone_poses=ref_bone_poses, delta_transforms=T, base_transform=base_transform)
            elif pose_parameterization == "world-orient":
                # Use the root bone as origin
                base_transform = (roma.Rigid.from_homogeneous(delta_transforms[:,0]) @ roma.Rigid.from_homogeneous(rest_bone_poses[:,0]).inverse()).to_homogeneous()
                bone_poses, bone_transforms = kinematics.parallel_forward_kinematic_absolute_orientations(self.kinematic_propagation_fronts, rest_bone_poses=rest_bone_poses, absolute_orientations=delta_transforms[...,:3,:3], base_transform=base_transform)
            else:
                raise NotImplementedError(f"Pose parameterization {pose_parameterization} not implemented")

        if bone_transforms is None:
            bone_transforms = bone_poses @ roma.Rigid.from_homogeneous(rest_bone_poses).inverse().to_homogeneous()
        return bone_transforms, bone_poses



    def forward(
        self,
        pose_parameters,
        blendshape_coeffs: torch.Tensor,
        pose_parameterization: PoseParameterization | None = None,
        return_bone_ends: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Helper function to compute the skinned vertices and bone poses.
        Args:
            - pose_parameters: BxJx4x4
            - blendshape_coeffs: BxN
        Returns:
            - A dictionary with:
                - blendshape_coeffs: BxN
                - vertices: BxVx3
                - bone_poses: BxJx4x4
        """
        output = self.get_rest_model(blendshape_coeffs)
        rest_bone_poses = output["rest_bone_poses"]
        bone_transforms, bone_poses = self.get_bone_transforms(pose_parameters, rest_bone_poses, pose_parameterization=pose_parameterization)
        vertices = self.get_skinned_vertices(bone_transforms=bone_transforms, rest_vertices=output["rest_vertices"].expand(bone_transforms.shape[0], -1, -1))
        output.update(vertices=vertices,
                    bone_poses=bone_poses)
        if return_bone_ends and self.bone_orientation == "procrustes":
            raise NotImplementedError(
                "return_bone_ends=True is not supported for "
                "bone_orientation='procrustes'."
            )
        if return_bone_ends:
            rest_bone_heads = output["rest_bone_heads"]
            rest_bone_tails = output["rest_bone_tails"]
            bone_heads, bone_tails = self.get_bone_ends(rest_bone_heads, rest_bone_tails, rest_bone_poses, bone_poses)
            output["bone_heads"] = bone_heads
            output["bone_tails"] = bone_tails
        return output

    def get_pose_parameterization(
        self,
        model_output: dict[str, torch.Tensor],
        pose_parameterization: PoseParameterization,
    ) -> torch.Tensor:
        rest_bone_poses = model_output["rest_bone_poses"]
        bone_poses = model_output["bone_poses"]

        if pose_parameterization == "world":
            return bone_poses
        elif pose_parameterization == "world-orient":
            output = bone_poses.clone()
            output[...,1:,:3,3] = 0.0
            return output
        elif pose_parameterization == "local-bone":
            if self.reference_bone_orientations is not None:
                # Use the reference bone orientations
                ref_bone_poses, _ = kinematics.parallel_forward_kinematic_absolute_orientations(self.kinematic_propagation_fronts, rest_bone_poses=rest_bone_poses, absolute_orientations=self.reference_bone_orientations[None])
            else:
                ref_bone_poses = rest_bone_poses
            ref_relative = roma.Rigid.from_homogeneous(ref_bone_poses[:, self.bone_parents[1:]]).inverse().to_homogeneous() @ ref_bone_poses[:,1:]
            relative = roma.Rigid.from_homogeneous(bone_poses[:, self.bone_parents[1:]]).inverse().to_homogeneous() @ bone_poses[:,1:]
            local = ref_relative.inverse() @ relative
            return torch.cat((bone_poses[:,0, None], local), dim=1)
        elif pose_parameterization == "local-bone-world":
            if self.reference_bone_orientations is not None:
                # Use the reference bone orientations
                ref_bone_poses, _ = kinematics.parallel_forward_kinematic_absolute_orientations(self.kinematic_propagation_fronts, rest_bone_poses=rest_bone_poses, absolute_orientations=self.reference_bone_orientations[None])
            else:
                ref_bone_poses = rest_bone_poses
            ref_relative = roma.Rigid.from_homogeneous(ref_bone_poses[:, self.bone_parents[1:]]).inverse().to_homogeneous() @ ref_bone_poses[:,1:]
            relative = roma.Rigid.from_homogeneous(bone_poses[:, self.bone_parents[1:]]).inverse().to_homogeneous() @ bone_poses[:,1:]
            local = ref_relative.inverse() @ relative
            root = roma.Rigid.from_homogeneous(ref_bone_poses[:,0]).inverse().to_homogeneous() @ bone_poses[:,0]
            return torch.cat((root[:,None], local), dim=1)
        elif pose_parameterization == "local-ref":
            output = self.get_pose_parameterization(model_output, pose_parameterization="local-bone")
            if self.reference_bone_orientations is not None:
                # Use the reference bone orientations
                ref_bone_poses, _ = kinematics.parallel_forward_kinematic_absolute_orientations(self.kinematic_propagation_fronts, rest_bone_poses=rest_bone_poses, absolute_orientations=self.reference_bone_orientations[None])
            else:
                ref_bone_poses = rest_bone_poses
            reference_orientations = roma.Rigid(ref_bone_poses[:,:,:3,:3], translation=None)
            output = reference_orientations.to_homogeneous() @ output @ reference_orientations.inverse().to_homogeneous()
            return output
        else:
            raise NotImplementedError(f"Pose parametrization {pose_parameterization} not implemented")
