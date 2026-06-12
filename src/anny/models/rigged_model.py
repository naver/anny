# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
from typing import Literal
import torch
import anny.utils.kinematics as kinematics
import anny.skinning.skinning as skinning
from anny.utils.mesh_utils import triangulate_faces
import roma
import warnings

PoseParameterization = Literal["world", "local-bone-world", "local-bone", "local-ref", "world-orient"]
BoneOrientation = Literal["blender", "gramschmidtyx", "gramschmidtyz", "blender-rootidentity"]


class RiggedModelWithLinearBlendShapes(torch.nn.Module):
    def __init__(self,
                 template_vertices,
                 faces,
                 texture_coordinates,
                 face_texture_coordinate_indices,
                 blendshapes,
                 template_bone_heads,
                 bone_heads_blendshapes,
                 bone_parents,
                 bone_labels,
                 vertex_bone_weights,
                 vertex_bone_indices,
                 skinning_method : str = None,
                 reference_bone_orientations = None,
                 pose_parameterization : str = "root_relative_world"):
        super().__init__()
        self.template_vertices = torch.nn.Buffer(template_vertices, persistent=False)
        self.faces = faces
        self.texture_coordinates = torch.nn.Buffer(texture_coordinates, persistent=False)
        self.face_texture_coordinate_indices = torch.nn.Buffer(face_texture_coordinate_indices, persistent=False)
        self.blendshapes = torch.nn.Buffer(blendshapes, persistent=False)
        self.template_bone_heads = torch.nn.Buffer(template_bone_heads, persistent=False)
        self.bone_heads_blendshapes = torch.nn.Buffer(bone_heads_blendshapes, persistent=False)
        self.reference_bone_orientations = torch.nn.Buffer(reference_bone_orientations, persistent=False) if reference_bone_orientations is not None else None
        self.bone_parents = bone_parents
        self.kinematic_propagation_fronts = kinematics.get_kinematic_propagation_fronts(bone_parents)
        self.bone_labels = bone_labels
        self.vertex_bone_weights = torch.nn.Buffer(vertex_bone_weights, persistent=False)
        self.vertex_bone_indices = torch.nn.Buffer(vertex_bone_indices, persistent=False)
        self.set_skinning_method(skinning_method)
        self.pose_parameterization = pose_parameterization

    @property
    def bone_count(self):
        return len(self.bone_labels)

    @property
    def dtype(self):
        return self.template_vertices.dtype

    @property
    def device(self):
        return self.template_vertices.device

    def get_triangular_faces(self):
        """
        Return a triangulated version of the faces, splitting quads when needed.
        """
        triangular_faces = torch.tensor(triangulate_faces(vertices=self.template_vertices, faces=self.faces.detach().cpu().numpy().tolist()), device=self.device)
        return triangular_faces

    def set_skinning_method(self, skinning_method):
        self._skinning_method_name = skinning_method  # preserve original (None = auto-detect)
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

    def to_model_data(self):
        """Return a :class:`~anny.models.model_data.ModelData` representing this model.

        Implemented by concrete subclasses.
        """
        raise NotImplementedError

    def save_safetensors(self, path: str) -> None:
        """Serialize this model to a safetensors file."""
        self.to_model_data().save_safetensors(path)

    @classmethod
    def load_safetensors(cls, path: str):
        """Deserialize a model from a safetensors file written by :meth:`save_safetensors`."""
        from anny.models.model_data import ModelData, model_from_model_data
        return model_from_model_data(ModelData.load_safetensors(path))

    def get_rest_vertices(self, blendshape_coeffs):
        return skinning.apply_linear_blendshape(self.template_vertices, self.blendshapes, blendshape_coeffs)

    def get_rest_model(self, blendshape_coeffs):
        rest_vertices = self.get_rest_vertices(blendshape_coeffs)
        rest_bone_heads = skinning.apply_linear_blendshape(self.template_bone_heads, self.bone_heads_blendshapes, blendshape_coeffs)
        rest_bone_poses = torch.eye(4, device=rest_vertices.device, dtype=rest_vertices.dtype)[None,None].expand(rest_bone_heads.shape[0], rest_bone_heads.shape[1], 4, 4).clone()
        rest_bone_poses[...,:3,3] = rest_bone_heads
        return dict(rest_vertices=rest_vertices, rest_bone_heads=rest_bone_heads, rest_bone_poses=rest_bone_poses)


    def parse_delta_transforms_dict(self, delta_transforms_dict, batch_size=None):
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
            batch_size = batch_size if batch_size is not None else len(next(iter(delta_transforms_dict.values())))
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
            identity = torch.eye(4, dtype=self.template_vertices.dtype, device=self.template_vertices.device)[None].repeat(batch_size, len(self.bone_labels), 1, 1)
            return identity

        elif isinstance(delta_transforms_dict, torch.Tensor):
            return delta_transforms_dict

        else:
            raise NameError(f"delta_transforms_dict should be a dict, a namedtuple or a tensor, but got {type(delta_transforms_dict)}")

    def get_bone_ends(self, rest_bone_heads, rest_bone_tails, rest_bone_poses, bone_poses):
        relative_transform = roma.Rigid.from_homogeneous(bone_poses) @ roma.Rigid.from_homogeneous(rest_bone_poses).inverse()
        bone_heads = relative_transform.apply(rest_bone_heads)
        bone_tails = relative_transform.apply(rest_bone_tails)
        return bone_heads, bone_tails

    def get_skinned_vertices(self, rest_vertices, bone_transforms):
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

    def get_bone_transforms(self, pose_parameters, rest_bone_poses, batch_size, pose_parameterization=None):
         pose_parameterization = self.pose_parameterization if (pose_parameterization is None) else pose_parameterization
         delta_transforms = self.parse_delta_transforms_dict(pose_parameters, batch_size=batch_size)

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


    def forward(self, pose_parameters, blendshape_coeffs, pose_parameterization=None, return_bone_ends=False):
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
        bone_transforms, bone_poses = self.get_bone_transforms(pose_parameters, rest_bone_poses, batch_size=blendshape_coeffs.shape[0], pose_parameterization=pose_parameterization)
        vertices = self.get_skinned_vertices(bone_transforms=bone_transforms, rest_vertices=output["rest_vertices"])
        output.update(vertices=vertices,
                    bone_poses=bone_poses)
        if return_bone_ends:
            rest_bone_heads = output["rest_bone_heads"]
            rest_bone_tails = output["rest_bone_tails"]
            bone_heads, bone_tails = self.get_bone_ends(rest_bone_heads, rest_bone_tails, rest_bone_poses, bone_poses)
            output["bone_heads"] = bone_heads
            output["bone_tails"] = bone_tails
        return output

    def get_pose_parameterization(self,
                                model_output,
                                pose_parameterization):
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

class RiggedModelWithBoneTails(RiggedModelWithLinearBlendShapes):
    def __init__(self,
                 template_vertices,
                 faces,
                 texture_coordinates,
                 face_texture_coordinate_indices,
                 blendshapes,
                 template_bone_heads,
                 bone_heads_blendshapes,
                 template_bone_tails,
                 bone_tails_blendshapes,
                 bone_rolls_rotmat,
                 bone_parents,
                 bone_labels,
                 vertex_bone_weights,
                 vertex_bone_indices,
                 skinning_method : str = None,
                 pose_parameterization : str = "local-bone",
                 bone_orientation = "blender-rootidentity"):
        super().__init__(
            template_vertices=template_vertices,
            faces=faces,
            texture_coordinates=texture_coordinates,
            face_texture_coordinate_indices=face_texture_coordinate_indices,
            blendshapes=blendshapes,
            template_bone_heads=template_bone_heads,
            bone_heads_blendshapes=bone_heads_blendshapes,
            bone_parents=bone_parents,
            bone_labels=bone_labels,
            vertex_bone_weights=vertex_bone_weights,
            vertex_bone_indices=vertex_bone_indices,
            skinning_method=skinning_method,
            pose_parameterization=pose_parameterization)

        self.template_bone_tails = torch.nn.Buffer(template_bone_tails, persistent=False)
        self.bone_tails_blendshapes = torch.nn.Buffer(bone_tails_blendshapes, persistent=False)
        self.y_axis = torch.nn.Buffer(torch.as_tensor([0.,1.,0.], dtype=self.template_vertices.dtype), persistent=False)
        self.degenerate_rotation = torch.nn.Buffer(torch.tensor([[1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.]], dtype=self.template_vertices.dtype), persistent=False)
        self.bone_rolls_rotmat = torch.nn.Buffer(bone_rolls_rotmat, persistent=False)

        if bone_orientation == "blender-rootidentity":
            self.bone_orientation = "blender"
            self.root_identity_orientation = True
        else:
            self.bone_orientation = bone_orientation
            self.root_identity_orientation = False

    def get_rest_model(self, blendshape_coeffs):
        rest_vertices = self.get_rest_vertices(blendshape_coeffs)

        rest_bone_heads = skinning.apply_linear_blendshape(self.template_bone_heads, self.bone_heads_blendshapes, blendshape_coeffs)
        rest_bone_tails = skinning.apply_linear_blendshape(self.template_bone_tails, self.bone_tails_blendshapes, blendshape_coeffs)

        if self.bone_orientation == "blender":
            rest_bone_poses = kinematics.get_bone_poses(rest_bone_heads, rest_bone_tails, self.bone_rolls_rotmat, y_axis=self.y_axis, degenerate_rotation=self.degenerate_rotation)
        elif self.bone_orientation == "gramschmidtyx":
            # We want to maintain the same X direction as the template pose, and only align the Y direction to the head-tail direction of the rest pose.
            y = rest_bone_tails - rest_bone_heads
            y = y / torch.linalg.norm(y, dim=-1, keepdim=True)
            template_x = self.template_bone_poses[...,:3,0]
            yxmz = roma.special_gramschmidt(torch.stack([y, template_x.expand_as(y)], dim=-1))
            R = yxmz[...,[1,0,2]] * torch.tensor([1, 1, -1], device=yxmz.device, dtype=yxmz.dtype).reshape(1,1,1,3)

            rest_bone_poses = torch.empty(R.shape[:-2] + (4, 4), device=R.device, dtype=R.dtype)
            rest_bone_poses[..., :3, :3] = R
            rest_bone_poses[..., :3, 3] = rest_bone_heads
            rest_bone_poses[..., 3, :3] = 0.0
            rest_bone_poses[..., 3, 3] = 1.0
        elif self.bone_orientation == "gramschmidtyz":
            # We want to maintain the same Z direction as the template pose, and only align the Y direction to the head-tail direction of the rest pose.
            y = rest_bone_tails - rest_bone_heads
            y = y / torch.linalg.norm(y, dim=-1, keepdim=True)
            template_z = self.template_bone_poses[...,:3,2]
            yzx = roma.special_gramschmidt(torch.stack([y, template_z.expand_as(y)], dim=-1))
            R = yzx[..., [2,0,1]]

            rest_bone_poses = torch.empty(R.shape[:-2] + (4, 4), device=R.device, dtype=R.dtype)
            rest_bone_poses[..., :3, :3] = R
            rest_bone_poses[..., :3, 3] = rest_bone_heads
            rest_bone_poses[..., 3, :3] = 0.0
            rest_bone_poses[..., 3, 3] = 1.0
        else:
            raise NotImplementedError(f"Bone orientation {self.bone_orientation} not implemented. Supported orientations are 'blender', 'gramschmidtyx' and 'gramschmidtyz'.")

        if self.root_identity_orientation:
            # Manually set root bone orientation to identity
            rest_bone_poses[:,0, :3, :3] = torch.eye(3, device=rest_bone_poses.device, dtype=rest_bone_poses.dtype)

        return dict(rest_vertices=rest_vertices, rest_bone_heads=rest_bone_heads, rest_bone_tails=rest_bone_tails, rest_bone_poses=rest_bone_poses)


class RiggedModelWithBoneVertices(RiggedModelWithLinearBlendShapes):
    def __init__(self,
                 template_vertices,
                 faces,
                 texture_coordinates,
                 face_texture_coordinate_indices,
                 blendshapes,
                 template_bone_heads,
                 bone_heads_blendshapes,
                 bone_parents,
                 bone_labels,
                 vertex_bone_weights,
                 vertex_bone_indices,
                 bone_nonzeroweight_mask,
                 bone_vertex_indices,
                 bone_vertex_weights,
                 template_bone_vertices,
                 reference_bone_orientations=None,
                 skinning_method : str = None,
                 pose_parameterization : PoseParameterization = "local-bone"):
        super().__init__(
            template_vertices=template_vertices,
            faces=faces,
            texture_coordinates=texture_coordinates,
            face_texture_coordinate_indices=face_texture_coordinate_indices,
            blendshapes=blendshapes,
            template_bone_heads=template_bone_heads,
            bone_heads_blendshapes=bone_heads_blendshapes,
            bone_parents=bone_parents,
            bone_labels=bone_labels,
            vertex_bone_weights=vertex_bone_weights,
            vertex_bone_indices=vertex_bone_indices,
            skinning_method=skinning_method,
            reference_bone_orientations=reference_bone_orientations,
            pose_parameterization=pose_parameterization)

        self.bone_nonzeroweight_mask = torch.nn.Buffer(bone_nonzeroweight_mask, persistent=False)
        self.bone_vertex_indices = torch.nn.Buffer(bone_vertex_indices, persistent=False)
        self.bone_vertex_weights = torch.nn.Buffer(bone_vertex_weights, persistent=False)
        self.template_bone_vertices = torch.nn.Buffer(template_bone_vertices, persistent=False)

        # Register zero-valued tail buffers for compatibility with code that reads these attributes
        # (e.g. retopology functions). The procrustes orientation does not use tails.
        zero_tails = torch.zeros_like(template_bone_heads)
        zero_tails_blendshapes = torch.zeros_like(bone_heads_blendshapes)
        identity_rolls = torch.eye(3, dtype=template_vertices.dtype, device=template_vertices.device).expand(1, self.bone_count, 3, 3).clone()
        self.template_bone_tails = torch.nn.Buffer(zero_tails, persistent=False)
        self.bone_tails_blendshapes = torch.nn.Buffer(zero_tails_blendshapes, persistent=False)
        self.bone_rolls_rotmat = torch.nn.Buffer(identity_rolls, persistent=False)

    def get_rest_model(self, blendshape_coeffs):
        rest_vertices = self.get_rest_vertices(blendshape_coeffs)
        rest_bone_heads = skinning.apply_linear_blendshape(self.template_bone_heads, self.bone_heads_blendshapes, blendshape_coeffs)

        batch_size = rest_vertices.shape[0]
        bone_vertices = torch.gather(rest_vertices[:,None].expand(-1, self.bone_vertex_indices.shape[0], -1, -1), dim=2, index=self.bone_vertex_indices[None,:,:,None].expand(batch_size, -1, -1, 3))
        bone_vertices = bone_vertices - rest_bone_heads[:,self.bone_nonzeroweight_mask,None,:]
        R = roma.rigid_vectors_registration(self.template_bone_vertices[None], bone_vertices, weights=self.bone_vertex_weights[None])
        rest_bone_orientation = torch.eye(3, device=rest_bone_heads.device, dtype=rest_bone_heads.dtype).expand(batch_size, self.bone_count, 3, 3).clone()
        rest_bone_orientation[:,self.bone_nonzeroweight_mask] = R
        rest_bone_poses = roma.Rigid(linear=rest_bone_orientation, translation=rest_bone_heads).to_homogeneous()
        return dict(rest_vertices=rest_vertices, rest_bone_heads=rest_bone_heads, rest_bone_poses=rest_bone_poses)
