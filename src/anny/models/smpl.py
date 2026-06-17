import inspect
if not hasattr(inspect, 'getargspec'):
    inspect.getargspec = inspect.getfullargspec

import numpy as np
if not hasattr(np, 'int'):
    np.int = np.int_
    np.float = np.float64
    np.bool = np.bool_
    np.complex = np.complex128
    np.object = np.object_
    np.str = np.str_
    np.unicode = np.str_

import torch
import smplx
import anny.models.rigged_model
import roma
import anny.models.model_transforms
from anny.paths import get_anny2smpl_data_path, get_anny2smplx_data_path

class SMPLX(torch.nn.Module):
    def __init__(self, *smplx_args, pose_corrective=True, topology="smplx", **smplx_kwargs):
        super().__init__()
        # Original model
        model = smplx.create(*smplx_args, **smplx_kwargs)

        # Parse useful data
        template_vertices = model.v_template
        base_blendshapes = model.shapedirs.permute(2,0,1)

        template_bone_heads = model.J_regressor @ template_vertices

        # Add expression blendshapes
        blendshapes = torch.concatenate((base_blendshapes, model.expr_dirs.permute(2, 0, 1)), dim=0)

        if pose_corrective:
            # Add pose corrective blend shapes
            pose_corrective_blendshapes = model.posedirs.reshape((len(model.posedirs),-1,3))
            blendshapes = torch.concatenate((blendshapes, pose_corrective_blendshapes), dim=0)

        # Expression and pose corrective blend shapes should not influence bone location
        masked_blendshapes = blendshapes.clone()
        masked_blendshapes[len(base_blendshapes):] = 0.
        # J = joint, V = vertex, S = shape blendshape, D = dimension (3D)
        bone_heads_blendshapes = torch.einsum("JV,SVD->SJD", model.J_regressor, masked_blendshapes)

        bone_count = template_bone_heads.shape[0]
        vertex_bone_weights = model.lbs_weights
        vertex_bone_indices = torch.arange(bone_count).unsqueeze(0).expand(vertex_bone_weights.shape[0], -1)
        bone_labels = [f"bone_{i}" for i in range(bone_count)]

        self.pose_mean = model.pose_mean.reshape(1,-1,3)
        self.use_pca = model.use_pca
        if self.use_pca:
            self.left_hand_components = model.left_hand_components
            self.right_hand_components = model.right_hand_components
        self.pose_corrective = pose_corrective
        metadata = anny.ModelMetadata(
            model_type="tail",
            bone_parents=model.parents,
            bone_labels=bone_labels,
            pose_parameterization="local-bone-world",
            skinning_method=None,
            local_change_labels=[], # Dummy values to workaround ModelMetadata limitations
            all_phenotypes=False,
            extrapolate_phenotypes=False,
            bone_orientation="blender",
        )
        data = anny.ModelData(metadata=metadata,
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
                                    base_mesh_vertex_indices=None, # Dummy values to workaround ModelData limitations
                                    stacked_phenotype_blend_shapes_mask=None,
                                    )
        if topology == "smplx":
            pass
        else:
            # Load the SMPL-X/Anny correspondences
            anny2smplx_state_dict = torch.load(get_anny2smplx_data_path(),
                                    map_location="cpu",
                                    weights_only=True)
            barycentric_coordinates = anny2smplx_state_dict["dst2anny_barycentric_coordinates"]
            reference_vertex_indices = anny2smplx_state_dict["dst2anny_vertex_indices"]
            vertices = barycentric_coordinates[0][:,None] * data.template_vertices[reference_vertex_indices[:,0]] + \
                    barycentric_coordinates[1][:,None] * data.template_vertices[reference_vertex_indices[:,1]] + \
                    barycentric_coordinates[2][:,None] * data.template_vertices[reference_vertex_indices[:,2]]
            faces = anny2smplx_state_dict["anny_faces"]
            anny_data = anny.models.model_transforms.apply_retopology(
                data,
                vertices=vertices,
                faces=faces,
                reference_vertex_indices=reference_vertex_indices,
                barycentric_coordinates=barycentric_coordinates)
        if topology == "anny":
            data = anny_data
            # Store the validity mask regarding Anny vertices
            self.vertex_mask = torch.nn.Buffer(anny2smplx_state_dict["anny_vertex_mask"], persistent = False)
        elif topology == "smpl":
            # Load the SMPL topology
            state_dict = torch.load(get_anny2smpl_data_path(),
                                    map_location="cpu",
                                    weights_only=True)
            barycentric_coordinates = state_dict["anny2dst_barycentric_coordinates"]
            reference_vertex_indices = state_dict["anny2dst_vertex_indices"]
            vertices = barycentric_coordinates[0][:,None] * anny_data.template_vertices[reference_vertex_indices[:,0]] + \
                        barycentric_coordinates[1][:,None] * anny_data.template_vertices[reference_vertex_indices[:,1]] + \
                        barycentric_coordinates[2][:,None] * anny_data.template_vertices[reference_vertex_indices[:,2]]
            faces = state_dict["dst_faces"]
            data = anny.models.model_transforms.apply_retopology(
                anny_data,
                vertices=vertices,
                faces=faces,
                reference_vertex_indices=reference_vertex_indices,
                barycentric_coordinates=barycentric_coordinates,
            )
            # No vertex masking
            self.vertex_mask = torch.nn.Buffer(torch.ones(len(vertices), dtype=torch.float64), persistent = False)
        else:
            assert topology == "smplx"

        self.rigged_model = anny.models.rigged_model.RiggedModelWithLinearBlendShapes(
            template_vertices=data.template_vertices,
            faces=data.faces,
            texture_coordinates=data.texture_coordinates,
            face_texture_coordinate_indices=data.face_texture_coordinate_indices,
            blendshapes=data.blendshapes,
            template_bone_heads=data.template_bone_heads,
            bone_heads_blendshapes=data.bone_heads_blendshapes,
            bone_parents=data.metadata.bone_parents,
            bone_labels=data.metadata.bone_labels,
            vertex_bone_weights=data.vertex_bone_weights,
            vertex_bone_indices=data.vertex_bone_indices,
            skinning_method=data.metadata.skinning_method,
            reference_bone_orientations=data.reference_bone_orientations,
            pose_parameterization=data.metadata.pose_parameterization,)

    @property
    def faces(self):
        return self.rigged_model.faces

    def forward(self, betas, expression, global_orient, transl, body_pose, leye_pose, reye_pose, left_hand_pose, right_hand_pose, jaw_pose):
        if self.use_pca:
            left_hand_pose = torch.einsum(
                'bi,ij->bj', [left_hand_pose, self.left_hand_components])
            right_hand_pose = torch.einsum(
                'bi,ij->bj', [right_hand_pose, self.right_hand_components])

        # Anny equivalent (without pose corrective blend shapes)
        rotvec = torch.cat([global_orient.reshape(-1, 1, 3),
                                body_pose.reshape(-1, 21, 3),
                                jaw_pose.reshape(-1, 1, 3),
                                leye_pose.reshape(-1, 1, 3),
                                reye_pose.reshape(-1, 1, 3),
                                left_hand_pose.reshape(-1, 15, 3),
                                right_hand_pose.reshape(-1, 15, 3)],
                                dim=1)

        pose_parameters = torch.eye(4).unsqueeze(0).expand(1, self.rigged_model.bone_count, 4, 4).clone()
        pose_parameters[:, :, :3, :3] = roma.rotvec_to_rotmat(rotvec + self.pose_mean)
        pose_parameters[:,0,:3,3] = transl
        blendshape_coeffs = torch.cat((betas, expression), dim=1)

        if self.pose_corrective:
            batch_size = len(pose_parameters)
            pose_corrective_blendshape_coeffs = (pose_parameters[:,1:,:3, :3] - torch.eye(3, dtype=pose_parameters.dtype)[None,None]).view(batch_size, -1)
            full_blendshape_coeffs = torch.concatenate((blendshape_coeffs, pose_corrective_blendshape_coeffs), dim=-1)
        else:
            full_blendshape_coeffs = blendshape_coeffs

        

        return self.rigged_model(pose_parameters=pose_parameters, blendshape_coeffs=full_blendshape_coeffs)
    
class SMPL(torch.nn.Module):
    def __init__(self, *smpl_args, pose_corrective=True, topology="smpl", **smpl_kwargs):
        super().__init__()
        # Original model
        model = smplx.create(*smpl_args, **smpl_kwargs)

        # Parse useful data
        template_vertices = model.v_template
        base_blendshapes = model.shapedirs.permute(2, 0, 1)

        template_bone_heads = model.J_regressor @ template_vertices

        if pose_corrective:
            # Add pose corrective blend shapes
            pose_corrective_blendshapes = model.posedirs.reshape((len(model.posedirs), -1, 3))
            blendshapes = torch.concatenate((base_blendshapes, pose_corrective_blendshapes), dim=0)
        else:
            blendshapes = base_blendshapes

        # Pose corrective blend shapes should not influence bone location
        masked_blendshapes = blendshapes.clone()
        masked_blendshapes[len(base_blendshapes):] = 0.
        # J = joint, V = vertex, S = shape blendshape, D = dimension (3D)
        bone_heads_blendshapes = torch.einsum("JV,SVD->SJD", model.J_regressor, masked_blendshapes)

        bone_count = template_bone_heads.shape[0]
        vertex_bone_weights = model.lbs_weights
        vertex_bone_indices = torch.arange(bone_count).unsqueeze(0).expand(vertex_bone_weights.shape[0], -1)
        bone_labels = [f"bone_{i}" for i in range(bone_count)]

        self.pose_corrective = pose_corrective
        metadata = anny.ModelMetadata(
            model_type="tail",
            bone_parents=model.parents,
            bone_labels=bone_labels,
            pose_parameterization="local-bone-world",
            skinning_method=None,
            local_change_labels=[], # Dummy values to workaround ModelMetadata limitations
            all_phenotypes=False,
            extrapolate_phenotypes=False,
            bone_orientation="blender",
        )
        data = anny.ModelData(metadata=metadata,
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
                                    base_mesh_vertex_indices=None, # Dummy values to workaround ModelData limitations
                                    stacked_phenotype_blend_shapes_mask=None,
                                    )
        if topology == "smpl":
            pass
        else:
            # Load the SMPL/Anny correspondences
            anny_state_dict = torch.load(get_anny2smpl_data_path(),
                                    map_location="cpu",
                                    weights_only=True)
            barycentric_coordinates = anny_state_dict["dst2anny_barycentric_coordinates"]
            reference_vertex_indices = anny_state_dict["dst2anny_vertex_indices"]
            vertices = barycentric_coordinates[0][:,None] * data.template_vertices[reference_vertex_indices[:,0]] + \
                    barycentric_coordinates[1][:,None] * data.template_vertices[reference_vertex_indices[:,1]] + \
                    barycentric_coordinates[2][:,None] * data.template_vertices[reference_vertex_indices[:,2]]
            faces = anny_state_dict["anny_faces"]
            anny_data = anny.models.model_transforms.apply_retopology(
                data,
                vertices=vertices,
                faces=faces,
                reference_vertex_indices=reference_vertex_indices,
                barycentric_coordinates=barycentric_coordinates,
            )

            if topology == "anny":
                # Store the validity mask regarding Anny vertices
                self.vertex_mask = torch.nn.Buffer(anny_state_dict["anny_vertex_mask"], persistent = False)
            elif topology == "smplx":
                # Load the SMPLX topology
                state_dict = torch.load(get_anny2smplx_data_path(),
                                        map_location="cpu",
                                        weights_only=True)
                barycentric_coordinates = state_dict["anny2dst_barycentric_coordinates"]
                reference_vertex_indices = state_dict["anny2dst_vertex_indices"]
                vertices = barycentric_coordinates[0][:,None] * anny_data.template_vertices[reference_vertex_indices[:,0]] + \
                            barycentric_coordinates[1][:,None] * anny_data.template_vertices[reference_vertex_indices[:,1]] + \
                            barycentric_coordinates[2][:,None] * anny_data.template_vertices[reference_vertex_indices[:,2]]
                faces = state_dict["dst_faces"]
                data = anny.models.model_transforms.apply_retopology(
                    anny_data,
                    vertices=vertices,
                    faces=faces,
                    reference_vertex_indices=reference_vertex_indices,
                    barycentric_coordinates=barycentric_coordinates,
                )
                # No vertex masking
                self.vertex_mask = torch.nn.Buffer(torch.ones(len(vertices), dtype=torch.float64), persistent = False)
            else:
                raise ValueError()

        self.rigged_model = anny.models.rigged_model.RiggedModelWithLinearBlendShapes(
            template_vertices=data.template_vertices,
            faces=data.faces,
            texture_coordinates=data.texture_coordinates,
            face_texture_coordinate_indices=data.face_texture_coordinate_indices,
            blendshapes=data.blendshapes,
            template_bone_heads=data.template_bone_heads,
            bone_heads_blendshapes=data.bone_heads_blendshapes,
            bone_parents=data.metadata.bone_parents,
            bone_labels=data.metadata.bone_labels,
            vertex_bone_weights=data.vertex_bone_weights,
            vertex_bone_indices=data.vertex_bone_indices,
            skinning_method=data.metadata.skinning_method,
            reference_bone_orientations=data.reference_bone_orientations,
            pose_parameterization=data.metadata.pose_parameterization,)

    @property
    def faces(self):
        return self.rigged_model.faces

    def forward(self, betas, global_orient, transl, body_pose):
        # Anny equivalent (without pose corrective blend shapes)
        rotvec = torch.cat([global_orient.reshape(-1, 1, 3),
                                body_pose.reshape(-1, self.rigged_model.bone_count - 1, 3)],
                                dim=1)

        pose_parameters = torch.eye(4).unsqueeze(0).expand(1, self.rigged_model.bone_count, 4, 4).clone()
        pose_parameters[:, :, :3, :3] = roma.rotvec_to_rotmat(rotvec)
        pose_parameters[:, 0, :3, 3] = transl
        blendshape_coeffs = betas

        if self.pose_corrective:
            batch_size = len(pose_parameters)
            pose_corrective_blendshape_coeffs = (pose_parameters[:, 1:, :3, :3] - torch.eye(3, dtype=pose_parameters.dtype)[None, None]).view(batch_size, -1)
            full_blendshape_coeffs = torch.concatenate((blendshape_coeffs, pose_corrective_blendshape_coeffs), dim=-1)
        else:
            full_blendshape_coeffs = blendshape_coeffs

        return self.rigged_model(pose_parameters=pose_parameters, blendshape_coeffs=full_blendshape_coeffs)