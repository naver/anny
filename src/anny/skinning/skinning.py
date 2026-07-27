# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
import torch
import roma
import torch.nn.functional as F

def linear_blend_skinning(vertices, bone_weights, bone_indices, bone_transforms):
    """
    Apply linear blend skinning to a batch of sets of vertices using sparse bone weights.

    Args:
    vertices (torch.Tensor): Tensor of shape (batch_size, num_vertices, 3) containing the vertex positions.
    bone_weights (torch.Tensor): Tensor of shape (batch_size, num_vertices, max_bones_per_vertex) containing the bone weights for each vertex.
    bone_indices (torch.Tensor): Tensor of shape (batch_size, num_vertices, max_bones_per_vertex) containing the bone indices for each vertex.
    bone_transforms (torch.Tensor): Tensor of shape (batch_size, num_bones, 4, 4) containing the bone transformation matrices.

    Returns:
    torch.Tensor: Transformed vertices of shape (batch_size, num_vertices, 3).
    """
    B1, _, _ = vertices.shape
    batch_size = max(B1, bone_transforms.shape[0])

    vertices = vertices.expand(batch_size, -1, -1)
    bone_weights = bone_weights.expand(batch_size, -1, -1)
    bone_indices = bone_indices.expand(batch_size, -1, -1)
    bone_transforms = bone_transforms.expand(batch_size, -1, -1, -1)

    batch_indices = torch.arange(batch_size, device=bone_indices.device)[:, None, None]
    selected_bone_transforms = bone_transforms[batch_indices, bone_indices, :3, :]

    # Blend the transformations
    transforms = torch.sum(bone_weights.unsqueeze(-1).unsqueeze(dim=-1) * selected_bone_transforms, dim=2)

    # Apply the transformations
    # Strong assumption here to get consistent results:
    # The weights are supposed to sum to one, and the transformations are assumed to be affine.
    vertices_homo = torch.cat([vertices, torch.ones_like(vertices[..., :1])], dim=-1)
    return torch.einsum("bvik, bvk -> bvi", transforms, vertices_homo)

def homogeneous_to_dual_quaternion(homogeneous):
    """
    XYZW convention
    Result is q + epsilon * q_tr
    """
    T = roma.Rigid.from_homogeneous(homogeneous)
    q = roma.rotmat_to_unitquat(T.linear)
    q_tr = torch.empty_like(q)
    q_tr[...,:3] = 0.5 * T.translation
    q_tr[...,3] = 1.
    q_tr = roma.quat_product(q_tr, q)
    return q, q_tr

def unit_dual_quaternion_to_homogeneous(dual_quat):
    q, q_tr = dual_quat
    tr = (2. * roma.quat_product(q_tr, roma.quat_conjugation(q)))[...,:3]
    return roma.RigidUnitQuat(q, tr).to_homogeneous()

def dual_quaternion_skinning(vertices, bone_weights, bone_indices, bone_transforms):
    """
    Apply dual quaternion blend skinning to a batch of sets of vertices using sparse bone weights.
    Take homogeneous transformations as input for compatibility with linear_blend_skinning

    Args:
    vertices (torch.Tensor): Tensor of shape (batch_size, num_vertices, 3) containing the vertex positions.
    bone_weights (torch.Tensor): Tensor of shape (batch_size, num_vertices, max_bones_per_vertex) containing the bone weights for each vertex.
    bone_indices (torch.Tensor): Tensor of shape (batch_size, num_vertices, max_bones_per_vertex) containing the bone indices for each vertex.
    bone_transforms (torch.Tensor): Tensor of shape (batch_size, num_bones, 4, 4) containing the bone transformation matrices.

    Returns:
    torch.Tensor: Transformed vertices of shape (batch_size, num_vertices, 3).
    """
    B1, num_vertices, _ = vertices.shape
    max_bones_per_vertex = bone_indices.shape[2]
    num_bones = bone_transforms.shape[1]
    batch_size = max(B1, bone_transforms.shape[0])

    # Cast transfotmations to dual quaternions
    # Two (batch_size, num_vertices, num_bones, 4) quaternion tensors.
    rotation_part, translation_part = homogeneous_to_dual_quaternion(bone_transforms)

    # Gather the relevant bone transforms for each vertex.
    # Note: this creates suboptimal copies. One could switch to custom CUDA/Warp kernels for better performance.
    # Expand bone_transforms so that the vertex dimension matches bone_indices for gathering
    rotation_part = rotation_part.unsqueeze(1).expand(batch_size, num_vertices, num_bones, 4)
    translation_part = translation_part.unsqueeze(1).expand(batch_size, num_vertices, num_bones, 4)
    # We need to make sure bone_indices is aligned to match the gathering axis
    bone_indices_expanded = bone_indices.unsqueeze(-1).expand(batch_size, num_vertices, max_bones_per_vertex, 4)

    # Gather along the bone dimension (axis 2) for the entire batch
    rotation_part = torch.gather(rotation_part, 2, bone_indices_expanded)
    translation_part = torch.gather(translation_part, 2, bone_indices_expanded)

    # Solve the antipodal ambiguity in dual quaternion orientation, considering the first one as reference
    dot_prod = torch.sum(rotation_part * rotation_part[:,:,0,None], dim=-1, keepdim=True) + torch.sum(translation_part * translation_part[:,:,0,None], dim=-1, keepdim=True)
    # Apply sign fipping where required
    sign = (dot_prod >= 0).to(dtype=rotation_part.dtype) * 2. - 1
    rotation_part = rotation_part * sign
    translation_part = translation_part * sign

    # Dual quaternion linear blending
    mean_rotation_part = torch.sum(bone_weights[:,:,:,None] * rotation_part, dim=2)
    mean_translation_part = torch.sum(bone_weights[:,:,:,None] * translation_part, dim=2)
    # Dual quaternion normalization
    norm = torch.linalg.norm(mean_rotation_part, dim=-1, keepdim=True)
    mean_rotation_part = mean_rotation_part / norm
    mean_translation_part = mean_translation_part / norm

    tr = (2. * roma.quat_product(mean_translation_part, roma.quat_conjugation(mean_rotation_part)))[...,:3]
    skinned_vertices = roma.quat_action(mean_rotation_part, vertices) + tr
    return skinned_vertices

def apply_linear_blendshape(template_vertices,
                            blendshapes, blendshape_coeffs):
        """
        Args:
            - template_vertices: Px3 tensor of vertices
            - blendshapes: CxPx3 tensor of blend shape offsets
            - blendshape_coeffs: BxC tensor

        """
        return template_vertices[None] + torch.einsum("cpd, bc -> bpd", blendshapes, blendshape_coeffs)

if __name__ == "__main__":
    # Example usage
    batch_size = 192
    num_vertices = 1000
    max_bones_per_vertex = 8
    num_bones = 20

    vertices = torch.rand((batch_size, num_vertices, 3), device='cuda')  # Example vertices
    bone_indices = torch.randint(0, num_bones, (batch_size, num_vertices, max_bones_per_vertex), device='cuda')  # Example bone indices
    bone_weights = torch.rand((batch_size, num_vertices, max_bones_per_vertex), device='cuda')  # Example bone weights
    bone_weights = bone_weights / bone_weights.sum(dim=2, keepdim=True)  # Normalize bone weights
    bone_transforms = torch.eye(4, device='cuda').unsqueeze(0).unsqueeze(0).expand(batch_size, num_bones, -1, -1)  # Example bone transforms

    transformed_vertices = linear_blend_skinning(vertices, bone_weights, bone_indices, bone_transforms)
    transformed_vertices_qual_quat = dual_quaternion_skinning(vertices, bone_weights, bone_indices, bone_transforms)

    import timeit
    print("LBS", timeit.timeit(lambda : linear_blend_skinning(vertices, bone_weights, bone_indices, bone_transforms), number=200))
    print("DQS", timeit.timeit(lambda : dual_quaternion_skinning(vertices, bone_weights, bone_indices, bone_transforms), number=200))
