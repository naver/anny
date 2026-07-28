# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
import torch
import roma


def get_kinematic_propagation_fronts(parent_indices):
    """
    Group nodes of a kinematic tree into subsets allowing parallel computations of forward kinematics within a set,
    i.e. propagation fronts of forward kinematic information.
    Args:
        - parent_indices: A list in which each element is the index of the parent joint in the kinematic tree.
                         Absence of parent (i.e. root joints) is encoded using -1.

    Returns:
        A tuple of two lists:
        - The first list contains lists of joint indices that can be processed in parallel.
        - The second list contains lists of parent indices associated with each joint in the parallel groups.
    """
    num_joints = len(parent_indices)
    grouped_joints_indices = []
    grouped_joints_parents = []

    # Create a list to keep track of whether each joint has been assigned to a group.
    assigned = [False] * num_joints

    # Start with the first level: joints with parent -1 (root joints)
    current_level = [i for i in range(num_joints) if parent_indices[i] < 0]

    while current_level:
        # Add the current level of independent joints and their parents to the result
        grouped_joints_indices.append(current_level)
        grouped_joints_parents.append([parent_indices[i] for i in current_level])

        # Mark joints in the current level as assigned
        for joint in current_level:
            assigned[joint] = True

        # Find the next level: joints whose parent is in the current level and not yet assigned
        next_level = []
        for i in range(num_joints):
            if not assigned[i] and parent_indices[i] in current_level:
                next_level.append(i)

        # Move to the next level
        current_level = next_level

    assert all(assigned)
    return grouped_joints_indices, grouped_joints_parents


def identity_rotation_like(translation):
    d = translation.shape[-1]
    return torch.eye(d, dtype=translation.dtype, device=translation.device)[
        [None] * len(translation.shape[:-1])
    ]


def forward_kinematic(bone_parents, rest_bone_poses, delta_transforms):
    """
    Args:
        - bone_parents: torch.Tensor of shape [B]
        - rest_bone_poses: torch.Tensor of shape [bs,B,4,4]
        - delta_transforms: torch.Tensor of shape [bs,B,4,4]
        - mode: string
    Return:
        - poses: [bs,B,4,4]
        - transforms: [bs,B,4,4]
    """
    # Pose of each bone
    poses = torch.empty_like(rest_bone_poses)
    # Relative pose of each bone with respect to its rest pose
    transforms = torch.empty_like(rest_bone_poses)

    bs, B = rest_bone_poses.shape[:2]

    for bone_id in range(len(bone_parents)):
        delta = delta_transforms[:, bone_id]
        rest_pose = rest_bone_poses[:, bone_id]
        parent_id = bone_parents[bone_id]
        T = rest_pose @ delta

        if parent_id >= 0:
            pose = transforms[:, parent_id] @ T
        else:
            pose = T
        transform = pose @ rest_pose.inverse()
        poses[:, bone_id] = pose
        transforms[:, bone_id] = transform
    return poses, transforms


def forward_kinematic_absolute_orientations(
    bone_parents, rest_bone_poses, absolute_orientations
):
    """
    Args:
        - bone_parents: torch.Tensor of shape [B]
        - rest_bone_poses: torch.Tensor of shape [bs,B,4,4]
        - absolute_orientations: list of ([bs,3,3] rotation matrices or None)
        - mode: string
    Return:
        - poses: [bs,B,4,4]
        - transforms: [bs,B,4,4]
    """
    # Pose of each bone
    poses = torch.empty_like(rest_bone_poses)
    # Relative pose of each bone with respect to its rest pose
    transforms = torch.empty_like(rest_bone_poses)

    bs, B = rest_bone_poses.shape[:2]

    for bone_id in range(len(bone_parents)):
        absolute_orientation = absolute_orientations[bone_id]
        rest_pose = rest_bone_poses[:, bone_id]
        parent_id = bone_parents[bone_id]
        if parent_id >= 0:
            pose = transforms[:, parent_id] @ rest_pose
        else:
            pose = (
                rest_pose.clone()
            )  # Use clone to avoid in-place modification of rest_pose
        if absolute_orientation is not None:
            pose[:, :3, :3] = absolute_orientation
        transform = pose @ rest_pose.inverse()
        poses[:, bone_id] = pose
        transforms[:, bone_id] = transform
    return poses, transforms


def forward_kinematic_v2(bone_parents, rest_bone_poses, delta_transforms):
    """
    Args:
        - bone_parents: torch.Tensor of shape [B]
        - rest_bone_poses: torch.Tensor of shape [bs,B,4,4]
        - delta_transforms: torch.Tensor of shape [bs,B,4,4]
        - mode: string
    Return:
        - poses: [bs,B,4,4]
        - transforms: [bs,B,4,4]
    """

    bs, B, _, _ = rest_bone_poses.shape

    T = rest_bone_poses @ delta_transforms

    bone_rest_poses_inv = rest_bone_poses.inverse()

    poses, transforms = [], []
    for bone_id in range(len(bone_parents)):
        parent_id = bone_parents[bone_id]
        rest_pose_inv = bone_rest_poses_inv[:, bone_id]

        if parent_id >= 0:
            pose = torch.matmul(transforms[parent_id], T[:, bone_id])
        else:
            pose = T[:, bone_id]

        transform = torch.matmul(pose, rest_pose_inv)

        transforms.append(transform)
        poses.append(pose)

    return torch.stack(poses, 1), torch.stack(transforms, 1)


def parallel_forward_kinematic(
    kinematic_propagation_fronts, rest_bone_poses, delta_transforms, base_transform=None
):
    """
    Args:
        - kinematic_propagation_fronts: output of get_kinematic_propagation_fronts
        - rest_bone_poses: torch.Tensor of shape [bs,B,4,4]
        - delta_transforms: torch.Tensor of shape [bs,B,4,4]
        - mode: string
    Return:
        - poses: [bs,B,4,4]
        - transforms: [bs,B,4,4]
    """

    l_grouped_indices, l_grouped_parents = kinematic_propagation_fronts

    bs, B, _, _ = rest_bone_poses.shape

    T = rest_bone_poses @ delta_transforms

    bone_rest_poses_inv = (
        roma.Rigid.from_homogeneous(rest_bone_poses).inverse().to_homogeneous()
    )

    # Preallocate tensors for poses and transforms with the same shape as rest_bone_poses
    poses = torch.empty_like(rest_bone_poses)
    transforms = torch.empty_like(rest_bone_poses)

    for i in range(len(l_grouped_indices)):
        grouped_joints_indices = l_grouped_indices[i]
        grouped_joints_parents = l_grouped_parents[i]

        # Handle bones with no parents first
        for bone_id, parent_id in zip(grouped_joints_indices, grouped_joints_parents):
            if parent_id == -1:  # Root bones
                root_poses = (
                    T[:, bone_id]
                    if (base_transform is None)
                    else base_transform @ T[:, bone_id]
                )
                poses[:, bone_id] = root_poses
                transforms[:, bone_id] = torch.einsum(
                    "bij,bjk->bik", root_poses, bone_rest_poses_inv[:, bone_id]
                )

        # Now handle bones with parents
        children_and_parents = [
            (child, parent)
            for child, parent in zip(grouped_joints_indices, grouped_joints_parents)
            if parent >= 0
        ]
        if len(children_and_parents) > 0:
            children, parents = zip(*children_and_parents)
            children = torch.tensor(children, dtype=torch.long, device=T.device)
            parents = torch.tensor(parents, dtype=torch.long, device=T.device)
            children_poses = torch.einsum(
                "blij,bljk->blik", transforms[:, parents], T[:, children]
            )
            poses[:, children] = children_poses
            transforms[:, children] = torch.einsum(
                "blij,bljk->blik", children_poses, bone_rest_poses_inv[:, children]
            )
    return poses, transforms


def parallel_forward_kinematic_absolute_orientations(
    kinematic_propagation_fronts,
    rest_bone_poses,
    absolute_orientations,
    base_transform=None,
):
    """
    Args:
        - kinematic_propagation_fronts: output of get_kinematic_propagation_fronts
        - rest_bone_poses: torch.Tensor of shape [bs,B,4,4]
        - absolute_orientations: torch.Tensor of shape [bs,B,3,3]
    Return:
        - poses: [bs,B,4,4]
        - transforms: [bs,B,4,4]
    """

    l_grouped_indices, l_grouped_parents = kinematic_propagation_fronts

    bs, B, _, _ = rest_bone_poses.shape

    T = rest_bone_poses

    bone_rest_poses_inv = (
        roma.Rigid.from_homogeneous(rest_bone_poses).inverse().to_homogeneous()
    )

    # Preallocate tensors for poses and transforms with the same shape as rest_bone_poses
    poses = torch.empty_like(rest_bone_poses)
    transforms = torch.empty_like(rest_bone_poses)

    for i in range(len(l_grouped_indices)):
        grouped_joints_indices = l_grouped_indices[i]
        grouped_joints_parents = l_grouped_parents[i]

        # Handle bones with no parents first
        for bone_id, parent_id in zip(grouped_joints_indices, grouped_joints_parents):
            if parent_id == -1:  # Root bones
                root_poses = (
                    T[:, bone_id]
                    if (base_transform is None)
                    else base_transform @ T[:, bone_id]
                )
                poses[:, bone_id] = root_poses
                poses[:, bone_id, :3, :3] = absolute_orientations[:, bone_id]
                transforms[:, bone_id] = torch.einsum(
                    "bij,bjk->bik", root_poses, bone_rest_poses_inv[:, bone_id]
                )

        # Now handle bones with parents
        children_and_parents = [
            (child, parent)
            for child, parent in zip(grouped_joints_indices, grouped_joints_parents)
            if parent >= 0
        ]
        if len(children_and_parents) > 0:
            children, parents = zip(*children_and_parents)
            children = torch.tensor(children, dtype=torch.long, device=T.device)
            parents = torch.tensor(parents, dtype=torch.long, device=T.device)
            children_poses = torch.einsum(
                "blij,bljk->blik", transforms[:, parents], T[:, children]
            )
            children_poses[:, :, :3, :3] = absolute_orientations[:, children, :, :]
            poses[:, children] = children_poses
            transforms[:, children] = torch.einsum(
                "blij,bljk->blik", children_poses, bone_rest_poses_inv[:, children]
            )
    return poses, transforms


def get_bone_poses(
    bone_heads, bone_tails, bone_rolls_rotmat, y_axis, degenerate_rotation, epsilon=0.1
):
    """
    Return pose of bones specified by head and tail coordinates, as well as some 'roll' parameter around the bone axis.
    Orient bones consistently with Blender: the y axis is aligned with the head-tail direction by the rotation of smallest angle possible.

    Args:
        - bone_heads: torch.Tensor (B,V,3)
        - bone_tails: torch.Tensor (B,V,3)
        - bone_rolls: tensor of rotation matrices (B,V,3,3)
    Return:
        - rest_bone_poses: homogeneous matrix - torch.Tensor (B,V,4,4)
    """
    # Compute vectors from head to tail
    vectors = bone_tails - bone_heads
    y = vectors / torch.linalg.norm(vectors, dim=-1, keepdim=True)
    cross_p = torch.linalg.cross(y, y_axis[None, None, :])
    dot_p = torch.sum(y * y_axis, dim=-1)
    cross_p_norm = torch.linalg.norm(cross_p, dim=-1)
    angle = torch.atan2(cross_p_norm, dot_p)

    ## Special case: cross_p_norm == 0 will produce NaN values and an undefined rotation axis.
    ## When dot_p > 0, angle==0 and R should be the identity rotation, thus the rotation axis does not matter.
    ## When dot_p < 0, angle==\pi and one should choose a rotation of 180 degrees around some axis.
    ## This case occurs at least for the 'tongue02' bone in a special body shape configuration.
    # Considering -X as rotation axis ensures continuity in bone orientation for this degenerate configuration,
    # but it may not be the general case.
    ## We detect and manually fix this degenerate case.
    axis = (cross_p / cross_p_norm.unsqueeze(-1)).clone()
    R = roma.rotvec_to_rotmat(-angle.unsqueeze(-1) * axis)
    with torch.no_grad():
        is_valid = (torch.abs(torch.sum(torch.square(axis), dim=-1) - 1) < epsilon)[
            ..., None, None
        ].expand_as(R)
    R = torch.where(is_valid, R, degenerate_rotation[None, None, :, :].expand_as(R))

    # Apply bone roll correction and finalize transformation
    R = R @ bone_rolls_rotmat
    H = torch.empty(R.shape[:-2] + (4, 4), device=R.device, dtype=R.dtype)
    H[..., :3, :3] = R
    H[..., :3, 3] = bone_heads
    H[..., 3, :3] = 0.0
    H[..., 3, 3] = 1.0
    return H
