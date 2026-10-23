# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
import torch
import roma


def get_bone_poses(
    bone_heads, bone_tails, bone_rolls_rotmat, y_axis, degenerate_rotation, epsilon=0.1
):
    """
    Return pose of bones specified by head and tail coordinates, as well as some 'roll' parameter around the bone axis.
    Orient bones consistently with Blender: the y axis is aligned with the head-tail
    direction by the rotation of smallest angle possible.

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

    ## Special case: cross_p_norm == 0 gives an undefined rotation axis.
    ## When dot_p > 0, angle==0 and R should be the identity rotation, thus the rotation axis does not matter.
    ## When dot_p < 0, angle==\pi and one should choose a rotation of 180 degrees around some axis.
    ## This case occurs at least for the 'tongue02' bone in a special body shape configuration.
    # Considering -X as rotation axis ensures continuity in bone orientation for this degenerate configuration,
    # but it may not be the general case.
    ## We detect and manually fix this degenerate case.
    ## Where cross_p_norm == 0 the axis is undefined: the norm is replaced by 1 there so that the
    ## backward pass stays finite (the value is discarded by the degenerate-case selection below).
    with torch.no_grad():
        has_axis = cross_p_norm > 0
    safe_norm = torch.where(has_axis, cross_p_norm, torch.ones_like(cross_p_norm))
    axis = (cross_p / safe_norm.unsqueeze(-1)).clone()
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


# 3x3 and 4x4 products as broadcast sums: on a V100, torch.matmul was slower in the chain product below
# (1.9 vs 1.6 ms for one state, 3.8 vs 1.6 ms for 1024).
def _apply(r, v):
    return (r * v[..., None, :]).sum(-1)


def _mm(a, b):
    return (a[..., :, :, None] * b[..., None, :, :]).sum(-2)


def get_kinematic_chain(bone_parents):
    """
    The bones in depth order, for forward_kinematic: the root (a single one, at index 0), then its children, then
    theirs, and so on, so that each depth level is a contiguous slice whose parents all lie in the level before it.

    Args:
        - bone_parents: list of B parent indices, the root's being negative
    Return:
        - levels: list of the (first, last + 1) positions of each depth level
        - indices: LongTensor (4, B): each bone's parent's position within the level before it (root: 0),
          the depth order, its inverse, and each bone's parent (root: itself)
    """
    bone_parents = [int(parent) for parent in bone_parents]
    assert bone_parents[0] < 0 and min(bone_parents[1:], default=0) >= 0, (
        "Expected a single root bone, at index 0."
    )
    children = [[] for _ in bone_parents]
    for bone, parent in enumerate(bone_parents[1:], start=1):
        children[parent].append(bone)
    levels, order, local = [], [0], [0]
    first = 0
    while first < len(order):
        last = len(order)
        levels.append((first, last))
        for position in range(first, last):
            for child in children[order[position]]:
                order.append(child)
                local.append(position - first)
        first = last
    assert len(order) == len(bone_parents), (
        "Expected every bone to descend from the root."
    )
    order = torch.tensor(order)
    indices = torch.stack(
        [
            torch.tensor(local),
            order,
            torch.argsort(order),
            torch.tensor([0] + bone_parents[1:]),
        ]
    )
    return levels, indices


def forward_kinematic(
    kinematic_chain,
    rest_bone_poses,
    delta_transforms,
    pose_parameterization,
    reference_orientations=None,
):
    """
    Forward kinematics: the posed bone poses for delta transforms in a pose parameterization, and the bone transforms
    that map the rest pose to them.

    Notation, for a bone c of parent p, [X | t] being the rigid transform x -> X x + t:
        - P_c = [R_c | h_c]: the rest pose, h_c being the bone's head.
        - Q_c = [U_c | q_c]: the reference pose, P_c if None. Each bone takes its reference orientation U_c, rotating
          its descendants' heads with it about its own head, except the root, whose rotation leaves the skeleton in
          place.
        - [D_c | d_c]: the delta transform. Only its top 3x4 block is read.
        - W_c: the posed pose, returned as poses.
        - T_c = W_c P_c^-1 is returned as transforms (for skinning).

    The delta transforms in each pose parameterization:
        - 'local-bone': W_c = W_p (Q_p^-1 Q_c) D_c. The delta is the bone's pose relative to its reference pose, in
          its reference frame. The root's delta is its world pose, W_0 = D_0.
        - 'local-bone-world': like 'local-bone', but the root's delta is relative to its reference pose too,
          W_0 = Q_0 D_0. Identity deltas then give the reference poses Q_c.
        - 'local-ref': like 'local-bone', but each delta is expressed in world-aligned axes rather than in the bone's
          reference axes: D_c is replaced by [U_c^T | 0] D_c [U_c | 0], the root's included. This is 'local-bone'
          when the reference orientations are I.
        - 'world-orient': D_c is the bone's world orientation, and its head follows its parent: W_c = [D_c | T_p h_c].
          d_c is ignored except for the root, whose delta is its world pose, W_0 = [D_0 | d_0].
        - 'world': the deltas are the posed poses, W_c = D_c.

    Implementation: a chain X_c = X_p K_c on the poses W, or on the transforms T for 'world-orient', with the K_c
    computed for every bone at once, so the chain takes one batched product per depth level. Rotations are inverted
    by transposition.

    Args:
        - kinematic_chain: output of get_kinematic_chain
        - rest_bone_poses: torch.Tensor [..., B, 4, 4]
        - delta_transforms: torch.Tensor [..., B, 4, 4]
        - pose_parameterization: 'local-bone', 'local-bone-world', 'local-ref', 'world-orient' or 'world'
        - reference_orientations: torch.Tensor [B, 3, 3], or None for the rest orientations
    Return:
        - transforms: [..., B, 4, 4]
        - poses: [..., B, 4, 4]
    """
    levels, (local, order, inverse, parent) = kinematic_chain
    R, h = rest_bone_poses[..., :3, :3], rest_bone_poses[..., :3, 3]
    D, d = delta_transforms[..., :3, :3], delta_transforms[..., :3, 3]
    if pose_parameterization == "world-orient":
        # W_c = [D_c | T_p h_c] gives, with G_c = D_c R_c^T, T_c = [G_c | T_p h_c - G_c h_c].
        # We can define K_c = T_p^-1 T_c = [G_p^T G_c | h_c - G_p^T G_c h_c] to compute
        # the chain T_c = T_p K_c for every bone.
        # The root's T_0 = [D_0 | d_0] P_0^-1 = [G_0 | d_0 - G_0 h_0].
        G = _mm(D, R.transpose(-1, -2))
        K_rotations = _mm(G[..., parent, :, :].transpose(-1, -2), G)
        K = _rigid(K_rotations, h - _apply(K_rotations, h))
        G0 = G[..., 0, :, :]
        T0 = _rigid(G0, d[..., 0, :] - _apply(G0, h[..., 0, :]))
        T = _chain_product(levels, local, order, inverse, T0, K)
        return T, _mm(T, rest_bone_poses)

    if pose_parameterization == "world":
        W = delta_transforms
    elif pose_parameterization in ("local-ref", "local-bone", "local-bone-world"):
        # The reference head offsets q_offsets = q_c - q_p = A_p (h_c - h_p),
        # with A_p = U_p R_p^T the parent's rotation
        # from its rest to its reference orientation, and A_0 taken as I.
        h_offsets = h - h[..., parent, :]
        if reference_orientations is None:
            U, q_offsets = R, h_offsets
        else:
            U = reference_orientations
            eye = torch.eye(3, dtype=R.dtype, device=R.device).expand_as(
                R[..., :1, :, :]
            )
            A = _mm(U[..., 1:, :, :], R[..., 1:, :, :].transpose(-1, -2))
            A = torch.cat([eye, A], -3)
            q_offsets = _apply(A[..., parent, :, :], h_offsets)
        # W_c = W_p K_c with K_c = [U_p^T | 0] [I | q_c - q_p] [M_c | m_c] = [U_p^T M_c | U_p^T (q_c - q_p + m_c)],
        # where:
        # [M_c | m_c] = [D_c | d_c] [U_c | 0] for 'local-ref',
        # [M_c | m_c] = [U_c | 0] [D_c | d_c] for 'local-bone-*'.
        if pose_parameterization == "local-ref":
            M, m = _mm(D, U), d
        else:
            M, m = _mm(U, D), _apply(U, d)
        U_pt = U[..., parent, :, :].transpose(-1, -2)
        K = _rigid(_mm(U_pt, M), _apply(U_pt, m + q_offsets))
        M0, m0 = M[..., 0, :, :], m[..., 0, :]
        if pose_parameterization == "local-ref":
            # W_0 = [U_0^T | 0] [D_0 | d_0] [U_0 | 0]
            U0_t = U[..., 0, :, :].transpose(-1, -2)
            W0 = _rigid(_mm(U0_t, M0), _apply(U0_t, m0))
        elif pose_parameterization == "local-bone":
            # W_0 = [D_0 | d_0]
            W0 = _rigid(D[..., 0, :, :], d[..., 0, :])
        else:  # local-bone-world
            # W_0 = Q_0 [D_0 | d_0] = [I | h_0] [U_0 | 0] [D_0 | d_0]
            W0 = _rigid(M0, m0 + h[..., 0, :])
        W = _chain_product(levels, local, order, inverse, W0, K)
    else:
        raise NotImplementedError(
            f"Pose parameterization {pose_parameterization} not implemented"
        )
    # T_c = W_c P_c^-1 = [V_c R_c^T | w_c - V_c R_c^T h_c] for W_c = [V_c | w_c].
    T_rotations = _mm(W[..., :3, :3], R.transpose(-1, -2))
    return _rigid(T_rotations, W[..., :3, 3] - _apply(T_rotations, h)), W


def _rigid(rotation, translation):
    return roma.Rigid(rotation, translation).to_homogeneous()


def _chain_product(levels, local, order, inverse, X0, K):
    """X_c = X_p K_c level by level from X_0 at the root."""
    K = K[..., order, :, :]
    X = [X0[..., None, :, :]]
    for first, last in levels[1:]:
        X_p = X[-1][..., local[first:last], :, :]
        X.append(_mm(X_p, K[..., first:last, :, :]))
    return torch.cat(X, -3)[..., inverse, :, :]
