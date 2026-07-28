# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
"""Optional post-processing of rest bone orientations.

The procrustes rest orientations are estimated independently per bone from their attached vertices.
Some rigs (e.g. SOMA) additionally reorient each bone so that its frame follows the direction of its
child joints, and copy the parent orientation onto end bones. This behaviour is isolated here as a
pluggable :class:`torch.nn.Module` so that :class:`RiggedModelWithLinearBlendShapes` stays generic:
the base model simply applies a refiner when the rig data provides one.
"""

import roma
import torch

from anny.torch_compat import make_buffer


def _shortest_arc_rotation(
    target: torch.Tensor, source: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """Rotation matrices of smallest angle mapping *source* vectors onto *target* vectors (shape (...,3)),
    replicating the convention of the SOMA skeleton fit for single-child joints."""
    dtype, device = target.dtype, target.device
    target_u = target / torch.clamp(
        torch.linalg.norm(target, dim=-1, keepdim=True), min=eps
    )
    source_u = source / torch.clamp(
        torch.linalg.norm(source, dim=-1, keepdim=True), min=eps
    )
    dot = torch.clamp((target_u * source_u).sum(dim=-1, keepdim=True), -1.0, 1.0)
    v = torch.cross(source_u, target_u, dim=-1)
    zeros = torch.zeros_like(v[..., 0])
    skew_v = torch.stack(
        [
            torch.stack([zeros, -v[..., 2], v[..., 1]], dim=-1),
            torch.stack([v[..., 2], zeros, -v[..., 0]], dim=-1),
            torch.stack([-v[..., 1], v[..., 0], zeros], dim=-1),
        ],
        dim=-2,
    )
    eye = torch.eye(3, dtype=dtype, device=device).expand(target.shape[:-1] + (3, 3))
    R = eye + skew_v + (skew_v @ skew_v) / (1.0 + dot[..., None])

    # Antiparallel case: 180 degree rotation around an arbitrary axis orthogonal to the source vector.
    antiparallel_mask = dot[..., 0] < -1.0 + 1e-6
    if torch.any(antiparallel_mask):
        source_anti = source_u[antiparallel_mask]
        basis = torch.zeros_like(source_anti)
        use_y = torch.abs(source_anti[..., 0]) > 0.6
        basis[..., 1] = use_y.to(dtype)
        basis[..., 0] = (~use_y).to(dtype)
        axis = torch.cross(source_anti, basis, dim=-1)
        axis = axis / torch.linalg.norm(axis, dim=-1, keepdim=True)
        R_180 = 2.0 * axis[..., :, None] * axis[..., None, :] - torch.eye(
            3, dtype=dtype, device=device
        )
        R = R.clone()
        R[antiparallel_mask] = R_180
    return R


class ChildOffsetOrientationRefiner(torch.nn.Module):
    """Reorient rest bones to follow their child joints, matching the SOMA skeleton fit.

    Each non-root bone with children gets a secondary rotation aligning its bind-pose child offsets
    onto the current-shape ones (a Kabsch fit for several children, a shortest-arc rotation for a
    single one). End bones (no children) copy their parent's refined orientation.
    """

    def __init__(
        self,
        bone_parents: list[int],
        bone_children_indices: torch.Tensor,
        bone_children_mask: torch.Tensor,
        bone_children_local_offsets: torch.Tensor,
    ) -> None:
        super().__init__()
        bone_count = bone_children_mask.shape[0]
        self.bone_children_indices = make_buffer(
            self, "bone_children_indices", bone_children_indices, persistent=False
        )
        self.bone_children_mask = make_buffer(
            self, "bone_children_mask", bone_children_mask, persistent=False
        )
        self.bone_children_local_offsets = make_buffer(
            self,
            "bone_children_local_offsets",
            bone_children_local_offsets,
            persistent=False,
        )

        # Static index groups. The root is never refined.
        counts = bone_children_mask.sum(dim=-1)
        counts[0] = 0
        leaf_bones = [
            bone_idx
            for bone_idx in range(1, bone_count)
            if bone_idx not in bone_parents
        ]
        self.refined_single_child_bone_indices = make_buffer(
            self,
            "refined_single_child_bone_indices",
            torch.nonzero(counts == 1).squeeze(1),
            persistent=False,
        )
        self.refined_multi_child_bone_indices = make_buffer(
            self,
            "refined_multi_child_bone_indices",
            torch.nonzero(counts >= 2).squeeze(1),
            persistent=False,
        )
        self.leaf_bone_indices = make_buffer(
            self,
            "leaf_bone_indices",
            torch.tensor(leaf_bones, dtype=torch.int64),
            persistent=False,
        )
        self.leaf_bone_parent_indices = make_buffer(
            self,
            "leaf_bone_parent_indices",
            torch.tensor(
                [bone_parents[bone_idx] for bone_idx in leaf_bones], dtype=torch.int64
            ),
            persistent=False,
        )

    def forward(
        self, rest_bone_orientation: torch.Tensor, rest_bone_heads: torch.Tensor
    ) -> torch.Tensor:
        # Offsets towards children bones: current ones, and bind-pose ones rotated by the current orientations.
        target_offsets = (
            rest_bone_heads[:, self.bone_children_indices]
            - rest_bone_heads[:, :, None, :]
        )
        target_offsets = target_offsets * self.bone_children_mask[None, :, :, None]
        source_offsets = torch.einsum(
            "bkij,kcj->bkci", rest_bone_orientation, self.bone_children_local_offsets
        )

        rest_bone_orientation = rest_bone_orientation.clone()

        idx = self.refined_multi_child_bone_indices
        if idx.numel() > 0:
            A = target_offsets[:, idx]
            B = source_offsets[:, idx]
            H = torch.einsum("bkci,bkcj->bkij", A, B)
            # Virtual-normal conditioning from the first two children, as in the SOMA skeleton fit.
            eps = 1e-8
            n_target = torch.cross(A[..., 0, :], A[..., 1, :], dim=-1)
            n_source = torch.cross(B[..., 0, :], B[..., 1, :], dim=-1)
            len_n_target = torch.linalg.norm(n_target, dim=-1, keepdim=True)
            len_n_source = torch.linalg.norm(n_source, dim=-1, keepdim=True)
            v_target = (
                n_target
                * torch.linalg.norm(A[..., 0, :], dim=-1, keepdim=True)
                / (len_n_target + eps)
            )
            v_source = (
                n_source
                * torch.linalg.norm(B[..., 0, :], dim=-1, keepdim=True)
                / (len_n_source + eps)
            )
            valid_normal = (
                (len_n_target[..., 0] > 1e-9) & (len_n_source[..., 0] > 1e-9)
            )[..., None, None]
            H = H + torch.where(
                valid_normal,
                torch.einsum("bki,bkj->bkij", v_target, v_source),
                torch.zeros_like(H),
            )
            align = roma.special_procrustes(H)
            rest_bone_orientation[:, idx] = align @ rest_bone_orientation[:, idx]

        idx = self.refined_single_child_bone_indices
        if idx.numel() > 0:
            align = _shortest_arc_rotation(
                target_offsets[:, idx, 0], source_offsets[:, idx, 0]
            )
            rest_bone_orientation[:, idx] = align @ rest_bone_orientation[:, idx]

        # End bones copy their parent's refined orientation.
        rest_bone_orientation[:, self.leaf_bone_indices] = rest_bone_orientation[
            :, self.leaf_bone_parent_indices
        ]
        return rest_bone_orientation
