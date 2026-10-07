# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0

import unittest
import torch
import anny.utils.kinematics as kinematics
import roma


class TestBonePoses(unittest.TestCase):
    def bone_inputs(self):
        generator = torch.Generator().manual_seed(0)
        dtype = torch.float64
        up = torch.tensor([0.0, 1.0, 0.0], dtype=dtype)
        heads = torch.randn((2, 4, 3), generator=generator, dtype=dtype)
        offsets = up.expand(2, 4, 3).clone()
        offsets[:, 3] = torch.tensor([0.5, 1.0, -0.3], dtype=dtype)
        rolls = roma.random_rotmat((2, 4), dtype=dtype)
        return heads, offsets, rolls, up, torch.eye(3, dtype=dtype)

    def test_degenerate_bones_have_finite_gradients(self):
        heads, offsets, rolls, up, identity = self.bone_inputs()
        heads.requires_grad_()
        offsets.requires_grad_()
        poses = kinematics.get_bone_poses(heads, heads + offsets, rolls, up, identity)
        poses.sum().backward()
        self.assertTrue(torch.all(torch.isfinite(heads.grad)))
        self.assertTrue(torch.all(torch.isfinite(offsets.grad)))

    def test_gradcheck_with_degenerate_bones(self):
        heads, offsets, rolls, up, identity = self.bone_inputs()
        heads.requires_grad_()
        self.assertTrue(
            torch.autograd.gradcheck(
                lambda h: kinematics.get_bone_poses(
                    h, h + offsets, rolls, up, identity
                ),
                (heads,),
            )
        )


class TestKinematics(unittest.TestCase):
    def test_forward_kinematics(self):
        parent_indices = [-1, 0, 0, 1, 2, 1, 2]
        batch_size = 10
        dtype = torch.float64

        n = len(parent_indices)
        rest_bone_poses = roma.Rigid(
            roma.random_rotmat((batch_size, n), dtype=dtype),
            torch.randn((batch_size, n, 3), dtype=dtype),
        ).to_homogeneous()
        delta_transforms = roma.Rigid(
            roma.random_rotmat((batch_size, n), dtype=dtype),
            torch.randn((batch_size, n, 3), dtype=dtype),
        ).to_homogeneous()

        poses1, transforms1 = kinematics.forward_kinematic(
            parent_indices, rest_bone_poses, delta_transforms
        )

        propagation_fronts = kinematics.get_kinematic_propagation_fronts(parent_indices)
        poses2, transforms2 = kinematics.parallel_forward_kinematic(
            propagation_fronts, rest_bone_poses, delta_transforms
        )

        epsilon = 1e-7
        self.assertTrue(torch.all(torch.abs(poses1 - poses2) < epsilon))
        self.assertTrue(torch.all(torch.abs(transforms1 - transforms2) < epsilon))
