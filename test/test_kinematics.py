# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0

import unittest
import torch
import anny.utils.kinematics as kinematics
import roma

class TestKinematics(unittest.TestCase):
    def test_forward_kinematics(self):
        parent_indices = [-1, 0, 0, 1, 2, 1, 2]
        batch_size = 10
        dtype = torch.float64

        n = len(parent_indices)
        rest_bone_poses = roma.Rigid(roma.random_rotmat((batch_size, n), dtype=dtype), torch.randn((batch_size, n, 3), dtype=dtype)).to_homogeneous()
        delta_transforms = roma.Rigid(roma.random_rotmat((batch_size, n), dtype=dtype), torch.randn((batch_size, n, 3), dtype=dtype)).to_homogeneous()

        poses1, transforms1 = kinematics.forward_kinematic(parent_indices, rest_bone_poses, delta_transforms)

        propagation_fronts = kinematics.get_kinematic_propagation_fronts(parent_indices)
        poses2, transforms2 = kinematics.parallel_forward_kinematic(propagation_fronts, rest_bone_poses, delta_transforms)

        epsilon = 1e-7
        self.assertTrue(torch.all(torch.abs(poses1 - poses2) < epsilon))
        self.assertTrue(torch.all(torch.abs(transforms1 - transforms2) < epsilon))