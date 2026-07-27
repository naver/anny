# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0

import unittest
import torch
import anny
import copy
import numpy as np
import roma

class TestDegenerateConfiguration(unittest.TestCase):
    def test_degenerate_tongue02(self):
        # This particular shape create a bone head-tail direction perfectly aligned with the y axis for 'tongue02'.
        # It is a special case that makes bone orientation a bit ill-defined.
        # This test ensure that we keep orientation continuity around this edge case.
        naughty_shape = {
                'gender': 0.4645,
                'age': 0.6078, 
                'muscle': 0.2637, 
                'weight': 0.7545, 
                'height': 0.5872, 
                'proportions': 0.7788, 
                'cupsize': 0.4095, 
                'firmness': 0.8335, 
                'african': 0.3333, 
                'asian': 0.3333, 
                'caucasian': 0.3333, 
            }
        model = anny.Anny(rig="makehuman").to(dtype=torch.float32)

        def return_tongue_pose(shape):
            shape = {k: torch.Tensor([v]) for k, v in shape.items()}
            blendshape_coeffs = model.get_phenotype_blendshape_coefficients(**shape) # [bs,564]
            rest_model = model.get_rest_model(blendshape_coeffs) # _,_, [bs,163,4,4]
            rest_bone_poses = rest_model["rest_bone_poses"]
            assert not torch.isnan(rest_bone_poses).any()
            tongue02_bone_index = model.bone_labels.index('tongue02')
            return rest_bone_poses[0, tongue02_bone_index]

        naughty_pose = return_tongue_pose(naughty_shape)

        for _ in range(500):
            # Apply small shape perturbation
            devious_shape = dict()
            for k, v in naughty_shape.items():
                devious_shape[k] = v + np.random.uniform(low=-0.0005, high=0.0005)

            devious_pose = return_tongue_pose(devious_shape)
            # print("Naughty pose", naughty_pose)
            # print("Devious pose", devious_pose)
            deviation = torch.linalg.norm(devious_pose - naughty_pose)
            # print("Deviation", deviation)
            self.assertTrue(deviation.item() < 0.002)