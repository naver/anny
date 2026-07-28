# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0

import torch
import anny
import roma
import unittest
import anny.models.rigged_model


class TestPoseParametrization(unittest.TestCase):
    def test_pose_parameterization_conversions(self):
        dtype = torch.float64
        model = anny.Anny().to(dtype=dtype)

        batch_size = 32

        parametrization_list = [
            "world",
            "local-bone-world",
            "local-ref",
            "local-bone",
            "world-orient",
        ]

        for source_pose_parameterization in parametrization_list:
            for target_pose_parameterization in parametrization_list:
                phenotype_kwargs = {
                    key: torch.rand(batch_size, dtype=dtype)
                    for key in model.phenotype_labels
                }
                source_pose_parameters = roma.Rigid(
                    roma.random_rotmat((batch_size, model.bone_count), dtype=dtype),
                    torch.randn((batch_size, model.bone_count, 3), dtype=dtype),
                ).to_homogeneous()
                if (
                    source_pose_parameterization == "world-orient"
                    or target_pose_parameterization == "world-orient"
                ):
                    # No translation except for the root bone
                    source_pose_parameters[:, 1:, :3, 3] = 0.0

                if target_pose_parameterization == "world-orient":
                    if source_pose_parameterization == "world":
                        # the 'world' parametterization allow some non-articulated transformations, which cannot be parameterized by 'world-orient'.
                        continue

                source_output = model(
                    pose_parameters=source_pose_parameters,
                    phenotype_kwargs=phenotype_kwargs,
                    pose_parameterization=source_pose_parameterization,
                )

                target_pose_parameters = model.get_pose_parameterization(
                    source_output, pose_parameterization=target_pose_parameterization
                )
                target_output = model(
                    pose_parameters=target_pose_parameters,
                    phenotype_kwargs=phenotype_kwargs,
                    pose_parameterization=target_pose_parameterization,
                )
                self.assertTrue(
                    torch.allclose(
                        source_output["vertices"], target_output["vertices"], atol=1e-4
                    ),
                    f"Pose parametrization conversion error from {source_pose_parameterization} to {target_pose_parameterization}",
                )
