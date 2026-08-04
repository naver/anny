# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
import unittest

import roma
import torch

import anny

COCO_LABELS = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

# A plausible pose: arms down and bent, one leg lifted. Baked and regressed keypoints diverge with
# the amount of articulation, so a wilder pose would only measure how far apart we let them drift.
POSED_BONE_ROTATION_VECTORS = {
    "upperarm01.L": [0.0, 0.0, -0.9],
    "upperarm01.R": [0.0, 0.0, 0.9],
    "lowerarm01.L": [0.0, -1.1, 0.0],
    "lowerarm01.R": [0.0, 1.1, 0.0],
    "upperleg01.L": [0.5, 0.0, 0.0],
    "lowerleg01.L": [-0.8, 0.0, 0.0],
}


class TestCocoKeypoints(unittest.TestCase):
    dtype = torch.float64
    device = torch.device("cpu")
    batch_size = 3

    @classmethod
    def setUpClass(cls):
        cls.model = anny.Anny().to(dtype=cls.dtype, device=cls.device)
        cls.regressor = anny.KeypointsRegressor.coco(cls.model, COCO_LABELS)
        generator = torch.Generator(device=cls.device).manual_seed(0)
        cls.phenotype_kwargs = {
            label: torch.rand(
                (cls.batch_size,),
                dtype=cls.dtype,
                device=cls.device,
                generator=generator,
            )
            for label in cls.model.phenotype_labels
        }

    def posed_parameters(self):
        pose_parameters = (
            torch.eye(4, dtype=self.dtype, device=self.device)[None, None]
            .expand(self.batch_size, self.model.bone_count, 4, 4)
            .clone()
        )
        for bone_label, rotation_vector in POSED_BONE_ROTATION_VECTORS.items():
            bone_index = self.model.bone_labels.index(bone_label)
            pose_parameters[:, bone_index, :3, :3] = roma.rotvec_to_rotmat(
                torch.tensor(rotation_vector, dtype=self.dtype, device=self.device)
            )
        return pose_parameters

    def test_sparse_regressor_matches_dense_regressor(self):
        # Dropping all but the dominant vertices of each convex combination barely moves it.
        support_size = 64
        weights, indices = torch.topk(
            self.regressor.regression_weights, support_size, dim=1
        )
        weights = weights / weights.sum(dim=1, keepdim=True)
        sparse_regressor = anny.KeypointsRegressor(
            weights, self.regressor.labels, regression_indices=indices
        )
        self.assertTrue(sparse_regressor.is_sparse)
        model_output = self.model(
            pose_parameters=self.posed_parameters(),
            phenotype_kwargs=self.phenotype_kwargs,
        )
        distances = torch.norm(
            self.regressor(model_output) - sparse_regressor(model_output), dim=-1
        )
        self.assertLess(distances.max(), 1e-3)


if __name__ == "__main__":
    unittest.main()
