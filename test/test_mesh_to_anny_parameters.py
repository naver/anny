# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
import unittest
import torch
import roma
import anny
from anny.shape_distribution import SimpleShapeDistribution


class TestParametersRegressor(unittest.TestCase):
    def test_fit_synthetic_mesh_roundtrip(self):
        torch.manual_seed(0)

        dtype = torch.float32
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batch_size = 2

        rig = "anny"
        model = anny.Anny(rig=rig, local_changes="default", facial_actions="all").to(
            dtype=dtype, device=device
        )

        pose_parameters = {}
        for i, bone in enumerate(model.bone_labels):
            rotvec = 0.15 * torch.randn(batch_size, 3, dtype=dtype, device=device)
            rotmat = roma.rotvec_to_rotmat(rotvec)
            translation = (
                torch.randn(batch_size, 3, dtype=dtype, device=device)
                if i == 0
                else None
            )
            pose_parameters[bone] = roma.Rigid(linear=rotmat, translation=translation)

        shape_dist = SimpleShapeDistribution(
            model,
            morphological_age_distribution=torch.distributions.Uniform(
                low=torch.tensor(20.0, dtype=dtype, device=device),
                high=torch.tensor(90.0, dtype=dtype, device=device),
            ),
        )
        _, phenotype_gt = shape_dist.sample(batch_size)

        with torch.no_grad():
            target = model(
                pose_parameters=pose_parameters,
                phenotype_kwargs=phenotype_gt,
            )["vertices"]

        initial_phenotype_kwargs = {}
        initial_phenotype_kwargs["age"] = torch.full(
            (batch_size,), 0.8, dtype=dtype, device=device
        )  # assuming it is always adults

        fitter = anny.AnnyInverter(
            model=model,
            verbose=True,
        )

        pose, shape, vertices_hat = fitter(
            vertices_target=target,
            initial_phenotype_kwargs=initial_phenotype_kwargs,
            optimize_phenotypes=True,
            excluded_phenotypes=["age"],
            post_gd=True,
            post_gd_steps=20,
            post_gd_optimize_local_changes=False,
            post_gd_optimize_facial_actions=False,
            multistart_anchors={"muscle": [0.01, 0.5, 0.99]},
        )

        pve = torch.norm(vertices_hat - target, dim=-1).mean()

        print(f"Mean PVE: {1000.0 * pve.item():.2f} mm")

        self.assertLess(
            pve.item(),
            0.01,
            f"Mean PVE too high: {1000.0 * pve.item():.2f} mm",
        )

        # self.assertTrue(
        #     torch.allclose(shape[0]["age"], initial_phenotype_kwargs["age"], atol=1e-5)
        # )
