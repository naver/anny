# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
import types
import unittest
import torch
from anny.shape_distribution import SimpleShapeDistribution


class TestShapeDistribution(unittest.TestCase):
    def test_conditional_tables_are_indexed_by_anny_age(self):
        model = types.SimpleNamespace(
            dtype=torch.float64,
            device="cpu",
            phenotype_labels=[
                "gender",
                "age",
                "muscle",
                "weight",
                "height",
                "proportions",
            ],
        )
        shape_dist = SimpleShapeDistribution(model)
        # Anny age of an 18 year old, which is an anchor of the weight and muscle tables.
        age = shape_dist.morphological_age_mapping.morphological_to_anny_age(
            torch.tensor([18.0], dtype=torch.float64)
        )
        for sex in ["boys", "girls"]:
            for name in ["weight", "muscle"]:
                dist = getattr(shape_dist, f"{sex}_conditional_{name}_distribution")
                alpha, beta = dist.get_distribution_params(age)
                with self.subTest(sex=sex, name=name):
                    torch.testing.assert_close(alpha, dist.alpha_anchors[5:6])
                    torch.testing.assert_close(beta, dist.beta_anchors[5:6])


if __name__ == "__main__":
    unittest.main()
