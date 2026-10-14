# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
import unittest

import roma
import torch

import anny
from anny.models import build_model_data
from anny.models.model_data import RigConfig, TopologyConfig, resolve_phenotypes
from anny.models.model_transforms import compact_phenotype_blendshapes


def _random_inputs(model, batch_size=8, seed=0, phenotype_range=(0.0, 1.0)):
    generator = torch.Generator().manual_seed(seed)
    low, high = phenotype_range
    kwargs = {
        "phenotype_kwargs": low
        + (high - low)
        * torch.rand(batch_size, len(model.phenotype_labels), generator=generator).to(
            model.dtype
        )
    }
    if model.local_change_labels:
        kwargs["local_changes_kwargs"] = (
            2
            * torch.rand(
                batch_size, len(model.local_change_labels), generator=generator
            ).to(model.dtype)
            - 1
        )
    if model.facial_action_labels:
        kwargs["facial_actions"] = torch.rand(
            batch_size, len(model.facial_action_labels), generator=generator
        ).to(model.dtype)
    rotvecs = 0.3 * torch.randn(
        batch_size, model.bone_count, 3, generator=generator
    ).to(model.dtype)
    kwargs["pose_parameters"] = {
        label: roma.Rigid(
            linear=roma.rotvec_to_rotmat(rotvecs[:, i]),
            translation=torch.zeros(batch_size, 3, dtype=model.dtype),
        ).to_homogeneous()
        for i, label in enumerate(model.bone_labels)
    }
    return kwargs


def _assert_same_outputs(test, reference, model, atol=1e-9, **kwargs):
    expected = reference(**kwargs)
    actual = model(**kwargs)
    for key in ("vertices", "bone_poses", "rest_vertices", "rest_bone_heads"):
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=atol)


class TestCompactBlendshapes(unittest.TestCase):
    def test_reproduces_original_model(self):
        configurations = [
            dict(),
            dict(local_changes="default", facial_actions="all"),
            dict(rig="makehuman", topology="makehuman"),
            dict(phenotypes="all"),
            dict(rig="anny-hand.R", topology="hand.R"),
            dict(rig="soma", topology="soma"),
        ]
        for config in configurations:
            with self.subTest(**config):
                reference = anny.Anny(phenotype_blendshapes="original", **config)
                model = anny.Anny(phenotype_blendshapes="compact", **config)
                self.assertLess(
                    model.blendshapes.shape[0], reference.blendshapes.shape[0]
                )
                # The template is the reference of the compact basis and is kept as is.
                torch.testing.assert_close(
                    model.template_vertices, reference.template_vertices, rtol=0, atol=0
                )
                self.assertEqual(model.config.phenotype_blendshapes, "compact")
                self.assertEqual(
                    model.local_change_labels, reference.local_change_labels
                )
                self.assertEqual(
                    model.facial_action_labels, reference.facial_action_labels
                )
                _assert_same_outputs(self, reference, model, **_random_inputs(model))

    def test_default_model_rank(self):
        model = anny.Anny(phenotype_blendshapes="compact")
        self.assertEqual(model.blendshapes.shape[0], 136)
        self.assertEqual(model.phenotype_blendshape_projection.shape, (624, 136))
        self.assertEqual(model.stacked_phenotype_blend_shapes_mask.shape[0], 624)
        self.assertTrue(
            all(
                label.startswith("phenotype_basis:")
                for label in model.blendshape_labels
            )
        )

    def test_extrapolation(self):
        reference = anny.Anny(
            phenotype_blendshapes="original", extrapolate_phenotypes=True
        )
        model = anny.Anny(phenotype_blendshapes="compact", extrapolate_phenotypes=True)
        _assert_same_outputs(
            self,
            reference,
            model,
            **_random_inputs(model, phenotype_range=(-0.5, 1.5)),
        )

    def test_model_data_roundtrip(self):
        model = anny.Anny(phenotype_blendshapes="compact")
        data = model.to_model_data()
        self.assertEqual(data.metadata.phenotype_labels, model.phenotype_labels)
        restored = anny.Anny.from_model_data(
            data,
            pose_parameterization=model.pose_parameterization,
            bone_orientation=model.bone_orientation,
            root_identity_orientation=model.root_identity_orientation,
        )
        _assert_same_outputs(self, model, restored, **_random_inputs(model))

    def test_phenotypes_default_to_metadata(self):
        # Unrestricted data: None resolves to the default phenotypes.
        plain = anny.Anny(phenotype_blendshapes="original").to_model_data()
        self.assertIsNone(plain.metadata.phenotype_labels)
        model = anny.Anny.from_model_data(plain, bone_orientation="cached")
        self.assertEqual(model.phenotype_labels, resolve_phenotypes("default"))
        # Compacted data for all phenotypes: None resolves to them.
        data = anny.Anny(
            phenotype_blendshapes="compact", phenotypes="all"
        ).to_model_data()
        model = anny.Anny.from_model_data(data, bone_orientation="cached")
        self.assertEqual(model.phenotype_labels, resolve_phenotypes("all"))
        # An explicit matching value is accepted too.
        anny.Anny.from_model_data(data, bone_orientation="cached", phenotypes="all")

    def test_phenotype_mismatch_raises(self):
        data = anny.Anny(phenotype_blendshapes="compact").to_model_data()
        with self.assertRaisesRegex(ValueError, "restricted to the phenotypes"):
            anny.Anny.from_model_data(
                data,
                bone_orientation="cached",
                phenotypes="all",
            )

    def test_transform_arguments(self):
        data = build_model_data(
            rig=RigConfig.from_string("anny"),
            topology=TopologyConfig.from_string("anny"),
            local_changes="none",
            facial_actions="none",
        )
        compacted = compact_phenotype_blendshapes(data)
        with self.assertRaisesRegex(ValueError, "already restricted"):
            compact_phenotype_blendshapes(compacted)
        with self.assertRaisesRegex(ValueError, "numerical rank"):
            compact_phenotype_blendshapes(data, rank=1000)

    def test_rank_truncation(self):
        reference = anny.Anny(phenotype_blendshapes="original")
        model = anny.Anny(phenotype_blendshapes=20)
        self.assertEqual(model.config.phenotype_blendshapes, 20)
        self.assertEqual(model.blendshapes.shape[0], 20)
        # Same result through the transform.
        data = compact_phenotype_blendshapes(reference.to_model_data(), rank=20)
        torch.testing.assert_close(data.blendshapes, model.blendshapes)
        inputs = _random_inputs(model)
        error = (model(**inputs)["vertices"] - reference(**inputs)["vertices"]).norm(
            dim=-1
        )
        self.assertTrue(torch.isfinite(error).all())
        # An approximation: close, but no longer exact.
        self.assertLess(error.max().item(), 0.1)
        self.assertGreater(error.max().item(), 1e-6)

    def test_compact_is_default(self):
        model = anny.Anny()
        self.assertEqual(model.config.phenotype_blendshapes, "compact")
        self.assertIsNotNone(model.phenotype_blendshape_projection)
        self.assertEqual(model.blendshapes.shape[0], 136)

    def test_original(self):
        model = anny.Anny(phenotype_blendshapes="original")
        self.assertEqual(model.config.phenotype_blendshapes, "original")
        self.assertIsNone(model.phenotype_blendshape_projection)
        self.assertEqual(model.blendshapes.shape[0], 624)

    def test_invalid_phenotype_blendshapes_argument(self):
        for value in [0, -3, 2.5, True, "all", "exact"]:
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "phenotype_blendshapes"):
                    anny.Anny(phenotype_blendshapes=value)


if __name__ == "__main__":
    unittest.main()
