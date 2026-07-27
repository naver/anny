# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
"""Equivalence between the three ways of constructing the full MakeHuman rig model.

The deprecated legacy factory, an explicit ``RigConfig`` and the ``"makehuman"`` string
spec are meant to be interchangeable entry points into the same model. This test pins that
contract so a future change to the legacy shim, the rig-spec parser or the string->config
defaults cannot silently break it.
"""
import unittest
import warnings

import torch
import roma

import anny
from anny.models.model_data import RigConfig


class TestMakehumanConstructionAliases(unittest.TestCase):
    dtype = torch.float64
    atol = 1e-8

    def _build_models(self):
        # create_fullbody_model is deprecated on purpose here; silence the expected warning.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            model_a = anny.create_fullbody_model(
                rig="default",
                bone_orientation="blender-rootidentity",
                pose_parameterization="local-bone",
                triangulate_faces=True,
            ).to(dtype=self.dtype)
        model_b = anny.Anny(
            rig=RigConfig(
                base_rig="makehuman",
                bone_orientation="blender",
                root_identity_orientation=True,
            ),
            pose_parameterization="local-bone",
        ).to(dtype=self.dtype)
        model_c = anny.Anny(
            rig="makehuman",
            pose_parameterization="local-bone",
        ).to(dtype=self.dtype)
        return model_a, model_b, model_c

    def _assert_close_all(self, key, out_a, out_b, out_c):
        torch.testing.assert_close(
            out_a[key], out_b[key], rtol=0, atol=self.atol,
            msg=f"'{key}' differs between config A (legacy factory) and config B (RigConfig)")
        torch.testing.assert_close(
            out_a[key], out_c[key], rtol=0, atol=self.atol,
            msg=f"'{key}' differs between config A (legacy factory) and config C (string spec)")

    def test_construction_equivalence(self):
        model_a, model_b, model_c = self._build_models()
        models = [model_a, model_b, model_c]

        # --- Structural equivalence -------------------------------------------------
        for model in models:
            self.assertEqual(model.bone_orientation, "blender")
            self.assertIs(model.root_identity_orientation, True)

        for attr in ["bone_count", "phenotype_labels", "blendshape_labels", "bone_labels"]:
            values = [getattr(m, attr) for m in models]
            self.assertEqual(values[0], values[1], f"'{attr}' differs between config A and B")
            self.assertEqual(values[0], values[2], f"'{attr}' differs between config A and C")

        self.assertEqual(model_a.template_vertices.shape, model_b.template_vertices.shape)
        self.assertEqual(model_a.template_vertices.shape, model_c.template_vertices.shape)
        self.assertTrue(torch.equal(model_a.faces, model_b.faces), "faces differ between config A and B")
        self.assertTrue(torch.equal(model_a.faces, model_c.faces), "faces differ between config A and C")

        # --- Static template geometry -----------------------------------------------
        torch.testing.assert_close(
            model_a.template_vertices, model_b.template_vertices, rtol=0, atol=self.atol)
        torch.testing.assert_close(
            model_a.template_vertices, model_c.template_vertices, rtol=0, atol=self.atol)

        # --- Rest-pose forward (default pose and phenotypes) ------------------------
        rest_a, rest_b, rest_c = model_a(), model_b(), model_c()
        for key in ["rest_vertices", "rest_bone_poses", "vertices", "bone_poses"]:
            self._assert_close_all(key, rest_a, rest_b, rest_c)

        # --- Random shape + random pose forward -------------------------------------
        batch_size = 8
        bone_count = model_b.bone_count
        phenotype_labels = model_b.phenotype_labels

        torch.manual_seed(0)
        rots = roma.random_rotmat((batch_size, bone_count), dtype=self.dtype)
        translations = torch.randn((batch_size, bone_count, 3), dtype=self.dtype)
        pose_parameters = roma.Rigid(rots, translations).to_homogeneous()
        phenotype_kwargs = {k: torch.rand(batch_size, dtype=self.dtype) for k in phenotype_labels}

        def run(model):
            return model(
                pose_parameters=pose_parameters,
                phenotype_kwargs=phenotype_kwargs,
            )

        posed_a, posed_b, posed_c = run(model_a), run(model_b), run(model_c)
        for key in ["vertices", "bone_poses"]:
            self._assert_close_all(key, posed_a, posed_b, posed_c)


if __name__ == "__main__":
    unittest.main()
