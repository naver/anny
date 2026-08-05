# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
"""Smoke test for the deprecated legacy construction syntax.

Old user code combining the deprecated ``create_fullbody_model`` factory with
legacy argument spellings (boolean ``local_changes``, the ``'root_relative'``
pose parameterization, the ``'notoes_collapse5pc'`` topology alias) must keep
building a working model, only emitting deprecation warnings.
"""

import unittest
import warnings

import torch

import anny


class TestLegacySyntax(unittest.TestCase):
    def test_legacy_fullbody_construction_runs(self):
        for topology in ["notoes_collapse5pc", "default"]:
            with self.subTest(topology=topology):
                with warnings.catch_warnings():
                    warnings.simplefilter("error")
                    warnings.simplefilter("ignore", DeprecationWarning)
                    model = anny.create_fullbody_model(
                        local_changes=True,
                        pose_parameterization="root_relative",
                        remove_unattached_vertices=False,
                        all_phenotypes=False,
                        topology=topology,
                    ).to(dtype=torch.float32)

                output = model()
                self.assertIn("vertices", output)
                self.assertIn("bone_poses", output)
                self.assertEqual(output["vertices"].shape[-1], 3)
                self.assertTrue(torch.isfinite(output["vertices"]).all())

    def test_legacy_fullbody_construction_warns(self):
        for topology in ["notoes_collapse5pc", "default"]:
            with self.subTest(topology=topology):
                with self.assertWarns(DeprecationWarning):
                    anny.create_fullbody_model(
                        local_changes=True,
                        pose_parameterization="root_relative",
                        remove_unattached_vertices=False,
                        all_phenotypes=False,
                        topology=topology,
                    )


if __name__ == "__main__":
    unittest.main()
