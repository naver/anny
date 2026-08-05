# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
"""Smoke test for the deprecated legacy construction syntax.

Old user code combining the deprecated ``create_fullbody_model`` factory with
legacy argument spellings (boolean ``local_changes``, the ``'root_relative'``
and ``'root_relative_world'`` pose parameterizations, the
``'notoes_collapse5pc'`` topology alias, the ``'default-notongue-noeyes'``
rig/topology aliases) must keep building a working model, only emitting
deprecation warnings.
"""

import unittest
import warnings

import torch

import anny

LEGACY_CALL_CASES = [
    dict(
        local_changes=True,
        pose_parameterization="root_relative",
        remove_unattached_vertices=False,
        all_phenotypes=False,
        topology="notoes_collapse5pc",
    ),
    dict(
        local_changes=True,
        pose_parameterization="root_relative",
        remove_unattached_vertices=False,
        all_phenotypes=False,
        topology="default",
    ),
    dict(
        all_phenotypes=True,
        remove_unattached_vertices=True,
    ),
    dict(
        rig="default-notongue-noeyes",
        topology="default-notongue-noeyes",
        remove_unattached_vertices=False,
        all_phenotypes=False,
    ),
    dict(
        local_changes=True,
        pose_parameterization="root_relative_world",
        remove_unattached_vertices=False,
        all_phenotypes=True,
    ),
    dict(
        local_changes=True,
        pose_parameterization="root_relative_world",
        remove_unattached_vertices=False,
        all_phenotypes=True,
        topology="notoes_collapse5pc",
    ),
]


class TestLegacySyntax(unittest.TestCase):
    def test_legacy_fullbody_construction_runs(self):
        for kwargs in LEGACY_CALL_CASES:
            with self.subTest(kwargs=kwargs):
                with warnings.catch_warnings():
                    warnings.simplefilter("error")
                    warnings.simplefilter("ignore", DeprecationWarning)
                    model = anny.create_fullbody_model(**kwargs).to(torch.float32)

                output = model()
                self.assertIn("vertices", output)
                self.assertIn("bone_poses", output)
                self.assertEqual(output["vertices"].shape[-1], 3)
                self.assertTrue(torch.isfinite(output["vertices"]).all())

    def test_legacy_fullbody_construction_warns(self):
        for kwargs in LEGACY_CALL_CASES:
            with self.subTest(kwargs=kwargs):
                with self.assertWarns(DeprecationWarning):
                    anny.create_fullbody_model(**kwargs)


if __name__ == "__main__":
    unittest.main()
