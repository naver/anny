# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
import os
import pathlib
import unittest

import anny
from anny.paths import ANNY_ROOT_DIR
from anny.models.rigged_model import RiggedModelWithLinearBlendShapes





class TestInstantiation(unittest.TestCase):

    def test_instantiation(self):
        for rig in ["default", "soma"]:
            for topology in ["default", "soma", "smplx"]:
                with self.subTest(rig=rig, topology=topology):
                    model = anny.Anny(rig=rig, topology=topology)
                    self.assertIsNotNone(model)

    def test_procrustes_model(self):
        for rig in ["soma"]:
            for topology in ["default", "soma"]:
                with self.subTest(rig=rig, topology=topology):
                    model = anny.Anny(rig=rig, topology=topology, bone_orientation="procrustes")
                    self.assertIsNotNone(model)

    def test_anny_is_normal_rigged_model_instance(self):
        model = anny.Anny()

        self.assertIsInstance(model, anny.Anny)
        self.assertIsInstance(model, RiggedModelWithLinearBlendShapes)
        self.assertFalse(hasattr(model, "model_type"))
        self.assertEqual(model._bone_orientation_method, "tail")
        self.assertTrue(hasattr(model, "template_vertices"))
        self.assertTrue(hasattr(model, "bone_labels"))
        self.assertTrue(hasattr(model, "vertex_bone_weights"))
        self.assertTrue(callable(model.get_rest_model))
        self.assertTrue(callable(model.get_pose_parameterization))
        self.assertTrue(callable(model.set_skinning_method))

    def test_custom_rig_path_accepts_pathlike(self):
        rig_filename = pathlib.Path(ANNY_ROOT_DIR) / "data/mpfb2/rigs/standard/rig.default.json"
        weights_filename = pathlib.Path(ANNY_ROOT_DIR) / "data/mpfb2/rigs/standard/weights.default.json"

        try:
            model = anny.Anny(rig=rig_filename, weights_filename=weights_filename)
        except Exception as exc:
            self.fail(f"PathLike custom rig should instantiate, raised {exc!r}")

        self.assertIsNotNone(model)

    def test_custom_rig_path_with_smplx_topology_passes_weights(self):
        rig_filename = pathlib.Path(ANNY_ROOT_DIR) / "data/mpfb2/rigs/standard/rig.default.json"
        weights_filename = pathlib.Path(ANNY_ROOT_DIR) / "data/mpfb2/rigs/standard/weights.default.json"

        try:
            model = anny.Anny(
                rig=rig_filename,
                weights_filename=weights_filename,
                topology="smplx",
            )
        except Exception as exc:
            self.fail(f"PathLike custom rig with SMPL-X topology should instantiate, raised {exc!r}")

        self.assertIsNotNone(model)

    def test_public_factories_return_anny(self):
        fullbody = anny.create_fullbody_model()
        hand = anny.create_hand_model()
        head = anny.create_head_model()

        self.assertIsInstance(fullbody, anny.Anny)
        self.assertIsInstance(hand, anny.Anny)
        self.assertIsInstance(head, anny.Anny)
        self.assertIsInstance(fullbody, RiggedModelWithLinearBlendShapes)
        self.assertIsInstance(hand, RiggedModelWithLinearBlendShapes)
        self.assertIsInstance(head, RiggedModelWithLinearBlendShapes)

  


if __name__ == "__main__":
    unittest.main()
