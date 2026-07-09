# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
import unittest

from anny.models.model_data import RigConfig
import torch
import anny
from anny.paths import get_anny_cache_path, get_anny_root_dir
from anny.models.rigged_model import RiggedModelWithLinearBlendShapes


class TestInstantiation(unittest.TestCase):
    def test_instantiation(self):
        for rig in ["anny", "soma"]:
            for topology in ["anny", "soma", "smplx"]:
                with self.subTest(rig=rig, topology=topology):
                    model = anny.Anny(rig=rig, topology=topology)
                    self.assertIsNotNone(model)

    def test_bone_orientation_default(self):
        for rig in ["anny", "cmu_mb", "game_engine", "mixamo"]:
            with self.subTest(rig=rig):
                model = anny.Anny(
                    rig=rig,
                    topology="anny",
                    skinning_method="lbs",
                )
                self.assertIsNotNone(model)
                self.assertEqual(model.bone_orientation, "procrustes" if rig =="anny" else "blender")

    def test_procrustes_anny_rig_retopology_paths(self):
        for topology in ["smplx", "smpl", "soma"]:
            with self.subTest(topology=topology):
                if topology == "smpl":
                    anny2smpl_path = (
                        get_anny_cache_path() / "noncommercial" / "anny2smpl.pth"
                    )
                    if not anny2smpl_path.exists():
                        self.skipTest("SMPL retopology data is not available.")
                model = anny.Anny(
                    rig="anny",
                    topology=topology,
                    skinning_method="lbs",
                )
                self.assertIsNotNone(model)
                self.assertEqual(model.bone_orientation, "procrustes")

    def test_procrustes_custom_rig_path_accepts_rig_spec(self):
        rig_filename = get_anny_root_dir() / "data/mpfb2/rigs/standard/rig.default.json"
        weights_filename = get_anny_root_dir() / "data/mpfb2/rigs/standard/weights.default.json"

        model = anny.Anny(
            rig=RigConfig(rig_filename, weights_filename=weights_filename, bone_orientation="procrustes"),
            skinning_method="lbs",
        )

        self.assertIsNotNone(model)
        self.assertEqual(model.bone_orientation, "procrustes")

    def test_procrustes_anny_rig_forward_outputs(self):
        model = anny.Anny(
            rig="anny",
            skinning_method="lbs",
        )

        output = model()

        self.assertIn("vertices", output)
        self.assertIn("bone_poses", output)
        self.assertIn("rest_vertices", output)
        self.assertIn("rest_bone_heads", output)
        self.assertIn("rest_bone_poses", output)
        self.assertNotIn("rest_bone_tails", output)

    def test_procrustes_root_identity_orientation(self):
        model = anny.Anny(
            rig="anny",
            skinning_method="lbs",
        )
        blendshape_coeffs = torch.zeros(
            (1, model.blendshapes.shape[0]),
            dtype=model.dtype,
            device=model.device,
        )

        rest_bone_poses = model.get_rest_model(blendshape_coeffs)["rest_bone_poses"]

        torch.testing.assert_close(
            rest_bone_poses[0, 0, :3, :3],
            torch.eye(3, dtype=model.dtype, device=model.device),
        )

    def test_procrustes_return_bone_ends_raises_clear_error(self):
        model = anny.Anny(
            rig="anny",
            skinning_method="lbs",
        )

        with self.assertRaisesRegex(NotImplementedError, "return_bone_ends=True is not supported"):
            model(return_bone_ends=True)

    def test_anny_is_normal_rigged_model_instance(self):
        model = anny.Anny()

        self.assertIsInstance(model, anny.Anny)
        self.assertIsInstance(model, RiggedModelWithLinearBlendShapes)
        self.assertFalse(hasattr(model, "model_type"))
        self.assertEqual(model.bone_orientation, "procrustes")
        self.assertTrue(hasattr(model, "template_vertices"))
        self.assertTrue(hasattr(model, "bone_labels"))
        self.assertTrue(hasattr(model, "vertex_bone_weights"))
        self.assertTrue(callable(model.get_rest_model))
        self.assertTrue(callable(model.get_pose_parameterization))
        self.assertTrue(callable(model.set_skinning_method))
        self.assertIsNotNone(model)



if __name__ == "__main__":
    unittest.main()
