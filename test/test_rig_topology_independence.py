# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
import unittest

import torch

import anny
from anny.models import full_model
from anny.models.model_data import ModelData, ModelMetadata, RigConfig

HAND_BONE_NAMES = [
    "wrist",
    "finger1-1",
    "finger1-2",
    "finger1-3",
    "metacarpal1",
    "finger2-1",
    "finger2-2",
    "finger2-3",
    "metacarpal2",
    "finger3-1",
    "finger3-2",
    "finger3-3",
    "metacarpal3",
    "finger4-1",
    "finger4-2",
    "finger4-3",
    "metacarpal4",
    "finger5-1",
    "finger5-2",
    "finger5-3",
]
LEFT_HAND_BONES = [f"{name}.L" for name in HAND_BONE_NAMES]
RIGHT_HAND_BONES = [f"{name}.R" for name in HAND_BONE_NAMES]


class TestRigFiltering(unittest.TestCase):
    def test_outside_subtree_weights_are_aggregated_at_synthetic_root(self):
        data = ModelData(
            metadata=ModelMetadata(
                bone_labels=["root", "arm.L", "wrist.L", "finger.L"],
                bone_parents=[-1, 0, 1, 2],
            ),
            template_vertices=torch.zeros((2, 3), dtype=torch.float64),
            faces=torch.empty((0, 4), dtype=torch.int64),
            blendshapes=torch.zeros((2, 2, 3), dtype=torch.float64),
            stacked_phenotype_blend_shapes_mask=None,
            template_bone_heads=torch.arange(12, dtype=torch.float64).reshape(4, 3),
            bone_heads_blendshapes=torch.arange(24, dtype=torch.float64).reshape(
                2, 4, 3
            ),
            vertex_bone_weights=torch.tensor(
                [[0.2, 0.3, 0.1, 0.4], [0.1, 0.2, 0.3, 0.4]],
                dtype=torch.float64,
            ),
            vertex_bone_indices=torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]]),
            base_mesh_vertex_indices=torch.arange(2),
            template_bone_tails=torch.arange(12, 24, dtype=torch.float64).reshape(4, 3),
            bone_tails_blendshapes=torch.arange(24, 48, dtype=torch.float64).reshape(
                2, 4, 3
            ),
            bone_rolls_rotmat=torch.arange(36, dtype=torch.float64).reshape(1, 4, 3, 3),
        )

        filtered = full_model._filter_rig(
            data, bones_to_remove=set(), subtree_root="wrist.L"
        )

        self.assertEqual(filtered.metadata.bone_labels, ["root", "wrist.L", "finger.L"])
        self.assertEqual(filtered.metadata.bone_parents, [-1, 0, 1])
        transform_indices = [2, 2, 3]
        torch.testing.assert_close(
            filtered.template_bone_heads,
            data.template_bone_heads[transform_indices],
        )
        torch.testing.assert_close(
            filtered.bone_heads_blendshapes,
            data.bone_heads_blendshapes[:, transform_indices],
        )
        torch.testing.assert_close(
            filtered.template_bone_tails,
            data.template_bone_tails[transform_indices],
        )
        torch.testing.assert_close(
            filtered.bone_tails_blendshapes,
            data.bone_tails_blendshapes[:, transform_indices],
        )
        torch.testing.assert_close(
            filtered.bone_rolls_rotmat,
            data.bone_rolls_rotmat[:, transform_indices],
        )
        self.assertTrue(
            torch.equal(
                filtered.vertex_bone_indices,
                torch.tensor([[0, 1, 2], [0, 1, 2]]),
            )
        )
        torch.testing.assert_close(
            filtered.vertex_bone_weights,
            torch.tensor([[0.5, 0.1, 0.4], [0.3, 0.3, 0.4]], dtype=torch.float64),
        )

    def test_cached_subtree_root_matches_selected_root_pose(self):
        model = anny.Anny(rig="anny-head", topology="anny")

        bone_poses = model()["bone_poses"]

        torch.testing.assert_close(bone_poses[:, 0], bone_poses[:, 1])


class TestRigSubtreeParsing(unittest.TestCase):
    def test_subtree_selectors_resolve_to_source_bones(self):
        cases = {
            "anny-head": "neck01",
            "makehuman-head": "neck01",
            "anny-hand.L": "wrist.L",
            "makehuman-hand.R": "wrist.R",
        }
        for spec, expected in cases.items():
            with self.subTest(spec=spec):
                self.assertEqual(
                    RigConfig.from_string(spec).subtree_root,
                    expected,
                )

    def test_removal_modifier_combines_with_subtree_selector(self):
        config = RigConfig.from_string("anny-nohands-hand.L")

        self.assertEqual(config.subtree_root, "wrist.L")
        self.assertFalse(config.root_identity_orientation)
        self.assertIn("finger1-1.L", config.bones_to_remove)
        self.assertNotIn("wrist.L", config.bones_to_remove)

    def test_multiple_subtree_selectors_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "multiple subtree selectors"):
            RigConfig.from_string("anny-head-hand.L")

    def test_plural_hand_selector_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown rig specifier: hands.L"):
            RigConfig.from_string("anny-hands.L")

    def test_soma_subtree_selector_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "does not support subtree selectors"):
            RigConfig.from_string("soma-head")
