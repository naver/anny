# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0

"""Test that the skinning weights of each bone are compact on the body shell:
the set of body-shell vertices a bone influences forms a connected region.
Other mesh shells (eyes, teeth, ...) are excluded so bones that legitimately
span multiple shells are not flagged."""

import unittest

import torch
import trimesh.graph

import anny
from anny.models.model_data import _RIG_PRESET_FILES
from anny.models.model_transforms import _get_symmetric_bone_name
from anny.utils.mesh_utils import get_edge_vertex_indices, get_symmetric_vertex_indices


class TestSkinningWeightCompactness(unittest.TestCase):
    def test_each_bone_skinning_region_is_connected(self):
        model = anny.Anny()
        edges = get_edge_vertex_indices(model.faces).cpu().numpy()

        # Body shell = largest connected component; other shells (eyes, teeth) are excluded.
        mesh_components = trimesh.graph.connected_components(edges=edges)
        body_vertex_mask = torch.zeros(
            model.template_vertices.shape[0], dtype=torch.bool
        )
        body_vertex_mask[
            torch.as_tensor(max(mesh_components, key=len), dtype=torch.int64)
        ] = True

        offending = []
        for bone_id, bone_name in enumerate(model.bone_labels):
            bone_vertex_mask = (
                (model.vertex_bone_indices == bone_id) & (model.vertex_bone_weights > 0)
            ).any(dim=-1) & body_vertex_mask
            if not bone_vertex_mask.any():
                continue
            skinned_nodes = bone_vertex_mask.cpu().numpy().nonzero()[0]
            components = trimesh.graph.connected_components(
                edges=edges, nodes=skinned_nodes
            )
            if len(components) != 1:
                offending.append((bone_name, [len(c) for c in components]))

        self.assertEqual(
            offending,
            [],
            msg=f"Bones with non-compact skinning region on the body shell: {offending}",
        )


class TestSkinningWeightNormalization(unittest.TestCase):
    def test_per_vertex_weights_sum_to_one(self):
        for rig in _RIG_PRESET_FILES:
            with self.subTest(rig=rig):
                model = anny.Anny(rig=rig)
                row_sums = model.vertex_bone_weights.sum(dim=-1)
                torch.testing.assert_close(
                    row_sums,
                    torch.ones_like(row_sums),
                    atol=1e-6,
                    rtol=0,
                )


class TestSkinningWeightSymmetry(unittest.TestCase):
    def test_symmetric_bone_name_supports_builtin_rig_naming_schemes(self):
        cases = {
            "upperarm02.L": "upperarm02.R",
            "upperarm02.R": "upperarm02.L",
            "mixamorig:LeftArm": "mixamorig:RightArm",
            "mixamorig:RightForeArm": "mixamorig:LeftForeArm",
            "LeftHandThumb1": "RightHandThumb1",
            "RightEye": "LeftEye",
            "spine": "spine",
        }
        for bone_name, expected in cases.items():
            with self.subTest(bone_name=bone_name):
                self.assertEqual(_get_symmetric_bone_name(bone_name), expected)

    def test_lr_bone_pairs_have_mirrored_skinning_weights(self):
        model = anny.Anny()
        N = model.template_vertices.shape[0]
        B = len(model.bone_labels)

        sym = get_symmetric_vertex_indices(
            model.template_vertices,
            axis=0,
            threshold=1e-4,
        )

        dense = torch.zeros(N, B, dtype=model.vertex_bone_weights.dtype)
        dense.scatter_add_(
            1,
            model.vertex_bone_indices,
            model.vertex_bone_weights,
        )

        name_to_id = {n: i for i, n in enumerate(model.bone_labels)}
        missing, mismatched = [], []
        for name, bid in name_to_id.items():
            mate = _get_symmetric_bone_name(name)
            if mate == name:
                continue  # central bone — not an L/R pair
            if mate not in name_to_id:
                missing.append(name)
                continue
            w_self = dense[:, bid]
            w_mate_mirrored = dense[:, name_to_id[mate]][sym]
            if not torch.allclose(w_self, w_mate_mirrored, atol=1e-6, rtol=0):
                max_err = (w_self - w_mate_mirrored).abs().max().item()
                mismatched.append((name, mate, max_err))

        self.assertEqual(missing, [], msg=f"L/R bones without counterpart: {missing}")
        self.assertEqual(
            mismatched, [], msg=f"L/R bones with non-mirrored weights: {mismatched}"
        )


if __name__ == "__main__":
    unittest.main()
