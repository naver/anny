# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
import unittest
import io
import tempfile
import os

import torch

from anny.models.rigged_model import RiggedModelWithLinearBlendShapes
from anny.models.phenotype import Anny
from anny.models.model_data import ModelData, ModelMetadata, AnnyModelMetadata
from anny.typing import SkinningMethod

class TestPickle(unittest.TestCase):

    def test_rigged_model_pickle(self):
        """
        Tests that RiggedModelWithLinearBlendShapes can be pickled and unpickled using torch.save and torch.load.
        This is important for distributed training and saving model checkpoints.
        We test with all available skinning methods.
        """
        device = torch.device("cpu")
        dtype = torch.float32

        # Define dummy data for RiggedModelWithLinearBlendShapes
        v_count = 10
        f_count = 5
        b_count = 3
        bs_count = 2
        max_bones = 4

        template_vertices = torch.randn(v_count, 3, dtype=dtype)
        faces = torch.randint(0, v_count, (f_count, 3), dtype=torch.long)
        texture_coordinates = torch.randn(v_count, 2, dtype=dtype)
        face_texture_coordinate_indices = torch.randint(0, v_count, (f_count, 3), dtype=torch.long)
        blendshapes = torch.randn(bs_count, v_count, 3, dtype=dtype)
        template_bone_heads = torch.randn(b_count, 3, dtype=dtype)
        bone_heads_blendshapes = torch.randn(bs_count, b_count, 3, dtype=dtype)
        template_bone_tails = template_bone_heads + torch.tensor([0.0, 1.0, 0.0], dtype=dtype)
        bone_tails_blendshapes = torch.randn(bs_count, b_count, 3, dtype=dtype)
        bone_rolls_rotmat = torch.eye(3, dtype=dtype).unsqueeze(0).unsqueeze(0).expand(1, b_count, 3, 3).clone()
        bone_parents = [-1, 0, 1]
        bone_labels = ["root", "spine", "head"]
        vertex_bone_weights = torch.rand(v_count, max_bones, dtype=dtype)
        vertex_bone_weights /= vertex_bone_weights.sum(dim=-1, keepdim=True)
        vertex_bone_indices = torch.randint(0, b_count, (v_count, max_bones), dtype=torch.long)

        skinning_methods: list[SkinningMethod] = ["lbs", "dqs"]
        try:
            import warp
            skinning_methods.append("warp_lbs")
        except ImportError:
            pass

        for method in skinning_methods:
            with self.subTest(skinning_method=method):
                model = RiggedModelWithLinearBlendShapes(
                    template_vertices=template_vertices,
                    faces=faces,
                    texture_coordinates=texture_coordinates,
                    face_texture_coordinate_indices=face_texture_coordinate_indices,
                    blendshapes=blendshapes,
                    template_bone_heads=template_bone_heads,
                    bone_heads_blendshapes=bone_heads_blendshapes,
                    bone_parents=bone_parents,
                    bone_labels=bone_labels,
                    vertex_bone_weights=vertex_bone_weights,
                    vertex_bone_indices=vertex_bone_indices,
                    base_mesh_vertex_indices=torch.arange(v_count, dtype=torch.long),
                    skinning_method=method,
                    pose_parameterization="local-bone",
                    template_bone_tails=template_bone_tails,
                    bone_tails_blendshapes=bone_tails_blendshapes,
                    bone_rolls_rotmat=bone_rolls_rotmat,
                ).to(device=device, dtype=dtype)

                # Test pickling with a buffer
                buffer = io.BytesIO()
                torch.save(model, buffer)
                buffer.seek(0)

                unpickled_model = torch.load(buffer, weights_only=False)

                # Check if unpickled model is of the correct type
                self.assertIsInstance(unpickled_model, RiggedModelWithLinearBlendShapes)

                # Check some attributes to ensure data integrity
                torch.testing.assert_close(model.template_vertices, unpickled_model.template_vertices)
                self.assertEqual(model.bone_parents, unpickled_model.bone_parents)
                self.assertEqual(model.bone_labels, unpickled_model.bone_labels)
                self.assertEqual(model.pose_parameterization, unpickled_model.pose_parameterization)

                # Test running forward pass on unpickled model
                batch_size = 2
                pose_params = None
                blendshape_coeffs = torch.zeros(batch_size, bs_count, dtype=dtype)

                with torch.no_grad():
                    out_orig = model(pose_params, blendshape_coeffs)["vertices"]
                    out_unpickled = unpickled_model(pose_params, blendshape_coeffs)["vertices"]

                torch.testing.assert_close(out_orig, out_unpickled)

class TestSafetensors(unittest.TestCase):

    def _make_tail_model_data(self, dtype=torch.float64):
        v_count = 8
        f_count = 4
        b_count = 3
        n_pheno_bs = 6
        max_bones = 2

        template_vertices = torch.randn(v_count, 3, dtype=dtype)
        faces = torch.randint(0, v_count, (f_count, 3), dtype=torch.long)
        blendshapes = torch.randn(n_pheno_bs, v_count, 3, dtype=dtype)
        stacked_phenotype_blend_shapes_mask = torch.rand(n_pheno_bs, 26, dtype=dtype).clamp(0, 1).round()
        template_bone_heads = torch.randn(b_count, 3, dtype=dtype)
        template_bone_tails = template_bone_heads + torch.tensor([0.0, 1.0, 0.0], dtype=dtype)
        bone_heads_blendshapes = torch.randn(n_pheno_bs, b_count, 3, dtype=dtype)
        bone_tails_blendshapes = torch.randn(n_pheno_bs, b_count, 3, dtype=dtype)
        bone_rolls_rotmat = torch.eye(3, dtype=dtype).unsqueeze(0).unsqueeze(0).expand(1, b_count, 3, 3).clone()
        vertex_bone_weights = torch.rand(v_count, max_bones, dtype=dtype)
        vertex_bone_weights /= vertex_bone_weights.sum(dim=-1, keepdim=True)
        vertex_bone_indices = torch.randint(0, b_count, (v_count, max_bones), dtype=torch.long)

        return ModelData(
            metadata=AnnyModelMetadata(
                bone_parents=[-1, 0, 1],
                bone_labels=["root", "spine", "head"],
                pose_parameterization="local-bone",
                skinning_method="lbs",
                bone_orientation="blender-rootidentity",
                local_change_labels=[],
                all_phenotypes=False,
                extrapolate_phenotypes=False,
            ),
            template_vertices=template_vertices,
            faces=faces,
            texture_coordinates=None,
            face_texture_coordinate_indices=None,
            blendshapes=blendshapes,
            stacked_phenotype_blend_shapes_mask=stacked_phenotype_blend_shapes_mask,
            template_bone_heads=template_bone_heads,
            bone_heads_blendshapes=bone_heads_blendshapes,
            vertex_bone_weights=vertex_bone_weights,
            vertex_bone_indices=vertex_bone_indices,
            base_mesh_vertex_indices=torch.arange(v_count, dtype=torch.long),
            template_bone_tails=template_bone_tails,
            bone_tails_blendshapes=bone_tails_blendshapes,
            bone_rolls_rotmat=bone_rolls_rotmat,
        )

    def _make_phenotype_model(self, dtype=torch.float64):
        return Anny.from_model_data(self._make_tail_model_data(dtype=dtype))

    def _make_procrustes_model_data(self, dtype=torch.float64):
        data = self._make_tail_model_data(dtype=dtype)
        b_count = len(data.metadata.bone_labels)
        vertex_ids = torch.arange(0, 6, dtype=torch.long).reshape(3, 2)
        return ModelData(
            metadata=ModelMetadata(
                bone_parents=data.metadata.bone_parents,
                bone_labels=data.metadata.bone_labels,
                pose_parameterization=data.metadata.pose_parameterization,
                skinning_method=data.metadata.skinning_method,
                bone_orientation="procrustes",
            ),
            template_vertices=data.template_vertices,
            faces=data.faces,
            texture_coordinates=data.texture_coordinates,
            face_texture_coordinate_indices=data.face_texture_coordinate_indices,
            blendshapes=data.blendshapes,
            stacked_phenotype_blend_shapes_mask=data.stacked_phenotype_blend_shapes_mask,
            template_bone_heads=data.template_bone_heads,
            bone_heads_blendshapes=data.bone_heads_blendshapes,
            vertex_bone_weights=data.vertex_bone_weights,
            vertex_bone_indices=data.vertex_bone_indices,
            base_mesh_vertex_indices=data.base_mesh_vertex_indices,
            bone_nonzeroweight_mask=torch.ones(b_count, dtype=torch.bool),
            bone_vertex_indices=vertex_ids,
            bone_vertex_weights=torch.ones(b_count, 2, dtype=dtype),
            template_bone_vertices=torch.randn(b_count, 2, 3, dtype=dtype),
            reference_bone_orientations=torch.eye(3, dtype=dtype).expand(b_count, 3, 3).clone(),
        )

    def test_model_data_round_trip(self):
        model = self._make_phenotype_model()
        data = model.to_model_data()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model_data.safetensors")
            data.save_safetensors(path)

            loaded_data = ModelData.load_safetensors(path)

        self.assertEqual(data.metadata.bone_orientation, loaded_data.metadata.bone_orientation)
        self.assertEqual(data.metadata.bone_labels, loaded_data.metadata.bone_labels)
        torch.testing.assert_close(data.template_vertices, loaded_data.template_vertices)
        torch.testing.assert_close(data.blendshapes, loaded_data.blendshapes)




if __name__ == "__main__":
    unittest.main()
