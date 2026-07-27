# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0

import unittest

import torch

import anny


class TestTorchCapture(unittest.TestCase):
    def _make_base_model_inputs(self):
        model = anny.Anny(skinning_method="lbs")
        pose_parameters = torch.eye(4, dtype=torch.float64)[None, None].expand(1, model.bone_count, 4, 4).clone()
        return model, pose_parameters

    @unittest.skipIf(not hasattr(torch, "compile"), "torch.compile is not available")
    def test_rigged_model_supports_fullgraph_compile(self):
        model, pose_parameters = self._make_base_model_inputs()
        eager_output = model(pose_parameters)

        compiled_model = torch.compile(model, backend="eager", fullgraph=True)
        compiled_output = compiled_model(pose_parameters)

        torch.testing.assert_close(compiled_output["vertices"], eager_output["vertices"])
        torch.testing.assert_close(compiled_output["bone_poses"], eager_output["bone_poses"])

    @unittest.skipIf(
        not hasattr(torch, "export") or not hasattr(torch.export, "export"),
        "torch.export.export is not available",
    )
    def test_rigged_model_supports_export(self):
        model, pose_parameters = self._make_base_model_inputs()

        exported_program = torch.export.export(model, (pose_parameters, ))

        self.assertIsNotNone(exported_program)

    @unittest.skipIf(not hasattr(torch, "compile"), "torch.compile is not available")
    def test_rigged_model_supports_fullgraph_compiled_backward(self):
        model, pose_parameters = self._make_base_model_inputs()
        eager_phenotype = torch.full(
            (1, len(model.phenotype_labels)),
            0.5,
            dtype=model.dtype,
            requires_grad=True,
        )
        compiled_phenotype = eager_phenotype.detach().clone().requires_grad_()

        eager_vertices = model(
            pose_parameters,
            phenotype_kwargs=eager_phenotype,
        )["vertices"]
        eager_vertices.square().mean().backward()

        compiled_model = torch.compile(
            model,
            backend="aot_eager",
            fullgraph=True,
        )
        compiled_vertices = compiled_model(
            pose_parameters,
            phenotype_kwargs=compiled_phenotype,
        )["vertices"]
        compiled_vertices.square().mean().backward()

        torch.testing.assert_close(compiled_vertices, eager_vertices)
        torch.testing.assert_close(compiled_phenotype.grad, eager_phenotype.grad)

if __name__ == "__main__":
    unittest.main()
