import unittest

import roma
import torch

import anny


class TestSomaRigVertexCount(unittest.TestCase):

    def test_soma_rig_soma_topology_vertex_count_matches_anny_rig(self):
        anny_model = anny.Anny(rig="anny", topology="soma")
        soma_model = anny.Anny(rig="soma", topology="soma")
        self.assertEqual(soma_model.template_vertices.shape,
                         anny_model.template_vertices.shape)

    def test_soma_rig_default_topology_vertex_count_matches_anny_rig(self):
        anny_model = anny.Anny(rig="anny", topology="anny")
        soma_model = anny.Anny(rig="soma", topology="anny")
        self.assertEqual(soma_model.template_vertices.shape,
                         anny_model.template_vertices.shape)

    def test_soma_rig_preserves_alternative_topology(self):
        model = anny.Anny(rig="soma", topology="notoes_collapse10pc")
        self.assertEqual(model.template_vertices.shape, (1229, 3))


class TestSomaRigProcrustesOrientation(unittest.TestCase):

    def _max_rest_pose_difference(self, model_a, model_b, blendshape_coeffs):
        poses_a = model_a.get_rest_model(blendshape_coeffs)["rest_bone_poses"]
        poses_b = model_b.get_rest_model(blendshape_coeffs)["rest_bone_poses"]
        return torch.max(torch.abs(poses_a - poses_b))

    def test_precomputed_orientation_data(self):
        model = anny.Anny(rig="soma", topology="soma")
        self.assertIsNotNone(model.bone_template_orientation_matrices)
        self.assertIsNotNone(model.bone_orientation_blendshapes)
        self.assertIsNotNone(model.reference_bone_orientations)
        # With zero blendshape coefficients, rest orientations reduce to the template matrices.
        blendshape_coeffs = torch.zeros((1, model.blendshapes.shape[0]), dtype=model.dtype)
        rest_bone_poses = model.get_rest_model(blendshape_coeffs)["rest_bone_poses"]
        expected = roma.special_procrustes(model.bone_template_orientation_matrices)
        self.assertLess(torch.max(roma.rotmat_geodesic_distance(
            rest_bone_poses[0, :, :3, :3], expected)), 1e-6)

    def test_orientation_topology_independence(self):
        soma_topology_model = anny.Anny(rig="soma", topology="soma")
        anny_topology_model = anny.Anny(rig="soma", topology="anny")
        torch.manual_seed(0)
        blendshape_coeffs = torch.rand((4, soma_topology_model.blendshapes.shape[0]),
                                       dtype=soma_topology_model.dtype)
        self.assertLess(self._max_rest_pose_difference(
            soma_topology_model, anny_topology_model, blendshape_coeffs), 1e-6)

    def test_orientation_local_changes_slicing(self):
        all_model = anny.Anny(rig="soma", topology="soma", local_changes="all")
        default_model = anny.Anny(rig="soma", topology="soma", local_changes="default")
        base_count = all_model.blendshapes.shape[0] - 2 * len(all_model.local_change_labels)
        self.assertEqual(base_count,
                         default_model.blendshapes.shape[0] - 2 * len(default_model.local_change_labels))
        # Coefficients touching only the shared phenotype/facial-action block must yield the
        # same orientations in both configurations.
        torch.manual_seed(0)
        base_coeffs = torch.rand((4, base_count), dtype=all_model.dtype)
        all_coeffs = torch.zeros((4, all_model.blendshapes.shape[0]), dtype=all_model.dtype)
        all_coeffs[:, :base_count] = base_coeffs
        default_coeffs = torch.zeros((4, default_model.blendshapes.shape[0]), dtype=default_model.dtype)
        default_coeffs[:, :base_count] = base_coeffs
        poses_all = all_model.get_rest_model(all_coeffs)["rest_bone_poses"]
        poses_default = default_model.get_rest_model(default_coeffs)["rest_bone_poses"]
        self.assertLess(torch.max(roma.rotmat_geodesic_distance(
            poses_all[:, :, :3, :3], poses_default[:, :, :3, :3])), 1e-6)


if __name__ == "__main__":
    unittest.main()
