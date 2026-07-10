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

    def test_blendshape_labels(self):
        for model in [anny.Anny(rig="soma", topology="soma"), anny.Anny(rig="anny", topology="anny")]:
            self.assertEqual(len(model.blendshape_labels), model.blendshapes.shape[0])
            self.assertEqual(len(set(model.blendshape_labels)), len(model.blendshape_labels))

    def test_orientation_local_changes_slicing(self):
        all_model = anny.Anny(rig="soma", topology="soma", local_changes="all")
        all_index = {label: i for i, label in enumerate(all_model.blendshape_labels)}
        arbitrary_subset = [all_model.local_change_labels[0], all_model.local_change_labels[-1]]
        for local_changes in ["default", arbitrary_subset]:
            subset_model = anny.Anny(rig="soma", topology="soma", local_changes=local_changes)
            # Assign a coefficient to each blend shape of the subset model, and compare against
            # the full model with the same coefficients on the corresponding rows.
            torch.manual_seed(0)
            subset_coeffs = torch.rand((4, subset_model.blendshapes.shape[0]), dtype=subset_model.dtype)
            all_coeffs = torch.zeros((4, all_model.blendshapes.shape[0]), dtype=all_model.dtype)
            for j, label in enumerate(subset_model.blendshape_labels):
                all_coeffs[:, all_index[label]] = subset_coeffs[:, j]
            poses_subset = subset_model.get_rest_model(subset_coeffs)["rest_bone_poses"]
            poses_all = all_model.get_rest_model(all_coeffs)["rest_bone_poses"]
            self.assertLess(torch.max(torch.abs(poses_subset - poses_all)), 1e-6)


if __name__ == "__main__":
    unittest.main()
