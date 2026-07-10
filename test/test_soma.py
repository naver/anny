import unittest
from unittest import mock

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
        refiner = model.rest_orientation_refiner
        self.assertIsNotNone(refiner)
        blendshape_coeffs = torch.zeros((1, model.blendshapes.shape[0]), dtype=model.dtype)
        rest_bone_poses = model.get_rest_model(blendshape_coeffs)["rest_bone_poses"]
        # Orientations are valid rotation matrices.
        R = rest_bone_poses[0, :, :3, :3]
        self.assertLess(torch.max(torch.abs(R @ R.transpose(-1, -2) - torch.eye(3, dtype=model.dtype))), 1e-6)
        # End bones copy their parent's orientation (SOMA skeleton fit convention).
        self.assertLess(torch.max(roma.rotmat_geodesic_distance(
            rest_bone_poses[0, refiner.leaf_bone_indices, :3, :3],
            rest_bone_poses[0, refiner.leaf_bone_parent_indices, :3, :3])), 1e-9)

    def test_skeleton_fit_parity_with_soma_package(self):
        """anny's precomputed procrustes orientations plus the child-offset/end-joint refinement
        reproduce the SOMA skeleton fit for every non-root bone.

        The only structural difference is that SOMA's per-bone skinning Kabsch (``R_init``) augments
        its covariance with a *virtual normal* (``soma.geometry.transforms.compute_covariance``,
        ``virtual_normal=True``): a synthetic correspondence built from two specific attached
        vertices. That term is nonlinear in the blendshape coefficients and vertex-index dependent,
        so it cannot live in anny's linear, topology-independent covariance ``M0 + Σc·B``. With it on
        (SOMA's default) it rotates ``R_init`` by up to ~1e-2 rad on thin/near-degenerate skinning
        clouds (forearms, finger segments); its effect on the posed mesh is dominated by the
        (deferred) topology-transfer difference. This test therefore disables that one term to verify
        the part anny is responsible for. The child-offset alignment (small N) keeps its virtual
        normal, which anny does replicate from topology-independent joint offsets. The root is a
        mesh-irrelevant orientation convention (differs by pi/2) and is excluded."""
        try:
            import soma
            from soma.geometry.transforms import compute_covariance, kabsch, newton_schulz, rodrigues_rotation
        except ImportError:
            self.skipTest("soma package not installed")

        def align_vectors_no_skinning_vnorm(A, B, eps=1e-8, method="kabsch"):
            # Drop the virtual-normal conditioning for the large-N skinning fit (R_init) only; the
            # small-N child-offset alignment (<= a handful of children) keeps it.
            if A.shape[-2] == 1:
                return rodrigues_rotation(A[..., 0, :], B[..., 0, :], eps=eps)
            H = compute_covariance(A, B, virtual_normal=A.shape[-2] <= 6, eps=eps)
            return newton_schulz(H, num_iters=20, eps=eps) if method == "newton-schulz" else kabsch(H)

        dtype = torch.float64
        soma_layer = soma.SOMALayer(identity_model_type="anny", mode="warp",
                                    device=torch.device("cpu")).to(dtype=dtype)
        model = anny.Anny(rig="soma", topology="soma").to(dtype=dtype)
        torch.manual_seed(0)
        blendshape_coeffs = torch.rand((2, model.blendshapes.shape[0]), dtype=dtype)
        rest = model.get_rest_model(blendshape_coeffs)
        # anny geometry is Z-up, the SOMA skeleton data Y-up.
        P = torch.tensor([[1., 0., 0.], [0., 0., 1.], [0., -1., 0.]], dtype=dtype)
        vertices = rest["rest_vertices"] @ P.T
        skeleton_transfer = soma_layer.skeleton_transfer
        with mock.patch("soma.geometry.skeleton_transfer.align_vectors",
                        align_vectors_no_skinning_vnorm):
            fitted = skeleton_transfer.fit_joint_rotations(
                skeleton_transfer.fit_joint_positions(vertices), vertices)
        geodesic = roma.rotmat_geodesic_distance(
            fitted[:, 1:, :3, :3], (P[None, None] @ rest["rest_bone_poses"][:, :, :3, :3])[:, 1:])
        self.assertLess(torch.max(geodesic), 1e-6)

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
