import unittest
import os

import torch

from anny.models.rigged_model import RiggedModelWithLinearBlendShapes

import anny
SKIP_SMPL = False
SKIP_SMPL_REASON = "SMPL/SMPL-X models are not available, please install anny[smpl] to run these tests"
try:
    from anny.models.smpl import SMPL, SMPLX
except ImportError:
    SKIP_SMPL = True

SMPLX_MODEL_PATH = os.environ.get("SMPLX_MODEL_PATH")
SMPLX_MODEL_SKIP_REASON = "SMPLX_MODEL_PATH is not defined"

@unittest.skipIf(SKIP_SMPL, SKIP_SMPL_REASON)
@unittest.skipIf(SMPLX_MODEL_PATH is None, SMPLX_MODEL_SKIP_REASON)
class TestSMPLForward(unittest.TestCase):
    dtype = torch.float32

    def test_smplx_is_normal_rigged_model_instance(self):
        model = SMPLX(SMPLX_MODEL_PATH, pose_corrective=True)

        self.assertIsInstance(model, RiggedModelWithLinearBlendShapes)
        self.assertEqual(model._bone_orientation_method, "tail")
        self.assertEqual(model.bone_count, 55)
        self.assertTrue(hasattr(model, "faces"))
        

    def test_smpl_is_normal_rigged_model_instance(self):
        model = SMPL(SMPLX_MODEL_PATH, pose_corrective=True)

        self.assertIsInstance(model, RiggedModelWithLinearBlendShapes)
        self.assertEqual(model._bone_orientation_method, "tail")
        self.assertEqual(model.bone_count, 24)
        self.assertTrue(hasattr(model, "faces"))
        self.assertEqual(model.template_vertices.shape[0], 6890)

    def test_smplx_with_anny_topology(self):
        model = SMPLX(SMPLX_MODEL_PATH, pose_corrective=True, topology="anny")
        anny_model = anny.Anny(remove_unattached_vertices=False)
        self.assertEqual(model.template_vertices.shape[0],  anny_model.template_vertices.shape[0])


    def test_smpl_forward_builds_batched_pose_and_coefficients(self):
        batch_size = 2
        model = SMPL(SMPLX_MODEL_PATH, pose_corrective=True)
        betas = torch.arange(batch_size * 10, dtype=self.dtype).reshape(batch_size, 10)
        global_orient = torch.zeros((batch_size, 3), dtype=self.dtype)
        transl = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=self.dtype)
        body_pose = torch.zeros((batch_size, model.bone_count - 1, 3), dtype=self.dtype)

        _ = model(
                betas=betas,
                global_orient=global_orient,
                transl=transl,
                body_pose=body_pose,
        )

    def test_smplx_forward_builds_batched_pose_and_coefficients_pca(self):
        batch_size = 2
        model = SMPLX(SMPLX_MODEL_PATH, pose_corrective=True)
        betas = torch.arange(batch_size * 10, dtype=self.dtype).reshape(batch_size, 10)
        expression = torch.arange(batch_size * 10, dtype=self.dtype).reshape(batch_size, 10) + 100.0
        global_orient = torch.zeros((batch_size, 3), dtype=self.dtype)
        transl = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=self.dtype)
        body_pose = torch.zeros((batch_size, 21, 3), dtype=self.dtype)
        eye_pose = torch.zeros((batch_size, 3), dtype=self.dtype)
        hand_pose = torch.zeros((batch_size, 6), dtype=self.dtype)
        jaw_pose = torch.zeros((batch_size, 3), dtype=self.dtype)

        _ = model(
                betas=betas,
                expression=expression,
                global_orient=global_orient,
                transl=transl,
                body_pose=body_pose,
                leye_pose=eye_pose,
                reye_pose=eye_pose,
                left_hand_pose=hand_pose,
                right_hand_pose=hand_pose,
                jaw_pose=jaw_pose,
            )

        

    def test_smplx_forward_builds_batched_pose_and_coefficients_no_pca(self):
        batch_size = 2
        model = SMPLX(SMPLX_MODEL_PATH, pose_corrective=True, use_pca=False)
        betas = torch.arange(batch_size * 10, dtype=self.dtype).reshape(batch_size, 10)
        expression = torch.arange(batch_size * 10, dtype=self.dtype).reshape(batch_size, 10) + 100.0
        global_orient = torch.zeros((batch_size, 3), dtype=self.dtype)
        transl = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=self.dtype)
        body_pose = torch.zeros((batch_size, 21, 3), dtype=self.dtype)
        eye_pose = torch.zeros((batch_size, 3), dtype=self.dtype)
        hand_pose = torch.zeros((batch_size, 15, 3), dtype=self.dtype)
        jaw_pose = torch.zeros((batch_size, 3), dtype=self.dtype)

        _ = model(
                betas=betas,
                expression=expression,
                global_orient=global_orient,
                transl=transl,
                body_pose=body_pose,
                leye_pose=eye_pose,
                reye_pose=eye_pose,
                left_hand_pose=hand_pose,
                right_hand_pose=hand_pose,
                jaw_pose=jaw_pose,
            )
        
if __name__ == "__main__":
    unittest.main()
