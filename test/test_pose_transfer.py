import unittest

import torch
import roma

import anny
from anny.utils.pose import transfer_pose_parameters


class TestPoseTransfer(unittest.TestCase):
    def test_transfer_reproduces_pose_across_rest_orientations(self):
        dtype = torch.float64
        # Two rigs sharing the mesh, skinning weights and bone origins but with different rest bone
        # orientations: 'makehuman' uses the tail-based (blender) orientation, 'anny' the procrustes one.
        src_model = anny.Anny(rig="makehuman", topology="anny", local_changes="default").to(dtype=dtype)
        target_model = anny.Anny(rig="anny", topology="anny", local_changes="default").to(dtype=dtype)

        torch.manual_seed(0)
        phenotype_kwargs = {label: torch.rand((), dtype=dtype) for label in target_model.phenotype_labels}
        local_changes_kwargs = {label: 2 * torch.rand((), dtype=dtype) - 1 for label in target_model.local_change_labels}

        # Pose only the bones the target shares (matched by label); leave source-only bones
        # (tongue/expression) at rest, since the target rig cannot reproduce their deformation.
        shared_indices = [src_model.bone_labels.index(label) for label in target_model.bone_labels]
        rotvecs = torch.zeros((1, len(src_model.bone_labels), 3), dtype=dtype)
        translations = torch.zeros((1, len(src_model.bone_labels), 3), dtype=dtype)
        rotvecs[:, shared_indices] = 0.3 * torch.randn((1, len(shared_indices), 3), dtype=dtype)
        # Per-bone translations are not anatomically realistic (they detach bones from their parents),
        # but since the transfer merely re-expresses the source's world bone poses it reproduces them.
        translations[:, shared_indices] = 0.05 * torch.randn((1, len(shared_indices), 3), dtype=dtype)
        src_pose_parameters = roma.Rigid(roma.rotvec_to_rotmat(rotvecs), translations).to_homogeneous()

        target_pose_parameters = transfer_pose_parameters(
            src_model=src_model,
            src_pose_parameters=src_pose_parameters,
            phenotype_kwargs=phenotype_kwargs,
            local_changes_kwargs=local_changes_kwargs,
            target_model=target_model,
        )

        src_output = src_model(pose_parameters=src_pose_parameters, phenotype_kwargs=phenotype_kwargs, local_changes_kwargs=local_changes_kwargs)
        target_output = target_model(pose_parameters=target_pose_parameters, phenotype_kwargs=phenotype_kwargs, local_changes_kwargs=local_changes_kwargs)
        max_error = torch.linalg.norm(src_output["vertices"] - target_output["vertices"], dim=-1).max()
        self.assertLess(max_error, 1e-4)

    def test_transfer_raises_when_target_bone_missing(self):
        dtype = torch.float64
        # The target keeps face/tongue bones that the source rig prunes, so no complete label mapping
        # exists and the pose cannot be transferred.
        src_model = anny.Anny(rig="anny", topology="anny").to(dtype=dtype)
        target_model = anny.Anny(rig="makehuman", topology="anny").to(dtype=dtype)
        src_pose_parameters = torch.eye(4, dtype=dtype)[None, None].expand(
            1, len(src_model.bone_labels), 4, 4)
        with self.assertRaises(AssertionError):
            transfer_pose_parameters(
                src_model=src_model,
                src_pose_parameters=src_pose_parameters,
                phenotype_kwargs=dict(),
                local_changes_kwargs=dict(),
                target_model=target_model,
            )


if __name__ == "__main__":
    unittest.main()
