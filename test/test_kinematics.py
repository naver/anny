# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0

import unittest
import torch
import anny
import anny.utils.kinematics as kinematics
import roma


class TestBonePoses(unittest.TestCase):
    def bone_inputs(self):
        generator = torch.Generator().manual_seed(0)
        dtype = torch.float64
        up = torch.tensor([0.0, 1.0, 0.0], dtype=dtype)
        heads = torch.randn((2, 4, 3), generator=generator, dtype=dtype)
        offsets = up.expand(2, 4, 3).clone()
        offsets[:, 3] = torch.tensor([0.5, 1.0, -0.3], dtype=dtype)
        rolls = roma.random_rotmat((2, 4), dtype=dtype)
        return heads, offsets, rolls, up, torch.eye(3, dtype=dtype)

    def test_degenerate_bones_have_finite_gradients(self):
        heads, offsets, rolls, up, identity = self.bone_inputs()
        heads.requires_grad_()
        offsets.requires_grad_()
        poses = kinematics.get_bone_poses(heads, heads + offsets, rolls, up, identity)
        poses.sum().backward()
        self.assertTrue(torch.all(torch.isfinite(heads.grad)))
        self.assertTrue(torch.all(torch.isfinite(offsets.grad)))

    def test_gradcheck_with_degenerate_bones(self):
        heads, offsets, rolls, up, identity = self.bone_inputs()
        heads.requires_grad_()
        self.assertTrue(
            torch.autograd.gradcheck(
                lambda h: kinematics.get_bone_poses(
                    h, h + offsets, rolls, up, identity
                ),
                (heads,),
            )
        )


PARAMETERIZATIONS = (
    "local-ref",
    "local-bone",
    "local-bone-world",
    "world-orient",
    "world",
)


def random_rigid(shape, scale=1.0):
    return roma.Rigid(
        roma.random_rotmat(shape, dtype=torch.float64),
        scale * torch.randn(shape + (3,), dtype=torch.float64),
    ).to_homogeneous()


def reference_forward_kinematic(
    bone_parents, rest_bone_poses, delta_transforms, parameterization, orientations
):
    # The pose parameterizations by their definitions, bone by bone (parents first) with general inverses:
    # a reference pass of the reference orientations, then the posing pass.
    inv = torch.linalg.inv
    order = []
    while len(order) < len(bone_parents):
        order += [
            c
            for c, p in enumerate(bone_parents)
            if c not in order and (p < 0 or p in order)
        ]
    if parameterization == "world":
        return delta_transforms @ inv(rest_bone_poses), delta_transforms
    if parameterization == "world-orient":
        # The delta rotations are the bones' world orientations, and the root's delta its world pose.
        transforms, poses = {}, {}
        for c in order:
            p = bone_parents[c]
            if p < 0:
                poses[c] = delta_transforms[:, c]
            else:
                poses[c] = (transforms[p] @ rest_bone_poses[:, c]).clone()
                poses[c][..., :3, :3] = delta_transforms[:, c, :3, :3]
            transforms[c] = poses[c] @ inv(rest_bone_poses[:, c])
        return (
            torch.stack([transforms[c] for c in range(len(order))], 1),
            torch.stack([poses[c] for c in range(len(order))], 1),
        )
    ref_bone_poses = rest_bone_poses.clone()
    if orientations is not None:
        ref_transforms = {}
        for c in order:
            p = bone_parents[c]
            pose = (
                rest_bone_poses[:, c]
                if p < 0
                else ref_transforms[p] @ rest_bone_poses[:, c]
            ).clone()
            pose[..., :3, :3] = orientations[c]
            ref_bone_poses[:, c] = pose
            ref_transforms[c] = (
                torch.eye(4, dtype=pose.dtype)
                if p < 0
                else pose @ inv(rest_bone_poses[:, c])
            )
    if parameterization == "local-ref":
        # The deltas expressed in the reference frames.
        frames = torch.zeros_like(ref_bone_poses)
        frames[..., :3, :3] = ref_bone_poses[..., :3, :3]
        frames[..., 3, 3] = 1.0
        delta_transforms = inv(frames) @ delta_transforms @ frames
    transforms, poses = {}, {}
    for c in order:
        p = bone_parents[c]
        if p >= 0:
            parent = transforms[p]
        elif parameterization == "local-bone-world":
            parent = torch.eye(4, dtype=rest_bone_poses.dtype)
        else:
            # The root's reference pose as the origin.
            parent = inv(ref_bone_poses[:, c])
        poses[c] = parent @ ref_bone_poses[:, c] @ delta_transforms[:, c]
        transforms[c] = poses[c] @ inv(ref_bone_poses[:, c])
    poses = torch.stack([poses[c] for c in range(len(order))], 1)
    return poses @ inv(rest_bone_poses), poses


class TestKinematics(unittest.TestCase):
    def test_forward_kinematic(self):
        parent_indices = [-1, 3, 0, 0, 1, 2, 1]  # bone 1's parent comes after it
        n = len(parent_indices)
        rest_bone_poses = random_rigid((10, n))
        delta_transforms = random_rigid((10, n))
        chain = kinematics.get_kinematic_chain(parent_indices)
        for orientations in (None, roma.random_rotmat(n, dtype=torch.float64)):
            for parameterization in PARAMETERIZATIONS:
                with self.subTest(
                    parameterization=parameterization,
                    orientations=orientations is not None,
                ):
                    actual = kinematics.forward_kinematic(
                        chain,
                        rest_bone_poses,
                        delta_transforms,
                        parameterization,
                        orientations,
                    )
                    expected = reference_forward_kinematic(
                        parent_indices,
                        rest_bone_poses,
                        delta_transforms,
                        parameterization,
                        orientations,
                    )
                    for a, e in zip(actual, expected):
                        torch.testing.assert_close(a, e, rtol=0, atol=1e-12)

    def test_model_forward_kinematic(self):
        # The hand's root has A_0 != I, and makehuman has no reference orientations.
        for kwargs in (
            dict(),
            dict(rig="anny-hand.R", topology="hand.R"),
            dict(rig="makehuman", topology="makehuman"),
        ):
            model = anny.Anny(skinning_method="lbs", **kwargs).to(dtype=torch.float64)
            delta_transforms = random_rigid((4, model.bone_count), scale=0.01)
            phenotypes = torch.rand(4, len(model.phenotype_labels), dtype=model.dtype)
            rest_bone_poses = model(delta_transforms, phenotypes)["rest_bone_poses"]
            for parameterization in PARAMETERIZATIONS:
                with self.subTest(
                    rig=kwargs.get("rig", "anny"), parameterization=parameterization
                ):
                    actual = model.get_bone_transforms(
                        delta_transforms, rest_bone_poses, parameterization
                    )
                    expected = reference_forward_kinematic(
                        model.bone_parents,
                        rest_bone_poses,
                        delta_transforms,
                        parameterization,
                        model.reference_bone_orientations,
                    )
                    for a, e in zip(actual, expected):
                        torch.testing.assert_close(a, e, rtol=0, atol=1e-12)


class TestFrames(unittest.TestCase):
    def test_frames_are_orthonormal(self):
        # The kinematics invert the reference and rest frames by transposition.
        eye = torch.eye(3, dtype=torch.float64)
        for kwargs in (
            dict(),
            dict(rig="anny-hand.R", topology="hand.R"),
            dict(rig="makehuman", topology="makehuman"),
            dict(rig="soma", topology="anny"),
        ):
            with self.subTest(**kwargs):
                model = anny.Anny(**kwargs).to(dtype=torch.float64)
                phenotypes = torch.rand(
                    4, len(model.phenotype_labels), dtype=model.dtype
                )
                rest = model(phenotype_kwargs=phenotypes)["rest_bone_poses"]
                frames = [rest[..., :3, :3]]
                if model.reference_bone_orientations is not None:
                    frames.append(model.reference_bone_orientations)
                for frame in frames:
                    self.assertLess((frame.mT @ frame - eye).abs().max(), 1e-12)
                    self.assertTrue((torch.linalg.det(frame) > 0).all())
