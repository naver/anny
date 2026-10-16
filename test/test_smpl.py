# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0

import unittest
import os

import torch

from anny.models.rigged_model import RiggedModelWithLinearBlendShapes

import anny

try:
    from anny.models.smpl import SMPL, SMPLH, SMPLX
except ImportError:
    SMPL = SMPLH = SMPLX = None

SMPLX_MODEL_PATH = os.environ.get("SMPLX_MODEL_PATH")

requires_smpl_models = unittest.skipIf(
    SMPL is None or SMPLX_MODEL_PATH is None,
    "SMPL-family tests need anny[smpl] and SMPLX_MODEL_PATH",
)

DTYPE = torch.float32


# Shared helpers -----------------------------------------------------------------------------


def zero_inputs(dims, batch_size=1):
    """Zero-valued forward inputs from (name, dimension) pairs."""
    return {name: torch.zeros((batch_size, dim), dtype=DTYPE) for name, dim in dims}


def reference_inputs(dims, batch_size=2):
    """Deterministic, non-trivial forward inputs from (name, dimension) pairs.

    Each input is a linspace of moderate amplitude, with alternating signs across inputs so
    that no two inputs carry the same values.
    """
    return {
        name: (-1.0) ** i
        * torch.linspace(-0.25, 0.25, batch_size * dim, dtype=DTYPE).reshape(
            batch_size, dim
        )
        for i, (name, dim) in enumerate(dims)
    }


def reference_model(model_type, **kwargs):
    """The model of the smplx package that an anny model is compared against."""
    import smplx

    return smplx.create(SMPLX_MODEL_PATH, model_type=model_type, **kwargs)


def assert_anny_topology(test, model, inputs):
    """The model is retopologized onto the pruned Anny mesh and usable as built."""
    full_anny_vertex_count = anny.Anny(
        "anny", topology="makehuman"
    ).template_vertices.shape[0]
    # All vertices must be referenced by at least one face (no unattached vertices)
    test.assertEqual(
        torch.unique(model.faces.flatten()).shape[0],
        model.template_vertices.shape[0],
    )
    # Pruned mesh must be a strict subset of the full Anny topology
    test.assertLess(model.template_vertices.shape[0], full_anny_vertex_count)
    test.assertTrue(torch.all(model.base_mesh_vertex_indices < full_anny_vertex_count))
    # No unbound (zero skinning weight) vertex should survive the pruning
    test.assertTrue(torch.all(model.vertex_bone_weights.sum(dim=-1) > 0.5))
    # The retopologized model must be usable as built, without a dtype cast
    output = model(**inputs)
    test.assertEqual(output["vertices"].dtype, DTYPE)
    test.assertTrue(torch.all(torch.isfinite(output["vertices"])))


def assert_matches_reference_model(test, model, reference, inputs, tolerance=1e-5):
    """Vertices and bone positions agree with the reference model on the same inputs."""
    model = model.to(dtype=DTYPE)
    model.set_skinning_method("lbs")
    reference_output = reference(**inputs)
    model_output = model(**inputs)

    vertex_error = torch.linalg.norm(
        model_output["vertices"] - reference_output.vertices, dim=-1
    )
    joint_error = torch.linalg.norm(
        model_output["bone_poses"][:, :, :3, 3]
        - reference_output.joints[:, : model.bone_count],
        dim=-1,
    )
    test.assertLess(vertex_error.max().item(), tolerance)
    test.assertLess(joint_error.max().item(), tolerance)


# SMPL ---------------------------------------------------------------------------------------


def smpl_input_dims(num_betas=10):
    return (
        ("betas", num_betas),
        ("global_orient", 3),
        ("transl", 3),
        ("body_pose", 23 * 3),
    )


@requires_smpl_models
class TestSMPL(unittest.TestCase):
    def test_is_normal_rigged_model_instance(self):
        model = SMPL(SMPLX_MODEL_PATH, pose_corrective=True)

        self.assertIsInstance(model, RiggedModelWithLinearBlendShapes)
        self.assertEqual(model.bone_orientation, "blender")
        self.assertEqual(model.bone_count, 24)
        self.assertTrue(hasattr(model, "faces"))
        self.assertEqual(model.template_vertices.shape[0], 6890)

    def test_with_anny_topology(self):
        model = SMPL(SMPLX_MODEL_PATH, topology="anny")
        assert_anny_topology(self, model, zero_inputs(smpl_input_dims()))

    def test_forward_builds_batched_pose_and_coefficients(self):
        model = SMPL(SMPLX_MODEL_PATH, pose_corrective=True)
        inputs = zero_inputs(smpl_input_dims(), batch_size=2)
        inputs["transl"] = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=DTYPE)

        _ = model(**inputs)

    def test_rejects_other_model_types(self):
        with self.assertRaises(ValueError):
            SMPL(SMPLX_MODEL_PATH, model_type="smplx")

    def test_forward_matches_reference_model(self):
        reference = reference_model("smpl", gender="neutral", batch_size=2)
        model = SMPL(
            SMPLX_MODEL_PATH, gender="neutral", pose_corrective=True, topology="smpl"
        )
        assert_matches_reference_model(
            self, model, reference, reference_inputs(smpl_input_dims())
        )


# SMPL-X -------------------------------------------------------------------------------------


def smplx_input_dims(hand_pose_dim=6):
    return (
        ("betas", 10),
        ("expression", 10),
        ("global_orient", 3),
        ("transl", 3),
        ("body_pose", 21 * 3),
        ("jaw_pose", 3),
        ("leye_pose", 3),
        ("reye_pose", 3),
        ("left_hand_pose", hand_pose_dim),
        ("right_hand_pose", hand_pose_dim),
    )


@requires_smpl_models
class TestSMPLX(unittest.TestCase):
    def test_is_normal_rigged_model_instance(self):
        model = SMPLX(SMPLX_MODEL_PATH, pose_corrective=True)

        self.assertIsInstance(model, RiggedModelWithLinearBlendShapes)
        self.assertEqual(model.bone_orientation, "blender")
        self.assertEqual(model.bone_count, 55)
        self.assertTrue(hasattr(model, "faces"))

    def test_with_anny_topology(self):
        model = SMPLX(SMPLX_MODEL_PATH, pose_corrective=True, topology="anny")
        assert_anny_topology(self, model, zero_inputs(smplx_input_dims()))

    def test_forward_builds_batched_pose_and_coefficients_pca(self):
        model = SMPLX(SMPLX_MODEL_PATH, pose_corrective=True)
        inputs = zero_inputs(smplx_input_dims(hand_pose_dim=6), batch_size=2)
        inputs["transl"] = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=DTYPE)

        _ = model(**inputs)

    def test_forward_builds_batched_pose_and_coefficients_no_pca(self):
        model = SMPLX(SMPLX_MODEL_PATH, pose_corrective=True, use_pca=False)
        inputs = zero_inputs(smplx_input_dims(hand_pose_dim=15 * 3), batch_size=2)
        inputs["transl"] = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=DTYPE)

        _ = model(**inputs)

    def test_rejects_other_model_types(self):
        with self.assertRaises(ValueError):
            SMPLX(SMPLX_MODEL_PATH, model_type="smpl")

    def test_forward_matches_reference_model(self):
        reference = reference_model(
            "smplx", gender="neutral", use_pca=True, num_pca_comps=6, batch_size=2
        )
        model = SMPLX(SMPLX_MODEL_PATH, gender="neutral", use_pca=True, num_pca_comps=6)
        assert_matches_reference_model(
            self, model, reference, reference_inputs(smplx_input_dims(hand_pose_dim=6))
        )


# SMPL+H -------------------------------------------------------------------------------------


def smplh_input_dims(num_betas=10, hand_pose_dim=6):
    return (
        ("betas", num_betas),
        ("global_orient", 3),
        ("transl", 3),
        ("body_pose", 21 * 3),
        ("left_hand_pose", hand_pose_dim),
        ("right_hand_pose", hand_pose_dim),
    )


@requires_smpl_models
class TestSMPLH(unittest.TestCase):
    genders = ("male", "female", "neutral")

    def require_smplh_pkl(self, gender):
        filename = f"SMPLH_{gender.upper()}.pkl"
        if not os.path.isfile(os.path.join(SMPLX_MODEL_PATH, "smplh", filename)):
            self.skipTest(f"{filename} is not in SMPLX_MODEL_PATH/smplh")

    def find_extended_npz(self, gender):
        for folder in ("smplh", "smplh_raw"):
            path = os.path.join(SMPLX_MODEL_PATH, folder, gender, "model.npz")
            if os.path.isfile(path):
                return path
        return None

    def check_matches_reference_model(self, use_pca, num_pca_comps, flat_hand_mean):
        hand_pose_dim = num_pca_comps if use_pca else 15 * 3
        for gender in self.genders:
            with self.subTest(gender=gender):
                self.require_smplh_pkl(gender)
                reference = reference_model(
                    "smplh",
                    gender=gender,
                    ext="pkl",
                    use_pca=use_pca,
                    num_pca_comps=num_pca_comps,
                    flat_hand_mean=flat_hand_mean,
                    batch_size=2,
                )
                model = SMPLH(
                    SMPLX_MODEL_PATH,
                    gender=gender,
                    use_pca=use_pca,
                    num_pca_comps=num_pca_comps,
                    flat_hand_mean=flat_hand_mean,
                )
                assert_matches_reference_model(
                    self,
                    model,
                    reference,
                    reference_inputs(smplh_input_dims(hand_pose_dim=hand_pose_dim)),
                )

    def test_is_normal_rigged_model_instance(self):
        self.require_smplh_pkl("neutral")
        model = SMPLH(SMPLX_MODEL_PATH)

        self.assertIsInstance(model, RiggedModelWithLinearBlendShapes)
        self.assertEqual(model.bone_orientation, "blender")
        self.assertEqual(model.bone_count, 52)
        self.assertEqual(model.template_vertices.shape[0], 6890)
        self.assertTrue(
            torch.equal(model.faces, SMPL(SMPLX_MODEL_PATH, topology="smpl").faces)
        )

    def test_with_anny_topology(self):
        self.require_smplh_pkl("neutral")
        model = SMPLH(SMPLX_MODEL_PATH, topology="anny")
        assert_anny_topology(self, model, zero_inputs(smplh_input_dims()))

    def test_forward_matches_reference_model_full_hand_pose(self):
        self.check_matches_reference_model(
            use_pca=False, num_pca_comps=6, flat_hand_mean=True
        )

    def test_forward_matches_reference_model_pca(self):
        self.check_matches_reference_model(
            use_pca=True, num_pca_comps=12, flat_hand_mean=False
        )

    def test_extended_npz_uses_16_betas(self):
        mano_model_path = os.path.join(SMPLX_MODEL_PATH, "mano")
        if not os.path.isfile(os.path.join(mano_model_path, "MANO_LEFT.pkl")):
            self.skipTest("MANO model files are not in SMPLX_MODEL_PATH/mano")
        for gender in self.genders:
            with self.subTest(gender=gender):
                npz_path = self.find_extended_npz(gender)
                if npz_path is None:
                    self.skipTest("extended SMPL+H model.npz files are not available")
                model = SMPLH(
                    npz_path,
                    num_betas=16,
                    use_pca=True,
                    num_pca_comps=12,
                    mano_model_path=mano_model_path,
                ).to(dtype=DTYPE)
                self.assertEqual(model.bone_count, 52)

                inputs = reference_inputs(
                    smplh_input_dims(num_betas=16, hand_pose_dim=12)
                )
                truncated_betas = inputs["betas"].clone()
                truncated_betas[:, 10:] = 0.0
                vertices = model(**inputs)["vertices"]
                inputs["betas"] = truncated_betas
                truncated_vertices = model(**inputs)["vertices"]

                self.assertEqual(vertices.shape, (2, 6890, 3))
                self.assertTrue(torch.all(torch.isfinite(vertices)))
                self.assertGreater(
                    (vertices - truncated_vertices).abs().max().item(), 1e-4
                )


# Gradients ----------------------------------------------------------------------------------


def random_inputs(dims, batch_size=2, dtype=DTYPE, seed=0):
    """Random forward inputs from (name, dimension) pairs, all requiring gradients."""
    generator = torch.Generator().manual_seed(seed)
    return {
        name: (
            0.3 * torch.randn((batch_size, dim), generator=generator, dtype=dtype)
        ).requires_grad_()
        for name, dim in dims
    }


def vertices_and_joints_loss(model, inputs):
    output = model(**inputs)
    return output["vertices"].sum() + output["bone_poses"][:, :, :3, 3].sum()


def smpl_family():
    """(model class, model file, model arguments, forward input dimensions) for each
    SMPL-family model."""
    return (
        (SMPL, os.path.join("smpl", "SMPL_NEUTRAL.pkl"), {}, smpl_input_dims()),
        (
            SMPLH,
            os.path.join("smplh", "SMPLH_NEUTRAL.pkl"),
            {"use_pca": False},
            smplh_input_dims(hand_pose_dim=15 * 3),
        ),
        (
            SMPLH,
            os.path.join("smplh", "SMPLH_NEUTRAL.pkl"),
            {"use_pca": True, "num_pca_comps": 6},
            smplh_input_dims(hand_pose_dim=6),
        ),
        (
            SMPLX,
            os.path.join("smplx", "SMPLX_NEUTRAL.npz"),
            {"use_pca": False},
            smplx_input_dims(hand_pose_dim=15 * 3),
        ),
        (
            SMPLX,
            os.path.join("smplx", "SMPLX_NEUTRAL.npz"),
            {"use_pca": True, "num_pca_comps": 6},
            smplx_input_dims(hand_pose_dim=6),
        ),
    )


def check_each_model(test, check):
    """Run check(model, dims) on each SMPL-family model whose file is available."""
    for model_class, filename, kwargs, dims in smpl_family():
        with test.subTest(model=model_class.__name__, **kwargs):
            if not os.path.isfile(os.path.join(SMPLX_MODEL_PATH, filename)):
                test.skipTest(f"{filename} is not in SMPLX_MODEL_PATH")
            check(model_class(SMPLX_MODEL_PATH, **kwargs), dims)


@requires_smpl_models
class TestSMPLFamilyDtypes(unittest.TestCase):
    def test_forward_supports_dtypes(self):
        def check(model, dims):
            for dtype in (torch.float32, torch.float64):
                with self.subTest(dtype=dtype):
                    model = model.to(dtype=dtype)
                    inputs = {
                        name: value.to(dtype)
                        for name, value in zero_inputs(dims).items()
                    }
                    self.assertEqual(model(**inputs)["vertices"].dtype, dtype)

        check_each_model(self, check)


@requires_smpl_models
class TestSMPLFamilyGradients(unittest.TestCase):
    """Synthetic bone tails make every bone direction degenerate (zero cross product
    with the y axis); their gradients must stay finite all the same."""

    def test_gradients_are_finite(self):
        def check(model, dims):
            model = model.to(dtype=DTYPE)
            inputs = random_inputs(dims)
            vertices_and_joints_loss(model, inputs).backward()
            for name, value in inputs.items():
                self.assertIsNotNone(value.grad, name)
                self.assertTrue(torch.all(torch.isfinite(value.grad)), name)

        check_each_model(self, check)

    def test_gradient_matches_finite_differences(self):
        def check(model, dims):
            model = model.to(dtype=torch.float64)
            model.set_skinning_method("lbs")
            inputs = random_inputs(dims, batch_size=1, dtype=torch.float64)
            vertices_and_joints_loss(model, inputs).backward()
            step = 1e-6
            for name, index in (
                ("betas", 0),
                ("betas", 5),
                ("body_pose", 0),
                ("body_pose", 10),
                ("body_pose", 40),
            ):
                value = inputs[name]
                original = value[0, index].item()
                with torch.no_grad():
                    value[0, index] = original + step
                    loss_plus = vertices_and_joints_loss(model, inputs)
                    value[0, index] = original - step
                    loss_minus = vertices_and_joints_loss(model, inputs)
                    value[0, index] = original
                finite_difference = (loss_plus - loss_minus).item() / (2 * step)
                self.assertAlmostEqual(
                    value.grad[0, index].item(),
                    finite_difference,
                    delta=1e-5 * max(1.0, abs(finite_difference)),
                    msg=f"{name}[{index}]",
                )

        check_each_model(self, check)


if __name__ == "__main__":
    unittest.main()
