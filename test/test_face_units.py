import unittest

import roma
import torch

import anny
from anny.paths import ANNY_ROOT_DIR
from anny.models.face_units import FACE_UNIT_LABELS, load_face_unit_blendshapes


class TestFaceUnits(unittest.TestCase):
    dtype = torch.float64
    device = torch.device("cpu")

    @classmethod
    def setUpClass(cls):
        cls.model = anny.Anny(
            face_units=True,
            remove_unattached_vertices=False,
        ).to(dtype=cls.dtype, device=cls.device)

    def _values(self, batch_size=1, **kwargs):
        values = torch.zeros(
            (batch_size, len(self.model.face_unit_labels)),
            dtype=self.dtype,
            device=self.device,
        )
        for label, value in kwargs.items():
            values[:, self.model.face_unit_labels.index(label)] = value
        return values

    def test_face_unit_label_count_and_order(self):
        self.assertEqual(len(FACE_UNIT_LABELS), 52)
        self.assertEqual(len(set(FACE_UNIT_LABELS)), 52)
        self.assertEqual(self.model.face_unit_labels, FACE_UNIT_LABELS)
        self.assertEqual(self.model.face_unit_labels[0], "browDownLeft")
        self.assertEqual(self.model.face_unit_labels[-1], "tongueOut")
        self.assertIn("jawOpen", self.model.face_unit_labels)
        self.assertIn("mouthSmileLeft", self.model.face_unit_labels)

    def test_plain_target_loader_returns_canonical_stack(self):
        world_transformation = roma.Linear(
            0.1 * roma.euler_to_rotmat("X", [90], degrees=True, dtype=self.dtype)
        )[None]
        labels, blendshapes = load_face_unit_blendshapes(
            root_dirname=ANNY_ROOT_DIR,
            vertices_count=self.model.template_vertices.shape[0],
            world_transformation=world_transformation,
            dtype=self.dtype,
        )

        self.assertEqual(labels, FACE_UNIT_LABELS)
        self.assertEqual(
            blendshapes.shape,
            (52, self.model.template_vertices.shape[0], 3),
        )
        jaw_open = blendshapes[labels.index("jawOpen")]
        self.assertGreater(torch.count_nonzero(jaw_open).item(), 0)

    def test_dict_input_changes_output_vertices(self):
        zero_output = self.model(face_units={})
        moved_output = self.model(face_units={"jawOpen": 1.0})

        self.assertFalse(
            torch.allclose(
                zero_output["vertices"],
                moved_output["vertices"],
                atol=0.0,
                rtol=0.0,
            )
        )

    def test_tensor_input_matches_equivalent_dict_input(self):
        values = self._values(
            batch_size=2,
            jawOpen=0.8,
            mouthSmileLeft=0.4,
            eyeBlinkRight=0.25,
        )
        dict_values = {
            "jawOpen": torch.full((2,), 0.8, dtype=self.dtype, device=self.device),
            "mouthSmileLeft": torch.full((2,), 0.4, dtype=self.dtype, device=self.device),
            "eyeBlinkRight": torch.full((2,), 0.25, dtype=self.dtype, device=self.device),
        }

        tensor_output = self.model(face_units=values)
        dict_output = self.model(face_units=dict_values)

        torch.testing.assert_close(tensor_output["rest_vertices"], dict_output["rest_vertices"])
        torch.testing.assert_close(tensor_output["vertices"], dict_output["vertices"])

    def test_out_of_range_dict_values_raise_value_error(self):
        with self.assertRaisesRegex(ValueError, "Face unit values must be in \\[0, 1\\]"):
            self.model(face_units={"jawOpen": -0.1})
        with self.assertRaisesRegex(ValueError, "Face unit values must be in \\[0, 1\\]"):
            self.model(face_units={"jawOpen": 1.1})

    def test_out_of_range_tensor_values_raise_value_error(self):
        values = self._values(jawOpen=0.5)
        values[:, self.model.face_unit_labels.index("jawOpen")] = 1.2

        with self.assertRaisesRegex(ValueError, "Face unit values must be in \\[0, 1\\]"):
            self.model(face_units=values)

    def test_unknown_dict_label_raises_value_error(self):
        with self.assertRaisesRegex(ValueError, "Unknown face unit labels"):
            self.model(face_units={"notAUnit": 0.5})

    def test_wrong_tensor_shape_raises_value_error(self):
        with self.assertRaisesRegex(ValueError, "face_units tensor must have shape"):
            self.model(face_units=torch.zeros((1, 53), dtype=self.dtype))
        with self.assertRaisesRegex(ValueError, "face_units tensor must have shape"):
            self.model(face_units=torch.zeros((52,), dtype=self.dtype))

    def test_default_model_rejects_non_empty_face_units(self):
        model = anny.Anny().to(dtype=self.dtype, device=self.device)
        self.assertEqual(model.face_unit_labels, [])

        with self.assertRaisesRegex(ValueError, "built with face_units='none'"):
            model(face_units={"jawOpen": 0.5})

    def test_empty_face_units_are_allowed_on_default_model(self):
        model = anny.Anny().to(dtype=self.dtype, device=self.device)
        output = model(face_units={})

        self.assertEqual(output["vertices"].shape[0], 1)

    def test_face_unit_scalar_dict_expands_to_pose_batch(self):
        batch_size = 3
        pose_parameters = torch.eye(4, dtype=self.dtype, device=self.device)[
            None, None
        ].expand(batch_size, self.model.bone_count, 4, 4).clone()

        output = self.model(
            pose_parameters=pose_parameters,
            face_units={"jawOpen": 0.5},
        )

        self.assertEqual(output["vertices"].shape[0], batch_size)
        self.assertEqual(output["bone_poses"].shape[0], batch_size)

    def test_face_unit_tensor_batch_defines_output_batch(self):
        values = self._values(batch_size=4, jawOpen=0.25)

        output = self.model(face_units=values)

        self.assertEqual(output["vertices"].shape[0], 4)
        self.assertEqual(output["rest_vertices"].shape[0], 4)

    def test_head_model_exposes_labels_and_accepts_input(self):
        model = anny.create_head_model(face_units=True).to(
            dtype=self.dtype,
            device=self.device,
        )

        output = model(face_units={"jawOpen": 0.5})

        self.assertEqual(model.face_unit_labels, FACE_UNIT_LABELS)
        self.assertEqual(output["vertices"].shape[0], 1)
        self.assertEqual(output["vertices"].shape[1], model.template_vertices.shape[0])

    def test_builtin_retopology_preserves_labels_and_accepts_input(self):
        model = anny.Anny(topology="notoes", face_units=True).to(
            dtype=self.dtype,
            device=self.device,
        )

        output = model(face_units={"jawOpen": 0.5})

        self.assertEqual(model.face_unit_labels, FACE_UNIT_LABELS)
        self.assertEqual(output["vertices"].shape[1], model.template_vertices.shape[0])


if __name__ == "__main__":
    unittest.main()
