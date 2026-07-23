import unittest

import torch

import anny
from anny.models.facial_actions import FACIAL_ACTION_LABELS


class TestFacialActions(unittest.TestCase):
    dtype = torch.float64
    device = torch.device("cpu")

    @classmethod
    def setUpClass(cls):
        cls.model = anny.Anny(
            facial_actions=True,
            topology="anny"
        ).to(dtype=cls.dtype, device=cls.device)

    def _values(self, batch_size=1, **kwargs):
        values = torch.zeros(
            (batch_size, len(self.model.facial_action_labels)),
            dtype=self.dtype,
            device=self.device,
        )
        for label, value in kwargs.items():
            values[:, self.model.facial_action_labels.index(label)] = value
        return values

    def test_facial_action_label_count_and_order(self):
        self.assertEqual(len(FACIAL_ACTION_LABELS), 52)
        self.assertEqual(len(set(FACIAL_ACTION_LABELS)), 52)
        self.assertEqual(self.model.facial_action_labels, FACIAL_ACTION_LABELS)
        self.assertEqual(self.model.facial_action_labels[0], "browDownLeft")
        self.assertEqual(self.model.facial_action_labels[-1], "tongueOut")
        self.assertIn("jawOpen", self.model.facial_action_labels)
        self.assertIn("mouthSmileLeft", self.model.facial_action_labels)

    def test_dict_input_changes_output_vertices(self):
        zero_output = self.model(facial_actions={})
        moved_output = self.model(facial_actions={"jawOpen": 1.0})

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

        tensor_output = self.model(facial_actions=values)
        dict_output = self.model(facial_actions=dict_values)

        torch.testing.assert_close(tensor_output["rest_vertices"], dict_output["rest_vertices"])
        torch.testing.assert_close(tensor_output["vertices"], dict_output["vertices"])

    def test_unknown_dict_label_raises_value_error(self):
        with self.assertRaisesRegex(ValueError, "Unknown face unit labels"):
            self.model(facial_actions={"notAUnit": 0.5})

    def test_wrong_tensor_shape_raises_value_error(self):
        with self.assertRaisesRegex(ValueError, "facial_actions tensor must have shape"):
            self.model(facial_actions=torch.zeros((1, 53), dtype=self.dtype))
        with self.assertRaisesRegex(ValueError, "facial_actions tensor must have shape"):
            self.model(facial_actions=torch.zeros((52,), dtype=self.dtype))

    def test_empty_facial_actions_are_allowed_on_default_model(self):
        model = anny.Anny().to(dtype=self.dtype, device=self.device)
        output = model(facial_actions={})

        self.assertEqual(output["vertices"].shape[0], 1)

    def test_disabled_facial_actions_are_filtered(self):
        model = anny.Anny(facial_actions=False)

        self.assertEqual(model.facial_action_labels, [])
        self.assertFalse(any(
            label.startswith("facial_action:")
            for label in model.blendshape_labels
        ))
        with self.assertRaisesRegex(
            ValueError,
            "model was built with facial_actions='none'",
        ):
            model(facial_actions={"jawOpen": 0.5})

    def test_facial_action_scalar_dict_expands_to_pose_batch(self):
        batch_size = 3
        pose_parameters = torch.eye(4, dtype=self.dtype, device=self.device)[
            None, None
        ].expand(batch_size, self.model.bone_count, 4, 4).clone()

        output = self.model(
            pose_parameters=pose_parameters,
            facial_actions={"jawOpen": 0.5},
        )

        self.assertEqual(output["vertices"].shape[0], batch_size)
        self.assertEqual(output["bone_poses"].shape[0], batch_size)

    def test_facial_action_tensor_batch_defines_output_batch(self):
        values = self._values(batch_size=4, jawOpen=0.25)

        output = self.model(facial_actions=values)

        self.assertEqual(output["vertices"].shape[0], 4)
        self.assertEqual(output["rest_vertices"].shape[0], 4)

    def test_head_model_exposes_labels_and_accepts_input(self):
        model = anny.create_head_model(facial_actions=True).to(
            dtype=self.dtype,
            device=self.device,
        )

        output = model(facial_actions={"jawOpen": 0.5})

        self.assertEqual(model.facial_action_labels, FACIAL_ACTION_LABELS)
        self.assertEqual(output["vertices"].shape[0], 1)
        self.assertEqual(output["vertices"].shape[1], model.template_vertices.shape[0])

    def test_builtin_retopology_preserves_labels_and_accepts_input(self):
        model = anny.Anny(topology="notoes", facial_actions=True).to(
            dtype=self.dtype,
            device=self.device,
        )

        output = model(facial_actions={"jawOpen": 0.5})

        self.assertEqual(model.facial_action_labels, FACIAL_ACTION_LABELS)
        self.assertEqual(output["vertices"].shape[1], model.template_vertices.shape[0])


if __name__ == "__main__":
    unittest.main()
