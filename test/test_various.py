import unittest
import torch
import anny
import roma
from anny.models.model_data import PHENOTYPE_LABELS

class TestVarious(unittest.TestCase):
    device = torch.device('cpu')
    dtype = torch.float64

    def setUp(self):
        self._deterministic_algorithms = torch.are_deterministic_algorithms_enabled()

    def tearDown(self):
        torch.use_deterministic_algorithms(self._deterministic_algorithms)

    def test_batch_consistency(self):
        batch_size = 32
        model = anny.Anny().to(dtype=self.dtype, device=self.device)
        torch.use_deterministic_algorithms(True)

        joints_relative_transforms = {}
        for k in model.bone_labels:
            rot = roma.random_rotmat(batch_size, dtype=self.dtype, device=self.device)
            joints_relative_transforms[k] = roma.Rigid(rot, torch.zeros((batch_size,3), dtype=self.dtype, device=self.device)).to_homogeneous()
        delta_transforms = model.parse_delta_transforms_dict(joints_relative_transforms)

        generator = None
        phenotype_kwargs = { key : torch.rand((batch_size,), dtype=self.dtype, device=self.device, generator=generator) for key in model.phenotype_labels}

        epsilon = 1e-8
        skinning_methods = ['lbs', 'dqs']
        try:
            import warp
            skinning_methods.append('warp_lbs')
        except ImportError:
            pass

        for skinning_method in skinning_methods:
            model.set_skinning_method(skinning_method)

            # Run the model
            batched_results = model(phenotype_kwargs=phenotype_kwargs, pose_parameters=delta_transforms)

            # Ensure batch consistency by performing computations for a single element
            for i in range(batch_size):
                results = model(phenotype_kwargs={key : value[None,i] for key, value in phenotype_kwargs.items()}, pose_parameters=delta_transforms[None,i])
                for key in batched_results.keys():
                    self.assertTrue(torch.all(torch.abs(batched_results[key][i] - results[key].squeeze(dim=0)) < epsilon))

    def test_default_phenotypes_expand_to_pose_batch_size(self):
        batch_size = 4
        model = anny.Anny().to(dtype=self.dtype, device=self.device)

        pose_parameters = torch.eye(4, dtype=self.dtype, device=self.device)[None, None].expand(batch_size, model.bone_count, 4, 4).clone()

        results = model(pose_parameters=pose_parameters)

        self.assertEqual(results["vertices"].shape[0], batch_size)
        self.assertEqual(results["bone_poses"].shape[0], batch_size)

    def test_single_phenotype_batch_expands_to_pose_batch_size(self):
        batch_size = 4
        model = anny.Anny().to(dtype=self.dtype, device=self.device)

        pose_parameters = torch.eye(4, dtype=self.dtype, device=self.device)[None, None].expand(batch_size, model.bone_count, 4, 4).clone()
        phenotype_kwargs = {key: torch.full((1,), 0.5, dtype=self.dtype, device=self.device) for key in model.phenotype_labels}

        results = model(phenotype_kwargs=phenotype_kwargs, pose_parameters=pose_parameters)

        self.assertEqual(results["vertices"].shape[0], batch_size)
        self.assertEqual(results["bone_poses"].shape[0], batch_size)

    def test_tensor_phenotype_kwargs_forward(self):
        batch_size = 3
        dtype = torch.float64
        device = torch.device("cpu")
        model = anny.Anny().to(dtype=dtype, device=device)

        phenotype_tensor = torch.full(
            (batch_size, len(model.phenotype_labels)),
            0.5,
            dtype=dtype,
            device=device,
        )

        results = model(phenotype_kwargs=phenotype_tensor)

        self.assertEqual(results["vertices"].shape[0], batch_size)
        self.assertEqual(results["rest_vertices"].shape[0], batch_size)
        self.assertEqual(results["bone_poses"].shape[0], batch_size)

    def test_forward_dict_and_tensor_parameters_match(self):
        model = anny.Anny(all_phenotypes=True, local_changes="default").to(
            dtype=self.dtype,
            device=self.device,
        )
        phenotype = torch.rand((2, len(model.phenotype_labels)), dtype=self.dtype)
        local_changes = torch.rand((2, len(model.local_change_labels)), dtype=self.dtype)
        phenotype_dict = dict(zip(model.phenotype_labels, phenotype.unbind(dim=1)))
        local_changes_dict = dict(zip(model.local_change_labels, local_changes.unbind(dim=1)))

        tensor_output = model(
            phenotype_kwargs=phenotype,
            local_changes_kwargs=local_changes,
        )
        dict_output = model(
            phenotype_kwargs=phenotype_dict,
            local_changes_kwargs=local_changes_dict,
        )

        torch.testing.assert_close(tensor_output["vertices"], dict_output["vertices"])
        torch.testing.assert_close(tensor_output["bone_poses"], dict_output["bone_poses"])


    def test_local_changes(self):
        """
        Ensure that default local changes params have no impact on
        """
        batch_size = 32
        model = anny.Anny(rig="anny", all_phenotypes=True).to(dtype=self.dtype, device=self.device)
        model_local_changes = anny.Anny(rig="anny", local_changes="default", all_phenotypes=True).to(dtype=self.dtype, device=self.device)
        torch.use_deterministic_algorithms(True)

        generator = None
        phenotype_kwargs = dict(gender=torch.rand((batch_size,), dtype=self.dtype, device=self.device, generator=generator),
                                age=torch.rand((batch_size,), dtype=self.dtype, device=self.device, generator=generator),
                                muscle=torch.rand((batch_size,), dtype=self.dtype, device=self.device, generator=generator),
                                weight=torch.rand((batch_size,), dtype=self.dtype, device=self.device, generator=generator),
                                height=torch.rand((batch_size,), dtype=self.dtype, device=self.device, generator=generator),
                                proportions=torch.rand((batch_size,), dtype=self.dtype, device=self.device, generator=generator),
                                cupsize=torch.rand((batch_size,), dtype=self.dtype, device=self.device, generator=generator),
                                firmness=torch.rand((batch_size,), dtype=self.dtype, device=self.device, generator=generator),
                                african=torch.rand((batch_size,), dtype=self.dtype, device=self.device, generator=generator),
                                asian=torch.rand((batch_size,), dtype=self.dtype, device=self.device, generator=generator),
                                caucasian=torch.rand((batch_size,), dtype=self.dtype, device=self.device, generator=generator))

        blendshape_coeffs0 = model.get_phenotype_blendshape_coefficients(**phenotype_kwargs)
        rest_model0 = model.get_rest_model(blendshape_coeffs0)

        blendshape_coeffs1 = model_local_changes.get_phenotype_blendshape_coefficients(**phenotype_kwargs)
        rest_model1 = model_local_changes.get_rest_model(blendshape_coeffs1)

        _, phenotype_parameters, _, _ = model_local_changes.get_tensor_inputs(phenotype_kwargs=phenotype_kwargs, pose_parameters=None, local_changes_kwargs=None, facial_actions=None)
        local_changes = torch.zeros((batch_size, len(model_local_changes.local_change_labels)), dtype=self.dtype, device=self.device)
        facial_actions = torch.zeros((batch_size, len(model_local_changes.facial_action_labels)), dtype=self.dtype, device=self.device)

        blendshape_coeffs2 = model_local_changes._get_phenotype_blendshape_coefficients(phenotype_parameters, local_changes, facial_actions)
        rest_model2 = model_local_changes.get_rest_model(blendshape_coeffs2)

        for key in ["rest_vertices", "rest_bone_poses"]:
            self.assertTrue(torch.all(torch.abs(rest_model1[key] - rest_model0[key]) < 1e-3))
            self.assertTrue(torch.all(torch.abs(rest_model2[key] - rest_model0[key]) < 1e-3))
