import unittest
import torch
import anny
import roma

class TestVarious(unittest.TestCase):
    def test_batch_consistency(self):
        batch_size = 32
        dtype = torch.float64
        device = torch.device('cpu')
        model = anny.Anny().to(dtype=dtype, device=device)
        torch.use_deterministic_algorithms(True)

        joints_relative_transforms = {}
        for k in model.bone_labels:
            rot = roma.random_rotmat(batch_size, dtype=dtype, device=device)
            joints_relative_transforms[k] = roma.Rigid(rot, torch.zeros((batch_size,3), dtype=dtype, device=device)).to_homogeneous()      
        delta_transforms = model.parse_delta_transforms_dict(joints_relative_transforms)

        generator = None
        phenotype_kwargs = { key : torch.rand((batch_size,), dtype=dtype, device=device, generator=generator) for key in model.phenotype_labels}

        epsilon = 1e-8
        for skinning_method in ['lbs', 'dqs', 'warp_lbs']:
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
        dtype = torch.float64
        device = torch.device('cpu')
        model = anny.Anny().to(dtype=dtype, device=device)

        pose_parameters = torch.eye(4, dtype=dtype, device=device)[None, None].expand(batch_size, model.bone_count, 4, 4).clone()

        results = model(pose_parameters=pose_parameters)

        self.assertEqual(results["vertices"].shape[0], batch_size)
        self.assertEqual(results["bone_poses"].shape[0], batch_size)

    def test_single_phenotype_batch_expands_to_pose_batch_size(self):
        batch_size = 4
        dtype = torch.float64
        device = torch.device('cpu')
        model = anny.Anny().to(dtype=dtype, device=device)

        pose_parameters = torch.eye(4, dtype=dtype, device=device)[None, None].expand(batch_size, model.bone_count, 4, 4).clone()
        phenotype_kwargs = {key: torch.full((1,), 0.5, dtype=dtype, device=device) for key in model.phenotype_labels}

        results = model(phenotype_kwargs=phenotype_kwargs, pose_parameters=pose_parameters)

        self.assertEqual(results["vertices"].shape[0], batch_size)
        self.assertEqual(results["bone_poses"].shape[0], batch_size)

    def test_local_changes(self):
        """
        Ensure that default local changes params have no impact on 
        """
        batch_size = 32
        dtype = torch.float64
        device = torch.device('cpu')
        model = anny.Anny().to(dtype=dtype, device=device)
        model_local_changes = anny.Anny(local_changes="default").to(dtype=dtype, device=device)
        torch.use_deterministic_algorithms(True)

        generator = None
        phenotype_kwargs = dict(gender=torch.rand((batch_size,), dtype=dtype, device=device, generator=generator),
                                age=torch.rand((batch_size,), dtype=dtype, device=device, generator=generator),
                                muscle=torch.rand((batch_size,), dtype=dtype, device=device, generator=generator),
                                weight=torch.rand((batch_size,), dtype=dtype, device=device, generator=generator),
                                height=torch.rand((batch_size,), dtype=dtype, device=device, generator=generator),
                                proportions=torch.rand((batch_size,), dtype=dtype, device=device, generator=generator),
                                cupsize=torch.rand((batch_size,), dtype=dtype, device=device, generator=generator),
                                firmness=torch.rand((batch_size,), dtype=dtype, device=device, generator=generator),
                                african=torch.rand((batch_size,), dtype=dtype, device=device, generator=generator),
                                asian=torch.rand((batch_size,), dtype=dtype, device=device, generator=generator),
                                caucasian=torch.rand((batch_size,), dtype=dtype, device=device, generator=generator))
        
        blendshape_coeffs0 = model.get_phenotype_blendshape_coefficients(**phenotype_kwargs)
        rest_model0 = model.get_rest_model(blendshape_coeffs0)

        blendshape_coeffs1 = model_local_changes.get_phenotype_blendshape_coefficients(**phenotype_kwargs)
        rest_model1 = model_local_changes.get_rest_model(blendshape_coeffs1)

        blendshape_coeffs2 = model_local_changes.get_phenotype_blendshape_coefficients(**phenotype_kwargs, local_changes={key: torch.zeros((batch_size,), dtype=dtype, device=device) for key in model_local_changes.local_change_labels})
        rest_model2 = model_local_changes.get_rest_model(blendshape_coeffs2)

        for key in ["rest_vertices", "rest_bone_heads", "rest_bone_tails", "rest_bone_poses"]:
            self.assertTrue(torch.all(torch.abs(rest_model1[key] - rest_model0[key]) < 1e-3))
            self.assertTrue(torch.all(torch.abs(rest_model2[key] - rest_model0[key]) < 1e-3))