import unittest
import torch
import roma
import anny
from anny.paths import ANNY_ROOT_DIR
from anny.shape_distribution import SimpleShapeDistribution, MorphologicalAgeMapping
import trimesh
import os
from pathlib import Path

class TestUtils(unittest.TestCase):

    def test_fixed_shape(self):
        """
        Verify that pose parameters can be recovered when the phenotype is fixed.

        A random pose and phenotype are sampled from the model and used to generate
        a target mesh. The ParametersRegressor then optimizes only the pose while
        keeping the phenotype fixed to the ground-truth values.

        The test checks that the reconstructed mesh closely matches the target mesh
        by ensuring that the mean per-vertex error (PVE) is below a small threshold (i.e. 5mm).
        """

        torch.manual_seed(0)

        dtype = torch.float32
        device = torch.device("cpu")

        # create model
        rig = 'default-noeyes-notongue-noexpression-notoes-nohands-nobreasts'
        # rig = 'mixamo'
        model = anny.Anny(rig=rig).to(dtype=dtype, device=device)
        batch_size = 32

        # random pose
        pose_parameters = {}
        for i, bone in enumerate(model.bone_labels):
            # small random rotation vector
            rotvec = 0.2 * torch.randn((batch_size, 3), dtype=dtype, device=device)
            rotmat = roma.rotvec_to_rotmat(rotvec)

            # only the root has translation
            if i == 0:
                translation = torch.randn((batch_size, 3), dtype=dtype, device=device)
            else:
                translation = None

            pose_parameters[bone] = roma.Rigid(linear=rotmat, translation=translation)

        # random phenotype
        shape_dist = SimpleShapeDistribution(
            model,
            morphological_age_distribution=torch.distributions.Uniform(
                low=torch.tensor(20., dtype=model.dtype, device=model.device),
                high=torch.tensor(90., dtype=model.dtype, device=model.device)
            )
        )
        _, phenotype_kwargs = shape_dist.sample(batch_size)

        # generate target mesh
        with torch.no_grad():
            output = model(pose_parameters=pose_parameters, phenotype_kwargs=phenotype_kwargs)

        vertices_target = output['vertices']

        fitter = anny.ParametersRegressor(model=model, verbose=True, max_n_iters=10)

        # fit pose while keeping phenotype fixed
        pose, macro, vertices_hat = fitter(
            vertices_target=vertices_target,
            initial_phenotype_kwargs=phenotype_kwargs,
            optimize_phenotypes=False
        )

        # compute reconstruction error
        pve = torch.norm(vertices_hat - vertices_target, dim=-1).mean() # in mm

        # verify reconstruction is very good
        epsilon = 5e-3 # 5mm tolerance
        self.assertTrue(pve < epsilon, f"Reconstruction error is too high: {1000.*pve:.4f} mm")

    def test_optimize_shape_except_age_gender(self):
        """
        Verify that phenotype optimization works while keeping specific attributes fixed.

        A random pose and phenotype are sampled (adult only) to generate a target mesh. The
        ParametersRegressor is then used to optimize both pose and phenotype
        parameters while explicitly excluding the 'age' and 'gender' attributes
        from optimization (i.e., they remain fixed with 'age'=0.7 and 'gender'=0.5).

        The test verifies that:
            1. The reconstructed mesh matches the target mesh within a reasonable
            vertex error tolerance (i.e. 10 mm).
            2. The excluded phenotype parameters ('age' and 'gender') remain
            unchanged compared to their ground-truth values.

        This test is restricted to adult shapes (age >= 20) to avoid ambiguity
        caused by large morphological changes during growth.
        """

        torch.manual_seed(0)

        dtype = torch.float32
        device = torch.device("cpu")

        rig = 'default-noeyes-notongue-noexpression-notoes-nohands-nobreasts'
        model = anny.Anny(rig=rig).to(dtype=dtype, device=device)

        batch_size = 32

        # random pose
        pose_parameters = {}
        for i, bone in enumerate(model.bone_labels):

            rotvec = 0.2 * torch.randn((batch_size, 3), dtype=dtype, device=device)
            rotmat = roma.rotvec_to_rotmat(rotvec)

            if i == 0:
                translation = torch.randn((batch_size, 3), dtype=dtype, device=device)
            else:
                translation = None

            pose_parameters[bone] = roma.Rigid(linear=rotmat, translation=translation)

        # random phenotype
        shape_dist = SimpleShapeDistribution(
            model,
            morphological_age_distribution=torch.distributions.Uniform(
                low=torch.tensor(20., dtype=model.dtype, device=model.device),
                high=torch.tensor(90., dtype=model.dtype, device=model.device)
            )
        )

        _, phenotype_kwargs = shape_dist.sample(batch_size)

        # generate GT mesh
        with torch.no_grad():
            output = model(
                pose_parameters=pose_parameters,
                phenotype_kwargs=phenotype_kwargs
            )

        vertices_target = output['vertices']

        fitter = anny.ParametersRegressor(
            model=model,
            verbose=True,
            max_n_iters=10
        )

        # fit pose + phenotype but keep age and gender fixed
        initial_phenotype_kwargs = {k: v for k, v in phenotype_kwargs.items() if k not in ["age", "gender"]}
        initial_phenotype_kwargs['age'] = 0.7 * torch.ones(batch_size, dtype=dtype, device=device)  # fixed age (adult)
        initial_phenotype_kwargs['gender'] = 0.5 * torch.ones(batch_size, dtype=dtype, device=device)  # fixed gender (neutral)
        pose, macro, vertices_hat = fitter(
            vertices_target=vertices_target,
            initial_phenotype_kwargs=initial_phenotype_kwargs,
            optimize_phenotypes=True,
            excluded_phenotypes=["age", "gender"]
        )

        # reconstruction error
        pve = torch.norm(vertices_hat - vertices_target, dim=-1).mean()

        epsilon = 1e-2  # 10 mm
        self.assertTrue(
            pve < epsilon,
            f"Reconstruction error too high: {1000.*pve:.4f} mm"
        )

        # check age and gender stayed unchanged
        if "age" in macro:
            self.assertTrue(torch.allclose(macro["age"], initial_phenotype_kwargs["age"], atol=1e-5))

        if "gender" in macro:
            self.assertTrue(torch.allclose(macro["gender"], initial_phenotype_kwargs["gender"], atol=1e-5))
    
    def test_fit_smplx_templates(self):
        """
        Fit Anny model to SMPL-X template OBJ meshes and verify reconstruction error.

        The SMPL-X templates should already match the SMPL-X topology used by Anny.
        This test loads each OBJ, fits pose + phenotype parameters, and verifies that
        the reconstruction error remains small (i.e. below 15 mm mean PVE).
        """

        torch.manual_seed(0)

        dtype = torch.float32
        device = torch.device("cpu")

        rig = 'default'
        topology = 'smplx'
        model = anny.Anny(rig=rig, topology=topology).to(dtype=dtype, device=device)

        boys_state_dict = torch.load(
            Path(ANNY_ROOT_DIR / "data" / "shape_calibration/boys.pth"),
            weights_only=True,
            map_location="cpu"
        )
        age_mapping = MorphologicalAgeMapping()
        age_mapping.load_state_dict(boys_state_dict["morphological_age_mapping"])

        from anny.paths import download_noncommercial_data, get_anny_cache_path
        download_noncommercial_data()
        data_dir = get_anny_cache_path() / "noncommercial" / "smplx_examples"
        obj_files = [
            data_dir / "smplx_male_template.obj",
            data_dir / "smplx_female_template.obj",
            data_dir / "smplx_neutral_template.obj"
        ]

        fitter = anny.ParametersRegressor(
            model=model,
            verbose=True,
            max_n_iters=10
        )

        for obj_path in obj_files:

            if not os.path.exists(obj_path):
                print(f"Skipping {obj_path} (file not found).")
                continue

            mesh = trimesh.load(obj_path, process=False)

            vertices_target = torch.tensor(
                mesh.vertices,
                dtype=dtype,
                device=device
            )[None]  # batch=1

            # initialize phenotype parameters to neutral values
            initial_phenotype_kwargs = {key: 0.5 * torch.ones(1, dtype=dtype, device=device) for key in model.phenotype_labels}
            if 'female' in str(obj_path):
                initial_phenotype_kwargs['gender'] = 1.0 * torch.ones(1, dtype=dtype, device=device)
            elif 'male' in str(obj_path):
                initial_phenotype_kwargs['gender'] = 0.0 * torch.ones(1, dtype=dtype, device=device)
            else:
                initial_phenotype_kwargs['gender'] = 0.5 * torch.ones(1, dtype=dtype, device=device)
            
            morpho_age = 20
            initial_phenotype_kwargs['age'] = age_mapping.morphological_to_anny_age(torch.Tensor([morpho_age])).to(dtype=dtype, device=device)

            pose, macro, vertices_hat = fitter(
                vertices_target=vertices_target,
                initial_phenotype_kwargs=initial_phenotype_kwargs,
                excluded_phenotypes=['age', 'gender'],
                optimize_phenotypes=True
            )

            # compute reconstruction error
            pve = torch.norm(vertices_hat - vertices_target, dim=-1).mean()

            print(f"{obj_path} → mean PVE: {1000.*pve:.4f} mm")

            epsilon = 0.015  # 15 mm tolerance
            self.assertTrue(
                pve < epsilon,
                f"Reconstruction error too high for {obj_path}: {1000.*pve:.4f} mm"
            )
    