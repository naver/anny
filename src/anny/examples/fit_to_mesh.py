# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0

import torch
import roma
import anny
import time
import trimesh
from typing import List, Optional
from anny.shape_distribution import SimpleShapeDistribution


def fit_to_mesh(
    seed: int = 3993,
    N: int = 1,
    max_n_iters: int = 5,
    verbose: bool = False,
    batch_size: int = 1,
    eps: float = 0.1,
    excluded_phenotypes: Optional[List[str]] = ["age", "gender"],
    max_delta: float = 0.1,
    optimize_phenotypes: bool = True,
    rig: str = "anny",
    topology: str = "anny",
    n_points: int = 5000,
):
    print("seed", seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    dtype = torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = anny.Anny(rig="mixamo", topology=topology).to(dtype=dtype, device=device)

    # generate a target mesh to fit
    pose_parameters = {}
    pose_parameters["mixamorig:Hips"] = roma.Rigid(
        linear=roma.random_rotmat(batch_size, dtype=dtype, device=device),
        translation=torch.randn((batch_size, 3), dtype=dtype, device=device),
    )
    pose_parameters["mixamorig:RightLeg"] = roma.Rigid(
        linear=roma.rotvec_to_rotmat(
            torch.Tensor([[1.5, 0, 0]])
            .to(dtype=dtype, device=device)
            .repeat(batch_size, 1)
        ),
        translation=None,
    )
    pose_parameters["mixamorig:LeftArm"] = roma.Rigid(
        linear=roma.rotvec_to_rotmat(
            torch.Tensor([[0, 2.0, 0]])
            .to(dtype=dtype, device=device)
            .repeat(batch_size, 1)
        ),
        translation=None,
    )
    shape_dist = SimpleShapeDistribution(
        model,
        morphological_age_distribution=torch.distributions.Uniform(
            low=torch.tensor(20, dtype=model.dtype, device=model.device),
            high=torch.tensor(90.0, dtype=model.dtype, device=model.device),
        ),
    )
    age, phenotype_kwargs = shape_dist.sample(batch_size)

    output = model(pose_parameters=pose_parameters, phenotype_kwargs=phenotype_kwargs)

    # instantate fitter and find the best pose/shape
    _model = anny.Anny(rig=rig, topology=topology).to(dtype=dtype, device=device)
    fitter = anny.ParametersRegressor(
        _model, verbose=verbose, max_n_iters=max_n_iters, eps=eps, n_points=n_points
    )

    # fitting by optimizing both pose and all phenotypes
    start = time.time()
    # ensure default excluded_phenotypes is an empty list
    excluded_phenotypes = excluded_phenotypes or []

    for i in range(N):
        initial_phenotype_kwargs = {} if optimize_phenotypes else phenotype_kwargs
        pose, macro, vertices_hat = fitter(
            vertices_target=output["vertices"],
            initial_phenotype_kwargs=initial_phenotype_kwargs,
            excluded_phenotypes=excluded_phenotypes,
            optimize_phenotypes=optimize_phenotypes,
            max_delta=max_delta,
        )
        # print(macro['height'])
    duration = 1000.0 * (time.time() - start) / N
    print(f"\nFitting took {duration:.1f} ms with batch_size={batch_size}")
    pve = 1000.0 * torch.norm(vertices_hat - output["vertices"], dim=-1).mean(1)
    print(
        f"PVE: {pve.mean():.2f} mm (median={pve.median():.1f} - min={pve.min():.1f} - max={pve.max():.1f})"
    )
    v_hat = vertices_hat

    # saving
    i = pve.argmax().item()
    trimesh.Trimesh(
        vertices=output["vertices"][i].cpu().numpy(), faces=model.faces.cpu().numpy()
    ).export("y.ply")
    trimesh.Trimesh(
        vertices=v_hat[i].cpu().numpy(), faces=model.faces.cpu().numpy()
    ).export("y_hat.ply")
    print("Meshes saved into y.ply and y_hat.ply")


if __name__ == "__main__":
    from jsonargparse import auto_cli

    auto_cli(fit_to_mesh)
