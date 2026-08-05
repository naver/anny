# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0

import os
import gc
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import roma
import trimesh
import anny

from anny.shape_distribution import SimpleShapeDistribution

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

ABLATION_JOBS: List[Dict[str, Any]] = [
    # SAME-RIG FITTING
    {
        "id": "A",
        "target_rig": "anny",
        "fitting_rig": "anny",
        "optimize_phenotypes": False,
        "post_gd": False,
        "post_gd_steps": 0,
        "purpose": "Same-rig pose baseline",
    },
    {
        "id": "B",
        "target_rig": "anny",
        "fitting_rig": "anny",
        "optimize_phenotypes": False,
        "post_gd": True,
        "post_gd_steps": 100,
        "purpose": "Measure the effect of post-GD with known shape",
    },
    {
        "id": "C",
        "target_rig": "anny",
        "fitting_rig": "anny",
        "optimize_phenotypes": False,
        "max_n_iters": 10,
        "post_gd": False,
        "post_gd_steps": 0,
        "purpose": "Measure known-shape same-rig fitting with more iterations and no post-GD",
    },
    {
        "id": "D",
        "target_rig": "anny",
        "fitting_rig": "anny",
        "optimize_phenotypes": True,
        "post_gd": False,
        "post_gd_steps": 0,
        "purpose": "Full inverse problem with matched rig before post-GD",
    },
    {
        "id": "E",
        "target_rig": "anny",
        "fitting_rig": "anny",
        "optimize_phenotypes": True,
        "post_gd": True,
        "post_gd_steps": 100,
        "purpose": "Full inverse problem with matched rig",
    },
    {
        "id": "F",
        "target_rig": "anny",
        "fitting_rig": "anny",
        "optimize_phenotypes": True,
        "post_gd": True,
        "post_gd_steps": 40,
        "purpose": "Short post-GD budget with matched rig and unknown shape",
    },
    {
        "id": "F-ms",
        "target_rig": "anny",
        "fitting_rig": "anny",
        "optimize_phenotypes": True,
        "post_gd": True,
        "post_gd_steps": 40,
        "multistart_anchors": {"muscle": [0.01, 0.5, 0.99]},
        "purpose": "Matched-rig unknown-shape fit with muscle multistart anchors",
    },
    # CROSS-RIG FITTING
    {
        "id": "G",
        "target_rig": "soma",
        "fitting_rig": "anny",
        "optimize_phenotypes": False,
        "post_gd": False,
        "post_gd_steps": 0,
        "post_gd_optimize_local_changes": False,
        "post_gd_optimize_facial_actions": False,
        "purpose": "Cross-rig pose baseline",
    },
    {
        "id": "H",
        "target_rig": "soma",
        "fitting_rig": "anny",
        "optimize_phenotypes": False,
        "post_gd": True,
        "post_gd_steps": 100,
        "post_gd_optimize_local_changes": False,
        "post_gd_optimize_facial_actions": False,
        "purpose": "Measure the effect of post-GD under cross-rig fitting",
    },
    {
        "id": "I",
        "target_rig": "soma",
        "fitting_rig": "anny",
        "optimize_phenotypes": False,
        "max_n_iters": 10,
        "post_gd": False,
        "post_gd_steps": 0,
        "post_gd_optimize_local_changes": False,
        "post_gd_optimize_facial_actions": False,
        "purpose": "Measure known-shape cross-rig fitting with more iterations and no post-GD",
    },
    {
        "id": "J",
        "target_rig": "soma",
        "fitting_rig": "anny",
        "optimize_phenotypes": True,
        "post_gd": False,
        "post_gd_steps": 0,
        "post_gd_optimize_local_changes": True,
        "post_gd_optimize_facial_actions": True,
        "purpose": "Full inverse problem with cross-rig fitting before post-GD",
    },
    {
        "id": "K",
        "target_rig": "soma",
        "fitting_rig": "anny",
        "optimize_phenotypes": True,
        "post_gd": True,
        "post_gd_steps": 100,
        "post_gd_optimize_local_changes": True,
        "post_gd_optimize_facial_actions": True,
        "purpose": "Full inverse problem with cross-rig fitting",
    },
    {
        "id": "K-ms",
        "target_rig": "soma",
        "fitting_rig": "anny",
        "optimize_phenotypes": True,
        "post_gd": True,
        "post_gd_steps": 100,
        "post_gd_optimize_local_changes": True,
        "post_gd_optimize_facial_actions": True,
        "multistart_anchors": {"muscle": [0.01, 0.5, 0.99]},
        "purpose": "Cross-rig unknown-shape fit with muscle multistart anchors",
    },
]


def _format_markdown_value(value: Any, precision: int = 2) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{precision}f}"
    return str(value)


def _format_ablation_markdown(results: List[Dict[str, Any]]) -> str:
    header = (
        "| ID | Target rig | Fitting rig | Shape known | Max iters | Post-GD | "
        "PVE mean (mm) | PVE median (mm) | PVE max (mm) | "
        "Samples/s | Purpose |"
    )
    separator = (
        "|:--:|:----------:|:-----------:|:-----------:|:---------:|:-------:|"
        ":-------------:|:---------------:|:------------:|:------------:|---------|"
    )
    rows = [header, separator]

    for result in results:
        shape_known = "Yes" if not result["optimize_phenotypes"] else "No"
        post_gd = f"Yes ({result['post_gd_steps']})" if result["post_gd"] else "No"
        rows.append(
            "| "
            f"**{result['id']}** | "
            f"`{result['target_rig']}` | "
            f"`{result['fitting_rig']}` | "
            f"**{shape_known}** | "
            f"{result['max_n_iters']} | "
            f"{post_gd} | "
            f"{_format_markdown_value(result['pve_mean_mm'])} | "
            f"{_format_markdown_value(result['pve_median_mm'])} | "
            f"{_format_markdown_value(result['pve_max_mm'])} | "
            f"{_format_markdown_value(result['throughput_samples_per_s'], precision=1)} | "
            f"{result['purpose']} |"
        )

    return "\n".join(rows)


def _format_phenotype_impact_markdown(results: List[Dict[str, Any]]) -> str:
    header = (
        "| Phenotype | Rig | Max iters | Post-GD | "
        "PVE mean (mm) | PVE median (mm) | PVE max (mm) | Samples/s |"
    )
    separator = (
        "|:---------:|:---:|:---------:|:-------:|"
        ":-------------:|:---------------:|:------------:|:------------:|"
    )
    rows = [header, separator]

    for result in results:
        post_gd = f"Yes ({result['post_gd_steps']})" if result["post_gd"] else "No"
        rows.append(
            "| "
            f"**{result['phenotype']}** | "
            f"`{result['target_rig']}` | "
            f"{result['max_n_iters']} | "
            f"{post_gd} | "
            f"{_format_markdown_value(result['pve_mean_mm'])} | "
            f"{_format_markdown_value(result['pve_median_mm'])} | "
            f"{_format_markdown_value(result['pve_max_mm'])} | "
            f"{_format_markdown_value(result['throughput_samples_per_s'], precision=1)} |"
        )

    return "\n".join(rows)


def _save_meshes(
    vertices_target: torch.Tensor,
    vertices_hat: torch.Tensor,
    faces: torch.Tensor,
    out_dir: str,
    prefix: str,
    max_samples: Optional[int] = 4,
):
    os.makedirs(out_dir, exist_ok=True)
    faces_np = faces.detach().cpu().numpy()

    batch_size = vertices_target.shape[0]
    if max_samples is not None:
        batch_size = min(batch_size, max_samples)
    for b in range(batch_size):
        fn_target = f"{out_dir}/{prefix}_sample-{b:03d}_target.ply"
        fn_fitted = f"{out_dir}/{prefix}_sample-{b:03d}_fitted.ply"
        trimesh.Trimesh(
            vertices=vertices_target[b].detach().cpu().numpy(),
            faces=faces_np,
            process=False,
        ).export(fn_target)

        trimesh.Trimesh(
            vertices=vertices_hat[b].detach().cpu().numpy(),
            faces=faces_np,
            process=False,
        ).export(fn_fitted)

        print(f"Saved target mesh to {fn_target}")
        print(f"Saved fitted mesh to {fn_fitted}")


def cross_rig_random_meshes(
    seed: int = 3993,
    batch_size: int = 16,
    N: int = 2,
    target_rig: str = "soma",
    fitting_rig: str = "anny",
    topology: str = "anny",
    phenotypes: str = "none",
    max_n_iters: int = 5,
    eps: float = 0.1,
    max_delta: float = 0.1,
    optimize_phenotypes: bool = True,
    excluded_phenotypes: Optional[List[str]] = ["age", "gender"],
    joint_min_weight: float = 0.01,
    joint_top_k: Optional[int] = 1024,
    post_gd: bool = True,
    post_gd_steps: int = 100,
    post_gd_lr: float = 1e-3,
    post_gd_prior_weight: float = 0.0,
    post_gd_optimize_local_changes: bool = False,
    post_gd_optimize_facial_actions: bool = False,
    n_points: Optional[int] = None,
    verbose: bool = True,
    out_dir: str = "mesh_to_anny_params_output_examples",
    save_meshes: bool = True,
    print_optimized_phenotypes: bool = False,
    multistart_anchors: Optional[Dict[str, List[float]]] = None,
):
    """
    Generate random meshes with SOMA rig + Anny topology, then fit them with Anny.

    By default:
        target_rig='soma'
        fitting_rig='anny'
        topology='anny'
    """

    print("seed", seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if excluded_phenotypes is None:
        excluded_phenotypes = ["age", "gender"]

    dtype = torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    target_model = anny.Anny(
        rig=target_rig,
        topology=topology,
        phenotypes=phenotypes,
        local_changes="default",
        facial_actions="all",
    ).to(dtype=dtype, device=device)

    fitting_model = anny.Anny(
        rig=fitting_rig,
        topology=topology,
        phenotypes=phenotypes,
        local_changes="default",
        facial_actions="all",
    ).to(dtype=dtype, device=device)

    pose_parameters = {}
    for i, bone in enumerate(target_model.bone_labels):
        if "toe" in bone.lower() or "jaw" in bone.lower() or "eye" in bone.lower():
            continue

        rotvec = 0.2 * torch.randn((batch_size, 3), dtype=dtype, device=device)
        rotmat = roma.rotvec_to_rotmat(rotvec)

        translation = (
            torch.randn((batch_size, 3), dtype=dtype, device=device) if i == 0 else None
        )

        pose_parameters[bone] = roma.Rigid(
            linear=rotmat,
            translation=translation,
        )

    shape_dist = SimpleShapeDistribution(
        target_model,
        morphological_age_distribution=torch.distributions.Uniform(
            low=torch.tensor(
                20.0, dtype=target_model.dtype, device=target_model.device
            ),
            high=torch.tensor(
                90.0, dtype=target_model.dtype, device=target_model.device
            ),
        ),
    )

    _, phenotype_kwargs = shape_dist.sample(batch_size)

    with torch.no_grad():
        output = target_model(
            pose_parameters=pose_parameters,
            phenotype_kwargs=phenotype_kwargs,
        )

    vertices_target = output["vertices"]

    fitter = anny.AnnyInverter(
        model=fitting_model,
        verbose=verbose,
        max_n_iters=max_n_iters,
        eps=eps,
        n_points=n_points,
        joint_min_weight=joint_min_weight,
        joint_top_k=joint_top_k,
    )

    timed_n_iters = max(N - 1, 0)
    start = None

    def phenotype_parameters(fit_parameters):
        if isinstance(fit_parameters, tuple):
            return fit_parameters[0]
        return fit_parameters

    for iter_idx in range(N):
        if optimize_phenotypes:
            initial_phenotype_kwargs = {"age": 0.7}
            initial_phenotype_kwargs.update(
                {
                    phenotype: phenotype_kwargs[phenotype]
                    for phenotype in excluded_phenotypes
                    if phenotype in phenotype_kwargs
                }
            )
        else:
            initial_phenotype_kwargs = phenotype_kwargs

        pose, macro, vertices_hat = fitter(
            vertices_target=vertices_target,
            initial_phenotype_kwargs=initial_phenotype_kwargs,
            excluded_phenotypes=excluded_phenotypes,
            optimize_phenotypes=optimize_phenotypes,
            max_delta=max_delta,
            post_gd=post_gd,
            post_gd_steps=post_gd_steps,
            post_gd_lr=post_gd_lr,
            post_gd_prior_weight=post_gd_prior_weight,
            post_gd_optimize_local_changes=post_gd_optimize_local_changes,
            post_gd_optimize_facial_actions=post_gd_optimize_facial_actions,
            multistart_anchors=multistart_anchors,
        )

        if N > 1 and iter_idx == 0:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()

    pve = 1000.0 * torch.norm(vertices_hat - vertices_target, dim=-1).mean(dim=1)

    if print_optimized_phenotypes and optimize_phenotypes:
        fitted_phenotypes = phenotype_parameters(macro)
        optimized_phenotypes = [
            phenotype
            for phenotype in fitting_model.phenotype_labels
            if phenotype not in excluded_phenotypes
        ]

        print("\nOptimized phenotype target vs fitted")
        for phenotype in optimized_phenotypes:
            if phenotype not in phenotype_kwargs or phenotype not in fitted_phenotypes:
                continue

            target_values = phenotype_kwargs[phenotype].detach().cpu()
            fitted_values = fitted_phenotypes[phenotype].detach().cpu()
            error_values = torch.abs(fitted_values - target_values)
            pve_values = pve.detach().cpu()

            print(
                f"{phenotype}: "
                f"target_mean={target_values.mean():.3f}, "
                f"fitted_mean={fitted_values.mean():.3f}, "
                f"mae={error_values.mean():.3f}, "
                f"max_error={error_values.max():.3f}"
            )
            print("sample | target | fitted | abs_error | pve_mm")
            for sample_idx in range(target_values.shape[0]):
                print(
                    f"{sample_idx:03d} | "
                    f"{target_values[sample_idx]:.3f} | "
                    f"{fitted_values[sample_idx]:.3f} | "
                    f"{error_values[sample_idx]:.3f} | "
                    f"{pve_values[sample_idx]:.2f}"
                )

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    duration_ms = None
    if start is not None:
        elapsed = time.perf_counter() - start
        duration_ms = 1000.0 * elapsed / timed_n_iters

    throughput_samples_per_s = None
    if duration_ms is not None:
        throughput_samples_per_s = 1000.0 * batch_size / duration_ms

    if duration_ms is not None:
        print(f"\nFitting took {duration_ms:.1f} ms with batch_size={batch_size}")
        print(f"Throughput: {throughput_samples_per_s:.1f} samples/s")
    else:
        print("\nFitting timing skipped because N <= 1")
    print(
        f"PVE: {pve.mean():.2f} mm "
        f"(median={pve.median():.1f} - min={pve.min():.1f} - max={pve.max():.1f})\n"
    )

    prefix = f"target-{target_rig}_fit-{fitting_rig}_topology-{topology}"
    prefix += f"_shapeopt-{optimize_phenotypes}_iters-{max_n_iters}_postgd-{post_gd}_steps-{post_gd_steps}"
    prefix += f"_localopt-{post_gd_optimize_local_changes}_faceopt-{post_gd_optimize_facial_actions}"
    prefix = prefix.replace(".", "p").replace("None", "all")

    if save_meshes:
        _save_meshes(
            vertices_target=vertices_target,
            vertices_hat=vertices_hat,
            faces=fitting_model.faces,
            out_dir=out_dir,
            prefix=prefix,
        )

        print(f"Meshes saved into {out_dir}")

    return {
        "target_rig": target_rig,
        "fitting_rig": fitting_rig,
        "topology": topology,
        "phenotypes": phenotypes,
        "optimize_phenotypes": optimize_phenotypes,
        "max_n_iters": max_n_iters,
        "post_gd": post_gd,
        "post_gd_steps": post_gd_steps,
        "post_gd_optimize_local_changes": post_gd_optimize_local_changes,
        "post_gd_optimize_facial_actions": post_gd_optimize_facial_actions,
        "batch_size": batch_size,
        "duration_ms": duration_ms,
        "throughput_samples_per_s": throughput_samples_per_s,
        "pve_mean_mm": pve.mean().item(),
        "pve_median_mm": pve.median().item(),
        "pve_min_mm": pve.min().item(),
        "pve_max_mm": pve.max().item(),
    }


def phenotype_impact(
    seed: int = 3993,
    batch_size: int = 32,
    N: int = 2,
    rig: str = "anny",
    topology: str = "anny",
    phenotypes: str = "none",
    max_n_iters: int = 5,
    eps: float = 0.1,
    max_delta: float = 0.1,
    joint_min_weight: float = 0.01,
    joint_top_k: Optional[int] = 1024,
    post_gd: bool = False,
    post_gd_steps: int = 100,
    post_gd_lr: float = 1e-3,
    post_gd_prior_weight: float = 0.0,
    post_gd_optimize_local_changes: bool = False,
    post_gd_optimize_facial_actions: bool = False,
    n_points: Optional[int] = None,
    verbose: bool = True,
    out_dir: str = "mesh_to_mesh_to_anny_params.py",
    save_meshes: bool = False,
    markdown_path: Optional[str] = None,
    print_optimized_phenotypes: bool = False,
    multistart_anchors: Optional[Dict[str, List[float]]] = None,
):
    """Fit random same-rig targets while optimizing one phenotype at a time."""

    dtype = torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    phenotype_labels = (
        anny.Anny(
            rig=rig,
            topology=topology,
            phenotypes=phenotypes,
            local_changes="default",
            facial_actions="all",
        )
        .to(dtype=dtype, device=device)
        .phenotype_labels
    )

    results = []
    for phenotype in phenotype_labels:
        excluded_phenotypes = [
            label for label in phenotype_labels if label != phenotype
        ]
        print(
            f"\nRunning phenotype impact for {phenotype}: "
            f"rig={rig} max_n_iters={max_n_iters} post_gd={post_gd} "
            f"post_gd_steps={post_gd_steps}"
        )
        result = cross_rig_random_meshes(
            seed=seed,
            batch_size=batch_size,
            N=N,
            target_rig=rig,
            fitting_rig=rig,
            topology=topology,
            phenotypes=phenotypes,
            max_n_iters=max_n_iters,
            eps=eps,
            max_delta=max_delta,
            optimize_phenotypes=True,
            excluded_phenotypes=excluded_phenotypes,
            joint_min_weight=joint_min_weight,
            joint_top_k=joint_top_k,
            post_gd=post_gd,
            post_gd_steps=post_gd_steps,
            post_gd_lr=post_gd_lr,
            post_gd_prior_weight=post_gd_prior_weight,
            post_gd_optimize_local_changes=post_gd_optimize_local_changes,
            post_gd_optimize_facial_actions=post_gd_optimize_facial_actions,
            n_points=n_points,
            verbose=verbose,
            out_dir=out_dir,
            save_meshes=save_meshes,
            print_optimized_phenotypes=print_optimized_phenotypes,
            multistart_anchors=multistart_anchors,
        )
        result["phenotype"] = phenotype
        results.append(result)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    markdown = _format_phenotype_impact_markdown(results)
    print("\nPhenotype impact results\n")
    print(markdown)

    if markdown_path is not None:
        markdown_file = Path(markdown_path)
        markdown_file.parent.mkdir(parents=True, exist_ok=True)
        markdown_file.write_text(markdown + "\n")
        print(f"\nMarkdown results saved to {markdown_path}")

    return results


def ablation(
    seed: int = 3993,
    batch_size: int = 32,
    N: int = 2,
    topology: str = "anny",
    max_n_iters: int = 10,
    eps: float = 0.1,
    max_delta: float = 0.1,
    excluded_phenotypes: Optional[List[str]] = ["age", "gender"],
    joint_min_weight: float = 0.01,
    joint_top_k: Optional[int] = 1024,
    post_gd_lr: float = 1e-3,
    post_gd_prior_weight: float = 0.0,
    n_points: Optional[int] = None,
    verbose: bool = True,
    out_dir: str = "mesh_to_mesh_to_anny_params.py",
    save_meshes: bool = False,
    markdown_path: Optional[str] = None,
    multistart_anchors: Optional[Dict[str, List[float]]] = None,
):
    """Run the mesh-to-parameters fitting ablation and print a markdown results table."""

    results = []
    for job in ABLATION_JOBS:
        print(
            f"\nRunning ablation {job['id']}: "
            f"target_rig={job['target_rig']} "
            f"fitting_rig={job['fitting_rig']} "
            f"optimize_phenotypes={job['optimize_phenotypes']} "
            f"max_n_iters={job.get('max_n_iters', max_n_iters)} "
            f"post_gd={job['post_gd']} "
            f"post_gd_steps={job['post_gd_steps']} "
            f"post_gd_optimize_local_changes={job.get('post_gd_optimize_local_changes', False)} "
            f"post_gd_optimize_facial_actions={job.get('post_gd_optimize_facial_actions', False)} "
            f"multistart_anchors={job.get('multistart_anchors', multistart_anchors)}"
        )
        result = cross_rig_random_meshes(
            seed=seed,
            batch_size=batch_size,
            N=N,
            target_rig=job["target_rig"],
            fitting_rig=job["fitting_rig"],
            topology=topology,
            phenotypes="none",
            max_n_iters=job.get("max_n_iters", max_n_iters),
            eps=eps,
            max_delta=max_delta,
            optimize_phenotypes=job["optimize_phenotypes"],
            excluded_phenotypes=excluded_phenotypes,
            joint_min_weight=joint_min_weight,
            joint_top_k=joint_top_k,
            post_gd=job["post_gd"],
            post_gd_steps=job["post_gd_steps"],
            post_gd_lr=post_gd_lr,
            post_gd_prior_weight=post_gd_prior_weight,
            post_gd_optimize_local_changes=job.get(
                "post_gd_optimize_local_changes", False
            ),
            post_gd_optimize_facial_actions=job.get(
                "post_gd_optimize_facial_actions", False
            ),
            n_points=n_points,
            verbose=verbose,
            out_dir=out_dir,
            save_meshes=save_meshes,
            multistart_anchors=job.get("multistart_anchors", multistart_anchors),
        )
        result.update({"id": job["id"], "purpose": job["purpose"]})
        results.append(result)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    markdown = _format_ablation_markdown(results)
    print("\nResults\n")
    print(markdown)

    if markdown_path is not None:
        markdown_file = Path(markdown_path)
        markdown_file.parent.mkdir(parents=True, exist_ok=True)
        markdown_file.write_text(markdown + "\n")
        print(f"\nMarkdown results saved to {markdown_path}")

    return results


if __name__ == "__main__":
    import sys
    from jsonargparse import auto_cli

    commands = {
        "fit": cross_rig_random_meshes,
        "ablation": ablation,
        "phenotype_impact": phenotype_impact,
        "phenotype-impact": phenotype_impact,
    }
    if len(sys.argv) > 1 and sys.argv[1] in commands:
        auto_cli(commands)
    else:
        auto_cli(cross_rig_random_meshes)
