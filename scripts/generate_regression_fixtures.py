"""Generate regression fixture NPZ files for test/test_regression_vs_fixtures.py.

Run from the repo root:
    uv run python scripts/generate_regression_fixtures.py fixtures
"""

import os
import random
import json

import numpy as np
import roma
import torch
from jsonargparse import auto_cli

import anny

CONFIGS = [
    {},
    {"rig": "anny-notoes", "local_changes": "all"},
    {"all_phenotypes": True},
    {"pose_parameterization": "world-orient"},
    {"topology": "soma"},
    {"topology": "smplx"},
    {"topology": "anny-quads"},
    {"rig": "soma"},
]

SEED = 42
BATCH_SIZE = 1


def config_to_slug(cfg: dict) -> str:
    if not cfg:
        return "default"
    return "__".join(f"{k}_{v}" for k, v in sorted(cfg.items()))

def generate_config(cfg: dict, seed: int) -> dict:
    model = anny.Anny(**cfg)
    dtype = model.dtype
    slug = config_to_slug(cfg)

    torch.manual_seed(seed + hash(slug))
    rots = roma.random_rotmat((BATCH_SIZE, model.bone_count), dtype=dtype)
    pose_parameters = roma.Rigid(
        rots, torch.zeros((BATCH_SIZE, model.bone_count, 3), dtype=dtype)
    ).to_homogeneous()

    phenotype_kwargs = {
        key: torch.rand((BATCH_SIZE,), dtype=dtype)
        for key in model.phenotype_labels
    }

    local_changes = random.sample(model.local_change_labels, k=min(2, len(model.local_change_labels)))
    local_changes_kwargs = {
        key: torch.rand((BATCH_SIZE,), dtype=dtype)
        for key in local_changes    }
    return {
        "slug": slug,
        "config": cfg,
        "model_kwargs": {
        "phenotype_kwargs": {key: val.tolist() for key, val in phenotype_kwargs.items()},
        "pose_parameters": pose_parameters.tolist(),
        "local_changes_kwargs": {key: val.tolist() for key, val in local_changes_kwargs.items()}
        }
    }

def generate_fixture(cfg: dict, output_dir: str) -> None:
    slug = cfg["slug"]
    out_path = os.path.join(output_dir, f"regression_{slug}.npz")
    if os.path.exists(out_path):
        print(f"Fixture already exists: {out_path} - loading existing data for comparison.")
        old_data = np.load(out_path)
    else:
        old_data = None

    model = anny.Anny(**cfg["config"])

    pose_parameters = torch.tensor(cfg["model_kwargs"]["pose_parameters"], dtype=model.dtype)
    phenotype_kwargs = {key: torch.tensor(val, dtype=model.dtype) for key, val in cfg["model_kwargs"]["phenotype_kwargs"].items()}
    local_changes_kwargs = {key: torch.tensor(val, dtype=model.dtype) for key, val in cfg["model_kwargs"]["local_changes_kwargs"].items()}
    with torch.no_grad():
        fwd_output = model(pose_parameters=pose_parameters, phenotype_kwargs=phenotype_kwargs, local_changes_kwargs=local_changes_kwargs)

    data = {
        "template_vertices": model.template_vertices.numpy(),
        "rest_vertices": fwd_output["rest_vertices"].numpy(),
        "rest_bone_poses": fwd_output["rest_bone_poses"].numpy(),
        "forward_vertices": fwd_output["vertices"].numpy(),
        "forward_bone_poses": fwd_output["bone_poses"].numpy(),
    }

    if old_data is not None:
        for key in data:
            if not np.allclose(data[key], old_data[key]):
                print(f"Warning: Data mismatch for {slug} key {key} - overwriting fixture.")
                np.savez_compressed(out_path, **data)
                print(f"Saved: {out_path}")
                break
    else:
        np.savez_compressed(out_path, **data)
        print(f"Saved: {out_path}")



def generate_configs(output_dir: str = "test/data") -> None:
    """Generate regression fixture configs JSON file."""
    os.makedirs("test/data", exist_ok=True)
    configs = []
    for cfg in CONFIGS:
        print(f"Generating config for: {cfg}")
        cfg_full = generate_config(cfg, seed)
        configs.append(cfg_full)
    with open("test/data/regression_configs.json", "w") as f:
        json.dump(configs, f, indent=4)
    print("Saved: test/data/regression_configs.json")

def generate_fixtures(output_dir: str = "test/data", config_slugs: list[str] | None = None) -> None:
    """Generate regression fixture NPZ files."""
    os.makedirs(output_dir, exist_ok=True)
    configs = json.load(open("test/data/regression_configs.json"))
    for cfg in configs:
        if config_slugs and cfg["slug"] not in config_slugs:
            print(f"Skipping fixture for config: {cfg['slug']}")
            continue
        generate_fixture(cfg, output_dir=output_dir)


if __name__ == "__main__":
    auto_cli({"configs": generate_configs, "fixtures": generate_fixtures})
