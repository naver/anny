# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0

# %% [markdown]
# ## Keypoints
#
# Anny can regress named anatomical keypoints from the body surface. Each keypoint is defined as
# a weighted average of mesh vertices, so it follows the body as the shape and the pose change,
# and it stays differentiable with respect to every model parameter.
#
# This tutorial uses the bundled COCO keypoints, shows how to keep only a sparse set of vertices
# per keypoint, and finally bakes the regression into the model itself so that it outputs the
# keypoints directly, without evaluating the full mesh.

# %% [markdown]
# #### Imports and helper functions

# %%
import roma  # A PyTorch library useful to deal with space transformations.
import torch
import trimesh  # For 3D mesh visualization.
import trimesh.viewer.notebook as nb
from IPython.display import Markdown, display

import anny  # The main library for the Anny model.

trimesh_scene_transform = (
    roma.Rigid(
        linear=roma.euler_to_rotmat("x", [-90.0], degrees=True), translation=None
    )
    .to_homogeneous()
    .cpu()
    .numpy()
)

mesh_material = trimesh.visual.material.PBRMaterial(
    baseColorFactor=[0.6, 0.8, 0.7, 0.5],
    metallicFactor=0.5,
    doubleSided=False,
    alphaMode="BLEND",
)

keypoint_sphere = trimesh.creation.icosphere(radius=0.015, subdivisions=2)
keypoint_sphere.visual = trimesh.visual.TextureVisuals(
    material=trimesh.visual.material.PBRMaterial(
        baseColorFactor=[0.9, 0.3, 0.2, 1.0],
        metallicFactor=0.0,
        roughnessFactor=1.0,
        alphaMode="OPAQUE",
    )
)


def show_keypoints(vertices, faces, keypoints, labels):
    """Display a mesh with a sphere at each keypoint location."""
    mesh = trimesh.Trimesh(
        vertices=vertices.squeeze(0).detach().cpu().numpy(),
        faces=faces.cpu().numpy(),
    )
    mesh.visual.material = mesh_material
    scene = trimesh.Scene([mesh])
    for label, keypoint in zip(labels, keypoints.squeeze(0).detach().cpu()):
        scene.add_geometry(
            keypoint_sphere,
            transform=roma.Rigid(
                linear=torch.eye(3), translation=keypoint
            ).to_homogeneous(),
            node_name=label,
        )
    scene.apply_transform(trimesh_scene_transform)
    return nb.scene_to_notebook(scene)


# %% [markdown]
# ### Regressing COCO keypoints
#
# `KeypointsRegressor.coco` loads the bundled COCO regression weights. Passing an explicit list of
# labels fixes the output order; omitting it returns every keypoint of the file.

# %%
model = anny.Anny()

# Typical COCO 23 ordering
keypoint_labels = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_big_toe",
    "left_small_toe",
    "left_heel",
    "right_big_toe",
    "right_small_toe",
    "right_heel",
]
keypoints_regressor = anny.KeypointsRegressor.coco(model, keypoint_labels)

# %% [markdown]
# The regressor is a `torch.nn.Module` taking a model output dictionary and returning a
# `(batch, keypoints, 3)` tensor. Keypoints track the body under any pose and phenotype.

# %%
pose_parameters = (
    torch.eye(4, dtype=model.dtype)[None, None]
    .expand(1, model.bone_count, 4, 4)
    .clone()
)
for bone_label, rotation_vector in [
    ("upperarm01.L", [0.0, 0.0, -0.9]),
    ("upperarm01.R", [0.0, 0.0, 0.9]),
    ("lowerarm01.L", [0.0, -1.1, 0.0]),
    ("lowerarm01.R", [0.0, 1.1, 0.0]),
]:
    bone_index = model.bone_labels.index(bone_label)
    pose_parameters[0, bone_index, :3, :3] = roma.rotvec_to_rotmat(
        torch.tensor(rotation_vector, dtype=model.dtype)
    )
phenotype_kwargs = {"gender": 0.2, "age": 0.8, "weight": 0.7}

output = model(pose_parameters=pose_parameters, phenotype_kwargs=phenotype_kwargs)
keypoints = keypoints_regressor(output)
print(f"{keypoints.shape=}")

display(
    show_keypoints(
        output["vertices"], model.faces, keypoints, keypoints_regressor.labels
    )
)

# %% [markdown]
# ### Sparse regression weights
#
# The COCO weights are defined as a dense matrix with one weight per vertex,
# but most of the weights are 0 by design. We can make the regressor more efficient
# by taking only a sparse support for each keypoint.

# %%
support_size = 64
sparse_weights, sparse_indices = torch.topk(
    keypoints_regressor.regression_weights, support_size, dim=1
)
sparse_weights = sparse_weights / sparse_weights.sum(dim=1, keepdim=True)
sparse_regressor = anny.KeypointsRegressor(
    sparse_weights, keypoints_regressor.labels, regression_indices=sparse_indices
)

sparse_keypoints = sparse_regressor(output)
distances = torch.norm(keypoints - sparse_keypoints, dim=-1)
display(
    Markdown(
        f"Keeping {support_size} of {keypoints_regressor.regression_weights.shape[1]} vertices "
        f"per keypoint moves them by at most **{1000 * distances.max():.2f} mm** "
        f"(mean {1000 * distances.mean():.2f} mm)."
    )
)

# %%
