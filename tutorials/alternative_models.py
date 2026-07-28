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
# ### Alternative models
#
#
# In this tutorial, we show how to use different rigs and topologies with Anny.

# %% [markdown]
# #### Imports and helper functions

# %%
import torch
import roma  # A PyTorch library useful to deal with space transformations.
import anny  # The main library for the Anny model.
import trimesh  # For 3D mesh visualization.

# Create and show multiple rigs in one cell
from IPython.display import display, Markdown
import trimesh.viewer.notebook as nb

# Some helper objects for visualization.
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

world_axis = trimesh.creation.axis(axis_length=1.0)
axis = trimesh.creation.axis(axis_length=0.1)


def add_skeleton_to_scene(scene, model, output):
    # Add bones visualization. Procrustes rigs (the default) have no bone tails,
    # so we draw each bone as a segment from its head to its parent's head.
    bone_poses = output["bone_poses"].squeeze(dim=0).cpu()
    bone_heads = bone_poses[:, :3, 3]
    bone_color = [0.8, 0.3, 0.3, 1.0]
    bone_visual = trimesh.visual.TextureVisuals(
        material=trimesh.visual.material.PBRMaterial(
            baseColorFactor=bone_color,
            metallicFactor=0.0,
            roughnessFactor=1.0,
            doubleSided=True,
            alphaMode="BLEND",
        )
    )
    for i in range(1, len(bone_heads)):
        bone_head = bone_heads[model.bone_parents[i]]
        bone_tail = bone_heads[i]
        length = torch.norm(bone_tail - bone_head).item()
        if length < 1e-6:
            continue
        cylinder = trimesh.creation.cylinder(radius=0.005, height=length, sections=16)
        t = (bone_head + bone_tail) / 2
        M = roma.special_gramschmidt(
            torch.stack(
                [
                    bone_tail - bone_head,
                    torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32),
                ],
                dim=-1,
            )
        )
        R = torch.stack([M[:, 2], M[:, 1], M[:, 0]], dim=-1)
        cylinder.visual = bone_visual
        scene.add_geometry(
            cylinder,
            transform=roma.Rigid(R, t).to_homogeneous().numpy(),
            node_name=f"bone_{model.bone_labels[i]}",
        )

    # Add some spheres at the joints
    joint_sphere = trimesh.creation.icosphere(radius=0.008, subdivisions=2)
    joint_sphere.visual = trimesh.visual.TextureVisuals(
        material=trimesh.visual.material.PBRMaterial(
            baseColorFactor=[0.1, 0.1, 0.1, 1.0],
            metallicFactor=0.5,
            roughnessFactor=1.0,
            doubleSided=True,
            alphaMode="OPAQUE",
        )
    )
    for i in range(len(bone_poses)):
        scene.add_geometry(
            joint_sphere,
            transform=bone_poses[i],
            node_name=f"joint_{model.bone_labels[i]}",
        )


# %% [markdown]
# ## Rigs and topology
#
# Anny supports various skeletal rigs, including:
# - "makehuman", the default rig provided by MPFB2 (https://github.com/makehumancommunity/mpfb2) with minor vertex weights fixes.
# - "anny" (the default), an adaptation of the "makehuman" rig with more stable bone orientations when the shape changes.
# - "mixamo", inspired by characters from https://www.mixamo.com/.
# - "soma", for compatibility with https://www.github.com/NVlabs/SOMA-X/.
#
# Bones that are not useful for your application can be removed from the Anny rig. Using "anny-notoes" will ignore bones animating individual toes, for example.
#
# Anny also supports various mesh topologies. A topology such as "notoes_collapse5pc" provides coarser mesh output for example, allowing to speed up inference and reduce memory consumption.
# We provide a "smplx" topology for interoperability with the SMPL-X model (https://smpl-x.is.tue.mpg.de/), **for non-commercial use only**.
#
# We show below a few combinations of meshes and topologies supported by the model:

# %%
viewers = []

for rig, topology in [
    ("anny", "anny"),
    (
        "mixamo",
        "anny",
    ),
    (
        "anny",
        "smplx",
    ),
    (
        "anny",
        "soma",
    ),
    ("anny-notoes-noeyes", "makehuman"),
]:
    model = anny.Anny(rig=rig, topology=topology)
    output = model()

    mesh = trimesh.Trimesh(
        vertices=output["vertices"].squeeze(0).cpu().numpy(),
        faces=model.faces.cpu().numpy(),
    )
    mesh.visual.material = mesh_material
    scene = trimesh.Scene([mesh])

    add_skeleton_to_scene(scene, model, output)
    scene.apply_transform(trimesh_scene_transform)

    # Convert to a notebook widget/HTML
    viewers.append(
        Markdown(
            f"#### '{rig}' rig ({model.bone_count} bones) / '{topology}' topology ({len(output['vertices'].squeeze(0))} vertices, {len(model.faces)} faces)"
        )
    )
    viewers.append(Markdown("  - " + ", ".join([label for label in model.bone_labels])))
    viewers.append(nb.scene_to_notebook(scene))


# Display all viewers
display(*viewers)

# %% [markdown]
# ## Interoperability with SMPL-X
#
# Beyond the SMPL-X *topology* shown above, Anny ships a first-class `SMPLX` model class
# (`from anny.models.smpl import SMPLX`) that wraps the official
# [SMPL-X](https://smpl-x.is.tue.mpg.de/) model. It follows the same construction and forward
# conventions as `anny.Anny`, but is driven by SMPL-X `betas` / `expression` / pose parameters
# rather than Anny phenotypes, making it a drop-in differentiable SMPL-X implementation.
#
# Running it requires the SMPL-X model files, which are distributed separately (for
# **non-commercial use only**) at https://smpl-x.is.tue.mpg.de/. Download them and point the
# `SMPLX_MODEL_PATH` environment variable at the directory that contains them. The cell below
# skips itself automatically when that variable is not set.
#
# Here we request the `"anny"` output topology (`topology="anny"`) rather than the native SMPL-X
# mesh. The retopology is folded into the model at construction time, so a single forward pass
# directly outputs the body in Anny's common mesh — there is no separate conversion step. This is
# what makes the outputs interoperable: a SMPL-X-driven body and a phenotype-driven `anny.Anny`
# body live in the same topology, so per-vertex operations (texturing, correspondences, losses)
# transfer directly between them. Use `topology="smplx"` instead if you need the native SMPL-X mesh.
#
# Note that some Anny vertices have no SMPL-X counterpart — for instance the internal mouth bag,
# which SMPL-X does not model — and are therefore ignored by this operation.

# %%
import os

smplx_model_path = os.environ.get("SMPLX_MODEL_PATH")
if not smplx_model_path or not os.path.isdir(smplx_model_path):
    display(
        Markdown(
            "> **SMPL-X example skipped.** Download the SMPL-X model files from "
            "[smpl-x.is.tue.mpg.de](https://smpl-x.is.tue.mpg.de/) and set the `SMPLX_MODEL_PATH` "
            "environment variable to the directory that contains them to run this cell."
        )
    )
else:
    from anny.models.smpl import SMPLX

    dtype = torch.float32
    # topology="anny" retopologizes the SMPL-X output onto Anny's common mesh (see above).
    model = SMPLX(smplx_model_path, gender="neutral", use_pca=True, topology="anny").to(
        dtype=dtype
    )

    # Random shape, expression and pose parameters, using SMPL-X's standard parameter
    # dimensions (10 shape betas, 10 expression coefficients, 21 body joints, 6 PCA hand
    # components).
    torch.manual_seed(0)
    pose_kwargs = dict(
        betas=0.5 * torch.randn((1, 10), dtype=dtype),
        expression=0.5 * torch.randn((1, 10), dtype=dtype),
        global_orient=0.1 * torch.randn((1, 3), dtype=dtype),
        body_pose=0.1 * torch.randn((1, 21 * 3), dtype=dtype),
        transl=torch.zeros((1, 3), dtype=dtype),
        jaw_pose=0.1 * torch.randn((1, 3), dtype=dtype),
        leye_pose=0.1 * torch.randn((1, 3), dtype=dtype),
        reye_pose=0.1 * torch.randn((1, 3), dtype=dtype),
        left_hand_pose=torch.randn((1, 6), dtype=dtype),
        right_hand_pose=torch.randn((1, 6), dtype=dtype),
    )

    output = model(**pose_kwargs)

    # Anny's SMPL-X mesh with its skeleton.
    mesh = trimesh.Trimesh(
        vertices=output["vertices"].squeeze(0).detach().cpu().numpy(),
        faces=model.faces.cpu().numpy(),
    )
    mesh.visual.material = mesh_material
    scene = trimesh.Scene([mesh])

    add_skeleton_to_scene(scene, model, output)
    scene.apply_transform(trimesh_scene_transform)

    display(
        Markdown(
            f"#### `anny.SMPLX` model ({model.bone_count} bones, "
            f"{len(output['vertices'].squeeze(0))} vertices)"
        )
    )
    display(nb.scene_to_notebook(scene))
