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
# - "makehuman", the default rig provided by MPFB2
#   (https://github.com/makehumancommunity/mpfb2) with minor vertex weights fixes.
# - "anny" (the default), an adaptation of the "makehuman" rig with more stable bone
#   orientations when the shape changes.
# - "mixamo", inspired by characters from https://www.mixamo.com/.
# - "soma", for compatibility with https://www.github.com/NVlabs/SOMA-X/.
#
# Bones that are not useful for your application can be removed from the Anny rig.
# Using "anny-notoes" will ignore bones animating individual toes, for example.
#
# Anny also supports various mesh topologies. A topology such as "notoes_collapse5pc"
# provides coarser mesh output for example, allowing to speed up inference and reduce
# memory consumption.
# We provide a "smplx" topology for interoperability with the SMPL-X model
# (https://smpl-x.is.tue.mpg.de/), **for non-commercial use only**.
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
            f"#### '{rig}' rig ({model.bone_count} bones) / '{topology}' topology "
            f"({len(output['vertices'].squeeze(0))} vertices, {len(model.faces)} faces)"
        )
    )
    viewers.append(Markdown("  - " + ", ".join([label for label in model.bone_labels])))
    viewers.append(nb.scene_to_notebook(scene))


# Display all viewers
display(*viewers)

# %% [markdown]
# ## Interoperability with the SMPL family
#
# Anny ships differentiable re-implementations of the SMPL body models as first-class
# model classes (`from anny.models.smpl import SMPL, SMPLH, SMPLX`). They follow the same
# construction and forward conventions as `anny.Anny`, but are driven by SMPL models
# parameters, rather than Anny phenotypes:
#
# | Class | Model (bones) | Additional inputs |
# |---|---|---|
# | `SMPL` | [SMPL](https://smpl.is.tue.mpg.de/) (24) | – |
# | `SMPLH` | [SMPL+H](https://mano.is.tue.mpg.de/) (52) | `left_hand_pose`, `right_hand_pose` (MANO, PCA by default) |
# | `SMPLX` | [SMPL-X](https://smpl-x.is.tue.mpg.de/) (55) | hand poses, `expression`, `jaw_pose`, eye poses |
#
# All three accept different output topologies (`"anny"`, `"smpl"`, `"smplx"`), the
# default being the model's native mesh. With `topology="anny"` the retopology is folded
# into the model at construction time, so a single forward pass outputs the body directly
# in Anny's common mesh, and per-vertex operations (texturing, correspondences, losses)
# transfer between a SMPL-driven body and a phenotype-driven `anny.Anny`. Anny vertices
# with no counterpart in the source model (e.g. the internal mouth bag) are ignored.
# Note that the `"smpl"` and `"smplx"` topologies rely on retopology data that Anny
# provides **for non-commercial use only**.
#
# The model files are distributed separately, under their own license terms, from the
# sites linked above. Point `SMPLX_MODEL_PATH` at a directory containing them in `smpl/`,
# `smplh/` and `smplx/` subfolders; the cell below skips itself otherwise.

# %%
import os

smplx_model_path = os.environ.get("SMPLX_MODEL_PATH")
if not smplx_model_path or not os.path.isdir(smplx_model_path):
    display(
        Markdown(
            "> **SMPL family examples skipped.** Download the model files and set the "
            "`SMPLX_MODEL_PATH` environment variable to the directory that contains them "
            "to run this cell."
        )
    )
else:
    from anny.models.smpl import SMPL, SMPLH, SMPLX

    dtype = torch.float32
    torch.manual_seed(0)
    # Shared shape and root parameters (10 betas).
    shape_kwargs = dict(
        betas=0.5 * torch.randn((1, 10), dtype=dtype),
        global_orient=0.1 * torch.randn((1, 3), dtype=dtype),
        transl=torch.zeros((1, 3), dtype=dtype),
    )
    # SMPL poses 23 body joints, the last two being the hands; SMPL+H and SMPL-X pose the
    # first 21 only and articulate the hands with MANO (6 PCA components by default).
    body_pose = 0.1 * torch.randn((1, 23 * 3), dtype=dtype)
    hand_kwargs = dict(
        left_hand_pose=torch.randn((1, 6), dtype=dtype),
        right_hand_pose=torch.randn((1, 6), dtype=dtype),
    )
    # SMPL-X adds a face: 10 expression coefficients, jaw and eye rotations.
    face_kwargs = dict(
        expression=0.5 * torch.randn((1, 10), dtype=dtype),
        jaw_pose=0.1 * torch.randn((1, 3), dtype=dtype),
        leye_pose=0.1 * torch.randn((1, 3), dtype=dtype),
        reye_pose=0.1 * torch.randn((1, 3), dtype=dtype),
    )

    for cls, pose_kwargs in [
        (SMPL, dict(body_pose=body_pose)),
        (SMPLH, dict(body_pose=body_pose[:, : 21 * 3], **hand_kwargs)),
        (SMPLX, dict(body_pose=body_pose[:, : 21 * 3], **hand_kwargs, **face_kwargs)),
    ]:
        # topology="anny" retopologizes the output onto Anny's common mesh (see above).
        model = cls(smplx_model_path, gender="neutral", topology="anny").to(dtype=dtype)
        output = model(**shape_kwargs, **pose_kwargs)

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
                f"#### `anny.{cls.__name__}` model ({model.bone_count} bones, "
                f"{len(output['vertices'].squeeze(0))} vertices)"
            )
        )
        display(nb.scene_to_notebook(scene))
