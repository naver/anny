# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

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
import numpy as np
import roma # A PyTorch library useful to deal with space transformations.
import anny # The main library for the Anny model.
import trimesh # For 3D mesh visualization.
# Create and show multiple rigs in one cell
from IPython.display import display, Markdown
import trimesh.viewer.notebook as nb

# Some helper objects for visualization.
trimesh_scene_transform = roma.Rigid(linear=roma.euler_to_rotmat('x', [-90.], degrees=True), translation=None).to_homogeneous().cpu().numpy()

mesh_material = trimesh.visual.material.PBRMaterial(baseColorFactor=[0.6, 0.8, 0.7, 0.5],
                                                        metallicFactor=0.5,
                                                        doubleSided=False,
                                                        alphaMode='BLEND')

world_axis = trimesh.creation.axis(axis_length=1.)
axis = trimesh.creation.axis(axis_length=0.1)


def add_skeleton_to_scene(scene, model, output):
    # Add bones visualization. Procrustes rigs (the default) have no bone tails,
    # so we draw each bone as a segment from its head to its parent's head.
    bone_poses = output["bone_poses"].squeeze(dim=0).cpu()
    bone_heads = bone_poses[:, :3, 3]
    bone_color = [0.8, 0.3, 0.3, 1.0]
    bone_visual = trimesh.visual.TextureVisuals(material=trimesh.visual.material.PBRMaterial(baseColorFactor=bone_color,
                                                            metallicFactor=0.,
                                                            roughnessFactor=1.,
                                                            doubleSided=True,
                                                            alphaMode='BLEND'))
    for i in range(1, len(bone_heads)):
        bone_head = bone_heads[model.bone_parents[i]]
        bone_tail = bone_heads[i]
        length = torch.norm(bone_tail - bone_head).item()
        if length < 1e-6:
            continue
        cylinder = trimesh.creation.cylinder(radius=0.005, height=length, sections=16)
        t = (bone_head + bone_tail) / 2
        M = roma.special_gramschmidt(torch.stack([bone_tail - bone_head, torch.tensor([0., 0., 1.], dtype=torch.float32)], dim=-1))
        R = torch.stack([M[:, 2], M[:, 1], M[:,0]], dim=-1)
        cylinder.visual = bone_visual
        scene.add_geometry(cylinder, transform=roma.Rigid(R, t).to_homogeneous().numpy(),
                            node_name=f"bone_{model.bone_labels[i]}")


    # Add some spheres at the joints
    joint_sphere = trimesh.creation.icosphere(radius=0.008, subdivisions=2)
    joint_sphere.visual = trimesh.visual.TextureVisuals(material=trimesh.visual.material.PBRMaterial(
            baseColorFactor=[0.1, 0.1, 0.1, 1.0],
            metallicFactor=0.5,
            roughnessFactor=1.,
            doubleSided=True,
            alphaMode='OPAQUE'))
    for i in range(len(bone_poses)):
        scene.add_geometry(joint_sphere, transform=bone_poses[i], node_name=f"joint_{model.bone_labels[i]}")

# %% [markdown]
# ## Rigs and topology
#
# Anny supports two main rigs: "anny" and "mixamo". Bones that are not useful for your application can be removed from the Anny rig. Using "anny-notoes" will ignore bones animating individual toes, for example.
#
# Anny also supports various mesh topologies. A topology such as "notoes_collapse5pc" provides coarser mesh output for example, allowing to speed up inference and reduce memory consumption.
# We provide a "smplx" topology for interoperability with the SMPL-X model (https://smpl-x.is.tue.mpg.de/), **for non-commercial use only**.
#
# We show below a few combinations of meshes and topologies supported by the model:

# %%
viewers = []

for rig, topology in [("anny", "anny"),
                      ("mixamo", "anny",),
                      ("anny", "smplx",),
                      ("anny-notoes-noexpression-nobreasts", "makehuman"),
                      ("anny-notoes-nohands-noexpression-nobreasts", "notoes_collapse5pc"),
                      ]:
    model = anny.Anny(rig=rig, topology=topology)
    output = model()

    mesh = trimesh.Trimesh(
        vertices=output["vertices"].squeeze(0).cpu().numpy(),
        faces=model.faces.cpu().numpy()
    )
    mesh.visual.material = mesh_material
    scene = trimesh.Scene([mesh])

    add_skeleton_to_scene(scene, model, output)
    scene.apply_transform(trimesh_scene_transform)

    # Convert to a notebook widget/HTML
    viewers.append(Markdown(f"#### '{rig}' rig ({model.bone_count} bones)"))
    viewers.append(Markdown( "  - " + ", ".join([label for label in model.bone_labels])))
    viewers.append(Markdown(f"#### '{topology}' topology ({len(output['vertices'].squeeze(0))} vertices, {len(model.faces)} faces)"))
    viewers.append(nb.scene_to_notebook(scene))


# Display all viewers
display(*viewers)
