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

# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0

# %% [markdown]
# ## Transferring a pose between rigs
#
# Anny can describe the same body with different rig conventions: the `makehuman` rig follows
# MPFB2 convention, aligning bones along the head-to-tail direction, while the newer `anny` rig
# uses a bone attachment strategy more stable to pose deformations.
# As a result, the *same* numerical parameters mean different things on the two rigs.
# `anny.utils.pose.transfer_pose_parameters` re-expresses a pose defined on a *source* model as the
# pose parameters of a *target* model, so that both produce the same posed mesh. It works as long as
# the shared bones (matched by name) have the same rest origins in both rigs and the two models
# describe the same body at rest.

# %% [markdown]
# #### Imports and helpers

# %%
import torch
import roma  # A PyTorch library useful to deal with space transformations.
import anny  # The main library for the Anny model.
import trimesh  # For 3D mesh visualization.
import anny.utils.pose

# The gradio/trimesh viewers use a Y-up camera; rotate the scene to compensate for Anny's Z-up frame.
trimesh_scene_transform = roma.Rigid(linear=roma.euler_to_rotmat('x', [-90.], degrees=True), translation=None).to_homogeneous().cpu().numpy()

def body_material(color):
    return trimesh.visual.material.PBRMaterial(baseColorFactor=color, metallicFactor=0.5,
                                               doubleSided=False, alphaMode='BLEND')

def body_mesh(vertices, faces, color):
    mesh = trimesh.Trimesh(vertices=vertices.squeeze(0).cpu().numpy(), faces=faces.cpu().numpy(), process=False)
    mesh.visual = trimesh.visual.TextureVisuals(material=body_material(color))
    return mesh

# %% [markdown]
# #### The source and target models
#
# Both use the `anny` topology (same mesh) but different rigs, hence different rest bone orientations.

# %%
src_model = anny.Anny(rig="makehuman", topology="anny").to(dtype=torch.float32)      # blender (tail) orientation
target_model = anny.Anny(rig="anny", topology="anny").to(dtype=torch.float32)        # procrustes orientation

phenotype_kwargs=dict()
local_changes_kwargs=dict()

# %% [markdown]
# #### Pose the source model
#
# We raise the left upper arm by rotating its bone on the source rig.

# %%
src_pose_parameters = {label: torch.eye(4)[None] for label in src_model.bone_labels}
src_pose_parameters["lowerarm01.L"] = roma.Rigid(roma.euler_to_rotmat("x", [-60.], degrees=True), translation=None).to_homogeneous()[None]

src_output = src_model(phenotype_kwargs=phenotype_kwargs, local_changes_kwargs=local_changes_kwargs, pose_parameters=src_pose_parameters)

scene = trimesh.Scene()
scene.add_geometry(body_mesh(src_output["vertices"], src_model.faces, [0.4, 0.8, 0.8, 1.0]))
scene.apply_transform(trimesh_scene_transform)
scene.show()

# %% [markdown]
# #### Transfer the pose to the target rig
#
# `transfer_pose_parameters` returns the target rig's pose parameters that reproduce the same pose.
# We then overlay the two posed meshes: the target (orange) matches the source (teal).

# %%
target_pose_parameters = anny.utils.pose.transfer_pose_parameters(
    src_model=src_model,
    src_pose_parameters=src_pose_parameters,
    phenotype_kwargs=local_changes_kwargs,
    local_changes_kwargs=local_changes_kwargs,
    target_model=target_model,
)
target_output = target_model(phenotype_kwargs=phenotype_kwargs, local_changes_kwargs=local_changes_kwargs, pose_parameters=target_pose_parameters)

scene = trimesh.Scene()
scene.add_geometry(body_mesh(src_output["vertices"], src_model.faces, [0.4, 0.8, 0.8, 0.5]))
scene.add_geometry(body_mesh(target_output["vertices"], target_model.faces, [0.9, 0.6, 0.2, 0.5]))
scene.apply_transform(trimesh_scene_transform)
scene.show()

# %% [markdown]
# The two meshes coincide up to numerical precision:

# %%
max_error = torch.linalg.norm(src_output["vertices"] - target_output["vertices"], dim=-1).max()
print(f"max vertex distance between the source and re-posed target meshes: {max_error:.2e} m")

# %% [markdown]
# #### How it works
#
# Linear blend skinning deforms each vertex by a per-bone *delta* `bone_pose @ rest_bone_pose⁻¹`,
# relative to the rig's own rest frame. `transfer_pose_parameters` reconstructs the target's world
# bone poses as `BP_src @ RP_src⁻¹ @ RP_tgt` so that the target's delta `BP_tgt @ RP_tgt⁻¹` equals the
# source's delta `BP_src @ RP_src⁻¹`, cancelling the difference in rest orientations. The pose is
# encoded in the target's own pose parameterization, so it can be edited or exported like any other
# Anny pose. Only bones shared by name and origin are transferred; bones unique to the source (e.g.
# tongue or expression bones absent from the target) are ignored.
