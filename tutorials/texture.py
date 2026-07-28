# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0

# %% [markdown]
# ## Playing with texture coordinates

# %% [markdown]
# Basic imports

# %%
import numpy as np
import torch
import anny
import PIL.Image
import PIL.ImageDraw
from anny.paths import get_anny_root_dir
import trimesh
import yaml
from IPython.display import display

# %% [markdown]
# Instanciate the body model.
#

# %%
anny_model = anny.Anny()
trimesh.Trimesh(
    anny_model.template_vertices.cpu().numpy(), faces=anny_model.faces.cpu().numpy()
).show()

# %% [markdown]
# Each vertex of each face of the model is associated with some 2D ST texture coordinates.
# It enables to unwrap the mesh onto a 2D image, as illustrated here.

# %%
# Create an empty image with white background
width, height = 1024, 1024
uv_unwrap_image = PIL.Image.new("RGB", (width, height), (0, 0, 0))

# Draw face contours on the texture image
faces = anny_model.faces.cpu().numpy()
face_texture_coordinates_indices = anny_model.face_texture_coordinate_indices.numpy()
st = anny_model.texture_coordinates.numpy()
vertex_absolute_texture_coordinates = (
    np.array([0, height])[None] + st * np.array([width, -height])[None]
)
draw = PIL.ImageDraw.Draw(uv_unwrap_image)
for face_texture_ids in face_texture_coordinates_indices:
    u0, v0 = vertex_absolute_texture_coordinates[face_texture_ids[-1]]
    for i in face_texture_ids:
        u, v = vertex_absolute_texture_coordinates[i]
        draw.line(((u0, v0), (u, v)), fill=(128, 128, 128), width=1)
        u0, v0 = u, v  # Update the starting point for the next line
display(uv_unwrap_image)

# %% [markdown]
# ## Body part segmentation

# %% [markdown]
# We provide a basic segmentation of the mesh of Anny into different semantic body parts.

# %%
path = get_anny_root_dir() / "data/segmentation/body_parts_segmentation.png"
body_parts_segmentation_image = PIL.Image.open(path).convert("RGB")

overlay_image = body_parts_segmentation_image.copy()
mask = PIL.Image.fromarray(np.all(np.asarray(uv_unwrap_image) != 0, axis=-1))
overlay_image.paste(uv_unwrap_image, mask=mask)
display(overlay_image)

with open(
    get_anny_root_dir() / "data/segmentation/body_parts_segmentation.yaml", "r"
) as f:
    body_parts_segmentation = yaml.safe_load(f)
display(f"Body parts: {list(body_parts_segmentation['colors'].keys())}")

# %% [markdown]
# ### 3D visualization
# **Note:** we need to duplicate vertices as trimesh expects one texture coordinate per vertex.

# %%
vertices = anny_model.template_vertices.detach().cpu().numpy()
faces = faces
uv = anny_model.texture_coordinates.cpu().numpy()
duplicated_vertices = vertices[faces.flatten()]
duplicated_faces = np.arange(3 * len(faces)).reshape(-1, 3)
duplicated_uvs = uv[anny_model.face_texture_coordinate_indices.cpu().numpy().flatten()]

mesh = trimesh.Trimesh(
    vertices=duplicated_vertices,
    faces=duplicated_faces,
    process=False,
    maintain_order=True,
)

material = trimesh.visual.material.PBRMaterial(
    baseColorFactor=np.ones(4),
    baseColorTexture=body_parts_segmentation_image,
    metallicFactor=0.5,
    doubleSided=True,
)
import trimesh.visual

mesh.visual = trimesh.visual.texture.TextureVisuals(
    uv=duplicated_uvs, material=material
)

mesh.show()

# %%
# Retrieve the central color of each face
body_parts_segmentation_array = np.asarray(body_parts_segmentation_image)
face_center_texture_coordinates = anny_model.texture_coordinates[
    anny_model.face_texture_coordinate_indices
].mean(dim=1)

u = (
    torch.round(
        face_center_texture_coordinates[:, 0] * body_parts_segmentation_array.shape[1]
    )
    .to(dtype=torch.int64)
    .clamp_max(body_parts_segmentation_array.shape[0] - 1)
    .detach()
    .cpu()
    .numpy()
)
v = (
    torch.round(
        (1 - face_center_texture_coordinates[:, 1])
        * body_parts_segmentation_array.shape[0]
    )
    .to(dtype=torch.int64)
    .clamp_max(body_parts_segmentation_array.shape[1] - 1)
    .detach()
    .cpu()
    .numpy()
)

face_colors = body_parts_segmentation_array[v, u]

# %%
# Segment the head based on face colors
face_mask = np.zeros(len(faces), dtype=bool)

labels = [
    "head",
    "eye_cavity.R",
    "eye_cavity.L",
    "mouth_cavity",
    "eye_front.L",
    "eye_back.L",
    "eye_front.R",
    "eye_back.L",
    "tongue",
]
for label in labels:
    face_mask |= np.all(
        face_colors == np.asarray(body_parts_segmentation["colors"][label]), axis=-1
    )

trimesh.Trimesh(
    vertices=vertices,
    faces=faces[face_mask],
).show()
