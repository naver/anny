# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
import torch
import numpy as np
import yaml
import PIL.Image
from anny.paths import get_anny_root_dir

def get_face_segmentation_mask(anny_model, labels,
                               image_path=None,
                               metadata_path=None):
    if image_path is None:
        image_path = get_anny_root_dir() / "data/segmentation/body_parts_segmentation.png"

    if metadata_path is None:
        metadata_path = get_anny_root_dir() / "data/segmentation/body_parts_segmentation.yaml"

    body_parts_segmentation_image = PIL.Image.open(image_path).convert("RGB")

    with open(metadata_path, "r") as f:
        body_parts_segmentation = yaml.safe_load(f)

    # Retrieve the central color of each face
    body_parts_segmentation_array = np.asarray(body_parts_segmentation_image)
    face_center_texture_coordinates = anny_model.texture_coordinates[anny_model.face_texture_coordinate_indices].mean(dim=1)

    u = torch.round(face_center_texture_coordinates[:, 0] * body_parts_segmentation_array.shape[1]).to(dtype=torch.int64).clamp_max(body_parts_segmentation_array.shape[0] - 1).detach().cpu().numpy()
    v = torch.round((1-face_center_texture_coordinates[:, 1]) * body_parts_segmentation_array.shape[0]).to(dtype=torch.int64).clamp_max(body_parts_segmentation_array.shape[1] - 1).detach().cpu().numpy()
    face_colors = body_parts_segmentation_array[v,u]

    face_mask = np.zeros(len(anny_model.faces), dtype=bool)

    for label in labels:
        face_mask |= np.all(face_colors == np.asarray(body_parts_segmentation['colors'][label]), axis=-1)

    face_mask = torch.as_tensor(face_mask, device=anny_model.device, dtype=torch.bool)
    return face_mask
