# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
import torch
from anny.paths import ANNY_ROOT_DIR
from anny.torch_compat import make_buffer
from pathlib import Path

class KeypointsRegressor(torch.nn.Module):
    """Regresses named anatomical keypoints from model output vertices via a linear blend.

    Args:
        model: an Anny model exposing ``template_vertices`` and ``base_mesh_vertex_indices``.
        path: either a built-in identifier (``"coco"``) or a filesystem path to a ``.pth`` file
            containing a dict mapping keypoint label → per-base-mesh-vertex regression weights
            that sum to 1.
        labels: keypoint label names to regress, in the desired output order. Each must be a
            key of the loaded keypoints dict.

    Forward input: a model output dict containing ``vertices`` of shape ``(B, V, D)``.
    Forward output: tensor of shape ``(B, K, D)`` where ``K = len(labels)``.
    """
    def __init__(self,
                 model,
                 labels : list[str],
                 path : str = "coco"):
        super().__init__()
        if path == "coco":
            path = Path(ANNY_ROOT_DIR) / "data/keypoints/coco.pth"
        keypoints_data = torch.load(path, weights_only=True)

        if len(labels) == 0:
            labels = list(keypoints_data.keys())

        K = len(labels)
        V = len(model.template_vertices)
        dtype = model.dtype
        device = model.template_vertices.device

        regression_weights = torch.zeros((K, V), dtype=dtype, device=device)
        for k, label in enumerate(labels):
            weights = keypoints_data[label][model.base_mesh_vertex_indices].to(dtype=dtype, device=device)
            assert torch.abs(weights.sum() - 1) < 1e-3
            regression_weights[k] = weights
        self.regression_weights = make_buffer(self, "regression_weights", regression_weights, persistent=False)
        self.labels = labels

    def forward(self, model_output):
        return torch.einsum("kv, bvd -> bkd", self.regression_weights, model_output["vertices"])