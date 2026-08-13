# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
import torch

import anny
from anny.paths import get_anny_root_dir, PathLike
from anny.torch_compat import make_buffer


def _load_keypoint_data(
    model: anny.Anny, path: PathLike, labels: list[str] | None = None
) -> tuple[torch.Tensor, list[str]]:
    dtype = model.dtype
    device = model.device
    keypoints_data = torch.load(path, weights_only=True, map_location=device)
    if labels is None:
        labels = list(keypoints_data.keys())
    K = len(labels)
    V = len(model.template_vertices)

    regression_weights = torch.zeros((K, V), dtype=dtype, device=device)
    for k, label in enumerate(labels):
        weights = keypoints_data[label][model.base_mesh_vertex_indices].to(
            dtype=dtype, device=device
        )
        assert torch.abs(weights.sum() - 1) < 1e-3
        regression_weights[k] = weights
    return regression_weights, labels


class KeypointsRegressor(torch.nn.Module):
    """Regresses named anatomical keypoints from model output vertices via a linear blend.

    Args:
        model: an Anny model exposing ``template_vertices`` and ``base_mesh_vertex_indices``.
        path: either a built-in identifier (``"coco"``) or a filesystem path to a ``.pth`` file
            containing a dict mapping keypoint label → per-base-mesh-vertex regression weights
            that sum to 1.
        labels: keypoint label names to regress, in the desired output order. Each must be a
            key of the loaded keypoints dict.

    Weights are either dense, one per model vertex, or sparse, given together with the
    ``regression_indices`` of the vertices they apply to.

    Forward input: a model output dict containing ``vertices`` of shape ``(B, V, D)``.
    Forward output: tensor of shape ``(B, K, D)`` where ``K = len(labels)``.
    """

    @classmethod
    def coco(cls, model: anny.Anny, labels: list[str] | None = None):
        return cls.load_precomputed(
            model,
            get_anny_root_dir() / "data" / "keypoints" / "coco.pth",
            labels=labels,
        )

    @classmethod
    def load_precomputed(
        cls, model: anny.Anny, path: PathLike, labels: list[str] | None = None
    ):
        regression_weights, labels = _load_keypoint_data(model, path, labels)
        return cls(regression_weights, labels)

    def __init__(
        self,
        regression_weights: torch.Tensor,
        labels: list[str],
        regression_indices: torch.Tensor | None = None,
    ):
        super().__init__()
        self.regression_weights = make_buffer(
            self, "regression_weights", regression_weights, persistent=False
        )
        # Dense weights regress from every vertex, so their indices are implicit; keeping them
        # implicit lets the forward stay a plain matrix product instead of a gather.
        self.is_sparse = regression_indices is not None
        if regression_indices is None:
            regression_indices = (
                torch.arange(0, self.regression_weights.shape[1])
                .to(device=self.regression_weights.device)[None]
                .expand(self.regression_weights.shape[0], -1)
            )
        else:
            if regression_indices.shape != regression_weights.shape:
                raise ValueError("Expecting indices and weights to have the same shape")
        self.regression_indices = make_buffer(
            self, "regression_indices", regression_indices, persistent=False
        )
        self.labels = labels

    def forward(self, model_output: dict[str, torch.Tensor]) -> torch.Tensor:
        vertices = model_output["vertices"]
        if not self.is_sparse:
            return torch.einsum("kv, bvd -> bkd", self.regression_weights, vertices)
        # (B, K, S, D) gather of the regressed vertices only.
        return torch.einsum(
            "ks, bksd -> bkd",
            self.regression_weights,
            vertices[:, self.regression_indices],
        )
