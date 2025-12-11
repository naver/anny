# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
import torch
import os
from anny.paths import ANNY_ROOT_DIR

def load_tensor(path, key=None):
    data = torch.load(path, map_location="cpu", weights_only=True)
    return data[key] if key else data

class VertexRegressor(torch.nn.Module):
    def __init__(self,
                 type="anny_to_smplx",
                 root_dirname=ANNY_ROOT_DIR):
        super().__init__()

        self.root_dir = root_dirname
        self.type = type
        coeffs_data = load_tensor(os.path.join(self.root_dir, "data/interop/smplx/regression_coefficients_202410.pth"))

        self.source_coeffs = coeffs_data["source_regression_coefficients"]
        self.target_coeffs = coeffs_data["target_regression_coefficients"]

        dispatch = {
            "anny_to_smplx": self._load_anny_to_smplx,
            "smplx_to_anny": self._load_smplx_to_anny,
        }

        if type not in dispatch:
            raise ValueError(f"Unknown type: {type}")
        
        regression_coefficients = dispatch[type]()
        self.register_buffer("regression_coefficients", regression_coefficients[None], persistent=False)

    def _load_anny_to_smplx(self):
        return self.target_coeffs

    def _load_smplx_to_anny(self):
        return self.source_coeffs

    def __call__(self, vertices):
        """
        Args:
            - vertices: torch.Tensor [batch_size,V,3]
        """

        # Loop over batch and apply sparse - not super efficient but cannot handle variable batch size otherwise
        batch_size, *_ = vertices.shape
        regressed_vertices = torch.cat([torch.bmm(self.regression_coefficients, vertices[[i]]) for i in range(batch_size)], dim=0)  # [B, J, 3]

        return regressed_vertices