# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0

import unittest
import torch
import anny.utils.interpolation


class TestUtils(unittest.TestCase):
    def test_interpolation(self):
        dtype = torch.float32
        device = torch.device('cpu')
        anchors = torch.as_tensor([2,3,4], dtype=dtype, device=device)
        value = torch.as_tensor([1.5, 4.1, 3.5, 2.], dtype=dtype, device=device)
        coeffs = anny.utils.interpolation.linear_interpolation_coefficients(value, anchors)
        self.assertTrue(coeffs.shape == (len(value),len(anchors)))
        epsilon = 1e-3
        self.assertTrue(torch.all(torch.abs(torch.sum(coeffs, dim=-1) - 1) < epsilon))
        self.assertTrue(torch.all(torch.abs(coeffs[0] - torch.as_tensor([1.,0.,0.])) < epsilon))
        self.assertTrue(torch.all(torch.abs(coeffs[1] - torch.as_tensor([0.,0.,1.])) < epsilon))
        self.assertTrue(torch.all(torch.abs(coeffs[2] - torch.as_tensor([0.,0.5,0.5])) < epsilon))
        self.assertTrue(torch.all(torch.abs(coeffs[3] - torch.as_tensor([1.,0.,0.])) < epsilon))

