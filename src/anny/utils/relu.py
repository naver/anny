# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
import torch


class ReLUWithGradientAtZero(torch.autograd.Function):
    """
    Custom ReLU activation function that has a gradient of 1 at zero.
    This is useful for certain applications where we want to maintain a gradient
    at zero to avoid dead variables.
    """

    @staticmethod
    def forward(x):
        output = torch.relu(x)
        return output

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(inputs[0])

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        grad_input = grad_output * (x >= 0)
        return grad_input


def relu_with_gradient_at_zero(x):
    return ReLUWithGradientAtZero.apply(x)
