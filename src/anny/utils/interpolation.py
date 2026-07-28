# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
import torch


def linear_interpolation_coefficients(value, anchors, extrapolate=False):
    """
    Args:
        value (torch.Tensor of shape [bs]): batch of values
        anchors (torch.Tensor of shape [n]): array of monotonically increasing reference values to interpolate from.
    Returns:
        torch.Tensor of shape n x bs : interpolation coefficients corresponding to each value and anchor.
    """
    batch_size = len(value)
    n = len(anchors)

    # # Initialize the weights tensor of shape [bs, n]
    weights_tensor = torch.zeros(
        batch_size, n, dtype=anchors.dtype, device=anchors.device
    )

    # Find the indices where each value falls in the anchors
    idx = torch.searchsorted(anchors, value.contiguous(), side="left")  # Shape: [bs]

    # Clamp idx to valid range
    idx = torch.clamp(idx, 1, n - 1)

    # Compute the interpolation weights for each value in the batch
    lower_anchor = anchors[idx - 1]  # Shape: [bs]
    upper_anchor = anchors[idx]  # Shape: [bs]

    # Compute alpha for linear interpolation
    alpha = (value - lower_anchor) / (upper_anchor - lower_anchor)  # Shape: [bs]
    if not extrapolate:
        alpha = torch.clamp(alpha, 0, 1)

    # Assign values to the weight tensor
    dummy_range = torch.arange(batch_size)
    weights_tensor[dummy_range, idx - 1] = 1.0 - alpha  # Left anchor
    weights_tensor[dummy_range, idx] = alpha  # Right anchor
    return weights_tensor


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Example usage
    anchors = torch.tensor([0.0, 0.5, 1.0])
    values = torch.tensor([0.2, 0.7, 0.9])

    x = torch.linspace(-0.1, 1.1, 50)  # Example values for interpolation
    coeffs = linear_interpolation_coefficients(x, anchors, extrapolate=True)
    y = torch.einsum("bk, k -> b", coeffs, values)
    fig, ax = plt.subplots()
    ax.plot(x, y, label="Interpolated values")
    ax.scatter(anchors, values, color="red", label="Anchors")
    ax.legend()
    plt.show()
    print(coeffs)  # Should print the interpolation coefficients for the given values
