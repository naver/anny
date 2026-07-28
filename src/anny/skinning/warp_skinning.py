# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
import torch
import warp as wp

wp.init()


def grad_or_none(array):
    return wp.to_torch(array.grad) if array.requires_grad else None


def get_kernel(
    max_bones_count,
    vertices_scalar_dtype,
    weights_dtype,
    indices_dtype,
    vec3_dtype=wp.vec3,
    mat44_dtype=wp.mat44,
):
    """
    Generate the kernel for linear blend skinning.
    """

    @wp.kernel
    def kernel(
        vertices: wp.array2d(dtype=vec3_dtype),
        bone_weights: wp.array(dtype=weights_dtype),
        bone_indices: wp.array(dtype=indices_dtype),
        bone_transforms: wp.array2d(dtype=mat44_dtype),
        output: wp.array2d(dtype=vec3_dtype),
    ):
        batch_id, vertex_id = wp.tid()
        vertex = vertices[batch_id, vertex_id]
        ws = bone_weights[vertex_id]
        ids = bone_indices[vertex_id]
        result = vec3_dtype(
            vertices_scalar_dtype(0.0),
            vertices_scalar_dtype(0.0),
            vertices_scalar_dtype(0.0),
        )
        for i in range(wp.static(max_bones_count)):
            result += ws[i] * wp.transform_point(
                bone_transforms[batch_id, ids[i]], vertex
            )
        # Weights are assumed to be normalized, so we can skip the normalization step
        output[batch_id, vertex_id] = result

    return kernel


class LinearBlendSkinning(torch.autograd.Function):
    @staticmethod
    def forward(ctx, vertices, bone_weights, bone_indices, bone_transforms):
        """
        Apply linear blend skinning to a batch of sets of vertices using sparse bone weights.

        Args:
        vertices (torch.Tensor): Tensor of shape (batch_size, num_vertices, 3) containing the vertex positions.
        bone_weights (torch.Tensor): Tensor of shape (num_vertices, max_bones_per_vertex) containing the bone weights for each vertex.
        bone_indices (torch.Tensor): Tensor of shape (num_vertices, max_bones_per_vertex) containing the bone indices for each vertex.
        bone_transforms (torch.Tensor): Tensor of shape (batch_size, num_bones, 4, 4) containing the bone transformation matrices.

        Returns:
        torch.Tensor: Transformed vertices of shape (batch_size, num_vertices, 3).
        """
        B1, num_vertices, _ = vertices.shape
        max_bones_per_vertex = bone_indices.shape[-1]
        batch_size = max(B1, bone_transforms.shape[0])
        assert bone_weights.shape == bone_indices.shape

        if vertices.dtype == torch.float32:
            ctx.vec3_dtype = wp.vec3f
            ctx.mat44_dtype = wp.mat44f
        elif vertices.dtype == torch.float64:
            ctx.vec3_dtype = wp.vec3d
            ctx.mat44_dtype = wp.mat44d
        elif vertices.dtype == torch.float16:
            ctx.vec3_dtype = wp.vec3h
            ctx.mat44_dtype = wp.mat44h
        else:
            raise NotImplementedError(f"Unsupported dtype {vertices.dtype}")
        ctx.vertices_scalar_dtype = wp.dtype_from_torch(vertices.dtype)
        # We detach gradients to avoid having them updated twice
        # (once in the Warp backward pass, once by torch autograd with the gradient values returned by backward)
        ctx.vertices = wp.from_torch(
            vertices.detach().contiguous(), dtype=ctx.vec3_dtype, requires_grad=False
        )
        ctx.bone_weights = wp.from_torch(
            bone_weights.detach().contiguous(),
            dtype=wp.types.vector(
                max_bones_per_vertex, dtype=wp.dtype_from_torch(bone_weights.dtype)
            ),
            requires_grad=False,
        )
        ctx.bone_indices = wp.from_torch(
            bone_indices.detach().contiguous(),
            dtype=wp.types.vector(
                max_bones_per_vertex, dtype=wp.dtype_from_torch(bone_indices.dtype)
            ),
            requires_grad=False,
        )
        ctx.bone_transforms = wp.from_torch(
            bone_transforms.detach().contiguous(),
            dtype=ctx.mat44_dtype,
            requires_grad=False,
        )
        ctx.output = wp.zeros_like(ctx.vertices, requires_grad=False)
        ctx.dim = (batch_size, num_vertices)
        ctx.max_bones_per_vertex = max_bones_per_vertex

        ctx.kernel = get_kernel(
            max_bones_per_vertex,
            vertices_scalar_dtype=ctx.vertices_scalar_dtype,
            weights_dtype=ctx.bone_weights.dtype,
            indices_dtype=ctx.bone_indices.dtype,
            vec3_dtype=ctx.vec3_dtype,
            mat44_dtype=ctx.mat44_dtype,
        )
        wp.launch(
            ctx.kernel,
            dim=ctx.dim,
            inputs=[
                ctx.vertices,
                ctx.bone_weights,
                ctx.bone_indices,
                ctx.bone_transforms,
            ],
            outputs=[ctx.output],
            device=ctx.vertices.device,
        )
        return wp.to_torch(ctx.output)

    @staticmethod
    def backward(ctx, adj_output):
        input_grad = [
            wp.zeros_like(ctx.vertices) if ctx.needs_input_grad[0] else None,
            wp.zeros_like(ctx.bone_weights) if ctx.needs_input_grad[1] else None,
            None,
            wp.zeros_like(ctx.bone_transforms) if ctx.needs_input_grad[3] else None,
        ]

        wp.launch(
            ctx.kernel,
            dim=ctx.dim,
            inputs=[
                ctx.vertices,
                ctx.bone_weights,
                ctx.bone_indices,
                ctx.bone_transforms,
            ],
            outputs=[ctx.output],
            adj_inputs=input_grad,
            adj_outputs=[wp.from_torch(adj_output.contiguous(), dtype=ctx.vec3_dtype)],
            adjoint=True,
            device=ctx.vertices.device,
        )

        return tuple(
            [wp.to_torch(grad) if grad is not None else None for grad in input_grad]
        )


def linear_blend_skinning(vertices, bone_weights, bone_indices, bone_transforms):
    return LinearBlendSkinning.apply(
        vertices, bone_weights.squeeze(0), bone_indices.squeeze(0), bone_transforms
    )


if __name__ == "__main__":
    # Example usage
    batch_size = 192
    num_vertices = 1000
    max_bones_per_vertex = 8
    num_bones = 20
    dtype = torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"

    vertices = torch.rand(
        (batch_size, num_vertices, 3), device=device, dtype=dtype
    )  # Example vertices
    bone_indices = torch.randint(
        0, num_bones, (num_vertices, max_bones_per_vertex), device=device
    )  # Example bone indices
    bone_weights = torch.rand(
        (num_vertices, max_bones_per_vertex), device=device, dtype=dtype
    )  # Example bone weights
    bone_weights = bone_weights / bone_weights.sum(
        dim=-1, keepdim=True
    )  # Normalize bone weights
    bone_transforms = (
        torch.eye(4, device=device, dtype=dtype)
        .unsqueeze(0)
        .unsqueeze(0)
        .expand(batch_size, num_bones, -1, -1)
    )  # Example bone transforms

    bone_transforms = bone_transforms.clone().requires_grad_(True)

    transformed_vertices = linear_blend_skinning(
        vertices, bone_weights, bone_indices, bone_transforms
    )
    foo = torch.sum(transformed_vertices)
    foo.backward()
