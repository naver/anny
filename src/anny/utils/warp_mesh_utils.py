# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
import torch
import warp as wp
import sys

# Initialize Warp (if not already initialized) without printing to stdout
sys.stdout = sys.stderr
wp.init()
sys.stdout = sys.__stdout__


def to_torch_or_none(array):
    return wp.to_torch(array) if array is not None else None


@wp.func
def point_segment_projection(point: wp.vec3f,
                            v0: wp.vec3f,
                            v1: wp.vec3f) -> wp.vec3f:
    """
    Return the projection of a 3D point on a segment.
    """
    # Compute the projection of the point onto the segment defined by v0 and v1
    v0_to_v1 = v1 - v0
    v0_to_point = point - v0
    t = wp.dot(v0_to_point, v0_to_v1) / wp.dot(v0_to_v1, v0_to_v1)
    beta = wp.clamp(t, 0.0, 1.0)
    alpha = 1.0 - beta
    return alpha * v0 + beta * v1

def get_length_func(safe_length: bool = False):
    """
    Return a length function that is numerically stable for small vectors if safe_length is True.
    """
    if safe_length:
        @wp.func
        def length_func(v: wp.vec3f) -> wp.float32:
            max = wp.max(wp.max(wp.abs(v.x), wp.abs(v.y)), wp.abs(v.z))
            if max == 0.0:
                return 0.0
            scaled_v = v / max
            return max * wp.sqrt(wp.dot(scaled_v, scaled_v))
    else:
        length_func = wp.length
    return length_func

def get_point_triangle_distance_func(safe_length: bool = False):
    length_func = get_length_func(safe_length)
    @wp.func
    def point_triangle_distance(point: wp.vec3f,
                                    v0: wp.vec3f,
                                    v1: wp.vec3f,
                                    v2: wp.vec3f) -> wp.float32:
        # Compute barycentric coordinates manually
        ab = v1 - v0
        ac = v2 - v0
        ax = point - v0
        bb = wp.length_sq(ab)
        cc = wp.length_sq(ac)
        bc = wp.dot(ab,ac)
        bx = wp.dot(ab,ax)
        cx = wp.dot(ac,ax)
        inv_det = 1.0 / (bb*cc - bc*bc)
        beta = inv_det * (cc*bx - bc*cx)
        gamma = inv_det * (bb*cx - bc*bx)
        alpha = 1.0 - beta - gamma

        inplane_projection = alpha * v0 + beta * v1 + gamma * v2

        is_inside = (alpha >= 0.0) and (beta >= 0.0) and (gamma >= 0.0)
        if is_inside:
            # The planar projection is inside the triangle
            return length_func(point - inplane_projection)
        else:
            # The planar projection is outside the triangle: consider point/segment and point/vertices distances to get proper gradients
            proj_01 = point_segment_projection(point, v0, v1)
            proj_02 = point_segment_projection(point, v0, v2)
            proj_12 = point_segment_projection(point, v1, v2)
            dist_01 = length_func(point - proj_01)
            dist_02 = length_func(point - proj_02)
            dist_12 = length_func(point - proj_12)
            return wp.min(wp.min(dist_01, dist_02), dist_12)
    return point_triangle_distance

_point_to_mesh_distance_kernel_cache = dict()
def get_point_to_mesh_distance_kernel(safe_length: bool = True):
    global _point_to_mesh_distance_kernel_cache
    try:
        return _point_to_mesh_distance_kernel_cache[safe_length]
    except KeyError:
        point_triangle_distance = get_point_triangle_distance_func(safe_length)

        @wp.kernel
        def point_to_mesh_distance_kernel(mesh_id: wp.uint64,
                                    points: wp.array(dtype=wp.vec3f),
                                    max_dist: wp.float32,
                                    distances: wp.array(dtype=wp.float32)):
            tid = wp.tid()
            point = points[tid]

            query = wp.mesh_query_point_no_sign(mesh_id, point, max_dist)
            if query.result:
                face_id = query.face
                # Vertex gradient does not flow properly when using wp.mesh_eval_position
                mesh = wp.mesh_get(mesh_id)
                i = 3 * face_id
                v0 = mesh.points[mesh.indices[i]]
                v1 = mesh.points[mesh.indices[i + 1]]
                v2 = mesh.points[mesh.indices[i + 2]]
                dist = point_triangle_distance(point, v0, v1, v2)
                distances[tid] = wp.min(dist, max_dist)
            else:
                distances[tid] = max_dist
        _point_to_mesh_distance_kernel_cache[safe_length] = point_to_mesh_distance_kernel
        return point_to_mesh_distance_kernel

_point_to_mesh_distance_and_face_kernel_cache = dict()
def get_point_to_mesh_distance_and_face_kernel(safe_length: bool = False):
    try:
        return _point_to_mesh_distance_and_face_kernel_cache[safe_length]
    except KeyError:
        point_triangle_distance = get_point_triangle_distance_func(safe_length)

        @wp.kernel
        def point_to_mesh_distance_and_face_kernel(mesh_id: wp.uint64,
                                    points: wp.array(dtype=wp.vec3f),
                                    max_dist: wp.float32,
                                    distances: wp.array(dtype=wp.float32),
                                    faces: wp.array(dtype=wp.int32)):
            tid = wp.tid()
            point = points[tid]

            query = wp.mesh_query_point_no_sign(mesh_id, point, max_dist)
            if query.result:
                face_id = query.face
                # Vertex gradient does not flow when using wp.mesh_eval_position
                mesh = wp.mesh_get(mesh_id)
                i = 3 * face_id
                v0 = mesh.points[mesh.indices[i]]
                v1 = mesh.points[mesh.indices[i + 1]]
                v2 = mesh.points[mesh.indices[i + 2]]
                # This should produce the same result as the above, but with better gradients? (point to plane distance or point to segment distance when the projection is not on the boundary)
                dist = point_triangle_distance(point, v0, v1, v2)
                distances[tid] = wp.min(dist, max_dist)
                faces[tid] = face_id
            else:
                distances[tid] = max_dist
                faces[tid] = -1  # No face found, set to -1 by convention
        _point_to_mesh_distance_and_face_kernel_cache[safe_length] = point_to_mesh_distance_and_face_kernel
        return point_to_mesh_distance_and_face_kernel

_point_to_mesh_distance_and_face_uvs_kernel_cache = dict()
def get_point_to_mesh_distance_and_face_uvs_kernel(safe_length: bool = False):
    try:
        return _point_to_mesh_distance_and_face_uvs_kernel_cache[safe_length]
    except KeyError:
        point_triangle_distance = get_point_triangle_distance_func(safe_length)
        @wp.kernel
        def point_to_mesh_distance_and_face_uvs_kernel(mesh_id: wp.uint64,
                                    points: wp.array(dtype=wp.vec3f),
                                    max_dist: wp.float32,
                                    distances: wp.array(dtype=wp.float32),
                                    faces: wp.array(dtype=wp.int32),
                                    uvs: wp.array(dtype=wp.vec2f)):
            tid = wp.tid()
            point = points[tid]

            query = wp.mesh_query_point_no_sign(mesh_id, point, max_dist)
            if query.result:
                face_id = query.face
                # Vertex gradient does not flow when using wp.mesh_eval_position
                mesh = wp.mesh_get(mesh_id)
                i = 3 * face_id
                v0 = mesh.points[mesh.indices[i]]
                v1 = mesh.points[mesh.indices[i + 1]]
                v2 = mesh.points[mesh.indices[i + 2]]
                dist = point_triangle_distance(point, v0, v1, v2)
                distances[tid] = wp.min(dist, max_dist)
                faces[tid] = face_id
                uvs[tid] = wp.vec2f(query.u, query.v)  # Store the UV coordinates of the closest point
            else:
                distances[tid] = max_dist
                faces[tid] = -1  # No face found, set to -1 or any invalid index
                uvs[tid] = wp.vec2f(0.0, 0.0)  # No UV coordinates available
        _point_to_mesh_distance_and_face_uvs_kernel_cache[safe_length] = point_to_mesh_distance_and_face_uvs_kernel
        return point_to_mesh_distance_and_face_uvs_kernel

class PointToMeshDistance(torch.autograd.Function):
    @staticmethod
    def forward(ctx, points, vertices, faces, max_dist, safe_length: bool = False):
        assert faces.shape[-1] == 3, "Faces must be a list of triangles"
        assert points.shape[-1] == 3, "Points must be a list of 3D coordinates"
        assert vertices.shape[-1] == 3, "Vertices must be a list of 3D coordinates"
        assert vertices.dim() == 2, "Vertices must be a 2D tensor"
        assert points.dim() == 2, "Points must be a 2D tensor"
        assert faces.dim() == 2, "Faces must be a 2D tensor"

        ctx.warp_device = wp.device_from_torch(points.device)
        ctx.dim = points.shape[0]
        # We detach gradients to avoid having them updated twice (once by warp backwarp pass, once by torch autograd)
        ctx.points = wp.from_torch(points.detach(), dtype=wp.vec3f, requires_grad=points.requires_grad)
        ctx.vertices = wp.from_torch(vertices.detach(), dtype=wp.vec3f, requires_grad=vertices.requires_grad)
        ctx.output = wp.zeros(ctx.dim, dtype=wp.float32, device=wp.device_from_torch(points.device), requires_grad=True)
        ctx.mesh = wp.Mesh(points=ctx.vertices,
                        indices=wp.from_torch(faces.flatten().to(dtype=torch.int32)))
        ctx.max_dist = max_dist
        ctx.kernel = get_point_to_mesh_distance_kernel(safe_length)
        wp.launch(ctx.kernel,
                dim=ctx.dim,
                inputs=[ctx.mesh.id, ctx.points, max_dist],
                outputs=[ctx.output],
                device=ctx.warp_device)
        return wp.to_torch(ctx.output, requires_grad=False).detach()

    @staticmethod
    def backward(ctx, grad_output):
        ctx.output.grad = wp.from_torch(grad_output.contiguous())
        wp.launch(ctx.kernel,
                  dim=ctx.dim,
                  inputs=[ctx.mesh.id, ctx.points, ctx.max_dist],
                  outputs=[ctx.output],
                  adj_inputs=[None, ctx.points.grad, None],
                  adj_outputs=[ctx.output.grad],
                  adjoint=True,
                  device=ctx.warp_device)
        return (to_torch_or_none(ctx.points.grad), to_torch_or_none(ctx.vertices.grad), None, None, None)

def point_to_mesh_distance(points, vertices, faces, max_dist, safe_length: bool = False):
    return PointToMeshDistance.apply(points.contiguous(), vertices.contiguous(), faces, max_dist, safe_length)

class PointToMeshDistanceAndFace(torch.autograd.Function):
    @staticmethod
    def forward(ctx, points, vertices, faces, max_dist, safe_length: bool = False):
        assert faces.shape[-1] == 3, "Faces must be a list of triangles"
        assert points.shape[-1] == 3, "Points must be a list of 3D coordinates"
        assert vertices.shape[-1] == 3, "Vertices must be a list of 3D coordinates"
        assert vertices.dim() == 2, "Vertices must be a 2D tensor"
        assert points.dim() == 2, "Points must be a 2D tensor"
        assert faces.dim() == 2, "Faces must be a 2D tensor"

        ctx.warp_device = wp.device_from_torch(points.device)
        ctx.dim = points.shape[0]
        # We detach gradients to avoid having them updated twice (once by warp backwarp pass, once by torch autograd)
        ctx.points = wp.from_torch(points.detach(), dtype=wp.vec3f, requires_grad=points.requires_grad)
        ctx.vertices = wp.from_torch(vertices.detach(), dtype=wp.vec3f, requires_grad=vertices.requires_grad)
        ctx.distances_output = wp.zeros(ctx.dim, dtype=wp.float32, device=wp.device_from_torch(points.device), requires_grad=True)
        ctx.faces_output = wp.zeros(ctx.dim, dtype=wp.int32, device=wp.device_from_torch(points.device), requires_grad=False)
        ctx.mesh = wp.Mesh(points=ctx.vertices,
                        indices=wp.from_torch(faces.flatten().to(dtype=torch.int32)))
        ctx.max_dist = max_dist
        ctx.kernel = get_point_to_mesh_distance_and_face_kernel(safe_length)
        wp.launch(ctx.kernel,
                dim=ctx.dim,
                inputs=[ctx.mesh.id, ctx.points, max_dist],
                outputs=[ctx.distances_output, ctx.faces_output],
                device=ctx.warp_device)
        return wp.to_torch(ctx.distances_output, requires_grad=False).detach(), wp.to_torch(ctx.faces_output, requires_grad=False).detach()

    @staticmethod
    def backward(ctx, grad_output, dummy_grad_faces):
        ctx.distances_output.grad = wp.from_torch(grad_output.contiguous())
        ctx.faces_output.grad = None
        # We use this kernel to compute the gradients, as the faces are not differentiable.
        wp.launch(ctx.kernel,
                dim=ctx.dim,
                inputs=[ctx.mesh.id, ctx.points, ctx.max_dist],
                outputs=[ctx.distances_output, ctx.faces_output],
                adj_inputs=[None, ctx.points.grad, None],
                adj_outputs=[ctx.distances_output.grad, None],
                adjoint=True,
                device=ctx.warp_device)
        # Note: one could use a kernel specific to compute gradients only for the closest face.
        return (to_torch_or_none(ctx.points.grad), to_torch_or_none(ctx.vertices.grad), None, None, None)

def point_to_mesh_distance_and_face(points, vertices, faces, max_dist, safe_length: bool = False):
    return PointToMeshDistanceAndFace.apply(points.contiguous(), vertices.contiguous(), faces, max_dist, safe_length)

class PointToMeshDistanceAndFaceUVs(torch.autograd.Function):
    @staticmethod
    def forward(ctx, points, vertices, faces, max_dist, safe_length: bool = False):
        assert faces.shape[-1] == 3, "Faces must be a list of triangles"
        assert points.shape[-1] == 3, "Points must be a list of 3D coordinates"
        assert vertices.shape[-1] == 3, "Vertices must be a list of 3D coordinates"
        assert vertices.dim() == 2, "Vertices must be a 2D tensor"
        assert points.dim() == 2, "Points must be a 2D tensor"
        assert faces.dim() == 2, "Faces must be a 2D tensor"

        ctx.warp_device = wp.device_from_torch(points.device)
        ctx.dim = points.shape[0]
        # We detach gradients to avoid having them updated twice (once by warp backwarp pass, once by torch autograd)
        ctx.points = wp.from_torch(points.detach(), dtype=wp.vec3f, requires_grad=points.requires_grad)
        ctx.vertices = wp.from_torch(vertices.detach(), dtype=wp.vec3f, requires_grad=vertices.requires_grad)
        ctx.distances_output = wp.zeros(ctx.dim, dtype=wp.float32, device=wp.device_from_torch(points.device), requires_grad=True)
        ctx.faces_output = wp.zeros(ctx.dim, dtype=wp.int32, device=wp.device_from_torch(points.device), requires_grad=False)
        ctx.uvs_output = wp.zeros(ctx.dim, dtype=wp.vec2f, device=wp.device_from_torch(points.device), requires_grad=True)
        ctx.mesh = wp.Mesh(points=ctx.vertices,
                        indices=wp.from_torch(faces.flatten().to(dtype=torch.int32)))
        ctx.max_dist = max_dist
        ctx.kernel = get_point_to_mesh_distance_and_face_uvs_kernel(safe_length)
        wp.launch(ctx.kernel,
                dim=ctx.dim,
                inputs=[ctx.mesh.id, ctx.points, max_dist],
                outputs=[ctx.distances_output, ctx.faces_output, ctx.uvs_output],
                device=ctx.warp_device)
        return (wp.to_torch(ctx.distances_output, requires_grad=False).detach(),
                wp.to_torch(ctx.faces_output, requires_grad=False).detach(),
                wp.to_torch(ctx.uvs_output, requires_grad=False).detach())

    @staticmethod
    def backward(ctx, grad_distances, dummy_grad_faces, grad_uvs):
        ctx.distances_output.grad = wp.from_torch(grad_distances.contiguous())
        ctx.faces_output.grad = None
        ctx.uvs_output.grad = wp.from_torch(grad_uvs.contiguous(), dtype=wp.vec2f)
        wp.launch(ctx.kernel,
                    dim=ctx.dim,
                    inputs=[ctx.mesh.id, ctx.points, ctx.max_dist],
                    outputs=[ctx.distances_output, ctx.faces_output, ctx.uvs_output],
                    adj_inputs=[None, ctx.points.grad, None],
                    adj_outputs=[ctx.distances_output.grad, ctx.faces_output.grad, ctx.uvs_output.grad],
                    adjoint=True,
                    device=ctx.warp_device)
        return (to_torch_or_none(ctx.points.grad), to_torch_or_none(ctx.vertices.grad), None, None)

def point_to_mesh_distance_and_face_uvs(points, vertices, faces, max_dist) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return PointToMeshDistanceAndFaceUVs.apply(points.contiguous(), vertices.contiguous(), faces, max_dist)
