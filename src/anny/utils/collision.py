# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
import torch
import math
import numpy as np
import warp as wp

wp.init()


def get_intersection_kernel(mask_uint32_length):
    IntersectionMask = wp.types.vector(length=mask_uint32_length, dtype=wp.uint32)

    @wp.func
    def cw_min(a: wp.vec3, b: wp.vec3):
        return wp.vec3(wp.min(a[0], b[0]), wp.min(a[1], b[1]), wp.min(a[2], b[2]))

    @wp.func
    def cw_max(a: wp.vec3, b: wp.vec3):
        return wp.vec3(wp.max(a[0], b[0]), wp.max(a[1], b[1]), wp.max(a[2], b[2]))

    @wp.func
    def project_triangle(axis: wp.vec3, p0: wp.vec3, p1: wp.vec3, p2: wp.vec3):
        """Projects a triangle onto an axis and returns min/max values."""
        p0_proj = wp.dot(p0, axis)
        p1_proj = wp.dot(p1, axis)
        p2_proj = wp.dot(p2, axis)
        return wp.min(wp.min(p0_proj, p1_proj), p2_proj), wp.max(
            wp.max(p0_proj, p1_proj), p2_proj
        )

    @wp.func
    def overlap(
        min_a: wp.Float, max_a: wp.Float, min_b: wp.Float, max_b: wp.Float
    ) -> wp.bool:
        """Checks if two 1D projections overlap."""
        return not (max_a < min_b or max_b < min_a)

    @wp.func
    def separable_axis(
        a0: wp.vec3,
        a1: wp.vec3,
        a2: wp.vec3,
        b0: wp.vec3,
        b1: wp.vec3,
        b2: wp.vec3,
        axis: wp.vec3,
    ):
        """Test if the projection of two triangles are separated along a given axis."""
        # Test face normals as separating axes
        min_a, max_a = project_triangle(axis, a0, a1, a2)
        min_b, max_b = project_triangle(axis, b0, b1, b2)
        return not overlap(min_a, max_a, min_b, max_b)

    @wp.func
    def separable_axis_edge_edge(
        a0: wp.vec3,
        a1: wp.vec3,
        a2: wp.vec3,
        b0: wp.vec3,
        b1: wp.vec3,
        b2: wp.vec3,
        e0: wp.vec3,
        e1: wp.vec3,
    ):
        """Test if the projection of two triangles are separated along an axis
        corresponding to the cross product of two edges."""
        axis = wp.cross(e0, e1)
        if wp.dot(axis, axis) > 1e-6:  # Ensure valid axis (avoid degenerate cases)
            axis = wp.normalize(axis)  # Normalize to avoid precision issues
            return separable_axis(a0, a1, a2, b0, b1, b2, axis)
        return False

    @wp.func
    def triangle_intersects_SAT(
        a0: wp.vec3, a1: wp.vec3, a2: wp.vec3, b0: wp.vec3, b1: wp.vec3, b2: wp.vec3
    ) -> wp.bool:
        """Test if two triangles intersect, based on the SAT theorem."""
        # Note: WARP provides an off-the-shelf "wp.intersect_tri_tri" function, but it returns many false positives.
        # Triangle edges
        e1_a = a1 - a0
        e2_a = a2 - a0

        e1_b = b1 - b0
        e2_b = b2 - b0

        # Face normals
        normal_a = wp.normalize(wp.cross(e1_a, e2_a))
        normal_b = wp.normalize(wp.cross(e1_b, e2_b))

        # Test face normals as separating axes
        if separable_axis(a0, a1, a2, b0, b1, b2, normal_a):
            return False
        if separable_axis(a0, a1, a2, b0, b1, b2, normal_b):
            return False
        e3_a = a2 - a1
        e3_b = b2 - b1
        if separable_axis_edge_edge(a0, a1, a2, b0, b1, b2, e1_a, e1_b):
            return False
        if separable_axis_edge_edge(a0, a1, a2, b0, b1, b2, e1_a, e2_b):
            return False
        if separable_axis_edge_edge(a0, a1, a2, b0, b1, b2, e1_a, e3_b):
            return False
        if separable_axis_edge_edge(a0, a1, a2, b0, b1, b2, e2_a, e1_b):
            return False
        if separable_axis_edge_edge(a0, a1, a2, b0, b1, b2, e2_a, e2_b):
            return False
        if separable_axis_edge_edge(a0, a1, a2, b0, b1, b2, e2_a, e3_b):
            return False
        if separable_axis_edge_edge(a0, a1, a2, b0, b1, b2, e3_a, e1_b):
            return False
        if separable_axis_edge_edge(a0, a1, a2, b0, b1, b2, e3_a, e2_b):
            return False
        if separable_axis_edge_edge(a0, a1, a2, b0, b1, b2, e3_a, e3_b):
            return False
        return True  # No separating axis found, triangles must intersect

    @wp.func
    def test_mask(
        a: IntersectionMask,  # type: ignore
        b: IntersectionMask,
    ) -> wp.bool:  # type: ignore
        for k in range(mask_uint32_length):
            # Static loop unrolling
            static_k = wp.static(k)
            if a[static_k] & b[static_k]:
                return True
        return False

    @wp.kernel
    def test_self_intersection(
        query_mesh_id: wp.uint64,
        intersection_mask: wp.array(dtype=IntersectionMask),  # type: ignore
        colliding_face: wp.array(dtype=wp.int32),
    ):  # type: ignore
        tid = wp.tid()

        query_face_id = tid
        query_mask = intersection_mask[query_face_id]

        mesh = wp.mesh_get(query_mesh_id)
        i0, i1, i2 = (
            mesh.indices[query_face_id * 3],
            mesh.indices[query_face_id * 3 + 1],
            mesh.indices[query_face_id * 3 + 2],
        )

        # Retrieve triangle vertices location
        v0, v1, v2 = mesh.points[i0], mesh.points[i1], mesh.points[i2]

        # compute bounds of the query triangle
        lower = cw_min(cw_min(v0, v1), v2)
        upper = cw_max(cw_max(v0, v1), v2)

        # Retrieve potentially intersecting triangles
        candidates = wp.mesh_query_aabb(query_mesh_id, lower, upper)
        for target_face_id in candidates:
            # Do not test a triangle with others sharing the same mask
            mask = intersection_mask[target_face_id]
            if test_mask(mask, query_mask):
                continue

            j0, j1, j2 = (
                mesh.indices[target_face_id * 3],
                mesh.indices[target_face_id * 3 + 1],
                mesh.indices[target_face_id * 3 + 2],
            )

            # Retrieve vertices location
            u0, u1, u2 = mesh.points[j0], mesh.points[j1], mesh.points[j2]
            res = triangle_intersects_SAT(v0, v1, v2, u0, u1, u2)
            if res > 0:
                # There is an intersection: store the face id (not that we only consider a single one)
                colliding_face[query_face_id] = target_face_id
                return
        # No intersection
        colliding_face[query_face_id] = -1

    return test_self_intersection, IntersectionMask


class SelfInterpenetrationModule:
    def __init__(self, body_model, group_toes=True, group_eyes=True, group_tongue=True):
        wp.init()
        self.triangular_faces = torch.as_tensor(
            body_model.get_triangular_faces(), device=body_model.device
        )
        self.wp_triangular_faces = wp.from_torch(
            self.triangular_faces.to(dtype=torch.int32).flatten()
        )

        if True:
            bone_collision_groups = []
            for bone_id, bone_label in enumerate(body_model.bone_labels):
                if "toe" in bone_label and group_toes:
                    if bone_label.endswith(".L"):
                        label = "left_toes"
                    else:
                        assert bone_label.endswith(".R")
                        label = "right_toes"
                elif "eye" in bone_label and group_eyes:
                    label = "head"
                elif "tongue" in bone_label and group_tongue:
                    label = "head"
                else:
                    label = bone_label
                bone_collision_groups.append(label)

            collision_group_labels = set(bone_collision_groups)
            collision_group_ids_map = {
                label: i for i, label in enumerate(collision_group_labels)
            }
            bone_id_to_mask_id = [
                collision_group_ids_map[label] for label in bone_collision_groups
            ]

        # Compute a per-vertex binary bone mask.
        # Use pure Python to avoid overflows, as the bone count is typically greater than 64
        # Note: we could exclude bones with no weights to reduce the mask size
        vertex_bone_indices = (
            body_model.vertex_bone_indices.detach().cpu().numpy().tolist()
        )
        vertex_bone_weights = body_model.vertex_bone_weights.detach().cpu().numpy()
        n = len(body_model.template_vertices)
        per_vertex_mask = [0 for _ in range(n)]
        for i in range(n):
            for bone_id, weight in zip(vertex_bone_indices[i], vertex_bone_weights[i]):
                mask_id = bone_id_to_mask_id[bone_id]
                if weight > 0:
                    per_vertex_mask[i] |= 1 << mask_id

        # Compute per face mask
        np_triangular_faces = self.triangular_faces.detach().cpu().numpy()
        face_count = len(np_triangular_faces)
        triangular_faces_mask = [0 for _ in range(face_count)]
        for face_id, (i, j, k) in enumerate(np_triangular_faces):
            triangular_faces_mask[face_id] = (
                per_vertex_mask[i] | per_vertex_mask[j] | per_vertex_mask[k]
            )
        mask_bit_length = int(math.log2(np.max(triangular_faces_mask))) + 1
        print("mask bit length", mask_bit_length)
        mask_uint32_length = int(math.log2(np.max(triangular_faces_mask)) / 32 + 1)

        np_triangular_faces_mask = np.zeros(
            (face_count, mask_uint32_length), dtype=np.uint32
        )
        for face_id in range(face_count):
            remainder = triangular_faces_mask[face_id]
            for block_id in range(mask_uint32_length):
                block_value = remainder & (2**32 - 1)
                np_triangular_faces_mask[face_id][block_id] = block_value
                remainder >>= 32

        # Map these masks to a wp vector
        self.test_self_intersection_kernel, IntersectionMask = get_intersection_kernel(
            mask_uint32_length
        )
        self.wp_triangular_faces_mask = wp.from_numpy(
            np_triangular_faces_mask,
            dtype=IntersectionMask,
            device=wp.device_from_torch(body_model.device),
        )

    def detect_self_intersections(self, vertices: torch.Tensor):
        """Returns for each triangle the index of a triangle it interesects with (-1 if there is no collision)."""
        assert vertices.dim() == 2
        wp_vertices = wp.from_torch(
            vertices.to(dtype=torch.float32).contiguous(),
            dtype=wp.vec3,
            requires_grad=False,
        )
        mesh = wp.Mesh(points=wp_vertices, indices=self.wp_triangular_faces)

        faces_count = len(self.triangular_faces)
        wp_colliding_face = wp.zeros(
            faces_count, dtype=wp.int32, device=wp_vertices.device
        )
        wp.launch(
            self.test_self_intersection_kernel,
            dim=len(self.triangular_faces),
            inputs=[mesh.id, self.wp_triangular_faces_mask, wp_colliding_face],
            device=wp_vertices.device,
        )
        colliding_face = wp.to_torch(wp_colliding_face)
        return colliding_face


if __name__ == "__main__":
    import anny

    anny_model = anny.Anny()

    collision_module = SelfInterpenetrationModule(anny_model)
    output = anny_model()
    colliding_triangular_faces = collision_module.detect_self_intersections(
        output["vertices"].squeeze(dim=0)
    )
    is_collision = torch.any(colliding_triangular_faces > 0)
    print(is_collision)
