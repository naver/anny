# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
import torch
import collections

def _split_quad1(data):
    a, b, c, d = data
    return ([a, b, c], [c, d, a])

def _split_quad2(data):
    a, b, c, d = data
    return ([a, b, d], [d, b, c])


def _split_quad(vertices, face_vertex_indices, face_texture_coordinate_indices=None):
    """
    Simple triangulation of a quad, consisting in splitting the face along its smaller diagonal.
    Remark: this leads to some interesting artefacts. See the link below for some interesting discussions.
    https://computergraphics.stackexchange.com/questions/10180/how-to-decide-which-way-to-triangulate-a-quad
    """
    assert len(face_vertex_indices) == 4
    v_a, v_b, v_c, v_d = [vertices[i] for i in face_vertex_indices]
    diag1 = torch.linalg.norm(v_a - v_c, dim=-1).item()
    diag2 = torch.linalg.norm(v_b - v_d, dim=-1).item()
    if diag1 < diag2:
        if face_texture_coordinate_indices is None:
            return _split_quad1(face_vertex_indices)
        else:
            return _split_quad1(face_vertex_indices), _split_quad1(face_texture_coordinate_indices)
    else:
        if face_texture_coordinate_indices is None:
            return _split_quad2(face_vertex_indices)
        else:
            return _split_quad2(face_vertex_indices), _split_quad2(face_texture_coordinate_indices)

def triangulate_faces(vertices, faces):
    triangulated_faces = []
    for face in faces:
        if len(face) == 3:
            triangulated_faces.append(face)
        else:
            triangulated_faces.extend(_split_quad(vertices, face))
    return triangulated_faces

def triangulate_faces_with_texture_coordinates(vertices, faces, face_texture_coordinate_indices):
    triangulated_faces = []
    triangulated_face_texture_coordinates_indices = []
    for face_id, face in enumerate(faces):
        if len(face) == 3:
            triangulated_faces.append(face)
            triangulated_face_texture_coordinates_indices.append(face_texture_coordinate_indices[face_id])
        else:
            f, t = _split_quad(vertices, face, face_texture_coordinate_indices[face_id])
            triangulated_faces.extend(f)
            triangulated_face_texture_coordinates_indices.extend(t)
    return triangulated_faces, triangulated_face_texture_coordinates_indices

def get_edge_vertex_indices(faces):
    edges = set()
    for face in faces:
        for i in range(len(face)):
            edge = (face[i-1], face[i]) if face[i-1] < face[i] else (face[i], face[i-1])
            edges.add(edge)
    return torch.as_tensor(list(edges))

def get_boundary_edges(faces):
    if type(faces) is torch.Tensor:
        faces = faces.detach().cpu().numpy().tolist()
    # Number of faces having a specific edge
    edges_face_count = collections.defaultdict(lambda : 0)
    for face in faces:
        for i in range(len(face)):
            edge = (face[i-1], face[i]) if face[i-1] < face[i] else (face[i], face[i-1])
            edges_face_count[edge] += 1
    boundary_edges_vertex_indices = [edge for edge, count in edges_face_count.items() if count == 1]
    return boundary_edges_vertex_indices

def get_corresponding_vertex_indices(source_vertices, target_vertices, threshold):
    """Naive implementation to get for each source vertex the closest corresponding vertex in a target point cloud.
    Args:
        - source_vertices, target_vertices: tensors of shape ...x3"""
    indices = []
    for i in range(len(source_vertices)):
        distances = torch.linalg.norm(source_vertices[i,None] - target_vertices, dim=-1)
        value, index = torch.min(distances, dim=0)
        value = value.item()
        index = index.item()
        if value >= threshold:
            raise ValueError(f"No corresponding vertex found within threshold {threshold} (closest distance: {value})")
        indices.append(index)
    return indices

def get_symmetric_vertex_indices(vertices, axis, threshold):
    """
    Return for each vertex the index of the corresponding symmetric vertex, along a given axis
    """
    symmetric_vertices = vertices.clone()
    symmetric_vertices[...,axis] *= -1
    indices = get_corresponding_vertex_indices(vertices, symmetric_vertices, threshold)
    indices = torch.as_tensor(indices, dtype=torch.int64)
    assert len(torch.unique(indices)) == len(indices), "Two indices point towards the same vertex"
    return indices
