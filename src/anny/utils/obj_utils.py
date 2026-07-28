# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
import torch


def _pack_face_list(faces, pack_as_tensor):
    """Pack face indices as a tensor when possible, otherwise return as list."""
    if not pack_as_tensor or not faces:
        return faces
    if len(set(len(f) for f in faces)) > 1:
        return faces  # mixed polygon sizes — cannot pack as uniform tensor
    return torch.as_tensor(faces)


def load_obj_file(mesh_filename, dtype=torch.float32, pack_as_tensor=True):
    """
    Simple OBJ file parser.
    """
    # Manual parsing of the base mesh
    vertices = []
    texture_coordinates = []

    as_tensor = torch.as_tensor if pack_as_tensor else lambda x, **kwargs: x

    groups = dict()
    # The first group is called "noname" by default
    group_name = "noname"
    face_vertex_indices = []
    face_texture_coordinate_indices = []
    with open(mesh_filename, "r") as f:
        for line in f.readlines():
            line = line.strip()
            if len(line) > 0 and not line.startswith("#") or line.startswith("mtllib"):
                split = line.split(" ")
                if split[0] == "o":
                    # New object. We consider only one object in this simple file parser
                    # and stop if vertices where already loaded.
                    if len(vertices) > 0:
                        break
                elif split[0] == "v":
                    coords = [float(x) for x in split[1:]]
                    assert len(coords) == 3
                    vertices.append(coords)
                elif split[0] == "vt":
                    coords = [float(x) for x in split[1:]]
                    assert len(coords) == 2
                    texture_coordinates.append(coords)
                elif split[0] == "g":
                    # New group
                    # Pack data from the previous group
                    if len(face_vertex_indices) > 0:
                        groups[group_name] = dict(
                            face_vertex_indices=_pack_face_list(
                                face_vertex_indices, pack_as_tensor
                            ),
                            face_texture_coordinate_indices=_pack_face_list(
                                face_texture_coordinate_indices, pack_as_tensor
                            ),
                        )
                    # Initialize group data
                    group_name = split[1]
                    if group_name in groups:
                        # Continue adding to existing group
                        fvi = groups[group_name]["face_vertex_indices"]
                        ftci = groups[group_name]["face_texture_coordinate_indices"]
                        face_vertex_indices = (
                            fvi.numpy().tolist()
                            if isinstance(fvi, torch.Tensor)
                            else list(fvi)
                        )
                        face_texture_coordinate_indices = (
                            ftci.numpy().tolist()
                            if isinstance(ftci, torch.Tensor)
                            else list(ftci)
                        )
                    else:
                        face_vertex_indices = []
                        face_texture_coordinate_indices = []

                elif split[0] == "f":
                    vids = []
                    vtids = []
                    for x in split[1:]:
                        # Use 0 as initial index (hence the minus 1)
                        data = [int(y) - 1 for y in x.split("/")]
                        vids.append(data[0])
                        if len(data) == 2:
                            vtids.append(data[1])
                    face_vertex_indices.append(vids)
                    face_texture_coordinate_indices.append(vtids)

    if len(face_vertex_indices) > 0:
        groups[group_name] = dict(
            face_vertex_indices=_pack_face_list(face_vertex_indices, pack_as_tensor),
            face_texture_coordinate_indices=_pack_face_list(
                face_texture_coordinate_indices, pack_as_tensor
            ),
        )

    vertices = as_tensor(vertices, dtype=dtype)
    if len(texture_coordinates) > 0:
        texture_coordinates = as_tensor(texture_coordinates, dtype=dtype)

    return vertices, texture_coordinates, groups


def save_obj_file(mesh_filename, vertices, faces):
    with open(mesh_filename, "w") as f:
        for vertex in vertices:
            f.write(f"v {' '.join([str(vertex[i]) for i in range(3)])}\n")

        for face in faces:
            f.write(f"f {' '.join([str(i + 1) for i in face])}\n")
