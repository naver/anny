import os
import json
import torch

from anny.models.model_transforms import (
    LocalChanges,
    apply_procrustes_retopology,
    apply_soma_rig,
)
from anny.utils.mesh_utils import triangulate_faces as _triangulate_faces
from anny.utils.warp_mesh_utils import point_to_mesh_distance_and_face_uvs
from anny.paths import ANNY_ROOT_DIR
from anny.models import retopology


def _load_soma_rig(root_dirname):
    """Load soma rig data, preferring .safetensors and falling back to legacy .pt."""
    safetensors_path = os.path.join(root_dirname, "data/soma/soma_rig.safetensors")
    pt_path = os.path.join(root_dirname, "data/soma/soma_rig.pt")
    if os.path.exists(safetensors_path):
        from safetensors import safe_open
        tensors = {}
        with safe_open(safetensors_path, framework="pt") as f:
            meta_str = f.metadata().get("rig_meta", "{}")
            for k in f.keys():
                tensors[k] = f.get_tensor(k)
        meta = json.loads(meta_str)
        tensors["bone_labels"] = meta.get("bone_labels", [])
        tensors["bone_parents"] = meta.get("bone_parents", [])
        return tensors
    return torch.load(pt_path, weights_only=True)


def build_soma_rig_and_topology_model_data(all_phenotypes=False,
                                       skinning_method=None,
                                       pose_parameterization="local-bone",
                                       extrapolate_phenotypes=False,
                                       local_changes: LocalChanges ="none",
                                       facial_actions: bool = False):
    soma_rig_data = _load_soma_rig(ANNY_ROOT_DIR)

    soma_data = retopology.build_soma_topology_model_data(rig="default",
                                           all_phenotypes=all_phenotypes,
                                           skinning_method=skinning_method,
                                           pose_parameterization=pose_parameterization,
                                           extrapolate_phenotypes=extrapolate_phenotypes,
                                           local_changes=local_changes,
                                           facial_actions=facial_actions)

    data = apply_soma_rig(soma_data, soma_rig_data)
    return data


def build_soma_rig_model_data(
        topology="soma",
        all_phenotypes=False,
        skinning_method=None,
        pose_parameterization="local-bone",
        extrapolate_phenotypes=False,
        local_changes: LocalChanges="none",
        facial_actions: bool = False,
        remove_unattached_vertices=True,
        triangulate_faces=False):

    soma_data = build_soma_rig_and_topology_model_data(
        all_phenotypes=all_phenotypes,
        skinning_method=skinning_method,
        pose_parameterization=pose_parameterization,
        extrapolate_phenotypes=extrapolate_phenotypes,
        local_changes=local_changes,
        facial_actions=facial_actions,
    )

    if topology == "soma":
        return soma_data

    source_vertices = soma_data.template_vertices
    source_triangular_faces = torch.tensor(
        _triangulate_faces(soma_data.template_vertices, soma_data.faces.cpu().tolist()),
        dtype=torch.int64,
    )

    # Lazy import to avoid circular dependency with models/__init__.py
    from anny.models import build_fullbody_model_data
    target_data = build_fullbody_model_data(topology=topology,
                                            rig="default",
                                            remove_unattached_vertices=remove_unattached_vertices,
                                            triangulate_faces=triangulate_faces)

    vertices = target_data.template_vertices

    _, target2source_face_ids, uvs = point_to_mesh_distance_and_face_uvs(
        points=vertices.to(dtype=torch.float32),
        vertices=source_vertices.to(dtype=torch.float32),
        faces=source_triangular_faces,
        max_dist=1000.,
    )

    uvs = uvs.to(dtype=source_vertices.dtype)

    u, v = uvs[:, 0], uvs[:, 1]
    w = 1. - u - v
    target2source_barycentric_coordinates = torch.stack([u, v, w], dim=0)
    reference_vertex_indices = source_triangular_faces[target2source_face_ids]

    ref_data = soma_data
    data = apply_procrustes_retopology(
        ref_data,
        vertices=vertices,
        faces=target_data.faces,
        source_model=soma_data,
        reference_vertex_indices=reference_vertex_indices,
        barycentric_coordinates=target2source_barycentric_coordinates,
        base_mesh_vertex_indices=target_data.base_mesh_vertex_indices,
    )
    return data
