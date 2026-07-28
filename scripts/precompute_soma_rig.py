# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0

import soma
import copy
import json
import torch

def precompute_soma_rig():
    """
    Creates and returns a model with SOMA topology and SOMA rig.
    """
    device = torch.device("cpu")
    dtype = torch.float64

    soma_layer = soma.SOMALayer(device=device).to(dtype=dtype)

    # Compute bone positions and blendshapes from neutral pose
    sparse_rbf_matrix = soma_layer.skeleton_transfer.sparse_rbf_matrix.to(dtype=dtype, device=device)
    skinning_weights=soma_layer.skeleton_transfer.skinning_weights.to(dtype=dtype, device=device)
    bind_world_transforms = soma_layer.skeleton_transfer.bind_world_transforms.to(dtype=dtype, device=device)


    bone_labels = [str(label) for label in soma_layer.rig_data["joint_names"]]
    bone_parents = copy.copy(soma_layer.skeleton_transfer.joint_parent_ids)
    bone_parents[0] = -1
    t_pose_world = soma_layer.t_pose_world.to(dtype=dtype, device=device)
    bind_shape = soma_layer.skeleton_transfer.bind_shape.to(dtype=dtype, device=device)

    rig_data = dict(
        bind_shape=bind_shape,
        sparse_rbf_matrix=sparse_rbf_matrix,
        skinning_weights=skinning_weights,
        bind_world_transforms=bind_world_transforms,
        t_pose_world=t_pose_world,
        bone_labels=bone_labels,
        bone_parents=bone_parents,
    )
    return rig_data


def save_soma_rig_safetensors(data, path):
    import safetensors.torch
    tensors = {k: v.to_dense().contiguous() for k, v in data.items() if isinstance(v, torch.Tensor)}
    meta = {
        "bone_labels": data["bone_labels"],
        "bone_parents": data["bone_parents"],
    }
    safetensors.torch.save_file(tensors, path, metadata={"rig_meta": json.dumps(meta)})


if __name__ == "__main__":
    data = precompute_soma_rig()
    save_soma_rig_safetensors(data, "src/anny/data/soma/soma_rig.safetensors")
    # Also keep legacy .pt for backward compatibility
    torch.save(data, "src/anny/data/soma/soma_rig.pt")
