# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
"""Utilities operating on pose parameters."""
import torch
import roma

def transfer_pose_parameters(src_model, src_pose_parameters, phenotype_kwargs, local_changes_kwargs, target_model):
    """Re-express a pose defined on *src_model* in *target_model*'s pose parameterization.

    The source model is posed to obtain the world poses of its bones; the poses of the bones the
    target shares (matched by label) are then encoded into the target's pose parameterization.
    This only reproduces the pose when the shared bones have the same rest origins in both models or have no weights.
    Returns the target model's pose parameters yielding the same pose.
    """
    missing = [label for label in target_model.bone_labels if label not in src_model.bone_labels]
    assert not missing, f"target_model has bones absent from src_model, cannot transfer the pose: {missing}"
    bone_indices = [src_model.bone_labels.index(label) for label in target_model.bone_labels]

    src_output = src_model(phenotype_kwargs=phenotype_kwargs, local_changes_kwargs=local_changes_kwargs, pose_parameters=src_pose_parameters)
    base_output = target_model(phenotype_kwargs=phenotype_kwargs, local_changes_kwargs=local_changes_kwargs)

    assert (base_output["rest_vertices"] - src_output["rest_vertices"]).abs().max() < 1e-6, \
        "src_model and target_model have different rest meshes; they must represent the same body at rest to transfer the pose."

    # Preserve the LBS skinning delta (bone_pose @ rest_bone_pose^-1) across the two rigs' differing rest orientations.
    src_rest_inverse = roma.Rigid.from_homogeneous(src_output["rest_bone_poses"][:, bone_indices]).inverse().to_homogeneous()
    target_bone_poses = src_output["bone_poses"][:, bone_indices] @ src_rest_inverse @ base_output["rest_bone_poses"]

    input_dict = dict(bone_poses = target_bone_poses,
             rest_bone_poses = base_output["rest_bone_poses"])
    target_pose_parameters = target_model.get_pose_parameterization(input_dict, pose_parameterization=target_model.pose_parameterization)
    return target_pose_parameters
