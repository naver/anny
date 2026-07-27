# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0

import os
import torch
from anny.models.smpl import SMPLX
import smplx
import trimesh

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compare SMPLX and Anny implementations of the SMPLX model.")
    parser.add_argument("--smplx_model_path", type=str, default=os.environ.get("SMPLX_MODEL_PATH"), help="Path to the SMPLX models directory.")
    parser.add_argument("--pose_corrective", action=argparse.BooleanOptionalAction, default=True, help="Include pose corrective blend shapes.")
    parser.add_argument("--use_pca", action=argparse.BooleanOptionalAction, default=True, help="Use PCA hand space.")
    parser.add_argument("--topology", type=str, default="anny", help="Output mesh topology.")
    args = parser.parse_args()

    dtype = torch.float32
    model = smplx.create(args.smplx_model_path, model_type="smplx", gender="neutral", use_pca=args.use_pca).to(dtype=dtype)
    model_wrapper = SMPLX(args.smplx_model_path, gender="neutral", use_pca=args.use_pca, pose_corrective=args.pose_corrective, topology=args.topology).to(dtype=dtype)

    betas = 0.5 * torch.randn((1, model.num_betas), dtype=torch.float32)
    expression = 0.5 * torch.randn((1, model.num_expression_coeffs), dtype=torch.float32)

    global_orient = 0.3 * torch.randn((1, 3), dtype=torch.float32)
    body_pose = 0.3 * torch.randn((1, model.NUM_BODY_JOINTS * 3), dtype=torch.float32)
    transl = torch.randn((1, 3), dtype=torch.float32)

    if not args.use_pca:
        left_hand_pose = 0.5 * torch.randn((1, 15 * 3), dtype=torch.float32)
        right_hand_pose = 0.5 * torch.randn((1, 15 * 3), dtype=torch.float32)
    else:
        left_hand_pose = torch.randn((1, 6), dtype=torch.float32)
        right_hand_pose = torch.randn((1, 6), dtype=torch.float32)
    leye_pose = 0.3 * torch.randn((1, 3), dtype=torch.float32)
    reye_pose = 0.3 * torch.randn((1, 3), dtype=torch.float32)
    jaw_pose = 0.3 * torch.randn((1, 3), dtype=torch.float32)

    smplx_output = model(betas=betas, expression=expression, global_orient=global_orient, transl=transl, body_pose=body_pose.view(1, -1),  jaw_pose=jaw_pose, leye_pose=leye_pose, reye_pose=reye_pose, left_hand_pose=left_hand_pose, right_hand_pose=right_hand_pose)
    anny_output = model_wrapper(betas=betas, expression=expression, global_orient=global_orient, transl=transl, body_pose=body_pose.view(1, -1),  jaw_pose=jaw_pose, leye_pose=leye_pose, reye_pose=reye_pose, left_hand_pose=left_hand_pose, right_hand_pose=right_hand_pose)

    # Export both meshes to a glb file for visual comparison in a 3D viewer
    scene = trimesh.Scene()

    mesh = trimesh.Trimesh(vertices=anny_output["vertices"][0].detach().numpy(), faces=model_wrapper.faces.numpy())
    mesh.visual.vertex_colors = [200, 200, 200, 255]
    scene.add_geometry(mesh, node_name="anny")

    mesh = trimesh.Trimesh(vertices=smplx_output.vertices[0].detach().numpy(), faces=model.faces_tensor.numpy())
    mesh.visual.vertex_colors = [200, 0, 0, 255]
    scene.add_geometry(mesh, node_name="smplx")

    # Add some axes for reference
    axes = trimesh.creation.axis(origin_size=0.05, axis_length=0.5)
    scene.add_geometry(axes, node_name="axes")

    scene.export("smplx_comparison.glb")
    print("Meshes exported to smplx_comparison.glb for visual comparison.")