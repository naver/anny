# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0

import torch
import anny
import roma
import soma
import trimesh

device = torch.device("cpu")
dtype = torch.float64


anny_soma = anny.Anny(
    rig="soma", topology="soma", pose_parameterization="local-ref", all_phenotypes=True
).to(device=device, dtype=dtype)
soma_layer = soma.SOMALayer(identity_model_type="anny", mode="warp", device=device).to(
    dtype=dtype
)

phenotype_kwargs = torch.rand(
    (1, len(anny_soma.phenotype_labels)), device=device, dtype=dtype
)
local_changes = dict()

rotvec = 0.2 * torch.randn((1, 77, 3), device=device, dtype=dtype)
transl = 1.0 * torch.randn((1, 3), device=device, dtype=dtype)

extended_rotvec = torch.cat(
    (torch.zeros((1, 1, 3), device=device, dtype=dtype), rotvec), dim=1
)
R = roma.rotvec_to_rotmat(extended_rotvec)
pose_parameters = roma.Rigid(R, translation=None).to_homogeneous()
pose_parameters[:, 0, :3, 3] = transl

anny_output = anny_soma(
    pose_parameters=pose_parameters,
    phenotype_kwargs=phenotype_kwargs,
    local_changes_kwargs=local_changes,
)
soma_output = soma_layer(
    poses=rotvec,
    transl=transl,
    identity_coeffs=phenotype_kwargs,
    scale_params=local_changes,
    apply_correctives=False,
)

anny_faces = anny_soma.faces.numpy()

anny_vertices = anny_output["vertices"][0].detach().numpy()
soma_vertices = soma_output["vertices"][0].detach().numpy()
soma_faces = soma_layer.faces.numpy()

anny_mesh = trimesh.Trimesh(vertices=anny_vertices, faces=anny_faces, process=False)
soma_mesh = trimesh.Trimesh(vertices=soma_vertices, faces=soma_faces, process=False)

# Blue for anny, orange for soma, both partially transparent
anny_mesh.visual = trimesh.visual.ColorVisuals(
    mesh=anny_mesh, face_colors=[0, 128, 255, 160]
)
soma_mesh.visual = trimesh.visual.ColorVisuals(
    mesh=soma_mesh, face_colors=[255, 128, 0, 160]
)

# Add some axes for reference
axes = trimesh.creation.axis(origin_size=0.1, axis_length=0.5)


scene = trimesh.Scene({"anny": anny_mesh, "soma": soma_mesh, "axes": axes})

scene.export("soma_comparison.glb")
print("Saved soma_comparison.glb")
